"""Shared, cross-tokenizer held-out evaluation.

Runs any trained tokenizer's own existing boundary/span-inducing entry point
(fairtok.policy.segment_bytes, magnet/flexitokens/manta.segment.induce_spans) over a
held-out parallel corpus (BOUQuET dev by default -- see common.data.oldi_data.
load_bouquet_dev, and fairtok/cli.py's _load_eval_groups docstring for why BOUQuET is
kept disjoint from every training --data-source) and reports the same fairness/
efficiency metrics common.eval.metrics already defines, so all four tokenizers in this
repo can be compared on IDENTICAL held-out data with IDENTICAL scoring code.

Each tokenizer's *.evaluate module supplies an `induce_spans_fn_by_lang` dict (a
`bytes -> list[bytes] spans` callable per language, already bound to that
tokenizer's loaded checkpoint + whatever extra argument its own induce_spans needs,
e.g. magnet's per-script boundary predictor) -- this module stays completely
agnostic to how that callable was built, the same pattern common.eval.stability already
uses for the boundary-stability diagnostic.
"""

import argparse
from collections import Counter, defaultdict

import numpy as np

from common.eval.metrics import compression_rate, fertility, gini_coefficient, renyi_efficiency
from common.eval.reporting import word_count
from common.eval.stability import sequences_by_lang_from_groups


def evaluate_on_groups(induce_spans_fn_by_lang, eval_groups):
    """induce_spans_fn_by_lang: dict[lang -> (bytes -> list[bytes] spans)] callable.
    eval_groups: list[dict[lang -> text]] (e.g. common.data.oldi_data.load_bouquet_dev()'s
    return value).

    Languages present in eval_groups but missing from induce_spans_fn_by_lang are
    silently skipped, not errored on -- a held-out comparison across independently
    trained checkpoints is exactly the situation where language coverage can
    legitimately differ (e.g. a checkpoint trained on fewer languages than BOUQuET
    covers).

    Returns {"token_freq": {lang: Counter}, "renyi": {lang: float}, "gini": float,
    "per_lang_compression": {lang: float}, "avg_compression": float,
    "fertility": {lang: float}}.
    """
    token_freq = defaultdict(Counter)
    compressions_by_lang = defaultdict(list)
    word_counts = defaultdict(int)

    for group in eval_groups:
        for lang, text in group.items():
            induce_fn = induce_spans_fn_by_lang.get(lang)
            if induce_fn is None:
                continue
            raw = text.encode("utf-8") if isinstance(text, str) else bytes(text)
            spans = induce_fn(raw)
            token_freq[lang].update(spans)
            compressions_by_lang[lang].append(compression_rate(len(raw), len(spans)))
            word_counts[lang] += word_count(text)

    renyi = {
        lang: renyi_efficiency(list(c.values())) for lang, c in token_freq.items() if c
    }
    gini = gini_coefficient(list(renyi.values())) if renyi else 0.0
    per_lang_compression = {
        lang: float(np.mean(vals)) for lang, vals in compressions_by_lang.items()
    }
    avg_compression = (
        float(np.mean(list(per_lang_compression.values())))
        if per_lang_compression
        else 0.0
    )
    fertility_by_lang = {
        lang: fertility(sum(counter.values()), word_counts.get(lang, 0))
        for lang, counter in token_freq.items()
    }
    return {
        "token_freq": token_freq,
        "renyi": renyi,
        "gini": gini,
        "per_lang_compression": per_lang_compression,
        "avg_compression": avg_compression,
        "fertility": fertility_by_lang,
    }


def report_eval(results, label=""):
    prefix = f"[{label}] " if label else ""
    print(f"\n{prefix}held-out evaluation:")
    print(f"  avg_compression={results['avg_compression']:.2f}  gini={results['gini']:.4f}")
    print("  per-language compression / renyi efficiency / fertility:")
    for lang in sorted(results["renyi"]):
        print(
            f"    {lang}: compression={results['per_lang_compression'].get(lang, 0.0):.2f}  "
            f"renyi={results['renyi'][lang]:.4f}  "
            f"fertility={results['fertility'].get(lang, 0.0):.2f}"
        )


def sample_eval_groups(eval_groups, max_eval_samples, seed=0):
    """Cap eval_groups to a random sample of max_eval_samples (without
    replacement), for periodic IN-TRAINING dev checks -- scoring hundreds of
    BOUQuET groups every epoch boundary in a long run is wasteful when it's just
    a training-time signal, not the final reported number (see evaluate.py's
    --eval-data-source bouquet_test, which always scores everything).
    max_eval_samples=0, or eval_groups already shorter than it, means score
    everything -- no sampling applied."""
    if not max_eval_samples or len(eval_groups) <= max_eval_samples:
        return eval_groups
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(eval_groups), size=max_eval_samples, replace=False)
    return [eval_groups[i] for i in idx]


def eval_wandb_log_dict(results, prefix="eval"):
    """Flatten an evaluate_on_groups result dict into a wandb.log-able dict,
    matching fairtok.train.GRPOTrainer's own eval/* naming convention for its
    (separately implemented, but equivalent-in-spirit) periodic held-out eval."""
    log_dict = {
        f"{prefix}/avg_compression": results["avg_compression"],
        f"{prefix}/gini": results["gini"],
    }
    log_dict.update(
        {f"{prefix}/renyi/{lang}": v for lang, v in results["renyi"].items()}
    )
    log_dict.update(
        {
            f"{prefix}/compression/{lang}": v
            for lang, v in results["per_lang_compression"].items()
        }
    )
    log_dict.update(
        {f"{prefix}/fertility/{lang}": v for lang, v in results["fertility"].items()}
    )
    return log_dict


_EVAL_DATA_SOURCE_HELP = (
    "'bouquet' (default): BOUQuET DEV, for tuning/exploratory comparisons; "
    "'bouquet_test': BOUQuET TEST, the genuinely held-out split -- reserve for final "
    "reported numbers, not repeated tuning checks; "
    "'synthetic': the placeholder corpus, for a quick sanity check with no network access"
)


def build_eval_arg_parser(system_label, checkpoint_help=None, eval_data_source_help=None):
    """--checkpoint/--eval-data-source/--num-groups/--device, confirmed live
    to be the ENTIRE argparse surface of all seven systems/*/evaluate.py in
    this repo (hf_frontier is the one evaluate.py NOT built on this: it scores
    externally-downloaded HF tokenizers, not a checkpoint this repo trained,
    and genuinely needs its own CLI shape) -- extracted verbatim rather than
    copy-pasted a seventh time."""
    parser = argparse.ArgumentParser(
        description=f"Evaluate a trained {system_label} checkpoint on held-out data."
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help=checkpoint_help or f"path to a {system_label} checkpoint (--output-dir at training time)",
    )
    parser.add_argument(
        "--eval-data-source", choices=["bouquet", "bouquet_test", "synthetic"], default="bouquet",
        help=eval_data_source_help or _EVAL_DATA_SOURCE_HELP,
    )
    parser.add_argument(
        "--num-groups", type=int, default=None,
        help="cap the number of held-out groups scored; omit for the full set",
    )
    parser.add_argument("--device", type=str, default="cpu")
    return parser


def load_eval_groups(args, synthetic_groups=None):
    """`args`: the Namespace build_eval_arg_parser's own parser produces (or
    anything with the same `.eval_data_source`/`.num_groups` attributes).
    synthetic_groups: an already-built list to slice for --eval-data-source
    synthetic instead of the default common.data.synthetic.
    make_synthetic_parallel_groups call -- bpe's own evaluate.py is the one
    system that passes this (its own _SMOKE_TEST_GROUPS, real short text
    rather than synthetic's byte generator, since --data-source synthetic
    means something different for bpe at TRAINING time too -- see
    bpe/train.py's own module docstring)."""
    if args.eval_data_source == "synthetic":
        if synthetic_groups is not None:
            return synthetic_groups[: args.num_groups] if args.num_groups else synthetic_groups
        from common.data.synthetic import make_synthetic_parallel_groups

        return make_synthetic_parallel_groups(args.num_groups or 40)
    # "all": every language BOUQuET covers, not just the 9-language panel --
    # evaluate_on_groups already skips languages this checkpoint has no entry
    # for, so this is always safe.
    from common.data.oldi_data import load_bouquet_dev, load_bouquet_test

    loader = load_bouquet_test if args.eval_data_source == "bouquet_test" else load_bouquet_dev
    groups = loader("all")
    if args.num_groups:
        groups = groups[: args.num_groups]
    return groups


def run_eval_cli(
    argv,
    system_label,
    load_model,
    build_induce_fn_by_lang,
    checkpoint_help=None,
    eval_data_source_help=None,
    synthetic_groups=None,
):
    """The shared main() body every systems/*/evaluate.py ran, near-verbatim
    (confirmed live: identical modulo the induce_spans call shape and the
    checkpoint-loading step) -- parses args, loads the checkpoint, loads held-
    out groups, scores, reports. The two callables below are the genuinely
    per-system pieces, deliberately left as callables rather than guessed at:

    load_model(checkpoint_path, device) -> model: e.g. plain `load_checkpoint`
    for the six systems whose own inference.load_checkpoint already takes
    (path, device); fairtok's own load_checkpoint takes no device (its
    checkpoint is a live nn.Module policy, not a frozen artifact), so fairtok
    passes `lambda path, device: load_checkpoint(path).to(device).eval()`.

    build_induce_fn_by_lang(model, sequences_by_lang, args) -> {lang: (bytes
    -> spans) callable}: e.g. magnet's own extra per-script boundary-predictor
    resolution + its own model.boundary_predictors coverage filter, fairtok's
    segment_bytes-over-a-live-policy, or the plain 2-arg (bpe/superbpe) /
    3-arg-with-device (flexitokens/manta/fanta) induce_spans call every other
    system makes.

    Returns the same results dict evaluate_on_groups does.
    """
    args = build_eval_arg_parser(system_label, checkpoint_help, eval_data_source_help).parse_args(argv)
    model = load_model(args.checkpoint, args.device)

    eval_groups = load_eval_groups(args, synthetic_groups=synthetic_groups)
    print(
        f"checkpoint={args.checkpoint} eval_data_source={args.eval_data_source} "
        f"groups={len(eval_groups)}"
    )

    sequences_by_lang = sequences_by_lang_from_groups(eval_groups)
    induce_fn_by_lang = build_induce_fn_by_lang(model, sequences_by_lang, args)
    results = evaluate_on_groups(induce_fn_by_lang, eval_groups)
    report_eval(results, label=system_label)
    return results
