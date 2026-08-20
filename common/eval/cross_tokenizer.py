"""Shared, cross-tokenizer held-out evaluation.

Runs any trained tokenizer's own boundary/span-inducing entry point over a held-out
parallel corpus (BOUQuET dev by default; kept disjoint from every training
--data-source) and reports the same fairness/efficiency metrics common.eval.metrics
defines, so all four tokenizers in this repo are compared on identical held-out
data with identical scoring code.

Each tokenizer's *.evaluate module supplies an `induce_spans_fn_by_lang` dict (a
`bytes -> list[bytes] spans` callable per language, already bound to that
tokenizer's loaded checkpoint + any extra argument its own induce_spans needs) --
this module stays agnostic to how that callable was built, same pattern as
common.eval.stability.
"""

import argparse
import json
from collections import Counter, defaultdict

import numpy as np

from common.config_file import parse_args_with_config
from common.eval.metrics import compression_rate, fertility, gini_coefficient, renyi_efficiency
from common.eval.parity import _find_anchor_key, anchor_invariant_parity
from common.eval.reporting import word_count
from common.eval.stability import sequences_by_lang_from_groups


def evaluate_on_groups(induce_spans_fn_by_lang, eval_groups, anchor_lang="eng"):
    """induce_spans_fn_by_lang: dict[lang -> (bytes -> list[bytes] spans)] callable.
    eval_groups: list[dict[lang -> text]] (e.g. load_bouquet_dev()'s return value).

    Languages present in eval_groups but missing from induce_spans_fn_by_lang are
    silently skipped, not errored on -- checkpoints trained on different language
    sets is expected, not exceptional.

    Also computes an ANCHOR-INVARIANT version of the disparity (token_parity_gm,
    token_parity_spread) via common.eval.parity.anchor_invariant_parity: a single
    fixed anchor inverts every ratio if a tokenizer is unusually good/bad
    specifically at the anchor (e.g. re-anchoring to Mandarin can flip a
    Chinese-optimized tokenizer from best to worst with no real change in
    per-language costs). token_parity_gm/spread avoid this at zero extra cost.

    anchor_lang: also computes TOKEN PARITY against this language (default
    "eng") -- the token-count analog of compute_lang_parity_ratios's
    byte-length ratio, from each group's own already-tokenized spans. ratio
    > 1 means this tokenizer produces more tokens for `lang` than the
    anchor for the same content -- distinct from byte-length disparity (a
    tokenizer can show ratio > 1 for a language whose byte-length ratio is
    ~1.0, if its vocab/merges just serve it worse). Resolved per-group via
    common.eval.parity._find_anchor_key (so "eng_Latn" still matches
    anchor_lang="eng"); a language never paired with the anchor gets ratio 1.0.

    Returns {"token_freq": {lang: Counter}, "renyi": {lang: float}, "gini": float,
    "per_lang_compression": {lang: float}, "avg_compression": float,
    "fertility": {lang: float}, "token_parity": {lang: float},
    "token_parity_anchor": str, "token_parity_gm": {lang: float},
    "token_parity_spread": float}.
    """
    token_freq = defaultdict(Counter)
    compressions_by_lang = defaultdict(list)
    word_counts = defaultdict(int)
    paired_anchor_counts = defaultdict(list)
    paired_lang_counts = defaultdict(list)
    anchor_found = False

    for group in eval_groups:
        num_tokens_this_group = {}
        for lang, text in group.items():
            induce_fn = induce_spans_fn_by_lang.get(lang)
            if induce_fn is None:
                continue
            raw = text.encode("utf-8") if isinstance(text, str) else bytes(text)
            spans = induce_fn(raw)
            token_freq[lang].update(spans)
            compressions_by_lang[lang].append(compression_rate(len(raw), len(spans)))
            word_counts[lang] += word_count(text)
            num_tokens_this_group[lang] = len(spans)

        anchor_key = _find_anchor_key(num_tokens_this_group, anchor_lang)
        if anchor_key is not None:
            anchor_found = True
            anchor_count = num_tokens_this_group[anchor_key]
            for lang, count in num_tokens_this_group.items():
                paired_anchor_counts[lang].append(anchor_count)
                paired_lang_counts[lang].append(count)

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
    token_parity_anchor = anchor_lang if anchor_found else next(iter(token_freq), anchor_lang)
    token_parity = {}
    for lang in token_freq:
        a_counts = paired_anchor_counts.get(lang, [])
        l_counts = paired_lang_counts.get(lang, [])
        if a_counts and sum(a_counts) > 0:
            token_parity[lang] = (sum(l_counts) / len(l_counts)) / (sum(a_counts) / len(a_counts))
        else:
            token_parity[lang] = 1.0
    token_parity_gm, token_parity_spread = anchor_invariant_parity(token_parity)
    return {
        "token_freq": token_freq,
        "renyi": renyi,
        "gini": gini,
        "per_lang_compression": per_lang_compression,
        "avg_compression": avg_compression,
        "fertility": fertility_by_lang,
        "token_parity": token_parity,
        "token_parity_anchor": token_parity_anchor,
        "token_parity_gm": token_parity_gm,
        "token_parity_spread": token_parity_spread,
    }


def evaluate_on_indigenous_panel(induce_spans_fn_by_lang, eval_groups):
    """Dedicated entry point for common.data.indigenous_panel's DELIBERATELY
    mixed-anchor panel (English for crk-en/iu-en, Spanish for the AmericasNLP
    pairs). Pooling every pair into one evaluate_on_groups(..., anchor_lang="eng")
    call would give every Spanish-anchored language an uninformative
    token_parity=1.0. Merging ratios from two different anchors into one
    anchor_invariant_parity call would also reintroduce anchor bias: a
    tokenizer's cost for "en" isn't guaranteed equal to its cost for "es",
    so ratio-vs-en and ratio-vs-es aren't on a common scale even after
    per-subgroup GM-normalization.

    eval_groups: list[dict[lang -> text]], the same flat shape every other
    evaluate_on_* consumer produces. Each group's anchor is inferred from its own
    keys against the current set of anchors in common.data.indigenous_panel.PAIRS,
    not a hardcoded pair, so a new anchor language added to that manifest needs no
    change here.

    Returns {
      "combined": <evaluate_on_groups over every pair's groups pooled together --
        token_freq/renyi/gini/per_lang_compression/avg_compression/fertility are
        per-language, anchor-free quantities, so meaningful pooled across the
        panel. Its token_parity* fields are NOT meaningful here and are dropped>,
      "token_parity_by_anchor": {anchor_lang: <that anchor's own subset's
        evaluate_on_groups result -- token_parity/token_parity_gm/
        token_parity_spread ARE meaningful within each, scoped to languages
        sharing that anchor>},
      "morphology_spread": {"fertility_spread": max/min fertility across the whole
        panel, "compression_spread": same for per_lang_compression} -- the
        panel-wide headline number that avoids the mixed-anchor problem entirely.
    }
    """
    from ..data.indigenous_panel import PAIRS

    known_anchors = {meta["anchor"] for meta in PAIRS.values()}
    groups_by_anchor = defaultdict(list)
    for group in eval_groups:
        anchors_present = known_anchors & set(group)
        if not anchors_present:
            raise ValueError(
                f"indigenous_panel group has none of this panel's known anchor languages "
                f"({sorted(known_anchors)}) as a key: {sorted(group)} -- see "
                "common.data.indigenous_panel.PAIRS"
            )
        for anchor in anchors_present:
            groups_by_anchor[anchor].append(group)

    combined = evaluate_on_groups(induce_spans_fn_by_lang, eval_groups)
    for key in ("token_parity", "token_parity_anchor", "token_parity_gm", "token_parity_spread"):
        combined.pop(key, None)

    token_parity_by_anchor = {
        anchor: evaluate_on_groups(induce_spans_fn_by_lang, anchor_groups, anchor_lang=anchor)
        for anchor, anchor_groups in groups_by_anchor.items()
    }

    fertility_vals = [v for v in combined["fertility"].values() if v > 0]
    compression_vals = [v for v in combined["per_lang_compression"].values() if v > 0]
    morphology_spread = {
        "fertility_spread": (max(fertility_vals) / min(fertility_vals)) if fertility_vals else 1.0,
        "compression_spread": (max(compression_vals) / min(compression_vals)) if compression_vals else 1.0,
    }

    return {
        "combined": combined,
        "token_parity_by_anchor": token_parity_by_anchor,
        "morphology_spread": morphology_spread,
    }


def report_indigenous_panel_eval(results, label=""):
    prefix = f"[{label}] " if label else ""
    print(f"\n{prefix}indigenous_panel evaluation (mixed-anchor -- see evaluate_on_indigenous_panel's own docstring):")
    print(
        f"  panel-wide fertility_spread={results['morphology_spread']['fertility_spread']:.3f}  "
        f"compression_spread={results['morphology_spread']['compression_spread']:.3f}"
    )
    combined = results["combined"]
    print(f"  avg_compression={combined['avg_compression']:.2f}  gini={combined['gini']:.4f}")
    print("  per-language compression / renyi efficiency / fertility (anchor-free, comparable across the whole panel):")
    for lang in sorted(combined["renyi"]):
        print(
            f"    {lang}: compression={combined['per_lang_compression'].get(lang, 0.0):.2f}  "
            f"renyi={combined['renyi'][lang]:.4f}  "
            f"fertility={combined['fertility'].get(lang, 0.0):.2f}"
        )
    for anchor, anchor_results in sorted(results["token_parity_by_anchor"].items()):
        print(f"  token_parity vs anchor={anchor!r} (only comparable within this anchor's own languages):")
        token_parity = anchor_results["token_parity"]
        token_parity_gm = anchor_results["token_parity_gm"]
        for lang in sorted(token_parity):
            if lang == anchor:
                continue
            print(
                f"    {lang}: token_parity={token_parity[lang]:.3f}  "
                f"token_parity_gm={token_parity_gm.get(lang, 1.0):.3f}"
            )


def report_eval(results, label=""):
    prefix = f"[{label}] " if label else ""
    print(f"\n{prefix}held-out evaluation:")
    print(f"  avg_compression={results['avg_compression']:.2f}  gini={results['gini']:.4f}")
    if "token_parity_spread" in results:
        print(f"  token_parity_spread (anchor-invariant, max/min across languages)={results['token_parity_spread']:.3f}")
    anchor = results.get("token_parity_anchor", "eng")
    print(
        f"  per-language compression / renyi efficiency / fertility / token parity vs "
        f"{anchor}=1.0 / anchor-invariant token parity vs the geometric mean=1.0:"
    )
    token_parity = results.get("token_parity", {})
    token_parity_gm = results.get("token_parity_gm", {})
    for lang in sorted(results["renyi"]):
        print(
            f"    {lang}: compression={results['per_lang_compression'].get(lang, 0.0):.2f}  "
            f"renyi={results['renyi'][lang]:.4f}  "
            f"fertility={results['fertility'].get(lang, 0.0):.2f}  "
            f"token_parity={token_parity.get(lang, 1.0):.3f}  "
            f"token_parity_gm={token_parity_gm.get(lang, 1.0):.3f}"
        )


def sample_eval_groups(eval_groups, max_eval_samples, seed=0):
    """Cap eval_groups to a random sample of max_eval_samples (without
    replacement), for periodic in-training dev checks -- scoring hundreds of
    BOUQuET groups every epoch boundary is wasteful for a training-time signal,
    not the final reported number (see --eval-data-source bouquet_test, which
    always scores everything). max_eval_samples=0, or eval_groups already
    shorter than it, means score everything."""
    if not max_eval_samples or len(eval_groups) <= max_eval_samples:
        return eval_groups
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(eval_groups), size=max_eval_samples, replace=False)
    return [eval_groups[i] for i in idx]


def eval_wandb_log_dict(results, prefix="eval"):
    """Flatten an evaluate_on_groups result dict into a wandb.log-able dict,
    matching fairtok.train.GRPOTrainer's own eval/* naming convention."""
    log_dict = {
        f"{prefix}/avg_compression": results["avg_compression"],
        f"{prefix}/gini": results["gini"],
    }
    if "token_parity_spread" in results:
        log_dict[f"{prefix}/token_parity_spread"] = results["token_parity_spread"]
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
    log_dict.update(
        {f"{prefix}/token_parity/{lang}": v for lang, v in results.get("token_parity", {}).items()}
    )
    log_dict.update(
        {f"{prefix}/token_parity_gm/{lang}": v for lang, v in results.get("token_parity_gm", {}).items()}
    )
    return log_dict


def strip_token_freq(results, is_indigenous_panel):
    """token_freq is {lang: Counter[bytes, int]} -- bytes keys aren't valid JSON,
    so strip it before writing --output (an earlier version of systems.pretraining.cli_eval
    hit the identical bug with tuple-keyed dicts). For --eval-data-source
    indigenous_panel, token_freq is nested inside "combined" and inside each
    anchor's entry in "token_parity_by_anchor", not at the top level."""
    if not is_indigenous_panel:
        return {k: v for k, v in results.items() if k != "token_freq"}
    return {
        "combined": {k: v for k, v in results["combined"].items() if k != "token_freq"},
        "token_parity_by_anchor": {
            anchor: {k: v for k, v in anchor_results.items() if k != "token_freq"}
            for anchor, anchor_results in results["token_parity_by_anchor"].items()
        },
        "morphology_spread": results["morphology_spread"],
    }


def indigenous_panel_wandb_log_dict(results, prefix="eval"):
    """Analog of eval_wandb_log_dict for --eval-data-source indigenous_panel's
    differently-shaped results (see evaluate_on_indigenous_panel)."""
    combined = results["combined"]
    log_dict = {
        f"{prefix}/avg_compression": combined["avg_compression"],
        f"{prefix}/gini": combined["gini"],
        **{f"{prefix}/renyi/{lang}": v for lang, v in combined["renyi"].items()},
        **{f"{prefix}/compression/{lang}": v for lang, v in combined["per_lang_compression"].items()},
        **{f"{prefix}/fertility/{lang}": v for lang, v in combined["fertility"].items()},
        **{f"{prefix}/morphology_spread/{k}": v for k, v in results["morphology_spread"].items()},
    }
    for anchor, anchor_results in results["token_parity_by_anchor"].items():
        log_dict.update(
            {
                f"{prefix}/token_parity_vs_{anchor}/{lang}": v
                for lang, v in anchor_results["token_parity"].items()
                if lang != anchor
            }
        )
    return log_dict


_EVAL_DATA_SOURCE_HELP = (
    "'bouquet' (default): BOUQuET DEV, for tuning/exploratory comparisons; "
    "'bouquet_test': BOUQuET TEST, the genuinely held-out split -- reserve for final "
    "reported numbers; "
    "'synthetic': placeholder corpus, for a quick sanity check with no network access; "
    "'indigenous_panel': common.data.indigenous_panel's curated panel (needs a one-time "
    "common.data.prepare_indigenous_panel run first) -- scored via "
    "evaluate_on_indigenous_panel since it's deliberately mixed-anchor; results have a "
    "different shape, not directly comparable to a bouquet/bouquet_test/synthetic run's"
)


def build_eval_arg_parser(system_label, checkpoint_help=None, eval_data_source_help=None):
    """--checkpoint/--eval-data-source/--num-groups/--device/--output/
    --result-key/--use-wandb/--wandb-project/--run-name: the entire argparse surface
    shared by all systems/*/evaluate.py in this repo (hf_frontier scores externally
    downloaded HF tokenizers rather than a trained checkpoint and needs its own CLI
    shape, so isn't built on this) -- extracted once rather than copy-pasted.

    --output/--result-key exist so results can join the same comparison pipeline
    (scripts/combine_eval_results.py, scripts/generate_tikz_figures.py), written as {result_key:
    results} (result_key defaults to system_label, e.g. "fanta"). Override
    --result-key to keep two differently-configured runs of the same system as
    distinct entries in one combined file.
    """
    parser = argparse.ArgumentParser(
        description=f"Evaluate a trained {system_label} checkpoint on held-out data."
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help=checkpoint_help or f"path to a {system_label} checkpoint (--output-dir at training time)",
    )
    parser.add_argument(
        "--eval-data-source",
        choices=["bouquet", "bouquet_test", "synthetic", "indigenous_panel"],
        default="bouquet",
        help=eval_data_source_help or _EVAL_DATA_SOURCE_HELP,
    )
    parser.add_argument(
        "--num-groups", type=int, default=None,
        help="cap the number of held-out groups scored; omit for the full set",
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--output", type=str, default=None,
        help="write {result_key: results} JSON here (default: print only, via report_eval/"
        "report_indigenous_panel_eval, no file written) -- the same per-tokenizer results "
        "shape systems/tokenization/hf_frontier/evaluate.py and systems/tokenization/claude_tokenizer/evaluate.py "
        "already write, so scripts/combine_eval_results.py can merge this in directly",
    )
    parser.add_argument(
        "--result-key", type=str, default=None,
        help=f"top-level key this run's results are written under in --output (default: "
        f"{system_label!r}, this system's own label) -- override to keep two differently-"
        f"configured runs of the same system as distinct entries in one combined file",
    )
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument(
        "--wandb-project", type=str, default=f"{system_label}_eval",
        help=f"own project, separate from {system_label}/train.py's own {system_label!r} "
        "training-time project -- this is a held-out EVALUATION run, not a training run",
    )
    parser.add_argument("--run-name", type=str, default="")
    return parser


def load_eval_groups(args, synthetic_groups=None):
    """`args`: the Namespace build_eval_arg_parser's parser produces (or anything
    with the same `.eval_data_source`/`.num_groups` attributes). synthetic_groups:
    an already-built list to slice for --eval-data-source synthetic instead of the
    default make_synthetic_parallel_groups call -- bpe's evaluate.py passes its own
    _SMOKE_TEST_GROUPS (real short text) here, since --data-source synthetic means
    something different for bpe at training time too."""
    if args.eval_data_source == "synthetic":
        if synthetic_groups is not None:
            return synthetic_groups[: args.num_groups] if args.num_groups else synthetic_groups
        from common.data.synthetic import make_synthetic_parallel_groups

        return make_synthetic_parallel_groups(args.num_groups or 40)
    if args.eval_data_source == "indigenous_panel":
        from common.data.corpora import stream_groups

        groups = list(stream_groups("indigenous_panel", config="all"))
        return groups[: args.num_groups] if args.num_groups else groups
    # "all": every language BOUQuET covers; evaluate_on_groups already skips
    # languages this checkpoint has no entry for, so this is always safe.
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
    """The shared main() body every systems/*/evaluate.py runs, near-verbatim across
    systems modulo the induce_spans call shape and checkpoint-loading step: parses
    args, loads the checkpoint, loads held-out groups, scores, reports. The two
    callables below are the genuinely per-system pieces, left as callables:

    load_model(checkpoint_path, device) -> model: e.g. plain `load_checkpoint` for
    systems whose inference.load_checkpoint already takes (path, device); fairtok's
    load_checkpoint takes no device (its checkpoint is a live nn.Module policy, not
    a frozen artifact), so fairtok passes `lambda path, device:
    load_checkpoint(path).to(device).eval()`.

    build_induce_fn_by_lang(model, sequences_by_lang, args) -> {lang: (bytes ->
    spans) callable}: e.g. magnet's per-script boundary-predictor resolution +
    coverage filter, fairtok's segment_bytes-over-a-live-policy, or the plain
    induce_spans call other systems make.

    Returns the same results dict evaluate_on_groups does (or
    evaluate_on_indigenous_panel's differently-shaped dict for --eval-data-source
    indigenous_panel).

    MAGNET CAVEAT: magnet/evaluate.py's build_induce_fn_by_lang resolves each
    language to a SCRIPT via eval_lang_to_script before looking it up in
    model.boundary_predictors. indigenous_panel's language keys (crk, iu, nah, es,
    ...) are plain codes not in magnet.train.LANG_SCRIPT, so each falls into its own
    one-off "script" bucket with no matching boundary_predictors entry -- a magnet
    checkpoint scores 0 languages on this panel. Not a bug to paper over: whether a
    per-script predictor generalizes to unseen scripts is a genuine capability
    question, not something to fake with an invented script mapping.
    """
    # parse_args_with_config (not plain .parse_args) adds -c/--config support so a
    # YAML file can supply these flags, matching every other CLI in this repo.
    args = parse_args_with_config(build_eval_arg_parser(system_label, checkpoint_help, eval_data_source_help), argv)
    model = load_model(args.checkpoint, args.device)

    eval_groups = load_eval_groups(args, synthetic_groups=synthetic_groups)
    print(
        f"checkpoint={args.checkpoint} eval_data_source={args.eval_data_source} "
        f"groups={len(eval_groups)}"
    )

    sequences_by_lang = sequences_by_lang_from_groups(eval_groups)
    induce_fn_by_lang = build_induce_fn_by_lang(model, sequences_by_lang, args)
    is_indigenous_panel = args.eval_data_source == "indigenous_panel"
    if is_indigenous_panel:
        results = evaluate_on_indigenous_panel(induce_fn_by_lang, eval_groups)
        report_indigenous_panel_eval(results, label=system_label)
    else:
        results = evaluate_on_groups(induce_fn_by_lang, eval_groups)
        report_eval(results, label=system_label)

    if args.output:
        result_key = args.result_key or system_label
        payload = {result_key: strip_token_freq(results, is_indigenous_panel)}
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"wrote results under key {result_key!r} to {args.output}")

    if args.use_wandb:
        import wandb

        run = wandb.init(
            project=args.wandb_project,
            name=args.run_name or None,
            job_type="eval",
            config={
                "checkpoint": args.checkpoint,
                "eval_data_source": args.eval_data_source,
                "num_groups": args.num_groups,
            },
        )
        log_fn = indigenous_panel_wandb_log_dict if is_indigenous_panel else eval_wandb_log_dict
        run.log(log_fn(results, prefix="eval"))
        run.finish()
        print(f"logged results to wandb project={args.wandb_project!r}")

    return results
