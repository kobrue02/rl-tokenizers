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
from common.eval.parity import _find_anchor_key, anchor_invariant_parity
from common.eval.reporting import word_count
from common.eval.stability import sequences_by_lang_from_groups


def evaluate_on_groups(induce_spans_fn_by_lang, eval_groups, anchor_lang="eng"):
    """induce_spans_fn_by_lang: dict[lang -> (bytes -> list[bytes] spans)] callable.
    eval_groups: list[dict[lang -> text]] (e.g. common.data.oldi_data.load_bouquet_dev()'s
    return value).

    Languages present in eval_groups but missing from induce_spans_fn_by_lang are
    silently skipped, not errored on -- a held-out comparison across independently
    trained checkpoints is exactly the situation where language coverage can
    legitimately differ (e.g. a checkpoint trained on fewer languages than BOUQuET
    covers).

    Also computes an ANCHOR-INVARIANT version of the same disparity (token_parity_gm,
    token_parity_spread) via common.eval.parity.anchor_invariant_parity -- a single
    fixed anchor silently assumes that language's own cost is the fairness "1.0"
    baseline, which inverts into every other ratio if a tokenizer happens to be
    unusually good or bad specifically AT the anchor (confirmed live: re-anchoring
    this project's own hf_frontier comparison to Mandarin flips Chinese-optimized
    tokenizers from best to worst and gpt2 from worst to best, with no actual change
    in any model's per-language token costs -- see that function's own docstring).
    token_parity_gm and token_parity_spread don't have this problem, at zero extra
    tokenization cost (derived from token_parity, already computed below).

    anchor_lang: also computes TOKEN PARITY against this language (default "eng") --
    the same "how many X does `lang` need vs. the anchor, for the exact same
    underlying content" question common.eval.parity.compute_lang_parity_ratios
    already answers for raw BYTE length, computed here instead from each group's
    OWN already-tokenized spans (no re-tokenization, no extra cost -- this reuses
    the exact `spans` the loop below already computed for the aggregate stats).
    ratio > 1 means THIS tokenizer produces more tokens for `lang` than for the
    anchor to say the same thing -- a real, tokenizer-specific fairness cost, not
    just the byte-length disparity common.eval.parity's own ratio already
    explains structurally (e.g. a tokenizer could still show token parity ratio
    > 1 for a language whose byte-length ratio is ~1.0, if its own vocabulary/
    merges just serve that language worse). anchor_lang's own key is resolved
    per-group via common.eval.parity._find_anchor_key (same short-code-vs-full-
    stem handling that module's own docstring explains), so a group keyed by
    "eng_Latn" still matches anchor_lang="eng". A language never paired with the
    anchor in any group gets ratio 1.0 (no evidence of a disparity), same
    convention compute_lang_parity_ratios uses.

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
    mixed-anchor panel (English for crk-en/iu-en, Spanish for the nine
    AmericasNLP pairs -- see that module's own docstring for why). Naively
    pooling every pair's groups into one plain evaluate_on_groups(...,
    anchor_lang="eng") call would silently give every Spanish-anchored
    language token_parity=1.0 (never paired with "eng" in any group, the
    same "no evidence of a disparity" fallback evaluate_on_groups already
    documents) -- not wrong, but uninformative for 9 of this panel's 11
    languages. Just as importantly, naively MERGING ratios computed against
    two DIFFERENT anchors into one anchor_invariant_parity call would
    silently reintroduce anchor bias in a subtler form: a tokenizer's own
    cost for "en" text isn't guaranteed equal to its cost for "es" text, so
    ratio-vs-en and ratio-vs-es aren't on a common scale even after each
    subgroup's own GM-normalization -- exactly the kind of anchor-dependence
    anchor_invariant_parity exists to eliminate, not something to smuggle
    back in.

    eval_groups: list[dict[lang -> text]], the SAME flat shape every other
    evaluate_on_* consumer already produces -- e.g. list(common.data.
    corpora.stream_groups("indigenous_panel", config="all")). Each group's
    own anchor is inferred directly from its own keys (every group
    genuinely contains its own anchor language as one of its two keys, by
    construction) against the CURRENT set of anchors in common.data.
    indigenous_panel.PAIRS, not a hardcoded {"en", "es"} pair -- adding a
    pair with a new anchor language to that manifest later needs no change
    here.

    Returns {
      "combined": <one evaluate_on_groups result over every pair's groups
        pooled together -- token_freq/renyi/gini/per_lang_compression/
        avg_compression/fertility are all per-language, unpaired
        quantities with no anchor concept at all, so these ARE meaningful
        pooled across the whole mixed-anchor panel. Its own token_parity/
        token_parity_anchor/token_parity_gm/token_parity_spread fields are
        NOT meaningful here (see above) and are dropped from this dict>,
      "token_parity_by_anchor": {anchor_lang: <that anchor's own subset's
        evaluate_on_groups result -- token_parity/token_parity_gm/
        token_parity_spread ARE meaningful within each of these, scoped
        strictly to the languages sharing that one anchor>},
      "morphology_spread": {"fertility_spread": max/min fertility across
        every language in the whole panel, "compression_spread": same for
        per_lang_compression} -- the panel-wide "how unfair is this
        tokenizer" headline number that avoids the mixed-anchor problem
        entirely, since fertility/compression need no anchor at all.
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


_EVAL_DATA_SOURCE_HELP = (
    "'bouquet' (default): BOUQuET DEV, for tuning/exploratory comparisons; "
    "'bouquet_test': BOUQuET TEST, the genuinely held-out split -- reserve for final "
    "reported numbers, not repeated tuning checks; "
    "'synthetic': the placeholder corpus, for a quick sanity check with no network access; "
    "'indigenous_panel': common.data.indigenous_panel's curated Indigenous-language panel "
    "(needs a one-time common.data.prepare_indigenous_panel run first) -- scored via "
    "evaluate_on_indigenous_panel, not evaluate_on_groups, since this panel is deliberately "
    "mixed-anchor (see that function's own docstring); results have a different shape, not "
    "directly comparable to a bouquet/bouquet_test/synthetic run's own"
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
    if args.eval_data_source == "indigenous_panel":
        from common.data.corpora import stream_groups

        groups = list(stream_groups("indigenous_panel", config="all"))
        return groups[: args.num_groups] if args.num_groups else groups
    # "all": every language BOUQuET covers -- evaluate_on_groups already
    # skips languages this checkpoint has no entry for, so this is always
    # safe.
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

    Returns the same results dict evaluate_on_groups does (or
    evaluate_on_indigenous_panel's own differently-shaped dict for
    --eval-data-source indigenous_panel -- see that function's own
    docstring).

    MAGNET CAVEAT, stated plainly: magnet/evaluate.py's own
    build_induce_fn_by_lang resolves each language to a SCRIPT via
    eval_lang_to_script before looking it up in model.boundary_predictors.
    indigenous_panel's own language keys (crk, iu, nah, es, ...) are plain
    codes with no lang_Script suffix and aren't in magnet.train.LANG_SCRIPT,
    so eval_lang_to_script's own fallback treats each one as its OWN
    one-off "script" bucket (see that function's docstring) -- a checkpoint
    trained the normal way (against BOUQuET/glot500's real lang_Script-keyed
    scripts) will have no matching boundary_predictors entry for any of
    them, so a magnet checkpoint scores 0 languages on this panel. Not a bug
    to work around here: it's a genuine capability question (does a
    per-script boundary predictor generalize to scripts it never saw a
    single example of?) rather than something this harness should paper
    over with an invented script mapping.
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
    if args.eval_data_source == "indigenous_panel":
        results = evaluate_on_indigenous_panel(induce_fn_by_lang, eval_groups)
        report_indigenous_panel_eval(results, label=system_label)
    else:
        results = evaluate_on_groups(induce_fn_by_lang, eval_groups)
        report_eval(results, label=system_label)
    return results
