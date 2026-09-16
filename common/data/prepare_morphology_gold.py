"""One-time local prep for common.eval.morphology's gold data (see that
module's own docstring for what MED / Consistency F1 measure and why).

Gold morpheme-boundary data only exists for a small subset of languages
with dedicated morphological resources -- unlike this project's main
259-language BPE-fairness eval, which needs only raw text. Three real,
independently-verified sources here, covering disjoint language sets:

  - MorphyNet (Batsuren et al. 2021, github.com/kbatsuren/MorphyNet):
    INFLECTIONAL data only (not derivational) for 14 of this project's
    languages -- see _MORPHYNET_LANGS -- of which 12 actually resolve (pol
    and rus don't -- see _MORPHYNET_UNAVAILABLE -- but rus is covered by
    sigmorphon below regardless, so the real net gap is pol alone: MorphyNet's
    pol/ directory has no inflectional file at all, only derivational, which
    this module doesn't use -- see below -- so pol has NO gold morphology
    data from any of this module's 3 sources; a genuine, reported coverage
    gap, not silently papered over). MorphyNet's own directory for Mongolian
    is named "mon", not this project's "khk_Cyrl" (Halh Mongolian,
    FLORES/BOUQuET convention) -- same language, different code, verified
    against results/bpe_comparison.json's own khk_Cyrl entry. MorphyNet's
    repo also has an "hbs" (Serbo-Croatian) directory, dropped here since
    hbs isn't one of this project's languages. Filenames are NOT uniform
    across languages (hun/por/spa each deviate -- see
    _MORPHYNET_FILENAME_OVERRIDES, verified live per-language). Each row is
    lemma / inflected_form / features / "|"-delimited segmentation with "-"
    marking affix-attachment points (verified by hand that stripping "-" and
    joining reconstructs inflected_form exactly, e.g.
    "zu-|füg|-end" -> "zufügend"). Derivational data (base_word /
    derived_word / pos / pos / affix / prefix_or_suffix) is deliberately
    NOT used: it does not reliably reconstruct the derived word via simple
    concatenation (e.g. "China"+"ese" != "Chinese", a spelling change), so
    its implied morph boundaries can't be trusted without ad-hoc repair
    this project has no way to verify is correct.

  - SIGMORPHON 2022 Segmentation Shared Task
    (github.com/sigmorphon/2022SegmentationST -- the SEGMENTATION task,
    distinct from the same year's InflectionST): eng/rus/hun specifically,
    OVERRIDING MorphyNet's own entries for those 3 languages entirely (not
    merged word-by-word). This is the SAME gold source Asgari et al.'s
    MorphBPE paper itself uses for exactly these 3 languages (confirmed by
    direct reading of that paper), so this project's own MED/Consistency-F1
    numbers for eng/rus/hun are directly comparable to MorphBPE's own
    reported Table 2 results, not just same-metric-name coincidentally-
    different data. Segmentation here is CANONICAL (lemma-based morphs,
    e.g. Russian "бородёнки" -> "борода"+"ёнка"+"и"), not a literal
    substring split of the surface form -- expected, and fine for this
    project's metrics, which compare morph/span lists as atomic units
    rather than requiring literal reconstruction.

  - UniMorph (per-language repos under github.com/unimorph): aym/cni/shp,
    this project's 3 indigenous-panel languages (see
    common/data/prepare_indigenous_panel.py and common/data/indigenous_panel.py
    for why these 3 specifically, and their bare-code convention -- no
    lang_Script suffix, matching indigenous_panel.py's own PAIRS dict keys'
    "code" field). UniMorph is lemma/form/tag triples, NOT pre-segmented --
    morphs are APPROXIMATED here via a longest-common-prefix/suffix
    alignment between lemma and form (tagged "unimorph_aligned_approx" in
    the output, distinct from the other sources' true gold boundaries --
    callers should treat this source's numbers as noisier).

A fourth source, UniSegments 1.0 (Žabokrtský et al., LREC 2022,
hdl.handle.net/11234/1-4629), was investigated per this feature's own
design plan and its real download mechanism resolved (a ~137MB tarball at
.../server/api/core/bitstreams/handle/11234/1-4629/UniSegments-1.0-public.tar.gz,
not discoverable from the landing page's static HTML -- it's populated by
client-side JS -- but present in the page's embedded Angular state JSON).
Its own per-language datasets confirmed labeled "-MorphyNet" for exactly
this project's same 14 MorphyNet languages (cat, ces, deu, eng, fin, fra,
hun, ita, mon, pol, por, rus, spa, swe) -- i.e. it re-packages the SAME
underlying MorphyNet data (with explicit character-index spans instead of
MorphyNet's own "-" convention, which is a real quality improvement, but
not a NEW language). Its own non-MorphyNet datasets (hye, kpv, mdf, mhr,
myv, udm, tgk, ben, hin, kan, mal, mar, fas, lat, hbs) are not languages
this project evaluates. Net incremental language coverage for this
project: zero -- so it is NOT used here; MorphyNet's own raw per-language
TSVs are fetched directly instead (simpler: small per-file fetches instead
of one 137MB tarball, for an equivalent, already hand-verified format).

Output format: one "{output_dir}/{lang_code}.jsonl" per language, each line
{"word": str, "morphs": list[str], "source": str}. load_morphology_gold
reads these back into the exact shape
common.eval.morphology.evaluate_morphological_segmentation expects.
"""

import argparse
import json
import os

import requests

_REQUEST_TIMEOUT = 60

# Cap on how many (deduplicated) gold words to keep per language per source --
# plenty for stable MED / Consistency-F1 statistics (the latter already
# subsamples to at most 2000 word PAIRS on top of this) without the fetch/
# parse cost of e.g. MorphyNet English's full 650k-row inflectional table.
_MAX_WORDS_PER_LANG = 3000

# MorphyNet source directory name -> this project's lang_Script code.
_MORPHYNET_LANGS = {
    "cat": "cat_Latn",
    "ces": "ces_Latn",
    "deu": "deu_Latn",
    "eng": "eng_Latn",
    "fin": "fin_Latn",
    "fra": "fra_Latn",
    "hun": "hun_Latn",
    "ita": "ita_Latn",
    "mon": "khk_Cyrl",
    "pol": "pol_Latn",
    "por": "por_Latn_braz1246",
    "rus": "rus_Cyrl",
    "spa": "spa_Latn",
    "swe": "swe_Latn",
}

# SIGMORPHON source code -> this project's lang_Script code. Overrides
# MorphyNet entirely for these 3 (see module docstring).
_SIGMORPHON_LANGS = {"eng": "eng_Latn", "rus": "rus_Cyrl", "hun": "hun_Latn"}

# UniMorph repo name -> this project's indigenous-panel code (bare, no
# lang_Script suffix -- matches common.data.indigenous_panel.PAIRS's "code").
_UNIMORPH_INDIGENOUS_LANGS = {"aym": "aym", "cni": "cni", "shp": "shp"}

_MORPHYNET_REPO = "kbatsuren/MorphyNet"
_MORPHYNET_BRANCH = "main"

# MorphyNet's own filenames are NOT uniformly "{src_code}.inflectional.v1.tsv"
# -- verified live per-language: hun's is named with an "hu" prefix and its
# own "segmentation" suffix (format-identical otherwise); por's uses a "pt"
# prefix; spa's is split into 2 parts needing concatenation. Anything not
# listed here uses the "{src_code}.inflectional.v1.tsv" default.
_MORPHYNET_FILENAME_OVERRIDES = {
    "hun": ["hu.inflectional.segmentation.v1.tsv"],
    "por": ["pt.inflectional.v1.tsv"],
    "spa": ["spa.inflectional.v1.part1.tsv", "spa.inflectional.v1.part2.tsv"],
}

# Languages with NO usable inflectional file in the source repo at all (only
# derivational, which this module deliberately doesn't use -- see module
# docstring) or only a non-TSV format not worth a one-off unzip path for a
# language sigmorphon already covers. Verified live via the repo's own
# per-language directory listing.
_MORPHYNET_UNAVAILABLE = {
    "pol": "MorphyNet's pol/ directory has only pol.derivational.v1.tsv -- no inflectional file exists",
    "rus": "MorphyNet's rus/ inflectional data is rus.inflectional.v1.zip (not a plain TSV) -- "
    "skipped since rus is already covered by sigmorphon",
}
_SIGMORPHON_REPO = "sigmorphon/2022SegmentationST"
_SIGMORPHON_BRANCH = "main"
_UNIMORPH_BRANCH = "main"


def _fetch_raw_github_lines(repo, branch, path, request_timeout=_REQUEST_TIMEOUT):
    """One raw.githubusercontent.com file -> list[str] of its lines. Same
    helper as common.data.prepare_indigenous_panel's own (not imported from
    there to avoid a cross-dependency between two independent one-time prep
    scripts over a five-line function)."""
    url = f"https://raw.githubusercontent.com/{repo}/{branch}/{path}"
    resp = requests.get(url, timeout=request_timeout)
    resp.raise_for_status()
    return resp.text.splitlines()


def _write_gold_jsonl(path, words):
    """words: dict[word -> (morphs, source)]."""
    with open(path, "w", encoding="utf-8") as f:
        for word, (morphs, source) in words.items():
            f.write(json.dumps({"word": word, "morphs": morphs, "source": source}, ensure_ascii=False) + "\n")


def _prepare_morphynet_language(src_code, request_timeout=_REQUEST_TIMEOUT):
    """One MorphyNet language's inflectional TSV -> dict[word -> (morphs,
    "morphynet")], capped at _MAX_WORDS_PER_LANG, skipping any row whose
    "-"-stripped segmentation doesn't reconstruct inflected_form exactly
    (a real data-quality guard, not just a formatting nicety -- a mismatch
    means the segmentation column doesn't actually describe this row's
    surface form, so using it as gold would be wrong)."""
    if src_code in _MORPHYNET_UNAVAILABLE:
        raise ValueError(_MORPHYNET_UNAVAILABLE[src_code])
    filenames = _MORPHYNET_FILENAME_OVERRIDES.get(src_code, [f"{src_code}.inflectional.v1.tsv"])
    lines = []
    for filename in filenames:
        lines.extend(_fetch_raw_github_lines(_MORPHYNET_REPO, _MORPHYNET_BRANCH, f"{src_code}/{filename}", request_timeout))
    words = {}
    for line in lines:
        parts = line.split("\t")
        if len(parts) != 4:
            continue
        _lemma, inflected_form, _features, segmentation = parts
        if segmentation == "-" or inflected_form in words:
            continue
        morphs = [piece.strip("-") for piece in segmentation.split("|")]
        if "".join(morphs) != inflected_form:
            continue
        words[inflected_form] = (morphs, "morphynet")
        if len(words) >= _MAX_WORDS_PER_LANG:
            break
    return words


def _prepare_morphynet(request_timeout=_REQUEST_TIMEOUT):
    """Returns (gold_by_lang, errors) across all of _MORPHYNET_LANGS --
    errors is {src_code: exception str} for any language whose fetch/parse
    failed, so one dead source doesn't abort the other 13."""
    gold_by_lang = {}
    errors = {}
    for src_code, lang in _MORPHYNET_LANGS.items():
        try:
            gold_by_lang[lang] = _prepare_morphynet_language(src_code, request_timeout)
        except Exception as e:  # noqa: BLE001 -- one bad source must not abort the rest
            errors[src_code] = str(e)
    return gold_by_lang, errors


def _prepare_sigmorphon_segmentation(request_timeout=_REQUEST_TIMEOUT):
    """SIGMORPHON 2022 Segmentation Shared Task's eng/rus/hun word-level
    training files -> dict[lang -> dict[word -> (morphs,
    "sigmorphon2022segmentation")]]. Segmentation column uses " @@" as a
    continuation marker between morphs (e.g. "penta @@azo @@ole" ->
    ["penta", "azo", "ole"])."""
    gold_by_lang = {}
    errors = {}
    for src_code, lang in _SIGMORPHON_LANGS.items():
        try:
            lines = _fetch_raw_github_lines(
                _SIGMORPHON_REPO, _SIGMORPHON_BRANCH, f"data/{src_code}.word.train.tsv", request_timeout
            )
            words = {}
            for line in lines:
                parts = line.split("\t")
                if len(parts) != 3:
                    continue
                word, segmented, _flags = parts
                if word in words:
                    continue
                morphs = [m.strip() for m in segmented.split(" @@") if m.strip()]
                if not morphs:
                    continue
                words[word] = (morphs, "sigmorphon2022segmentation")
                if len(words) >= _MAX_WORDS_PER_LANG:
                    break
            gold_by_lang[lang] = words
        except Exception as e:  # noqa: BLE001
            errors[src_code] = str(e)
    return gold_by_lang, errors


def _align_unimorph_lemma_form(lemma, form):
    """Approximate morph split for one UniMorph (lemma, form) pair via a
    longest-common-prefix/suffix alignment against the FORM (not the
    lemma): common prefix, then whatever's left in the middle, then common
    suffix (if non-overlapping with the prefix match). Not a linguistic
    segmentation (UniMorph carries no gold boundaries at all -- see module
    docstring) -- a documented, fully-specified heuristic approximation,
    tagged "unimorph_aligned_approx" downstream so callers can tell it
    apart from the other three sources' true gold data."""
    i = 0
    while i < len(lemma) and i < len(form) and lemma[i] == form[i]:
        i += 1
    max_suffix = min(len(lemma) - i, len(form) - i)
    j = 0
    while j < max_suffix and lemma[len(lemma) - 1 - j] == form[len(form) - 1 - j]:
        j += 1
    prefix = form[:i]
    suffix = form[len(form) - j :] if j > 0 else ""
    middle = form[i : len(form) - j] if j > 0 else form[i:]
    morphs = [m for m in (prefix, middle, suffix) if m]
    return morphs if morphs else [form]


def _prepare_unimorph_indigenous(request_timeout=_REQUEST_TIMEOUT):
    """aym/cni/shp's UniMorph lemma/form/tag files -> dict[lang ->
    dict[word -> (morphs, "unimorph_aligned_approx")]], word = the
    inflected FORM (not the lemma -- consistent with every other source
    here keying on the surface word)."""
    gold_by_lang = {}
    errors = {}
    for repo_name, lang in _UNIMORPH_INDIGENOUS_LANGS.items():
        try:
            lines = _fetch_raw_github_lines(
                f"unimorph/{repo_name}", _UNIMORPH_BRANCH, repo_name, request_timeout
            )
            words = {}
            for line in lines:
                parts = line.split("\t")
                if len(parts) != 3:
                    continue
                lemma, form, _tag = parts
                if not lemma or not form or form in words:
                    continue
                words[form] = (_align_unimorph_lemma_form(lemma, form), "unimorph_aligned_approx")
                if len(words) >= _MAX_WORDS_PER_LANG:
                    break
            gold_by_lang[lang] = words
        except Exception as e:  # noqa: BLE001
            errors[repo_name] = str(e)
    return gold_by_lang, errors


_SOURCE_PREPARERS = {
    "morphynet": _prepare_morphynet,
    "sigmorphon": _prepare_sigmorphon_segmentation,
    "unimorph": _prepare_unimorph_indigenous,
}


def prepare_morphology_gold(output_dir, sources=None, request_timeout=_REQUEST_TIMEOUT):
    """sources: subset of _SOURCE_PREPARERS's keys (default: all 3). Applies
    them in a fixed order (morphynet, then sigmorphon, then unimorph) so
    that sigmorphon's eng/rus/hun OVERRIDE morphynet's own entries for
    those languages entirely, per the module docstring -- unimorph's 3
    languages never overlap with the other two sources' languages, so its
    ordering relative to them doesn't matter.

    Returns {"languages": {lang: {"num_words": int, "source": str}},
    "errors": {source_name: {sub_source_key: error str}}} -- writes one
    "{output_dir}/{lang}.jsonl" per language with any gold words at all.
    """
    os.makedirs(output_dir, exist_ok=True)
    source_names = list(sources) if sources else list(_SOURCE_PREPARERS)
    unknown = set(source_names) - set(_SOURCE_PREPARERS)
    if unknown:
        raise ValueError(f"unknown source(s) {sorted(unknown)} -- choose from {sorted(_SOURCE_PREPARERS)}")

    gold_by_lang = {}
    all_errors = {}
    for name in ("morphynet", "sigmorphon", "unimorph"):
        if name not in source_names:
            continue
        lang_words, errors = _SOURCE_PREPARERS[name](request_timeout)
        for lang, words in lang_words.items():
            if words:
                gold_by_lang[lang] = words
        if errors:
            all_errors[name] = errors

    languages = {}
    for lang, words in gold_by_lang.items():
        _write_gold_jsonl(os.path.join(output_dir, f"{lang}.jsonl"), words)
        any_source = next(iter(words.values()))[1]
        languages[lang] = {"num_words": len(words), "source": any_source}

    return {"languages": languages, "errors": all_errors}


def load_morphology_gold(gold_dir):
    """{gold_dir}/*.jsonl -> dict[lang -> dict[word -> (gold_morphs,
    source)]], the exact shape common.eval.morphology.evaluate_morphological_segmentation
    consumes. Returns {} if gold_dir doesn't exist or has no *.jsonl files
    (a run without --morphology-gold-dir configured, not an error)."""
    if not os.path.isdir(gold_dir):
        return {}
    gold_by_lang = {}
    for filename in sorted(os.listdir(gold_dir)):
        if not filename.endswith(".jsonl"):
            continue
        lang = filename[: -len(".jsonl")]
        words = {}
        with open(os.path.join(gold_dir, filename), encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                words[row["word"]] = (row["morphs"], row["source"])
        gold_by_lang[lang] = words
    return gold_by_lang


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="One-time download+convert of gold morphological segmentation data for common.eval.morphology."
    )
    parser.add_argument("--output-dir", type=str, default="data/morphology_gold")
    parser.add_argument(
        "--sources", type=str, default=None,
        help=f"comma-separated subset of sources to prepare (default: all of {sorted(_SOURCE_PREPARERS)})",
    )
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    sources = args.sources.split(",") if args.sources else None
    summary = prepare_morphology_gold(args.output_dir, sources=sources)
    for lang, info in sorted(summary["languages"].items()):
        print(f"  {lang} [{info['source']}]: {info['num_words']} words")
    for source_name, errors in summary["errors"].items():
        for sub_key, err in errors.items():
            print(f"  FAILED {source_name}/{sub_key}: {err}")


if __name__ == "__main__":
    main()
