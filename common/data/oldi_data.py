"""Real data loader: the OLDI-and-friends collection
(https://huggingface.co/collections/openlanguagedata/oldi-and-friends), minus wmt24pp.

Shared by every tokenizer in this repo (fairtok, magnet, flexitokens,
manta) -- comparing them meaningfully requires training/evaluating on the
same real multilingual data.

Every loader defaults to `langs="all"` (every language the source offers,
no curated subset), for both TRAINING (load_oldi_seed/load_flores_plus) and
EVAL (load_bouquet_dev/load_bouquet_test). smol moved to corpora.py's
BITEXT_SOURCES (it's English/Russian-pivot bilingual pairs, not true N-way
parallel content like oldi_seed/flores_plus).

LANGS below is one fixed list of 9 codes -- the strict cross-lingual
intersection of flores_plus/oldi_seed/smol_sent once wmt24pp is dropped
(wmt24pp is high/mid-resource-only and unrelated to OLDI's low-resource
focus; including it collapses the intersection to just {eng}). Not used as
a default anywhere any more; its one remaining consumer is
systems/fairtok/train_morfessor_cli.py's default language list:

    arz (Egyptian Arabic), bam (Bambara), ben (Bengali), eng (English), kas (Kashmiri),
    lij (Ligurian), mni (Manipuri), nqo (N'Ko), spa (Spanish)

One canonical script is picked per language where a dataset offers more
than one (e.g. kas_Arab over kas_Deva) -- see LANG_SCRIPT below.

oldi_seed/flores_plus: one file per language, aligned by an explicit `id`
field (true N-way parallel; `langs="all"` covers ~46/~227 languages
respectively).

BOUQuET dev/test default to `langs="all"` too (all 259 languages) --
evaluate_on_groups already skips languages a checkpoint has no entry for.
"""

import json

from huggingface_hub import hf_hub_download, list_repo_files
from tqdm.auto import tqdm

LANGS = ["arz", "bam", "ben", "eng", "kas", "lij", "mni", "nqo", "spa"]

LANG_SCRIPT = {
    "arz": "arz_Arab",
    "bam": "bam_Latn",
    "ben": "ben_Beng",
    "eng": "eng_Latn",
    "kas": "kas_Arab",
    "lij": "lij_Latn",
    "mni": "mni_Beng",
    "nqo": "nqo_Nkoo",
    "spa": "spa_Latn",
}


def _download(repo_id, filename):
    return hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset")


def _load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _list_all_stems(repo_id, dir_prefix, ext=".jsonl"):
    """Every lang[_Script[_variant]] stem this dataset natively offers in
    `dir_prefix`, discovered from the repo's file listing rather than our
    curated LANG_SCRIPT table -- lets a group use everything a fully N-way
    parallel source (oldi_seed, flores_plus, bouquet) actually has. `ext`
    defaults to ".jsonl" (oldi_seed/flores_plus); bouquet's files are
    ".parquet"."""
    prefix = dir_prefix + "/"
    files = list_repo_files(repo_id, repo_type="dataset")
    return sorted(
        f[len(prefix) : -len(ext)]
        for f in files
        if f.startswith(prefix) and f.endswith(ext)
    )


def _load_ngram_parallel(repo_id, dir_prefix, langs):
    """oldi_seed / flores_plus layout: one file per language, `id` + `text` fields.

    langs="all" loads every language file the dataset offers (41 for
    oldi_seed, ~212 for flores_plus dev), keyed by its full lang_Script
    stem -- script variants of the same language (e.g. kas_Arab/kas_Deva)
    become distinct entries rather than one curated choice. Compute cost is
    bounded by subsampling each group at training time, not by restricting
    what's loaded here.
    """
    if langs == "all":
        stems = _list_all_stems(repo_id, dir_prefix)
        lang_to_stem = dict(zip(stems, stems))
    else:
        lang_to_stem = {lang: LANG_SCRIPT[lang] for lang in langs}

    per_lang = {}
    for lang, stem in tqdm(
        lang_to_stem.items(), desc=f"loading {repo_id}/{dir_prefix}", unit="lang"
    ):
        path = _download(repo_id, f"{dir_prefix}/{stem}.jsonl")
        rows = _load_jsonl(path)
        per_lang[lang] = {r["id"]: r["text"] for r in rows}
    common_ids = sorted(set.intersection(*(set(d) for d in per_lang.values())))
    return [{lang: per_lang[lang][i] for lang in lang_to_stem} for i in common_ids]


def load_oldi_seed(langs="all"):
    return _load_ngram_parallel("openlanguagedata/oldi_seed", "seed", langs)


def load_flores_plus(split="dev", langs="all"):
    return _load_ngram_parallel("openlanguagedata/flores_plus", split, langs)


def _load_bouquet_split(split, langs):
    """Shared implementation for load_bouquet_dev/load_bouquet_test. Combines
    BOTH paragraph_level and sentence_level for the given split, matching
    BOUQuET's own HF "default" config -- roughly doubles row count per
    language (e.g. English dev: 120 paragraph_level + 504 sentence_level
    rows), matching the dataset card's stated totals (~162k dev / ~272k
    test rows across all 259 languages).

    Joined by `uniq_id`, NOT `par_id`: paragraph_level rows use `uniq_id ==
    par_id` (e.g. "P001"), sentence_level rows use a finer id per sentence
    ("P001-S1", "P001-S2", ...) -- disjoint value spaces, so both levels
    combine into one dict per language, each row becoming its own
    independent group (a concatenation, not a merge into the parent
    paragraph's entry).

    UNION across languages, not intersection: a group is built for every
    uniq_id that AT LEAST ONE requested language has, with whichever subset
    of languages covers that id -- NOT the N-way intersection
    _load_ngram_parallel uses for training. This matters: training's
    fairness losses need aligned parallel content within a group, but
    evaluate_on_groups accumulates each language's stats independently, so
    it has no such requirement. Full-intersection across all 259 BOUQuET
    languages was confirmed (via a real FANTA test run) to throw away most
    of each language's rows whenever some unrelated language happened to be
    missing that id (1052 surviving groups vs. ~272k rows total) -- the
    union join uses every row every language actually has.

    Each {lang}.parquet has src_lang == lang and tgt_lang == eng_Latn fixed
    as a reference pivot -- `src_text` is this file's actual per-language
    content, `tgt_text` is a constant English gloss (same string for a
    given uniq_id across every language's file, NOT the per-language
    sentence -- an easy bug to make).

    Defaults to langs="all" (all 259 real BOUQuET languages, keyed by full
    lang_Script stem) -- same convention as _load_ngram_parallel. Safe:
    every real call site already passes "all" explicitly.
    """
    import pyarrow.parquet as pq

    if langs == "all":
        stems = _list_all_stems(
            "facebook/bouquet", f"data/paragraph_level/{split}", ext=".parquet"
        )
        lang_to_stem = dict(zip(stems, stems))
    else:
        lang_to_stem = {lang: LANG_SCRIPT[lang] for lang in langs}

    per_lang = {}
    for lang, stem in tqdm(
        lang_to_stem.items(), desc=f"loading facebook/bouquet/{split}", unit="lang"
    ):
        rows = []
        for level in ("paragraph_level", "sentence_level"):
            path = _download(
                "facebook/bouquet", f"data/{level}/{split}/{stem}.parquet"
            )
            rows.extend(pq.read_table(path).to_pylist())
        per_lang[lang] = {r["uniq_id"]: r["src_text"] for r in rows}

    all_ids = set()
    for id_to_text in per_lang.values():
        all_ids.update(id_to_text)
    return [
        {lang: per_lang[lang][i] for lang in lang_to_stem if i in per_lang[lang]}
        for i in sorted(all_ids)
    ]


def load_bouquet_dev(langs="all"):
    return _load_bouquet_split("dev", langs)


def load_bouquet_test(langs="all"):
    """The genuinely held-out counterpart to load_bouquet_dev -- reserve for
    FINAL reported numbers; use load_bouquet_dev for tuning/exploratory
    comparison, to avoid the equivalent of test-set leakage."""
    return _load_bouquet_split("test", langs)
