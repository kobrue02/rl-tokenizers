"""Real data loader: the OLDI-and-friends collection
(https://huggingface.co/collections/openlanguagedata/oldi-and-friends), minus wmt24pp.

Shared by every tokenizer in this repo (fairtok's RL policy, and the
magnet/flexitokens/manta baselines) -- comparing them meaningfully requires
training/evaluating them all on the same real multilingual data.

Language panel (see conversation for how this was derived): the strict intersection of
flores_plus / oldi_seed / smol_sent's language coverage is just {eng} once wmt24pp is
included, because wmt24pp is a high/mid-resource-only benchmark unrelated to OLDI's
low-resource focus. Dropping wmt24pp, the 3-way intersection of the OLDI-native sources
gives a real 9-language panel spanning both resource levels:

    arz (Egyptian Arabic), bam (Bambara), ben (Bengali), eng (English), kas (Kashmiri),
    lij (Ligurian), mni (Manipuri), nqo (N'Ko), spa (Spanish)

One canonical script is picked per language where a dataset offers more than one
(e.g. kas_Arab over kas_Deva, ben_Beng over ben_Latn) -- see LANG_SCRIPT below.

Three source datasets, three different join strategies, because they're structured
differently:
  - oldi_seed, flores_plus: one file per language, aligned by an explicit `id` field
    (true N-way parallel -- one group has all 9 languages).
  - smol (smolsent only, never GATITOS): English-pivot bilingual pairs, {code}_en.jsonl,
    joined by a shared `id` -- empirically confirmed 562 ids common across all 8
    non-English languages in this panel, so groups here also end up 9-wide, not just
    pairwise, though only over that 562-sentence subset.

Eval: BOUQuET dev covers 6 of the 9 panel languages (arz, bam, ben, eng, lij, spa) via
data/paragraph_level/dev/{lang}.parquet, joined by `par_id`, value column `tgt_text`.
kas/mni/nqo aren't in BOUQuET at all, so FLORES+ devtest is used as their eval fallback
(devtest is disjoint from the `dev` split used for training, per FLORES' own train/eval
split convention). BOUQuET test is never touched, per the plan's evaluation gate.
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

SMOL_CODE = {  # smolsent's own bare-code naming, only for the 8 non-English languages
    "arz": "arz",
    "bam": "bm",
    "ben": "bn",
    "kas": "ks",
    "lij": "lij",
    "mni": "mni-Mtei",
    "nqo": "nqo",
    "spa": "es",
}

BOUQUET_LANGS = ["arz", "bam", "ben", "eng", "lij", "spa"]
FLORES_FALLBACK_LANGS = [
    "kas",
    "mni",
    "nqo",
]  # not in BOUQuET -- eval via FLORES+ devtest


def _download(repo_id, filename):
    return hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset")


def _load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _list_all_stems(repo_id, dir_prefix):
    """Every lang[_Script[_variant]] stem this dataset natively offers in
    `dir_prefix`, discovered from the repo's file listing rather than our
    curated LANG_SCRIPT table -- this is what lets a group use everything a
    fully N-way parallel source (oldi_seed, flores_plus) actually has, instead
    of just the 9-language reporting panel."""
    prefix = dir_prefix + "/"
    files = list_repo_files(repo_id, repo_type="dataset")
    return sorted(
        f[len(prefix) : -len(".jsonl")]
        for f in files
        if f.startswith(prefix) and f.endswith(".jsonl")
    )


def _load_ngram_parallel(repo_id, dir_prefix, langs):
    """oldi_seed / flores_plus layout: one file per language, `id` + `text` fields.

    langs="all" discovers and loads every language file the dataset natively
    offers (41 for oldi_seed, ~212 for flores_plus dev), keyed by its full
    lang_Script stem -- this also means script variants of the same language
    (e.g. kas_Arab and kas_Deva) become distinct entries rather than one
    curated choice, which is more information, not less. Compute cost is kept
    bounded by randomly subsampling each group at training time (each
    trainer's own group_sample_size-equivalent field), not by restricting
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


def load_oldi_seed(langs=LANGS):
    return _load_ngram_parallel("openlanguagedata/oldi_seed", "seed", langs)


def load_flores_plus(split="dev", langs=LANGS):
    return _load_ngram_parallel("openlanguagedata/flores_plus", split, langs)


def load_smol_groups(langs=None):
    """smolsent English-pivot pairs, joined by shared `id` across all requested
    non-English languages. English text is read off any one file's `trg` field."""
    langs = langs or [l for l in LANGS if l != "eng"]
    per_lang_text = {}
    per_lang_eng = {}
    for lang in langs:
        path = _download("google/smol", f"smolsent/{SMOL_CODE[lang]}_en.jsonl")
        rows = _load_jsonl(path)
        per_lang_text[lang] = {r["id"]: r["src"] for r in rows}
        per_lang_eng[lang] = {r["id"]: r["trg"] for r in rows}

    common_ids = sorted(set.intersection(*(set(d) for d in per_lang_text.values())))
    groups = []
    for i in common_ids:
        group = {lang: per_lang_text[lang][i] for lang in langs}
        group["eng"] = per_lang_eng[langs[0]][
            i
        ]  # same shared English source for every lang
        groups.append(group)
    return groups


def load_bouquet_dev(langs=BOUQUET_LANGS):
    """paragraph_level/dev, joined by `par_id`. Each {lang}.parquet has src_lang == lang
    and tgt_lang == eng_Latn fixed as a reference pivot -- so `src_text` is this file's
    actual per-language content, and `tgt_text` is a constant English gloss (the same
    string appears in every language's file for a given par_id, which is NOT the
    per-language sentence -- easy bug to make, caught by comparing files directly)."""
    import pyarrow.parquet as pq

    per_lang = {}
    for lang in langs:
        path = _download(
            "facebook/bouquet", f"data/paragraph_level/dev/{LANG_SCRIPT[lang]}.parquet"
        )
        table = pq.read_table(path).to_pylist()
        per_lang[lang] = {r["par_id"]: r["src_text"] for r in table}
    common_ids = sorted(set.intersection(*(set(d) for d in per_lang.values())))
    return [{lang: per_lang[lang][i] for lang in langs} for i in common_ids]


def load_flores_devtest_fallback(langs=FLORES_FALLBACK_LANGS):
    return load_flores_plus(split="devtest", langs=langs)


def load_all_training_groups(langs=LANGS):
    """Every training group from all three sources, pooled. Groups from different
    sources are NOT the same underlying sentence, so they're independent groups --
    only rows *within* one source's join are the same parallel content.

    langs="all" expands oldi_seed and flores_plus to every language they
    natively offer. smol stays on the explicit 9-language panel regardless --
    "all" isn't supported there yet, since (unlike the other two) knowing which
    of its ~115 languages share a given sentence id requires rescanning every
    one of its per-language pair files, not just listing them (see module
    docstring)."""
    groups = []
    groups.extend(load_oldi_seed(langs))
    groups.extend(load_flores_plus("dev", langs))
    smol_langs = (
        [l for l in LANGS if l != "eng"]
        if langs == "all"
        else [l for l in langs if l != "eng"]
    )
    groups.extend(load_smol_groups(smol_langs))
    return groups
