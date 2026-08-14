"""Real data loader: the OLDI-and-friends collection
(https://huggingface.co/collections/openlanguagedata/oldi-and-friends), minus wmt24pp.

Shared by every tokenizer in this repo (fairtok's RL policy, and the
magnet/flexitokens/manta baselines) -- comparing them meaningfully requires
training/evaluating them all on the same real multilingual data.

TRAINING (load_oldi_seed/load_flores_plus below) now defaults to `langs="all"`
-- every language the source natively offers -- rather than a curated panel;
see common.data.corpora's own module docstring for why (and for smol, which
moved out of this module entirely into corpora.py's BITEXT_SOURCES pattern,
since it's English/Russian-pivot bilingual pairs, not true N-way parallel
content the way oldi_seed/flores_plus are).

LANGS below is NOT a training default any more -- it survives as a genuinely
useful reference: the strict cross-lingual intersection of flores_plus/
oldi_seed/smol_sent's language coverage once wmt24pp is dropped (wmt24pp is a
high/mid-resource-only benchmark unrelated to OLDI's low-resource focus, and
including it collapses the intersection to just {eng}). Still used directly
by systems/fairtok/train_morfessor_cli.py's own default panel, and it's the
9-language set this project's own EVAL side (BOUQuET, mostly) still centers
its held-out comparisons on:

    arz (Egyptian Arabic), bam (Bambara), ben (Bengali), eng (English), kas (Kashmiri),
    lij (Ligurian), mni (Manipuri), nqo (N'Ko), spa (Spanish)

One canonical script is picked per language where a dataset offers more than one
(e.g. kas_Arab over kas_Deva, ben_Beng over ben_Latn) -- see LANG_SCRIPT below.

oldi_seed/flores_plus: one file per language, aligned by an explicit `id` field
(true N-way parallel -- a `langs="all"` group can have every language both
datasets offer, ~46/~227 respectively at last count, not just LANGS's 9).

Eval: BOUQuET dev/test (load_bouquet_dev/load_bouquet_test below) default to
`langs="all"` too (every one of BOUQuET's 259 languages) -- common.eval.
cross_tokenizer.evaluate_on_groups already skips languages a given checkpoint
has no entry for, so this is always safe, and every real call site in this
repo already passes "all" explicitly regardless of this module's own default.
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
    curated LANG_SCRIPT table -- this is what lets a group use everything a
    fully N-way parallel source (oldi_seed, flores_plus, bouquet) actually has,
    instead of just the 9-language reporting panel. `ext` defaults to ".jsonl"
    (oldi_seed/flores_plus's own layout); bouquet's own files are ".parquet"."""
    prefix = dir_prefix + "/"
    files = list_repo_files(repo_id, repo_type="dataset")
    return sorted(
        f[len(prefix) : -len(ext)]
        for f in files
        if f.startswith(prefix) and f.endswith(ext)
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


def load_oldi_seed(langs="all"):
    return _load_ngram_parallel("openlanguagedata/oldi_seed", "seed", langs)


def load_flores_plus(split="dev", langs="all"):
    return _load_ngram_parallel("openlanguagedata/flores_plus", split, langs)


def _load_bouquet_split(split, langs):
    """Shared implementation for load_bouquet_dev/load_bouquet_test. Combines
    BOTH paragraph_level and sentence_level for the given split -- matching
    BOUQuET's own HF "default" config (data_files: "data/*_level/{split}/*.parquet"),
    not just paragraph_level alone. This roughly doubles row count per language
    (e.g. English dev: 120 paragraph_level + 504 sentence_level rows) and is what
    the dataset card's own stated totals (~162k dev / ~272k test rows, summed
    across all 259 languages) refer to -- paragraph_level alone is only a fraction
    of that.

    Joined by `uniq_id`, NOT `par_id`: paragraph_level rows use `uniq_id == par_id`
    (e.g. "P001"), sentence_level rows use a finer `uniq_id` per sentence within
    that paragraph (e.g. "P001-S1", "P001-S2", ...) -- disjoint value spaces (no
    collision risk), so paragraph- and sentence-level rows can be combined into
    ONE dict per language. Each row (whether a whole paragraph or a single
    sentence) becomes its own independent group -- this is a concatenation of
    the two levels' rows, not a merge of a sentence into its parent paragraph's
    entry.

    UNION across languages, not intersection: a group is built for every
    uniq_id that AT LEAST ONE requested language has, containing whichever
    subset of those languages actually covers that id -- NOT the full
    N-way-intersection join _load_ngram_parallel uses for TRAINING data. That
    distinction matters and is deliberate: training's fairness loss terms
    (fanta.model.fairness_loss, rate_anchor_loss, flexitokens' boundary_hinge_loss)
    compare languages' compression rates WITHIN one forward pass, which needs a
    group's languages to be genuinely aligned parallel content. common.eval.cross_tokenizer.
    evaluate_on_groups has no such requirement -- it accumulates each language's
    stats independently across every group containing that language, the same
    way macro-averaged metrics don't require every class to appear in every
    sample. Full-intersection across all 259 BOUQuET languages was confirmed
    (via a real FANTA test-set run) to throw away the vast majority of each
    language's own rows just because some UNRELATED language happened to be
    missing that particular id -- e.g. 1052 surviving groups for
    --eval-data-source bouquet_test langs="all", versus ~272k rows summed
    across languages per the dataset card. The union join uses every row every
    language actually has.

    Each {lang}.parquet has src_lang == lang and tgt_lang == eng_Latn fixed as a
    reference pivot -- so `src_text` is this file's actual per-language content,
    and `tgt_text` is a constant English gloss (the same string appears in every
    language's file for a given uniq_id, which is NOT the per-language sentence
    -- easy bug to make, caught by comparing files directly).

    Defaults to langs="all" (every one of BOUQuET's real 259 languages,
    confirmed via list_repo_files, keyed by its full lang_Script stem) --
    same "all" convention _load_ngram_parallel (load_oldi_seed/
    load_flores_plus) already uses. common.eval.cross_tokenizer's
    evaluate_on_groups already skips languages a given checkpoint has no
    entry for, so passing "all" here is always safe -- and every real call
    site in this repo already does, explicitly, regardless of this
    function's own default.
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
    """The genuinely held-out counterpart to load_bouquet_dev -- reserve this for
    FINAL reported numbers; use load_bouquet_dev for any hyperparameter tuning or
    exploratory comparison, to avoid the equivalent of test-set leakage from
    repeatedly checking results against the same held-out data decisions get
    tuned against."""
    return _load_bouquet_split("test", langs)
