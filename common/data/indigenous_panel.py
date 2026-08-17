"""A small, deliberately curated panel of Indigenous languages for a
dedicated tokenizer-fairness comparison alongside BOUQuET -- unlike other
sources in common.data.corpora, this one exists specifically to probe
polysynthetic/highly-synthetic morphology, not to cover "every language a
source offers" (there is no "all" for this source).

Every entry pairs one Indigenous language with an ANCHOR language it's
already aligned against in its own source -- English for crk/iu, Spanish
for the nine AmericasNLP shared-task languages (their own pivot language,
per https://github.com/AmericasNLP/americasnlp2021, not English). This is
a genuine mixed-anchor panel: common.eval.parity.anchor_invariant_parity's
gm_relative/spread metrics are anchor-agnostic by construction, so mixing
anchors is fine as long as callers keep each pair's own anchor field
visible (PAIRS[*]["anchor"]) rather than treating the whole panel as
English-anchored.

MORPHOLOGY: not every language here is equally "polysynthetic" under
standard typological classification -- Plains Cree, Inuktitut, Nahuatl,
Wixarika, Cherokee, Mapudungun, and (to varying degrees) Bribri, Rarámuri,
Shipibo-Konibo, and Asháninka are most consistently described as
polysynthetic; Quechua, Aymara, and Guaraní are more standardly
agglutinative (highly synthetic but not usually "polysynthetic" in the
stricter sense). Māori is a further exception in the other direction --
typologically closer to isolating/analytic than either category -- tagged
"agglutinative" here only as the nearest bucket this project's coarse
three-way scheme offers, included for broader Indigenous-language coverage
rather than because it fits this panel's polysynthesis theme narrowly. The
"morphology" field is this project's own best-effort classification -- a
starting point for figure grouping/filtering, not a citable typological
claim.

PROVENANCE:
  - crk-en (Plains Cree): KonradBRG/plains-cree-figurative on HF -- 228
    human-verified ("gold") + 10,619 LLM-labeled ("silver") sentence pairs
    from Bloomfield's 1934 Plains Cree Texts. CC-BY-4.0. Both splits are
    used (the figurative-language labels themselves are unused extra
    columns -- only text_cree/text_en matter).
  - iu-en (Inuktitut): The Nunavut Hansard Inuktitut-English Parallel
    Corpus 3.0.1 (NRC Digital Repository, CC BY 4.0) -- Legislative
    Assembly of Nunavut proceedings, 1999-2017, ~1.3M sentence pairs total.
    This panel uses only the held-out "test" split (13,082 pairs), matching
    the scale other panel entries use for a fairness comparison (not an MT
    training run).
  - the next nine pairs: AmericasNLP 2021 shared task
    (github.com/AmericasNLP/americasnlp2021), each language's own
    train.{code}/train.es line-aligned files, fetched via
    raw.githubusercontent.com (no git clone -- see prepare_indigenous_panel).
  - chr-en (Cherokee): ChrEn (Zhang, Frey & Bansal, EMNLP 2020,
    github.com/ZhangShiyue/ChrEn), data/parallel/{train,dev,test}.{chr,en}
    combined (not out_dev/out_test, the paper's separate out-of-domain eval
    split -- a different distribution). No LICENSE file in the source
    repo -- an academic research release with no declared license; used
    here for non-commercial research only, cite the paper.
  - mi-en (Māori): jinglishi0206/Maori_English_New_Zealand on HF -- 6,486
    sentence pairs scraped from Te Ara (the Encyclopedia of New Zealand).
    CC-BY-NC-3.0, dataset card states research-use-only.
  - arn-es (Mapudungun): the AVENUE project's Mapudungun corpus (CMU,
    Chilean Ministry of Education, Instituto de Estudios Indígenas at
    Universidad de La Frontera; github.com/mingjund/mapudungun-corpus) --
    translation-clean/*.txt per-recording transcripts (M:/C: line-paired
    Mapudungun/Castellano utterances), combining the training/dev/test file
    lists under dataset_splits/mt/. CC-BY-NC-SA-3.0 (research use,
    ShareAlike on any redistribution of derived data -- moot here since
    data/ is gitignored, see prepare_indigenous_panel).
"""

# code: this pair's ISO/AmericasNLP-native language code (the key used in
# every yielded {lang: text} group).
# anchor: the other language this pair is aligned against (its own key in
# the same group).
# family: genealogical language family (Ethnologue/Glottolog-style naming,
# standard textbook classification, not independently verified).
# morphology: this project's own best-effort tag -- see module docstring's
# MORPHOLOGY section.
# loader: which of prepare_indigenous_panel's loader functions builds this
# pair ("hf_cree", "nrc_hansard", or "americasnlp").
# dir: (americasnlp only) the pair's directory name in the americasnlp2021
# repo.
PAIRS = {
    "crk-en": {
        "language": "Plains Cree",
        "code": "crk",
        "anchor": "en",
        "family": "Algonquian",
        "morphology": "polysynthetic",
        "loader": "hf_cree",
    },
    "iu-en": {
        "language": "Inuktitut",
        "code": "iu",
        "anchor": "en",
        "family": "Eskimo-Aleut",
        "morphology": "polysynthetic",
        "loader": "nrc_hansard",
    },
    "nah-es": {
        "language": "Nahuatl",
        "code": "nah",
        "anchor": "es",
        "family": "Uto-Aztecan",
        "morphology": "polysynthetic",
        "loader": "americasnlp",
        "dir": "nahuatl-spanish",
    },
    "hch-es": {
        "language": "Wixarika (Huichol)",
        "code": "hch",
        "anchor": "es",
        "family": "Uto-Aztecan",
        "morphology": "polysynthetic",
        "loader": "americasnlp",
        "dir": "wixarika-spanish",
    },
    "oto-es": {
        "language": "Hñähñu (Otomi)",
        "code": "oto",
        "anchor": "es",
        "family": "Oto-Manguean",
        "morphology": "agglutinative",  # not standardly polysynthetic --
        # included for AmericasNLP-panel completeness (see module docstring).
        "loader": "americasnlp",
        "dir": "hñähñu-spanish",
    },
    "gn-es": {
        "language": "Guaraní",
        "code": "gn",
        "anchor": "es",
        "family": "Tupian",
        "morphology": "agglutinative",
        "loader": "americasnlp",
        "dir": "guarani-spanish",
    },
    "bzd-es": {
        "language": "Bribri",
        "code": "bzd",
        "anchor": "es",
        "family": "Chibchan",
        "morphology": "polysynthetic",
        "loader": "americasnlp",
        "dir": "bribri-spanish",
    },
    "quy-es": {
        "language": "Quechua (Ayacucho)",
        "code": "quy",
        "anchor": "es",
        "family": "Quechuan",
        "morphology": "agglutinative",
        "loader": "americasnlp",
        "dir": "quechua-spanish",
    },
    "aym-es": {
        "language": "Aymara",
        "code": "aym",
        "anchor": "es",
        "family": "Aymaran",
        "morphology": "agglutinative",
        "loader": "americasnlp",
        "dir": "aymara-spanish",
    },
    "tar-es": {
        "language": "Rarámuri",
        "code": "tar",
        "anchor": "es",
        "family": "Uto-Aztecan",
        "morphology": "polysynthetic",
        "loader": "americasnlp",
        "dir": "raramuri-spanish",
    },
    "shp-es": {
        "language": "Shipibo-Konibo",
        "code": "shp",
        "anchor": "es",
        "family": "Panoan",
        "morphology": "polysynthetic",
        "loader": "americasnlp",
        "dir": "shipibo_konibo-spanish",
    },
    "cni-es": {
        "language": "Asháninka",
        "code": "cni",
        "anchor": "es",
        "family": "Arawakan",
        "morphology": "polysynthetic",
        "loader": "americasnlp",
        "dir": "ashaninka-spanish",
    },
    "chr-en": {
        "language": "Cherokee",
        "code": "chr",
        "anchor": "en",
        "family": "Iroquoian",
        "morphology": "polysynthetic",
        "loader": "chren",
    },
    "mi-en": {
        "language": "Māori",
        "code": "mi",
        "anchor": "en",
        "family": "Polynesian",
        "morphology": "agglutinative",  # see module docstring's MORPHOLOGY
        # section -- typologically closer to isolating, tagged this way only
        # as the nearest bucket this project's coarse scheme offers.
        "loader": "hf_csv",
    },
    "arn-es": {
        "language": "Mapudungun",
        "code": "arn",
        "anchor": "es",
        "family": "Araucanian",
        "morphology": "polysynthetic",
        "loader": "mapudungun",
    },
}

AMERICASNLP_REPO = "AmericasNLP/americasnlp2021"
AMERICASNLP_BRANCH = "main"

HF_CREE_REPO = "KonradBRG/plains-cree-figurative"

NRC_HANSARD_URL = (
    "https://nrc-digital-repository.canada.ca/eng/view/dataset/"
    "?id=c7e34fa7-7629-43c2-bd6d-19b32bf64f60"
)
NRC_HANSARD_ARCHIVE_ROOT = "Nunavut-Hansard-Inuktitut-English-Parallel-Corpus-3.0"

CHREN_REPO = "ZhangShiyue/ChrEn"
CHREN_BRANCH = "main"

MAORI_HF_REPO = "jinglishi0206/Maori_English_New_Zealand"

MAPUDUNGUN_REPO = "mingjund/mapudungun-corpus"
MAPUDUNGUN_BRANCH = "master"
