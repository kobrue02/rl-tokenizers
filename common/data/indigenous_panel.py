"""A small, DELIBERATELY curated panel of Indigenous languages for a
dedicated tokenizer-fairness comparison alongside BOUQuET -- unlike every
other source in common.data.corpora, this one exists specifically to probe
polysynthetic/highly-synthetic morphology, not to cover "every language a
source natively offers" (there is no "all" for this source; see
common.data.corpora's own LOCAL_BITEXT_SOURCES docstring section for why).

Every entry pairs one Indigenous language with an ANCHOR language it comes
already aligned against in its own source -- English for crk/iu, Spanish
for the nine AmericasNLP shared-task languages (their own pivot language,
confirmed directly from https://github.com/AmericasNLP/americasnlp2021,
not English). This is a genuine mixed-anchor panel, not an oversight:
common.eval.parity.anchor_invariant_parity's own gm_relative/spread metrics
are anchor-agnostic by construction, so mixing anchors doesn't break the
underlying comparison -- callers that report results per-pair should just
keep each pair's own anchor field visible (see PAIRS[*]["anchor"]) rather
than silently treating the whole panel as English-anchored.

MORPHOLOGY, stated plainly: not every language here is equally
"polysynthetic" under standard typological classification -- Plains Cree,
Inuktitut, Nahuatl, Wixarika, and (to varying degrees discussed in the
literature) Bribri, Rarámuri, Shipibo-Konibo, and Asháninka are the ones
most consistently described as polysynthetic; Quechua, Aymara, and
Guaraní are more standardly classified as agglutinative (highly synthetic,
but not usually "polysynthetic" in the stricter sense). The "morphology"
field below records this project's own best-effort classification, not a
rigorously sourced typological survey -- treat it as a starting point for
figure grouping/filtering, not a citable claim.

PROVENANCE (verified live against each real source, not guessed):
  - crk-en (Plains Cree): KonradBRG/plains-cree-figurative on HF -- 228
    human-verified ("gold") + 10,619 LLM-labeled ("silver") sentence pairs
    from Bloomfield's 1934 Plains Cree Texts. CC-BY-4.0. Both splits are
    used here (the figurative-language labels themselves are unused extra
    columns for tokenizer-fairness purposes -- only text_cree/text_en
    matter).
  - iu-en (Inuktitut): The Nunavut Hansard Inuktitut-English Parallel
    Corpus 3.0.1 (NRC Digital Repository, CC BY 4.0) -- Legislative
    Assembly of Nunavut proceedings, 1999-2017, ~1.3M sentence pairs total.
    This panel uses only the corpus's own held-out "test" split (13,082
    pairs) rather than the full training-scale corpus, matching the scale
    BOUQuET/other panel entries already use for a fairness comparison
    (this isn't an MT training run).
  - the remaining nine pairs: AmericasNLP 2021 shared task
    (github.com/AmericasNLP/americasnlp2021), each language's own
    train.{code}/train.es line-aligned text files, fetched directly via
    raw.githubusercontent.com (no git clone -- see prepare_indigenous_panel
    for why).
"""

# code: this pair's own ISO/AmericasNLP-native language code (the key used
# in every yielded {lang: text} group -- see common.data.corpora's
# LOCAL_BITEXT_SOURCES docstring section).
# anchor: the OTHER language this pair is aligned against (its own key in
# the same group).
# family: genealogical language family (Ethnologue/Glottolog-style naming,
# not verified against either directly -- standard textbook classification).
# morphology: this project's own best-effort tag -- see module docstring's
# own MORPHOLOGY section for what this is and isn't.
# loader: which of prepare_indigenous_panel's loader functions builds this
# pair ("hf_cree", "nrc_hansard", or "americasnlp").
# dir: (americasnlp only) the pair's own directory name in the
# americasnlp2021 repo.
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
        "morphology": "agglutinative",  # not standardly classified as
        # polysynthetic -- included for AmericasNLP-panel completeness, not
        # because it's a strong polysynthesis example (see module docstring).
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
}

AMERICASNLP_REPO = "AmericasNLP/americasnlp2021"
AMERICASNLP_BRANCH = "main"

HF_CREE_REPO = "KonradBRG/plains-cree-figurative"

NRC_HANSARD_URL = (
    "https://nrc-digital-repository.canada.ca/eng/view/dataset/"
    "?id=c7e34fa7-7629-43c2-bd6d-19b32bf64f60"
)
NRC_HANSARD_ARCHIVE_ROOT = "Nunavut-Hansard-Inuktitut-English-Parallel-Corpus-3.0"
