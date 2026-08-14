"""Cross-lingual byte-length parity: how many bytes a language needs, on average, to
say the same thing an anchor language (English by default) says, computed directly
from genuinely parallel groups (common.data.oldi_data's loaders, or any other {lang: text}
dict list sharing content across languages).

Extracted from flexitokens.train.derive_alpha_beta, which was the first place this
ratio was computed (for FlexiTokens' own per-language alpha_L/beta_L band). Shared
here so any tokenizer's rate-control mechanism can derive a per-language, rather than
a single global, target compression rate from the same evidence -- see "Compute
Optimal Tokenization" (Limisiewicz et al. 2026), which finds real deployed tokenizers
systematically over-compress high-resource languages and under-compress low-resource
ones relative to the compute-optimal rate, i.e. a single global target rate is itself
a fairness problem, not a neutral simplification.

See common.eval.cross_tokenizer.evaluate_on_groups's own "token_parity" output for
the TOKEN-COUNT analog of this same ratio (this module's own ratio is computed from
raw untokenized bytes, an input-side property shared by every tokenizer scoring the
same content; token_parity is computed from a SPECIFIC tokenizer's own resulting
token counts, an output-side property that can differ between two tokenizers scoring
the exact same byte-length-parity-matched content).
"""

from collections import defaultdict


def _byte_len(text):
    return len(text.encode("utf-8")) if isinstance(text, str) else len(text)


def _short_code(lang):
    """'eng_Latn' -> 'eng', 'apc_Arab_nort3139' -> 'apc', 'eng' -> 'eng'. Language
    keys in this project come in two conventions depending on data source (see
    common.data.oldi_data._load_ngram_parallel's docstring): a bare ISO code (smol,
    and every curated fixed-langs load) or a full lang_Script[_variant] stem
    (oldi_seed/flores_plus under langs="all"). Splitting on the first
    underscore recovers the ISO code either way, since stems always start with
    it."""
    return lang.split("_", 1)[0]


def _find_anchor_key(group, anchor_lang):
    """Returns whichever key in `group` represents `anchor_lang`, checking an
    exact match first, then a short-code match (see _short_code) -- needed
    because a single pooled train_groups list (common.data.cli_data.
    load_groups, e.g. --data-source all or a single oldi_seed/flores_dev
    source, both now defaulting to langs="all") mixes groups keyed by bare
    code with groups keyed by full stem, and a single group only ever uses
    ONE of the two conventions. Returns None if `anchor_lang` isn't present
    in this group under either convention."""
    if anchor_lang in group:
        return anchor_lang
    for lang in group:
        if _short_code(lang) == anchor_lang:
            return lang
    return None


def compute_lang_parity_ratios(train_groups, anchor_lang="eng"):
    """Returns (ratio_by_lang: dict[str, float], anchor: str).

    ratio_by_lang[lang] = mean(byte_len(lang)) / mean(byte_len(anchor)), over every
    group containing BOTH lang and the anchor -- i.e. how many more (or fewer) bytes
    `lang` needs, on average, to express the same content the anchor does. ratio > 1
    means `lang` is less byte-dense than the anchor for equivalent content (e.g.
    multi-byte UTF-8 scripts, more verbose morphology); ratio ~= 1.0 for a language
    never paired with the anchor in any group (no evidence of a disparity, so it's
    treated like the anchor rather than penalized/boosted on no evidence).

    The anchor is located PER GROUP via _find_anchor_key, not by a single fixed
    key -- confirmed (via a real wandb run's logged target_rate_by_lang) that
    without this, pooling langs="all" oldi_seed/flores_plus groups (keyed by
    full lang_Script stem, e.g. "eng_Latn") together with smol groups (keyed by
    bare code, "eng") silently discarded EVERY stem-keyed group from pairing --
    anchor_lang="eng" never matched "eng_Latn" -- leaving ~95% of trained
    languages defaulted to the uninformative ratio=1.0 while a handful of
    bare-coded languages got real, very different ratios. That mismatch (most
    languages anchored to a flat target, a few anchored to genuinely disparate
    targets) directly fights the Gini fairness term, which wants all languages
    present in a batch to compress SIMILARLY to each other.

    Falls back to the first language present if anchor_lang isn't in train_groups at
    all, under either key convention (prints nothing -- callers that care about the
    fallback, like flexitokens.train.derive_alpha_beta, print their own notice using
    the returned `anchor`). Raises ValueError on empty train_groups.
    """
    lengths_by_lang = defaultdict(list)
    for group in train_groups:
        for lang, text in group.items():
            lengths_by_lang[lang].append(_byte_len(text))
    if not lengths_by_lang:
        raise ValueError(
            "train_groups is empty -- cannot derive parity ratios from no data"
        )

    paired_anchor_lengths = defaultdict(list)
    paired_lang_lengths = defaultdict(list)
    anchor_found = False
    for group in train_groups:
        anchor_key = _find_anchor_key(group, anchor_lang)
        if anchor_key is None:
            continue
        anchor_found = True
        anchor_len = _byte_len(group[anchor_key])
        for lang, text in group.items():
            paired_anchor_lengths[lang].append(anchor_len)
            paired_lang_lengths[lang].append(_byte_len(text))

    anchor = anchor_lang if anchor_found else next(iter(lengths_by_lang))

    ratio_by_lang = {}
    for lang in lengths_by_lang:
        anchor_lens = paired_anchor_lengths.get(lang, [])
        lang_lens = paired_lang_lengths.get(lang, [])
        if anchor_lens and sum(anchor_lens) > 0:
            ratio_by_lang[lang] = (sum(lang_lens) / len(lang_lens)) / (
                sum(anchor_lens) / len(anchor_lens)
            )
        else:
            ratio_by_lang[lang] = 1.0
    return ratio_by_lang, anchor
