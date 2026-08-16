"""Cross-lingual byte-length parity: how many bytes a language needs, on average, to
say the same thing an anchor language (English by default) says, computed from
genuinely parallel groups (common.data.oldi_data's loaders, or any {lang: text} dict
list sharing content across languages).

Extracted from flexitokens.train.derive_alpha_beta (FlexiTokens' per-language
alpha_L/beta_L band). Shared so any tokenizer's rate-control mechanism can derive a
per-language, rather than single global, target compression rate -- per "Compute
Optimal Tokenization" (Limisiewicz et al. 2026), a flat global target itself
over-compresses high-resource and under-compresses low-resource languages relative
to the compute-optimal rate.

Note: this ratio is computed from raw untokenized bytes (input-side, shared across
every tokenizer). common.eval.cross_tokenizer.evaluate_on_groups's "token_parity" is the
output-side analog, computed from a specific tokenizer's resulting token counts,
which can differ between tokenizers even on byte-length-parity-matched content.
"""

import math
from collections import defaultdict


def _byte_len(text):
    return len(text.encode("utf-8")) if isinstance(text, str) else len(text)


def _short_code(lang):
    """'eng_Latn' -> 'eng', 'apc_Arab_nort3139' -> 'apc', 'eng' -> 'eng'. Language
    keys come in two conventions depending on data source: bare ISO code, or full
    lang_Script[_variant] stem. Splitting on the first underscore recovers the ISO
    code either way, since stems always start with it."""
    return lang.split("_", 1)[0]


def _find_anchor_key(group, anchor_lang):
    """Returns whichever key in `group` represents `anchor_lang`: exact match
    first, then a short-code match (see _short_code). Needed because a pooled
    train_groups list can mix groups keyed by bare code with groups keyed by
    full stem, though any single group only uses one convention. Returns None
    if `anchor_lang` isn't present under either convention."""
    if anchor_lang in group:
        return anchor_lang
    for lang in group:
        if _short_code(lang) == anchor_lang:
            return lang
    return None


def compute_lang_parity_ratios(train_groups, anchor_lang="eng"):
    """Returns (ratio_by_lang: dict[str, float], anchor: str).

    ratio_by_lang[lang] = mean(byte_len(lang)) / mean(byte_len(anchor)), over every
    group containing BOTH lang and the anchor. ratio > 1 means `lang` is less
    byte-dense than the anchor for equivalent content (e.g. multi-byte UTF-8
    scripts, more verbose morphology); ratio ~= 1.0 for a language never paired
    with the anchor (no evidence of a disparity, so treated neutrally).

    The anchor is located PER GROUP via _find_anchor_key, not by a single fixed
    key: pooling groups keyed by full lang_Script stem (e.g. "eng_Latn") together
    with groups keyed by bare code ("eng") would otherwise silently discard every
    stem-keyed group from pairing, defaulting most languages to the uninformative
    ratio=1.0 while a handful got real ratios -- directly fighting the Gini
    fairness term, which wants all languages to compress similarly to each other.

    Falls back to the first language present if anchor_lang isn't in train_groups
    at all, under either convention (prints nothing; callers like
    flexitokens.train.derive_alpha_beta print their own notice using the returned
    `anchor`). Raises ValueError on empty train_groups.
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


def anchor_invariant_parity(ratio_by_lang):
    """Returns (gm_relative: dict[str, float], spread: float), computed from an
    existing anchor-relative ratio dict (e.g. this module's own ratio_by_lang, or
    cross_tokenizer.evaluate_on_groups's token_parity) WITHOUT needing to know which
    language was used as that ratio's anchor.

    A single fixed anchor (English, by convention) silently assumes that language's
    cost is fairness's "1.0" baseline. If a tokenizer is unusually good or bad
    specifically at the anchor, that inverts every other language's ratio (e.g.
    re-anchoring an English-anchored comparison to Mandarin can flip a
    Chinese-optimized tokenizer from best to worst, though no model's actual
    per-language cost changed).

    gm_relative[lang] = ratio_by_lang[lang] / geometric_mean(ratio_by_lang.values())
    replaces the fixed anchor with the geometric mean of every language present --
    provably anchor-invariant (re-deriving ratio_by_lang from any other anchor
    yields the same gm_relative, up to float noise), since GM-normalizing divides
    out whatever the original anchor was. spread = max/min needs no GM step: a
    common anchor divides out of both directly, so it's already anchor-invariant.

    Skips (excludes from both outputs) any non-positive ratio, since log() is
    undefined there; only reachable from a genuinely empty-content edge case.
    """
    positive = {lang: r for lang, r in ratio_by_lang.items() if r > 0}
    if not positive:
        return {}, 1.0

    gm = math.exp(sum(math.log(r) for r in positive.values()) / len(positive))
    gm_relative = {lang: r / gm for lang, r in positive.items()}
    spread = max(positive.values()) / min(positive.values())
    return gm_relative, spread
