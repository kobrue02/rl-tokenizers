"""Cross-lingual byte-length parity: how many bytes a language needs, on average, to
say the same thing an anchor language (English by default) says, computed directly
from genuinely parallel groups (common.oldi_data's loaders, or any other {lang: text}
dict list sharing content across languages).

Extracted from flexitokens.train.derive_alpha_beta, which was the first place this
ratio was computed (for FlexiTokens' own per-language alpha_L/beta_L band). Shared
here so any tokenizer's rate-control mechanism can derive a per-language, rather than
a single global, target compression rate from the same evidence -- see "Compute
Optimal Tokenization" (Limisiewicz et al. 2026), which finds real deployed tokenizers
systematically over-compress high-resource languages and under-compress low-resource
ones relative to the compute-optimal rate, i.e. a single global target rate is itself
a fairness problem, not a neutral simplification.
"""

from collections import defaultdict


def _byte_len(text):
    return len(text.encode("utf-8")) if isinstance(text, str) else len(text)


def compute_lang_parity_ratios(train_groups, anchor_lang="eng"):
    """Returns (ratio_by_lang: dict[str, float], anchor: str).

    ratio_by_lang[lang] = mean(byte_len(lang)) / mean(byte_len(anchor)), over every
    group containing BOTH lang and the anchor -- i.e. how many more (or fewer) bytes
    `lang` needs, on average, to express the same content the anchor does. ratio > 1
    means `lang` is less byte-dense than the anchor for equivalent content (e.g.
    multi-byte UTF-8 scripts, more verbose morphology); ratio ~= 1.0 for a language
    never paired with the anchor in any group (no evidence of a disparity, so it's
    treated like the anchor rather than penalized/boosted on no evidence).

    Falls back to the first language present if anchor_lang isn't in train_groups at
    all (prints nothing -- callers that care about the fallback, like
    flexitokens.train.derive_alpha_beta, print their own notice using the returned
    `anchor`). Raises ValueError on empty train_groups.
    """
    lengths_by_lang = defaultdict(list)
    for group in train_groups:
        for lang, text in group.items():
            lengths_by_lang[lang].append(_byte_len(text))

    anchor = (
        anchor_lang
        if anchor_lang in lengths_by_lang
        else next(iter(lengths_by_lang), None)
    )
    if anchor is None:
        raise ValueError(
            "train_groups is empty -- cannot derive parity ratios from no data"
        )

    paired_anchor_lengths = defaultdict(list)
    paired_lang_lengths = defaultdict(list)
    for group in train_groups:
        if anchor not in group:
            continue
        anchor_len = _byte_len(group[anchor])
        for lang, text in group.items():
            paired_anchor_lengths[lang].append(anchor_len)
            paired_lang_lengths[lang].append(_byte_len(text))

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
