"""Shared vocabulary-collapse and per-language fairness diagnostics -- used by every
tokenizer's CLI/smoke test in this repo (fairtok, magnet, flexitokens, manta) to
report the same checks, regardless of how that tokenizer's training loop works.
"""

from collections import defaultdict


def avg_span_length(token_freq):
    total_spans = sum(sum(c.values()) for c in token_freq.values())
    total_len = sum(len(s) * n for c in token_freq.values() for s, n in c.items())
    return total_len / total_spans if total_spans else 0.0


def collapse_stats(token_freq, final_vocab):
    return avg_span_length(token_freq), len(final_vocab)


def report_collapse(token_freq, final_vocab):
    avg_len, final_vocab_size = collapse_stats(token_freq, final_vocab)
    print(f"\nfinal vocab size={final_vocab_size}  avg span length={avg_len:.2f} bytes")
    if avg_len < 1.2:
        print("WARNING: near character-level collapse")
    elif avg_len > 40:
        print("WARNING: near full-sentence collapse")


def _word_count(text):
    # str.split()/bytes.split() both split on whitespace with no args -- using
    # bytes.split() directly (rather than decoding first) means this works
    # unchanged on common.data's synthetic placeholder corpus, whose "sentences"
    # are raw random bytes and not guaranteed to be valid UTF-8.
    if isinstance(text, str):
        return len(text.split())
    return len(bytes(text).split())


def fertility_by_lang(token_freq, train_groups):
    """Per-language fertility (see common.metrics.fertility): total tokens emitted for
    that language (sum of token_freq[lang]'s span counts) divided by total words in
    that language's raw training text (whitespace-split word count over every group's
    text for that language) -- both aggregated over the WHOLE corpus, not averaged
    per-sentence, matching how the tokenizer-fairness literature reports it. Reuses
    token_freq/train_groups already available at every tokenizer's post-training
    reporting call site -- no re-tokenization needed."""
    from common.metrics import fertility

    word_counts = defaultdict(int)
    for group in train_groups:
        for lang, text in group.items():
            word_counts[lang] += _word_count(text)
    return {
        lang: fertility(sum(counter.values()), word_counts.get(lang, 0))
        for lang, counter in token_freq.items()
    }


def report_fertility(fertility_by_lang_dict):
    if not fertility_by_lang_dict:
        return
    print("\nfertility (tokens/word) by language:")
    for lang in sorted(fertility_by_lang_dict):
        print(f"  {lang}: {fertility_by_lang_dict[lang]:.2f}")


def report_stability(stability_by_lang_dict):
    if not stability_by_lang_dict:
        return
    print("\nboundary stability under 10% random byte deletion (1.0 = fully stable):")
    for lang in sorted(stability_by_lang_dict):
        score = stability_by_lang_dict[lang]
        flag = "  <-- WARNING: unstable" if score < 0.5 else ""
        print(f"  {lang}: {score:.3f}{flag}")
