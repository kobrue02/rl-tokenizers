"""Two-stage fixed-vocabulary extraction, shared by every tokenizer in this repo.

The RL loop never enforces V_MAX in-loop (that would make the reward nonstationary --
see the fixed-vocab-size discussion). Instead it trains against a compression-rate
target, and the hard vocabulary budget is applied once, after training, by keeping the
V_MAX most frequent distinct byte spans seen across all languages. The differentiable
baselines (magnet/flexitokens/manta) don't have this nonstationary-reward concern at
all, but use the exact same two-stage extraction anyway so every tokenizer's final
vocabulary is produced identically and is directly comparable.

Spans are byte strings, so `Counter` keys them by content already -- there is no
arbitrary id to accidentally alias two spans onto, which is the invariant that keeps
this immune to Duplication-BPE-style gaming (see common.bytes_utils.spans_from_boundaries).
"""

import json
from collections import Counter


def _bytes_to_unicode():
    """The GPT-2 / HuggingFace byte-level BPE trick: map every byte value 0-255
    to its own printable unicode character, so any byte string (including one
    that isn't valid UTF-8 on its own -- a boundary policy can place a cut in the
    middle of a multi-byte character) becomes a safe, reversible, JSON-writable
    string. Same scheme `vocab.json` uses in a real HF byte-level tokenizer."""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("\xa1"), ord("\xac") + 1))
        + list(range(ord("\xae"), ord("\xff") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs)))


BYTE_TO_UNICODE = _bytes_to_unicode()


def span_to_token_string(span):
    return "".join(BYTE_TO_UNICODE[b] for b in span)


def vocab_with_stats(token_freq_by_lang, k):
    """Top-k spans by total frequency across all languages, each with its total
    count and a per-language breakdown -- the breakdown is what makes a vocab
    entry inspectable: whether it's genuinely shared across languages or an
    artifact of one language dominating the running frequency table.
    Returns a list of (span: bytes, total_count: int, per_lang: dict[str, int]),
    sorted by total_count descending.
    """
    merged = Counter()
    for counter in token_freq_by_lang.values():
        merged.update(counter)
    top = merged.most_common(k)
    return [
        (
            span,
            total,
            {lang: c[span] for lang, c in token_freq_by_lang.items() if c[span] > 0},
        )
        for span, total in top
    ]


def top_k_by_frequency(token_freq_by_lang, k):
    merged = Counter()
    for counter in token_freq_by_lang.values():
        merged.update(counter)
    return {span for span, _ in merged.most_common(k)}


def vocab_snapshot_stats(token_freq_by_lang, k):
    """Cheap periodic snapshot for progress tracking (coverage, cross-lingual
    sharing) -- deliberately avoids the full per-language breakdown
    vocab_with_stats computes (that's O(k * num_languages), fine to pay once
    at the end for vocab_stats.json, too expensive to pay every fairness
    refresh once many languages are in play, e.g. langs="all" pulling in
    dozens-to-hundreds of distinct language keys over a long run).

    Returns (top_spans: set[bytes], coverage: float, cross_lingual_share: float).
    coverage = fraction of all tokenized content (by frequency) the top-k spans
    capture. cross_lingual_share = fraction of the top-k spans used (count > 0)
    by more than one language -- an early exit after finding a 2nd language
    keeps this cheap for the common case (frequent spans are usually shared).
    """
    merged = Counter()
    for c in token_freq_by_lang.values():
        merged.update(c)
    total = sum(merged.values())
    top = merged.most_common(k)
    top_spans = {span for span, _ in top}

    top_total = sum(count for _, count in top)
    coverage = top_total / total if total else 0.0

    shared = 0
    for span, _ in top:
        seen = 0
        for c in token_freq_by_lang.values():
            if c.get(span, 0) > 0:
                seen += 1
                if seen > 1:
                    break
        if seen > 1:
            shared += 1
    cross_lingual_share = shared / len(top) if top else 0.0

    return top_spans, coverage, cross_lingual_share


def vocab_churn(prev_spans, curr_spans):
    """Jaccard overlap between two top-k span sets -- rising toward 1 as the
    vocabulary stops reshuffling and settles, independent of what the loss
    curve says."""
    if not prev_spans and not curr_spans:
        return 1.0
    union = prev_spans | curr_spans
    if not union:
        return 1.0
    return len(prev_spans & curr_spans) / len(union)


def save_vocab_json(entries, path):
    """Writes vocab.json in the same shape a HuggingFace byte-level tokenizer
    would: a flat {token_string: id} mapping, ids assigned in the given order
    (entries is expected sorted by frequency descending, as vocab_with_stats
    returns -- so id 0 is the most frequent span). Token strings use the same
    byte<->unicode encoding HF's ByteLevel tokenizers use, so this round-trips
    even for spans that aren't valid UTF-8 on their own.
    """
    vocab = {span_to_token_string(span): i for i, (span, _, _) in enumerate(entries)}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)


def save_vocab_stats(entries, path):
    """The richer companion file vocab.json intentionally doesn't carry: per-entry
    frequency and per-language usage breakdown, for actually inspecting fairness
    (is this token shared across languages, or a single language's artifact?)."""
    records = [
        {
            "token": span_to_token_string(span),
            "total_count": total,
            "per_lang": per_lang,
        }
        for span, total, per_lang in entries
    ]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def report_and_save_vocab(token_freq, vocab_size, vocab_out, vocab_stats_out, vocab_preview):
    """The final-vocab preview/save step every systems/*/cli.py runs identically
    after training, extracted verbatim from that repeated tail (confirmed
    byte-identical in substance across all seven, only cosmetic whitespace
    differed) -- terminal preview of the `vocab_preview` most frequent entries,
    then vocab.json (if `vocab_out`) and the richer vocab_stats.json (if
    `vocab_stats_out`), both skipped by passing "" (empty string), matching
    each flag's own existing --vocab-out/--vocab-stats-out convention.
    Returns `entries` (vocab_with_stats(token_freq, vocab_size)'s own return
    shape) in case a caller wants it after this -- none currently do, but
    nothing here needs entries to be recomputed to add that later.
    """
    entries = vocab_with_stats(token_freq, vocab_size)
    if vocab_preview:
        print(f"\ntop {min(vocab_preview, len(entries))} vocab entries by frequency:")
        for span, total, per_lang in entries[:vocab_preview]:
            langs = ", ".join(
                f"{lang}:{c}" for lang, c in sorted(per_lang.items(), key=lambda kv: -kv[1])
            )
            print(f"  {total:6d}  {span!r:20s} [{langs}]")
    if vocab_out:
        save_vocab_json(entries, vocab_out)
        print(f"\nsaved vocab ({len(entries)} entries) to {vocab_out}")
    if vocab_stats_out:
        save_vocab_stats(entries, vocab_stats_out)
        print(f"saved per-entry frequency/language stats to {vocab_stats_out}")
    return entries
