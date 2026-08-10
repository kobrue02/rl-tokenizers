"""Shared vocabulary-collapse diagnostics -- used by every tokenizer's CLI/smoke
test in this repo (fairtok, magnet, flexitokens, manta) to report the same
"did this run degenerate toward character-level or full-sentence spans"
sanity check, regardless of how that tokenizer's training loop actually works.
"""


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
