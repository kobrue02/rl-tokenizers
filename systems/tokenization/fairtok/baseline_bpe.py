"""Plain byte-level BPE baseline, trained from scratch on raw byte sequences.

Not HuggingFace `tokenizers`: HF's ByteLevel pre-tokenizer re-encodes strings as
UTF-8 internally, which breaks the 1:1 byte correspondence our synthetic (non-UTF-8)
corpus needs for values >127. Swap in HF `tokenizers` once real text corpora are wired in.
"""

from collections import Counter

from tqdm.auto import tqdm


def _apply_merge(seq, a, b, new_id):
    out = []
    i = 0
    n = len(seq)
    while i < n:
        if i < n - 1 and seq[i] == a and seq[i + 1] == b:
            out.append(new_id)
            i += 2
        else:
            out.append(seq[i])
            i += 1
    return out


def train_byte_bpe(byte_sequences, vocab_size):
    """Returns (vocab: id -> bytes, merges: list of (a, b, new_id))."""
    vocab = {i: bytes([i]) for i in range(256)}
    sequences = [list(seq) for seq in byte_sequences]
    next_id = 256
    merges = []

    # O(vocab_size * corpus_size): rescans every sequence on every merge. Fine for a
    # few thousand sample sentences (see _plain_bpe_target_rate), not for a full corpus.
    pbar = tqdm(
        total=vocab_size - 256, desc="plain-BPE baseline (merges)", unit="merge"
    )
    while len(vocab) < vocab_size:
        pair_counts = Counter()
        for seq in sequences:
            for a, b in zip(seq, seq[1:]):
                pair_counts[(a, b)] += 1
        if not pair_counts:
            break
        (a, b), _ = pair_counts.most_common(1)[0]
        vocab[next_id] = vocab[a] + vocab[b]
        merges.append((a, b, next_id))
        sequences = [_apply_merge(seq, a, b, next_id) for seq in sequences]
        next_id += 1
        pbar.update(1)
    pbar.close()

    return vocab, merges


def encode_with_merges(byte_seq, merges):
    seq = list(byte_seq)
    for a, b, new_id in merges:
        seq = _apply_merge(seq, a, b, new_id)
    return seq
