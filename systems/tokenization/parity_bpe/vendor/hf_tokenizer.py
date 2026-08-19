"""Vendored copy of build_vocab_from_merges from the official Parity-aware
BPE implementation's HF_tokenizer.py (github.com/swiss-ai/parity-aware-bpe,
MIT license) -- see parity_aware_learn_bpe.py's own module docstring for the
overall vendoring rationale.

Only this one function is reused: it's pure Python (no file I/O), converting
a learned merges list into the {token: id} vocab HuggingFace's
tokenizers.models.BPE expects, using ByteLevel.alphabet() for the base
256-ish single-byte-unicode-character vocabulary -- the same GPT-2/RoBERTa
byte<->unicode convention as common.vocab.BYTE_TO_UNICODE (see
systems.tokenization.bpe.model's own docstring, which already establishes
this equivalence for this project's OTHER classical-BPE baseline).
create_huggingface_tokenizer/load_custom_tokenizer (the rest of their
HF_tokenizer.py) are NOT reused -- they write vocab.json/merges.txt to disk
and reload from there, a round-trip this project's own model.py skips
entirely since tokenizers.models.BPE's Python bindings accept an in-memory
vocab dict + merges list directly (verified against the installed
`tokenizers` version -- see model.py).
"""


def build_vocab_from_merges(merges):
    """Creates a vocab dict from BPE merge rules.
    Args:
        merges (list[str]): Learned merge rules, one "token1 token2" pair per
            line (optionally with a leading "#version: ..." line, stripped).
    Returns:
        dict[str, int]: token -> id, base alphabet first, then one entry per
            merge in learned (== priority) order.
    """
    from tokenizers.pre_tokenizers import ByteLevel

    if merges[0].startswith("#version:"):
        merges = merges[1:]
    vocab = {}
    for idx, char in enumerate(ByteLevel.alphabet()):
        vocab[char] = idx

    index = len(vocab)
    for line in merges:
        token1, token2 = line.split()
        token1 = token1.strip()
        token2 = token2.strip()
        if token1 not in vocab:
            print(f"{token1} is not in the vocab!!!")
        if token2 not in vocab:
            print(f"{token2} is not in the vocab!!!")
        vocab[token1 + token2] = index
        index += 1
    return vocab
