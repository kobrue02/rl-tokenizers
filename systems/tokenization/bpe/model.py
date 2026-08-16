"""Standard byte-level BPE baseline -- unlike every other tokenizer here
(including superbpe/, a from-scratch reimplementation), this one wraps
HuggingFace's `tokenizers` library directly rather than reimplementing the
algorithm: plain BPE has no project-specific novelty, and `tokenizers` IS
the reference implementation (contrast superbpe/, whose two-stage
whitespace-wall-lifting mechanism has no off-the-shelf implementation).

Pre-tokenizer is ByteLevel (GPT-2/RoBERTa convention: every raw byte maps
to its own printable unicode character, BPE merges run over those) --
the same mapping as common.vocab.BYTE_TO_UNICODE (verified: round-trips
losslessly via the inverse mapping, see _token_to_bytes). Keeps BPE's
tokens byte-level, matching every other tokenizer here, rather than
`tokenizers`' default Unicode-string-level behavior.
"""

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

from common.vocab import BYTE_TO_UNICODE

_UNICODE_TO_BYTE = {v: k for k, v in BYTE_TO_UNICODE.items()}


def _to_str(text):
    # tokenizers' train_from_iterator/encode both want str, not bytes.
    # errors="replace" (not "surrogateescape"): ByteLevel's pre-tokenizer
    # RE-ENCODES the string to UTF-8 internally, and surrogate-escaped
    # codepoints can't be re-encoded (raises UnicodeEncodeError deep inside
    # the Rust extension, uncatchable per-call). "replace" avoids the crash
    # at the cost of fidelity for genuinely invalid byte runs -- an accepted
    # limitation of using tokenizers' str-based API. Every real text source
    # here is valid UTF-8 already; this only matters for
    # common.data.synthetic's placeholder corpus, which can generate invalid
    # UTF-8 -- that's why this package's smoke test uses real text instead
    # (see train.py's run_smoke_test).
    if isinstance(text, str):
        return text
    return bytes(text).decode("utf-8", errors="replace")


def _token_to_bytes(token_str):
    """Inverts the ByteLevel byte<->unicode mapping to recover a token's
    actual bytes -- Encoding.tokens gives back intermediate unicode-character
    strings, not raw bytes directly."""
    return bytes(_UNICODE_TO_BYTE[ch] for ch in token_str)


def fit_bpe(sentences, vocab_size):
    """Trains a standard byte-level BPE tokenizer over `sentences` (str/bytes,
    languages mixed -- BPE has no notion of language). Returns a BPEModel.
    min_frequency=0 (HF's default, "even a frequency-1 pair may be merged")
    is left as-is deliberately -- tightening it would starve training on
    this project's small smoke-test-scale corpora."""
    tokenizer = Tokenizer(BPE(unk_token=None))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    trainer = BpeTrainer(vocab_size=vocab_size, min_frequency=0, show_progress=False)
    tokenizer.train_from_iterator((_to_str(s) for s in sentences), trainer=trainer)
    return BPEModel(tokenizer)


class BPEModel:
    """Thin wrapper around a trained `tokenizers.Tokenizer` -- exposes
    encode_spans/num_parameters like every other tokenizer's model object,
    so callers don't need to know the byte<->unicode conversion themselves."""

    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer

    def num_parameters(self):
        return self.tokenizer.get_vocab_size()

    def encode_spans(self, raw):
        """str/bytes -> list[bytes] spans -- what common.eval.cross_tokenizer/common.vocab
        expect from every tokenizer's induce_spans (see segment.py)."""
        encoding = self.tokenizer.encode(_to_str(raw), add_special_tokens=False)
        return [_token_to_bytes(tok) for tok in encoding.tokens]
