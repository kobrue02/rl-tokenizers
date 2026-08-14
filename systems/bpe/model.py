"""Standard byte-level BPE baseline -- unlike every other tokenizer in this
repo (including superbpe/, a from-scratch reimplementation), this one wraps
HuggingFace's `tokenizers` library (github.com/huggingface/tokenizers)
directly rather than reimplementing the algorithm: plain BPE has no novel
mechanism specific to this project to reimplement, and `tokenizers` IS the
reference, production-grade implementation of exactly this algorithm --
reusing it here is a deliberate choice, not a shortcut around something that
needed reimplementing (contrast superbpe/, whose two-stage "lift the
whitespace wall partway through" mechanism has no off-the-shelf
implementation to reuse).

The pre-tokenizer is ByteLevel (GPT-2/RoBERTa's own convention: every raw
byte maps to its own printable unicode character, then ordinary BPE merges
run over those characters) -- the same convention this project's own
common.vocab.BYTE_TO_UNICODE already implements for vocab.json output,
confirmed to be the EXACT same mapping (verified: encoding round-trips
losslessly back to the original bytes via that mapping's inverse -- see
_token_to_bytes below). This keeps BPE's own tokens byte-level, matching
every other tokenizer here, rather than defaulting to `tokenizers`' own
Unicode-string-level behavior.
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
    # codepoints (Python's usual round-trip-safe way to represent invalid
    # bytes) cannot be re-encoded -- confirmed this raises UnicodeEncodeError
    # deep inside the Rust extension, not something catchable per-call.
    # errors="replace" avoids the crash at the cost of losing fidelity for
    # any genuinely invalid byte run -- a real, accepted limitation of
    # leaning on tokenizers' str-based API directly (see module docstring),
    # not a bug to work around. Every REAL text source in this project
    # (oldi_seed/flores_plus/smol/bouquet) is genuinely valid UTF-8 already,
    # so this only ever matters for common.data.synthetic's synthetic placeholder
    # corpus, which deliberately generates possibly-invalid-UTF-8 bytes --
    # this package's own smoke test therefore uses a small real-text corpus
    # instead of that generator (see train.py's run_smoke_test).
    if isinstance(text, str):
        return text
    return bytes(text).decode("utf-8", errors="replace")


def _token_to_bytes(token_str):
    """Inverts the ByteLevel byte<->unicode mapping (see module docstring) to
    recover a token's actual underlying bytes -- what every other tokenizer's
    induce_spans returns directly, but tokenizers' own Encoding.tokens gives
    back as the intermediate unicode-character strings instead."""
    return bytes(_UNICODE_TO_BYTE[ch] for ch in token_str)


def fit_bpe(sentences, vocab_size):
    """Trains a standard byte-level BPE tokenizer over `sentences` (str/bytes,
    any language mixed together -- BPE has no notion of language). Returns a
    BPEModel. min_frequency=1 (HF's own default is 0, i.e. "even a
    frequency-1 pair may be merged") is left at the library default
    deliberately, not tightened -- this project's smoke-test-scale corpora are
    small enough that raising it would starve training of learnable merges
    entirely, and every other tokenizer here also imposes no artificial
    frequency floor of its own."""
    tokenizer = Tokenizer(BPE(unk_token=None))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    trainer = BpeTrainer(vocab_size=vocab_size, min_frequency=0, show_progress=False)
    tokenizer.train_from_iterator((_to_str(s) for s in sentences), trainer=trainer)
    return BPEModel(tokenizer)


class BPEModel:
    """Thin wrapper around a trained `tokenizers.Tokenizer` -- exists so this
    package's segment.py/train.py can call encode_spans/num_parameters the
    same way every other tokenizer's model object supports, without every
    caller needing to know the byte<->unicode conversion (see module
    docstring) themselves."""

    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer

    def num_parameters(self):
        return self.tokenizer.get_vocab_size()

    def encode_spans(self, raw):
        """str/bytes -> list[bytes] spans -- what common.eval.cross_tokenizer/common.vocab
        expect from every tokenizer's induce_spans (see segment.py)."""
        encoding = self.tokenizer.encode(_to_str(raw), add_special_tokens=False)
        return [_token_to_bytes(tok) for tok in encoding.tokens]
