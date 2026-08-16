"""Loads an arbitrary HuggingFace repo's tokenizer (TOKENIZER ONLY) and
reconstructs exact BYTE spans from it, so it plugs into
common.eval.cross_tokenizer.evaluate_on_groups like every other
systems/*/segment.py's induce_spans.

SPAN RECONSTRUCTION is the real substance of this module: a frontier
tokenizer's `tokens`/`ids` aren't byte spans directly, and a naive approach
silently produces WRONG spans. Three schemes, detected empirically (never
assumed from model family), each with a real gotcha:

  1. Byte-level BPE (GPT-2/RoBERTa/tiktoken-style, e.g. gpt2, DeepSeek-V4-
     Pro): token characters encode raw bytes 1:1 via the GPT-2 byte<->unicode
     map (common.vocab.BYTE_TO_UNICODE, also used by systems/bpe/model.py).
     HF's `offset_mapping` is NOT usable here: a multi-byte UTF-8 char (e.g.
     "ü") can split across two tokens, and character-granularity offsets
     can't represent a boundary inside one character -- both half-tokens
     would map to the same range.

  2. Character-offset-based (SentencePiece, e.g. Llama-2, deberta-v3-base):
     `offset_mapping` IS exact for ordinary tokens (always whole
     characters), EXCEPT SentencePiece's `<0xXX>` byte-fallback tokens
     (used for out-of-vocab chars), which hit the same sub-character
     problem as scheme 1 (a 4-byte emoji -> four `<0xXX>` tokens sharing one
     offset) -- handled by reading the byte value out of the token's own hex
     digits instead of trusting the offset for those tokens.

     Same path also covers WordPiece (BERT/DistilBERT/ELECTRA), which has
     its own wrinkle: it discards whitespace entirely (word-boundary signal
     only, never encoded), so naive per-token offset slicing drops every
     space ("The quick" -> "Thequick"). Fix: _spans_via_offsets tracks a
     running `covered_until` char position and always starts the next span
     exactly where the last one left off (never at the token's own `start`)
     -- absorbs WordPiece's gaps into the following token's span (same
     "leading space belongs to next word" convention as BPE's "Ġ" prefix),
     and is a no-op when start == covered_until already.

     _detect_span_method always verifies its choice with a real round-trip
     (canary string) rather than trusting it -- correctly REJECTS
     xlm-roberta-base, whose SentencePiece tokenizer emits standalone
     metaspace-only tokens whose offset OVERLAPS the next token's start
     (not just a gap) -- a genuinely different quirk this module doesn't
     try to guess around.

  3. tiktoken-native (not loaded via transformers/HF Hub at all):
     `Encoding.decode_single_token_bytes(id)` returns exact raw bytes
     directly, no offset reconstruction needed. Selected via a
     "tiktoken:{encoding_name}" pseudo-repo-id (e.g. "tiktoken:cl100k_base")
     since these aren't HF Hub repos and a bare name could collide with one.
"""

import re

import tiktoken
from transformers import AutoTokenizer

from common.vocab import BYTE_TO_UNICODE

_UNICODE_TO_BYTE = {v: k for k, v in BYTE_TO_UNICODE.items()}
_BYTE_FALLBACK_RE = re.compile(r"^<0x([0-9A-Fa-f]{2})>$")
_TIKTOKEN_PREFIX = "tiktoken:"

# Mixes ASCII, a multi-byte accented Latin char (ü), CJK, and a 4-byte
# emoji -- exercises the sub-character-split edge case in both schemes
# above; a plain-ASCII canary would not have caught either bug.
_CANARY_TEXT = "The quick brown fox jumps über den Zaun. 你好世界 🎉"


def _token_to_bytes_gpt2(tok_str):
    # Literal ' ' always means a raw space byte -- the byte-to-unicode
    # scheme escapes byte 0x20 to 'Ġ', never to itself, so no collision.
    # Needed for e.g. ModernBERT-base, whose vocab has literal multi-space
    # tokens (for code/indentation) that bypass the escaping scheme.
    return bytes(ord(ch) if ch == " " else _UNICODE_TO_BYTE[ch] for ch in tok_str)


def _spans_via_byte_level(tokenizer, text):
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    tokens = tokenizer.convert_ids_to_tokens(ids)
    return [_token_to_bytes_gpt2(t) for t in tokens]


def _spans_via_tiktoken(encoding, text):
    # disallowed_special=(): tiktoken's default raises if text coincidentally
    # contains a substring that looks like a special token (e.g. a literal
    # "<|endoftext|>" in scraped text); we want plain byte-span
    # reconstruction, not a hard stop on a rare coincidence.
    ids = encoding.encode(text, disallowed_special=())
    return [encoding.decode_single_token_bytes(i) for i in ids]


def _load_tiktoken(encoding_name):
    valid = tiktoken.list_encoding_names()
    if encoding_name not in valid:
        raise ValueError(f"{encoding_name!r} is not a tiktoken encoding -- choose from {valid}")
    encoding = tiktoken.get_encoding(encoding_name)
    spans = _spans_via_tiktoken(encoding, _CANARY_TEXT)
    if b"".join(spans) != _CANARY_TEXT.encode("utf-8"):
        # Not expected to fire (tiktoken's byte-level BPE guarantees this),
        # but verified rather than assumed, same discipline as elsewhere here.
        raise ValueError(f"tiktoken encoding {encoding_name!r} failed the canary round-trip -- see module docstring")
    return encoding


def _spans_via_offsets(tokenizer, text):
    """`covered_until` is a running character high-water mark: each span
    starts exactly where the last one left off, never at the token's own
    `start` (see module docstring re: WordPiece gaps / xlm-roberta-base
    overlaps). One rule handles both -- a gap gets absorbed into the
    following span, an overlap gets clamped away -- and is a no-op when
    start == covered_until already (plain SentencePiece/byte-level case)."""
    enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"])
    raw = text.encode("utf-8")
    spans = []
    covered_until = 0
    for tok, (_start, end) in zip(tokens, enc["offset_mapping"]):
        m = _BYTE_FALLBACK_RE.match(tok)
        if m:
            spans.append(bytes([int(m.group(1), 16)]))
            covered_until = max(covered_until, end)
            continue
        span_end = max(end, covered_until)
        byte_start = len(text[:covered_until].encode("utf-8"))
        byte_end = len(text[:span_end].encode("utf-8"))
        spans.append(raw[byte_start:byte_end])
        covered_until = span_end
    if spans and covered_until < len(text):
        # Trailing gap after the last token (rare) -- attach to the last
        # span rather than drop it.
        spans[-1] = spans[-1] + raw[len(text[:covered_until].encode("utf-8")) :]
    return spans


def _detect_span_method(tokenizer):
    """Returns "byte_level" or "offset_mapping", chosen by actually trying
    each and checking a round-trip against the canary. Raises ValueError
    (not silently-wrong spans) if neither scheme reconstructs it exactly.

    Byte-level is tried FIRST regardless of tokenizer.is_fast: a "slow"
    (pure-Python) tokenizer (e.g. Kimi-K3) can still be genuine byte-level
    BPE, which only needs convert_ids_to_tokens. is_fast is only checked
    before the offset_mapping fallback, since return_offsets_mapping
    genuinely needs a fast tokenizer -- checked there, not up front, so a
    slow-but-byte-level tokenizer isn't rejected before it gets a chance."""
    try:
        spans = _spans_via_byte_level(tokenizer, _CANARY_TEXT)
        if b"".join(spans) == _CANARY_TEXT.encode("utf-8"):
            return "byte_level"
    except KeyError:
        pass

    if not tokenizer.is_fast:
        raise ValueError(
            f"{tokenizer.__class__.__name__} is not a 'fast' (Rust-backed) tokenizer and its "
            "byte-level-BPE reconstruction didn't round-trip -- the character-offset fallback "
            "needs offset_mapping, which only fast tokenizers provide (this repo likely has no "
            "tokenizer.json, and isn't byte-level-BPE either)"
        )

    spans = _spans_via_offsets(tokenizer, _CANARY_TEXT)
    if b"".join(spans) != _CANARY_TEXT.encode("utf-8"):
        raise ValueError(
            f"{tokenizer.__class__.__name__}: neither byte-level-BPE nor offset-mapping "
            "span reconstruction round-trips the canary string exactly -- this specific "
            "tokenizer's scheme isn't one of the two handled by systems/hf_frontier/model.py "
            "yet (see its own module docstring); do not use its results without extending "
            "this module first"
        )
    return "offset_mapping"


class HFFrontierTokenizer:
    """Construct via .load(repo_id), not directly. .induce_spans(raw) ->
    list[bytes], same shape as every systems/*/segment.py's induce_spans --
    plugs directly into common.eval.cross_tokenizer.evaluate_on_groups."""

    def __init__(self, tokenizer, span_method):
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.n_vocab if span_method == "tiktoken" else tokenizer.vocab_size
        self.span_method = span_method

    @classmethod
    def load(cls, repo_id, trust_remote_code=False, hf_token=None):
        """Loads ONLY the tokenizer from `repo_id` (AutoTokenizer.from_pretrained
        fetches tokenizer.json/config files, never model weights).

        A "tiktoken:{encoding_name}" repo_id (e.g. "tiktoken:cl100k_base")
        loads a tiktoken built-in encoding instead (see _load_tiktoken /
        scheme 3 above); the other args are unused for that path.

        trust_remote_code defaults to False: some repos (e.g. Kimi-K3) ship
        a custom tokenizer class that needs this to load, but it means
        executing that repo's own Python code -- opt in explicitly.

        hf_token: only needed for gated repos (e.g. Llama-3.1-8B-Instruct,
        which also needs license acceptance on huggingface.co). Falls back
        to HF_TOKEN env var / prior `huggingface-cli login`."""
        if repo_id.startswith(_TIKTOKEN_PREFIX):
            encoding = _load_tiktoken(repo_id[len(_TIKTOKEN_PREFIX):])
            return cls(encoding, "tiktoken")
        tokenizer = AutoTokenizer.from_pretrained(
            repo_id, trust_remote_code=trust_remote_code, token=hf_token
        )
        span_method = _detect_span_method(tokenizer)
        return cls(tokenizer, span_method)

    def induce_spans(self, raw):
        # errors="replace": same lossiness systems/bpe/model.py's _to_str
        # accepts for invalid UTF-8 -- never hit by real project text.
        text = raw.decode("utf-8", errors="replace") if isinstance(raw, (bytes, bytearray)) else raw
        if self.span_method == "byte_level":
            return _spans_via_byte_level(self.tokenizer, text)
        if self.span_method == "tiktoken":
            return _spans_via_tiktoken(self.tokenizer, text)
        return _spans_via_offsets(self.tokenizer, text)
