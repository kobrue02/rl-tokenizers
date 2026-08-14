"""Loads an arbitrary HuggingFace repo's tokenizer (TOKENIZER ONLY -- see
HFFrontierTokenizer.load's own docstring) and reconstructs exact BYTE spans
from it, so it can plug into common.eval.cross_tokenizer.evaluate_on_groups exactly
like every other systems/*/segment.py's own induce_spans.

SPAN RECONSTRUCTION is the real substance of this module. A frontier
tokenizer's own `tokens`/`ids` aren't byte spans directly -- getting from
"which token" to "which raw bytes" turns out to depend on which of (at
least) two genuinely different schemes a given tokenizer uses, CONFIRMED
empirically here (not assumed from a model's architecture/family name),
because a naive approach silently produces WRONG spans for either scheme:

  1. Byte-level BPE (GPT-2/RoBERTa/tiktoken-style -- confirmed for gpt2 and
     deepseek-ai/DeepSeek-V4-Pro): every token's own characters directly
     encode raw bytes one-for-one via the GPT-2 byte<->unicode convention
     this project already uses elsewhere (common.vocab.BYTE_TO_UNICODE,
     also reused by systems/bpe/model.py). HF's own `offset_mapping`
     (character offsets) is NOT sufficient here and was confirmed to give
     wrong results directly: this scheme can split a single multi-byte
     UTF-8 character across TWO tokens (e.g. "ü" = 2 UTF-8 bytes can become
     two separate byte-level tokens), and character-granularity offsets
     cannot represent a boundary that falls INSIDE one character -- both
     half-tokens end up mapped to the exact same character range.

  2. Character-offset-based (SentencePiece -- confirmed for
     NousResearch/Llama-2-7b-hf, an ungated mirror of Llama-2's own
     tokenizer, and microsoft/deberta-v3-base): HF's `offset_mapping` IS
     exact here, since ordinary SentencePiece tokens are always whole
     characters -- EXCEPT its own `<0xXX>` byte-fallback tokens (emitted
     for characters outside its normal vocabulary), which hit the identical
     sub-character problem as scheme 1: a 4-byte emoji can become four
     separate `<0xXX>` tokens, all sharing one character offset. Handled by
     reading the byte value directly out of the token's own hex digits
     instead of trusting the offset for those specific tokens.

     This same offset-based path ALSO covers WordPiece (BERT/DistilBERT/
     ELECTRA-style) -- confirmed for bert-base-cased/bert-base-multilingual-
     cased -- but WordPiece has its own real wrinkle offsets alone don't
     solve: it discards whitespace entirely during tokenization (used only
     as a word-boundary signal, never encoded in any token), so naive
     per-token offset slicing silently DROPS every space -- confirmed live:
     "The quick" round-tripped to "Thequick" without this. _spans_via_offsets
     tracks a running "covered_until" character position and always starts
     the next span exactly where the last one left off (never at the
     token's own reported `start`) -- this absorbs any gap (WordPiece's
     dropped whitespace) into the FOLLOWING token's span, the same "leading
     space belongs to the next word" convention byte-level BPE's own "Ġ"
     prefix already encodes, and is a no-op (start == covered_until anyway)
     for tokenizers with no such gaps, so it doesn't change anything for the
     plain-SentencePiece case already confirmed working.

Given a genuinely new/unverified repo could use some THIRD scheme this
module doesn't yet handle, _detect_span_method always verifies its choice
with a real round-trip (encode a canary string, reconstruct bytes, compare)
before trusting it, and raises a clear error rather than silently returning
wrong spans if neither known scheme round-trips correctly -- confirmed live
this correctly REJECTS xlm-roberta-base, whose SentencePiece tokenizer
emits a standalone metaspace-only token immediately before some words with
an offset that overlaps the following token's own reported start (not just
a gap) -- a genuinely different, not-yet-understood quirk this module
deliberately doesn't guess its way around.

  3. tiktoken-native (OpenAI's own tiktoken package, NOT loaded via
     transformers.AutoTokenizer/HF Hub at all -- confirmed live tiktoken's
     own `Encoding.decode_single_token_bytes(id)` returns each token's exact
     raw bytes directly, no offset reconstruction needed, and every one of
     the 7 encodings tiktoken 0.13.0 ships (gpt2, r50k_base, p50k_base,
     p50k_edit, cl100k_base, o200k_base, o200k_harmony) round-trips the
     canary exactly -- this is what tiktoken's own byte-level BPE design
     guarantees by construction, verified rather than assumed anyway (see
     _load_tiktoken's own round-trip check). Selected via a
     "tiktoken:{encoding_name}" pseudo-repo-id (e.g. "tiktoken:cl100k_base")
     instead of a real HF repo_id, since these aren't HF Hub repos at all --
     a bare name like "cl100k_base" could otherwise collide with (or be
     confused for) an actual HF repo id.
"""

import re

import tiktoken
from transformers import AutoTokenizer

from common.vocab import BYTE_TO_UNICODE

_UNICODE_TO_BYTE = {v: k for k, v in BYTE_TO_UNICODE.items()}
_BYTE_FALLBACK_RE = re.compile(r"^<0x([0-9A-Fa-f]{2})>$")
_TIKTOKEN_PREFIX = "tiktoken:"

# Deliberately mixes: plain ASCII words, a multi-byte-but-single-codepoint
# accented Latin character (ü), fully non-Latin multi-byte script (CJK),
# and a 4-byte emoji requiring surrogate-pair-free UTF-8 handling -- this
# specific mix is what actually exercises the sub-character-split edge case
# in both known schemes (see module docstring); a plain-ASCII canary would
# NOT have caught either bug during development.
_CANARY_TEXT = "The quick brown fox jumps über den Zaun. 你好世界 🎉"


def _token_to_bytes_gpt2(tok_str):
    return bytes(_UNICODE_TO_BYTE[ch] for ch in tok_str)


def _spans_via_byte_level(tokenizer, text):
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    tokens = tokenizer.convert_ids_to_tokens(ids)
    return [_token_to_bytes_gpt2(t) for t in tokens]


def _spans_via_tiktoken(encoding, text):
    # disallowed_special=(): real held-out text can coincidentally contain a
    # substring that LOOKS like a special token (e.g. a literal
    # "<|endoftext|>" appearing in some scraped corpus) -- tiktoken's own
    # default raises on that rather than tokenizing it as plain text; this
    # project wants byte-span reconstruction for arbitrary real text, not a
    # hard stop on a rare coincidental match, so nothing is treated as
    # special here.
    ids = encoding.encode(text, disallowed_special=())
    return [encoding.decode_single_token_bytes(i) for i in ids]


def _load_tiktoken(encoding_name):
    valid = tiktoken.list_encoding_names()
    if encoding_name not in valid:
        raise ValueError(f"{encoding_name!r} is not a tiktoken encoding -- choose from {valid}")
    encoding = tiktoken.get_encoding(encoding_name)
    spans = _spans_via_tiktoken(encoding, _CANARY_TEXT)
    if b"".join(spans) != _CANARY_TEXT.encode("utf-8"):
        # Not expected to ever fire (tiktoken's own byte-level BPE design
        # guarantees this), but verified rather than assumed anyway, same
        # discipline as every other scheme in this module.
        raise ValueError(f"tiktoken encoding {encoding_name!r} failed the canary round-trip -- see module docstring")
    return encoding


def _spans_via_offsets(tokenizer, text):
    """`covered_until` is a running character high-water mark, advanced
    monotonically -- each span always starts exactly where the last one
    left off, NEVER at the token's own reported `start` (see module
    docstring's own WordPiece/xlm-roberta-base discussion for why trusting
    `start` directly breaks on both a GAP -- WordPiece's dropped whitespace,
    start > covered_until -- and an OVERLAP -- xlm-roberta-base's isolated
    metaspace-only tokens, start < covered_until). This one rule handles
    both: a gap gets absorbed into the following span, an overlap gets
    clamped away from the following span (since covered_until already
    claimed those characters), and it's a complete no-op for a tokenizer
    with neither (start == covered_until already), so nothing changes for
    the plain-SentencePiece/byte-level cases already confirmed working."""
    enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"])
    raw = text.encode("utf-8")
    spans = []
    covered_until = 0
    for tok, (start, end) in zip(tokens, enc["offset_mapping"]):
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
        # Trailing gap after the last token (rare, but possible) -- attach
        # to the last span rather than drop it, same "no bytes belong to no
        # token" invariant as the leading/mid-sequence case above.
        spans[-1] = spans[-1] + raw[len(text[:covered_until].encode("utf-8")) :]
    return spans


def _detect_span_method(tokenizer):
    """Returns "byte_level" or "offset_mapping", chosen by ACTUALLY trying
    each and checking a real round-trip -- see module docstring. Raises
    ValueError (a clear, actionable failure, not silently wrong spans) if
    neither known scheme reconstructs the canary's bytes exactly.

    Byte-level is tried FIRST regardless of tokenizer.is_fast -- confirmed
    directly (moonshotai/Kimi-K3's own tokenizer): a "slow" (pure-Python,
    not Rust-backed) tokenizer can still be genuine byte-level BPE with a
    working convert_ids_to_tokens, which is all this path needs.
    tokenizer.is_fast is only required for the offset_mapping fallback,
    since HF's return_offsets_mapping genuinely needs a fast tokenizer --
    checked there, not up front, so a slow-but-byte-level tokenizer isn't
    rejected before it even gets a chance."""
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
    list[bytes], the same shape every other systems/*/segment.py's own
    induce_spans returns -- plugs directly into
    common.eval.cross_tokenizer.evaluate_on_groups."""

    def __init__(self, tokenizer, span_method):
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.n_vocab if span_method == "tiktoken" else tokenizer.vocab_size
        self.span_method = span_method

    @classmethod
    def load(cls, repo_id, trust_remote_code=False, hf_token=None):
        """Loads ONLY the tokenizer from `repo_id` -- confirmed directly
        (not assumed): a real load of deepseek-ai/DeepSeek-V4-Pro's
        tokenizer via AutoTokenizer.from_pretrained fetched ~6MB of
        tokenizer.json/config files, zero .safetensors/.bin model weight
        files, verified by inspecting the HF cache directory afterward.

        A "tiktoken:{encoding_name}" repo_id (e.g. "tiktoken:cl100k_base")
        loads one of tiktoken's own built-in encodings instead -- see
        _load_tiktoken and the module docstring's own scheme 3. Every other
        argument below is irrelevant to that path (tiktoken's encodings
        aren't HF Hub repos, need no token, execute no remote code) and is
        simply not used for it.

        trust_remote_code defaults to False and must be opted into
        explicitly -- some repos (confirmed: moonshotai/Kimi-K3 ships its
        own tokenization_kimi.py) define a custom tokenizer class not built
        into transformers and need this to load at all, but it means
        executing that repo's own Python code on your machine -- a real
        security consideration this project won't silently default around
        just to make one more repo load without an extra flag.

        hf_token: an explicit HF access token, only required for GATED
        repos (confirmed: meta-llama/Llama-3.1-8B-Instruct is one -- needs
        BOTH license acceptance on huggingface.co AND a token with that
        access actually granted). Falls back to the HF_TOKEN environment
        variable / a prior `huggingface-cli login` if not given, the same
        convention every other HF_TOKEN usage in this project's own
        jobs/*.sh scripts already follows."""
        if repo_id.startswith(_TIKTOKEN_PREFIX):
            encoding = _load_tiktoken(repo_id[len(_TIKTOKEN_PREFIX):])
            return cls(encoding, "tiktoken")
        tokenizer = AutoTokenizer.from_pretrained(
            repo_id, trust_remote_code=trust_remote_code, token=hf_token
        )
        span_method = _detect_span_method(tokenizer)
        return cls(tokenizer, span_method)

    def induce_spans(self, raw):
        # errors="replace": same accepted, documented lossiness
        # systems/bpe/model.py's own _to_str takes for invalid UTF-8 --
        # never triggered by any real text source in this project, only a
        # deliberately-adversarial byte string.
        text = raw.decode("utf-8", errors="replace") if isinstance(raw, (bytes, bytearray)) else raw
        if self.span_method == "byte_level":
            return _spans_via_byte_level(self.tokenizer, text)
        if self.span_method == "tiktoken":
            return _spans_via_tiktoken(self.tokenizer, text)
        return _spans_via_offsets(self.tokenizer, text)
