"""Unified encode/decode interface over any of the seven systems/ tokenizers,
for use by the pretraining pipeline (data_prep.py builds shards with it,
train.py's generation/eval helpers decode with it).

Every system's induce_spans turns raw bytes into a byte-level segmentation
(list[bytes]); this wraps that into stable integer ids for an embedding
table. Two families, handled differently:

  - bpe/superbpe ("native"): their checkpoint already is a complete int-id
    vocabulary (every byte 0-255 is a valid symbol by construction), so
    this calls straight through to encode_ids/tokenizer.encode.

  - fairtok/magnet/flexitokens/manta/fanta ("span"): these only produce a
    byte-span segmentation, and their vocab.json (harvested from a small
    tokenizer-fitting run) doesn't cover the far larger/more diverse text a
    pretraining corpus contains. This module builds its own complete id
    space: ids 0-255 are always reserved for raw byte values (regardless
    of vocab.json), ids 256+ are the multi-byte spans vocab.json
    harvested. Any span not in that harvested set falls back to its
    constituent bytes -- the same fallback guarantee byte-level BPE's base
    alphabet gives for free.

One wrinkle: MAGNET's induce_spans requires a `script` argument (its
boundary predictor is gated per script) -- see _MagnetScriptResolver for
how a `lang` hint resolves to one, falling back (with a one-time warning,
not a crash) to whichever script the checkpoint has when it can't.
"""

import json

from common.vocab import BYTE_TO_UNICODE

_UNICODE_TO_BYTE = {v: k for k, v in BYTE_TO_UNICODE.items()}

_NATIVE_SYSTEMS = {"bpe", "superbpe"}
_SPAN_SYSTEMS = {"fairtok", "magnet", "flexitokens", "manta", "fanta"}
ALL_SYSTEMS = sorted(_NATIVE_SYSTEMS | _SPAN_SYSTEMS)


def _token_string_to_bytes(token_str):
    return bytes(_UNICODE_TO_BYTE[ch] for ch in token_str)


def _to_bytes(text):
    return text.encode("utf-8") if isinstance(text, str) else bytes(text)


def _load_model(system, checkpoint_path, device):
    if system == "bpe":
        from systems.bpe.inference import load_checkpoint

        return load_checkpoint(checkpoint_path, device=device)
    if system == "superbpe":
        from systems.superbpe.inference import load_checkpoint

        return load_checkpoint(checkpoint_path, device=device)
    if system == "manta":
        from systems.manta.inference import load_checkpoint

        return load_checkpoint(checkpoint_path, device=device)
    if system == "fanta":
        from systems.fanta.inference import load_checkpoint

        return load_checkpoint(checkpoint_path, device=device)
    if system == "flexitokens":
        from systems.flexitokens.inference import load_checkpoint

        return load_checkpoint(checkpoint_path, device=device)
    if system == "magnet":
        from systems.magnet.inference import load_checkpoint

        return load_checkpoint(checkpoint_path, device=device)
    if system == "fairtok":
        from systems.fairtok.inference import load_checkpoint

        policy = load_checkpoint(checkpoint_path)  # always loads to CPU
        return policy.to(device)
    raise ValueError(f"unknown system {system!r} -- expected one of {ALL_SYSTEMS}")


class _MagnetScriptResolver:
    """Resolves a `lang` hint to one of MAGNET's own trained script keys,
    warning once per missing script (not per call, since encode() may run
    millions of times) when it has to fall back."""

    def __init__(self, model):
        self.available = sorted(model.boundary_predictors.keys())
        self.default = "Latn" if "Latn" in self.available else self.available[0]
        self._warned = set()

    def resolve(self, lang):
        if lang is None:
            return self.default
        # Accept either a bare script ("Latn") or a lang_Script stem ("eng_Latn").
        script = lang.rsplit("_", 1)[-1] if "_" in lang else lang
        if script in self.available:
            return script
        if lang not in self._warned:
            print(
                f"[tokenizer_adapter] MAGNET checkpoint has no boundary predictor "
                f"for script {script!r} (from lang={lang!r}); falling back to "
                f"{self.default!r}. Available: {self.available}"
            )
            self._warned.add(lang)
        return self.default


def _build_induce_fn(system, model, device):
    """Resolves one per-system induce-spans callable at adapter-construction
    time rather than re-importing per encode() call (avoidable overhead at
    millions-of-calls corpus scale).

    Returns a function (raw_bytes, lang) -> list[bytes] spans, with each
    system's signature difference normalized away (fairtok needs a
    pre-built tensor and no device kwarg; magnet needs a resolved script
    via _MagnetScriptResolver; flexitokens/manta/fanta take just (model,
    raw, device))."""
    if system == "magnet":
        from systems.magnet.segment import induce_spans

        script_resolver = _MagnetScriptResolver(model)

        def induce(raw, lang):
            return induce_spans(model, raw, script_resolver.resolve(lang), device)

        return induce

    if system == "fairtok":
        from common.bytes_utils import bytes_to_tensor
        from systems.fairtok.policy import segment_bytes

        def induce(raw, lang):
            tensor = bytes_to_tensor(raw, device)
            return segment_bytes(model, tensor, deterministic=True, device=device)

        return induce

    if system == "flexitokens":
        from systems.flexitokens.segment import induce_spans
    elif system == "manta":
        from systems.manta.segment import induce_spans
    elif system == "fanta":
        from systems.fanta.segment import induce_spans
    else:
        raise ValueError(f"{system!r} is not a span-family system")

    def induce(raw, lang):
        return induce_spans(model, raw, device)

    return induce


def _build_induce_batch_fn(system, model, device):
    """Batched counterpart to _build_induce_fn -- returns a callable
    (raws, langs) -> list[list[bytes]] processing the whole list in as few
    model calls as possible, or None if this system has no batched
    implementation (encode_batch then falls back to a per-item loop --
    correct, just no throughput gain).

    Only manta/fanta get a real batched path: both reuse
    systems.manta.segment.induce_spans_batch, padding the whole list to one
    common length and calling the model once (confirmed to substantially
    improve throughput over a data_prep cluster run bottleneck).

    magnet/flexitokens/fairtok don't: MAGNET's forward pass takes one
    `script` per call and a batch can span several scripts (would need
    grouping by script first, not implemented); flexitokens' induce_
    boundaries always builds a batch of exactly one sequence internally;
    fairtok's segment_bytes loops one byte at a time regardless of batch
    size. All three still work correctly via the per-item fallback."""
    if system in ("manta", "fanta"):
        from systems.manta.segment import induce_spans_batch

        def induce_batch(raws, langs):
            return induce_spans_batch(model, raws, device)

        return induce_batch
    return None


class TokenizerAdapter:
    """Construct via TokenizerAdapter.load(system, checkpoint_path[,
    vocab_json_path]), not directly.

    encode(text, lang=None) -> list[int]. lang is an optional hint (short
    code or lang_Script stem), ignored by every system except MAGNET,
    where it selects the boundary predictor's script. Every id returned is
    guaranteed valid (see module docstring's fallback guarantee).

    decode(ids) -> bytes. Exact inverse of encode's id assignment, so
    decode(encode(x)) == bytes(x) always holds.

    vocab_size: total id space including the reserved end-of-document id
    (eos_id) -- size an embedding table to this.
    """

    def __init__(self, system, model, id_to_bytes, span_to_id, device):
        self.system = system
        self.model = model
        self.device = device
        self._id_to_bytes = id_to_bytes  # list[bytes], index == id, length == vocab_size (excl. eos)
        self._span_to_id = span_to_id  # dict[bytes, int] or None (native family doesn't need it)
        self.eos_id = len(id_to_bytes)
        self.vocab_size = len(id_to_bytes) + 1
        # Resolved once here, not per encode() call (see _build_induce_fn).
        # None for the native family (bpe/superbpe), which never dispatches
        # through this.
        self._induce_fn = (
            _build_induce_fn(system, model, device) if system in _SPAN_SYSTEMS else None
        )
        # None for native systems and for span-family systems with no true
        # batched implementation (see _build_induce_batch_fn); encode_batch
        # falls back to a per-item loop in that case.
        self._induce_batch_fn = (
            _build_induce_batch_fn(system, model, device) if system in _SPAN_SYSTEMS else None
        )

    @classmethod
    def load(cls, system, checkpoint_path, vocab_json_path=None, device="cpu"):
        if system not in ALL_SYSTEMS:
            raise ValueError(f"unknown system {system!r} -- expected one of {ALL_SYSTEMS}")
        model = _load_model(system, checkpoint_path, device)

        if system in _NATIVE_SYSTEMS:
            id_to_bytes = cls._native_id_to_bytes(system, model)
            return cls(system, model, id_to_bytes, span_to_id=None, device=device)

        if vocab_json_path is None:
            raise ValueError(
                f"{system!r} is a span-family system and needs its saved "
                "vocab.json (--vocab-out at training time) to build a "
                "complete id space -- see module docstring."
            )
        id_to_bytes, span_to_id = cls._span_id_space(vocab_json_path)
        return cls(system, model, id_to_bytes, span_to_id, device=device)

    @staticmethod
    def _native_id_to_bytes(system, model):
        if system == "superbpe":
            return [model.id_to_bytes[i] for i in range(len(model.id_to_bytes))]
        # bpe: ask the underlying tokenizers.Tokenizer for each id's token
        # string, then invert the byte<->unicode mapping.
        tok = model.tokenizer
        return [
            _token_string_to_bytes(tok.id_to_token(i)) for i in range(tok.get_vocab_size())
        ]

    @staticmethod
    def _span_id_space(vocab_json_path):
        with open(vocab_json_path, "r", encoding="utf-8") as f:
            raw_vocab = json.load(f)  # {token_string: rank_id}, rank 0 = most frequent
        # Multi-byte spans only (single bytes are already covered by the
        # reserved 0-255 range), kept in frequency-rank order so common
        # spans get lower (256+) ids.
        by_rank = sorted(raw_vocab.items(), key=lambda kv: kv[1])
        spans_by_rank = (_token_string_to_bytes(tok_str) for tok_str, _ in by_rank)
        multi_byte_spans = [span for span in spans_by_rank if len(span) > 1]
        id_to_bytes = [bytes([i]) for i in range(256)] + multi_byte_spans
        span_to_id = {span: 256 + i for i, span in enumerate(multi_byte_spans)}
        return id_to_bytes, span_to_id

    def _spans_to_ids(self, spans):
        ids = []
        for span in spans:
            tid = self._span_to_id.get(span)
            if tid is not None:
                ids.append(tid)
            else:
                ids.extend(span)  # fallback: each byte IS its own id, 0-255
        return ids

    def encode(self, text, lang=None):
        raw = _to_bytes(text)
        if self.system == "bpe":
            return self.model.tokenizer.encode(
                raw.decode("utf-8", errors="replace"), add_special_tokens=False
            ).ids
        if self.system == "superbpe":
            return self.model.encode_ids(raw)

        spans = self._induce_fn(raw, lang)
        return self._spans_to_ids(spans)

    def encode_batch(self, texts, langs=None):
        """Batched counterpart to encode() -- returns the same result a
        per-item `[self.encode(t, lang=l) for t, l in zip(texts, langs)]`
        loop would (throughput optimization only), using one model call
        for the whole list where supported (manta/fanta today; see
        _build_induce_batch_fn). Falls back to the per-item loop for
        bpe/superbpe and any span-family system without a batched path."""
        langs = list(langs) if langs is not None else [None] * len(texts)
        if self._induce_batch_fn is None:
            return [self.encode(t, lang=l) for t, l in zip(texts, langs)]
        raws = [_to_bytes(t) for t in texts]
        spans_list = self._induce_batch_fn(raws, langs)
        return [self._spans_to_ids(spans) for spans in spans_list]

    def decode(self, ids):
        return b"".join(
            self._id_to_bytes[i] if i < len(self._id_to_bytes) else b""
            for i in ids
            if i != self.eos_id
        )
