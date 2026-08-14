"""Unified encode/decode interface over any of the seven systems/ tokenizers,
for use by the pretraining pipeline (data_prep.py builds token shards with
it, train.py's generation/eval helpers decode with it).

Every system's own induce_spans already turns raw bytes into a byte-level
segmentation (list[bytes]); this wraps that into STABLE INTEGER ids suitable
for an embedding table. Two genuinely different families exist here, and this
module treats them differently on purpose rather than papering over it:

  - bpe/superbpe ("native" family): their own checkpoint already IS a
    complete int-id vocabulary (every byte 0-255 is always a valid symbol by
    construction -- see systems.superbpe.model/systems.bpe.model), so this
    just calls straight through to encode_ids/tokenizer.encode.

  - fairtok/magnet/flexitokens/manta/fanta ("span" family): these only ever
    produce a byte-SPAN segmentation. Their "vocabulary" is whatever got
    harvested into a vocab.json after a (comparatively tiny) tokenizer-fitting
    run -- nowhere near covering the vastly larger, more diverse text a real
    pretraining corpus contains. This module builds its OWN guaranteed-
    complete id space for these: ids 0-255 are ALWAYS reserved for the 256
    raw byte values (regardless of whether vocab.json happens to list them),
    and ids 256+ are the multi-byte spans vocab.json harvested. Any span
    induce_spans produces that ISN'T in that harvested set falls back to its
    own constituent bytes -- the same "fall back to raw bytes when nothing
    bigger matches" guarantee any byte-level BPE tokenizer's base alphabet
    already provides, just constructed by hand here since these five systems
    don't carry a complete vocabulary the way bpe/superbpe do.

One further wrinkle, not hidden: MAGNET's induce_spans requires a `script`
argument (its boundary predictor is gated per script) -- see _magnet_script
below for how a `lang` hint gets resolved to one, and what happens when it
can't be (falls back to whichever script the checkpoint actually has, once,
with a printed warning -- not a crash, since a real pretraining corpus will
routinely contain languages/scripts a given checkpoint was never trained on).
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
    warning (once per missing script, not per call -- a real pretraining
    corpus makes many millions of encode() calls) when it has to fall back.
    See module docstring for why this exists only for MAGNET."""

    def __init__(self, model):
        self.available = sorted(model.boundary_predictors.keys())
        self.default = "Latn" if "Latn" in self.available else self.available[0]
        self._warned = set()

    def resolve(self, lang):
        if lang is None:
            return self.default
        # Accept either a bare script ("Latn") or a lang_Script stem
        # ("eng_Latn") -- same short-code/full-stem duality every real data
        # source in this project already produces (see common.data.oldi_data).
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
    """Resolves ONE per-system induce-spans callable at adapter-construction
    time -- NOT re-dispatched (with a fresh `from systems.X.segment import
    ...`) on every single encode() call, which matters at real pretraining-
    corpus scale: encode() can run many millions of times, and a repeated
    per-call import (even though cheap -- Python caches the module, this
    was just an avoidable dict lookup + attribute rebind every time) has no
    reason to happen more than once per loaded checkpoint.

    Returns a function (raw_bytes, lang) -> list[bytes] spans, with every
    system's own real signature difference already normalized away
    (fairtok needs a pre-built tensor and no device kwarg; magnet needs a
    resolved script, via one _MagnetScriptResolver built once here rather
    than per call; flexitokens/manta/fanta take just (model, raw, device))."""
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
    (raws, langs) -> list[list[bytes]] (spans per document) that processes
    the WHOLE list in as few underlying model calls as possible, or None if
    this system has no true batched implementation yet (TokenizerAdapter.
    encode_batch then falls back to a plain per-item encode() loop --
    correct, just no throughput gain).

    Only manta/fanta get a real batched path here today: both reuse
    systems.manta.segment.induce_spans_batch, which pads the whole list to
    one common length and calls the model ONCE -- confirmed (via
    pretraining.data_prep hitting exactly this bottleneck on a real cluster
    run) to give a substantial throughput improvement over one call per
    document.

    magnet/flexitokens/fairtok deliberately do NOT get a batched path here,
    stated plainly rather than silently pretending they're covered: MAGNET's
    forward pass takes one `script` per call, and a real multi-document
    batch can span several different scripts (would need grouping documents
    by resolved script first -- not implemented); flexitokens' own
    induce_boundaries always builds a batch of exactly one sequence
    internally, with no multi-document entry point exposed; fairtok's
    segment_bytes loops one byte at a time in Python regardless of batch
    size, so batching in front of it wouldn't address its actual
    bottleneck. All three still work correctly via encode_batch's per-item
    fallback -- just without the speedup."""
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
    code or lang_Script stem) -- ignored by every system except MAGNET,
    where it selects which trained script's boundary predictor to use (see
    _MagnetScriptResolver). Every id returned is guaranteed valid (see
    module docstring for the fallback guarantee).

    decode(ids) -> bytes. Exact inverse of encode's id assignment (not of
    encode() itself -- a span that fell back to individual bytes decodes
    back to those same bytes either way, so decode(encode(x)) == bytes(x)
    always holds regardless of which path a given span took).

    vocab_size: total id space INCLUDING the reserved end-of-document id
    (see eos_id) -- size an embedding table to this.
    """

    def __init__(self, system, model, id_to_bytes, span_to_id, device):
        self.system = system
        self.model = model
        self.device = device
        self._id_to_bytes = id_to_bytes  # list[bytes], index == id, length == vocab_size (excl. eos)
        self._span_to_id = span_to_id  # dict[bytes, int] or None (native family doesn't need it)
        self.eos_id = len(id_to_bytes)
        self.vocab_size = len(id_to_bytes) + 1
        # Resolved once here, not per encode() call -- see _build_induce_fn's
        # own docstring. None for the native family (bpe/superbpe), which
        # never dispatches through this at all (see encode() below).
        self._induce_fn = (
            _build_induce_fn(system, model, device) if system in _SPAN_SYSTEMS else None
        )
        # None for the native family AND for the span-family systems with no
        # true batched implementation yet (magnet/flexitokens/fairtok) --
        # see _build_induce_batch_fn's own docstring. encode_batch() checks
        # for None itself and falls back to a per-item encode() loop.
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
        # string directly, then invert the byte<->unicode mapping -- same
        # trick bpe.model.BPEModel.encode_spans already uses.
        tok = model.tokenizer
        return [
            _token_string_to_bytes(tok.id_to_token(i)) for i in range(tok.get_vocab_size())
        ]

    @staticmethod
    def _span_id_space(vocab_json_path):
        with open(vocab_json_path, "r", encoding="utf-8") as f:
            raw_vocab = json.load(f)  # {token_string: rank_id}, rank 0 = most frequent
        # Multi-byte spans only (see module docstring: single bytes are
        # already covered by the reserved 0-255 range, adding them again
        # under a second id would waste id space and make decode ambiguous
        # about which id a given byte "really" has), kept in their original
        # frequency-rank order so common spans still get lower (256+) ids.
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
        """Batched counterpart to encode() -- returns the SAME result a
        per-item `[self.encode(t, lang=l) for t, l in zip(texts, langs)]`
        loop would (this is a throughput optimization, not a behavior
        change -- verified directly against the per-item path), using ONE
        underlying model call for the whole list where the system supports
        it (self._induce_batch_fn is not None -- manta/fanta today, see
        _build_induce_batch_fn's own docstring for why not the others yet).
        Falls back to the plain per-item loop for bpe/superbpe (already
        fast, no length-driven memory cost to batch around) and for any
        span-family system without a true batched implementation."""
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
