"""Parity-aware BPE, wrapping the OFFICIAL implementation directly (per
explicit user instruction to reuse it wherever possible) rather than
reimplementing the algorithm from scratch.

Paper: Foroutan, Meister, Paul, Niklaus, Ahmadi, Bosselut & Sennrich,
"Parity-Aware Byte-Pair Encoding: Improving Cross-lingual Fairness in
Tokenization" (ACL 2026, aclanthology.org/2026.acl-long.342). Official code:
github.com/swiss-ai/parity-aware-bpe (MIT). The actual fitting functions
(learn_bpe/learn_bpe_moving_window and their dependencies) are vendored
near-verbatim in .vendor.parity_aware_learn_bpe -- see that module's own
docstring for exactly what's kept/dropped/why. This file is the thin
adapter layer between their file-based, argparse-oriented API and this
project's own {lang: text} group data + Config/Trainer/evaluate.py scaffold
-- like bpe/model.py (this project's OTHER classical-BPE baseline), which
wraps HuggingFace's `tokenizers` library directly rather than reimplementing
BPE, not like superbpe/model.py's from-scratch reimplementation (a
DIFFERENT choice for a DIFFERENT paper's baseline).

WHY io.StringIO, NOT REAL TEMP FILES: the vendored get_vocabulary only
touches a passed-in file object's `.name` attribute inside its num_workers>1
branch (confirmed by reading the vendored source) -- this project always
calls with num_workers=1, so that branch, and thus the `.name` requirement,
never fires. io.StringIO supports every OTHER file-object operation these
functions need (iteration over lines, `.read()`/`.readline()`/`.seek()` for
the preload path, `.write()` for the output), so the whole fit runs
in-memory, never touching disk.

WHY tokenizers.models.BPE(vocab=..., merges=...) directly, not their own
HF_tokenizer.py's disk round-trip: this project's installed `tokenizers`
version (0.22.2) accepts an in-memory {token: id} dict and list[(str,str)]
merges directly -- confirmed via BPE.__doc__ -- so building vocab.json/
merges.txt files on disk and reloading them (what their own
create_huggingface_tokenizer/load_custom_tokenizer do) is unneeded, extra
I/O this project skips. .vendor.hf_tokenizer.build_vocab_from_merges (their
own vocab-construction logic, pure Python, no file I/O) is still reused
directly for the {token: id} dict itself.

BYTE<->UNICODE CONVENTION: their own vocabulary uses HuggingFace's
ByteLevel.alphabet() single-character-per-byte remapping (the GPT-2/RoBERTa
convention) -- identical to common.vocab.BYTE_TO_UNICODE, already verified
equivalent by systems.tokenization.bpe.model's own docstring for this
project's other classical-BPE baseline. ParityBPEModel below reuses that
same inverse mapping to recover real bytes from encoded tokens, exactly like
bpe.model.BPEModel does.

CHECKPOINT/RESUME (checkpoint_dir, optional): learn_bpe/
learn_bpe_moving_window themselves are single monolithic calls with no
pause/resume hook of any kind. Rather than modify the vendored functions to
add one, .checkpointed_fit reimplements just their OUTER LOOP, reusing their
existing sub-functions (preprocess_input_data, prune_stats, replace_pair,
update_pair_statistics, replace_pair_dict, select_language_index)
unmodified -- see that module's own docstring for exactly what's persisted
and why a resume is provably BYTE-IDENTICAL to an uninterrupted run (not
merely "comparably fair"). fit_parity_bpe below calls
vendor.learn_bpe/learn_bpe_moving_window DIRECTLY (maximum reuse of the
official implementation, zero risk of the two code paths drifting apart)
whenever checkpoint_dir is None -- the checkpointed path is opt-in, used
only when actually needed for a real long-running fit.
"""

import io
import os
from collections import defaultdict

import langcodes
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel, Sequence, Whitespace

from common.vocab import BYTE_TO_UNICODE

from . import checkpointed_fit
from .vendor import parity_aware_learn_bpe as vendor
from .vendor.hf_tokenizer import build_vocab_from_merges

_UNICODE_TO_BYTE = {v: k for k, v in BYTE_TO_UNICODE.items()}


def _normalize_lang_code(code, _cache={}):
    """Best-effort canonicalization to a base ISO 639-3 code, so a short
    code (e.g. "en", "ar-MA" -- common.data.corpora's own smol source) and
    this project's own lang_Script convention (e.g. "eng_Latn",
    "apc_Arab_nort3139" -- BOUQuET/oldi_seed/flores_dev) that refer to the
    SAME language are recognized as the same key for train/dev pairing,
    instead of being silently treated as unrelated languages and dropped.

    Confirmed live to matter: a real --data-source all run (which pools
    oldi_seed+flores_dev+smol) showed 195 of 345 training languages
    excluded from the fit entirely, almost all of them smol's own
    short-code languages that DO have real BOUQuET dev coverage, just under
    a differently-spelled key ("en" vs. BOUQuET's own "eng_Latn").

    langcodes.Language.get(code).to_alpha3() handles both conventions
    directly (it already understands BCP-47-style hyphenated subtags like
    "ar-MA"/"pa-Arab" AND this project's own underscore-separated
    lang_Script[_variant] convention, extracting just the base language
    subtag either way -- verified against every code in the real dropped
    lists above). Falls back to the raw code on any parse failure, so a
    genuinely unrecognized code stays self-consistent within its own
    dataset rather than crashing the whole fit."""
    if code in _cache:
        return _cache[code]
    try:
        normalized = langcodes.Language.get(code).to_alpha3()
    except Exception:
        normalized = code
    _cache[code] = normalized
    return normalized


def _group_sentences_by_normalized_lang(sentences_by_lang):
    """{raw_lang_code: [sentence, ...]} -> ({normalized_code: [sentence,
    ...]}, {normalized_code: [raw_code, ...]}) -- merges raw codes that
    normalize to the SAME base language (see _normalize_lang_code) by
    pooling their sentences together, e.g. if a corpus somehow had both
    "en" and "eng_Latn" as separate keys, both would be treated as one
    language's data. The second return value records which raw code(s)
    contributed to each normalized key, purely for a clear diagnostic
    message -- not needed for the fit itself."""
    grouped = defaultdict(list)
    sources = defaultdict(list)
    for code, sentences in sentences_by_lang.items():
        norm = _normalize_lang_code(code)
        grouped[norm].extend(sentences)
        sources[norm].append(code)
    return dict(grouped), dict(sources)

# Matches the official CLI's own --pretokenize default (['whitespace',
# 'bytelevel']) exactly -- see vendor/parity_aware_learn_bpe.py's own module
# docstring for why this must be set as a module attribute on the vendored
# module rather than passed as a parameter (get_vocabulary references it as
# a bare global, a structural quirk of the original CLI-oriented code). Used
# ONLY for FITTING (vocabulary/pair counting over multi-line "files" -- see
# fit_parity_bpe), where ByteLevel's own add_prefix_space=True default
# (unset here, so it applies) is harmless: every sentence in a joined
# multi-line file already has genuine preceding whitespace except the very
# first word of the very first sentence in the whole corpus.
_FIT_PRE_TOKENIZER = Sequence([Whitespace(), ByteLevel(use_regex=False)])

# Used for the RETURNED model's own encode_spans -- DELIBERATELY not the
# Whitespace()+ByteLevel(use_regex=False) chain used for fitting. Confirmed
# live: that chain lets Whitespace() split off and DISCARD every whitespace
# character before ByteLevel ever sees it, and since Sequence re-processes
# each resulting piece through ByteLevel INDEPENDENTLY, add_prefix_space
# then applies to EVERY piece uniformly, not just the first -- so
# add_prefix_space=True (the official default) reconstructs inter-word
# spaces correctly but adds a spurious leading space at the very start of
# any independent encode() call, while add_prefix_space=False loses EVERY
# space, not just that spurious one (b"".join(encode_spans("hello world"))
# produced "hellow world" and "helloworld" respectively -- both wrong).
# ByteLevel's OWN internal regex (use_regex=True, its default, no separate
# Whitespace() stage at all) instead reads the RAW string directly and
# correctly attaches genuine leading whitespace (any run of it, not just a
# single space) to the following word as part of one pretoken, matching
# GPT-2/RoBERTa's real convention and this project's OTHER `tokenizers`-
# backed baseline (bpe.model, which uses exactly ByteLevel(add_prefix_space=
# False) alone for the same byte-exact-reconstruction reason). Functionally
# compatible with merges LEARNED under the official Whitespace()+ByteLevel
# chain: BPE merges operate on the resulting remapped-unicode strings
# regardless of which pre-tokenizer produced them, and for ordinary
# single-space-separated text both conventions produce the same "Ġword"-
# style pretoken strings for every word except a call's very first (where
# this one is correct and the official default isn't).
_ENCODE_PRE_TOKENIZER = ByteLevel(add_prefix_space=False)


def _to_str(text):
    # Same rationale as bpe.model._to_str: tokenizers' encode wants str, and
    # ByteLevel re-encodes to UTF-8 internally, so "replace" (not
    # "surrogateescape") avoids an uncatchable crash on invalid byte runs.
    if isinstance(text, str):
        return text
    return bytes(text).decode("utf-8", errors="replace")


def _token_to_bytes(token_str):
    """Inverts the ByteLevel byte<->unicode mapping to recover a token's
    actual bytes -- same as bpe.model._token_to_bytes."""
    return bytes(_UNICODE_TO_BYTE[ch] for ch in token_str)


def fit_parity_bpe(
    sentences_by_lang, dev_sentences_by_lang, vocab_size,
    num_global_merges=0, use_moving_window=False, window_size=100, alpha=2,
    min_frequency=2, verbose=False,
    checkpoint_dir=None, checkpoint_every=500,
):
    """Fits a ParityBPEModel by calling the OFFICIAL learn_bpe/
    learn_bpe_moving_window implementation directly (see module docstring).

    sentences_by_lang: dict[lang -> list[str/bytes]], the training corpus.
    dev_sentences_by_lang: dict[lang -> list[str/bytes]], a SEPARATE small
    multi-parallel development corpus used purely to track each language's
    current compression rate (this project's own ParityBPETrainer passes
    BOUQuET dev here -- exactly the corpus shape the paper recommends).

    Only languages appearing in BOTH corpora, in a shared DETERMINISTIC order
    (sorted language code -- their own file-list-positional API has no
    named-language concept, see README's "nth input corpus corresponds to
    the nth dev corpus"), can drive the fairness objective. Train/dev
    language codes are matched after NORMALIZATION (see
    _normalize_lang_code) -- a short code (e.g. "en") and this project's
    own lang_Script convention (e.g. "eng_Latn") for the same language are
    recognized as the same key, not silently treated as unrelated and
    dropped. Reports (never silently drops without saying so) any
    still-unmatched train-only/dev-only languages after normalization.

    checkpoint_dir (optional): if given, fits via .checkpointed_fit instead
    of calling vendor.learn_bpe/learn_bpe_moving_window directly -- a
    provably byte-identical-on-resume reimplementation of the same loop
    (see that module's own docstring), for real long-running fits that may
    span multiple SLURM job time limits. None (default): calls the official
    implementation directly, maximizing reuse for the common case where a
    fit finishes in one run.

    Returns a ParityBPEModel.
    """
    vendor.pre_tokenizer = _FIT_PRE_TOKENIZER

    train_grouped, train_sources = _group_sentences_by_normalized_lang(sentences_by_lang)
    dev_grouped, dev_sources = _group_sentences_by_normalized_lang(dev_sentences_by_lang)

    merged_train = {norm: raws for norm, raws in train_sources.items() if len(raws) > 1}
    merged_dev = {norm: raws for norm, raws in dev_sources.items() if len(raws) > 1}
    if merged_train or merged_dev:
        print(
            f"[parity_bpe] language-code normalization merged multiple raw codes into one "
            f"language: {len(merged_train)} on the train side, {len(merged_dev)} on the dev side "
            f"(e.g. train: {dict(list(merged_train.items())[:3])}, dev: {dict(list(merged_dev.items())[:3])})"
        )

    train_langs = set(train_grouped)
    dev_langs = set(dev_grouped)
    usable_langs = sorted(train_langs & dev_langs)
    if not usable_langs:
        raise ValueError(
            "fit_parity_bpe: no language appears in BOTH the training data and the dev set, even "
            f"after code normalization -- train languages: {sorted(train_langs)}, dev languages "
            f"({len(dev_langs)} total): {sorted(dev_langs)[:20]}{'...' if len(dev_langs) > 20 else ''}. "
            "Parity-aware BPE needs at least one shared language to compute a compression-rate signal at all."
        )
    dropped_dev_only = dev_langs - train_langs
    if dropped_dev_only:
        print(
            f"[parity_bpe] {len(dropped_dev_only)} dev-set language(s) have no corresponding "
            "training data even after code normalization -- excluded (a dev set routinely covers "
            f"more languages than any one training run's --langs subset): "
            f"{sorted(dropped_dev_only)[:10]}{'...' if len(dropped_dev_only) > 10 else ''}"
        )
    train_only = train_langs - dev_langs
    if train_only:
        print(
            f"[parity_bpe] {len(train_only)} training language(s) have no dev-set compression "
            f"signal even after code normalization -- excluded from this fit entirely (the "
            f"official implementation's --input/--dev lists are positionally paired, with no way "
            f"to include a training-only language that contributes no dev signal): {sorted(train_only)}"
        )

    infiles = [io.StringIO("\n".join(_to_str(s) for s in train_grouped[lang]) + "\n") for lang in usable_langs]
    devfiles = [io.StringIO("\n".join(_to_str(s) for s in dev_grouped[lang]) + "\n") for lang in usable_langs]

    num_symbols = max(0, vocab_size - len(ByteLevel.alphabet()))
    common_kwargs = dict(
        min_frequency=min_frequency, verbose=verbose, num_global=num_global_merges,
    )
    if checkpoint_dir:
        checkpoint_path = os.path.join(checkpoint_dir, "parity_bpe_fit_checkpoint.pkl")
        os.makedirs(checkpoint_dir, exist_ok=True)
        merges = checkpointed_fit.fit_checkpointed(
            infiles, devfiles, num_symbols,
            use_moving_window=use_moving_window, window_size=window_size, alpha=alpha,
            checkpoint_path=checkpoint_path, checkpoint_every=checkpoint_every,
            **common_kwargs,
        )
    else:
        outfile = io.StringIO()
        if use_moving_window:
            vendor.learn_bpe_moving_window(
                infiles, outfile, devfiles, num_symbols, window_size=window_size, alpha=alpha, **common_kwargs
            )
        else:
            vendor.learn_bpe(infiles, outfile, devfiles, num_symbols, **common_kwargs)
        lines = [line for line in outfile.getvalue().splitlines() if line]
        merges = lines[1:] if lines and lines[0].startswith("#version:") else lines

    merge_lines = merges
    vocab = build_vocab_from_merges(merge_lines)
    merge_pairs = [tuple(line.split()) for line in merge_lines]

    tokenizer = Tokenizer(BPE(vocab=vocab, merges=merge_pairs, unk_token=None))
    tokenizer.pre_tokenizer = _ENCODE_PRE_TOKENIZER
    return ParityBPEModel(tokenizer)


class ParityBPEModel:
    """Thin wrapper around a trained `tokenizers.Tokenizer` -- exposes
    encode_spans/num_parameters like every other tokenizer's model object,
    same shape as bpe.model.BPEModel (this project's other `tokenizers`-
    backed baseline)."""

    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer

    def num_parameters(self):
        return self.tokenizer.get_vocab_size()

    def encode_spans(self, raw):
        """str/bytes -> list[bytes] spans -- what common.eval.cross_tokenizer/common.vocab
        expect from every tokenizer's induce_spans (see segment.py)."""
        encoding = self.tokenizer.encode(_to_str(raw), add_special_tokens=False)
        return [_token_to_bytes(tok) for tok in encoding.tokens]
