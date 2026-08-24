"""Offline pipeline: stream a corpus from common.data.corpora's shared
registry, tokenize every document with a systems/ checkpoint (via
TokenizerAdapter), and pack the token ids into fixed-size binary shards on
disk -- tokenize-once, train-many-times, reading flat memory-mapped files
(see shard_dataset.py) instead of re-streaming/re-tokenizing every run.

Each (lang, text) document is tokenized with its own language as encode()'s
`lang` hint (needed for MAGNET's per-script boundary predictor), then
packed with the tokenizer's own eos_id as the only document-boundary
marker. Shards are written as they fill (--shard-size tokens).

LANGUAGE TRACKING (shards_meta.json's "lang_counts") is corpus-level only,
not per training step -- ShardedTokenDataset's random windows can straddle
documents of different languages. Tracks "bytes" alongside "tokens"/"docs"
for a real compression_rate plus a Gini coefficient over per-language token share.

DEDUP (--dedup, on by default): common.data.dedup.Deduplicator drops exact
and near-duplicate (MinHash+LSH) documents before tokenization.

MAX_DOC_BYTES (--max-doc-bytes, default 4096): truncates each document
before encode(). Guards a confirmed incident: an unusually long Glot500
document reached a neural system's dense attention and OOM'd a cluster GPU.

PERFORMANCE (--encode-batch-size, default 32): documents batch through
TokenizerAdapter.encode_batch -- confirmed fix: a cluster run measured
~56.8k tok/s on FANTA (A100) vs bpe/superbpe's ~74-83k tok/s CPU, because
manta.segment.induce_boundaries_batch paid full GPU kernel-launch overhead
per document at batch size 1. Only manta/fanta get a real speedup from
this; the rest fall back to a correct but unsped-up per-item loop.
MEMORY CAVEAT: a batch pads to its longest member, so memory scales with
(batch max length)^2 * batch_size -- tune --max-doc-bytes and
--encode-batch-size together.

BUCKETING (--bucket-pool-multiplier, default 8): glot500 "all" round-robins
~411 languages, so consecutive documents vary wildly in length and an
unsorted batch pays the longest member's O(T^2) cost -- confirmed a real
~1000x throughput collapse, not just a memory risk. Fix: accumulate a pool
of encode_batch_size * bucket_pool_multiplier documents, sort by length,
slice into encode_batch_size chunks. Reorders documents relative to the
source stream (harmless for ShardedTokenDataset's random sampling);
max_tokens/max_docs are checked at pool granularity. =1 disables bucketing.

PREFETCH (--prefetch, default off): confirmed live that pulling documents
from glot500's ~411-config HF stream is NETWORK-bound, not CPU-bound (a
real run took multiple days despite encode_batch() itself measuring fine).
A background thread runs the network pull AND this loop's own per-document
dedup + truncation (_dedup_and_truncate) -- the two per-document CPU costs
that would otherwise sit in the same thread as the pool-accumulation/
encode/pack work below -- pushing the resulting ("dup", ...)/("keep", ...)
records onto a bounded queue (--prefetch-queue-size, default 2 * pool_size)
that the consumer reads instead of iterating directly. This changes only
*when* a document's dedup/truncation outcome is COMPUTED relative to the
consumer's own pool/encode/pack work, not its order or content (dedup still
sees every document in stream order, from a single thread, exactly as
without --prefetch) -- every RESUME/dedup/checkpoint invariant below is
unaffected. Off by default: verify with a real timing comparison before
relying on it, since the win depends on how much of the slowness is really
network/dedup wait vs. e.g. HF `datasets`' own decode overhead or the
encode_batch() step this overlaps with. A producer-thread exception is
re-raised in the main thread via a queue sentinel, not swallowed.

RESUME (--prep-checkpoint-path, default "<output-dir>/prep_checkpoint.json"):
lets a run killed mid-way (e.g. a SLURM time limit) continue instead of
restarting from token 0. Checkpointed once per pool that crosses a shard
boundary, never mid-pool (bucketing sorts by length, so stream_docs_consumed
is only a safe resume point once a whole pool finishes). Written atomically
(temp + os.replace, buffer file first) so a crash never references buffer
data that isn't on disk. Loaded automatically if present at startup (no
separate --resume flag); deleted once shards_meta.json is written.

Two accepted tradeoffs: the Deduplicator's state isn't persisted (a
duplicate straddling the resume boundary could slip through), and
fast-forwarding assumes stream_groups yields documents in the same order
across invocations -- neither is a correctness hazard, since
shard_dataset.py samples random windows regardless of corpus order.
"""

import argparse
import json
import os
import queue
import threading
from collections import defaultdict

import numpy as np
from tqdm.auto import tqdm

from common.bytes_utils import truncate_to_max_bytes
from common.config_file import parse_args_with_config
from common.data.corpora import (
    ALL_SOURCES,
    BITEXT_SOURCES,
    FINEWEB_EDU_CONFIGS,
    MONOLINGUAL_SOURCES,
    OLMO_MIX_CONFIGS,
    stream_groups,
)
from common.data.dedup import Deduplicator
from common.eval.metrics import compression_rate, gini_coefficient

from .tokenizer_adapter import ALL_SYSTEMS, TokenizerAdapter

SHARD_SIZE = 100_000_000  # tokens/shard (~200MB uint16, ~400MB uint32) --
# large enough that per-shard overhead is negligible, small enough that
# already-flushed shards from a partial run stay individually valid.

MAX_DOC_BYTES = 4096  # default --max-doc-bytes; see module docstring's
# MAX_DOC_BYTES section. Deliberately non-zero by default (unlike
# FantaConfig.max_seq_length's "0 disables" convention) -- an unbounded
# default caused two separate real OOM crashes from the same underlying
# cause (long document -> uncapped dense attention).

ENCODE_BATCH_SIZE = 32  # default --encode-batch-size; see module
# docstring's PERFORMANCE and MEMORY CAVEAT sections.

BUCKET_POOL_MULTIPLIER = 8  # default --bucket-pool-multiplier; see module
# docstring's BUCKETING section (fixes a real throughput regression).


def _dtype_for_vocab(vocab_size):
    return "uint16" if vocab_size <= 65536 else "uint32"


class _ProducerError:
    """Wraps an exception raised on _produce's background thread so
    it can travel through a queue.Queue and be re-raised on the consuming
    (main) thread -- see PREFETCH docstring section. A bare background
    thread's exception would otherwise vanish silently."""

    def __init__(self, exc):
        self.exc = exc


_DONE = object()  # sentinel: the producer thread exhausted the stream normally


def _document_source(stream, skip_target):
    """Single-threaded generator yielding (lang, text, stream_docs_consumed)
    for every document in `stream`, in stream order -- this is exactly
    prep_dataset's own pre-prefetch iteration (see RESUME docstring
    section), just factored out so it can be wrapped in a --prefetch
    background thread (see _dedup_and_truncate/_prefetch below). Applies
    the RESUME fast-forward skip: documents at or
    below skip_target are counted (stream_docs_consumed still advances,
    matching the checkpoint's own semantics) but never yielded."""
    stream_docs_consumed = 0
    for group in stream:
        for lang, text in group.items():
            if stream_docs_consumed < skip_target:
                stream_docs_consumed += 1
                continue
            stream_docs_consumed += 1
            yield lang, text, stream_docs_consumed


def _produce(gen_factory, q, stop_event, put_timeout=0.5):
    """Runs gen_factory() (a zero-arg callable building the generator to
    iterate -- called ON this background thread, not the caller's, so any
    per-item work the generator does, e.g. _dedup_and_truncate's dedup/
    truncation, also runs here) on a background thread, pushing each item
    onto `q` instead of returning it directly (see PREFETCH docstring
    section). q.put() loops on a timeout rather than blocking indefinitely,
    so a full queue can't deadlock this thread after the consumer has
    already decided to stop (stop_event set, e.g. because max_tokens/
    max_docs was reached) while this thread is blocked trying to enqueue.
    Always pushes exactly one terminal item (_DONE, or a _ProducerError
    wrapping any exception) so the consumer's q.get() can never block
    forever if this thread dies or is asked to stop early."""

    def put(item):
        while not stop_event.is_set():
            try:
                q.put(item, timeout=put_timeout)
                return True
            except queue.Full:
                continue
        return False

    try:
        for item in gen_factory():
            if not put(item):
                return  # stop_event fired while blocked on a full queue
            if stop_event.is_set():
                return
    except Exception as e:
        put(_ProducerError(e))
        return
    put(_DONE)


def _prefetch(gen_factory, queue_size):
    """Wraps _produce's background thread + bounded queue behind the SAME
    plain-generator interface gen_factory() itself exposes, so a consumer
    loop can be written once and reused whether or not it's threaded (see
    prep_dataset's own record_source/build_records). Raises in the
    CONSUMING thread if the producer thread itself raised. Closing this
    generator early (e.g. the consumer `break`s out of its loop, or an
    exception propagates through it) signals stop_event and joins the
    producer thread before returning -- callers that iterate this with a
    `for ... in` loop and then `break` should call .close() (or rely on
    Python's own GeneratorExit-on-garbage-collection) before doing anything
    that assumes the producer has stopped, e.g. the final
    process_pending(final=True)/flush() in prep_dataset."""
    q = queue.Queue(maxsize=queue_size)
    stop_event = threading.Event()
    thread = threading.Thread(target=_produce, args=(gen_factory, q, stop_event), daemon=True)
    thread.start()
    try:
        while True:
            item = q.get()
            if item is _DONE:
                return
            if isinstance(item, _ProducerError):
                raise item.exc
            yield item
    finally:
        stop_event.set()
        thread.join()


def _dedup_and_truncate(document_source, deduper, max_doc_bytes):
    """Wraps document_source (the RESUME-aware (lang, text,
    stream_docs_consumed) stream _document_source yields) with the two
    per-document CPU costs -- dedup and truncation -- that used to run
    inline in prep_dataset's own consumer loop. Factored out so --prefetch
    can run this on its background thread too (see PREFETCH docstring
    section), overlapping it with the consumer's pool-accumulation/encode/
    pack work instead of paying for both sequentially in the same thread.

    deduper.is_duplicate(text) mutates deduper's own internal (MinHash/LSH)
    state and must see every document in stream order -- guaranteed here
    since this generator (run directly, or as --prefetch's single producer
    thread) is document_source's only consumer, so document order into
    deduper is identical to iterating document_source directly.

    Yields one of:
      ("dup", lang, raw_byte_len, stream_docs_consumed)
      ("keep", lang, encode_bytes, raw_byte_len, was_truncated, stream_docs_consumed)
    Empty documents are dropped silently, matching prep_dataset's own prior
    inline `if not text: continue`."""
    for lang, text, stream_docs_consumed in document_source:
        if not text:
            continue
        raw_bytes = text.encode("utf-8") if isinstance(text, str) else bytes(text)
        if deduper is not None and deduper.is_duplicate(text):
            yield ("dup", lang, len(raw_bytes), stream_docs_consumed)
            continue
        # Truncated for encoding only -- dedup already compared the
        # untruncated text, and the caller's lang_counts["bytes"] should
        # still count the full raw_bytes length, so a truncated document's
        # compression_rate is very slightly inflated. Accepted since
        # truncation is meant to be rare (see MAX_DOC_BYTES docstring section).
        encode_bytes, was_truncated = truncate_to_max_bytes(raw_bytes, max_doc_bytes)
        yield ("keep", lang, encode_bytes, len(raw_bytes), was_truncated, stream_docs_consumed)


def prep_dataset(
    dataset_name,
    langs=None,
    dataset_config=None,
    system=None,
    checkpoint_path=None,
    output_dir=None,
    vocab_json_path=None,
    max_tokens=None,
    max_docs=None,
    device="cpu",
    shard_size=SHARD_SIZE,
    dedup=True,
    dedup_near_threshold=0.8,
    dedup_num_perm=128,
    dedup_shingle_size=13,
    dedup_min_words_for_near_dup=50,
    max_doc_bytes=MAX_DOC_BYTES,
    encode_batch_size=ENCODE_BATCH_SIZE,
    bucket_pool_multiplier=BUCKET_POOL_MULTIPLIER,
    prep_checkpoint_path=None,
    prefetch=False,
    prefetch_queue_size=None,
):
    """dataset_name: one of common.data.corpora.ALL_SOURCES. langs: codes for
    the language-selectable sources (synthetic/oldi_seed/flores_dev/glot500
    default to "all"; bible_nlp takes an arbitrary subset); ignored for
    fineweb_edu/olmo_mix and BITEXT_SOURCES, which use `dataset_config`
    instead (see common.data.corpora.stream_groups). max_tokens/max_docs:
    stop once either is reached (None disables); checked against KEPT
    (post-dedup) counts. dedup/dedup_*: see common.data.dedup.Deduplicator
    and the module docstring's DEDUPLICATION section. max_doc_bytes: see
    MAX_DOC_BYTES section above; 0 or None disables truncation entirely.
    encode_batch_size/bucket_pool_multiplier: see PERFORMANCE and BUCKETING
    sections above -- max_tokens/max_docs are checked at POOL granularity
    (every encode_batch_size * bucket_pool_multiplier documents), so a
    capped run may overshoot by up to one pool's worth. prep_checkpoint_path:
    see RESUME section; defaults to "<output_dir>/prep_checkpoint.json".
    prefetch/prefetch_queue_size: see PREFETCH section; prefetch_queue_size
    defaults to 2 * pool_size (encode_batch_size * bucket_pool_multiplier).

    Returns the shards_meta.json dict this also writes to output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)
    adapter = TokenizerAdapter.load(system, checkpoint_path, vocab_json_path, device=device)
    dtype_name = _dtype_for_vocab(adapter.vocab_size)
    dtype = np.uint16 if dtype_name == "uint16" else np.uint32

    if prep_checkpoint_path is None:
        prep_checkpoint_path = os.path.join(output_dir, "prep_checkpoint.json")

    if dataset_name in ("fineweb_edu", "olmo_mix") or dataset_name in BITEXT_SOURCES:
        stream = stream_groups(dataset_name, config=dataset_config)
    else:
        # `config` is also passed here (not just langs) even though most
        # sources in this branch ignore it -- glot500 and bible_nlp both
        # use it as a local disk cache directory override (see
        # common.data.corpora.stream_groups's own docstring); omitting it
        # made that override unreachable from this CLI entirely.
        stream = stream_groups(dataset_name, langs=langs, config=dataset_config)

    resume = os.path.exists(prep_checkpoint_path)
    if resume:
        with open(prep_checkpoint_path) as f:
            _ckpt = json.load(f)
        print(f"\nresuming from {prep_checkpoint_path}: {_ckpt['total_tokens']:,} tokens / "
              f"{_ckpt['num_docs']:,} docs / {len(_ckpt['shard_files'])} shards already written")
    else:
        _ckpt = None

    shard_files = list(_ckpt["shard_files"]) if resume else []
    buffer = np.empty(shard_size, dtype=dtype)
    buffer_pos = 0
    if resume and _ckpt["buffer_pos"] > 0:
        # Restore whatever was sitting in the buffer, not yet flushed to a
        # shard, at checkpoint time -- see write_prep_checkpoint's own
        # docstring for why this is persisted rather than forcing an early
        # partial-shard flush on every checkpoint.
        buffer_pos = _ckpt["buffer_pos"]
        buffer[:buffer_pos] = np.fromfile(
            prep_checkpoint_path + ".buffer.bin", dtype=dtype, count=buffer_pos
        )
    total_tokens = _ckpt["total_tokens"] if resume else 0
    num_docs = _ckpt["num_docs"] if resume else 0
    shard_idx = _ckpt["shard_idx"] if resume else 0
    stream_docs_consumed = 0  # this RUN's own count of (lang, text) pairs
    # pulled from a FRESH stream_groups(...) call above -- compared against
    # skip_target below to fast-forward past whatever a prior run already
    # consumed (see RESUME docstring section); unrelated to num_docs, which
    # only counts KEPT (post-dedup) documents.
    skip_target = _ckpt["stream_docs_consumed"] if resume else 0
    # {lang: {"docs": int, "tokens": int, "bytes": int}} -- the REALIZED
    # corpus makeup, not the requested `langs` (see module docstring's own
    # LANGUAGES ENCOUNTERED section for why this is tracked here, at the
    # corpus level, rather than per training step). "tokens" counts each
    # document's ids INCLUDING its trailing eos_id, so summing every
    # language's "tokens" here always equals total_tokens exactly -- a
    # cheap, useful invariant to sanity-check this against (see the assert
    # right after the loop). "bytes" is the RAW UTF-8 byte length before
    # tokenization -- together with "tokens" this gives a real
    # bytes-per-token compression rate (common.eval.metrics.compression_rate) at
    # actual pretraining-corpus scale, not just the much smaller sample a
    # systems/ tokenizer's own training-time smoke test measures against.
    lang_counts = defaultdict(
        lambda: {"docs": 0, "tokens": 0, "bytes": 0}, _ckpt["lang_counts"] if resume else {}
    )
    # Deliberately NOT restored from the checkpoint -- see RESUME docstring
    # section's own tradeoffs paragraph for why a fresh Deduplicator on
    # resume is an accepted imprecision, not an oversight.
    deduper = (
        Deduplicator(
            near_dup_threshold=dedup_near_threshold,
            num_perm=dedup_num_perm,
            shingle_size=dedup_shingle_size,
            min_words_for_near_dup=dedup_min_words_for_near_dup,
        )
        if dedup
        else None
    )
    dropped_dup_docs = _ckpt["dropped_duplicate_docs"] if resume else 0
    dropped_dup_bytes = _ckpt["dropped_duplicate_bytes"] if resume else 0
    dropped_dup_by_lang = defaultdict(
        lambda: {"docs": 0, "bytes": 0}, _ckpt["dropped_duplicates_by_lang"] if resume else {}
    )
    num_truncated_docs = _ckpt["num_truncated_docs"] if resume else 0
    num_truncated_by_lang = defaultdict(
        int, _ckpt["num_truncated_by_lang"] if resume else {}
    )

    def flush():
        nonlocal buffer_pos, shard_idx
        if buffer_pos == 0:
            return
        name = f"shard_{shard_idx:05d}.bin"
        buffer[:buffer_pos].tofile(os.path.join(output_dir, name))
        shard_files.append(name)
        shard_idx += 1
        buffer_pos = 0

    buffer_checkpoint_path = prep_checkpoint_path + ".buffer.bin"

    def write_prep_checkpoint():
        # Called only once per process_pending() call, right after its
        # ENTIRE pending pool has been fully processed (see the call site
        # below and RESUME docstring section) -- NEVER mid-pool. This
        # matters because bucketing (see BUCKETING docstring section)
        # processes a pool's documents in LENGTH-SORTED order, not stream
        # order, so stream_docs_consumed (incremented in original stream
        # order, in the outer loop below) only correctly corresponds to
        # "everything up to here is fully accounted for" once the WHOLE
        # pool that reordering was drawn from has been processed, not at
        # some arbitrary partial point inside it.
        #
        # buffer_pos may be > 0 here (a shard boundary doesn't necessarily
        # land exactly at a pool boundary) -- buffer[:buffer_pos] is
        # persisted to a sibling .buffer.bin file rather than forcing an
        # early partial shard flush that would fragment shard files.
        ckpt = {
            "stream_docs_consumed": stream_docs_consumed,
            "total_tokens": total_tokens,
            "num_docs": num_docs,
            "shard_idx": shard_idx,
            "shard_files": shard_files,
            "buffer_pos": buffer_pos,
            "lang_counts": dict(lang_counts),
            "dropped_duplicate_docs": dropped_dup_docs,
            "dropped_duplicate_bytes": dropped_dup_bytes,
            "dropped_duplicates_by_lang": dict(dropped_dup_by_lang),
            "num_truncated_docs": num_truncated_docs,
            "num_truncated_by_lang": dict(num_truncated_by_lang),
        }
        buffer_tmp_path = buffer_checkpoint_path + ".tmp"
        buffer[:buffer_pos].tofile(buffer_tmp_path)
        os.replace(buffer_tmp_path, buffer_checkpoint_path)
        tmp_path = prep_checkpoint_path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(ckpt, f)
        os.replace(tmp_path, prep_checkpoint_path)  # atomic on POSIX; written
        # after the buffer file so a crash between the two never leaves a
        # checkpoint referencing buffer content that isn't on disk yet.

    # (lang, encode_bytes, raw_byte_len) tuples awaiting a bucketed, batched
    # encode() call (see PERFORMANCE/BUCKETING docstring sections). Dedup
    # and truncation stay per-document since they're cheap and only need
    # to run once each regardless of batching.
    pending = []
    pool_size = encode_batch_size * max(bucket_pool_multiplier, 1)

    def process_pending(final=False):
        nonlocal buffer_pos, shard_idx, total_tokens, num_docs
        if not pending:
            return
        if not final and len(pending) < pool_size:
            return  # keep accumulating until the pool is full (or the stream ends)
        # Sorted by length so each encode_batch_size chunk below contains
        # similarly-sized documents (see BUCKETING docstring section).
        pending.sort(key=lambda p: len(p[1]))
        flushed_this_call = False
        for start in range(0, len(pending), encode_batch_size):
            chunk = pending[start : start + encode_batch_size]
            langs_batch = [p[0] for p in chunk]
            bytes_batch = [p[1] for p in chunk]
            ids_list = adapter.encode_batch(bytes_batch, langs_batch)
            for (lang, _, raw_len), ids in zip(chunk, ids_list):
                ids = list(ids)
                ids.append(adapter.eos_id)
                num_docs += 1
                lang_counts[lang]["docs"] += 1
                lang_counts[lang]["tokens"] += len(ids)
                lang_counts[lang]["bytes"] += raw_len

                i = 0
                while i < len(ids):
                    n = min(len(ids) - i, shard_size - buffer_pos)
                    buffer[buffer_pos : buffer_pos + n] = ids[i : i + n]
                    buffer_pos += n
                    i += n
                    total_tokens += n
                    pbar.update(n)
                    if buffer_pos == shard_size:
                        flush()
                        flushed_this_call = True
        pending.clear()
        # Checkpointed once here, after the whole pool is done (mid-pool
        # would be unsafe given bucketing's reordering -- see
        # write_prep_checkpoint's docstring), and only when this pool
        # crossed a shard boundary (keeps checkpoint frequency ~once per
        # shard). Skipped on the final call since the checkpoint gets
        # deleted immediately after prep_dataset finishes anyway.
        if flushed_this_call and not final:
            write_prep_checkpoint()

    pbar = tqdm(
        desc=f"tokenizing {dataset_name}", unit="tok", unit_scale=True,
        initial=total_tokens if resume else 0,
    )
    def build_records():
        return _dedup_and_truncate(_document_source(stream, skip_target), deduper, max_doc_bytes)

    if prefetch:
        resolved_queue_size = prefetch_queue_size if prefetch_queue_size else 2 * pool_size
        record_source = _prefetch(build_records, resolved_queue_size)
    else:
        record_source = build_records()

    try:
        for record in record_source:
            if record[0] == "dup":
                # Dropped before tokenization; still counted per language
                # so the drop rate stays visible in reports.
                _, lang, raw_len, stream_docs_consumed = record
                dropped_dup_docs += 1
                dropped_dup_bytes += raw_len
                dropped_dup_by_lang[lang]["docs"] += 1
                dropped_dup_by_lang[lang]["bytes"] += raw_len
                continue
            _, lang, encode_bytes, raw_len, was_truncated, stream_docs_consumed = record
            if was_truncated:
                num_truncated_docs += 1
                num_truncated_by_lang[lang] += 1
            pending.append((lang, encode_bytes, raw_len))
            process_pending()  # no-op unless the pool has reached pool_size

            if (max_tokens and total_tokens >= max_tokens) or (max_docs and num_docs >= max_docs):
                break
    finally:
        # For --prefetch, this signals the producer thread to stop and
        # joins it (see _prefetch's own docstring) -- must happen BEFORE
        # the final process_pending/flush below, not concurrently with
        # them. A no-op for the plain build_records() generator.
        record_source.close()
    process_pending(final=True)  # flush any partial pool smaller than pool_size
    flush()
    pbar.close()

    assert sum(c["tokens"] for c in lang_counts.values()) == total_tokens, (
        "lang_counts token sum diverged from total_tokens -- a real bug in the "
        "accounting above, not just a cosmetic mismatch"
    )

    # Gini coefficient over each language's realized token share: 0 means
    # every language got an equal share, closer to 1 means a few languages
    # dominate -- a single top-line balance number for this run's data mix.
    lang_token_gini = gini_coefficient([c["tokens"] for c in lang_counts.values()])

    if dedup:
        docs_seen = num_docs + dropped_dup_docs
        drop_rate = dropped_dup_docs / docs_seen if docs_seen else 0.0
        print(
            f"\ndedup: dropped {dropped_dup_docs:,} of {docs_seen:,} documents seen "
            f"({drop_rate:.1%}, {dropped_dup_bytes:,} bytes) as exact/near duplicates"
        )
        for lang, counts in sorted(dropped_dup_by_lang.items(), key=lambda kv: -kv[1]["docs"]):
            print(f"    {lang:12s} dropped_docs={counts['docs']:8,d}  dropped_bytes={counts['bytes']:12,d}")

    if max_doc_bytes:
        trunc_rate = num_truncated_docs / num_docs if num_docs else 0.0
        print(
            f"\nmax_doc_bytes={max_doc_bytes}: truncated {num_truncated_docs:,}/{num_docs:,} "
            f"kept documents ({trunc_rate:.2%}) before encoding"
        )
        for lang, count in sorted(num_truncated_by_lang.items(), key=lambda kv: -kv[1]):
            print(f"    {lang:12s} truncated_docs={count:,}")

    print(f"\n{len(lang_counts)} language(s) encountered (token-share gini={lang_token_gini:.4f}):")
    for lang, counts in sorted(lang_counts.items(), key=lambda kv: -kv[1]["tokens"]):
        frac = counts["tokens"] / total_tokens if total_tokens else 0.0
        # "tokens" includes each document's trailing eos_id, so this rate
        # is very slightly lower than the tokenizer's true rate; not
        # corrected for since the bias is tiny and uniform across languages.
        rate = compression_rate(counts["bytes"], counts["tokens"])
        print(
            f"  {lang:12s} docs={counts['docs']:8,d}  tokens={counts['tokens']:12,d}  "
            f"bytes={counts['bytes']:12,d}  compression_rate={rate:6.3f}  ({frac:5.1%})"
        )

    meta = {
        "dataset": dataset_name,
        "langs": langs,
        "dataset_config": dataset_config,
        "system": system,
        "checkpoint": checkpoint_path,
        "vocab_size": adapter.vocab_size,
        "eos_id": adapter.eos_id,
        "dtype": dtype_name,
        "total_tokens": total_tokens,
        "num_docs": num_docs,
        "shard_files": shard_files,
        "lang_counts": dict(lang_counts),  # realized makeup, vs. "langs" (requested)
        "lang_token_gini": lang_token_gini,
        "dedup_enabled": dedup,
        "dropped_duplicate_docs": dropped_dup_docs,
        "dropped_duplicate_bytes": dropped_dup_bytes,
        "dropped_duplicates_by_lang": dict(dropped_dup_by_lang),
        "max_doc_bytes": max_doc_bytes,
        "num_truncated_docs": num_truncated_docs,
        "num_truncated_by_lang": dict(num_truncated_by_lang),
    }
    with open(os.path.join(output_dir, "shards_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    if os.path.exists(prep_checkpoint_path):
        # A finished run is no longer resumable; a stale checkpoint would
        # make a later identical invocation waste time fast-forwarding.
        os.remove(prep_checkpoint_path)
    if os.path.exists(buffer_checkpoint_path):
        os.remove(buffer_checkpoint_path)
    print(
        f"\nwrote {total_tokens:,} tokens ({num_docs:,} documents) across "
        f"{len(shard_files)} shards to {output_dir}"
    )
    return meta


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Tokenize a corpus (common.data.corpora registry) into packed token shards."
    )
    parser.add_argument("--dataset", choices=ALL_SOURCES, required=True)
    parser.add_argument(
        "--langs",
        type=str,
        default=None,
        help="comma-separated language codes -- defaults to 'all' for oldi_seed/flores_dev/"
        "glot500, arbitrary subset for bible_nlp; ignored for fineweb_edu/olmo_mix and "
        "BITEXT_SOURCES (smol/ccmatrix/un_pc/europarl/tatoeba_mt), which use --dataset-config",
    )
    parser.add_argument(
        "--dataset-config",
        type=str,
        default=None,
        help=f"HF config name for --dataset fineweb_edu (choices: {FINEWEB_EDU_CONFIGS}) or "
        f"olmo_mix (choices: {OLMO_MIX_CONFIGS}); a native pair name or 'all' for ccmatrix/"
        f"un_pc/europarl (see common.data.corpora.list_bitext_configs); a '{{split}}/{{pair-or-"
        f"all}}' string (e.g. 'test/deu-eng') or bare 'all' for tatoeba_mt (see "
        f"common.data.corpora.list_tatoeba_mt_pairs); for --dataset glot500/bible_nlp, an "
        f"OVERRIDE for their local disk cache directory (default GLOT500_LOCAL_DIR/"
        f"BIBLE_NLP_LOCAL_DIR -- see common.data.prepare_glot500/prepare_bible_nlp, which must "
        f"be run once first)",
    )
    parser.add_argument("--system", choices=ALL_SYSTEMS, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument(
        "--vocab-json",
        type=str,
        default=None,
        help="required for the five span-family systems (fairtok/magnet/flexitokens/manta/fanta); "
        "unused for bpe/superbpe",
    )
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--max-docs", type=int, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--shard-size", type=int, default=SHARD_SIZE)
    parser.add_argument(
        "--dedup", action=argparse.BooleanOptionalAction, default=True,
        help="drop exact/near-duplicate documents before tokenizing (see Deduplicator); "
        "--no-dedup disables entirely",
    )
    parser.add_argument("--dedup-near-threshold", type=float, default=0.8, help="MinHash/LSH Jaccard similarity threshold above which a document counts as a near-duplicate")
    parser.add_argument("--dedup-num-perm", type=int, default=128, help="MinHash permutation count -- higher is a more accurate Jaccard estimate at more memory/CPU per document")
    parser.add_argument("--dedup-shingle-size", type=int, default=13, help="word n-gram shingle length for near-dup comparison")
    parser.add_argument("--dedup-min-words-for-near-dup", type=int, default=50, help="documents shorter than this only get exact-dup checked, not MinHash near-dup")
    parser.add_argument(
        "--max-doc-bytes", type=int, default=MAX_DOC_BYTES,
        help="truncate each document to at most this many UTF-8 bytes before encoding -- guards "
        "against OOM from long documents reaching a neural system's dense attention uncapped; "
        "pass 0 to disable",
    )
    parser.add_argument(
        "--encode-batch-size", type=int, default=ENCODE_BATCH_SIZE,
        help="group this many documents into one encode_batch() call -- throughput fix for the "
        "neural systems; interacts with --max-doc-bytes (see MEMORY CAVEAT docstring section)",
    )
    parser.add_argument(
        "--bucket-pool-multiplier", type=int, default=BUCKET_POOL_MULTIPLIER,
        help="sort a pool of encode-batch-size * this-many documents by length before batching -- "
        "fixes a padding-waste throughput regression; 1 disables bucketing",
    )
    parser.add_argument(
        "--prep-checkpoint-path", type=str, default=None,
        help="defaults to '<output-dir>/prep_checkpoint.json'. Rerunning the same command against "
        "the same --output-dir after a mid-run interruption resumes automatically",
    )
    parser.add_argument(
        "--prefetch", action=argparse.BooleanOptionalAction, default=False,
        help="overlap network-bound stream pulling (confirmed the real bottleneck for --dataset "
        "glot500 --langs all) with this loop's own dedup/encode/pack work via a background "
        "producer thread -- see PREFETCH docstring section. Off by default: verify with a real "
        "timing comparison before relying on this for a full-scale run",
    )
    parser.add_argument(
        "--prefetch-queue-size", type=int, default=None,
        help="bounded queue size between the producer thread and this loop when --prefetch is "
        "set; defaults to 2 * (--encode-batch-size * --bucket-pool-multiplier), i.e. roughly two "
        "pools of overlap headroom. Ignored without --prefetch",
    )
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument(
        "--wandb-project", type=str, default="pretraining",
        help="same default project as train/cli_eval/cli_generate; logs job_type='data_prep'",
    )
    parser.add_argument("--run-name", type=str, default="")
    return parser


def main(argv=None):
    args = parse_args_with_config(build_arg_parser(), argv)
    if (args.dataset in MONOLINGUAL_SOURCES - {"glot500"} or args.dataset in BITEXT_SOURCES) and args.langs:
        print(
            f"warning: --langs is ignored for --dataset {args.dataset} "
            "(selected via --dataset-config instead)"
        )
    langs = None
    if args.langs is not None:
        langs = "all" if args.langs == "all" else args.langs.split(",")

    meta = prep_dataset(
        dataset_name=args.dataset,
        langs=langs,
        dataset_config=args.dataset_config,
        system=args.system,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        vocab_json_path=args.vocab_json,
        max_tokens=args.max_tokens,
        max_docs=args.max_docs,
        device=args.device,
        shard_size=args.shard_size,
        dedup=args.dedup,
        dedup_near_threshold=args.dedup_near_threshold,
        dedup_num_perm=args.dedup_num_perm,
        dedup_shingle_size=args.dedup_shingle_size,
        dedup_min_words_for_near_dup=args.dedup_min_words_for_near_dup,
        max_doc_bytes=args.max_doc_bytes,
        encode_batch_size=args.encode_batch_size,
        bucket_pool_multiplier=args.bucket_pool_multiplier,
        prep_checkpoint_path=args.prep_checkpoint_path,
        prefetch=args.prefetch,
        prefetch_queue_size=args.prefetch_queue_size,
    )

    if args.use_wandb:
        import wandb

        run = wandb.init(
            project=args.wandb_project,
            name=args.run_name or None,
            job_type="data_prep",
            config={
                "dataset": args.dataset,
                "langs": langs,
                "dataset_config": args.dataset_config,
                "system": args.system,
                "checkpoint": args.checkpoint,
                "output_dir": args.output_dir,
                "max_tokens": args.max_tokens,
                "max_docs": args.max_docs,
                "dedup": args.dedup,
                "dedup_near_threshold": args.dedup_near_threshold,
                "dedup_num_perm": args.dedup_num_perm,
                "dedup_shingle_size": args.dedup_shingle_size,
                "dedup_min_words_for_near_dup": args.dedup_min_words_for_near_dup,
                "max_doc_bytes": args.max_doc_bytes,
                "encode_batch_size": args.encode_batch_size,
                "bucket_pool_multiplier": args.bucket_pool_multiplier,
                "prefetch": args.prefetch,
                "prefetch_queue_size": args.prefetch_queue_size,
            },
        )
        lang_rows = [
            [
                lang,
                counts["docs"],
                counts["tokens"],
                counts["tokens"] / meta["total_tokens"],
                counts["bytes"],
                compression_rate(counts["bytes"], counts["tokens"]),
            ]
            for lang, counts in sorted(meta["lang_counts"].items(), key=lambda kv: -kv[1]["tokens"])
        ]
        docs_seen = meta["num_docs"] + meta["dropped_duplicate_docs"]
        dedup_rows = [
            [lang, counts["docs"], counts["bytes"]]
            for lang, counts in sorted(
                meta["dropped_duplicates_by_lang"].items(), key=lambda kv: -kv[1]["docs"]
            )
        ]
        run.log(
            {
                "total_tokens": meta["total_tokens"],
                "num_docs": meta["num_docs"],
                "num_languages": len(meta["lang_counts"]),
                "lang_token_gini": meta["lang_token_gini"],
                "lang_counts": wandb.Table(
                    columns=["lang", "docs", "tokens", "token_fraction", "bytes", "compression_rate"],
                    data=lang_rows,
                ),
                "dropped_duplicate_docs": meta["dropped_duplicate_docs"],
                "dropped_duplicate_bytes": meta["dropped_duplicate_bytes"],
                "dropped_duplicate_rate": (
                    meta["dropped_duplicate_docs"] / docs_seen if docs_seen else 0.0
                ),
                "dropped_duplicates_by_lang": wandb.Table(
                    columns=["lang", "dropped_docs", "dropped_bytes"], data=dedup_rows
                ),
                "num_truncated_docs": meta["num_truncated_docs"],
                "truncated_rate": (
                    meta["num_truncated_docs"] / meta["num_docs"] if meta["num_docs"] else 0.0
                ),
            }
        )
        run.finish()
        print(f"logged corpus language makeup to wandb project={args.wandb_project!r}")


if __name__ == "__main__":
    main()
