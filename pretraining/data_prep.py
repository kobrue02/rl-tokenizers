"""Offline pipeline: stream a corpus from common.data.corpora's shared registry
(the SAME registry common.data.cli_data.load_groups uses for tokenizer training --
no separate pretraining-only source list), tokenize every document with a
chosen systems/ checkpoint (via pretraining.tokenizer_adapter.TokenizerAdapter),
and pack the resulting token ids into fixed-size binary shards on disk -- the
standard approach real pretraining pipelines use (nanoGPT/OLMo-style):
tokenize once, train many times, reading flat memory-mapped files (see
shard_dataset.py) instead of re-streaming+re-tokenizing on every run, which
would be both slower and non-reproducible run to run (HF streaming order and
network conditions vary).

common.data.corpora.stream_groups yields {lang: text} dicts, not bare text rows --
multi-key for the genuinely N-way parallel sources (oldi_seed/flores_dev),
2-key for the bitext sources (smol/ccmatrix/un_pc/europarl/tatoeba_mt, one
language pair per group), single-key for the monolingual ones (glot500/
fineweb_edu/olmo_mix). This
module flattens every group into its constituent (lang, text) documents and
tokenizes each independently, passing that document's OWN language as
encode()'s `lang` hint -- which is what lets MAGNET's per-script boundary
predictor resolve correctly per document when prepping from a genuinely
multilingual source, rather than needing one global --lang flag for an
entire prep run (a real, incidental improvement from sharing one registry
with the parallel-aware tokenizer-training side, not something this module
would have had a reason to build on its own).

Each document's tokens are followed by the tokenizer's own EOS id (see
TokenizerAdapter.eos_id) before the next document starts -- the ONLY marker
of document boundaries; there is no separate boundary index. Shards are
written as they fill (--shard-size tokens each), so this streams through an
arbitrarily large source with roughly constant memory, regardless of how
many tokens are ultimately produced.

LANGUAGES ENCOUNTERED: this is tracked and reported at the CORPUS level
(per-language doc/token counts written to shards_meta.json's own
"lang_counts", optionally logged to wandb), not per training STEP -- the
packed token stream itself carries no per-token language tag (only the
`lang` hint at encode() time, which is discarded once tokenized), and
pretraining.shard_dataset.ShardedTokenDataset samples random windows that
can legitimately straddle two different-language documents, so "the
language of this batch" isn't even a well-defined single value once
packing has happened. The corpus-level realized distribution is the
actionable number for this project's fairness angle anyway -- it's what
tells you whether a --max-tokens/--max-docs cap combined with
common.data.corpora's round-robin interleaving actually produced the balanced
mix you asked for (see common/data/corpora.py's own module docstring for the
Glot500 incident this exact question was raised to catch). Per-language
"lang_counts" also tracks raw UTF-8 "bytes" alongside "tokens" and "docs" --
together these give a real bytes-per-token compression rate
(common.eval.metrics.compression_rate) at actual pretraining-corpus scale,
matching the same per-language diagnostics every systems/ tokenizer already
reports at its own (much smaller) training-time scale, plus a Gini
coefficient (common.eval.metrics.gini_coefficient) over each language's token
share -- the same "how balanced is this data mix" statistic frontier lab
reports disclose via per-SOURCE mixture proportions, adapted here to this
project's own per-LANGUAGE framing.

DEDUPLICATION (--dedup, on by default): every document is checked against
every OTHER document already seen so far in this same prep_dataset call,
via common.data.dedup.Deduplicator -- exact-hash dedup (verbatim repeats) for
every document regardless of length, plus MinHash+LSH near-duplicate
detection (small edits between otherwise-identical documents) for longer
ones. A document flagged as a duplicate is dropped BEFORE tokenization --
never encoded, never packed into a shard. Dropped-document/byte counts are
tracked per language (shards_meta.json's "dropped_duplicates"/
"dropped_duplicates_by_lang") and printed/logged the same way
"lang_counts" is -- see common/data/dedup.py's own module docstring for the
exact mechanism, its real memory-scaling cost, and why near-dup mostly
matters for the longer fineweb_edu/olmo_mix documents rather than short
individual parallel-corpus sentences.

MAX_DOC_BYTES (--max-doc-bytes, default MAX_DOC_BYTES=4096): every document
is truncated to at most this many UTF-8 bytes BEFORE encode() -- a real,
CONFIRMED incident this guards against, not a hypothetical: FantaConfig.
max_seq_length already guarded manta.model's dense (B,H,T,T) attention
during TRAINING, but this module calls a system's encode()/induce_spans
directly on whatever raw document text a source yields, with no cap of its
own -- an unusually long real Glot500 document reached that exact same
dense attention during TOKENIZATION and OOM'd a real cluster GPU
(confirmed via the job's own error log). truncate_to_max_bytes now lives
in common/bytes_utils.py (moved there from systems/fanta/train.py once
this module hit the identical failure mode from the identical underlying
cause) and is applied here to EVERY document regardless of system --
harmless for bpe/superbpe (no length-driven memory cost at all), essential
for the five neural systems. Truncated-document counts are tracked per
language (shards_meta.json's "num_truncated_docs"/"num_truncated_by_lang")
and printed/logged the same way dedup's own drop counts are -- never
silently absent from the run's own report.

PERFORMANCE / ENCODE_BATCH_SIZE (--encode-batch-size, default
ENCODE_BATCH_SIZE=32): documents are now grouped into batches of this size
and encoded via TokenizerAdapter.encode_batch, not one document per
adapter.encode() call -- a real, CONFIRMED bottleneck this fixes, not a
speculative optimization: a real cluster run of this module against FANTA
measured ~56.8k tok/s against bpe/superbpe's ~74-83k tok/s DESPITE running
on an A100 GPU bpe/superbpe never even use, because manta.segment's
induce_boundaries_batch (which the old one-document-per-call code path
already routed through, just always with a batch size of exactly 1) pays
full GPU kernel-launch/sync overhead per document for a tiny amount of
actual compute. See pretraining/tokenizer_adapter.py's own
_build_induce_batch_fn for which systems get a REAL batched speedup today
(manta/fanta) versus which still fall back to a correct-but-unsped-up
per-item loop (bpe/superbpe, magnet, flexitokens, fairtok -- stated
plainly, not silently pretended to be covered).

MEMORY CAVEAT, a real interaction worth understanding before raising
--encode-batch-size casually: every sequence in one batch gets padded to
that batch's OWN longest member, so a batch's memory cost scales with
(that batch's max length)^2 x batch_size, not just max length^2 -- see
systems/manta/segment.py's own induce_spans_batch docstring. --max-doc-bytes
(above) bounds the worst case per document, but a larger --encode-batch-size
still makes that worst case correspondingly larger; the two settings
interact and should be tuned together, not independently, especially for
the neural systems this batching actually speeds up.

BUCKETING (--bucket-pool-multiplier, default BUCKET_POOL_MULTIPLIER=8): the
padding cost above isn't just a memory risk, it's a real THROUGHPUT
regression the first version of ENCODE_BATCH_SIZE actually shipped with
and hit on a real cluster run -- forming batches straight from stream
order (as they arrive) means a batch can mix very-short and very-long
documents, and the whole batch then pays the long member's O(T^2) cost.
This is WORSE than it sounds for common.data.corpora's monolingual sources
specifically: glot500 "all" round-robins across ~411 different language
configs (see common/data/corpora.py), so CONSECUTIVE stream documents come from
alternating sources with very different typical lengths -- almost
maximizing the length heterogeneity any small window of consecutive
documents sees. Confirmed empirically (not just reasoned about): a
CPU microbenchmark with alternating short/long synthetic documents showed
stream-order batching taking ~2.75x longer than the same documents grouped
by length first, and the effect gets WORSE as documents get longer relative
to each other -- consistent with the real run's throughput collapsing to
roughly 1/1000th of its pre-batching rate.

The fix: documents accumulate into a POOL of encode_batch_size *
bucket_pool_multiplier before any encoding happens, the pool is sorted by
byte length, then sliced into encode_batch_size-sized chunks and encoded in
that order -- each chunk now contains similarly-sized documents, so padding
waste stays small. This REORDERS documents relative to the original corpus
stream (a document's position in the packed shard stream no longer matches
its position in the source stream) -- harmless for pretraining.
shard_dataset.ShardedTokenDataset, which samples random windows regardless
of corpus order anyway, but stated plainly as a real behavior change, not
hidden. max_tokens/max_docs are checked at POOL granularity now (every
encode_batch_size * bucket_pool_multiplier documents), a coarser version of
the same batch-granularity tradeoff ENCODE_BATCH_SIZE already introduced.
bucket_pool_multiplier=1 degrades to plain (unbucketed) batching -- useful
for reproducing the original bug directly, not recommended for real runs.

RESUME (--prep-checkpoint-path, default: "<output-dir>/prep_checkpoint.json"
-- named distinctly from --checkpoint, which is the SYSTEM's own tokenizer
checkpoint TokenizerAdapter.load reads, an unrelated file):
a real cluster run at 30B-token/hundreds-of-millions-of-document Glot500
scale can exceed a single SLURM job's time limit -- unlike pretraining.
train's --resume-from (which this mirrors in spirit), there was previously
no way to continue a prep_dataset call that got killed mid-run other than
starting over from token 0.

Checkpointed once per bucketing pool (see BUCKETING docstring section)
that happens to cross a shard boundary -- roughly once per shard on a real
run, not once per pool (encode_batch_size*bucket_pool_multiplier documents
would be far too frequent at production scale). NEVER checkpointed
mid-pool: bucketing processes a pool's documents in LENGTH-SORTED order,
not the order they were pulled from the stream, so stream_docs_consumed
(how many (lang, text) pairs have been pulled from the stream so far,
REGARDLESS of whether each was kept, deduped, or truncated -- the exact
count the skip-forward logic needs) only correctly corresponds to
"everything up to here is fully, durably accounted for" once an ENTIRE
pool has finished processing, not at some arbitrary point inside it.

The checkpoint captures total_tokens, num_docs, shard_idx, shard_files,
lang_counts, dedup/truncation counters, stream_docs_consumed, AND
buffer_pos plus the buffer's own unflushed contents (written to a sibling
"<prep_checkpoint_path>.buffer.bin" file) -- a shard boundary doesn't
necessarily land exactly at a pool boundary, so rather than forcing an
early, undersized shard flush just to keep buffer_pos always 0 at
checkpoint time (which would fragment a real run into far more, far
smaller shard files than shard_size intends), whatever's sitting in the
buffer gets persisted and restored as-is. Both files written atomically
(temp file + os.replace, buffer file first) so a crash mid-write never
leaves a checkpoint referencing buffer content that isn't actually on disk.

If prep_checkpoint_path exists when prep_dataset starts, it's loaded
automatically and the stream is fast-forwarded (each (lang, text) pair
discarded, uncounted against dedup/encode cost, until stream_docs_consumed
is reached) before normal processing resumes -- no separate --resume flag
needed, just rerun the same command against the same --output-dir. Deleted
(both files) once shards_meta.json is written (a finished run is no longer
resumable, and leaving it would make a later identical invocation waste
time fast-forwarding through the entire stream for nothing).

Two deliberate, minor tradeoffs, not oversights:
  - the Deduplicator itself is NOT persisted (no serialized Bloom
    filter/LSH state) -- resuming starts a FRESH deduplicator after
    fast-forwarding past whatever was already shipped, so a duplicate that
    straddles the resume boundary (one copy before, one after) could
    theoretically slip through where an uninterrupted run would have
    caught it. Serializing the Bloom filter's full bit array on every
    shard flush would cost real I/O time proportional to its size for
    negligible real-world benefit here.
  - fast-forwarding assumes stream_groups yields the same documents in the
    same order across separate process invocations -- already a known
    caveat of this module's own "tokenize once" design (see this
    docstring's own opening paragraph: "HF streaming order and network
    conditions vary"), not a new assumption introduced by resume. If order
    drifts, a resumed run may skip past slightly different documents than
    it otherwise would have, or duplicate a few -- not a correctness
    hazard for pretraining (shard_dataset.py samples random windows
    regardless of corpus order anyway), just an accepted imprecision.
"""

import argparse
import json
import os
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

SHARD_SIZE = 100_000_000  # tokens per shard file -- ~200MB at uint16,
# ~400MB at uint32; large enough that per-shard file overhead is negligible,
# small enough that one shard is a reasonable unit of both disk I/O and
# resumability (a partially-written run's already-flushed shards are still
# individually valid).

MAX_DOC_BYTES = 4096  # default --max-doc-bytes -- see prep_dataset's own
# docstring MAX_DOC_BYTES section for the real incident this guards
# against. Deliberately NOT 0/None-disabled-by-default (unlike
# FantaConfig.max_seq_length's own convention, which IS "0 disables"):
# that exact "unbounded by default" pattern has now caused two separate
# real OOM crashes in this project (FANTA's training loop, then this
# module's own tokenization loop) from the SAME underlying cause -- an
# unusually long document reaching a neural system's dense (B,H,T,T)
# attention uncapped. A conservative non-zero default here is a judgment
# call informed directly by that history, not a mechanical copy of the
# training-side field's own default.

ENCODE_BATCH_SIZE = 32  # default --encode-batch-size -- see this module's
# own PERFORMANCE / ENCODE_BATCH_SIZE docstring section for the real
# bottleneck this fixes and the MEMORY CAVEAT section for how this
# interacts with MAX_DOC_BYTES.

BUCKET_POOL_MULTIPLIER = 8  # default --bucket-pool-multiplier -- see this
# module's own BUCKETING docstring section for the real throughput
# regression (not just a memory risk) this fixes.


def _dtype_for_vocab(vocab_size):
    return "uint16" if vocab_size <= 65536 else "uint32"


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
):
    """dataset_name: one of common.data.corpora.ALL_SOURCES. langs: language
    codes for the language-selectable sources (synthetic/oldi_seed/
    flores_dev/glot500 -- all default to "all", every language natively
    offered, unless overridden; bible_nlp takes an arbitrary language subset
    here too, with no built-in default panel); ignored for fineweb_edu/
    olmo_mix and every BITEXT_SOURCES entry (smol/ccmatrix/un_pc/europarl/
    tatoeba_mt), which use `dataset_config` instead -- see common.data.
    corpora.stream_groups. max_tokens/max_docs: stop
    once either is reached (None disables that particular cap) -- both None
    together streams until the source itself is exhausted, only realistic
    for a genuinely bounded request (e.g. one small Glot500 language, or any
    of the parallel sources, which are all comparatively small and finite
    already). dedup/dedup_*: see common.data.dedup.Deduplicator and this
    module's own DEDUPLICATION docstring section -- max_tokens/max_docs are
    checked against KEPT (post-dedup) counts, so a capped run always ends
    up with that many genuinely distinct tokens/documents, not that many
    attempts before dropping duplicates. max_doc_bytes: see this module's
    own MAX_DOC_BYTES docstring section and MAX_DOC_BYTES's own comment --
    a REAL, confirmed OOM (a long glot500 document reaching a neural
    system's dense attention uncapped) this guards against, applied to
    every document BEFORE encode() regardless of system (harmless for
    bpe/superbpe, which have no length-driven memory cost at all). 0 or
    None disables entirely (truncate_to_max_bytes treats both as "off").
    encode_batch_size/bucket_pool_multiplier: see this module's own
    PERFORMANCE / ENCODE_BATCH_SIZE and BUCKETING docstring sections --
    max_tokens/max_docs are checked at POOL granularity (every
    encode_batch_size * bucket_pool_multiplier documents), not per
    document, so a capped run may overshoot the exact target by up to one
    pool's worth of tokens/documents -- a deliberate, minor tradeoff for
    the throughput gain from bucketed batched encoding, not an oversight.
    prep_checkpoint_path: see this module's own RESUME docstring section --
    defaults to "<output_dir>/prep_checkpoint.json" if None.

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
        stream = stream_groups(dataset_name, langs=langs)

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
        # early (partial, undersized) shard flush just to keep this
        # simpler, which would fragment real runs into far more, far
        # smaller shard files than shard_size intends.
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
        os.replace(tmp_path, prep_checkpoint_path)  # atomic on POSIX -- never
        # leaves a half-written (unreadable-on-resume) checkpoint behind if
        # the process dies mid-write. Written AFTER the buffer file so a
        # crash between the two can never leave a checkpoint that references
        # buffer content that isn't actually on disk yet.

    # (lang, encode_bytes, raw_byte_len) tuples awaiting a bucketed, batched
    # encode() call -- see this module's own PERFORMANCE / ENCODE_BATCH_SIZE
    # and BUCKETING docstring sections. Kept/truncated documents ACCUMULATE
    # here instead of being encoded immediately; dedup/truncation
    # themselves stay per-document (cheap, no GPU cost) since they only
    # need to run once each regardless of batching.
    pending = []
    pool_size = encode_batch_size * max(bucket_pool_multiplier, 1)

    def process_pending(final=False):
        nonlocal buffer_pos, shard_idx, total_tokens, num_docs
        if not pending:
            return
        if not final and len(pending) < pool_size:
            return  # keep accumulating until the pool is full (or the stream ends)
        # Sorted by length so each encode_batch_size-sized chunk below
        # contains similarly-sized documents -- see BUCKETING docstring
        # section for why unsorted chunks pay a real, confirmed throughput
        # penalty, not just a memory risk.
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
        # Checkpointed once here, AFTER the entire pool is done (not inside
        # the loop above) -- see write_prep_checkpoint's own docstring for
        # why mid-pool would be unsafe given bucketing's reordering.
        # Throttled to "only when this pool happened to cross a shard
        # boundary" so checkpoint frequency stays roughly one-per-shard on
        # a real run (~every encode_batch_size*bucket_pool_multiplier
        # documents would otherwise be far too frequent at production
        # scale -- see RESUME docstring section). Skipped on the final
        # call (final=True): prep_dataset finishes and deletes the
        # checkpoint immediately afterward regardless, so writing one here
        # would just be discarded unread.
        if flushed_this_call and not final:
            write_prep_checkpoint()

    pbar = tqdm(
        desc=f"tokenizing {dataset_name}", unit="tok", unit_scale=True,
        initial=total_tokens if resume else 0,
    )
    done = False
    for group in stream:
        for lang, text in group.items():
            if stream_docs_consumed < skip_target:
                # Fast-forwarding past a prior run's already-shipped
                # documents -- see RESUME docstring section. Counted but
                # otherwise untouched: not deduped, not encoded, not
                # counted toward num_docs/total_tokens (those already
                # reflect this document from the prior run's own
                # checkpoint).
                stream_docs_consumed += 1
                continue
            stream_docs_consumed += 1
            if not text:
                continue
            if deduper is not None and deduper.is_duplicate(text):
                # Dropped BEFORE tokenization -- never encoded, never
                # packed. Still counted (doc + byte, per language) so the
                # drop rate is visible, not silently absent from every
                # report this function produces (see module docstring's
                # own "no silent caps" convention elsewhere in this repo).
                raw_bytes = text.encode("utf-8") if isinstance(text, str) else bytes(text)
                dropped_dup_docs += 1
                dropped_dup_bytes += len(raw_bytes)
                dropped_dup_by_lang[lang]["docs"] += 1
                dropped_dup_by_lang[lang]["bytes"] += len(raw_bytes)
                continue
            # Encoded to raw bytes ONCE here (not left as a str for
            # adapter.encode_batch to re-encode internally) -- encode_batch
            # already accepts bytes directly (see tokenizer_adapter._to_bytes),
            # so this doubles as the exact byte count "bytes" below needs,
            # rather than encoding the same text to UTF-8 twice per document.
            raw_bytes = text.encode("utf-8") if isinstance(text, str) else bytes(text)
            # Truncated for ENCODING only -- dedup above already compared
            # the ORIGINAL untruncated text, and lang_counts["bytes"]
            # (in process_pending) still counts the FULL raw_bytes length
            # (a truncated document's own compression_rate is therefore
            # slightly inflated: more bytes counted than were actually
            # tokenized -- accepted since truncation is meant to be rare, a
            # safety net not a routine operation; see MAX_DOC_BYTES's own
            # docstring for the real OOM this guards against, confirmed on
            # a real cluster run).
            encode_bytes, was_truncated = truncate_to_max_bytes(raw_bytes, max_doc_bytes)
            if was_truncated:
                num_truncated_docs += 1
                num_truncated_by_lang[lang] += 1
            pending.append((lang, encode_bytes, len(raw_bytes)))
            process_pending()  # no-op unless the pool has reached pool_size

            if (max_tokens and total_tokens >= max_tokens) or (max_docs and num_docs >= max_docs):
                done = True
                break
        if done:
            break
    process_pending(final=True)  # flush any partial pool smaller than pool_size
    flush()
    pbar.close()

    assert sum(c["tokens"] for c in lang_counts.values()) == total_tokens, (
        "lang_counts token sum diverged from total_tokens -- a real bug in the "
        "accounting above, not just a cosmetic mismatch"
    )

    # Gini coefficient over each language's REALIZED token share -- 0 means
    # every language got an equal number of tokens, closer to 1 means a
    # small number of languages dominate the corpus. Same function
    # (common.eval.metrics.gini_coefficient) every systems/ tokenizer's own
    # per-language fairness reporting already uses, applied here to the
    # actual pretraining corpus rather than a tokenizer's harvested
    # vocabulary -- a single top-line number for how balanced this specific
    # run's training data was, the same kind of "data mix" statistic
    # frontier lab reports disclose (their per-source mixture proportions),
    # adapted to this project's own per-LANGUAGE framing.
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
        # JUDGMENT CALL: "tokens" includes each document's trailing eos_id
        # (see lang_counts's own comment above), so this compression rate is
        # very slightly lower than the tokenizer's own true rate -- one
        # extra token per document with no corresponding bytes, a bias that
        # shrinks as documents get longer. Not corrected for, since the
        # effect is tiny relative to real document lengths and every
        # language is biased the same direction, not selectively.
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
        "lang_counts": dict(lang_counts),  # realized corpus makeup -- see
        # module docstring's LANGUAGES ENCOUNTERED section; not the same as
        # the "langs" field above, which is just what was REQUESTED.
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
        # A finished run is no longer resumable -- see RESUME docstring
        # section for why a stale checkpoint left behind would make a later
        # identical invocation waste time fast-forwarding through the
        # entire stream for nothing.
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
        help="comma-separated language codes -- defaults to 'all' (every language natively "
        "offered) for oldi_seed/flores_dev/glot500 if omitted; an arbitrary subset for "
        "bible_nlp, no default panel) -- ignored for fineweb_edu/olmo_mix and every "
        "BITEXT_SOURCES entry (smol/ccmatrix/un_pc/europarl/tatoeba_mt), which use "
        "--dataset-config instead; see common.data.corpora",
    )
    parser.add_argument(
        "--dataset-config",
        type=str,
        default=None,
        help=f"HF config name for --dataset fineweb_edu (choices: {FINEWEB_EDU_CONFIGS}) or "
        f"olmo_mix (choices: {OLMO_MIX_CONFIGS}); a native pair name or 'all' for ccmatrix/"
        f"un_pc/europarl (see common.data.corpora.list_bitext_configs); a '{{split}}/{{pair-or-"
        f"all}}' string (e.g. 'test/deu-eng') or bare 'all' for tatoeba_mt (see "
        f"common.data.corpora.list_tatoeba_mt_pairs)",
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
        help="drop exact/near-duplicate documents before tokenizing (see common.data.dedup.Deduplicator "
        "and this module's own DEDUPLICATION docstring section); --no-dedup disables entirely",
    )
    parser.add_argument("--dedup-near-threshold", type=float, default=0.8, help="MinHash/LSH Jaccard similarity threshold above which a document counts as a near-duplicate")
    parser.add_argument("--dedup-num-perm", type=int, default=128, help="MinHash permutation count -- higher is a more accurate Jaccard estimate at more memory/CPU per document")
    parser.add_argument("--dedup-shingle-size", type=int, default=13, help="word n-gram shingle length for near-dup comparison")
    parser.add_argument("--dedup-min-words-for-near-dup", type=int, default=50, help="documents shorter than this only get exact-dup checked, not MinHash near-dup (see common/data/dedup.py's own docstring for why)")
    parser.add_argument(
        "--max-doc-bytes", type=int, default=MAX_DOC_BYTES,
        help="truncate each document to at most this many UTF-8 bytes before encoding -- guards "
        "against a real, confirmed OOM (a long document reaching a neural system's dense "
        "attention uncapped, see this module's own MAX_DOC_BYTES docstring section); pass 0 to disable",
    )
    parser.add_argument(
        "--encode-batch-size", type=int, default=ENCODE_BATCH_SIZE,
        help="group this many documents into one encode_batch() call -- a real, confirmed "
        "throughput fix for the neural systems (see this module's own PERFORMANCE / "
        "ENCODE_BATCH_SIZE docstring section); interacts with --max-doc-bytes, see the "
        "MEMORY CAVEAT section before raising this",
    )
    parser.add_argument(
        "--bucket-pool-multiplier", type=int, default=BUCKET_POOL_MULTIPLIER,
        help="sort a pool of encode-batch-size * this-many documents by length before batching -- "
        "a real, confirmed fix for a padding-waste throughput regression (see this module's own "
        "BUCKETING docstring section), not just a memory optimization; 1 disables bucketing",
    )
    parser.add_argument(
        "--prep-checkpoint-path", type=str, default=None,
        help="see this module's own RESUME docstring section -- defaults to "
        "'<output-dir>/prep_checkpoint.json' if omitted. Rerunning the same command against the "
        "same --output-dir after a mid-run interruption resumes automatically; no separate flag "
        "needed to trigger it",
    )
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument(
        "--wandb-project", type=str, default="pretraining",
        help="SAME default project as pretraining.train/cli_eval/cli_generate's own "
        "wandb_project -- this run logs job_type='data_prep', filterable apart in the wandb UI",
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
