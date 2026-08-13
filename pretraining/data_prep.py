"""Offline pipeline: stream a corpus from common.corpora's shared registry
(the SAME registry common.cli_data.load_groups uses for tokenizer training --
no separate pretraining-only source list), tokenize every document with a
chosen systems/ checkpoint (via pretraining.tokenizer_adapter.TokenizerAdapter),
and pack the resulting token ids into fixed-size binary shards on disk -- the
standard approach real pretraining pipelines use (nanoGPT/OLMo-style):
tokenize once, train many times, reading flat memory-mapped files (see
shard_dataset.py) instead of re-streaming+re-tokenizing on every run, which
would be both slower and non-reproducible run to run (HF streaming order and
network conditions vary).

common.corpora.stream_groups yields {lang: text} dicts, not bare text rows --
multi-key for the genuinely parallel sources (oldi_seed/flores_dev/smol),
single-key for the monolingual ones (glot500/fineweb_edu/olmo_mix). This
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
common.corpora's round-robin interleaving actually produced the balanced
mix you asked for (see common/corpora.py's own module docstring for the
Glot500 incident this exact question was raised to catch). Per-language
"lang_counts" also tracks raw UTF-8 "bytes" alongside "tokens" and "docs" --
together these give a real bytes-per-token compression rate
(common.metrics.compression_rate) at actual pretraining-corpus scale,
matching the same per-language diagnostics every systems/ tokenizer already
reports at its own (much smaller) training-time scale, plus a Gini
coefficient (common.metrics.gini_coefficient) over each language's token
share -- the same "how balanced is this data mix" statistic frontier lab
reports disclose via per-SOURCE mixture proportions, adapted here to this
project's own per-LANGUAGE framing.

DEDUPLICATION (--dedup, on by default): every document is checked against
every OTHER document already seen so far in this same prep_dataset call,
via common.dedup.Deduplicator -- exact-hash dedup (verbatim repeats) for
every document regardless of length, plus MinHash+LSH near-duplicate
detection (small edits between otherwise-identical documents) for longer
ones. A document flagged as a duplicate is dropped BEFORE tokenization --
never encoded, never packed into a shard. Dropped-document/byte counts are
tracked per language (shards_meta.json's "dropped_duplicates"/
"dropped_duplicates_by_lang") and printed/logged the same way
"lang_counts" is -- see common/dedup.py's own module docstring for the
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

PERFORMANCE, stated plainly rather than discovered the hard way later (this
project has already hit this exact class of problem once -- see
systems/fanta's max_seq_length field and jobs/evaluate.sh's time-limit fix,
both from an unbatched-per-sequence cost surfacing only at real scale):
encoding here is ONE DOCUMENT AT A TIME, single process. bpe/superbpe
(Rust-backed / a pure-Python but still O(document length) merge apply) are
fast. The five span-family systems call their own model's induce_spans PER
DOCUMENT -- fairtok's specifically loops one byte at a time in Python (see
systems.fairtok.policy.segment_bytes), the others at least batch internally
per call but still one document per Python-level call here. For a genuinely
large prep run with one of the five neural systems, this will be
substantially slower than bpe/superbpe; parallelizing across CPU processes
or batching multiple documents through one model call are the natural next
steps if that becomes a real bottleneck, neither of which is implemented yet
-- flagged here, not hidden.
"""

import argparse
import json
import os
from collections import defaultdict

import numpy as np
from tqdm.auto import tqdm

from common.bytes_utils import truncate_to_max_bytes
from common.config_file import parse_args_with_config
from common.corpora import ALL_SOURCES, FINEWEB_EDU_CONFIGS, MONOLINGUAL_SOURCES, OLMO_MIX_CONFIGS, stream_groups
from common.dedup import Deduplicator
from common.metrics import compression_rate, gini_coefficient

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
):
    """dataset_name: one of common.corpora.ALL_SOURCES. langs: language
    codes for the language-selectable sources (synthetic/oldi_seed/
    flores_dev/smol/glot500 -- "all" is valid for glot500); ignored for
    fineweb_edu/olmo_mix, which use `dataset_config` (an HF config name)
    instead -- see common.corpora.stream_groups. max_tokens/max_docs: stop
    once either is reached (None disables that particular cap) -- both None
    together streams until the source itself is exhausted, only realistic
    for a genuinely bounded request (e.g. one small Glot500 language, or any
    of the parallel sources, which are all comparatively small and finite
    already). dedup/dedup_*: see common.dedup.Deduplicator and this
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

    Returns the shards_meta.json dict this also writes to output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)
    adapter = TokenizerAdapter.load(system, checkpoint_path, vocab_json_path, device=device)
    dtype_name = _dtype_for_vocab(adapter.vocab_size)
    dtype = np.uint16 if dtype_name == "uint16" else np.uint32

    if dataset_name in ("fineweb_edu", "olmo_mix"):
        stream = stream_groups(dataset_name, config=dataset_config)
    else:
        stream = stream_groups(dataset_name, langs=langs)

    shard_files = []
    buffer = np.empty(shard_size, dtype=dtype)
    buffer_pos = 0
    total_tokens = 0
    num_docs = 0
    shard_idx = 0
    # {lang: {"docs": int, "tokens": int, "bytes": int}} -- the REALIZED
    # corpus makeup, not the requested `langs` (see module docstring's own
    # LANGUAGES ENCOUNTERED section for why this is tracked here, at the
    # corpus level, rather than per training step). "tokens" counts each
    # document's ids INCLUDING its trailing eos_id, so summing every
    # language's "tokens" here always equals total_tokens exactly -- a
    # cheap, useful invariant to sanity-check this against (see the assert
    # right after the loop). "bytes" is the RAW UTF-8 byte length before
    # tokenization -- together with "tokens" this gives a real
    # bytes-per-token compression rate (common.metrics.compression_rate) at
    # actual pretraining-corpus scale, not just the much smaller sample a
    # systems/ tokenizer's own training-time smoke test measures against.
    lang_counts = defaultdict(lambda: {"docs": 0, "tokens": 0, "bytes": 0})
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
    dropped_dup_docs = 0
    dropped_dup_bytes = 0
    dropped_dup_by_lang = defaultdict(lambda: {"docs": 0, "bytes": 0})
    num_truncated_docs = 0
    num_truncated_by_lang = defaultdict(int)

    def flush():
        nonlocal buffer_pos, shard_idx
        if buffer_pos == 0:
            return
        name = f"shard_{shard_idx:05d}.bin"
        buffer[:buffer_pos].tofile(os.path.join(output_dir, name))
        shard_files.append(name)
        shard_idx += 1
        buffer_pos = 0

    pbar = tqdm(desc=f"tokenizing {dataset_name}", unit="tok", unit_scale=True)
    done = False
    for group in stream:
        for lang, text in group.items():
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
            # adapter.encode to re-encode internally) -- encode() already
            # accepts bytes directly (see tokenizer_adapter._to_bytes), so
            # this doubles as the exact byte count "bytes" below needs,
            # rather than encoding the same text to UTF-8 twice per document.
            raw_bytes = text.encode("utf-8") if isinstance(text, str) else bytes(text)
            # Truncated for ENCODING only -- dedup above already compared
            # the ORIGINAL untruncated text, and lang_counts["bytes"]
            # below still counts the FULL raw_bytes length (a truncated
            # document's own compression_rate is therefore slightly
            # inflated: more bytes counted than were actually tokenized --
            # accepted since truncation is meant to be rare, a safety net
            # not a routine operation; see MAX_DOC_BYTES's own docstring
            # for the real OOM this guards against, confirmed on a real
            # cluster run).
            encode_bytes, was_truncated = truncate_to_max_bytes(raw_bytes, max_doc_bytes)
            if was_truncated:
                num_truncated_docs += 1
                num_truncated_by_lang[lang] += 1
            ids = adapter.encode(encode_bytes, lang=lang)
            ids.append(adapter.eos_id)
            num_docs += 1
            lang_counts[lang]["docs"] += 1
            lang_counts[lang]["tokens"] += len(ids)
            lang_counts[lang]["bytes"] += len(raw_bytes)

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

            if (max_tokens and total_tokens >= max_tokens) or (max_docs and num_docs >= max_docs):
                done = True
                break
        if done:
            break
    flush()
    pbar.close()

    assert sum(c["tokens"] for c in lang_counts.values()) == total_tokens, (
        "lang_counts token sum diverged from total_tokens -- a real bug in the "
        "accounting above, not just a cosmetic mismatch"
    )

    # Gini coefficient over each language's REALIZED token share -- 0 means
    # every language got an equal number of tokens, closer to 1 means a
    # small number of languages dominate the corpus. Same function
    # (common.metrics.gini_coefficient) every systems/ tokenizer's own
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
    print(
        f"\nwrote {total_tokens:,} tokens ({num_docs:,} documents) across "
        f"{len(shard_files)} shards to {output_dir}"
    )
    return meta


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Tokenize a corpus (common.corpora registry) into packed token shards."
    )
    parser.add_argument("--dataset", choices=ALL_SOURCES, required=True)
    parser.add_argument(
        "--langs",
        type=str,
        default=None,
        help="comma-separated language codes (or 'all' for glot500) -- ignored for "
        "fineweb_edu/olmo_mix, which use --dataset-config instead; see common.corpora",
    )
    parser.add_argument(
        "--dataset-config",
        type=str,
        default=None,
        help=f"HF config name, only meaningful for --dataset fineweb_edu (choices: "
        f"{FINEWEB_EDU_CONFIGS}) or olmo_mix (choices: {OLMO_MIX_CONFIGS})",
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
        help="drop exact/near-duplicate documents before tokenizing (see common.dedup.Deduplicator "
        "and this module's own DEDUPLICATION docstring section); --no-dedup disables entirely",
    )
    parser.add_argument("--dedup-near-threshold", type=float, default=0.8, help="MinHash/LSH Jaccard similarity threshold above which a document counts as a near-duplicate")
    parser.add_argument("--dedup-num-perm", type=int, default=128, help="MinHash permutation count -- higher is a more accurate Jaccard estimate at more memory/CPU per document")
    parser.add_argument("--dedup-shingle-size", type=int, default=13, help="word n-gram shingle length for near-dup comparison")
    parser.add_argument("--dedup-min-words-for-near-dup", type=int, default=50, help="documents shorter than this only get exact-dup checked, not MinHash near-dup (see common/dedup.py's own docstring for why)")
    parser.add_argument(
        "--max-doc-bytes", type=int, default=MAX_DOC_BYTES,
        help="truncate each document to at most this many UTF-8 bytes before encoding -- guards "
        "against a real, confirmed OOM (a long document reaching a neural system's dense "
        "attention uncapped, see this module's own MAX_DOC_BYTES docstring section); pass 0 to disable",
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
    if args.dataset in MONOLINGUAL_SOURCES - {"glot500"} and args.langs:
        print(
            f"warning: --langs is ignored for --dataset {args.dataset} "
            "(single-language source, selected via --dataset-config instead)"
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
