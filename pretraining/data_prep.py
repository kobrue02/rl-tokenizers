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

import numpy as np
from tqdm.auto import tqdm

from common.config_file import parse_args_with_config
from common.corpora import ALL_SOURCES, FINEWEB_EDU_CONFIGS, MONOLINGUAL_SOURCES, OLMO_MIX_CONFIGS, stream_groups

from .tokenizer_adapter import ALL_SYSTEMS, TokenizerAdapter

SHARD_SIZE = 100_000_000  # tokens per shard file -- ~200MB at uint16,
# ~400MB at uint32; large enough that per-shard file overhead is negligible,
# small enough that one shard is a reasonable unit of both disk I/O and
# resumability (a partially-written run's already-flushed shards are still
# individually valid).


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
    already).

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
            ids = adapter.encode(text, lang=lang)
            ids.append(adapter.eos_id)
            num_docs += 1

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
    }
    with open(os.path.join(output_dir, "shards_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(
        f"wrote {total_tokens:,} tokens ({num_docs:,} documents) across "
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

    prep_dataset(
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
    )


if __name__ == "__main__":
    main()
