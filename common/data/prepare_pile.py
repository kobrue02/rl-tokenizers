"""One-time local download of EleutherAI/the_pile_deduplicated (~451GB
across 1650 parquet shards, confirmed live via HfApi().dataset_info(...,
files_metadata=True)) into this project's own local disk cache.

Same motivation/lesson as common.data.prepare_glot500: common.data.corpora's
`pile` source used to live-stream every row from the HF Hub, and a real
full-scale (~300B-token, matching EleutherAI's own Pythia-suite training
budget) prep run needs multiple SLURM resubmits -- re-streaming ~451GB from
scratch on every resume would be exactly the glot500 mistake repeated.

UNLIKE glot500 (~411 SEPARATE per-language configs, each needing its own
custom JSONL write and its own per-language completion marker), Pile is one
flat corpus already sharded into 1650 parquet files on the HF Hub -- the
simplest, most robust local cache is downloading those files VERBATIM via
huggingface_hub.snapshot_download rather than re-encoding them into JSONL.
snapshot_download already resumes partial downloads and skips
already-complete files on its own (its own local cache under `local_dir`
tracks this), so no custom per-shard resumability logic is needed here the
way prepare_glot500's per-language ".tmp"-then-atomic-rename dance was.

Usage (run once; ~451GB will take a while over a real network link -- try a
small --allow-patterns smoke test first, e.g. a single shard):

    python -m common.data.prepare_pile --output-dir data/pile --max-workers 4  # smoke test: pass --limit 1
    python -m common.data.prepare_pile --output-dir data/pile

Then --dataset pile (with --dataset-config pointing at the same
--output-dir, if not the default PILE_LOCAL_DIR) reads local parquet
directly instead of live HF streaming -- see
common.data.corpora.stream_groups's own pile branch / _stream_pile_local.
No live fallback once this has run.
"""

import argparse
import os

from huggingface_hub import snapshot_download

from .corpora import PILE_LOCAL_DIR, PILE_REPO


def prepare_pile(output_dir, max_workers=8, limit=None):
    """Downloads every ("data/*.parquet" only -- skips .gitattributes/README/
    etc., the only other files in the repo) parquet shard of PILE_REPO into
    `output_dir`, preserving the repo's own "data/" subdirectory layout (so
    _stream_pile_local's glob, `{output_dir}/data/*.parquet`, finds them).

    limit: download at most this many shards (a quick smoke test via
    allow_patterns on the first N shard filenames -- the repo's own shard
    filenames sort the same way list_repo_files returns them); None
    (default) downloads every shard, the real intended use.

    Returns the local path snapshot_download resolved to (== output_dir).
    """
    os.makedirs(output_dir, exist_ok=True)
    allow_patterns = ["data/*.parquet"]
    if limit:
        from huggingface_hub import HfApi

        files = sorted(
            f for f in HfApi().list_repo_files(PILE_REPO, repo_type="dataset")
            if f.startswith("data/") and f.endswith(".parquet")
        )
        allow_patterns = files[:limit]

    path = snapshot_download(
        repo_id=PILE_REPO,
        repo_type="dataset",
        local_dir=output_dir,
        allow_patterns=allow_patterns,
        max_workers=max_workers,
    )
    data_dir = os.path.join(output_dir, "data")
    n_files = sum(1 for f in os.listdir(data_dir) if f.endswith(".parquet")) if os.path.isdir(data_dir) else 0
    print(f"pile local cache at {path}: {n_files}/1650 parquet shard(s) present")
    return path


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="One-time download of EleutherAI/the_pile_deduplicated into this project's own local parquet cache."
    )
    parser.add_argument("--output-dir", type=str, default=PILE_LOCAL_DIR)
    parser.add_argument(
        "--max-workers", type=int, default=8,
        help="concurrent shard downloads (huggingface_hub.snapshot_download's own parallelism)",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="download at most this many shards -- for a quick smoke test; omit to download "
        "every shard (the real, intended use, but a much longer ~451GB run)",
    )
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    prepare_pile(args.output_dir, max_workers=args.max_workers, limit=args.limit)


if __name__ == "__main__":
    main()
