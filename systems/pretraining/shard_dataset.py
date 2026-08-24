"""Memory-mapped reader for packed token shards written by data_prep.py.

Shards are raw binary files of fixed-width unsigned integers (uint16 if
vocab_size+1 fits under 65536, else uint32 -- see data_prep.py's
_dtype_for_vocab), one flat stream of token ids per file. Document
boundaries are marked by the tokenizer's own EOS id in that stream directly
-- no separate boundary index. A sidecar shards_meta.json records the
dtype/vocab_size/eos_id/system/checkpoint used to build them.
"""

import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset


def load_shard_meta(shard_dir):
    with open(os.path.join(shard_dir, "shards_meta.json"), "r") as f:
        return json.load(f)


def open_shards(shard_dir, meta, min_len, shard_files=None):
    """Opens every shard (or shard_files, an explicit subset of meta's own
    "shard_files") as a read-only memmap, dropping any shorter than min_len
    tokens -- shared by ShardedTokenDataset below (decoder: needs seq_len+1
    tokens per window, for its x/y shift) and encoder_data.MLMShardedTokenDataset
    (encoder: needs exactly seq_len, no shift)."""
    dtype = np.uint16 if meta["dtype"] == "uint16" else np.uint32
    names = shard_files if shard_files is not None else meta["shard_files"]
    shards = [np.memmap(os.path.join(shard_dir, name), dtype=dtype, mode="r") for name in names]
    return [s for s in shards if len(s) >= min_len]


class ShardedTokenDataset(Dataset):
    """One "epoch" is just `num_samples` random windows, not a real pass
    over the corpus -- a streamed corpus this large has no natural epoch
    boundary (pretraining/train.py steps by a token/step budget instead).

    Sampling is seeded from (self.seed, idx), not an RNG object stored on
    self, since a Dataset gets copied into every DataLoader worker process
    under num_workers > 0 -- RNG state on self would duplicate draws across
    workers. Deriving randomness purely from (seed, idx) gives the same
    sample for a given idx regardless of which worker serves it."""

    def __init__(self, shard_dir, seq_len, num_samples, seed=0, index_offset=0, shard_files=None):
        """index_offset: shifts __getitem__(idx) to draw (seed, idx +
        index_offset) so train.py's --resume-from can continue the same
        deterministic sequence from where a previous run left off, instead
        of a fresh dataset restarting at idx=0 and replaying already-trained
        samples. shard_files: explicit subset of shard_dir's own filenames
        to read from, instead of every file shards_meta.json lists -- lets
        a caller (e.g. train.py's train/val split) build two datasets over
        disjoint shards sharing one shard_dir; None (default) uses every
        shard, matching this class's original behavior exactly."""
        self.meta = load_shard_meta(shard_dir)
        self.seq_len = seq_len
        self.num_samples = num_samples
        self.seed = seed
        self.index_offset = index_offset
        # A shard shorter than seq_len+1 tokens can never serve a full
        # window -- open_shards drops it rather than letting the offset
        # sampler below divide by a non-positive range on one unlucky
        # trailing partial shard.
        self.shards = open_shards(shard_dir, self.meta, seq_len + 1, shard_files)
        if not self.shards:
            raise ValueError(
                f"no shard in {shard_dir} has more than seq_len={seq_len} tokens"
            )

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        rng = np.random.default_rng((self.seed, idx + self.index_offset))
        shard = self.shards[rng.integers(len(self.shards))]
        start = int(rng.integers(0, len(shard) - self.seq_len - 1))
        window = shard[start : start + self.seq_len + 1].astype(np.int64)
        x = torch.from_numpy(window[:-1].copy())
        y = torch.from_numpy(window[1:].copy())
        return x, y
