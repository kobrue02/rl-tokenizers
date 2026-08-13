"""Memory-mapped reader for packed token shards written by data_prep.py.

Shards are raw binary files of fixed-width unsigned integers (uint16 if the
tokenizer's vocab_size + 1, for the reserved EOS id, fits under 65536, else
uint32 -- see data_prep.py's _dtype_for_vocab), one flat stream of token ids
per file. Document boundaries are marked by the tokenizer's own EOS id
directly in that stream -- there is no separate boundary index to keep in
sync. A small sidecar shards_meta.json records the exact dtype/vocab_size/
eos_id/system/checkpoint used to build them, so this module never assumes or
re-derives that -- see data_prep.py for what writes it.
"""

import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset


def load_shard_meta(shard_dir):
    with open(os.path.join(shard_dir, "shards_meta.json"), "r") as f:
        return json.load(f)


class ShardedTokenDataset(Dataset):
    """One "epoch" here is just `num_samples` random windows, not a real
    pass over the corpus -- a streamed pretraining corpus this large has no
    natural epoch boundary at all (see pretraining/train.py, which steps by
    a token/step budget, not epochs).

    Sampling is (seed, idx)-seeded, not drawn from any RNG object stored on
    self: a Dataset instance gets COPIED into every DataLoader worker
    process under num_workers > 0, so RNG state living on self would either
    duplicate the same draws across workers or need a worker_init_fn to
    reseed correctly. Deriving each sample's randomness purely from
    (self.seed, idx) sidesteps that entirely -- no shared mutable state,
    same sample for the same idx regardless of which worker serves it,
    reproducible given the same seed."""

    def __init__(self, shard_dir, seq_len, num_samples, seed=0, index_offset=0):
        """index_offset: shifts every __getitem__(idx) to actually draw
        (seed, idx + index_offset) -- lets pretraining.train.train's own
        --resume-from path continue the SAME deterministic (seed, idx)
        sequence from where a previous run left off (idx=0..index_offset-1
        already consumed), instead of a fresh dataset instance restarting
        at idx=0 and silently replaying exactly the samples the original
        run already trained on early -- confirmed to actually happen this
        way before this parameter existed (train() built a brand new
        sequential-order DataLoader on every process launch, with no
        memory of how many samples a resumed run had already drawn)."""
        self.meta = load_shard_meta(shard_dir)
        self.seq_len = seq_len
        self.num_samples = num_samples
        self.seed = seed
        self.index_offset = index_offset
        dtype = np.uint16 if self.meta["dtype"] == "uint16" else np.uint32
        shards = [
            np.memmap(os.path.join(shard_dir, name), dtype=dtype, mode="r")
            for name in self.meta["shard_files"]
        ]
        # A shard shorter than seq_len+1 tokens can never serve a full
        # window -- skip it rather than let the offset sampler below divide
        # by a non-positive range on one unlucky trailing partial shard.
        self.shards = [s for s in shards if len(s) > seq_len]
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
