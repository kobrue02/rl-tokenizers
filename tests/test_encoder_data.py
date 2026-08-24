"""Tests for systems.pretraining.encoder_data: MLMShardedTokenDataset's
RESERVED-shifted windowing and mlm_collate_fn's 80/10/10 masking recipe."""

import numpy as np
import torch

from systems.pretraining.encoder_data import MLMShardedTokenDataset, mlm_collate_fn
from systems.pretraining.encoder_tokenizer import MASK_ID, PAD_ID, RESERVED


def _write_shard(tmp_path, name, vocab_size, length=2000, seed=0):
    """A shard of raw (unshifted) ids in [0, vocab_size), matching what
    data_prep.py actually writes on disk -- MLMShardedTokenDataset is the
    thing responsible for shifting them by RESERVED, not data_prep.py."""
    import json

    rng = np.random.default_rng(seed)
    dtype = np.uint16 if vocab_size <= 65536 else np.uint32
    ids = rng.integers(0, vocab_size, size=length).astype(dtype)
    ids.tofile(tmp_path / name)
    meta = {
        "dataset": "synthetic", "system": "bpe", "checkpoint": "unused",
        "vocab_size": vocab_size, "eos_id": vocab_size - 1,
        "dtype": "uint16" if dtype == np.uint16 else "uint32",
        "total_tokens": length, "num_docs": 1, "shard_files": [name],
    }
    with open(tmp_path / "shards_meta.json", "w") as f:
        json.dump(meta, f)
    return str(tmp_path)


def test_windows_are_shifted_by_reserved_and_never_collide_with_pad_or_mask(tmp_path):
    vocab_size = 300
    shard_dir = _write_shard(tmp_path, "shard_00000.bin", vocab_size)
    dataset = MLMShardedTokenDataset(shard_dir, seq_len=16, num_samples=50, seed=0)

    for i in range(len(dataset)):
        window = dataset[i]
        assert window.shape == (16,)
        # Raw on-disk ids span [0, vocab_size) -- shifted, they must span
        # [RESERVED, vocab_size + RESERVED), strictly above PAD_ID/MASK_ID.
        assert int(window.min()) >= RESERVED
        assert int(window.max()) < vocab_size + RESERVED
        assert not (window == PAD_ID).any()
        assert not (window == MASK_ID).any()


def test_same_index_and_seed_reproduces_the_same_window(tmp_path):
    shard_dir = _write_shard(tmp_path, "shard_00000.bin", vocab_size=300)
    a = MLMShardedTokenDataset(shard_dir, seq_len=16, num_samples=10, seed=7)
    b = MLMShardedTokenDataset(shard_dir, seq_len=16, num_samples=10, seed=7)
    assert torch.equal(a[3], b[3])


def test_mlm_collate_fn_masks_roughly_the_configured_fraction():
    torch.manual_seed(0)
    real_vocab_size = 1000
    batch = [torch.randint(RESERVED, RESERVED + real_vocab_size, (200,)) for _ in range(20)]

    input_ids, labels = mlm_collate_fn(batch, real_vocab_size, mlm_probability=0.15)

    masked_fraction = (labels != -100).float().mean().item()
    assert 0.10 < masked_fraction < 0.20  # 20*200=4000 draws -- generous band, not a tight statistical test


def test_mlm_collate_fn_never_produces_pad_or_mask_as_a_random_replacement():
    torch.manual_seed(0)
    real_vocab_size = 50
    batch = [torch.randint(RESERVED, RESERVED + real_vocab_size, (500,)) for _ in range(30)]

    input_ids, labels = mlm_collate_fn(batch, real_vocab_size, mlm_probability=0.5)

    # Every masked position is either MASK_ID or a real (>= RESERVED) id --
    # PAD_ID must never appear as a substitution (only MASK_ID/real ids are
    # ever written into input_ids by mlm_collate_fn).
    masked_values = input_ids[labels != -100]
    assert ((masked_values == MASK_ID) | (masked_values >= RESERVED)).all()


def test_mlm_collate_fn_unmasked_positions_are_untouched_and_ignored():
    torch.manual_seed(0)
    real_vocab_size = 1000
    original = [torch.randint(RESERVED, RESERVED + real_vocab_size, (200,)) for _ in range(10)]
    batch = [t.clone() for t in original]

    input_ids, labels = mlm_collate_fn(batch, real_vocab_size, mlm_probability=0.15)

    unmasked = labels == -100
    stacked_original = torch.stack(original)
    assert torch.equal(input_ids[unmasked], stacked_original[unmasked])
