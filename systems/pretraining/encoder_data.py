"""MLM training data: reuses the SAME packed token shards data_prep.py/
shard_dataset.py already build for the decoder (a flat stream of token ids,
document boundaries marked by the tokenizer's own eos_id) -- masking is a
batch-time transform, not a shard-format concern, so nothing about shard
construction needs to change to train an encoder over the identical corpus/
tokenizer a decoder run used.

Unlike ShardedTokenDataset (decoder), a sample here is ONE window of raw
token ids, not a shifted (x, y) pair -- next-token shifting is a decoder-only
concept. Masking (mlm_collate_fn) happens in the DataLoader's collate_fn,
not __getitem__, so every epoch/step re-masks the same underlying window
differently (dynamic masking, matching HF's own DataCollatorForLanguageModeling
convention rather than a fixed mask baked in at dataset-build time).

Every id a shard stores is shifted up by encoder_tokenizer.RESERVED here
(MLMShardedTokenDataset.__getitem__), and mlm_collate_fn's random-replacement
draw stays within that same shifted range -- see encoder_tokenizer's own
docstring for why pad/mask live at small FIXED ids (0/1) rather than
appended after the real vocabulary the way TokenizerAdapter's own eos_id is.
"""

import numpy as np
import torch
from torch.utils.data import Dataset

from .encoder_tokenizer import MASK_ID, RESERVED
from .shard_dataset import load_shard_meta, open_shards


class MLMShardedTokenDataset(Dataset):
    """See ShardedTokenDataset's own docstring (shard_dataset.py) for the
    (seed, idx)-derived sampling rationale and index_offset's resume
    semantics -- identical here, just without the x/y shift."""

    def __init__(self, shard_dir, seq_len, num_samples, seed=0, index_offset=0, shard_files=None):
        self.meta = load_shard_meta(shard_dir)
        self.seq_len = seq_len
        self.num_samples = num_samples
        self.seed = seed
        self.index_offset = index_offset
        self.shards = open_shards(shard_dir, self.meta, seq_len, shard_files)
        if not self.shards:
            raise ValueError(
                f"no shard in {shard_dir} has at least seq_len={seq_len} tokens"
            )

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        rng = np.random.default_rng((self.seed, idx + self.index_offset))
        shard = self.shards[rng.integers(len(self.shards))]
        start = int(rng.integers(0, len(shard) - self.seq_len + 1))
        window = shard[start : start + self.seq_len].astype(np.int64) + RESERVED
        return torch.from_numpy(window.copy())


def mlm_collate_fn(batch, real_vocab_size, mlm_probability=0.15):
    """batch: list of (seq_len,) LongTensors from MLMShardedTokenDataset
    (already shifted by RESERVED). real_vocab_size: the underlying
    TokenizerAdapter's own vocab_size (i.e. EncoderVocab.adapter.vocab_size,
    NOT EncoderVocab.vocab_size) -- shards_meta.json's own "vocab_size" is
    enough; nothing here needs the actual tokenizer model loaded (the
    shards are already tokenized ids).

    Standard BERT/RoBERTa masking recipe -- of the mlm_probability fraction
    of positions selected as masked, 80% are replaced with MASK_ID, 10% with
    a random real token id, 10% left unchanged (still counted as a
    prediction target). HF's own DataCollatorForLanguageModeling.
    torch_mask_tokens implements this identical scheme; reimplemented
    directly here rather than depended on since it expects an HF tokenizer
    object, and this project's tokenizers (bpe/magnet/fanta/etc, wrapped by
    TokenizerAdapter) aren't HF tokenizers.

    Returns (input_ids, labels): labels is -100 (ignore_index) at every
    UNMASKED position, matching transformers' MLM heads' own default
    CrossEntropyLoss(ignore_index=-100) inside model(input_ids=...,
    labels=...) -- only masked positions contribute to the loss."""
    input_ids = torch.stack(batch)
    labels = input_ids.clone()

    masked_indices = torch.bernoulli(torch.full(input_ids.shape, mlm_probability)).bool()
    labels[~masked_indices] = -100

    replace_with_mask = masked_indices & torch.bernoulli(torch.full(input_ids.shape, 0.8)).bool()
    input_ids[replace_with_mask] = MASK_ID

    # Of the masked positions NOT already replaced with MASK_ID, half (i.e.
    # 10% of all masked positions) get a random real token id -- drawn from
    # the RESERVED-shifted real-token range (excludes PAD_ID/MASK_ID:
    # neither is a sensible "random real token" substitute). The remaining
    # 10% fall through unchanged (still a prediction target, via labels above).
    replace_with_random = (
        masked_indices & ~replace_with_mask & torch.bernoulli(torch.full(input_ids.shape, 0.5)).bool()
    )
    random_ids = torch.randint(RESERVED, RESERVED + real_vocab_size, input_ids.shape, dtype=torch.long)
    input_ids[replace_with_random] = random_ids[replace_with_random]

    return input_ids, labels
