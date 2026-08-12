"""Checkpoint load for a trained SuperBPEModel -- mirrors every other
tokenizer's inference.py naming convention (load_checkpoint), even though
there is no nn.Module/state_dict here: superbpe.train.SuperBPETrainer.train()
saves {"config": ..., "merges": ..., "id_to_bytes": ..., "num_stage1_merges":
...} via torch.save (used purely as a convenient arbitrary-object pickler,
not for any tensor it contains -- there are none), in exactly the shape this
loads back.
"""

import torch

from .model import SuperBPEModel


def load_checkpoint(path, device="cpu"):
    # device is accepted (and ignored) only so every tokenizer's evaluate.py
    # can call load_checkpoint(path, device=args.device) identically --
    # SuperBPE has no tensors to place on a device.
    ckpt = torch.load(path, map_location=device, weights_only=False)
    return SuperBPEModel(
        merges=ckpt["merges"],
        id_to_bytes=ckpt["id_to_bytes"],
        num_stage1_merges=ckpt["num_stage1_merges"],
    )
