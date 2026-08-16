"""Checkpoint load for a trained SuperBPEModel -- mirrors every other
tokenizer's inference.py naming (load_checkpoint), though there's no
nn.Module/state_dict here: SuperBPETrainer.train() saves {"config": ...,
"merges": ..., "id_to_bytes": ..., "num_stage1_merges": ...} via torch.save
(used purely as a convenient object pickler, no tensors involved).
"""

import torch

from .model import SuperBPEModel


def load_checkpoint(path, device="cpu"):
    # device accepted (and ignored) only so evaluate.py can call
    # load_checkpoint(path, device=args.device) identically across tokenizers.
    ckpt = torch.load(path, map_location=device, weights_only=False)
    return SuperBPEModel(
        merges=ckpt["merges"],
        id_to_bytes=ckpt["id_to_bytes"],
        num_stage1_merges=ckpt["num_stage1_merges"],
    )
