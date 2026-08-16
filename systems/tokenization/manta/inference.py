"""Checkpoint load for a trained MantaModel -- mirrors fairtok.inference's
load_checkpoint naming. No save_checkpoint here: manta/train.py already saves
{"state_dict": ..., "config": dataclasses.asdict(cfg)} inline, in the shape
this loads back.
"""

import torch

from .model import MantaModel


def load_checkpoint(path, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    cfg = ckpt["config"]
    model = MantaModel(
        dim=cfg["dim"],
        window=cfg["window"],
        num_frontier_layers=cfg["num_frontier_layers"],
        num_frontier_heads=cfg["num_frontier_heads"],
        block_hidden_size=cfg["block_hidden_size"],
        num_block_layers=cfg["num_block_layers"],
        max_extra_sigma=cfg["max_extra_sigma"],
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model
