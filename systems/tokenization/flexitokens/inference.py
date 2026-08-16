"""Checkpoint load for a trained FlexiTokensModel -- mirrors fairtok.inference's
naming convention. No save_checkpoint here: train.py already saves
{"state_dict": ..., "config": ...} inline at the end of
FlexiTokensTrainer.train, in the shape this loads back.
"""

import torch

from .model import FlexiTokensModel


def load_checkpoint(path, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    cfg = ckpt["config"]
    model = FlexiTokensModel(
        d_model=cfg["d_model"],
        nhead=cfg["nhead"],
        num_pre_layers=cfg["num_pre_layers"],
        num_mid_layers=cfg["num_mid_layers"],
        num_post_layers=cfg["num_post_layers"],
        gumbel_temperature=cfg["gumbel_temperature"],
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model
