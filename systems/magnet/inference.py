"""Checkpoint save/load for a trained MagnetModel -- mirrors fairtok.inference's
naming convention.

Unlike FlexiTokens/MANTa, MagnetModel needs one extra piece of state beyond
MagnetConfig: `scripts`, the ISO 15924 script codes its boundary_predictors
ModuleDict was built with (derived from the training corpus's languages, not
a fixed config field) -- must be saved alongside the config to reconstruct
the model.
"""

import dataclasses

import torch

from .model import MagnetModel


def save_checkpoint(model, cfg, scripts, path):
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": dataclasses.asdict(cfg),
            "scripts": list(scripts),
        },
        path,
    )


def load_checkpoint(path, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    cfg = ckpt["config"]
    model = MagnetModel(
        scripts=ckpt["scripts"],
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        n_pre_layers=cfg["n_pre_layers"],
        n_shortened_layers=cfg["n_shortened_layers"],
        n_post_layers=cfg["n_post_layers"],
        boundary_temperature=cfg["boundary_temperature"],
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model
