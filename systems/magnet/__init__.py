"""MAGNET-style neural tokenizer baseline (scaled down) -- a sibling package to
fairtok/, reusing its data loading, metrics, and vocabulary-extraction utilities
unchanged, but with its own fully differentiable (no-REINFORCE) segmentation
model. See magnet/model.py's module docstring for the architecture and its
explicit list of simplifications vs. Ahia et al.'s MAGNET (arxiv.org/abs/2407.08818).
"""

from .model import BoundaryPredictor, MagnetModel, TransformerBlock
from .segment import induce_boundaries, induce_spans
from .train import MagnetConfig, MagnetTrainer, lang_to_script, run_smoke_test

__all__ = [
    "MagnetModel",
    "BoundaryPredictor",
    "TransformerBlock",
    "induce_boundaries",
    "induce_spans",
    "MagnetConfig",
    "MagnetTrainer",
    "lang_to_script",
    "run_smoke_test",
]
