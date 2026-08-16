"""MAGNET-style neural tokenizer baseline (scaled down): reuses fairtok's data
loading, metrics, and vocab-extraction utilities, but with its own fully
differentiable (no-REINFORCE) segmentation model. See model.py for the
architecture and simplifications vs. Ahia et al. (arxiv.org/abs/2407.08818).
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
