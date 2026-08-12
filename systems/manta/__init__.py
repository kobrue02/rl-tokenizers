"""MANTa-style gradient-based neural tokenizer baseline (see manta.model's
module docstring for the architecture, citations, and every deviation from
the original paper). Sibling package to fairtok/ -- imports FROM fairtok
(data loaders, metrics, vocab extraction, byte<->tensor helpers), never the
reverse, and never edits anything under fairtok/.
"""

from .model import MantaModel, MantaOutput, next_byte_loss
from .segment import (
    boundaries_from_assignment,
    induce_boundaries,
    induce_boundaries_batch,
    induce_spans,
)
from .train import MantaConfig, MantaTrainer, run_smoke_test

__all__ = [
    "MantaModel",
    "MantaOutput",
    "next_byte_loss",
    "boundaries_from_assignment",
    "induce_boundaries",
    "induce_boundaries_batch",
    "induce_spans",
    "MantaConfig",
    "MantaTrainer",
    "run_smoke_test",
]
