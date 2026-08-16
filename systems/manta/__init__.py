"""MANTa-style gradient-based neural tokenizer baseline (see manta.model for
architecture/citations). Sibling of fairtok/: imports from it (data loaders,
metrics, vocab, byte<->tensor helpers) but never the reverse, and never edits
fairtok/.
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
