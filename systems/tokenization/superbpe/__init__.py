"""SuperBPE baseline: a from-scratch reimplementation of the two-stage
byte-level BPE algorithm from Liu et al. 2025, "SuperBPE: Space Travel for
Language Models" (COLM 2025) -- see superbpe.model's docstring for the
algorithm and why the official release's training code isn't reused.

Unlike every other tokenizer here, SuperBPE has no neural model, gradient
descent, or fairness mechanism -- it's a classical BPE variant, included to
test whether cross-lingual fairness properties differ by tokenizer FAMILY
(learned boundary predictor vs. classical greedy merge counting), not just
by mechanism within this project's own neural family.
"""

from .model import SuperBPEModel, bpe_encode, fit_superbpe, pretokenize
from .segment import induce_spans
from .train import SuperBPEConfig, SuperBPETrainer, run_smoke_test

__all__ = [
    "SuperBPEModel",
    "bpe_encode",
    "fit_superbpe",
    "pretokenize",
    "induce_spans",
    "SuperBPEConfig",
    "SuperBPETrainer",
    "run_smoke_test",
]
