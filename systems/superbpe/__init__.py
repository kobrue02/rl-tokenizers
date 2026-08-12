"""SuperBPE baseline: a from-scratch reimplementation of the two-stage
byte-level BPE algorithm described in Liu et al. 2025, "SuperBPE: Space
Travel for Language Models" (COLM 2025) -- see superbpe.model's module
docstring for the algorithm, the official code release it's attributed to,
and why that release's own training code (a custom Rust fork of
huggingface/tokenizers) isn't reused directly.

Unlike every other tokenizer in this repo, SuperBPE has no neural model, no
gradient descent, and no fairness mechanism of its own -- it's a classical
BPE variant, included specifically to test whether cross-lingual fairness
properties differ by tokenizer FAMILY (learned boundary predictor vs.
classical greedy merge counting), not just by which mechanism is used
within this project's own neural family.
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
