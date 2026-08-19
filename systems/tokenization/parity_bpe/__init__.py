"""Parity-aware BPE baseline: wraps the OFFICIAL implementation of
Foroutan, Meister, Paul, Niklaus, Ahmadi, Bosselut & Sennrich, "Parity-Aware
Byte-Pair Encoding: Improving Cross-lingual Fairness in Tokenization" (ACL
2026) directly (github.com/swiss-ai/parity-aware-bpe, vendored in .vendor --
see parity_bpe.model's docstring for exactly what's reused vs. adapted, and
why -- per explicit user instruction to reuse the official code wherever
possible, rather than reimplementing the algorithm from scratch).

Like SuperBPE, Parity-aware BPE has no neural model, gradient descent, or
learned mechanism -- it's a classical greedy-merge BPE variant, but unlike
every other classical baseline here (bpe, superbpe), its merge SELECTION
rule is explicitly fairness-driven (fair-max over per-language compression
rate) rather than fairness-agnostic. Included to test whether an EXPLICIT,
externally-published cross-lingual fairness objective at the tokenizer-
learning stage outperforms this project's own from-scratch fairness-aware
neural tokenizers (fairtok/magnet/flexitokens/manta), which bake fairness
into a LEARNED boundary predictor instead of a classical merge-counting rule.
"""

from .model import ParityBPEModel, fit_parity_bpe
from .segment import induce_spans
from .train import ParityBPEConfig, ParityBPETrainer, run_smoke_test

__all__ = [
    "ParityBPEModel",
    "fit_parity_bpe",
    "induce_spans",
    "ParityBPEConfig",
    "ParityBPETrainer",
    "run_smoke_test",
]
