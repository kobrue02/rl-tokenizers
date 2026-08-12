"""Standard byte-level BPE baseline -- wraps HuggingFace's `tokenizers`
library directly (see bpe.model's module docstring for why this one, unlike
every other package in this repo, doesn't reimplement its own algorithm: BPE
is a solved, standard building block with no project-specific novelty, and
`tokenizers` is the reference implementation of it).

Like superbpe/ and manta/, this has no fairness mechanism of its own -- a
second fairness-agnostic control, from the SAME classical tokenizer family
superbpe/ is (frequency-count merges, not a learned boundary predictor), but
WITHOUT superbpe's whitespace-wall-lifting mechanism. Together, bpe/ and
superbpe/ isolate exactly what superword merging changes relative to
ordinary BPE, holding the rest of the tokenizer family fixed.
"""

from .model import BPEModel, fit_bpe
from .segment import induce_spans
from .train import BPEConfig, BPETrainer, run_smoke_test

__all__ = [
    "BPEModel",
    "fit_bpe",
    "induce_spans",
    "BPEConfig",
    "BPETrainer",
    "run_smoke_test",
]
