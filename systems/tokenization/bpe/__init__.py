"""Standard byte-level BPE baseline -- wraps HuggingFace's `tokenizers`
library directly (see bpe.model's docstring for why this one, unlike every
other package here, doesn't reimplement its own algorithm: BPE is a solved,
standard building block, and `tokenizers` is the reference implementation).

Like superbpe/ and manta/, this has no fairness mechanism of its own -- a
fairness-agnostic control from the same classical tokenizer family as
superbpe/ (frequency-count merges, not a learned boundary predictor), but
without superbpe's whitespace-wall-lifting. Together bpe/ and superbpe/
isolate what superword merging changes relative to ordinary BPE.
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
