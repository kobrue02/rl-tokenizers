"""FANTA reuses MANTa's architecture completely unchanged (see fanta/model.py's
module docstring) -- only the LOSS used to train it differs. Boundary/span
induction on a trained model is therefore identical too, so this just re-exports
manta.segment unmodified, rather than duplicating it: fanta.cli/fanta.evaluate can
import from `.segment` like every other tokenizer package does, without either of
them needing to know FANTA and MANTa share one induction implementation.
"""

from systems.manta.segment import (  # noqa: F401
    boundaries_from_assignment,
    induce_boundaries,
    induce_boundaries_batch,
    induce_spans,
)
