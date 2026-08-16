"""FANTA reuses MANTa's architecture unchanged (fanta/model.py) -- only the
loss differs -- so boundary/span induction is identical too. Re-exports
manta.segment unmodified so fanta.cli/fanta.evaluate can import from
`.segment` like every other tokenizer package.
"""

from systems.tokenization.manta.segment import (  # noqa: F401
    boundaries_from_assignment,
    induce_boundaries,
    induce_boundaries_batch,
    induce_spans,
    induce_spans_batch,
)
