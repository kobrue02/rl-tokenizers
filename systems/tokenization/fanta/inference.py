"""FANTA checkpoints are architecturally identical to MANTa's (same
MantaModel, same config fields) -- only the training loss differs, and
FantaConfig's extra fields (lambda_fair, group_sample_size) aren't looked up
by this loader, so it works unmodified. Re-exported so fanta.evaluate can
import from `.inference` like every other tokenizer package.
"""

from systems.tokenization.manta.inference import load_checkpoint  # noqa: F401
