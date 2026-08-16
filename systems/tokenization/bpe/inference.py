"""Checkpoint load for a trained BPEModel -- deliberately NOT torch.save/load
like other tokenizer packages: a `tokenizers.Tokenizer` wraps a Rust object
with its own native JSON serialization (.save()/.from_file()), the correct
way to persist it. No separate "config" needed either (unlike e.g.
manta/inference.py's saved dim/window/... fields) -- Tokenizer.from_file()
reconstructs vocabulary, merges, and pre-tokenizer settings directly.
"""

from tokenizers import Tokenizer

from .model import BPEModel


def load_checkpoint(path, device="cpu"):
    # device accepted (and ignored) only so evaluate.py can call
    # load_checkpoint(path, device=args.device) identically across tokenizers.
    return BPEModel(Tokenizer.from_file(path))
