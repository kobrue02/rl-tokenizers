"""Checkpoint load for a trained ParityBPEModel -- deliberately NOT torch.save/
load: a `tokenizers.Tokenizer` wraps a Rust object with its own native JSON
serialization (.save()/.from_file()), the correct way to persist it -- same
convention as bpe/inference.py (this project's other `tokenizers`-backed
baseline). No separate "config" needed either -- Tokenizer.from_file()
reconstructs vocabulary, merges, and pre-tokenizer settings directly.
"""

from tokenizers import Tokenizer

from .model import ParityBPEModel


def load_checkpoint(path, device="cpu"):
    # device accepted (and ignored) only so evaluate.py can call
    # load_checkpoint(path, device=args.device) identically across tokenizers.
    return ParityBPEModel(Tokenizer.from_file(path))
