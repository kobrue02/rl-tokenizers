"""Checkpoint load for a trained BPEModel -- deliberately NOT torch.save/load
like every other tokenizer package's inference.py: a `tokenizers.Tokenizer`
wraps a Rust object with its OWN native, self-describing JSON serialization
(.save()/.from_file()), which is the correct and idiomatic way to persist it
-- pickling it through torch.save would be working against the library, not
with it. Consequently there's no separate "config" to save alongside it
either (unlike e.g. manta/inference.py's saved dim/window/... fields, needed
to reconstruct an nn.Module before loading a state_dict into it):
Tokenizer.from_file() reconstructs the vocabulary, merges, and pre-tokenizer
settings directly from the file, nothing else is needed.
"""

from tokenizers import Tokenizer

from .model import BPEModel


def load_checkpoint(path, device="cpu"):
    # device is accepted (and ignored) only so every tokenizer's evaluate.py
    # can call load_checkpoint(path, device=args.device) identically -- BPE
    # has no tensors to place on a device.
    return BPEModel(Tokenizer.from_file(path))
