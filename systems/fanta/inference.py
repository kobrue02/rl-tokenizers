"""FANTA's checkpoints are architecturally identical to MANTa's (same MantaModel,
same config fields for dim/window/num_frontier_layers/etc.) -- only the loss used
to train them differs, and FantaConfig's extra fields (lambda_fair,
group_sample_size) aren't looked up by this loader, so it works unmodified. Re-
exported here so fanta.evaluate can import from `.inference` like every other
tokenizer package does.
"""

from systems.manta.inference import load_checkpoint  # noqa: F401
