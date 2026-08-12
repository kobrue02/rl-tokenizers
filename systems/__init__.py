"""Every tokenizer system compared in this thesis, one subpackage each:
fairtok (the RL-trained fairness-aware boundary policy), magnet/flexitokens/
manta (from-scratch baselines), fanta (MANTa's architecture + a fairness
loss), superbpe/bpe (classical BPE baselines -- see systems.base for the
shared Config/Trainer shape every one of the seven inherits from).

Named `systems/`, not `tokenizers/`, specifically to avoid shadowing the
real PyPI `tokenizers` package (systems.bpe.model imports it directly:
`from tokenizers import Tokenizer`) -- a local `tokenizers/` directory at
the repo root would be found first on sys.path when running scripts from
here, breaking that import.
"""
