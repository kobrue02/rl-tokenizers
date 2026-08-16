"""The two halves of this thesis's experimental pipeline: systems.tokenization (the
seven trained tokenizers plus baselines, compared against each other and against
frontier tokenizers) and systems.pretraining (an actual language model, pretrained
on top of any of them, to see whether tokenizer-level fairness gains survive
downstream).

Named `systems/`, not `tokenizers/`, specifically to avoid shadowing the real PyPI
`tokenizers` package (systems.tokenization.bpe.model imports it directly) -- a local
`tokenizers/` directory at the repo root would be found first on sys.path when
running scripts from here, breaking that import.
"""
