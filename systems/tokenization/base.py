"""Shared base Config/Trainer every tokenizer system in this repo inherits
from.

Deliberately minimal: captures ONLY what is genuinely identical, in name,
type, and meaning, across all seven systems (fairtok/magnet/flexitokens/
manta/fanta/superbpe/bpe) -- not a forced unification of things that really
differ. Confirmed by direct comparison before writing this: every one of the
seven Config dataclasses already independently declared these same five
fields, verbatim (default STRING/int values differ per system -- e.g.
fairtok's vocab_size defaults to 512, everyone else's to 384 -- which
subclasses override in the normal dataclass-inheritance way, by redeclaring
the field with their own default).

Everything that genuinely differs stays exactly where it already was, in
each system's own train.py: learning_rate/optimizer/batch-size fields for the
five gradient-based systems, nothing of the kind for the two classical BPE
systems; device resolution for the five that use torch tensors, nothing for
the two that don't; num_train_epochs/max_steps for the step-based trainers,
nothing for superbpe/bpe's single-shot fits. Forcing any of THAT into a
shared base would misrepresent a real architectural difference as an
accident of naming -- see e.g. superbpe/train.py's and manta/train.py's own
module docstrings for how deliberately each difference is documented.
"""

import dataclasses
from abc import ABC, abstractmethod


@dataclasses.dataclass
class BaseTokenizerConfig:
    vocab_size: int = 384  # final harvested-vocabulary budget -- every
    # system's own Trainer applies this once, after fitting/training, via
    # common.vocab.top_k_by_frequency (see that module's docstring for why
    # even the naturally-vocab-sized BPE/SuperBPE trainers still go through
    # the same two-stage harvesting step: it keeps every system's final
    # vocabulary produced identically, so they're directly comparable).
    output_dir: str = ""  # "" disables checkpoint saving; else a path each
    # system's own Trainer saves its trained artifact to (the artifact's
    # SHAPE differs a lot -- a torch state_dict for the five neural systems,
    # a merge table for superbpe, a native tokenizers.Tokenizer JSON file for
    # bpe -- see each system's own inference.py for how it loads back).
    use_wandb: bool = False
    wandb_project: str = "tokenizer"  # every subclass overrides this with
    # its own system name (see each system's own train.py).
    run_name: str = ""


class BaseTokenizerTrainer(ABC):
    """Construct with args + train_groups (a list of dicts {lang: text}, the
    shape every data loader in common.data.oldi_data/common.data.synthetic produces) and
    optionally eval_groups (BOUQuET dev, or None to skip held-out checks
    during/after fitting -- see common.data.cli_data.load_bouquet_dev_for_training).
    Call .train(), then read .model / .token_freq / .vocab off the instance --
    every subclass's own train() must set these three attributes and return
    them as a (model, token_freq, vocab) tuple, the convention every system's
    cli.py already relies on.

    Subclasses call super().__init__(args, train_groups, eval_groups) from
    their own __init__ and then add whatever extra setup THEY specifically
    need (e.g. the five neural systems resolve a torch device here; superbpe/
    bpe need nothing further at all) -- this base intentionally does not
    guess at that, since it differs by system (see module docstring).
    """

    def __init__(self, args, train_groups, eval_groups=None):
        self.args = args
        self.train_groups = train_groups
        self.eval_groups = eval_groups
        self.model = None
        self.token_freq = None
        self.vocab = None

    @abstractmethod
    def train(self):
        """Must set self.model/self.token_freq/self.vocab and return them as
        a (model, token_freq, vocab) tuple."""
