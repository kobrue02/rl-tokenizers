"""Held-out evaluation for a trained ParityBPE checkpoint -- mirrors
superbpe/evaluate.py's shape. ParityBPE's induce_spans is language-agnostic
at encode time (plain BPE has no notion of language once fitting is done --
only the LEARNING procedure was fairness-aware), same as bpe/superbpe, so no
per-language argument is needed.
"""

from common.eval.cross_tokenizer import run_eval_cli

from .inference import load_checkpoint
from .segment import induce_spans


def build_induce_fn_by_lang(model, sequences_by_lang, args):
    return {lang: (lambda raw, m=model: induce_spans(m, raw)) for lang in sequences_by_lang}


def main(argv=None):
    return run_eval_cli(argv, "parity_bpe", load_checkpoint, build_induce_fn_by_lang)


if __name__ == "__main__":
    main()
