"""Held-out evaluation for a trained SuperBPE checkpoint -- mirrors
manta/evaluate.py's shape. SuperBPE's induce_spans is language-agnostic at
encode time (plain BPE has no notion of language), same as flexitokens/manta,
so no per-language argument is needed.
"""

from common.eval.cross_tokenizer import run_eval_cli

from .inference import load_checkpoint
from .segment import induce_spans


def build_induce_fn_by_lang(model, sequences_by_lang, args):
    return {lang: (lambda raw, m=model: induce_spans(m, raw)) for lang in sequences_by_lang}


def main(argv=None):
    return run_eval_cli(argv, "superbpe", load_checkpoint, build_induce_fn_by_lang)


if __name__ == "__main__":
    main()
