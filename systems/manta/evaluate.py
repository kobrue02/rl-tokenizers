"""Held-out evaluation for a trained MANTa checkpoint -- mirrors
fairtok/evaluate.py's shape (BOUQuET as the held-out set). induce_spans
discretizes the soft assignment matrix via argmax and is language-agnostic
at inference time (like flexitokens), so no per-language argument is needed.
"""

from common.eval.cross_tokenizer import run_eval_cli

from .inference import load_checkpoint
from .segment import induce_spans


def build_induce_fn_by_lang(model, sequences_by_lang, args):
    return {
        lang: (lambda raw, m=model, d=args.device: induce_spans(m, raw, d))
        for lang in sequences_by_lang
    }


def main(argv=None):
    return run_eval_cli(argv, "manta", load_checkpoint, build_induce_fn_by_lang)


if __name__ == "__main__":
    main()
