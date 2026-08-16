"""Held-out evaluation for a trained FlexiTokens checkpoint -- mirrors
fairtok/evaluate.py's shape (see that module for the BOUQuET-as-held-out-set
rationale). FlexiTokens' induce_spans is language-agnostic at inference time
(alpha_L/beta_L bands only shape training, not the forward pass), so this is
the simplest of the evaluate.py modules -- no extra per-language argument
needed, unlike magnet's (script).
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
    return run_eval_cli(argv, "flexitokens", load_checkpoint, build_induce_fn_by_lang)


if __name__ == "__main__":
    main()
