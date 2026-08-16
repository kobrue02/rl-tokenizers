"""Held-out evaluation for a trained MAGNET checkpoint -- mirrors
fairtok/evaluate.py's shape (see that module for the BOUQuET-as-held-out-set
rationale).

MAGNET's induce_spans needs an extra `script` per language (its boundary
predictor is per-SCRIPT, not per-language). BOUQuET's langs="all" mode keys
groups by full lang_Script stem (e.g. "arz_Arab"), which lang_to_script's
LANG_SCRIPT lookup can't resolve -- eval_lang_to_script handles that;
synthetic data uses plain profile names instead, so build_induce_fn_by_lang
picks the resolver from --eval-data-source rather than guessing from string
shape (guessing is unsafe: "high_resource" also has an underscore).
"""

from common.eval.cross_tokenizer import run_eval_cli

from .inference import load_checkpoint
from .segment import induce_spans
from .train import eval_lang_to_script, lang_to_script


def build_induce_fn_by_lang(model, sequences_by_lang, args):
    # Languages whose script this checkpoint never saw have no entry in
    # model.boundary_predictors -- skip rather than error, same policy
    # evaluate_on_groups applies to languages missing from induce_fn_by_lang.
    script_of = lang_to_script if args.eval_data_source == "synthetic" else eval_lang_to_script
    return {
        lang: (
            lambda raw, m=model, s=script_of(lang), d=args.device: induce_spans(m, raw, s, d)
        )
        for lang in sequences_by_lang
        if script_of(lang) in model.boundary_predictors
    }


def main(argv=None):
    return run_eval_cli(argv, "magnet", load_checkpoint, build_induce_fn_by_lang)


if __name__ == "__main__":
    main()
