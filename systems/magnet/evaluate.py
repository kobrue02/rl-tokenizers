"""Held-out evaluation for a trained MAGNET checkpoint -- mirrors fairtok/evaluate.py's
shape; see that module's docstring for the BOUQuET-as-held-out-set rationale.

MAGNET's induce_spans needs an extra `script` argument per language (its boundary
predictor is per-SCRIPT, not per-language -- see magnet/segment.py, magnet/train.py's
lang_to_script), which is the one real difference from fairtok/flexitokens/manta's
own evaluate.py here. BOUQuET's langs="all" mode keys groups by full lang_Script
stem (e.g. "arz_Arab"), which lang_to_script's LANG_SCRIPT lookup can't resolve
(only plain codes like "arz") -- eval_lang_to_script handles that; synthetic data
still uses plain (if fake) profile names, so build_induce_fn_by_lang picks the
resolver based on --eval-data-source rather than guessing from string shape (see
eval_lang_to_script's own docstring for why that guess would be unsafe: "high_resource"
also contains an underscore but isn't a real stem).
"""

from common.eval.cross_tokenizer import run_eval_cli

from .inference import load_checkpoint
from .segment import induce_spans
from .train import eval_lang_to_script, lang_to_script


def build_induce_fn_by_lang(model, sequences_by_lang, args):
    # Languages whose SCRIPT this checkpoint never saw during training have no
    # entry in model.boundary_predictors -- skip them rather than erroring, same
    # policy as common.eval.cross_tokenizer.evaluate_on_groups already applies to
    # languages missing from induce_fn_by_lang entirely.
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
