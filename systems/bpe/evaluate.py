"""Held-out evaluation for a trained BPE checkpoint -- mirrors
superbpe/evaluate.py's shape; see that module's docstring for the
BOUQuET-as-held-out-set rationale. BPE's induce_spans is language-agnostic at
encode time (plain BPE has no notion of language at all), same as
flexitokens/manta/superbpe, so no extra per-language argument is needed here
either.
"""

from common.eval.cross_tokenizer import run_eval_cli

from .inference import load_checkpoint
from .segment import induce_spans
from .train import _SMOKE_TEST_GROUPS

_EVAL_DATA_SOURCE_HELP = (
    "'bouquet' (default): BOUQuET DEV, for tuning/exploratory comparisons; "
    "'bouquet_test': BOUQuET TEST, the genuinely held-out split -- reserve for final "
    "reported numbers, not repeated tuning checks; "
    "'synthetic': a small real-text placeholder (NOT common.data.synthetic's byte generator -- "
    "see bpe/train.py's module docstring), for a quick sanity check with no network access"
)


def build_induce_fn_by_lang(model, sequences_by_lang, args):
    return {lang: (lambda raw, m=model: induce_spans(m, raw)) for lang in sequences_by_lang}


def main(argv=None):
    return run_eval_cli(
        argv,
        "bpe",
        load_checkpoint,
        build_induce_fn_by_lang,
        eval_data_source_help=_EVAL_DATA_SOURCE_HELP,
        synthetic_groups=_SMOKE_TEST_GROUPS,
    )


if __name__ == "__main__":
    main()
