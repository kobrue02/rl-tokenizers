"""Held-out evaluation for a trained BPE checkpoint -- mirrors
superbpe/evaluate.py's shape; see that module's docstring for the
BOUQuET-as-held-out-set rationale. BPE's induce_spans is language-agnostic at
encode time (plain BPE has no notion of language at all), same as
flexitokens/manta/superbpe, so no extra per-language argument is needed here
either.
"""

import argparse

from common.eval.cross_tokenizer import evaluate_on_groups, report_eval
from common.data.oldi_data import load_bouquet_dev, load_bouquet_test
from common.eval.stability import sequences_by_lang_from_groups

from .inference import load_checkpoint
from .segment import induce_spans
from .train import _SMOKE_TEST_GROUPS


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained BPE checkpoint on held-out data."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="path to a BPE checkpoint (--output-dir at training time)",
    )
    parser.add_argument(
        "--eval-data-source",
        choices=["bouquet", "bouquet_test", "synthetic"],
        default="bouquet",
        help="'bouquet' (default): BOUQuET DEV, for tuning/exploratory comparisons; "
        "'bouquet_test': BOUQuET TEST, the genuinely held-out split -- reserve for final "
        "reported numbers, not repeated tuning checks; "
        "'synthetic': a small real-text placeholder (NOT common.data.synthetic's byte generator -- "
        "see bpe/train.py's module docstring), for a quick sanity check with no network access",
    )
    parser.add_argument(
        "--num-groups",
        type=int,
        default=None,
        help="cap the number of held-out groups scored; omit for the full set",
    )
    parser.add_argument("--device", type=str, default="cpu")
    return parser


def _load_eval_groups(args):
    if args.eval_data_source == "synthetic":
        groups = _SMOKE_TEST_GROUPS
        return groups[: args.num_groups] if args.num_groups else groups
    loader = load_bouquet_test if args.eval_data_source == "bouquet_test" else load_bouquet_dev
    groups = loader("all")
    if args.num_groups:
        groups = groups[: args.num_groups]
    return groups


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    model = load_checkpoint(args.checkpoint, device=args.device)

    eval_groups = _load_eval_groups(args)
    print(
        f"checkpoint={args.checkpoint} eval_data_source={args.eval_data_source} "
        f"groups={len(eval_groups)}"
    )

    sequences_by_lang = sequences_by_lang_from_groups(eval_groups)
    induce_fn_by_lang = {
        lang: (lambda raw, m=model: induce_spans(m, raw)) for lang in sequences_by_lang
    }
    results = evaluate_on_groups(induce_fn_by_lang, eval_groups)
    report_eval(results, label="bpe")
    return results


if __name__ == "__main__":
    main()
