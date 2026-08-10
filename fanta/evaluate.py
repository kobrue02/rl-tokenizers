"""Held-out evaluation for a trained FANTA checkpoint -- mirrors fairtok/evaluate.py's
shape; see that module's docstring for the BOUQuET-as-held-out-set rationale.
FANTA's induce_spans is identical to MANTa's (see fanta/segment.py) -- language-
agnostic at inference time, so no extra per-language argument is needed here.
"""

import argparse

from common.data import make_synthetic_parallel_groups
from common.eval_common import evaluate_on_groups, report_eval
from common.oldi_data import load_bouquet_dev
from common.stability import sequences_by_lang_from_groups

from .inference import load_checkpoint
from .segment import induce_spans


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained FANTA checkpoint on held-out data."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="path to a FANTA checkpoint (--output-dir at training time)",
    )
    parser.add_argument(
        "--eval-data-source",
        choices=["bouquet", "synthetic"],
        default="bouquet",
        help="'bouquet' (default): BOUQuET dev, held out from every training source; "
        "'synthetic': the placeholder corpus, for a quick sanity check with no network access",
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
        return make_synthetic_parallel_groups(args.num_groups or 40)
    groups = load_bouquet_dev()
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
        lang: (lambda raw, m=model, d=args.device: induce_spans(m, raw, d))
        for lang in sequences_by_lang
    }
    results = evaluate_on_groups(induce_fn_by_lang, eval_groups)
    report_eval(results, label="fanta")
    return results


if __name__ == "__main__":
    main()
