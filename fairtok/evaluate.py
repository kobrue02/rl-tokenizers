"""Held-out evaluation for a trained fairtok BytePolicy checkpoint.

Scores a SAVED checkpoint (see fairtok.inference.save_checkpoint -- the file
--output-dir points training's cli.py at) against a held-out dataset, as opposed to
fairtok.train.GRPOTrainer.evaluate's periodic in-training eval, which only ever
scores the live, currently-training policy. Held-out by default means BOUQuET dev
(common.oldi_data.load_bouquet_dev) -- disjoint from every --data-source
common.cli_data.load_groups trains on, covering 6 of the 9-language panel (arz, bam,
ben, eng, lij, spa; kas/mni/nqo aren't in BOUQuET at all -- see
common.oldi_data.load_flores_devtest_fallback for a fallback covering those, not
wired in here per this project's own choice to keep evaluation scoped to BOUQuET's
native coverage rather than mixing in a second held-out source).

Scoring itself (Rényi efficiency, Gini, compression rate, fertility) is
common.eval_common.evaluate_on_groups, shared verbatim with magnet/flexitokens/
manta's own evaluate.py -- only the checkpoint-loading and boundary-inducing steps
below are fairtok-specific.
"""

import argparse

from common.bytes_utils import bytes_to_tensor
from common.data import make_synthetic_parallel_groups
from common.eval_common import evaluate_on_groups, report_eval
from common.oldi_data import load_bouquet_dev
from common.stability import sequences_by_lang_from_groups

from .inference import load_checkpoint
from .policy import segment_bytes


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained fairtok checkpoint on held-out data."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="path to a fairtok checkpoint (see fairtok.inference.save_checkpoint / "
        "--output-dir at training time)",
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
    policy = load_checkpoint(args.checkpoint).to(args.device)
    policy.eval()

    eval_groups = _load_eval_groups(args)
    print(
        f"checkpoint={args.checkpoint} eval_data_source={args.eval_data_source} "
        f"groups={len(eval_groups)}"
    )

    sequences_by_lang = sequences_by_lang_from_groups(eval_groups)
    induce_fn_by_lang = {
        lang: (
            lambda raw, p=policy, d=args.device: segment_bytes(
                p, bytes_to_tensor(raw, d), deterministic=True, device=d
            )
        )
        for lang in sequences_by_lang
    }
    results = evaluate_on_groups(induce_fn_by_lang, eval_groups)
    report_eval(results, label="fairtok")
    return results


if __name__ == "__main__":
    main()
