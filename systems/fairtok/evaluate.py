"""Held-out evaluation for a trained fairtok BytePolicy checkpoint.

Scores a SAVED checkpoint (see fairtok.inference.save_checkpoint -- the file
--output-dir points training's cli.py at) against a held-out dataset, as opposed to
fairtok.train.GRPOTrainer.evaluate's periodic in-training eval, which only ever
scores the live, currently-training policy. Held-out by default means BOUQuET dev
(common.data.oldi_data.load_bouquet_dev("all")) -- disjoint from every --data-source
common.data.cli_data.load_groups trains on, and loading EVERY language BOUQuET's
paragraph_level/dev split actually offers (259, not just this project's own
9-language training panel) -- common.eval.cross_tokenizer.evaluate_on_groups already skips
languages a given checkpoint has no entry for, so this scores whatever the
checkpoint covers, out of everything BOUQuET has, with no manual language list
needed. kas/mni/nqo (3 of the 9-language panel) still aren't in BOUQuET at all --
see common.data.oldi_data.load_flores_devtest_fallback for a fallback covering those,
not wired in here.

Scoring itself (Rényi efficiency, Gini, compression rate, fertility) is
common.eval.cross_tokenizer.evaluate_on_groups, shared verbatim with magnet/flexitokens/
manta's own evaluate.py -- only the checkpoint-loading and boundary-inducing steps
below are fairtok-specific.
"""

from common.bytes_utils import bytes_to_tensor
from common.eval.cross_tokenizer import run_eval_cli

from .inference import load_checkpoint
from .policy import segment_bytes


def _load_policy(checkpoint, device):
    return load_checkpoint(checkpoint).to(device).eval()


def build_induce_fn_by_lang(policy, sequences_by_lang, args):
    return {
        lang: (
            lambda raw, p=policy, d=args.device: segment_bytes(
                p, bytes_to_tensor(raw, d), deterministic=True, device=d
            )
        )
        for lang in sequences_by_lang
    }


def main(argv=None):
    return run_eval_cli(
        argv,
        "fairtok",
        _load_policy,
        build_induce_fn_by_lang,
        checkpoint_help="path to a fairtok checkpoint (see fairtok.inference.save_checkpoint / "
        "--output-dir at training time)",
    )


if __name__ == "__main__":
    main()
