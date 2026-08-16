"""Held-out evaluation for a trained fairtok BytePolicy checkpoint.

Scores a SAVED checkpoint (fairtok.inference.save_checkpoint) against a held-out
dataset -- unlike GRPOTrainer.evaluate, which only scores the live, in-training
policy. Default held-out set is BOUQuET dev (all 259 languages it offers), disjoint
from every --data-source load_groups trains on; evaluate_on_groups skips languages
the checkpoint has no entries for, so no manual language list is needed. Note:
kas/mni/nqo aren't in BOUQuET at all, so a checkpoint trained on those gets no
reported numbers here.

Scoring (Rényi efficiency, Gini, compression rate, fertility) is
common.eval.cross_tokenizer.evaluate_on_groups, shared with magnet/flexitokens/
manta's evaluate.py -- only checkpoint-loading and boundary-inducing are fairtok-specific.
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
