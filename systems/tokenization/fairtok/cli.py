"""Command-line entry point. Every GRPOConfig field becomes a `--flag`, generated
from the dataclass itself so this can't drift out of sync with fairtok.train.GRPOConfig.
"""

import argparse

from common.bytes_utils import bytes_to_tensor
from common.config_file import parse_args_with_config
from common.data.cli_data import add_data_source_args, load_bouquet_dev_for_training, load_groups
from common.eval.reporting import fertility_by_lang, report_collapse, report_fertility, report_stability
from common.eval.stability import sequences_by_lang_from_groups, stability_by_lang
from common.vocab import report_and_save_vocab
from systems.tokenization.cli_common import add_dataclass_fields, add_vocab_output_args, config_from_args

from .policy import segment_bytes
from .train import GRPOConfig, GRPOTrainer

# Clarifying text for fields whose semantics aren't obvious from the
# auto-generated "(GRPOConfig.field, default: X)" help alone.
_HELP_OVERRIDES = {
    "max_steps": "0 derives the step count from num_train_epochs; set explicitly to override "
    "with a raw step count instead (bypasses epoch semantics entirely)",
    "num_train_epochs": "1 epoch = 1 full shuffled traversal of every loaded group, however many "
    "steps that takes given per_device_train_batch_size -- not a fixed step count",
}

def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Train the fairness-aware byte-boundary policy."
    )
    add_dataclass_fields(parser, GRPOConfig, help_overrides=_HELP_OVERRIDES)
    add_data_source_args(parser)
    add_vocab_output_args(
        parser,
        vocab_prefix="",
        vocab_out_help="where to save the final vocab as a HuggingFace-style {token: id} JSON file; empty string to skip",
        vocab_stats_help="companion file with per-entry frequency and per-language usage breakdown; empty string to skip",
        vocab_preview_help="print this many of the most frequent vocab entries to the terminal; 0 to skip",
    )
    return parser


def main(argv=None):
    args = parse_args_with_config(build_arg_parser(), argv)
    cfg = config_from_args(args, GRPOConfig)
    train_groups = load_groups(args)
    eval_groups = load_bouquet_dev_for_training(args)

    print(f"data_source={args.data_source} groups={len(train_groups)}\n{cfg}\n")
    trainer = GRPOTrainer(cfg, train_groups, eval_dataset=eval_groups)
    policy, token_freq, final_vocab, target_rate = trainer.train()
    report_collapse(token_freq, final_vocab)
    report_fertility(fertility_by_lang(token_freq, train_groups))

    device = next(policy.parameters()).device
    sequences_by_lang = sequences_by_lang_from_groups(train_groups)
    induce_fn_by_lang = {
        lang: (
            lambda raw, p=policy, d=device: segment_bytes(
                p, bytes_to_tensor(raw, d), deterministic=True, device=d
            )
        )
        for lang in sequences_by_lang
    }
    report_stability(
        stability_by_lang(induce_fn_by_lang, sequences_by_lang, seed=cfg.seed)
    )

    report_and_save_vocab(token_freq, cfg.vocab_size, args.vocab_out, args.vocab_stats_out, args.vocab_preview)
    return policy, token_freq, final_vocab, target_rate


if __name__ == "__main__":
    main()
