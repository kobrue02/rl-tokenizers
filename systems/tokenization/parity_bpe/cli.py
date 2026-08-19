"""Command-line entry point for ParityBPETrainer, mirroring superbpe.cli's
shape: every ParityBPEConfig field becomes a `--flag`. Real-data loading
reuses common.data.cli_data.load_groups, shared by every tokenizer CLI here.
"""

import argparse

from common.config_file import parse_args_with_config
from common.data.cli_data import add_data_source_args, load_bouquet_dev_for_training, load_groups
from common.eval.reporting import fertility_by_lang, report_collapse, report_fertility, report_stability
from common.eval.stability import sequences_by_lang_from_groups, stability_by_lang
from common.vocab import report_and_save_vocab
from systems.tokenization.cli_common import add_dataclass_fields, add_vocab_output_args, config_from_args

from .segment import induce_spans
from .train import ParityBPEConfig, ParityBPETrainer


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Fit a Parity-aware BPE tokenizer (fair-max cross-lingual merge selection)."
    )
    add_dataclass_fields(parser, ParityBPEConfig)
    add_data_source_args(parser)
    parser.add_argument("--seed", type=int, default=0, help="unused by Parity-aware BPE itself "
        "(fitting is deterministic given the corpus); kept so --data-source synthetic's corpus "
        "generation, which does take a seed, has one to forward.")
    add_vocab_output_args(parser, "parity_bpe_")
    return parser


def main(argv=None):
    args = parse_args_with_config(build_arg_parser(), argv)
    cfg = config_from_args(args, ParityBPEConfig)
    train_groups = load_groups(args)
    eval_groups = load_bouquet_dev_for_training(args)

    print(f"data_source={args.data_source} groups={len(train_groups)}\n{cfg}\n")
    trainer = ParityBPETrainer(cfg, train_groups, eval_groups=eval_groups)
    model, token_freq, final_vocab = trainer.train()
    report_collapse(token_freq, final_vocab)
    report_fertility(fertility_by_lang(token_freq, train_groups))

    sequences_by_lang = sequences_by_lang_from_groups(train_groups)
    induce_fn_by_lang = {
        lang: (lambda raw, m=model: induce_spans(m, raw)) for lang in sequences_by_lang
    }
    report_stability(
        stability_by_lang(induce_fn_by_lang, sequences_by_lang, seed=args.seed)
    )

    report_and_save_vocab(token_freq, cfg.vocab_size, args.vocab_out, args.vocab_stats_out, args.vocab_preview)
    return model, token_freq, final_vocab


if __name__ == "__main__":
    main()
