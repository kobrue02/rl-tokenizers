"""Command-line entry point for MantaTrainer, mirroring fairtok.cli's shape:
every MantaConfig field becomes a `--flag`, generated from the dataclass itself.
Real-data loading is NOT reimplemented here -- common.data.cli_data.load_groups already
does exactly what's needed (--data-source/--langs/--num-groups against the real
OLDI-and-friends corpus or the synthetic placeholder), and is shared verbatim by
every tokenizer's CLI in this repo (fairtok, magnet, flexitokens, manta).
"""

import argparse

from common.config_file import parse_args_with_config
from common.data.cli_data import add_data_source_args, load_bouquet_dev_for_training, load_groups
from common.eval.reporting import fertility_by_lang, report_collapse, report_fertility, report_stability
from common.eval.stability import sequences_by_lang_from_groups, stability_by_lang
from common.vocab import report_and_save_vocab
from systems.cli_common import add_dataclass_fields, add_vocab_output_args, config_from_args

from .segment import induce_spans
from .train import MantaConfig, MantaTrainer


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Train the MANTa baseline tokenizer.")
    add_dataclass_fields(parser, MantaConfig)
    add_data_source_args(parser)
    add_vocab_output_args(parser, "manta_")
    return parser


def main(argv=None):
    args = parse_args_with_config(build_arg_parser(), argv)
    cfg = config_from_args(args, MantaConfig)
    train_groups = load_groups(args)
    eval_groups = load_bouquet_dev_for_training(args)

    print(f"data_source={args.data_source} groups={len(train_groups)}\n{cfg}\n")
    trainer = MantaTrainer(cfg, train_groups, eval_groups=eval_groups)
    model, token_freq, final_vocab = trainer.train()
    report_collapse(token_freq, final_vocab)
    report_fertility(fertility_by_lang(token_freq, train_groups))

    device = next(model.parameters()).device
    sequences_by_lang = sequences_by_lang_from_groups(train_groups)
    induce_fn_by_lang = {
        lang: (lambda raw, m=model, d=device: induce_spans(m, raw, d))
        for lang in sequences_by_lang
    }
    report_stability(
        stability_by_lang(induce_fn_by_lang, sequences_by_lang, seed=cfg.seed)
    )

    report_and_save_vocab(token_freq, cfg.vocab_size, args.vocab_out, args.vocab_stats_out, args.vocab_preview)
    return model, token_freq, final_vocab


if __name__ == "__main__":
    main()
