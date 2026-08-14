"""Command-line entry point for SuperBPETrainer, mirroring manta.cli's shape:
every SuperBPEConfig field becomes a `--flag`, generated from the dataclass
itself. Real-data loading is NOT reimplemented here -- common.data.cli_data.load_groups
already does exactly what's needed, and is shared verbatim by every tokenizer's
CLI in this repo (fairtok, magnet, flexitokens, manta, fanta, superbpe).
"""

import argparse

from common.config_file import parse_args_with_config
from common.data.cli_data import add_data_source_args, load_bouquet_dev_for_training, load_groups
from common.eval.reporting import fertility_by_lang, report_collapse, report_fertility, report_stability
from common.eval.stability import sequences_by_lang_from_groups, stability_by_lang
from common.vocab import report_and_save_vocab
from systems.cli_common import add_dataclass_fields, add_vocab_output_args, config_from_args

from .segment import induce_spans
from .train import SuperBPEConfig, SuperBPETrainer


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Fit a SuperBPE tokenizer (two-stage byte-level BPE)."
    )
    add_dataclass_fields(parser, SuperBPEConfig)
    add_data_source_args(parser)
    parser.add_argument("--seed", type=int, default=0, help="unused by SuperBPE itself "
        "(fitting is deterministic given the corpus -- see SuperBPEConfig's module "
        "docstring); kept only so --data-source synthetic's own corpus generation, "
        "which DOES take a seed, has one to forward.")
    add_vocab_output_args(parser, "superbpe_")
    return parser


def main(argv=None):
    args = parse_args_with_config(build_arg_parser(), argv)
    cfg = config_from_args(args, SuperBPEConfig)
    train_groups = load_groups(args)
    eval_groups = load_bouquet_dev_for_training(args)

    print(f"data_source={args.data_source} groups={len(train_groups)}\n{cfg}\n")
    trainer = SuperBPETrainer(cfg, train_groups, eval_groups=eval_groups)
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
