"""Command-line entry point for SuperBPETrainer, mirroring manta.cli's shape:
every SuperBPEConfig field becomes a `--flag`, generated from the dataclass
itself. Real-data loading is NOT reimplemented here -- common.cli_data.load_groups
already does exactly what's needed, and is shared verbatim by every tokenizer's
CLI in this repo (fairtok, magnet, flexitokens, manta, fanta, superbpe).
"""

import argparse
import dataclasses

from common.cli_data import DATA_SOURCES, load_bouquet_dev_for_training, load_groups
from common.reporting import (
    fertility_by_lang,
    report_collapse,
    report_fertility,
    report_stability,
)
from common.stability import sequences_by_lang_from_groups, stability_by_lang
from common.vocab import save_vocab_json, save_vocab_stats, vocab_with_stats

from .segment import induce_spans
from .train import SuperBPEConfig, SuperBPETrainer


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Fit a SuperBPE tokenizer (two-stage byte-level BPE)."
    )

    for field in dataclasses.fields(SuperBPEConfig):
        flag = "--" + field.name.replace("_", "-")
        help_text = f"(SuperBPEConfig.{field.name}, default: {field.default})"
        if field.type is bool:
            parser.add_argument(
                flag,
                action=argparse.BooleanOptionalAction,
                default=field.default,
                help=help_text,
            )
        else:
            parser.add_argument(
                flag, type=field.type, default=field.default, help=help_text
            )

    parser.add_argument(
        "--data-source",
        choices=DATA_SOURCES,
        default="all",
        help="'all' pools oldi_seed+flores_dev+smol (default); 'synthetic' is the placeholder corpus",
    )
    parser.add_argument(
        "--num-groups",
        type=int,
        default=None,
        help="cap the number of parallel groups loaded (real sources are large; omit for the full set)",
    )
    parser.add_argument(
        "--langs",
        type=str,
        default=None,
        help="comma-separated language codes; 'all' to use every language oldi_seed/flores_dev natively "
        "offer; defaults to the 9-language panel for the chosen data source",
    )
    parser.add_argument("--seed", type=int, default=0, help="unused by SuperBPE itself "
        "(fitting is deterministic given the corpus -- see SuperBPEConfig's module "
        "docstring); kept only so --data-source synthetic's own corpus generation, "
        "which DOES take a seed, has one to forward.")
    parser.add_argument("--vocab-out", type=str, default="superbpe_vocab.json")
    parser.add_argument("--vocab-stats-out", type=str, default="superbpe_vocab_stats.json")
    parser.add_argument("--vocab-preview", type=int, default=20)
    return parser


def _config_from_args(args):
    field_names = {f.name for f in dataclasses.fields(SuperBPEConfig)}
    kwargs = {k: v for k, v in vars(args).items() if k in field_names}
    return SuperBPEConfig(**kwargs)


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    cfg = _config_from_args(args)
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

    entries = vocab_with_stats(token_freq, cfg.vocab_size)
    if args.vocab_preview:
        print(
            f"\ntop {min(args.vocab_preview, len(entries))} vocab entries by frequency:"
        )
        for span, total, per_lang in entries[: args.vocab_preview]:
            langs = ", ".join(
                f"{lang}:{c}"
                for lang, c in sorted(per_lang.items(), key=lambda kv: -kv[1])
            )
            print(f"  {total:6d}  {span!r:20s} [{langs}]")
    if args.vocab_out:
        save_vocab_json(entries, args.vocab_out)
        print(f"\nsaved vocab ({len(entries)} entries) to {args.vocab_out}")
    if args.vocab_stats_out:
        save_vocab_stats(entries, args.vocab_stats_out)
        print(f"saved per-entry frequency/language stats to {args.vocab_stats_out}")

    return model, token_freq, final_vocab


if __name__ == "__main__":
    main()
