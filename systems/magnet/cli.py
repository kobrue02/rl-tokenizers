"""Command-line entry point for MagnetTrainer, mirroring fairtok.cli's shape:
every MagnetConfig field becomes a `--flag`, generated from the dataclass itself.
Real-data loading is NOT reimplemented here -- common.cli_data.load_groups already
does exactly what's needed (--data-source/--langs/--num-groups against the real
OLDI-and-friends corpus or the synthetic placeholder), and is shared verbatim by
every tokenizer's CLI in this repo (fairtok, magnet, flexitokens, manta).
"""

import argparse
import dataclasses

from common.cli_data import DATA_SOURCES, load_bouquet_dev_for_training, load_groups
from common.config_file import parse_args_with_config
from common.reporting import (
    fertility_by_lang,
    report_collapse,
    report_fertility,
    report_stability,
)
from common.stability import sequences_by_lang_from_groups, stability_by_lang
from common.vocab import save_vocab_json, save_vocab_stats, vocab_with_stats

from .segment import induce_spans
from .train import MagnetConfig, MagnetTrainer, lang_to_script


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Train the MAGNET baseline tokenizer.")

    for field in dataclasses.fields(MagnetConfig):
        if field.default is dataclasses.MISSING:
            # per_script_boundary_prior uses default_factory=dict -- a {script: float}
            # override map isn't representable as a single scalar CLI flag; leave it
            # at its empty-dict default (falls back to default_boundary_prior
            # everywhere) rather than inventing a JSON-string flag for a rarely-needed
            # per-script override.
            continue
        flag = "--" + field.name.replace("_", "-")
        help_text = f"(MagnetConfig.{field.name}, default: {field.default})"
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
    parser.add_argument("--vocab-out", type=str, default="magnet_vocab.json")
    parser.add_argument(
        "--vocab-stats-out", type=str, default="magnet_vocab_stats.json"
    )
    parser.add_argument("--vocab-preview", type=int, default=20)
    return parser


def _config_from_args(args):
    field_names = {f.name for f in dataclasses.fields(MagnetConfig)}
    kwargs = {k: v for k, v in vars(args).items() if k in field_names}
    return MagnetConfig(**kwargs)


def main(argv=None):
    args = parse_args_with_config(build_arg_parser(), argv)
    cfg = _config_from_args(args)
    train_groups = load_groups(args)
    eval_groups = load_bouquet_dev_for_training(args)

    print(f"data_source={args.data_source} groups={len(train_groups)}\n{cfg}\n")
    trainer = MagnetTrainer(cfg, train_groups, eval_groups=eval_groups)
    model, token_freq, final_vocab, loss_trace, boundary_rate_trace = trainer.train()
    report_collapse(token_freq, final_vocab)
    report_fertility(fertility_by_lang(token_freq, train_groups))

    device = next(model.parameters()).device
    sequences_by_lang = sequences_by_lang_from_groups(train_groups)
    induce_fn_by_lang = {
        lang: (
            lambda raw, m=model, s=lang_to_script(lang), d=device: induce_spans(
                m, raw, s, d
            )
        )
        for lang in sequences_by_lang
    }
    report_stability(
        stability_by_lang(induce_fn_by_lang, sequences_by_lang, seed=cfg.seed)
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

    return model, token_freq, final_vocab, loss_trace, boundary_rate_trace


if __name__ == "__main__":
    main()
