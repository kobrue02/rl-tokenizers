"""Command-line entry point. Every GRPOConfig field becomes a `--flag`, generated
from the dataclass itself so this can't drift out of sync with fairtok.train.GRPOConfig.
"""

import argparse
import dataclasses

from common.bytes_utils import bytes_to_tensor
from common.cli_data import DATA_SOURCES, load_bouquet_dev_for_training, load_groups
from common.reporting import (
    fertility_by_lang,
    report_collapse,
    report_fertility,
    report_stability,
)
from common.stability import sequences_by_lang_from_groups, stability_by_lang
from common.vocab import save_vocab_json, save_vocab_stats, vocab_with_stats

from .policy import segment_bytes
from .train import GRPOConfig, GRPOTrainer

# Extra clarifying text for fields whose semantics aren't obvious from
# "(GRPOConfig.field, default: X)" alone -- merged into the auto-generated help.
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

    for field in dataclasses.fields(GRPOConfig):
        flag = "--" + field.name.replace("_", "-")
        help_text = f"(GRPOConfig.{field.name}, default: {field.default})"
        if field.name in _HELP_OVERRIDES:
            help_text = f"{_HELP_OVERRIDES[field.name]} {help_text}"
        if field.type is bool:
            # type=bool would make "--flag false" truthy (any non-empty string is
            # truthy) -- BooleanOptionalAction gives a real --flag/--no-flag pair.
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
        help="comma-separated language codes; 'all' to use every language oldi_seed/flores_dev "
        "natively offer (smol stays on the 9-language panel regardless); "
        "defaults to the 9-language panel for the chosen data source",
    )
    parser.add_argument(
        "--vocab-out",
        type=str,
        default="vocab.json",
        help="where to save the final vocab as a HuggingFace-style {token: id} JSON file; empty string to skip",
    )
    parser.add_argument(
        "--vocab-stats-out",
        type=str,
        default="vocab_stats.json",
        help="companion file with per-entry frequency and per-language usage breakdown; empty string to skip",
    )
    parser.add_argument(
        "--vocab-preview",
        type=int,
        default=20,
        help="print this many of the most frequent vocab entries to the terminal; 0 to skip",
    )
    return parser


def _config_from_args(args):
    field_names = {f.name for f in dataclasses.fields(GRPOConfig)}
    kwargs = {k: v for k, v in vars(args).items() if k in field_names}
    return GRPOConfig(**kwargs)


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    cfg = _config_from_args(args)
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

    return policy, token_freq, final_vocab, target_rate


if __name__ == "__main__":
    main()
