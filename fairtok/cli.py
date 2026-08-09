"""Command-line entry point. Every Config field becomes a `--flag`, generated from
the dataclass itself so this can't drift out of sync with fairtok.train.Config."""

import argparse
import dataclasses

from .data import LANG_PROFILES, make_synthetic_parallel_groups
from .oldi_data import LANGS, load_all_training_groups, load_flores_plus, load_oldi_seed, load_smol_groups
from .train import Config, _report_collapse, run_training
from .vocab import save_vocab_json, save_vocab_stats, vocab_with_stats

DATA_SOURCES = ["synthetic", "oldi_seed", "flores_dev", "smol", "all"]

# Extra clarifying text for fields whose semantics aren't obvious from
# "(Config.field, default: X)" alone -- merged into the auto-generated help.
_HELP_OVERRIDES = {
    "num_steps": "0 derives the step count from num_epochs; set explicitly to override "
                 "with a raw step count instead (bypasses epoch semantics entirely)",
    "num_epochs": "1 epoch = 1 full shuffled traversal of every loaded group, however many "
                  "steps that takes given batch_groups -- not a fixed step count",
}


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Train the fairness-aware byte-boundary policy.")

    for field in dataclasses.fields(Config):
        flag = "--" + field.name.replace("_", "-")
        help_text = f"(Config.{field.name}, default: {field.default})"
        if field.name in _HELP_OVERRIDES:
            help_text = f"{_HELP_OVERRIDES[field.name]} {help_text}"
        if field.type is bool:
            # type=bool would make "--flag false" truthy (any non-empty string is
            # truthy) -- BooleanOptionalAction gives a real --flag/--no-flag pair.
            parser.add_argument(flag, action=argparse.BooleanOptionalAction, default=field.default, help=help_text)
        else:
            parser.add_argument(flag, type=field.type, default=field.default, help=help_text)

    parser.add_argument(
        "--data-source", choices=DATA_SOURCES, default="all",
        help="'all' pools oldi_seed+flores_dev+smol (default); 'synthetic' is the placeholder corpus",
    )
    parser.add_argument(
        "--num-groups", type=int, default=None,
        help="cap the number of parallel groups loaded (real sources are large; omit for the full set)",
    )
    parser.add_argument(
        "--langs", type=str, default=None,
        help="comma-separated language codes; 'all' to use every language oldi_seed/flores_dev "
             "natively offer (smol stays on the 9-language panel regardless); "
             "defaults to the 9-language panel for the chosen data source",
    )
    parser.add_argument(
        "--vocab-out", type=str, default="vocab.json",
        help="where to save the final vocab as a HuggingFace-style {token: id} JSON file; empty string to skip",
    )
    parser.add_argument(
        "--vocab-stats-out", type=str, default="vocab_stats.json",
        help="companion file with per-entry frequency and per-language usage breakdown; empty string to skip",
    )
    parser.add_argument(
        "--vocab-preview", type=int, default=20,
        help="print this many of the most frequent vocab entries to the terminal; 0 to skip",
    )
    return parser


def _config_from_args(args):
    field_names = {f.name for f in dataclasses.fields(Config)}
    kwargs = {k: v for k, v in vars(args).items() if k in field_names}
    return Config(**kwargs)


def _load_groups(args):
    if args.langs is None:
        langs = None
    elif args.langs == "all":
        langs = "all"
    else:
        langs = args.langs.split(",")

    if langs == "all" and args.data_source in ("synthetic", "smol"):
        raise ValueError(
            f"--langs all isn't supported for --data-source {args.data_source} "
            "(synthetic has a fixed toy panel; smol needs a per-language-file rescan -- see oldi_data.py)"
        )

    if args.data_source == "synthetic":
        groups = make_synthetic_parallel_groups(400, langs=langs or list(LANG_PROFILES), seed=args.seed)
    elif args.data_source == "oldi_seed":
        groups = load_oldi_seed(langs=langs or LANGS)
    elif args.data_source == "flores_dev":
        groups = load_flores_plus(split="dev", langs=langs or LANGS)
    elif args.data_source == "smol":
        groups = load_smol_groups(langs=langs or [l for l in LANGS if l != "eng"])
    elif args.data_source == "all":
        groups = load_all_training_groups(langs=langs or LANGS)
    else:
        raise ValueError(f"unknown data source: {args.data_source}")

    if args.num_groups is not None:
        groups = groups[: args.num_groups]
    return groups


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    cfg = _config_from_args(args)
    train_groups = _load_groups(args)

    print(f"data_source={args.data_source} groups={len(train_groups)}\n{cfg}\n")
    policy, token_freq, final_vocab, target_rate = run_training(cfg, train_groups)
    _report_collapse(token_freq, final_vocab)

    entries = vocab_with_stats(token_freq, cfg.vocab_budget)

    if args.vocab_preview:
        print(f"\ntop {min(args.vocab_preview, len(entries))} vocab entries by frequency:")
        for span, total, per_lang in entries[: args.vocab_preview]:
            langs = ", ".join(f"{lang}:{c}" for lang, c in sorted(per_lang.items(), key=lambda kv: -kv[1]))
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
