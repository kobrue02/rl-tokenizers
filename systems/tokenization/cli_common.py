"""Shared CLI-scaffolding helpers every systems/*/cli.py uses to build its own
argparse parser from its own Config dataclass -- extracted after confirming
these exact pieces were copy-pasted verbatim (modulo the Config class name
itself) across all seven systems' cli.py, the same "implemented exactly once"
reasoning as common.data.cli_data's own load_groups.

Deliberately NOT a single mega build_arg_parser(system) that also owns
--data-source/--vocab-out (see common.data.cli_data.add_data_source_args and
common.vocab.report_and_save_vocab for those) or main()'s own body: each
system's build_arg_parser/main stay their own real functions, since a
system's --seed quirk (bpe), extra _HELP_OVERRIDES (fairtok), or differing
trainer.train() return arity (3 values for manta/bpe/superbpe, 4 for fairtok,
5 for magnet/flexitokens/fanta) are genuine differences worth keeping visible
in each system's own file, not papered over by a one-size-fits-all driver
(see systems/base.py's own docstring for this same "only unify what's
genuinely identical" principle applied to Config/Trainer).
"""

import argparse
import dataclasses


def add_dataclass_fields(parser, config_cls, help_overrides=None):
    """One `--flag` per field of `config_cls`, generated from the dataclass
    itself so a CLI can't drift out of sync with its own Config -- the
    per-field loop every systems/*/cli.py's build_arg_parser already ran
    itself, verbatim. help_overrides: optional {field_name: extra_text}
    merged in front of the auto-generated "(ConfigCls.field, default: X)"
    help (see fairtok/cli.py's own _HELP_OVERRIDES for the one system that
    uses this -- clarifying prose for fields whose semantics aren't obvious
    from the default alone, e.g. max_steps/num_train_epochs's interaction)."""
    help_overrides = help_overrides or {}
    for field in dataclasses.fields(config_cls):
        if field.default is dataclasses.MISSING:
            # A default_factory field (e.g. MagnetConfig.per_script_boundary_prior,
            # a {script: float} override map) has no single scalar representable
            # as one CLI flag -- skip it and leave it at its own default_factory
            # value, same as magnet/cli.py's own original skip logic did.
            continue
        flag = "--" + field.name.replace("_", "-")
        help_text = f"({config_cls.__name__}.{field.name}, default: {field.default})"
        if field.name in help_overrides:
            help_text = f"{help_overrides[field.name]} {help_text}"
        if field.type is bool:
            # type=bool would make "--flag false" truthy (any non-empty string
            # is truthy) -- BooleanOptionalAction gives a real --flag/--no-flag pair.
            parser.add_argument(
                flag, action=argparse.BooleanOptionalAction, default=field.default, help=help_text
            )
        else:
            parser.add_argument(flag, type=field.type, default=field.default, help=help_text)


def config_from_args(args, config_cls):
    """Builds a `config_cls` instance from an argparse Namespace, ignoring
    every attribute that isn't one of its own dataclass fields (e.g. `config`,
    always present once common.config_file.parse_args_with_config is wired
    in; --data-source/--langs/--num-groups/--vocab-out and friends, which are
    CLI-only concerns no Config dataclass declares) -- the exact
    `_config_from_args` every systems/*/cli.py already declared, verbatim."""
    field_names = {f.name for f in dataclasses.fields(config_cls)}
    kwargs = {k: v for k, v in vars(args).items() if k in field_names}
    return config_cls(**kwargs)


def add_vocab_output_args(
    parser, vocab_prefix, vocab_out_help=None, vocab_stats_help=None, vocab_preview_help=None
):
    """--vocab-out/--vocab-stats-out/--vocab-preview, the trailing three flags
    every systems/*/cli.py's build_arg_parser already added identically
    (confirmed live: six of seven pass no help text at all on these three;
    fairtok is the one exception, with its own richer explanatory text --
    passed through the optional *_help params rather than silently dropped).
    vocab_prefix: the default filename stem -- "{system}_" for six systems
    ("manta_vocab.json" etc.), "" for fairtok specifically (its own
    established "vocab.json"/"vocab_stats.json", predating the other six's
    per-system-prefixed convention -- kept as-is rather than renamed, since
    changing a checkpoint-adjacent default filename is a real behavior change
    for anyone with existing scripts/configs pointing at it)."""
    parser.add_argument(
        "--vocab-out", type=str, default=f"{vocab_prefix}vocab.json", help=vocab_out_help
    )
    parser.add_argument(
        "--vocab-stats-out", type=str, default=f"{vocab_prefix}vocab_stats.json", help=vocab_stats_help
    )
    parser.add_argument("--vocab-preview", type=int, default=20, help=vocab_preview_help)
