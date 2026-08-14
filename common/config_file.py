"""Shared -c/--config YAML support, used identically by every CLI entry
point in this repo (all seven systems/*/cli.py, plus pretraining/cli.py,
data_prep.py, cli_eval.py, cli_generate.py) -- one place to define "how does
a YAML file override this command's flags", not eleven copies of the same
logic (same reuse pattern as common/data/cli_data.py's load_groups).

Precedence, low to high: this command's own argparse defaults < the YAML
file's values < flags passed explicitly on the command line. So
`--vocab-size 512` on the command line always wins over `vocab_size: 256` in
the YAML file, which lets one full_run.yml define a whole experiment while
still allowing a one-off override without editing the file.

YAML keys are argparse DEST names (underscores, not dashes -- e.g.
`vocab_size`, `data_source`, `tokenizer_checkpoint`, matching what
`--vocab-size`/`--data-source`/`--tokenizer-checkpoint` turn into), not the
`--flag` spelling -- this mirrors how every cli.py in this repo already maps
dataclass field names to flags (see e.g. systems/bpe/cli.py's
`_config_from_args`), so one mental model covers both directions.

A REPEATED flag (argparse action="append", e.g. pretraining.cli_generate's
--prompt) takes a YAML LIST directly (`prompt: ["a", "b"]`) -- not the
"repeat the flag" convention that has no analogue in YAML. Every other flag
takes the SAME scalar type argparse's own `type=` would produce from a
command-line string (e.g. `vocab_size: 50000` as a YAML int, not a quoted
string -- though a quoted numeral still coerces correctly, see below).
"""

import argparse
import sys

import yaml


def load_yaml_config(path):
    with open(path) as f:
        return yaml.safe_load(f) or {}


def parse_args_with_config(parser, argv=None):
    """Adds -c/--config to `parser` and returns the merged argparse.Namespace.

    Required-ness (argparse's own `required=True`) is deliberately enforced
    AFTER merging, not by argparse itself during the initial parse -- a
    required flag can be satisfied by the YAML file alone (e.g. `-c
    full_run.yml` with no other flags at all), which argparse's built-in
    check has no way to know about until the file has actually been read.
    Every action found `required=True` here is temporarily relaxed, then
    re-checked by hand once the YAML file (if any) has been merged in --
    still a hard error, just raised at the right point in the sequence.
    """
    argv = sys.argv[1:] if argv is None else list(argv)

    parser.add_argument(
        "-c", "--config", type=str, default=None,
        help="YAML file of default values for this command's own flags (dest names, "
        "e.g. vocab_size/data_source/output_dir -- see common.config_file's own "
        "docstring) -- flags passed explicitly here on the command line still "
        "override whatever the YAML file says",
    )

    required_dests = [a.dest for a in parser._actions if getattr(a, "required", False)]
    for action in parser._actions:
        if action.required:
            action.required = False

    args = parser.parse_args(argv)

    if args.config:
        yaml_values = load_yaml_config(args.config)
        by_dest = {a.dest: a for a in parser._actions}
        unknown_keys = set(yaml_values) - set(by_dest)
        if unknown_keys:
            raise ValueError(
                f"{args.config}: unknown key(s) {sorted(unknown_keys)} -- not a flag this "
                f"command recognizes (expected some of {sorted(by_dest)})"
            )
        explicitly_passed = {
            dest for dest, action in by_dest.items()
            if dest != "config" and any(opt in argv for opt in action.option_strings)
        }

        for key, value in yaml_values.items():
            if key in explicitly_passed:
                continue  # an explicit CLI flag always wins over the YAML file
            action = by_dest[key]
            is_append = type(action).__name__ == "_AppendAction"

            if isinstance(value, list) and not is_append:
                flag = action.option_strings[-1] if action.option_strings else key
                raise ValueError(
                    f"{args.config}: key {key!r} is a YAML list, but {flag} expects a "
                    "single value (a comma-separated string, for this repo's own "
                    "comma-separated flags like --langs/--lang-pairs/--benchmark) -- "
                    "only flags that are repeatable on the command line (action="
                    "'append', e.g. --prompt) take a YAML list"
                )

            if action.choices is not None:
                check_values = value if is_append else [value]
                bad = [v for v in check_values if v not in action.choices]
                if bad:
                    raise ValueError(
                        f"{args.config}: key {key!r} has invalid value(s) {bad} -- "
                        f"choices are {list(action.choices)}"
                    )

            if action.type is not None and value is not None and not is_append:
                try:
                    value = action.type(value)
                except (TypeError, ValueError) as e:
                    raise ValueError(
                        f"{args.config}: key {key!r}={value!r} isn't a valid "
                        f"{action.type.__name__} -- {e}"
                    )

            setattr(args, key, value)

    missing = [dest for dest in required_dests if getattr(args, dest, None) is None]
    if missing:
        raise SystemExit(
            f"missing required value(s) for {missing} -- pass on the command line or "
            "set in the YAML file given via --config"
        )
    return args
