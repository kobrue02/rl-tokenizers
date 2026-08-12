"""Single driver CLI for every tokenizer in this repo: fairtok (the RL/GRPO-trained
byte-boundary policy) and the magnet/flexitokens/manta baselines. Replaces the four
separate entry points this project used to have (main.py, train_magnet.py,
train_flexitokens.py, train_manta.py) with one shared front door.

Usage:
    python train.py <tokenizer> [tokenizer-specific flags...]
    python train.py {fairtok,magnet,flexitokens,manta} --help   # that tokenizer's own flags

This file does NOT merge the four tokenizers' flags into one giant parser -- each
tokenizer keeps its own dataclass-driven parser (fairtok.cli.build_arg_parser,
magnet.cli.build_arg_parser, ...), built from ONLY that tokenizer's own Config
fields (+ the shared common.cli_data data-loading flags). This dispatcher just
picks which one to hand the remaining argv to. The consequence -- and the actual
point of keeping it this way rather than one merged parser -- is that passing a
flag that belongs to a DIFFERENT tokenizer is a hard error, not silently accepted:
each tokenizer's parser calls argparse's plain .parse_args() (not
.parse_known_args()), which already rejects anything it doesn't recognize. E.g.
`python train.py magnet --lambda-target 5.0` fails with "unrecognized arguments:
--lambda-target 5.0", since --lambda-target is a fairtok-only GRPOConfig field
that MagnetConfig's parser was never given.
"""

import importlib
import sys

TOKENIZERS = {
    "fairtok": "systems.fairtok.cli",
    "magnet": "systems.magnet.cli",
    "flexitokens": "systems.flexitokens.cli",
    "manta": "systems.manta.cli",
    "fanta": "systems.fanta.cli",
    "superbpe": "systems.superbpe.cli",
    "bpe": "systems.bpe.cli",
}


def _print_usage(file=sys.stdout):
    names = ", ".join(sorted(TOKENIZERS))
    print(
        f"usage: python train.py {{{'|'.join(sorted(TOKENIZERS))}}} [tokenizer-specific args...]",
        file=file,
    )
    print(f"\navailable tokenizers: {names}", file=file)
    print(
        "see a specific tokenizer's own flags with e.g.:  python train.py magnet --help",
        file=file,
    )


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)

    # Deliberately NOT argparse here: a positional "which tokenizer" argument
    # combined with each tokenizer's OWN full flag set doesn't compose cleanly
    # through argparse's required-positional-vs-early-help interaction (a
    # required positional makes `python train.py --help` (no tokenizer given)
    # fail with "the following arguments are required" before argparse's
    # built-in --help handling gets a chance to run). Plain argv[0] inspection
    # sidesteps that entirely and keeps `python train.py <tok> --help` forwarding
    # cleanly to that tokenizer's own parser, showing ITS flags, not this
    # dispatcher's.
    if not argv:
        _print_usage(sys.stderr)
        return 2
    if argv[0] in ("-h", "--help"):
        _print_usage()
        return 0
    tokenizer = argv[0]
    if tokenizer not in TOKENIZERS:
        _print_usage(sys.stderr)
        print(f"\nerror: unknown tokenizer {tokenizer!r}", file=sys.stderr)
        return 2

    module = importlib.import_module(TOKENIZERS[tokenizer])
    # Deliberately ignore module.main()'s own return value here: every
    # tokenizer's cli.main() returns (model, token_freq, vocab, ...) for
    # interactive/notebook use, not an exit code -- passing that straight to
    # sys.exit() would print the whole tuple to stderr and exit 1 on every
    # SUCCESSFUL run. A CLI-level failure inside module.main() (bad flags, a
    # data-loading error, ...) already raises/calls sys.exit() on its own
    # (argparse's unrecognized-arguments error does this directly), which
    # propagates through this call and past the `return 0` below untouched.
    module.main(argv[1:])
    return 0


if __name__ == "__main__":
    sys.exit(main())
