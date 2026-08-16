"""Single driver CLI for every tokenizer in this repo (fairtok's RL/GRPO-trained
byte-boundary policy plus the magnet/flexitokens/manta baselines), replacing the
four separate entry points this project used to have.

Usage:
    python train.py <tokenizer> [tokenizer-specific flags...]
    python train.py {fairtok,magnet,flexitokens,manta} --help   # that tokenizer's own flags

Each tokenizer keeps its own dataclass-driven argparse parser (built from only its
own Config fields); this dispatcher just picks which one to hand argv to, rather
than merging them into one giant parser. Consequence: a flag belonging to a
DIFFERENT tokenizer is a hard error, not silently accepted -- e.g.
`python train.py magnet --lambda-target 5.0` fails, since --lambda-target is a
fairtok-only field MagnetConfig's parser never sees.
"""

import importlib
import sys

TOKENIZERS = {
    "fairtok": "systems.tokenization.fairtok.cli",
    "magnet": "systems.tokenization.magnet.cli",
    "flexitokens": "systems.tokenization.flexitokens.cli",
    "manta": "systems.tokenization.manta.cli",
    "fanta": "systems.tokenization.fanta.cli",
    "superbpe": "systems.tokenization.superbpe.cli",
    "bpe": "systems.tokenization.bpe.cli",
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

    # Deliberately not argparse: a required positional "which tokenizer" arg would
    # make `python train.py --help` (no tokenizer given) fail on "required
    # arguments" before argparse's own --help handling runs. Plain argv[0]
    # inspection avoids that and lets `<tok> --help` forward to that tokenizer's
    # own parser.
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
    # Ignore module.main()'s return value: cli.main() returns (model, token_freq,
    # vocab, ...) for notebook use, not an exit code -- passing that to sys.exit()
    # would print the tuple and exit 1 on every SUCCESSFUL run. Real CLI failures
    # already raise/sys.exit() on their own and propagate past `return 0` below.
    module.main(argv[1:])
    return 0


if __name__ == "__main__":
    sys.exit(main())
