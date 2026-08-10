"""Single driver CLI for evaluating any trained tokenizer checkpoint in this repo on
held-out data -- mirrors train.py's dispatch pattern exactly (see that file's module
docstring for the full rationale: each tokenizer keeps its own argparse parser, so an
unrecognized flag is a hard error, not silently accepted).

Usage:
    python evaluate.py <tokenizer> --checkpoint PATH [tokenizer-specific flags...]
    python evaluate.py {fairtok,magnet,flexitokens,manta} --help   # that tokenizer's own flags

Held-out data defaults to BOUQuET dev (disjoint from every --data-source train.py
trains on) -- see fairtok/evaluate.py's module docstring for the full rationale and
common.eval_common for the shared scoring logic every tokenizer's own evaluate.py uses.
"""

import importlib
import sys

TOKENIZERS = {
    "fairtok": "fairtok.evaluate",
    "magnet": "magnet.evaluate",
    "flexitokens": "flexitokens.evaluate",
    "manta": "manta.evaluate",
}


def _print_usage(file=sys.stdout):
    names = ", ".join(sorted(TOKENIZERS))
    print(
        f"usage: python evaluate.py {{{'|'.join(sorted(TOKENIZERS))}}} --checkpoint PATH [tokenizer-specific args...]",
        file=file,
    )
    print(f"\navailable tokenizers: {names}", file=file)
    print(
        "see a specific tokenizer's own flags with e.g.:  python evaluate.py magnet --help",
        file=file,
    )


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)

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
    # Same reasoning as train.py: every tokenizer's evaluate.main() returns a results
    # dict for interactive/notebook use, not an exit code -- discard it here so a
    # successful run exits 0 instead of sys.exit()-ing on a truthy dict.
    module.main(argv[1:])
    return 0


if __name__ == "__main__":
    sys.exit(main())
