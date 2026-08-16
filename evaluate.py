"""Driver CLI for evaluating any trained tokenizer checkpoint on held-out data.
Mirrors train.py's dispatch pattern: each tokenizer keeps its own argparse parser,
so an unrecognized flag is a hard error, not silently accepted.

Usage:
    python evaluate.py <tokenizer> --checkpoint PATH [tokenizer-specific flags...]
    python evaluate.py {fairtok,magnet,flexitokens,manta} --help   # that tokenizer's own flags

Held-out data defaults to BOUQuET dev (disjoint from anything train.py trains on).
"""

import importlib
import sys

TOKENIZERS = {
    "fairtok": "systems.tokenization.fairtok.evaluate",
    "magnet": "systems.tokenization.magnet.evaluate",
    "flexitokens": "systems.tokenization.flexitokens.evaluate",
    "manta": "systems.tokenization.manta.evaluate",
    "fanta": "systems.tokenization.fanta.evaluate",
    "superbpe": "systems.tokenization.superbpe.evaluate",
    "bpe": "systems.tokenization.bpe.evaluate",
    "hf_frontier": "systems.tokenization.hf_frontier.evaluate",  # arbitrary HF tokenizer (--hf-repo-id), not one this project trains
    "claude_tokenizer": "systems.tokenization.claude_tokenizer.evaluate",  # token counts only via Anthropic's count_tokens API; no renyi/gini
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
    # evaluate.main() returns a results dict (for notebook use), not an exit code --
    # discard it so a successful run exits 0 instead of sys.exit()-ing on a truthy dict.
    module.main(argv[1:])
    return 0


if __name__ == "__main__":
    sys.exit(main())
