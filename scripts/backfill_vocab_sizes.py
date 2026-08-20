"""Backfills results/vocab_sizes.json: {tokenizer_name: {"vocab_size": int|null,
"source": str}} for every tokenizer in a results comparison file (default
results/all_tokenizers_comparison.json). No results/*.json produced by
combine_eval_results.py records vocab_size -- it was never part of
common.eval.cross_tokenizer.evaluate_on_groups's return contract -- so this
is sourced separately, per tokenizer family:

  - This project's own 7 tokenizers (bpe/superbpe/magnet/flexitokens/manta/
    fanta/parity_bpe): every jobs/train_*.sh usage example trains with
    --vocab-size 50000 -- taken as the declared value. NOT verified against
    a live checkpoint (none are present in this local checkout; the actual
    training runs happen on the cluster). Flagged accordingly in "source".
  - claude-opus-5: Anthropic does not publish a tokenizer or vocab size --
    recorded as vocab_size=null, not guessed.
  - Everything else (HF Hub repos + tiktoken: encodings): loaded LIVE via
    systems.tokenization.hf_frontier.model.HFFrontierTokenizer.load (the
    exact same loader used to produce results/hf_frontier_comparison.json),
    reading .vocab_size after load. One repo failing (gated without access,
    a transient network error) is recorded as vocab_size=null with the
    error message, not allowed to abort the whole run -- same per-repo
    isolation philosophy as systems/tokenization/hf_frontier/evaluate.py.

Usage:
    python3 scripts/backfill_vocab_sizes.py \\
        --input results/all_tokenizers_comparison.json \\
        --output results/vocab_sizes.json --trust-remote-code
"""

import argparse
import json
import sys

from scripts.generate_tikz_figures import _REPO_TOKENIZER_NAMES
from systems.tokenization.hf_frontier.model import HFFrontierTokenizer

_OWN_VOCAB_SIZE = 50000
_OWN_SOURCE = (
    "declared training target (--vocab-size 50000, see jobs/train_*.sh usage "
    "examples) -- not verified against a live checkpoint, none present locally"
)


def backfill(names, trust_remote_code, hf_token=None):
    """names: iterable of tokenizer_name strings. Returns {name: {"vocab_size":
    int|None, "source": str}}, one entry per name, in the same order."""
    result = {}
    for name in names:
        if name in _REPO_TOKENIZER_NAMES:
            result[name] = {"vocab_size": _OWN_VOCAB_SIZE, "source": _OWN_SOURCE}
            continue
        if "claude" in name.lower():
            result[name] = {
                "vocab_size": None,
                "source": "Anthropic does not publish a tokenizer or vocab size",
            }
            continue
        try:
            tok = HFFrontierTokenizer.load(name, trust_remote_code=trust_remote_code, hf_token=hf_token)
            result[name] = {"vocab_size": tok.vocab_size, "source": "live HFFrontierTokenizer.load()"}
        except Exception as e:  # noqa: BLE001 -- one bad repo must not abort the run
            result[name] = {"vocab_size": None, "source": f"load failed: {e}"}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="results/all_tokenizers_comparison.json")
    parser.add_argument("--output", default="results/vocab_sizes.json")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--hf-token", default=None)
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as f:
        data = json.load(f)
    names = [k for k in data if k != "_failed"]

    result = backfill(names, trust_remote_code=args.trust_remote_code, hf_token=args.hf_token)

    failed = [n for n, v in result.items() if v["vocab_size"] is None and "load failed" in v["source"]]
    if failed:
        print(f"warning: {len(failed)}/{len(names)} tokenizers failed to load: {failed}", file=sys.stderr)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, sort_keys=True)
    print(f"wrote {len(result)} entries to {args.output} ({len(failed)} failed)")


if __name__ == "__main__":
    main()
