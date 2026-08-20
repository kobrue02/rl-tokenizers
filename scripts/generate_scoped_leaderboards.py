"""Generates two additional standalone leaderboard figures, each scoped to a
SUBSET of results/all_tokenizers_comparison.json, alongside (not replacing)
the main multi-family comparison scripts/generate_tikz_figures.py itself
produces:

  1. hf_frontier: every "frontier" tokenizer -- results/hf_frontier_comparison.json's
     own key set (every model actually hosted on the HF Hub) PLUS claude-opus-5
     explicitly (Anthropic doesn't publish its tokenizer on the HF Hub, so it's
     evaluated via a separate claude_tokenizer pipeline and merged in from
     results/claude_comparison.json -- but it's unambiguously a frontier model's
     tokenizer and belongs in a "broad frontier comparison" alongside GPT/Llama/
     DeepSeek/etc., just not literally an "HF Hub" one). Isolated from this
     project's OWN tokenizers, so the sheer number of frontier models isn't
     diluted by them.
  2. our_work_vs_other_approaches: fanta (this thesis's own contribution) vs.
     manta/magnet/parity_bpe/flexitokens (other published fairness-aware
     tokenization methods, reproduced/reused here) -- a row-level family
     OVERRIDE reclassifies the latter four under "Other approaches" (see
     scripts/generate_tikz_figures.py's own _FAMILY_COLORS comment for why
     this needs a manual override rather than family_of() itself, which
     buckets all of this project's own tokenizers as "This work" everywhere
     else, correctly).

Both reuse gen_spread_leaderboard_tex/compute_families/write_bar_data
directly (no new figure logic) -- only the INPUT ROW SET and (for #2) the
row-level family assignment differ from the main comparison.

Usage:
    python3 -m scripts.generate_scoped_leaderboards \\
        --input results/all_tokenizers_comparison.json \\
        --hf-frontier-keys results/hf_frontier_comparison.json \\
        --output-dir figures/tikz
"""

import argparse
import json
import os

from scripts.generate_tikz_figures import (
    MIN_ROWS_FOR_TWO_COLUMN_LEADERBOARD,
    _assert_well_formed,
    compute_families,
    gen_spread_leaderboard_tex,
    load_rows,
    write_bar_data,
)


def _expected_tikzpictures(num_rows):
    """Mirrors gen_spread_leaderboard_tex's own column-count decision (see
    that function's docstring) so callers here can assert the right shape
    regardless of how many rows a given subset happens to have."""
    return 2 if num_rows >= MIN_ROWS_FOR_TWO_COLUMN_LEADERBOARD else 1

_OTHER_APPROACHES = {"manta", "magnet", "parity_bpe", "flexitokens"}
_OUR_WORK = {"fanta"}
# Not an HF Hub model, so absent from results/hf_frontier_comparison.json's
# own key set -- added explicitly to the "frontier" leaderboard anyway (see
# module docstring). family_of() already classifies "claude-opus-5" as
# "Anthropic", not "Other"/HF-frontier-ish, so it renders in its own
# distinct color here too, same as everywhere else in this project.
_EXTRA_FRONTIER_KEYS = {"claude-opus-5"}


def _write_filtered_json(all_results_path, keys, tmp_path):
    with open(all_results_path, encoding="utf-8") as f:
        data = json.load(f)
    missing = [k for k in keys if k not in data]
    if missing:
        raise ValueError(f"{all_results_path!r} is missing expected key(s): {missing}")
    filtered = {k: data[k] for k in keys if k in data}
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(filtered, f, indent=2)
    return len(filtered)


def generate_hf_frontier_leaderboard(
    all_results_path, hf_frontier_keys_path, out_dir, data_prefix=None,
    extra_keys=frozenset(_EXTRA_FRONTIER_KEYS),
):
    """Figure #1 -- see module docstring. hf_frontier_keys_path: any results
    JSON whose OWN key set defines "every HF-frontier tokenizer" (this
    project's own results/hf_frontier_comparison.json, not re-derived from
    naming heuristics -- that file IS the authoritative list of what got
    evaluated as an HF-frontier tokenizer). extra_keys: additional model(s)
    to include even though they're absent from that file -- defaults to
    {"claude-opus-5"} (see module docstring for why); pass an empty set/
    frozenset to get literally only the HF-Hub-hosted models."""
    with open(hf_frontier_keys_path, encoding="utf-8") as f:
        hf_frontier_keys = sorted(set(json.load(f).keys()) | set(extra_keys))

    os.makedirs(out_dir, exist_ok=True)
    tmp_json = os.path.join(out_dir, "_filtered_hf_frontier.json")
    n = _write_filtered_json(all_results_path, hf_frontier_keys, tmp_json)

    base_prefix = out_dir.replace(os.sep, "/") if data_prefix is None else data_prefix
    if base_prefix and not base_prefix.endswith("/"):
        base_prefix += "/"

    rows, _models = load_rows(tmp_json)
    families = compute_families(rows)
    write_bar_data(rows, families, out_dir)
    tex = gen_spread_leaderboard_tex(
        rows, families, out_dir, data_prefix=base_prefix,
        fig_name="spread_leaderboard_hf_frontier",
    )
    _assert_well_formed(tex, "fig_spread_leaderboard_hf_frontier.tex", expected_tikzpictures=_expected_tikzpictures(len(rows)))
    os.remove(tmp_json)
    print(f"wrote hf_frontier leaderboard ({n} models) to {out_dir}")
    return rows, families


def generate_our_work_vs_other_approaches_leaderboard(all_results_path, out_dir, data_prefix=None):
    """Figure #2 -- see module docstring's row-level family override."""
    keys = sorted(_OUR_WORK | _OTHER_APPROACHES)
    os.makedirs(out_dir, exist_ok=True)
    tmp_json = os.path.join(out_dir, "_filtered_our_work_vs_other.json")
    n = _write_filtered_json(all_results_path, keys, tmp_json)

    base_prefix = out_dir.replace(os.sep, "/") if data_prefix is None else data_prefix
    if base_prefix and not base_prefix.endswith("/"):
        base_prefix += "/"

    rows, _models = load_rows(tmp_json)
    for r in rows:
        r["family"] = "This work" if r["name"] in _OUR_WORK else "Other approaches"
    rows.sort(key=lambda r: r["spread"])
    for i, r in enumerate(rows):
        r["idx"] = i

    families = compute_families(rows)
    write_bar_data(rows, families, out_dir)
    tex = gen_spread_leaderboard_tex(
        rows, families, out_dir, data_prefix=base_prefix,
        fig_name="spread_leaderboard_our_work_vs_other_approaches",
    )
    _assert_well_formed(
        tex, "fig_spread_leaderboard_our_work_vs_other_approaches.tex",
        expected_tikzpictures=_expected_tikzpictures(len(rows)),
    )
    os.remove(tmp_json)
    print(f"wrote our-work-vs-other-approaches leaderboard ({n} models) to {out_dir}")
    return rows, families


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=str, default="results/all_tokenizers_comparison.json")
    parser.add_argument("--hf-frontier-keys", type=str, default="results/hf_frontier_comparison.json")
    parser.add_argument("--output-dir", type=str, default="figures/tikz")
    parser.add_argument("--data-prefix", type=str, default=None)
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    generate_hf_frontier_leaderboard(
        args.input, args.hf_frontier_keys,
        os.path.join(args.output_dir, "spread_leaderboard_hf_frontier"),
        data_prefix=f"{args.data_prefix}/spread_leaderboard_hf_frontier" if args.data_prefix else None,
    )
    generate_our_work_vs_other_approaches_leaderboard(
        args.input,
        os.path.join(args.output_dir, "spread_leaderboard_our_work_vs_other_approaches"),
        data_prefix=f"{args.data_prefix}/spread_leaderboard_our_work_vs_other_approaches" if args.data_prefix else None,
    )


if __name__ == "__main__":
    main()
