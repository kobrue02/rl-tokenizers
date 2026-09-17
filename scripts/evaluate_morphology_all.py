"""Runs common.eval.morphology's MED/Consistency-F1 (via each system's own
--morphology-gold-dir flag) for every tokenizer EXCEPT claude_tokenizer, in
ONE process, driven by a YAML config listing each system's own CLI args.

Generalizes scripts/evaluate_own_tokenizers_indigenous_panel.py's pattern
(list-of-systems + per-entry skip-if-exists + combine_eval_results) to a
genuinely heterogeneous CLI surface: the 6 "own trained" systems take
--checkpoint, hf_frontier takes --hf-repo-id, blt takes neither -- so,
unlike that script, each system's FULL args list comes from the config
itself (entry["args"]) rather than a fixed --checkpoint field. --result-key
is deliberately never passed: the run_eval_cli-based systems already
default it to their own system_label (== entry["name"]), and hf_frontier/
blt don't have that flag at all -- omitting it uniformly avoids
special-casing which systems support it.

claude_tokenizer is deliberately excluded, not just left unconfigured: its
count_tokens API returns only a bare token count, never actual token
spans/boundaries, so MED/Consistency-F1 (both need the tokenizer's actual
predicted segmentation) are structurally uncomputable for it -- see
systems/tokenization/claude_tokenizer/evaluate.py's own module docstring.
Including it in this config's systems list is a hard error, not a silent
skip, so a stale config never quietly claims to have scored it.
fairtok/flexitokens are also excluded, matching this project's own
2026-09-16 scope decision (scripts/generate_tikz_figures.py's own
_REPO_TOKENIZER_NAMES comment: fairtok never trained successfully,
flexitokens no longer pursued).

A system whose output_path already exists is SKIPPED (reusing that file)
rather than re-evaluated, unless --force is passed -- same durable-
completion-marker convention as evaluate_own_tokenizers_indigenous_panel.py
(a real run here spans a real forward-pass model (blt) and real gated-repo
network calls (hf_frontier), so losing already-completed systems to a
later failure would be wasteful).

One system failing doesn't abort the rest -- recorded under a "_failed" key
in the combined output, same per-entry error isolation as every other
multi-system driver in this project.

Usage:
    python3 -m scripts.evaluate_morphology_all -c configs/eval/morphology_all.yml
"""

import argparse
import json
import os

import yaml

import evaluate as evaluate_cli
from scripts.combine_eval_results import main as combine_main

_EXCLUDED_SYSTEMS = {
    "claude_tokenizer": "count_tokens API has no token spans -- see systems/tokenization/"
    "claude_tokenizer/evaluate.py's own module docstring",
    "fairtok": "never trained successfully -- dropped from the project 2026-09-16",
    "flexitokens": "no longer pursued -- dropped from the project 2026-09-16",
}


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "-c", "--config", required=True,
        help="YAML: {morphology_gold_dir, systems: [{name, args: [...]}, ...], "
        "output_dir (default 'results'), combined_output}",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="re-evaluate every system even if its output_path already exists "
        "(default: skip and reuse it -- see the module docstring)",
    )
    return parser


def run_morphology_all(cfg, force=False):
    """cfg: parsed YAML dict (see build_arg_parser's --config help). Returns
    (per_system_paths, failed) -- failed maps system name -> error string for
    any entry whose evaluate_cli.main() call raised (e.g. a stale checkpoint
    path or a transient network error), skipped rather than aborting the
    remaining systems."""
    output_dir = cfg.get("output_dir", "results")
    morphology_gold_dir = cfg["morphology_gold_dir"]
    # Not every output_dir already exists (e.g. configs/eval/
    # morphology_indigenous_panel.yml's own results/indigenous_panel_morphology/,
    # a fresh subdirectory) -- none of the per-system evaluate.py scripts
    # create their own --output's parent directory, so this driver must,
    # once, up front, for every system's output_path below. Confirmed live:
    # without this, every system failed with ENOENT on its own --output path.
    os.makedirs(output_dir, exist_ok=True)
    per_system_paths = []
    failed = {}
    for entry in cfg["systems"]:
        name = entry["name"]
        if name in _EXCLUDED_SYSTEMS:
            raise ValueError(
                f"{name!r} can't be scored here: {_EXCLUDED_SYSTEMS[name]} -- "
                "remove it from this config's systems list"
            )
        output_path = f"{output_dir}/{name}_morphology.json"
        if not force and os.path.exists(output_path):
            print(f"=== {name}: {output_path} already exists, skipping (pass --force to redo) ===")
            per_system_paths.append(output_path)
            continue
        print(f"=== evaluating {name} (morphology_gold_dir={morphology_gold_dir}) ===")
        try:
            evaluate_cli.main([
                name,
                *entry.get("args", []),
                "--morphology-gold-dir", morphology_gold_dir,
                "--output", output_path,
            ])
            per_system_paths.append(output_path)
        except Exception as e:
            print(f"  {name}: FAILED -- {e}")
            failed[name] = str(e)
    return per_system_paths, failed


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    per_system_paths, failed = run_morphology_all(cfg, force=args.force)

    output_dir = cfg.get("output_dir", "results")
    combined_output = cfg.get("combined_output", f"{output_dir}/morphology_all.json")
    if not per_system_paths:
        print(f"no systems succeeded ({len(failed)} failed) -- nothing to combine: {failed}")
        return

    combine_main(["--input", *per_system_paths, "--output", combined_output])
    if failed:
        with open(combined_output, encoding="utf-8") as f:
            combined = json.load(f)
        combined.setdefault("_failed", {}).update(failed)
        with open(combined_output, "w", encoding="utf-8") as f:
            json.dump(combined, f, indent=2)
        print(f"note: {len(failed)} system(s) failed, recorded under _failed in {combined_output}: {failed}")
    print(f"wrote combined results for {len(per_system_paths)} system(s) to {combined_output}")


if __name__ == "__main__":
    main()
