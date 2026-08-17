"""Runs common.data.indigenous_panel evaluation for every one of this
project's own trained tokenizers in ONE process, driven by a YAML config
listing each system's checkpoint -- evaluate.py's own dispatcher only
handles one system per invocation (see its module docstring), and
jobs/evaluate.sh mirrors that one-system-per-sbatch shape; this exists so a
single SLURM job can cover all 5 rather than five separate submissions,
since each is a cheap CPU-only eval anyway. Calling evaluate.py's main()
repeatedly in one process is safe: run_eval_cli's own wandb.init()/
run.finish() pair is fully scoped within each call, and neither it nor
evaluate.py's own main() ever sys.exit()s on success.

Not built on common.config_file.parse_args_with_config -- that helper
merges a flat YAML file into ONE argparse Namespace; this config is a LIST
of per-system entries, a genuinely different shape.

Writes one results/<system>_indigenous_panel.json per system (the same
{result_key: results} shape every systems/*/evaluate.py --output already
writes) plus one combined file via scripts.combine_eval_results -- safe
here since all 5 outputs share that identical shape with disjoint
top-level keys (unlike combining indigenous_panel with a bouquet-shaped
file, see jobs/combine_and_generate_figures.sh's own warning about that).

MAGNET CAVEAT (see run_eval_cli's own docstring): magnet resolves each
language to a script via eval_lang_to_script before looking up a boundary
predictor; indigenous_panel's codes (crk, iu, chr, mi, arn, ...) aren't in
magnet.train.LANG_SCRIPT, so magnet scores 0 languages on this panel --
expected, not a bug in this driver.

One system failing (e.g. a stale/placeholder --checkpoint path) doesn't
abort the rest of the list -- same per-entry error isolation as
systems/tokenization/hf_frontier/evaluate.py's own per-repo loop, recorded
under a "_failed" key in the combined output (only present if something
failed) rather than losing every OTHER system's real results.

A system whose output_path already exists is SKIPPED (reusing that file)
rather than re-evaluated, unless --force is passed -- confirmed live to
matter: a real run OOM-killed on the 2nd system after the 1st had already
completed, and a naive resubmit would otherwise waste that completed work
every time. Same "don't redo what a durable completion marker already
proves finished" fix as systems/tokenization/superbpe/model.py's own
stage1_result.json.

Usage:
    python -m scripts.evaluate_own_tokenizers_indigenous_panel \\
        -c configs/eval_own_tokenizers_indigenous_panel.yml
"""

import argparse
import json
import os

import yaml

import evaluate as evaluate_cli
from scripts.combine_eval_results import main as combine_main


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "-c", "--config", required=True,
        help="YAML: {systems: [{name, checkpoint, extra_args: [...]}, ...], "
        "output_dir (default 'results'), combined_output}",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="re-evaluate every system even if its output_path already exists "
        "(default: skip and reuse it -- see the module docstring)",
    )
    return parser


def run_own_tokenizers_indigenous_panel(cfg, force=False):
    """cfg: the parsed YAML dict (see build_arg_parser's --config help).
    Returns (per_system_paths, failed) -- failed maps system name -> error
    string for any entry whose evaluate_cli.main() call raised (e.g. a
    stale/placeholder --checkpoint), which is skipped rather than aborting
    the remaining systems."""
    output_dir = cfg.get("output_dir", "results")
    per_system_paths = []
    failed = {}
    for entry in cfg["systems"]:
        name = entry["name"]
        output_path = f"{output_dir}/{name}_indigenous_panel.json"
        if not force and os.path.exists(output_path):
            print(f"=== {name}: {output_path} already exists, skipping (pass --force to redo) ===")
            per_system_paths.append(output_path)
            continue
        print(f"=== evaluating {name} on indigenous_panel ===")
        try:
            evaluate_cli.main([
                name,
                "--checkpoint", entry["checkpoint"],
                "--eval-data-source", "indigenous_panel",
                "--output", output_path,
                "--result-key", name,
                *entry.get("extra_args", []),
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

    per_system_paths, failed = run_own_tokenizers_indigenous_panel(cfg, force=args.force)

    output_dir = cfg.get("output_dir", "results")
    combined_output = cfg.get("combined_output", f"{output_dir}/own_tokenizers_indigenous_panel.json")
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
