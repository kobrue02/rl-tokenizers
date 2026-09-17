"""Read-only pipeline status audit: for every experiment this project's own
configs/ describes (tokenizer training, data prep, pretraining, generation,
downstream/intrinsic eval), reports DONE / IN_PROGRESS / MISSING / UNKNOWN
by checking the real filesystem artifact each stage's own code writes on
completion -- never guessed, always derived from that stage's own
docstring/source (see each _check_* function for exactly which file and
why). Makes no filesystem changes and submits no jobs -- run directly on
the login node (or locally, though most artifacts only exist on the
cluster's own WORK_ROOT-backed paths).

Every config file in this project starts with a `# <module.py> -- sbatch ...`
comment line naming its own target CLI -- this is what routes each config to
the right stage checker below, not the directory it lives in (configs/eval/
alone holds THREE genuinely different shapes: systems/pretraining/
cli_eval.py's downstream-benchmark configs, systems/tokenization/*/
evaluate.py's intrinsic-comparison configs, and this project's own
multi-system wrapper scripts' list-of-systems configs).

Tokenizer training is the one stage NOT walked from configs/train_tokenizer/
-- magnet/manta were trained directly via their own jobs/train_tokenizer/*.sh
with no -c config at all (job-ID-tagged checkpoint path hardcoded in the
.sh), so a config-only walk would silently miss them. Checked instead via
the same glob-per-system convention jobs/eval/latest_checkpoints.sh already
uses (checkpoints/{pattern}), independent of whether any config exists for
that system.

CAVEAT (parity_bpe specifically): three configs (parity_bpe_50k/_hybrid_50k/
_window_50k) all produce a checkpoint matching the SAME
checkpoints/parity_bpe_*.json glob (job-ID-tagged, not variant-tagged) --
this script can report "at least one parity_bpe checkpoint exists" but
cannot attribute a specific glob match to a specific variant config. See
configs/prep/parity_bpe_50k.yml's own comment for how that's disambiguated
by hand today (checking job timestamps against `ls checkpoints/`).

Usage:
    python3 -m scripts.check_pipeline_status            # human-readable report
    python3 -m scripts.check_pipeline_status --json      # machine-readable
    python3 -m scripts.check_pipeline_status --stage prep pretrain  # subset
"""

import argparse
import glob
import json
import os
import re

import yaml

_CHECKPOINT_GLOBS = {
    "fairtok": "policy_*.pt",
    "magnet": "magnet_*.pt",
    "flexitokens": "flexitokens_*.pt",
    "manta": "manta_*.pt",
    "fanta": "fanta_*.pt",
    "superbpe": "superbpe_*.pt",
    "bpe": "bpe_*.json",
    "parity_bpe": "parity_bpe_*.json",
}

_MODULE_LINE_RE = re.compile(r"^#\s*(\S+\.py)\b")

_STAGE_ORDER = ["train_tokenizer", "prep", "pretrain", "generate", "eval"]


def _module_target(config_path):
    """The `<module.py>` named on a config file's own first-line comment
    (e.g. "systems/pretraining/data_prep.py"), or None if that convention
    isn't followed (every config in this project does, as of 2026-09-16)."""
    with open(config_path, encoding="utf-8") as f:
        first_line = f.readline()
    m = _MODULE_LINE_RE.match(first_line)
    return m.group(1) if m else None


def _load_yaml(config_path):
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _latest_by_mtime(paths):
    return max(paths, key=os.path.getmtime) if paths else None


def check_tokenizer_training():
    """Static per-system checkpoint glob (checkpoints/{pattern}) -- NOT
    derived from configs/train_tokenizer/, since magnet/manta have no config
    there at all (see module docstring). Independent of any config file."""
    rows = []
    for name, pattern in sorted(_CHECKPOINT_GLOBS.items()):
        matches = glob.glob(os.path.join("checkpoints", pattern))
        if not matches:
            rows.append({
                "name": name, "config": None, "status": "missing",
                "detail": f"no checkpoints/{pattern}",
            })
            continue
        latest = _latest_by_mtime(matches)
        rows.append({
            "name": name, "config": None, "status": "done",
            "detail": f"{len(matches)} checkpoint(s), latest {latest}",
        })
    return rows


def _check_prep(cfg):
    output_dir = cfg.get("output_dir")
    if not output_dir:
        return "unknown", "no output_dir key in config"
    meta_path = os.path.join(output_dir, "shards_meta.json")
    if os.path.exists(meta_path):
        try:
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
            detail = (
                f"{meta_path} exists ({len(meta.get('shard_files', []))} shards, "
                f"{meta.get('total_tokens', '?'):,} tokens)"
                if isinstance(meta.get("total_tokens"), int)
                else f"{meta_path} exists"
            )
        except (json.JSONDecodeError, OSError):
            detail = f"{meta_path} exists but is unreadable/corrupt"
        return "done", detail
    checkpoint_path = os.path.join(output_dir, "prep_checkpoint.json")
    if os.path.exists(checkpoint_path):
        return "in_progress", f"{checkpoint_path} exists (resumable), shards_meta.json not yet written"
    if os.path.isdir(output_dir):
        return "in_progress", f"{output_dir} exists but has neither shards_meta.json nor prep_checkpoint.json"
    return "missing", f"{output_dir} does not exist"


def _check_pretrain(cfg):
    output_dir = cfg.get("output_dir", "checkpoints/pretrain")
    final_path = os.path.join(output_dir, "final.pt")
    if os.path.exists(final_path):
        return "done", f"{final_path} exists"
    steps = []
    if os.path.isdir(output_dir):
        for name in os.listdir(output_dir):
            m = re.match(r"step_(\d+)\.pt$", name)
            if m:
                steps.append(int(m.group(1)))
    if steps:
        latest_step = max(steps)
        total_steps = cfg.get("total_steps")
        of_total = f"/{total_steps}" if total_steps else ""
        return "in_progress", f"latest checkpoint step_{latest_step}.pt{of_total} in {output_dir}"
    return "missing", f"{output_dir} has neither final.pt nor any step_*.pt"


def _check_output_file(cfg, key="output"):
    output = cfg.get(key)
    if not output:
        return "unknown", f"no {key!r} key in config (results only printed to stdout, not saved)"
    if os.path.exists(output):
        return "done", f"{output} exists"
    return "missing", f"{output} does not exist"


# Each multi-system wrapper script writes {output_dir}/{name}_{suffix}.json
# per system -- an EXACT suffix per wrapper, not a glob: a loose
# "{name}_*.json" glob would false-positive on an unrelated pre-existing
# file that happens to share the {name}_ prefix (confirmed live: every
# _comparison.json from this project's normal per-system eval runs already
# matches "{name}_*.json", so morphology_all.yml's own per-system status
# would otherwise always read DONE even before it's ever been run).
_WRAPPER_SUFFIXES = {
    "scripts/evaluate_own_tokenizers_indigenous_panel.py": "indigenous_panel",
    "scripts/evaluate_morphology_all.py": "morphology",
}


def _check_wrapper(cfg, suffix):
    """Multi-system driver configs (scripts/evaluate_own_tokenizers_indigenous_panel.py,
    scripts/evaluate_morphology_all.py): {systems: [{name, ...}], output_dir,
    combined_output}. `suffix` is this wrapper's own exact per-system output
    suffix (see _WRAPPER_SUFFIXES) -- e.g. {output_dir}/{name}_morphology.json,
    checked for EXACT existence, not a glob."""
    output_dir = cfg.get("output_dir", "results")
    combined_output = cfg.get("combined_output")
    systems = cfg.get("systems", [])
    per_system = {}
    for entry in systems:
        name = entry["name"]
        path = os.path.join(output_dir, f"{name}_{suffix}.json")
        per_system[name] = os.path.basename(path) if os.path.exists(path) else None
    n_done = sum(1 for v in per_system.values() if v)
    if combined_output and os.path.exists(combined_output):
        status = "done"
        detail = f"{combined_output} exists ({n_done}/{len(systems)} systems)"
    elif n_done:
        status = "in_progress"
        detail = f"{n_done}/{len(systems)} systems done, combined output not yet written"
    else:
        status = "missing"
        detail = f"0/{len(systems)} systems done"
    return status, detail, per_system


# module path -> (stage label, checker). Checkers either return (status,
# detail) or (status, detail, per_system_breakdown) for wrapper configs.
_MODULE_DISPATCH = {
    "systems/pretraining/data_prep.py": ("prep", _check_prep),
    "systems/pretraining/cli.py": ("pretrain", _check_pretrain),
    "systems/pretraining/encoder_cli.py": ("pretrain", _check_pretrain),
    "systems/pretraining/cli_generate.py": ("generate", _check_output_file),
    "systems/pretraining/cli_eval.py": ("eval", _check_output_file),
    "systems/pretraining/cli_lm_eval.py": ("eval", _check_output_file),
}


def _classify_module(module):
    """Dispatch a config's own first-line module path to (stage, checker).
    train_tokenizer configs are intentionally NOT routed here -- see
    check_tokenizer_training's own docstring for why that stage is checked
    independently of any config file."""
    if module in _MODULE_DISPATCH:
        return _MODULE_DISPATCH[module]
    if re.match(r"^systems/tokenization/[^/]+/cli\.py$", module):
        return "train_tokenizer", None  # handled statically, see check_tokenizer_training
    if re.match(r"^systems/tokenization/[^/]+/evaluate\.py$", module):
        return "eval", _check_output_file
    if module in _WRAPPER_SUFFIXES:
        suffix = _WRAPPER_SUFFIXES[module]
        return "eval", lambda cfg: _check_wrapper(cfg, suffix)
    return "unknown", None


def check_configs(config_dir="configs", stages=None):
    """Walks every configs/*/*.yml, classifies it via its own first-line
    module comment, and returns one row per config (skipping
    train_tokenizer's own configs -- see check_tokenizer_training)."""
    rows = []
    for path in sorted(glob.glob(os.path.join(config_dir, "*", "*.yml"))):
        module = _module_target(path)
        if module is None:
            rows.append({"name": path, "config": path, "status": "unknown", "detail": "no `# module.py --` header line found"})
            continue
        stage, checker = _classify_module(module)
        if stage == "train_tokenizer" or checker is None:
            continue  # tokenizer training handled statically; genuinely unroutable configs skipped
        if stages and stage not in stages:
            continue
        cfg = _load_yaml(path) or {}
        result = checker(cfg)
        if len(result) == 3:
            status, detail, per_system = result
            rows.append({
                "name": os.path.basename(path), "config": path, "stage": stage,
                "status": status, "detail": detail, "per_system": per_system,
            })
        else:
            status, detail = result
            rows.append({"name": os.path.basename(path), "config": path, "stage": stage, "status": status, "detail": detail})
    return rows


_STATUS_LABEL = {"done": "DONE", "in_progress": "IN_PROGRESS", "missing": "MISSING", "unknown": "UNKNOWN"}


def _print_report(rows_by_stage, stages):
    for stage in _STAGE_ORDER:
        if stages and stage not in stages:
            continue
        rows = rows_by_stage.get(stage, [])
        if not rows:
            continue
        print(f"=== {stage} ===")
        for row in rows:
            label = _STATUS_LABEL[row["status"]]
            print(f"  [{label:11}] {row['name']}: {row['detail']}")
            if row.get("per_system"):
                for name, latest in sorted(row["per_system"].items()):
                    sub_label = "DONE" if latest else "MISSING"
                    print(f"      - {name}: {sub_label}" + (f" ({latest})" if latest else ""))
        print()


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config-dir", type=str, default="configs")
    parser.add_argument(
        "--stage", nargs="+", choices=_STAGE_ORDER, default=None,
        help="only report these stage(s) (default: all)",
    )
    parser.add_argument("--json", action="store_true", help="machine-readable output instead of the printed report")
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    stages = set(args.stage) if args.stage else None

    rows = []
    if not stages or "train_tokenizer" in stages:
        for row in check_tokenizer_training():
            rows.append({**row, "stage": "train_tokenizer"})
    rows.extend(check_configs(args.config_dir, stages))

    if args.json:
        print(json.dumps(rows, indent=2))
        return rows

    rows_by_stage = {}
    for row in rows:
        rows_by_stage.setdefault(row["stage"], []).append(row)
    _print_report(rows_by_stage, stages)

    n_missing = sum(1 for r in rows if r["status"] == "missing")
    n_in_progress = sum(1 for r in rows if r["status"] == "in_progress")
    print(f"{len(rows)} experiment(s) checked: {n_missing} missing, {n_in_progress} in progress.")
    return rows


if __name__ == "__main__":
    main()
