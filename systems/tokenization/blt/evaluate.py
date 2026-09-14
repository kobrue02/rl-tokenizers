"""Held-out evaluation of BLT's dynamic entropy-based patch boundaries
(model.py/segment.py) on the same held-out data (BOUQuET) and metrics as
every other systems/*/evaluate.py, so it's directly comparable in the
"token tax" comparison. Like bpe/superbpe/flexitokens/manta/hf_frontier, BLT's
patcher takes no language argument, so one induce_fn covers every language.

UNLIKE hf_frontier's --hf-repo-id (a comma-separated LIST of arbitrary
tokenizers), this evaluates exactly ONE fixed system (facebook/blt-1b's
entropy model) -- there's nothing to parameterize by repo id, so no
--trust-remote-code/--hf-token/multi-repo machinery here.

PERFORMANCE NOTE (read before running the full BOUQuET test split): unlike
every other tokenizer here, this runs a real ~100M-param transformer forward
pass per sentence (~30-40ms/sentence on CPU, measured), not free
tokenization -- a full bouquet_test run (~272k sentences across 259
languages) is on the order of HOURS, not seconds. Use --num-groups for a
quick sanity check or a capped preliminary run.
"""

import json

from common.config_file import parse_args_with_config
from common.data.corpora import stream_groups
from common.data.oldi_data import load_bouquet_dev, load_bouquet_test
from common.eval.cross_tokenizer import (
    evaluate_on_groups,
    evaluate_on_indigenous_panel,
    report_eval,
    report_indigenous_panel_eval,
)
from systems.tokenization.bpe.train import _SMOKE_TEST_GROUPS

from .model import load_entropy_model
from .segment import induce_spans


def build_arg_parser():
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate BLT's entropy-based dynamic patch boundaries "
        "(facebook/blt-1b's entropy model) on held-out data."
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="torch device for the entropy model's forward pass; this project's own "
        "development machine has no CUDA, so 'cpu' is the tested default",
    )
    parser.add_argument(
        "--eval-data-source",
        choices=["bouquet", "bouquet_test", "synthetic", "indigenous_panel"],
        default="bouquet",
        help="same choices/meaning as every other systems/*/evaluate.py -- see e.g. "
        "hf_frontier/evaluate.py's own --eval-data-source help",
    )
    parser.add_argument(
        "--num-groups", type=int, default=None,
        help="cap the number of held-out groups scored -- see module docstring's "
        "performance note before omitting this for a full bouquet_test run",
    )
    parser.add_argument("--output", type=str, default=None, help="write results here (default: print to stdout)")
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="blt")
    parser.add_argument("--run-name", type=str, default="")
    return parser


def _load_eval_groups(args):
    if args.eval_data_source == "synthetic":
        groups = _SMOKE_TEST_GROUPS
        return groups[: args.num_groups] if args.num_groups else groups
    if args.eval_data_source == "indigenous_panel":
        groups = list(stream_groups("indigenous_panel", config="all"))
        return groups[: args.num_groups] if args.num_groups else groups
    loader = load_bouquet_test if args.eval_data_source == "bouquet_test" else load_bouquet_dev
    groups = loader("all")
    if args.num_groups:
        groups = groups[: args.num_groups]
    return groups


def main(argv=None):
    args = parse_args_with_config(build_arg_parser(), argv)
    eval_groups = _load_eval_groups(args)
    is_indigenous_panel = args.eval_data_source == "indigenous_panel"
    print(f"eval_data_source={args.eval_data_source} groups={len(eval_groups)} device={args.device}")

    model = load_entropy_model(device=args.device)
    print(f"loaded facebook/blt-1b entropy model: {model.num_parameters():,} parameters")

    induce_fn_by_lang = {
        lang: (lambda raw, m=model, d=args.device: induce_spans(m, raw, device=d))
        for group in eval_groups
        for lang in group
    }

    if is_indigenous_panel:
        results = evaluate_on_indigenous_panel(induce_fn_by_lang, eval_groups)
        report_indigenous_panel_eval(results, label="blt")
        # token_freq isn't valid JSON (bytes keys) and isn't needed for the
        # summary (report_indigenous_panel_eval already printed it) -- same
        # stripping as hf_frontier/evaluate.py.
        clean_results = {
            "combined": {k: v for k, v in results["combined"].items() if k != "token_freq"},
            "token_parity_by_anchor": {
                anchor: {k: v for k, v in anchor_results.items() if k != "token_freq"}
                for anchor, anchor_results in results["token_parity_by_anchor"].items()
            },
            "morphology_spread": results["morphology_spread"],
        }
    else:
        results = evaluate_on_groups(induce_fn_by_lang, eval_groups)
        report_eval(results, label="blt")
        clean_results = {k: v for k, v in results.items() if k != "token_freq"}

    all_results = {"facebook/blt-1b": clean_results}
    payload = json.dumps(all_results, indent=2)
    if args.output:
        with open(args.output, "w") as f:
            f.write(payload)
        print(f"\nwrote results to {args.output}")

    if args.use_wandb:
        import wandb

        run = wandb.init(
            project=args.wandb_project,
            name=args.run_name or None,
            job_type="eval",
            config={
                "eval_data_source": args.eval_data_source,
                "num_groups": args.num_groups,
                "device": args.device,
                "num_parameters": model.num_parameters(),
            },
        )
        summary = clean_results["combined"] if is_indigenous_panel else clean_results
        log_dict = {
            "blt/avg_compression": summary["avg_compression"],
            "blt/gini": summary["gini"],
            **{f"blt/renyi/{lang}": v for lang, v in summary["renyi"].items()},
            **{f"blt/compression/{lang}": v for lang, v in summary["per_lang_compression"].items()},
            **{f"blt/fertility/{lang}": v for lang, v in summary["fertility"].items()},
        }
        if is_indigenous_panel:
            log_dict.update(
                {f"blt/morphology_spread/{k}": v for k, v in clean_results["morphology_spread"].items()}
            )
        run.log(log_dict)
        run.finish()
        print(f"logged to wandb project={args.wandb_project!r}")

    return all_results


def run_smoke_test():
    """Real network+weights call (facebook/blt-1b's entropy checkpoint,
    ~200MB, gated -- needs prior HF access approval and login), unlike most
    other systems' smoke tests -- there's no local, network-free path since
    there's no "trained model" to construct without downloading the real
    checkpoint. Kept small (synthetic data, a handful of groups) specifically
    because of this module's real per-sentence compute cost -- see the
    module docstring's performance note."""
    from .model import load_entropy_model
    from .segment import induce_spans

    model = load_entropy_model()
    text = "The quick brown fox jumps über den Zaun. 你好世界 🎉"
    spans = induce_spans(model, text)
    assert b"".join(spans) == text.encode("utf-8"), "patch reconstruction did not round-trip"
    assert len(spans) < len(text.encode("utf-8")), "at least some bytes should merge into multi-byte patches"

    all_results = main(["--eval-data-source", "synthetic", "--num-groups", "2"])
    assert set(all_results) == {"facebook/blt-1b"}
    result = all_results["facebook/blt-1b"]
    assert result["avg_compression"] >= 1.0
    assert 0.0 <= result["gini"] <= 1.0

    print("\nblt smoke test passed.")
    return all_results


if __name__ == "__main__":
    main()
