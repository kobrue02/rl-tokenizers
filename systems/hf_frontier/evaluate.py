"""Held-out evaluation for one or more arbitrary HuggingFace frontier models'
own tokenizers (--hf-repo-id, loaded TOKENIZER-ONLY -- see model.py's own
docstring) -- mirrors every other systems/*/evaluate.py's shape so a real
frontier tokenizer scores on IDENTICAL held-out data (BOUQuET) with the SAME
metrics as fairtok/magnet/flexitokens/manta/fanta/superbpe/bpe. Like
bpe/superbpe/flexitokens/manta, a frontier tokenizer's own encode() takes no
language argument, so ONE induce_fn covers every language (no MAGNET-style
per-script resolution needed).

--hf-repo-id takes a COMMA-SEPARATED list (same convention pretraining.
cli_eval's own --benchmark uses) -- one job scores every listed model
against the SAME loaded eval_groups (loaded ONCE, not re-downloaded per
model) and writes one combined JSON, rather than one sbatch call per model.
--trust-remote-code/--hf-token apply to the WHOLE list uniformly (passing a
token to a repo that doesn't need one, or --trust-remote-code to a repo
that doesn't require custom code, is a harmless no-op) -- if different
repos in a real comparison genuinely need different tokens, run them as
separate invocations instead.
"""

import json

from common.config_file import parse_args_with_config
from common.eval_common import evaluate_on_groups, report_eval
from common.oldi_data import load_bouquet_dev, load_bouquet_test
from systems.bpe.train import _SMOKE_TEST_GROUPS

from .model import HFFrontierTokenizer
from .segment import induce_spans


def build_arg_parser():
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate one or more HuggingFace repos' own tokenizers on held-out data "
        "(loaded TOKENIZER ONLY -- no model weights are ever downloaded)."
    )
    parser.add_argument(
        "--hf-repo-id", type=str, required=True,
        help="comma-separated HuggingFace model repos, e.g. deepseek-ai/DeepSeek-V4-Pro,"
        "moonshotai/Kimi-K3,meta-llama/Llama-3.1-8B-Instruct -- only each repo's tokenizer "
        "is downloaded/loaded",
    )
    parser.add_argument(
        "--trust-remote-code", action="store_true",
        help="required by some repos (e.g. moonshotai/Kimi-K3, which ships its own tokenizer "
        "class) to load at all -- executes that repo's own Python code; off by default, "
        "see model.py's own docstring before turning this on; applies to every repo in "
        "--hf-repo-id uniformly",
    )
    parser.add_argument(
        "--hf-token", type=str, default=None,
        help="explicit HF access token -- only needed for GATED repos (e.g. "
        "meta-llama/Llama-3.1-8B-Instruct, which also needs its license accepted on "
        "huggingface.co first); falls back to HF_TOKEN / a prior huggingface-cli login; "
        "applies to every repo in --hf-repo-id uniformly",
    )
    parser.add_argument(
        "--eval-data-source",
        choices=["bouquet", "bouquet_test", "synthetic"],
        default="bouquet",
        help="'bouquet' (default): BOUQuET DEV, for tuning/exploratory comparisons; "
        "'bouquet_test': BOUQuET TEST, the genuinely held-out split -- reserve for final "
        "reported numbers, not repeated tuning checks; "
        "'synthetic': a small real-text placeholder (reuses systems.bpe.train's own "
        "_SMOKE_TEST_GROUPS -- NOT common.data's byte generator, which isn't guaranteed "
        "valid UTF-8), for a quick sanity check with no BOUQuET network access",
    )
    parser.add_argument(
        "--num-groups", type=int, default=None,
        help="cap the number of held-out groups scored; omit for the full set",
    )
    parser.add_argument("--output", type=str, default=None, help="write combined JSON results here (default: print to stdout)")
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument(
        "--wandb-project", type=str, default="hf_frontier",
        help="own project, separate from the systems/*/train.py per-system convention "
        "(fanta->'fanta', bpe->'bpe', ...) and from pretraining's own 'pretraining' project -- "
        "this compares EXTERNAL tokenizers, not one this project trained itself",
    )
    parser.add_argument("--run-name", type=str, default="")
    return parser


def _load_eval_groups(args):
    if args.eval_data_source == "synthetic":
        groups = _SMOKE_TEST_GROUPS
        return groups[: args.num_groups] if args.num_groups else groups
    loader = load_bouquet_test if args.eval_data_source == "bouquet_test" else load_bouquet_dev
    groups = loader("all")
    if args.num_groups:
        groups = groups[: args.num_groups]
    return groups


def _evaluate_one(repo_id, eval_groups, trust_remote_code, hf_token):
    wrapped = HFFrontierTokenizer.load(repo_id, trust_remote_code=trust_remote_code, hf_token=hf_token)
    print(f"\nhf_repo_id={repo_id} span_method={wrapped.span_method} native_vocab_size={wrapped.vocab_size}")

    # ONE induce_fn for every language -- frontier tokenizers' own encode()
    # takes no language argument, same as bpe/superbpe/flexitokens/manta.
    induce_fn_by_lang = {
        lang: (lambda raw, w=wrapped: induce_spans(w, raw))
        for group in eval_groups
        for lang in group
    }
    results = evaluate_on_groups(induce_fn_by_lang, eval_groups)
    report_eval(results, label=repo_id)
    return wrapped, results


def main(argv=None):
    args = parse_args_with_config(build_arg_parser(), argv)
    repo_ids = [r.strip() for r in args.hf_repo_id.split(",")]

    eval_groups = _load_eval_groups(args)
    print(f"eval_data_source={args.eval_data_source} groups={len(eval_groups)} repos={repo_ids}")

    all_results = {}
    all_wrapped = {}
    for repo_id in repo_ids:
        wrapped, results = _evaluate_one(repo_id, eval_groups, args.trust_remote_code, args.hf_token)
        all_wrapped[repo_id] = wrapped
        # token_freq is {lang: Counter[bytes, int]} -- bytes keys aren't
        # valid JSON, and aren't needed for the summary metrics this writes
        # out (report_eval's own printed output already covers them);
        # excluded here rather than left to crash json.dumps below on a
        # real multi-repo run (confirmed directly: an earlier version of
        # this file's own sibling, pretraining.cli_eval, hit the identical
        # class of bug with tuple-keyed dicts -- checked for it here before
        # shipping, not after).
        all_results[repo_id] = {k: v for k, v in results.items() if k != "token_freq"}

    payload = json.dumps(all_results, indent=2)
    if args.output:
        with open(args.output, "w") as f:
            f.write(payload)
        print(f"\nwrote combined results to {args.output}")

    if args.use_wandb:
        import wandb

        run = wandb.init(
            project=args.wandb_project,
            name=args.run_name or None,
            job_type="eval",
            config={
                "hf_repo_ids": repo_ids,
                "trust_remote_code": args.trust_remote_code,
                "eval_data_source": args.eval_data_source,
                "num_groups": args.num_groups,
            },
        )
        summary_rows = [
            [repo_id, all_wrapped[repo_id].span_method, all_wrapped[repo_id].vocab_size,
             r["avg_compression"], r["gini"]]
            for repo_id, r in all_results.items()
        ]
        run.log(
            {
                "comparison": wandb.Table(
                    columns=["hf_repo_id", "span_method", "native_vocab_size", "avg_compression", "gini"],
                    data=summary_rows,
                ),
            }
        )
        for repo_id, r in all_results.items():
            run.log(
                {
                    f"{repo_id}/avg_compression": r["avg_compression"],
                    f"{repo_id}/gini": r["gini"],
                    **{f"{repo_id}/renyi/{lang}": v for lang, v in r["renyi"].items()},
                    **{f"{repo_id}/compression/{lang}": v for lang, v in r["per_lang_compression"].items()},
                    **{f"{repo_id}/fertility/{lang}": v for lang, v in r["fertility"].items()},
                }
            )
        run.finish()
        print(f"logged comparison to wandb project={args.wandb_project!r}")

    return all_results


def run_smoke_test():
    """Mirrors every other systems/*/evaluate.py's own testing convention,
    with one real difference stated plainly: this module has no local,
    network-free path at all (unlike a from-scratch checkpoint, there is no
    "trained model" to construct without an HF call) -- gpt2 is used here
    specifically because it's small, fast, ungated, and always available,
    not because it's the recommended default for a real comparison run.
    Verified against real synthetic (non-network BOUQuET) eval data, plus
    a direct round-trip assertion on the span reconstruction itself so a
    regression in model.py's own logic fails LOUDLY here, not just as a
    quietly-wrong downstream metric. Also exercises the multi-repo/--output
    JSON path directly (gpt2 twice under different labels -- no need for a
    second real network-distinct tokenizer just to prove the fan-out and
    JSON serialization both work), including a real json.dumps call, which
    is exactly the step that would catch a tuple/bytes-key regression."""
    import tempfile
    import os

    from .model import _CANARY_TEXT, HFFrontierTokenizer

    wrapped = HFFrontierTokenizer.load("gpt2")
    assert wrapped.span_method == "byte_level"
    spans = wrapped.induce_spans(_CANARY_TEXT)
    assert b"".join(spans) == _CANARY_TEXT.encode("utf-8"), "span reconstruction did not round-trip"

    with tempfile.TemporaryDirectory() as d:
        out_path = os.path.join(d, "results.json")
        all_results = main(["--hf-repo-id", "gpt2,gpt2", "--eval-data-source", "synthetic", "--output", out_path])
        assert set(all_results) == {"gpt2"}, "comma-separated dupes should collapse to one dict key, as expected"
        with open(out_path) as f:
            reloaded = json.load(f)
        assert reloaded == all_results, "the JSON file written to --output must match what main() returned"

    print("\nhf_frontier smoke test passed.")
    return all_results


if __name__ == "__main__":
    main()
