"""Held-out evaluation for one or more arbitrary HuggingFace frontier
tokenizers (--hf-repo-id, loaded TOKENIZER-ONLY, see model.py) -- mirrors
every other systems/*/evaluate.py's shape so a frontier tokenizer scores on
the same held-out data (BOUQuET) with the same metrics as every other
systems/ tokenizer. Like bpe/superbpe/flexitokens/manta, a frontier
tokenizer's encode() takes no language argument, so one induce_fn covers
every language.

--hf-repo-id takes a COMMA-SEPARATED list -- one job scores every listed
model against the same loaded eval_groups (loaded once) and writes one
combined JSON, instead of one sbatch call per model. --trust-remote-code/
--hf-token apply uniformly to the whole list (harmless no-op for repos that
don't need them); run separate invocations if different repos need
different tokens.

A single repo's failure (gated access, an unhandled tokenizer scheme,
network error) does NOT abort the run -- main() catches per-repo and
records it under a "_failed" key (only present if something failed) rather
than losing every other repo's completed results.
"""

import json

from common.config_file import parse_args_with_config
from common.eval.cross_tokenizer import (
    evaluate_on_groups,
    evaluate_on_indigenous_panel,
    report_eval,
    report_indigenous_panel_eval,
)
from common.data.corpora import stream_groups
from common.data.oldi_data import load_bouquet_dev, load_bouquet_test
from systems.tokenization.bpe.train import _SMOKE_TEST_GROUPS

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
        help="required by some repos (e.g. Kimi-K3, custom tokenizer class) to load at all -- "
        "executes that repo's own Python code; off by default; applies to all --hf-repo-id entries",
    )
    parser.add_argument(
        "--hf-token", type=str, default=None,
        help="explicit HF access token -- only needed for gated repos (e.g. Llama-3.1-8B-Instruct, "
        "also needs license acceptance on huggingface.co); falls back to HF_TOKEN / "
        "huggingface-cli login; applies to all --hf-repo-id entries",
    )
    parser.add_argument(
        "--eval-data-source",
        choices=["bouquet", "bouquet_test", "synthetic", "indigenous_panel"],
        default="bouquet",
        help="'bouquet' (default): BOUQuET DEV, for tuning/exploratory comparisons; "
        "'bouquet_test': BOUQuET TEST, the held-out split -- use for final reported numbers only; "
        "'synthetic': small real-text placeholder (systems.tokenization.bpe.train's _SMOKE_TEST_GROUPS, not "
        "common.data.synthetic's byte generator which isn't guaranteed valid UTF-8), for a quick "
        "sanity check with no network access; "
        "'indigenous_panel': common.data.indigenous_panel's curated panel (needs a one-time "
        "common.data.prepare_indigenous_panel run first) -- scored via evaluate_on_indigenous_panel "
        "(mixed-anchor), not evaluate_on_groups; results have a different shape, not comparable "
        "to a bouquet/bouquet_test/synthetic run's",
    )
    parser.add_argument(
        "--num-groups", type=int, default=None,
        help="cap the number of held-out groups scored; omit for the full set",
    )
    parser.add_argument("--output", type=str, default=None, help="write combined JSON results here (default: print to stdout)")
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument(
        "--wandb-project", type=str, default="hf_frontier",
        help="own project, separate from the systems/*/train.py per-system convention -- "
        "this compares EXTERNAL tokenizers, not ones trained by this project",
    )
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


def _evaluate_one(repo_id, eval_groups, trust_remote_code, hf_token, indigenous_panel=False):
    wrapped = HFFrontierTokenizer.load(repo_id, trust_remote_code=trust_remote_code, hf_token=hf_token)
    print(f"\nhf_repo_id={repo_id} span_method={wrapped.span_method} native_vocab_size={wrapped.vocab_size}")

    # One induce_fn for every language -- frontier tokenizers' encode() takes
    # no language argument, same as bpe/superbpe/flexitokens/manta.
    induce_fn_by_lang = {
        lang: (lambda raw, w=wrapped: induce_spans(w, raw))
        for group in eval_groups
        for lang in group
    }
    if indigenous_panel:
        results = evaluate_on_indigenous_panel(induce_fn_by_lang, eval_groups)
        report_indigenous_panel_eval(results, label=repo_id)
    else:
        results = evaluate_on_groups(induce_fn_by_lang, eval_groups)
        report_eval(results, label=repo_id)
    return wrapped, results


def main(argv=None):
    args = parse_args_with_config(build_arg_parser(), argv)
    repo_ids = [r.strip() for r in args.hf_repo_id.split(",")]

    eval_groups = _load_eval_groups(args)
    print(f"eval_data_source={args.eval_data_source} groups={len(eval_groups)} repos={repo_ids}")

    is_indigenous_panel = args.eval_data_source == "indigenous_panel"
    all_results = {}
    all_wrapped = {}
    failed = {}
    for repo_id in repo_ids:
        try:
            wrapped, results = _evaluate_one(
                repo_id, eval_groups, args.trust_remote_code, args.hf_token,
                indigenous_panel=is_indigenous_panel,
            )
        except Exception as e:
            # One bad repo must not lose every other repo's completed results.
            print(f"\n[{repo_id}] FAILED: {type(e).__name__}: {e}")
            failed[repo_id] = f"{type(e).__name__}: {e}"
            continue
        all_wrapped[repo_id] = wrapped
        # token_freq is {lang: Counter[bytes, int]} -- bytes keys aren't
        # valid JSON and aren't needed for the summary (report_eval /
        # report_indigenous_panel_eval already printed it); stripped here
        # rather than crashing json.dumps below. indigenous_panel's results
        # nest token_freq inside "combined" and inside each anchor entry of
        # "token_parity_by_anchor", so both need stripping.
        if is_indigenous_panel:
            all_results[repo_id] = {
                "combined": {k: v for k, v in results["combined"].items() if k != "token_freq"},
                "token_parity_by_anchor": {
                    anchor: {k: v for k, v in anchor_results.items() if k != "token_freq"}
                    for anchor, anchor_results in results["token_parity_by_anchor"].items()
                },
                "morphology_spread": results["morphology_spread"],
            }
        else:
            all_results[repo_id] = {k: v for k, v in results.items() if k != "token_freq"}

    if failed:
        print(f"\n{len(failed)}/{len(repo_ids)} repo(s) failed: {list(failed)} -- see FAILED lines above for why")
        all_results["_failed"] = failed  # only added when non-empty

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
                "failed_repos": list(failed),
            },
        )
        successful = {r: res for r, res in all_results.items() if r != "_failed"}
        # indigenous_panel nests avg_compression/gini/renyi/per_lang_compression/
        # fertility under "combined"; everything else has them at the top level.
        summary = (lambda r: r["combined"]) if is_indigenous_panel else (lambda r: r)
        summary_rows = [
            [repo_id, all_wrapped[repo_id].span_method, all_wrapped[repo_id].vocab_size,
             summary(r)["avg_compression"], summary(r)["gini"]]
            for repo_id, r in successful.items()
        ]
        run.log(
            {
                "comparison": wandb.Table(
                    columns=["hf_repo_id", "span_method", "native_vocab_size", "avg_compression", "gini"],
                    data=summary_rows,
                ),
            }
        )
        for repo_id, r in successful.items():
            c = summary(r)
            log_dict = {
                f"{repo_id}/avg_compression": c["avg_compression"],
                f"{repo_id}/gini": c["gini"],
                **{f"{repo_id}/renyi/{lang}": v for lang, v in c["renyi"].items()},
                **{f"{repo_id}/compression/{lang}": v for lang, v in c["per_lang_compression"].items()},
                **{f"{repo_id}/fertility/{lang}": v for lang, v in c["fertility"].items()},
            }
            if is_indigenous_panel:
                log_dict.update(
                    {f"{repo_id}/morphology_spread/{k}": v for k, v in r["morphology_spread"].items()}
                )
                for anchor, anchor_results in r["token_parity_by_anchor"].items():
                    log_dict.update(
                        {
                            f"{repo_id}/token_parity_vs_{anchor}/{lang}": v
                            for lang, v in anchor_results["token_parity"].items()
                            if lang != anchor
                        }
                    )
            run.log(log_dict)
        run.finish()
        print(f"logged comparison to wandb project={args.wandb_project!r}")

    return all_results


def run_smoke_test():
    """Unlike other systems/*/evaluate.py smoke tests, this module has no
    local network-free path (there's no "trained model" to construct
    without an HF call) -- gpt2 is used since it's small, fast, and always
    available, not as a recommended default. Checks: a direct round-trip
    assertion on span reconstruction (so a model.py regression fails
    loudly here, not as a quiet downstream metric), the multi-repo/--output
    JSON path (gpt2 twice under different labels, including a real
    json.dumps to catch tuple/bytes-key regressions), and per-repo error
    isolation (a nonexistent repo alongside a real one lands under
    "_failed" without losing gpt2's result)."""
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
        gpt2_result = all_results["gpt2"]
        assert set(gpt2_result["token_parity_gm"]) == set(gpt2_result["token_parity"])
        assert gpt2_result["token_parity_spread"] >= 1.0

        # Anchor-invariance against a real tokenizer: re-run with anchor_lang="deu" instead
        # of "eng" and confirm token_parity_gm is identical (see
        # common.eval.parity.anchor_invariant_parity).
        from common.eval.cross_tokenizer import evaluate_on_groups
        from systems.tokenization.bpe.train import _SMOKE_TEST_GROUPS

        induce_fn = {"eng": wrapped.induce_spans, "deu": wrapped.induce_spans}
        eng_anchored = evaluate_on_groups(induce_fn, _SMOKE_TEST_GROUPS, anchor_lang="eng")
        deu_anchored = evaluate_on_groups(induce_fn, _SMOKE_TEST_GROUPS, anchor_lang="deu")
        for lang in eng_anchored["token_parity_gm"]:
            assert abs(eng_anchored["token_parity_gm"][lang] - deu_anchored["token_parity_gm"][lang]) < 1e-9, (
                f"token_parity_gm[{lang!r}] must be anchor-invariant, "
                f"got {eng_anchored['token_parity_gm'][lang]} (eng-anchored) vs "
                f"{deu_anchored['token_parity_gm'][lang]} (deu-anchored)"
            )
        assert abs(eng_anchored["token_parity_spread"] - deu_anchored["token_parity_spread"]) < 1e-9
        with open(out_path) as f:
            reloaded = json.load(f)
        assert reloaded == all_results, "the JSON file written to --output must match what main() returned"

        mixed_results = main([
            "--hf-repo-id", "gpt2,this-repo-genuinely-does-not-exist/nope",
            "--eval-data-source", "synthetic",
        ])
        assert set(mixed_results) == {"gpt2", "_failed"}, "a bad repo must not take gpt2's real result down with it"
        assert "this-repo-genuinely-does-not-exist/nope" in mixed_results["_failed"]
        assert mixed_results["gpt2"] == all_results["gpt2"], "the good repo's result must be unaffected by the bad one"

    print("\nhf_frontier smoke test passed.")
    return all_results


if __name__ == "__main__":
    main()
