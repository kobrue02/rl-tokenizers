"""Command-line entry point for running EleutherAI/lm-evaluation-harness's
own registered tasks against a systems.pretraining.train checkpoint, via
lm_eval_adapter.ThesisLM -- the "use the standard harness" counterpart to
cli_eval.py's own hand-rolled xnli/xcopa/blimp implementation (see
lm_eval_adapter.py's own module docstring for the full list of tasks
confirmed usable here -- xnli, xcopa, blimp, xstorycloze,
lambada_multilingual, global_piqa -- and exactly why, plus why flores_mt/
cola/squad stay on cli_eval.py's own path).

Usage:
    python3 -m systems.pretraining.cli_lm_eval --checkpoint checkpoints/pretrain/final.pt \\
        --system bpe --tokenizer-checkpoint checkpoints/bpe_12345.json \\
        --tasks xnli,xcopa,blimp --output results/harness_bpe.json

    python3 -m systems.pretraining.cli_lm_eval --checkpoint checkpoints/pretrain/final.pt \\
        --system fanta --tokenizer-checkpoint checkpoints/fanta_12345.pt \\
        --vocab-json vocab_out/fanta_vocab_12345.json \\
        --tasks xnli_sw,xcopa_et --num-fewshot 8 --limit 200 \\
        --output results/harness_fanta_fewshot.json
        # "xnli"/"xcopa"/"blimp" (bare) are real task GROUPS the harness
        # itself expands to every one of their per-language/per-paradigm
        # subtasks with a size-weighted aggregate metric computed
        # automatically -- pass a specific "xnli_sw"-style name instead to
        # run just one language/paradigm. --num-fewshot applies uniformly
        # to every task in --tasks (the harness's own convention, unlike
        # cli_eval.py's per-benchmark --num-fewshot handling).

    python3 -m systems.pretraining.cli_lm_eval --checkpoint checkpoints/pretrain/final.pt \\
        --system bpe --tokenizer-checkpoint checkpoints/bpe_12345.json \\
        --tasks xstorycloze,lambada_multilingual,global_piqa_nonparallel_cloze \\
        --output results/harness_bpe_extra.json
        # xstorycloze (11 langs)/lambada_multilingual (5 langs) are simple
        # 2-way-choice / plain-loglikelihood tasks respectively; global_piqa's
        # OWN README explicitly recommends the "cloze" config (not
        # "generation") for base/small non-instruction-tuned models like
        # these checkpoints -- confirmed by reading it directly, so
        # "_generation" variants aren't used here.

--output's JSON is the harness's own native `results` dict (per-task
accuracy/acc_norm + stderr, already bootstrap-derived -- see lm_eval's own
--bootstrap-iters), wrapped the same {"label", "checkpoint", "system",
"tokenizer_checkpoint", "results"} shape cli_eval.py's own --output uses,
so scripts.combine_decoder_results can read either file interchangeably.

Infrastructure only -- verified via run_smoke_test below against a tiny
freshly-initialized model AND a real (tiny, --limit-capped) blimp task,
not a real pretrained checkpoint.
"""

import argparse
import json

from common.config_file import parse_args_with_config

from .lm_eval_adapter import ThesisLM
from .tokenizer_adapter import ALL_SYSTEMS


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Evaluate a systems.pretraining.train checkpoint on lm-evaluation-harness's own registered tasks."
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="systems.pretraining.train checkpoint (.pt)")
    parser.add_argument("--system", choices=ALL_SYSTEMS, required=True)
    parser.add_argument("--tokenizer-checkpoint", type=str, required=True, help="systems/ tokenizer checkpoint")
    parser.add_argument(
        "--vocab-json", type=str, default=None,
        help="required for the five span-family systems (fairtok/magnet/flexitokens/manta/fanta)",
    )
    parser.add_argument(
        "--tasks", type=str, required=True,
        help="comma-separated lm-evaluation-harness task/group names, e.g. 'xnli,xcopa,blimp' (bare group "
        "names expand to every per-language/per-paradigm subtask) or 'xnli_sw,xcopa_et' for specific ones",
    )
    parser.add_argument(
        "--num-fewshot", type=int, default=0,
        help="in-context demonstrations, applied uniformly to every task in --tasks (the harness's own "
        "convention) -- see lm_eval_adapter.py's own module docstring for why this differs from "
        "cli_eval.py's per-benchmark handling",
    )
    parser.add_argument("--limit", type=int, default=None, help="cap examples scored per task (None = full split)")
    parser.add_argument(
        "--bootstrap-iters", type=int, default=1000,
        help="resamples for each task's own stderr (lm_eval's own default is 100000 -- much slower than this "
        "project's own eval_harness.bootstrap_ci default of 1000; lowered here to match that default speed/"
        "precision tradeoff rather than lm_eval's own, pass a higher value for a tighter estimate",
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--label", type=str, default="", help="name this run compares under -- defaults to --system. "
        "scripts.combine_decoder_results groups records by this key, same convention as cli_eval.py's own --label",
    )
    parser.add_argument("--output", type=str, default=None, help="write JSON results here (default: print to stdout)")
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument(
        "--wandb-project", type=str, default="pretraining",
        help="same default project as train.py/cli_eval.py; this run logs job_type='lm_eval'",
    )
    parser.add_argument("--run-name", type=str, default="")
    return parser


def _wandb_log_dict(harness_results):
    """Flattens lm_eval's own {task_name: {metric_key: value, ...}} results
    dict into a flat dict wandb.log can take -- e.g.
    "xnli_sw/acc,none" -> "xnli_sw/acc". Drops the ",none"/",<filter>"
    suffix lm_eval's own metric keys carry (its own multi-filter
    convention, unused here since no filters are configured) since it adds
    nothing readable in the wandb UI."""
    log_dict = {}
    for task_name, metrics in harness_results.items():
        for key, value in metrics.items():
            if key in ("alias", "samples"):
                continue
            clean_key = key.split(",")[0]
            if isinstance(value, (int, float)):
                log_dict[f"{task_name}/{clean_key}"] = value
    return log_dict


def main(argv=None):
    import lm_eval

    args = parse_args_with_config(build_arg_parser(), argv)
    task_names = [t.strip() for t in args.tasks.split(",")]

    lm = ThesisLM.from_pretrained(
        args.checkpoint, args.system, args.tokenizer_checkpoint,
        vocab_json=args.vocab_json, device=args.device,
    )

    eval_results = lm_eval.simple_evaluate(
        model=lm,
        tasks=task_names,
        num_fewshot=args.num_fewshot,
        limit=args.limit,
        bootstrap_iters=args.bootstrap_iters,
        verbosity="ERROR",
    )
    harness_results = eval_results["results"]

    record = {
        "label": args.label or args.system,
        "tasks": task_names,
        "checkpoint": args.checkpoint,
        "system": args.system,
        "tokenizer_checkpoint": args.tokenizer_checkpoint,
        "num_fewshot": args.num_fewshot,
        "results": harness_results,
    }
    payload = json.dumps(record, indent=2, default=str)
    if args.output:
        with open(args.output, "w") as f:
            f.write(payload)
        print(f"wrote results to {args.output}")
    print(payload)

    if args.use_wandb:
        import wandb

        run = wandb.init(
            project=args.wandb_project,
            name=args.run_name or None,
            job_type="lm_eval",
            config={
                "checkpoint": args.checkpoint,
                "system": args.system,
                "tokenizer_checkpoint": args.tokenizer_checkpoint,
                "tasks": task_names,
                "num_fewshot": args.num_fewshot,
                "limit": args.limit,
                "bootstrap_iters": args.bootstrap_iters,
            },
        )
        run.log(_wandb_log_dict(harness_results))
        run.finish()
        print(f"logged lm-eval-harness results to wandb project={args.wandb_project!r}")


def run_smoke_test():
    """Verifies the ThesisLM<->lm_eval wiring end-to-end: a tiny freshly-
    initialized (untrained) model + a real bpe tokenizer fit on a handful
    of sentences, scored against a REAL (network-fetched, --limit-capped)
    lm-evaluation-harness task -- not a claim about accuracy, just that
    loglikelihood/generate_until/task-loading/aggregation run without
    shape/dtype/device errors when driven by the harness itself, not just
    hand-built Instance objects. Needs network access (downloads blimp's
    real dataset, cached after the first run) -- unlike cli_eval.py's own
    run_smoke_test, which is fully offline."""
    from systems.tokenization.bpe.model import fit_bpe
    from systems.tokenization.bpe.train import _SMOKE_TEST_GROUPS

    import lm_eval

    from .model import TransformerLM
    from .model_configs import get_preset
    from .tokenizer_adapter import TokenizerAdapter

    sentences = [text for group in _SMOKE_TEST_GROUPS for text in group.values()]
    bpe_model = fit_bpe(sentences, vocab_size=384)
    id_to_bytes = TokenizerAdapter._native_id_to_bytes("bpe", bpe_model)
    adapter = TokenizerAdapter("bpe", bpe_model, id_to_bytes, span_to_id=None, device="cpu")

    model_cfg = get_preset("tiny")
    model_cfg.max_seq_len = 64
    model = TransformerLM(model_cfg, adapter.vocab_size)
    model.eval()

    lm = ThesisLM(model=model, adapter=adapter, device="cpu")
    results = lm_eval.simple_evaluate(
        model=lm, tasks=["blimp_adjunct_island"], limit=5, bootstrap_iters=10, verbosity="ERROR",
    )
    task_result = results["results"]["blimp_adjunct_island"]
    assert 0.0 <= task_result["acc,none"] <= 1.0
    assert task_result["sample_len"] == 5

    log_dict = _wandb_log_dict(results["results"])
    assert log_dict["blimp_adjunct_island/acc"] == task_result["acc,none"]

    print("systems.pretraining.cli_lm_eval smoke test passed:")
    print(f"  blimp_adjunct_island: acc={task_result['acc,none']:.3f} n={task_result['sample_len']}")


if __name__ == "__main__":
    main()
