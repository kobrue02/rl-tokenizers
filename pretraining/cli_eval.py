"""Command-line entry point for downstream benchmark evaluation (see
pretraining.benchmarks/pretraining.eval_harness): load a pretraining.train
checkpoint + its matching tokenizer_adapter, load one or more benchmarks'
examples, score each, write one combined JSON results file.

Usage:
    python3 -m pretraining.cli_eval --checkpoint checkpoints/pretrain/final.pt \\
        --system bpe --tokenizer-checkpoint checkpoints/bpe_12345.json \\
        --benchmark xnli --langs en,de,fr --max-examples 500 \\
        --output results/xnli_bpe.json

    python3 -m pretraining.cli_eval --checkpoint checkpoints/pretrain/final.pt \\
        --system fanta --tokenizer-checkpoint checkpoints/fanta_12345.pt \\
        --vocab-json vocab_out/fanta_vocab_12345.json \\
        --benchmark flores_mt --lang-pairs eng:spa,eng:arz,deu_Latn:fra_Latn \\
        --max-examples 200 --output results/flores_fanta.json
        # (flores_mt lang-pairs codes accept EITHER this project's own
        # 9-language short codes (arz/bam/ben/eng/kas/lij/mni/nqo/spa,
        # auto-resolved to their full stem) OR any of flores_plus's own
        # ~227 native lang_Script stems directly (e.g. deu_Latn, fra_Latn)
        # -- verified live to be genuinely fully N-way parallel across all
        # of them, see benchmarks.py's own module docstring)

    python3 -m pretraining.cli_eval --checkpoint checkpoints/pretrain/final.pt \\
        --system bpe --tokenizer-checkpoint checkpoints/bpe_12345.json \\
        --benchmark xnli,xcopa,flores_mt \\
        --langs en,de,fr --lang-pairs eng:spa,eng:arz --max-examples 500 \\
        --output results/all_bpe.json
        # --benchmark takes a COMMA-SEPARATED list -- one job, one combined
        # results file keyed by benchmark name, instead of one sbatch call
        # per benchmark. --langs only applies to xnli/xcopa in the list,
        # --lang-pairs only to flores_mt -- each benchmark just ignores the
        # flag it has no use for (see run_evaluation).

Infrastructure only (see eval_harness.py's own docstring) -- this module is
verified via run_smoke_test below, against a tiny freshly-initialized model,
not against a real pretrained checkpoint.
"""

import argparse
import itertools
import json

import torch

from common.config_file import parse_args_with_config

from . import benchmarks
from .eval_harness import evaluate_multiple_choice, evaluate_translation
from .model import TransformerLM
from .model_configs import get_preset
from .tokenizer_adapter import ALL_SYSTEMS, TokenizerAdapter
from .train import TrainConfig

_MULTIPLE_CHOICE_BENCHMARKS = {"xnli", "xcopa"}
_TRANSLATION_BENCHMARKS = {"flores_mt"}


def load_pretrained_model(checkpoint_path, device="cpu"):
    """Reconstructs a TransformerLM from a pretraining.train.save_checkpoint
    file: that checkpoint stores TrainConfig's fields (asdict) plus
    vocab_size, not a ModelConfig directly -- model_configs.get_preset(
    cfg.model_size) rebuilds the architecture the same way train.train()
    itself does, so this stays a single source of truth for "how do
    model_size + shard_dir's vocab_size become a TransformerLM"."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = TrainConfig(**ckpt["config"])
    model_cfg = get_preset(cfg.model_size)
    model_cfg.max_seq_len = max(model_cfg.max_seq_len, cfg.seq_len)
    model = TransformerLM(model_cfg, ckpt["vocab_size"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def _wandb_log_dict(results, wandb):
    """Flattens run_evaluation's {benchmark_name: results} into a single
    dict wandb.log can take in one call -- scalar metrics for every
    benchmark (top-level + per-language for xnli/xcopa, top-level +
    per-pair for flores_mt), PLUS a wandb.Table of raw generated samples per
    flores_mt pair (eval_harness.evaluate_translation already caps these at
    max_samples_per_pair -- see that module's docstring for why the full set
    isn't kept) so the actual generated text is something you can browse in
    the wandb UI, not just a chrF number."""
    log_dict = {}
    for name, result in results.items():
        if "accuracy" in result:  # xnli/xcopa shape
            log_dict[f"{name}/accuracy"] = result["accuracy"]
            log_dict[f"{name}/n"] = result["n"]
            for lang, stats in result["per_language"].items():
                log_dict[f"{name}/{lang}/accuracy"] = stats["accuracy"]
                log_dict[f"{name}/{lang}/n"] = stats["n"]
        elif "bleu" in result:  # flores_mt shape
            log_dict[f"{name}/bleu"] = result["bleu"]
            log_dict[f"{name}/chrf"] = result["chrf"]
            log_dict[f"{name}/n"] = result["n"]
            sample_rows = []
            for pair_key, stats in result["per_pair"].items():
                log_dict[f"{name}/{pair_key}/bleu"] = stats["bleu"]
                log_dict[f"{name}/{pair_key}/chrf"] = stats["chrf"]
                log_dict[f"{name}/{pair_key}/n"] = stats["n"]
                for sample in stats["samples"]:
                    sample_rows.append([pair_key, sample["source"], sample["hypothesis"], sample["reference"]])
            if sample_rows:
                log_dict[f"{name}/samples"] = wandb.Table(
                    columns=["pair", "source", "hypothesis", "reference"], data=sample_rows
                )
        else:
            raise AssertionError(f"benchmark {name!r}'s results have neither 'accuracy' nor 'bleu' -- unknown shape")
    return log_dict


def _parse_lang_pairs(raw):
    pairs = []
    for pair in raw.split(","):
        src, tgt = pair.split(":")
        pairs.append((src, tgt))
    return pairs


def _run_single_benchmark(
    model,
    adapter,
    benchmark,
    langs=None,
    lang_pairs=None,
    split=None,
    max_examples=None,
    device="cpu",
    length_normalize=False,
    max_new_tokens=128,
    temperature=1.0,
):
    if benchmark not in benchmarks.BENCHMARKS:
        raise ValueError(f"unknown benchmark {benchmark!r} -- expected one of {list(benchmarks.BENCHMARKS)}")

    if benchmark in _MULTIPLE_CHOICE_BENCHMARKS:
        loader = benchmarks.BENCHMARKS[benchmark]
        kwargs = {"langs": langs}
        if split is not None:
            kwargs["split"] = split
        examples = loader(**kwargs)
        if max_examples is not None:
            examples = itertools.islice(examples, max_examples)
        return evaluate_multiple_choice(model, adapter, examples, device=device, length_normalize=length_normalize)

    if benchmark in _TRANSLATION_BENCHMARKS:
        if not lang_pairs:
            raise ValueError(f"--benchmark {benchmark} needs --lang-pairs (e.g. eng:spa)")
        kwargs = {"lang_pairs": lang_pairs}
        if split is not None:
            kwargs["split"] = split
        examples = benchmarks.load_flores_mt(**kwargs)
        if max_examples is not None:
            examples = itertools.islice(examples, max_examples)
        return evaluate_translation(
            model, adapter, examples, device=device, max_new_tokens=max_new_tokens, temperature=temperature
        )

    raise AssertionError(f"benchmark {benchmark!r} in BENCHMARKS but not in either evaluator set")


def run_evaluation(
    model,
    adapter,
    benchmark,
    langs=None,
    lang_pairs=None,
    split=None,
    max_examples=None,
    device="cpu",
    length_normalize=False,
    max_new_tokens=128,
    temperature=1.0,
):
    """benchmark: a single name (str) or a list of names -- e.g. "xnli" or
    ["xnli", "xcopa", "flores_mt"]. Every requested benchmark is scored
    against the SAME model/adapter/max_examples/etc (langs feeds
    xnli/xcopa, lang_pairs feeds flores_mt -- a benchmark that has no use
    for one of those two just ignores it, so passing both --langs and
    --lang-pairs for a mixed --benchmark list is normal, not redundant).

    Returns {benchmark_name: <that benchmark's own results dict>} -- ALWAYS
    this shape, even for a single benchmark, rather than returning that one
    benchmark's dict unwrapped: one consistent return shape regardless of
    how many benchmarks were requested, so callers never need to branch on
    "was this a single name or a list" to find their results.
    """
    names = [benchmark] if isinstance(benchmark, str) else list(benchmark)
    unknown = [b for b in names if b not in benchmarks.BENCHMARKS]
    if unknown:
        raise ValueError(f"unknown benchmark(s) {unknown} -- expected some of {list(benchmarks.BENCHMARKS)}")

    return {
        name: _run_single_benchmark(
            model,
            adapter,
            name,
            langs=langs,
            lang_pairs=lang_pairs,
            split=split,
            max_examples=max_examples,
            device=device,
            length_normalize=length_normalize,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        for name in names
    }


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Evaluate a pretraining.train checkpoint on a benchmark.")
    parser.add_argument("--checkpoint", type=str, required=True, help="pretraining.train checkpoint (.pt)")
    parser.add_argument("--system", choices=ALL_SYSTEMS, required=True)
    parser.add_argument("--tokenizer-checkpoint", type=str, required=True, help="systems/ tokenizer checkpoint")
    parser.add_argument(
        "--vocab-json", type=str, default=None,
        help="required for the five span-family systems (fairtok/magnet/flexitokens/manta/fanta)",
    )
    parser.add_argument(
        "--benchmark", type=str, required=True,
        help=f"comma-separated list of benchmarks to run in one job (choices: {sorted(benchmarks.BENCHMARKS)})",
    )
    parser.add_argument("--langs", type=str, default=None, help="comma-separated, for xnli/xcopa")
    parser.add_argument(
        "--lang-pairs", type=str, default=None,
        help="comma-separated src:tgt pairs (e.g. eng:spa,deu_Latn:fra_Latn), for flores_mt -- "
        "accepts this project's 9-language short codes OR any of flores_plus's ~227 native "
        "lang_Script stems directly, see benchmarks.py",
    )
    parser.add_argument("--split", type=str, default=None, help="dataset split override (loader-specific default otherwise)")
    parser.add_argument("--max-examples", type=int, default=None, help="cap examples scored (None = full split)")
    parser.add_argument("--length-normalize", action="store_true", help="xnli/xcopa only -- see evaluate_multiple_choice")
    parser.add_argument("--max-new-tokens", type=int, default=128, help="flores_mt only")
    parser.add_argument("--temperature", type=float, default=1.0, help="flores_mt only")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output", type=str, default=None, help="write JSON results here (default: print to stdout)")
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument(
        "--wandb-project", type=str, default="pretraining",
        help="SAME default project as pretraining.train's own wandb_project -- see that "
        "module's job_type='train' comment; this run logs job_type='eval' so both share "
        "one project, filterable apart in the wandb UI",
    )
    parser.add_argument("--run-name", type=str, default="")
    return parser


def main(argv=None):
    args = parse_args_with_config(build_arg_parser(), argv)
    langs = args.langs.split(",") if args.langs else None
    lang_pairs = _parse_lang_pairs(args.lang_pairs) if args.lang_pairs else None
    benchmark_names = [b.strip() for b in args.benchmark.split(",")]

    model = load_pretrained_model(args.checkpoint, device=args.device)
    adapter = TokenizerAdapter.load(
        args.system, args.tokenizer_checkpoint, vocab_json_path=args.vocab_json, device=args.device
    )

    results = run_evaluation(
        model,
        adapter,
        benchmark_names,
        langs=langs,
        lang_pairs=lang_pairs,
        split=args.split,
        max_examples=args.max_examples,
        device=args.device,
        length_normalize=args.length_normalize,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )

    payload = json.dumps(results, indent=2, default=str)
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
            job_type="eval",
            config={
                "checkpoint": args.checkpoint,
                "system": args.system,
                "tokenizer_checkpoint": args.tokenizer_checkpoint,
                "benchmark": benchmark_names,
                "langs": langs,
                "lang_pairs": lang_pairs,
                "max_examples": args.max_examples,
                "length_normalize": args.length_normalize,
            },
        )
        run.log(_wandb_log_dict(results, wandb))
        run.finish()
        print(f"logged eval results to wandb project={args.wandb_project!r}")


def run_smoke_test():
    """Verifies the scoring plumbing end-to-end against a tiny, freshly-
    initialized (untrained) model + a real bpe tokenizer fit on a handful of
    sentences -- NOT a claim about accuracy (an untrained model scores at
    chance by construction), just that loglikelihood/evaluate_multiple_choice/
    evaluate_translation run without shape/dtype/device errors on synthetic
    examples matching benchmarks.py's own dataclass shapes.

    Builds a real TokenizerAdapter directly from an in-memory BPEModel
    (TokenizerAdapter._native_id_to_bytes + the constructor, bypassing
    TokenizerAdapter.load's checkpoint-file requirement) rather than a
    hand-rolled stand-in -- exercises the actual encode()/decode() path
    every real evaluation run will use, not a simplified lookalike."""
    from systems.bpe.model import fit_bpe
    from systems.bpe.train import _SMOKE_TEST_GROUPS
    from .benchmarks import MultipleChoiceExample, TranslationExample
    from .model_configs import get_preset

    sentences = [text for group in _SMOKE_TEST_GROUPS for text in group.values()]
    bpe_model = fit_bpe(sentences, vocab_size=384)
    id_to_bytes = TokenizerAdapter._native_id_to_bytes("bpe", bpe_model)
    adapter = TokenizerAdapter("bpe", bpe_model, id_to_bytes, span_to_id=None, device="cpu")

    model_cfg = get_preset("tiny")
    model_cfg.max_seq_len = 64
    model = TransformerLM(model_cfg, adapter.vocab_size)
    model.eval()

    mc_examples = [
        MultipleChoiceExample(lang="en", context="The cat sat on the mat", choices=[" happily", " because"], label=0),
        MultipleChoiceExample(lang="de", context="Die Katze saß auf der Matte", choices=[" glücklich", " weil"], label=1),
    ]
    mc_results = evaluate_multiple_choice(model, adapter, mc_examples, device="cpu")
    assert 0.0 <= mc_results["accuracy"] <= 1.0
    assert mc_results["n"] == len(mc_examples)
    assert set(mc_results["per_language"]) == {"en", "de"}

    mt_examples = [
        TranslationExample(source_lang="en", target_lang="de", source_text="The cat sat.", reference_text="Die Katze saß."),
    ]
    mt_results = evaluate_translation(model, adapter, mt_examples, device="cpu", max_new_tokens=8)
    assert mt_results["n"] == 1
    assert "en->de" in mt_results["per_pair"]
    assert len(mt_results["per_pair"]["en->de"]["samples"]) == 1

    # run_evaluation's comma-separated multi-benchmark dispatch, exercised
    # against fake loaders (not real xnli/xcopa/flores_mt network calls --
    # this is testing the CLI's own fan-out/return-shape logic, which
    # doesn't care what the individual loaders actually return) via
    # monkeypatching benchmarks.BENCHMARKS, restored in a finally so this
    # doesn't leak into any other test/run in the same process.
    original_benchmarks = dict(benchmarks.BENCHMARKS)
    try:
        benchmarks.BENCHMARKS["xnli"] = lambda langs=None, split="test": iter(mc_examples)
        benchmarks.BENCHMARKS["xcopa"] = lambda langs=None, split="test": iter(mc_examples)
        multi_results = run_evaluation(model, adapter, ["xnli", "xcopa"], device="cpu")
        assert set(multi_results) == {"xnli", "xcopa"}
        assert multi_results["xnli"]["n"] == len(mc_examples)
        assert multi_results["xcopa"]["n"] == len(mc_examples)
        json.dumps(multi_results, default=str)  # confirms the combined dict is actually JSON-serializable

        single_result = run_evaluation(model, adapter, "xnli", device="cpu")
        assert set(single_result) == {"xnli"}  # single-name input still comes back wrapped by benchmark name

        # _wandb_log_dict against BOTH result shapes (multiple-choice via
        # multi_results, translation via mt_results) -- built with the real
        # wandb module (for real wandb.Table construction) but never
        # actually wandb.init'd/logged, so this needs no network/login.
        import wandb

        combined = {**multi_results, "flores_mt": mt_results}
        log_dict = _wandb_log_dict(combined, wandb)
        assert log_dict["xnli/accuracy"] == multi_results["xnli"]["accuracy"]
        assert log_dict["flores_mt/bleu"] == mt_results["bleu"]
        assert isinstance(log_dict["flores_mt/samples"], wandb.Table)
    finally:
        benchmarks.BENCHMARKS.clear()
        benchmarks.BENCHMARKS.update(original_benchmarks)

    print("pretraining.cli_eval smoke test passed:")
    print(f"  multiple-choice: accuracy={mc_results['accuracy']:.3f} n={mc_results['n']}")
    print(f"  translation: bleu={mt_results['bleu']:.3f} chrf={mt_results['chrf']:.3f} n={mt_results['n']}")
    print(f"  multi-benchmark dispatch: {sorted(multi_results)}")


if __name__ == "__main__":
    main()
