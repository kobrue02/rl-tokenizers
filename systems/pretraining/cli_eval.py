"""Command-line entry point for downstream benchmark evaluation (see
systems.pretraining.benchmarks/systems.pretraining.eval_harness): load a systems.pretraining.train
checkpoint + its matching tokenizer_adapter, load one or more benchmarks'
examples, score each, write one combined JSON results file.

Usage:
    python3 -m systems.pretraining.cli_eval --checkpoint checkpoints/pretrain/final.pt \\
        --system bpe --tokenizer-checkpoint checkpoints/bpe_12345.json \\
        --benchmark xnli --langs en,de,fr --max-examples 500 \\
        --output results/xnli_bpe.json

    python3 -m systems.pretraining.cli_eval --checkpoint checkpoints/pretrain/final.pt \\
        --system fanta --tokenizer-checkpoint checkpoints/fanta_12345.pt \\
        --vocab-json vocab_out/fanta_vocab_12345.json \\
        --benchmark flores_mt --lang-pairs eng:spa,eng:arz,deu_Latn:fra_Latn \\
        --max-examples 200 --output results/flores_fanta.json
        # flores_mt lang-pairs codes accept either a short LANG_SCRIPT code
        # (auto-resolved) or a full flores_plus lang_Script stem directly

    python3 -m systems.pretraining.cli_eval --checkpoint checkpoints/pretrain/final.pt \\
        --system bpe --tokenizer-checkpoint checkpoints/bpe_12345.json \\
        --benchmark xnli,xcopa,flores_mt \\
        --langs sw,tr,zh --lang-pairs eng:spa,eng:arz --max-examples 500 \\
        --output results/all_bpe.json
        # --benchmark takes a comma-separated list -- one combined results
        # file instead of one job per benchmark. --langs applies to xnli/
        # xcopa, --lang-pairs to flores_mt; each benchmark ignores the flag
        # it has no use for. --langs codes invalid for a given benchmark
        # (e.g. xcopa has no English config) are dropped with a warning,
        # not a crash (see _resolve_multiple_choice_langs); sw/tr/zh above
        # are valid for both xnli and xcopa, so nothing gets dropped here.
        # blimp is also multiple-choice (--langs there means paradigm names,
        # see benchmarks.load_blimp); cola/squad ignore --langs/--lang-pairs
        # entirely and use --max-new-tokens/--temperature like flores_mt does.

Infrastructure only -- verified via run_smoke_test below against a tiny
freshly-initialized model, not a real pretrained checkpoint.
"""

import argparse
import itertools
import json

import torch

from common.config_file import parse_args_with_config

from . import benchmarks
from .eval_harness import evaluate_cola, evaluate_multiple_choice, evaluate_qa, evaluate_translation
from .model import TransformerLM
from .model_configs import get_preset
from .tokenizer_adapter import ALL_SYSTEMS, TokenizerAdapter
from .train import TrainConfig

_MULTIPLE_CHOICE_BENCHMARKS = {"xnli", "xcopa", "blimp"}
_MULTIPLE_CHOICE_LANGS = {
    "xnli": benchmarks.XNLI_LANGS,
    "xcopa": benchmarks.XCOPA_LANGS,
    # "langs" means PARADIGMS for blimp (see benchmarks.load_blimp) -- reuses
    # _resolve_multiple_choice_langs's filter/warn/raise logic unchanged.
    "blimp": benchmarks.BLIMP_PARADIGMS,
}
_ACCEPTABILITY_BENCHMARKS = {"cola"}
_QA_BENCHMARKS = {"squad"}


def _resolve_multiple_choice_langs(benchmark, langs):
    """A shared --langs list across a --benchmark list can legitimately
    include codes one specific benchmark doesn't support -- e.g. "en" is
    valid for xnli but cambridgeltl/xcopa has no English config at all
    (raises `ValueError: BuilderConfig 'en' not found`; XCOPA is a
    cross-lingual extension of English-only COPA, never given its own
    English translation). Filters `langs` to this benchmark's valid set,
    printing what got dropped and why, rather than crashing the whole run
    or silently scoring nothing. Raises if nothing requested is valid,
    since an empty result would otherwise look like a real "0 examples" finding."""
    if langs is None:
        return None
    valid = _MULTIPLE_CHOICE_LANGS[benchmark]
    resolved = [l for l in langs if l in valid]
    dropped = [l for l in langs if l not in valid]
    if dropped:
        print(
            f"warning: --benchmark {benchmark} doesn't support language(s) {dropped} "
            f"(its own valid set: {valid}) -- skipping just those for this benchmark, "
            "scoring the rest of --langs normally"
        )
    if not resolved:
        raise ValueError(
            f"--benchmark {benchmark}: none of the requested --langs {langs} are valid for it "
            f"-- choose from {valid}"
        )
    return resolved
_TRANSLATION_BENCHMARKS = {"flores_mt"}


def load_pretrained_model(checkpoint_path, device="cpu"):
    """Reconstructs a TransformerLM from a train.save_checkpoint file: that
    checkpoint stores TrainConfig's fields plus vocab_size, not a
    ModelConfig directly, so get_preset(cfg.model_size) rebuilds the
    architecture the same way train.train() does."""
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
    dict wandb.log can take -- scalar metrics per benchmark (top-level +
    per-language for xnli/xcopa/blimp, top-level + per-pair for flores_mt,
    mcc/threshold for cola, exact_match/f1 for squad), plus a wandb.Table of
    raw generated/scored samples for flores_mt/squad (already capped by
    evaluate_translation/evaluate_qa) so generated text is browsable in the
    wandb UI, not just an aggregate number."""
    log_dict = {}
    for name, result in results.items():
        if "accuracy" in result and "per_language" in result:  # xnli/xcopa/blimp shape
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
        elif "mcc" in result:  # cola shape
            log_dict[f"{name}/mcc"] = result["mcc"]
            log_dict[f"{name}/accuracy"] = result["accuracy"]
            log_dict[f"{name}/threshold"] = result["threshold"]
            log_dict[f"{name}/n"] = result["n"]
            log_dict[f"{name}/n_calibration"] = result["n_calibration"]
        elif "exact_match" in result:  # squad shape
            log_dict[f"{name}/exact_match"] = result["exact_match"]
            log_dict[f"{name}/f1"] = result["f1"]
            log_dict[f"{name}/n"] = result["n"]
            log_dict[f"{name}/n_skipped_too_long"] = result["n_skipped_too_long"]
            sample_rows = [
                [s["question"], s["prediction"], ", ".join(s["references"])] for s in result["samples"]
            ]
            if sample_rows:
                log_dict[f"{name}/samples"] = wandb.Table(
                    columns=["question", "prediction", "references"], data=sample_rows
                )
        else:
            raise AssertionError(
                f"benchmark {name!r}'s results match no known shape "
                "(accuracy/per_language, bleu, mcc, exact_match)"
            )
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
        kwargs = {"langs": _resolve_multiple_choice_langs(benchmark, langs)}
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

    if benchmark in _ACCEPTABILITY_BENCHMARKS:
        examples = benchmarks.load_cola(split=split or "validation")
        if max_examples is not None:
            examples = itertools.islice(examples, max_examples)
        # Calibration always uses the FULL train split, uncapped by
        # --max-examples -- a noisy/truncated threshold would undermine the
        # whole point of calibrating it, and it's cheap (8551 rows).
        calibration_examples = benchmarks.load_cola(split="train")
        return evaluate_cola(model, adapter, examples, calibration_examples, device=device)

    if benchmark in _QA_BENCHMARKS:
        kwargs = {}
        if split is not None:
            kwargs["split"] = split
        examples = benchmarks.load_squad(**kwargs)
        if max_examples is not None:
            examples = itertools.islice(examples, max_examples)
        return evaluate_qa(
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
    """benchmark: a single name (str) or a list of names. Every requested
    benchmark is scored against the same model/adapter/max_examples/etc
    (langs feeds xnli/xcopa/blimp, lang_pairs feeds flores_mt, max_new_tokens/
    temperature feed flores_mt/squad; each benchmark ignores the flags it has
    no use for, so passing all of them for a mixed list is normal).

    Returns {benchmark_name: <results dict>} always in this shape, even
    for a single benchmark, so callers never branch on single-vs-list input.
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
    parser = argparse.ArgumentParser(description="Evaluate a systems.pretraining.train checkpoint on a benchmark.")
    parser.add_argument("--checkpoint", type=str, required=True, help="systems.pretraining.train checkpoint (.pt)")
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
    parser.add_argument(
        "--langs", type=str, default=None,
        help="comma-separated, for xnli/xcopa (language codes) or blimp (paradigm names)",
    )
    parser.add_argument(
        "--lang-pairs", type=str, default=None,
        help="comma-separated src:tgt pairs (e.g. eng:spa,deu_Latn:fra_Latn), for flores_mt -- "
        "accepts a short code common.data.oldi_data.LANG_SCRIPT maps OR any of flores_plus's "
        "~227 native lang_Script stems directly, see benchmarks.py",
    )
    parser.add_argument("--split", type=str, default=None, help="dataset split override (loader-specific default otherwise)")
    parser.add_argument("--max-examples", type=int, default=None, help="cap examples scored (None = full split)")
    parser.add_argument(
        "--length-normalize", action="store_true",
        help="xnli/xcopa/blimp only -- see evaluate_multiple_choice",
    )
    parser.add_argument("--max-new-tokens", type=int, default=128, help="flores_mt/squad only")
    parser.add_argument("--temperature", type=float, default=1.0, help="flores_mt/squad only")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output", type=str, default=None, help="write JSON results here (default: print to stdout)")
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument(
        "--wandb-project", type=str, default="pretraining",
        help="same default project as train.py; this run logs job_type='eval' so both "
        "share one project, filterable apart in the wandb UI",
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
    sentences -- not a claim about accuracy, just that loglikelihood/
    evaluate_multiple_choice/evaluate_translation run without shape/dtype/
    device errors on synthetic examples matching benchmarks.py's shapes.

    Builds a real TokenizerAdapter directly from an in-memory BPEModel
    (bypassing TokenizerAdapter.load's checkpoint-file requirement) so this
    exercises the actual encode()/decode() path, not a simplified stand-in."""
    from systems.tokenization.bpe.model import fit_bpe
    from systems.tokenization.bpe.train import _SMOKE_TEST_GROUPS
    from .benchmarks import CoLAExample, MultipleChoiceExample, QAExample, TranslationExample
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

    cola_examples = [
        CoLAExample(lang="en", sentence="The cat sat on the mat.", label=1),
        CoLAExample(lang="en", sentence="Cat mat the sat on.", label=0),
    ]
    cola_calibration = [
        CoLAExample(lang="en", sentence="She walked to the store.", label=1),
        CoLAExample(lang="en", sentence="Store the walked to she.", label=0),
    ]
    cola_results = evaluate_cola(model, adapter, cola_examples, cola_calibration, device="cpu")
    assert -1.0 <= cola_results["mcc"] <= 1.0
    assert cola_results["n"] == len(cola_examples)
    assert cola_results["n_calibration"] == len(cola_calibration)

    qa_examples = [
        QAExample(lang="en", context="The cat sat on the mat.", question="Where did the cat sit?", answers=["the mat", "mat"]),
    ]
    qa_results = evaluate_qa(model, adapter, qa_examples, device="cpu", max_new_tokens=8)
    assert qa_results["n"] + qa_results["n_skipped_too_long"] == len(qa_examples)
    assert 0.0 <= qa_results["exact_match"] <= 1.0
    assert 0.0 <= qa_results["f1"] <= 1.0

    # run_evaluation's multi-benchmark dispatch, exercised against fake
    # loaders (testing fan-out/return-shape logic, not real network calls)
    # via monkeypatching, restored in a finally so it doesn't leak.
    original_benchmarks = dict(benchmarks.BENCHMARKS)
    try:
        benchmarks.BENCHMARKS["xnli"] = lambda langs=None, split="test": iter(mc_examples)
        benchmarks.BENCHMARKS["xcopa"] = lambda langs=None, split="test": iter(mc_examples)
        benchmarks.BENCHMARKS["blimp"] = lambda langs=None, split="train": iter(mc_examples)
        multi_results = run_evaluation(model, adapter, ["xnli", "xcopa", "blimp"], device="cpu")
        assert set(multi_results) == {"xnli", "xcopa", "blimp"}
        assert multi_results["xnli"]["n"] == len(mc_examples)
        assert multi_results["xcopa"]["n"] == len(mc_examples)
        assert multi_results["blimp"]["n"] == len(mc_examples)
        json.dumps(multi_results, default=str)  # confirms the combined dict is actually JSON-serializable

        single_result = run_evaluation(model, adapter, "xnli", device="cpu")
        assert set(single_result) == {"xnli"}  # single-name input still comes back wrapped by benchmark name

        # _wandb_log_dict against all four result shapes -- uses the real
        # wandb module for real Table construction but never wandb.init's/logs.
        import wandb

        combined = {**multi_results, "flores_mt": mt_results, "cola": cola_results, "squad": qa_results}
        log_dict = _wandb_log_dict(combined, wandb)
        assert log_dict["xnli/accuracy"] == multi_results["xnli"]["accuracy"]
        assert log_dict["flores_mt/bleu"] == mt_results["bleu"]
        assert isinstance(log_dict["flores_mt/samples"], wandb.Table)
        assert log_dict["cola/mcc"] == cola_results["mcc"]
        assert log_dict["squad/exact_match"] == qa_results["exact_match"]
    finally:
        benchmarks.BENCHMARKS.clear()
        benchmarks.BENCHMARKS.update(original_benchmarks)

    print("systems.pretraining.cli_eval smoke test passed:")
    print(f"  multiple-choice: accuracy={mc_results['accuracy']:.3f} n={mc_results['n']}")
    print(f"  translation: bleu={mt_results['bleu']:.3f} chrf={mt_results['chrf']:.3f} n={mt_results['n']}")
    print(f"  cola: mcc={cola_results['mcc']:.3f} accuracy={cola_results['accuracy']:.3f} n={cola_results['n']}")
    print(f"  squad: exact_match={qa_results['exact_match']:.3f} f1={qa_results['f1']:.3f} n={qa_results['n']}")
    print(f"  multi-benchmark dispatch: {sorted(multi_results)}")


if __name__ == "__main__":
    main()
