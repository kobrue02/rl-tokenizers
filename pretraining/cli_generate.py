"""Standalone text generation from a pretraining.train checkpoint -- a
dedicated entry point for TransformerLM.generate() to look at raw model
output given an arbitrary prompt (eval_harness.evaluate_translation uses
generate() too, but only surfaces an aggregate BLEU/chrF score).

Two usages:
  - Interactive, one-off (quick, no queueing needed): run directly on a
    login node or inside an existing GPU allocation.
        python3 -m pretraining.cli_generate --checkpoint checkpoints/pretrain/final.pt \\
            --system bpe --tokenizer-checkpoint checkpoints/bpe_50k.json \\
            --prompt "The quick brown fox" --max-new-tokens 100 --num-samples 3
  - Batch, as part of a SLURM pipeline (see jobs/generate_samples.sh) --
    chain via --dependency=afterok after a training job. --prompt can be
    repeated for multiple prompts in one invocation.
"""

import argparse
import json

from common.config_file import parse_args_with_config

from .cli_eval import load_pretrained_model
from .tokenizer_adapter import ALL_SYSTEMS, TokenizerAdapter


def generate_text(model, adapter, prompt, lang=None, max_new_tokens=100, temperature=1.0, top_k=None, device="cpu"):
    """Returns the decoded CONTINUATION only (not prompt + continuation) --
    what a caller actually wants to read as "what did the model say", with
    the echoed-back prompt left out."""
    import torch

    ids = adapter.encode(prompt, lang=lang)
    ids_tensor = torch.tensor([ids], dtype=torch.long, device=device)
    generated = model.generate(ids_tensor, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k)
    new_ids = generated[0, len(ids):].tolist()
    return adapter.decode(new_ids).decode("utf-8", errors="replace")


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Generate text from a pretraining.train checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True, help="pretraining.train checkpoint (.pt)")
    parser.add_argument("--system", choices=ALL_SYSTEMS, required=True)
    parser.add_argument("--tokenizer-checkpoint", type=str, required=True, help="systems/ tokenizer checkpoint")
    parser.add_argument(
        "--vocab-json", type=str, default=None,
        help="required for the five span-family systems (fairtok/magnet/flexitokens/manta/fanta)",
    )
    parser.add_argument(
        "--prompt", action="append", required=True,
        help="prompt text; repeat --prompt for multiple prompts scored in one run",
    )
    parser.add_argument("--lang", type=str, default=None, help="lang hint forwarded to encode() -- only MAGNET uses it, see tokenizer_adapter.py")
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--num-samples", type=int, default=1, help="independent completions to draw per prompt (each a fresh sample under `temperature`, not a beam search)")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output", type=str, default=None, help="write a JSON file of {prompt, completions} (default: print to stdout only)")
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument(
        "--wandb-project", type=str, default="pretraining",
        help="same default project as train.py/cli_eval.py; logs job_type='generate'",
    )
    parser.add_argument("--run-name", type=str, default="")
    return parser


def main(argv=None):
    args = parse_args_with_config(build_arg_parser(), argv)
    model = load_pretrained_model(args.checkpoint, device=args.device)
    adapter = TokenizerAdapter.load(
        args.system, args.tokenizer_checkpoint, vocab_json_path=args.vocab_json, device=args.device
    )

    results = []
    table_rows = []
    for prompt in args.prompt:
        print(f"prompt: {prompt!r}\n")
        completions = []
        for i in range(args.num_samples):
            continuation = generate_text(
                model,
                adapter,
                prompt,
                lang=args.lang,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                device=args.device,
            )
            completions.append(continuation)
            table_rows.append([prompt, i, continuation])
            print(f"--- sample {i + 1}/{args.num_samples} ---")
            print(continuation)
            print()
        results.append({"prompt": prompt, "completions": completions})

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"wrote {len(results)} prompt(s) x {args.num_samples} sample(s) to {args.output}")

    if args.use_wandb:
        import wandb

        run = wandb.init(
            project=args.wandb_project,
            name=args.run_name or None,
            job_type="generate",
            config={
                "checkpoint": args.checkpoint,
                "system": args.system,
                "tokenizer_checkpoint": args.tokenizer_checkpoint,
                "prompts": args.prompt,
                "max_new_tokens": args.max_new_tokens,
                "temperature": args.temperature,
                "top_k": args.top_k,
                "num_samples": args.num_samples,
            },
        )
        run.log({"generations": wandb.Table(columns=["prompt", "sample_index", "completion"], data=table_rows)})
        run.finish()
        print(f"logged {len(table_rows)} generated sample(s) to wandb project={args.wandb_project!r}")


def run_smoke_test():
    """Verifies generate_text runs end to end against a tiny, freshly-
    initialized (untrained) model + a real bpe tokenizer (same construction
    as cli_eval.run_smoke_test) -- not a claim that an untrained model's
    output means anything."""
    from systems.bpe.model import fit_bpe
    from systems.bpe.train import _SMOKE_TEST_GROUPS

    from .model import TransformerLM
    from .model_configs import get_preset

    sentences = [text for group in _SMOKE_TEST_GROUPS for text in group.values()]
    bpe_model = fit_bpe(sentences, vocab_size=384)
    id_to_bytes = TokenizerAdapter._native_id_to_bytes("bpe", bpe_model)
    adapter = TokenizerAdapter("bpe", bpe_model, id_to_bytes, span_to_id=None, device="cpu")

    model_cfg = get_preset("tiny")
    model_cfg.max_seq_len = 64
    model = TransformerLM(model_cfg, adapter.vocab_size)
    model.eval()

    continuation = generate_text(model, adapter, "The quick brown fox", max_new_tokens=16, device="cpu")
    assert isinstance(continuation, str)
    assert len(continuation) > 0

    # wandb.Table construction (the shape main() logs under --use-wandb) --
    # real wandb module, but never wandb.init'd/logged, so no network/login needed.
    import wandb

    table = wandb.Table(columns=["prompt", "sample_index", "completion"], data=[["The quick brown fox", 0, continuation]])
    assert table is not None

    print("pretraining.cli_generate smoke test passed:")
    print(f"  continuation: {continuation!r}")


if __name__ == "__main__":
    main()
