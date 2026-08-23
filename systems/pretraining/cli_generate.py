"""Standalone text generation from a systems.pretraining.train checkpoint -- a
dedicated entry point for TransformerLM.generate() to look at raw model
output given an arbitrary prompt (eval_harness.evaluate_translation uses
generate() too, but only surfaces an aggregate BLEU/chrF score).

Three usages:
  - Interactive "app" mode (no flags at all): browse checkpoints found on
    disk, pick one, then type prompts in a loop. The tokenizer (--system/
    --tokenizer-checkpoint) is auto-resolved from the checkpoint's own
    stored TrainConfig.shard_dir -> that shard_dir's own shards_meta.json
    (the exact same source train.py's own generate_samples() uses) --
    only asks you anything extra (--vocab-json) if the resolved system
    actually needs it (the five span-family tokenizers).
        python3 -m systems.pretraining.cli_generate
  - Interactive, but pointed at a specific checkpoint (skips the browse step):
        python3 -m systems.pretraining.cli_generate --checkpoint checkpoints/pretrain/final.pt
  - Batch/scripted, exactly as before (full manual control, no prompts) --
    as part of a SLURM pipeline (see jobs/generate_samples.sh), chained via
    --dependency=afterok after a training job. --prompt can be repeated for
    multiple prompts in one invocation. Passing --system/--tokenizer-checkpoint/
    --prompt is what selects this mode over the interactive ones above.
        python3 -m systems.pretraining.cli_generate --checkpoint checkpoints/pretrain/final.pt \\
            --system bpe --tokenizer-checkpoint checkpoints/bpe_50k.json \\
            --prompt "The quick brown fox" --max-new-tokens 100 --num-samples 3
"""

import argparse
import glob
import json
import os

from common.config_file import parse_args_with_config

from .cli_eval import load_pretrained_model
from .tokenizer_adapter import ALL_SYSTEMS, TokenizerAdapter, _SPAN_SYSTEMS


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


def discover_checkpoints(root="checkpoints"):
    """Every step_*.pt/final.pt under root, sorted newest-first by mtime.
    Deliberately stat-only (os.path.getmtime/getsize), never torch.load --
    a systems.pretraining.train checkpoint can be multi-GB (a "large"
    preset one is ~8.5GB), so listing candidates must not load any of them."""
    paths = glob.glob(os.path.join(root, "**", "*.pt"), recursive=True)
    return sorted(paths, key=os.path.getmtime, reverse=True)


def load_checkpoint_and_resolve_tokenizer(checkpoint_path, device="cpu"):
    """Loads `checkpoint_path` ONCE -- mirrors cli_eval.load_pretrained_model's
    own reconstruction logic exactly, but ALSO returns what's needed to
    resolve the matching tokenizer, which load_pretrained_model itself
    doesn't expose (and re-loading the checkpoint a second time just to
    get it would double the cost of what's often a multi-GB file).

    Resolution source: the checkpoint's own stored TrainConfig.shard_dir
    (train.save_checkpoint writes "config": dataclasses.asdict(cfg)) ->
    that shard_dir's own shards_meta.json ("system"/"checkpoint", written
    by data_prep.py from whatever tokenizer ACTUALLY built the shards this
    model trained on) -- not a separately-typed --system/--tokenizer-
    checkpoint pair that could drift out of sync with the real answer.

    Returns (model, system, tokenizer_checkpoint_path, trained_to_step).
    Raises FileNotFoundError if shard_dir no longer exists (e.g. moved/
    deleted since training) -- the caller decides whether to fall back to
    manual --system/--tokenizer-checkpoint entry."""
    import torch

    from .model import TransformerLM
    from .model_configs import get_preset
    from .shard_dataset import load_shard_meta
    from .train import TrainConfig

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = TrainConfig(**ckpt["config"])
    model_cfg = get_preset(cfg.model_size)
    model_cfg.max_seq_len = max(model_cfg.max_seq_len, cfg.seq_len)
    model = TransformerLM(model_cfg, ckpt["vocab_size"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    meta = load_shard_meta(cfg.shard_dir)
    return model, meta["system"], meta["checkpoint"], ckpt["step"]


def _prompt_for_checkpoint(checkpoints_dir):
    paths = discover_checkpoints(checkpoints_dir)
    if not paths:
        raise SystemExit(
            f"no .pt checkpoints found under {checkpoints_dir!r} -- pass --checkpoint "
            "directly, or --checkpoints-dir if your checkpoints live somewhere else"
        )
    print(f"Found {len(paths)} checkpoint(s) under {checkpoints_dir!r} (newest first):\n")
    for i, path in enumerate(paths):
        import datetime

        mtime = datetime.datetime.fromtimestamp(os.path.getmtime(path)).strftime("%Y-%m-%d %H:%M")
        size_gb = os.path.getsize(path) / 1e9
        print(f"  [{i}] {path}  ({mtime}, {size_gb:.1f}GB)")
    while True:
        choice = input(f"\nSelect a checkpoint [0-{len(paths) - 1}]: ").strip()
        try:
            return paths[int(choice)]
        except (ValueError, IndexError):
            print(f"enter a number between 0 and {len(paths) - 1}")


def interactive_main(args):
    checkpoint_path = args.checkpoint or _prompt_for_checkpoint(args.checkpoints_dir)

    print(f"\nLoading {checkpoint_path} ...")
    try:
        model, system, tokenizer_checkpoint, step = load_checkpoint_and_resolve_tokenizer(
            checkpoint_path, device=args.device
        )
        print(f"loaded -- system={system!r}, trained to step {step:,}")
    except FileNotFoundError as e:
        print(
            f"couldn't auto-resolve the tokenizer this checkpoint was trained with ({e}) "
            "-- falling back to manual entry."
        )
        model = load_pretrained_model(checkpoint_path, device=args.device)
        system = input(f"system ({', '.join(ALL_SYSTEMS)}): ").strip()
        tokenizer_checkpoint = input("tokenizer checkpoint path: ").strip()

    vocab_json = args.vocab_json
    if system in _SPAN_SYSTEMS and not vocab_json:
        vocab_json = input(f"{system!r} needs its vocab.json path (--vocab-out at training time): ").strip()
    adapter = TokenizerAdapter.load(system, tokenizer_checkpoint, vocab_json_path=vocab_json or None, device=args.device)

    print(
        f"\nReady -- generating up to {args.max_new_tokens} tokens per prompt "
        f"(temperature={args.temperature}, num_samples={args.num_samples}).\n"
        "Type a prompt and press enter (empty line or 'quit' to exit).\n"
    )
    while True:
        try:
            prompt = input("prompt> ").strip()
        except EOFError:
            break
        if not prompt or prompt.lower() in ("quit", "exit"):
            break
        for i in range(args.num_samples):
            continuation = generate_text(
                model, adapter, prompt, lang=args.lang, max_new_tokens=args.max_new_tokens,
                temperature=args.temperature, top_k=args.top_k, device=args.device,
            )
            prefix = f"[{i + 1}/{args.num_samples}] " if args.num_samples > 1 else ""
            print(f"{prefix}{continuation}\n")


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Generate text from a systems.pretraining.train checkpoint.")
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="systems.pretraining.train checkpoint (.pt) -- omit to browse/select interactively",
    )
    parser.add_argument(
        "--checkpoints-dir", type=str, default="checkpoints",
        help="root to search for checkpoints in interactive mode (only used when --checkpoint is omitted)",
    )
    parser.add_argument(
        "--system", choices=ALL_SYSTEMS, default=None,
        help="omit in interactive mode -- auto-resolved from the checkpoint's own shard_dir",
    )
    parser.add_argument(
        "--tokenizer-checkpoint", type=str, default=None,
        help="systems/ tokenizer checkpoint -- omit in interactive mode (see --system)",
    )
    parser.add_argument(
        "--vocab-json", type=str, default=None,
        help="required for the five span-family systems (fairtok/magnet/flexitokens/manta/fanta) -- "
        "interactive mode prompts for this itself if it turns out to be needed",
    )
    parser.add_argument(
        "--prompt", action="append", default=None,
        help="prompt text; repeat --prompt for multiple prompts scored in one run. Passing this at all "
        "is what selects batch/scripted mode over the interactive one -- see module docstring",
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

    if not args.prompt:
        # No --prompt at all -- interactive mode (see module docstring).
        # Everything else (--checkpoint, --system, --tokenizer-checkpoint,
        # --vocab-json) is optional here; interactive_main resolves or asks
        # for whatever it actually needs.
        interactive_main(args)
        return

    # Batch/scripted mode from here down -- unchanged from before
    # interactive mode existed. --prompt being present is what selects this
    # path, so these are the flags that mode actually needs.
    missing = [
        flag for flag, val in [
            ("--checkpoint", args.checkpoint), ("--system", args.system),
            ("--tokenizer-checkpoint", args.tokenizer_checkpoint),
        ] if not val
    ]
    if missing:
        raise SystemExit(f"--prompt was given (batch mode) but {', '.join(missing)} is also required")

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
    from systems.tokenization.bpe.model import fit_bpe
    from systems.tokenization.bpe.train import _SMOKE_TEST_GROUPS

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

    print("systems.pretraining.cli_generate smoke test passed:")
    print(f"  continuation: {continuation!r}")


if __name__ == "__main__":
    main()
