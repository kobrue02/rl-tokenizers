"""Pretraining loop for systems.pretraining.model.TransformerLM over packed token
shards (systems.pretraining.data_prep/shard_dataset).

Single-GPU and multi-GPU (via `torchrun --nproc_per_node=N`, plain
DistributedDataParallel) both work; this detects which via the RANK/
WORLD_SIZE/LOCAL_RANK env vars torchrun sets. Not yet built: FSDP/model
sharding. The "7b" preset needs roughly 24-28GB just for fp32 optimizer
state plus bf16 weights/gradients -- more than one A100 (80GB) can hold
alongside activations at a useful batch size, so a real 7B run needs
sharding across GPUs that plain DDP (which replicates the full model per
rank) can't provide. A deliberate, documented gap: DDP now, FSDP as
follow-up work.

Reuses common.training.lr_schedule.build_lr_scheduler (the same HF-Trainer-
style warmup+decay every systems/ tokenizer trainer uses) rather than a
second scheduler implementation.
"""

import dataclasses
import os
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from common.training.lr_schedule import build_lr_scheduler

from .model import TransformerLM
from .model_configs import get_preset
from .shard_dataset import ShardedTokenDataset, load_shard_meta


@dataclasses.dataclass
class TrainConfig:
    model_size: str = "small"  # key into model_configs.PRESETS
    shard_dir: str = ""  # required -- output_dir from a prior data_prep.py run.
    # vocab_size is read from shard_dir/shards_meta.json, not set here, so
    # the embedding table always matches whichever tokenizer built the shards.
    seq_len: int = 1024
    per_device_batch_size: int = 8
    grad_accum_steps: int = 1  # effective batch size = per_device_batch_size
    # * grad_accum_steps * world_size (world_size=1 outside torchrun)
    total_steps: int = 10_000
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95  # 0.95 (not PyTorch's default 0.999) matches GPT-3/
    # LLaMA practice; higher beta2 reacts too slowly to early-training
    # gradient-scale swings
    grad_clip: float = 1.0
    warmup_ratio: float = 0.02  # smaller than tokenizer trainers' 0.1 default,
    # but total_steps here is typically much larger, so 2% is still a long
    # absolute warmup (hundreds-low thousands of steps), matching GPT-3/
    # LLaMA-style practice
    lr_scheduler_type: str = "cosine"  # standard for pretraining (vs.
    # "linear", HF Trainer's/tokenizer trainers' default) -- cosine's
    # slower-then-faster decay is established practice for long runs
    log_steps: int = 10
    save_steps: int = 1000
    keep_last_n_checkpoints: int = 3  # rotating step_{step}.pt checkpoints
    # beyond this many most-recent ones are deleted. Real incident: a
    # "small" preset checkpoint is ~1.5GB, and unrotated saving every
    # save_steps=1000 crashed a real cluster run via disk-quota exhaustion
    # at step 103,000/250,000. final.pt is never rotated away.
    output_dir: str = "checkpoints/pretrain"
    resume_from: str = ""  # "" starts fresh; else a checkpoint path saved
    # by this same script (see save_checkpoint/load_checkpoint)
    seed: int = 0
    device: str = ""  # "" auto-detects cuda if available, else cpu; ignored
    # under torchrun/DDP, which uses the LOCAL_RANK-assigned GPU instead
    dtype: str = "bfloat16"  # or "float32" (CPU/debugging -- bf16 autocast
    # on CPU gains nothing without tensor cores)
    num_workers: int = 4  # DataLoader worker processes
    use_wandb: bool = False
    wandb_project: str = "pretraining"
    run_name: str = ""


def is_distributed():
    return "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1


def setup_distributed():
    """Call once, before any CUDA tensor allocation. Returns (rank,
    local_rank, world_size, device)."""
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size, torch.device(f"cuda:{local_rank}")


def save_checkpoint(path, model, optimizer, scheduler, step, cfg, vocab_size):
    raw_model = model.module if isinstance(model, DistributedDataParallel) else model
    torch.save(
        {
            "step": step,
            "model_state_dict": raw_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "config": dataclasses.asdict(cfg),
            "vocab_size": vocab_size,
        },
        path,
    )


def load_checkpoint(path, model, optimizer, scheduler, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    raw_model = model.module if isinstance(model, DistributedDataParallel) else model
    raw_model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    return ckpt["step"]


def train(cfg: TrainConfig):
    distributed = is_distributed()
    if distributed:
        rank, local_rank, world_size, device = setup_distributed()
        is_main = rank == 0
    else:
        rank, world_size = 0, 1
        device = torch.device(cfg.device) if cfg.device else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        is_main = True

    torch.manual_seed(cfg.seed + rank)  # each rank gets a different seed --
    # enough (no DistributedSampler needed) since ShardedTokenDataset derives
    # sampling purely from (seed, idx).

    meta = load_shard_meta(cfg.shard_dir)
    vocab_size = meta["vocab_size"]
    model_cfg = get_preset(cfg.model_size)
    model_cfg.max_seq_len = max(model_cfg.max_seq_len, cfg.seq_len)
    model = TransformerLM(model_cfg, vocab_size).to(device)
    model_params = model.num_parameters()
    effective_batch_size = cfg.per_device_batch_size * cfg.grad_accum_steps * world_size
    # Tokens/FLOPs this run is planned to see, known before the first step
    # runs. tokens_per_param compares against Chinchilla's ~20-tokens/param
    # compute-optimal ratio; estimated_flops uses the standard ~6ND
    # approximation with N = total parameters (embedding included, the
    # loose convention most reports use).
    planned_training_tokens = effective_batch_size * cfg.total_steps * cfg.seq_len
    tokens_per_param = planned_training_tokens / model_params if model_params else 0.0
    estimated_flops = 6 * model_params * planned_training_tokens
    if is_main:
        print(
            f"model_size={cfg.model_size} params={model_params:,} "
            f"vocab_size={vocab_size} (from {cfg.shard_dir}/shards_meta.json) "
            f"world_size={world_size}"
        )
        print(
            f"planned_training_tokens={planned_training_tokens:,} "
            f"tokens_per_param={tokens_per_param:.1f} (Chinchilla-optimal ~20) "
            f"estimated_flops={estimated_flops:.3e} (~6*N*D)"
        )
    if distributed:
        model = DistributedDataParallel(model, device_ids=[local_rank])

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        betas=(cfg.beta1, cfg.beta2),
        weight_decay=cfg.weight_decay,
    )
    scheduler = build_lr_scheduler(
        optimizer, cfg.total_steps, cfg.warmup_ratio, cfg.lr_scheduler_type
    )

    # Resumed before the dataset is built -- start_step feeds directly into
    # ShardedTokenDataset's index_offset below, so a resumed run continues
    # the same (seed, idx) sequence instead of restarting at idx=0 and
    # replaying already-trained samples (see ShardedTokenDataset.__init__).
    start_step = 0
    if cfg.resume_from:
        start_step = load_checkpoint(cfg.resume_from, model, optimizer, scheduler, device)
        if is_main:
            print(f"resumed from {cfg.resume_from} at step {start_step}")

    samples_per_step = cfg.grad_accum_steps * cfg.per_device_batch_size
    total_samples = cfg.total_steps * samples_per_step
    samples_already_consumed = start_step * samples_per_step
    dataset = ShardedTokenDataset(
        cfg.shard_dir,
        cfg.seq_len,
        num_samples=total_samples - samples_already_consumed,
        seed=cfg.seed + rank,
        index_offset=samples_already_consumed,
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg.per_device_batch_size,
        num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda",
    )

    run = None
    if cfg.use_wandb and is_main:
        import wandb

        run = wandb.init(
            project=cfg.wandb_project,
            name=cfg.run_name or None,
            job_type="train",  # lets cli_eval/cli_generate share this project
            # while staying filterable apart by job_type in the wandb UI
            config={
                **dataclasses.asdict(cfg),
                "vocab_size": vocab_size,
                "model_params": model_params,
                "world_size": world_size,
                "effective_batch_size": effective_batch_size,
                "planned_training_tokens": planned_training_tokens,
                "tokens_per_param": tokens_per_param,
                "estimated_flops": estimated_flops,
            },
        )

    amp_dtype = torch.bfloat16 if cfg.dtype == "bfloat16" else torch.float32
    if is_main:
        os.makedirs(cfg.output_dir, exist_ok=True)

    model.train()
    data_iter = iter(loader)
    step = start_step
    t_last_log = time.time()
    tokens_since_log = 0
    saved_checkpoint_paths = []  # step_{step}.pt paths saved by this process,
    # oldest first -- rotated per keep_last_n_checkpoints below. Not
    # pre-populated from pre-existing files on a --resume-from run, so
    # resuming never deletes a checkpoint from before the resume.

    while step < cfg.total_steps:
        optimizer.zero_grad(set_to_none=True)
        loss_accum = 0.0
        for _ in range(cfg.grad_accum_steps):
            x, y = next(data_iter)
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type if device.type != "cpu" else "cpu",
                dtype=amp_dtype,
                enabled=amp_dtype != torch.float32,
            ):
                _, loss = model(x, labels=y)
            (loss / cfg.grad_accum_steps).backward()
            loss_accum += loss.item() / cfg.grad_accum_steps
            tokens_since_log += x.numel() * world_size

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        scheduler.step()
        step += 1

        if is_main and step % cfg.log_steps == 0:
            dt = time.time() - t_last_log
            tok_per_sec = tokens_since_log / max(dt, 1e-9)
            lr = scheduler.get_last_lr()[0]
            print(
                f"[step {step:6d}/{cfg.total_steps}] loss={loss_accum:.4f} "
                f"grad_norm={float(grad_norm):.3f} lr={lr:.2e} tok/s={tok_per_sec:,.0f}"
            )
            if run is not None:
                run.log(
                    {
                        "train/loss": loss_accum,
                        "train/grad_norm": float(grad_norm),
                        "train/learning_rate": lr,
                        "train/tokens_per_second": tok_per_sec,
                    },
                    step=step,
                )
            t_last_log = time.time()
            tokens_since_log = 0

        if is_main and cfg.save_steps and step % cfg.save_steps == 0:
            path = os.path.join(cfg.output_dir, f"step_{step}.pt")
            save_checkpoint(path, model, optimizer, scheduler, step, cfg, vocab_size)
            print(f"saved checkpoint to {path}")
            saved_checkpoint_paths.append(path)
            if cfg.keep_last_n_checkpoints > 0:
                while len(saved_checkpoint_paths) > cfg.keep_last_n_checkpoints:
                    stale_path = saved_checkpoint_paths.pop(0)
                    os.remove(stale_path)
                    print(f"removed older checkpoint {stale_path} (keeping last {cfg.keep_last_n_checkpoints})")

    if is_main:
        final_path = os.path.join(cfg.output_dir, "final.pt")
        save_checkpoint(final_path, model, optimizer, scheduler, step, cfg, vocab_size)
        print(f"training complete, saved final checkpoint to {final_path}")
        if run is not None:
            run.finish()

    if distributed:
        dist.barrier()
        dist.destroy_process_group()
