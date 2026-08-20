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

import contextlib
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
    val_fraction: float = 0.05  # fraction of shard_dir's own shards reserved
    # for held-out validation loss (never sampled for training) -- 0
    # disables validation entirely. See _split_train_val_shard_files's own
    # docstring for why this is a fixed STRIDE across every shard rather
    # than just the last few: glot500-scale corpora have per-language sizes
    # spanning orders of magnitude, and _round_robin drops each language
    # once its own data is exhausted, so later shards in a long run skew
    # toward whichever languages had the most data -- a stride sampled
    # across the whole sequence avoids that skew.
    eval_interval: int = 1000  # run validation every this many steps (0 disables)
    eval_iters: int = 50  # validation batches averaged per validation pass --
    # a FIXED validation set (same eval_iters batches every single call, not
    # freshly reshuffled), so consecutive validation passes measure actual
    # model improvement rather than evaluation-sampling noise.
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


def _existing_checkpoints(output_dir):
    """Every step_{N}.pt already in output_dir (e.g. left over from an
    earlier --resume-from segment of the same multi-day run), sorted
    oldest-first by step number -- excludes final.pt (never rotated, see
    keep_last_n_checkpoints's own docstring), and anything that doesn't
    parse as step_{int}.pt.

    Used to SEED checkpoint rotation's bookkeeping at startup rather than
    starting it empty every process restart: rotation that resets on every
    resume can only ever delete checkpoints saved within ITS OWN lifetime,
    letting every PRIOR segment's checkpoints accumulate on disk forever
    across a long multi-resubmit run (jobs/train_pretraining.sh's own
    auto-resubmit convention means a real "large"-preset run spans roughly
    10 such segments) -- the exact same failure mode that already caused a
    real disk-quota crash once (see keep_last_n_checkpoints's own comment),
    just not yet triggered a second time because it needs several resume
    cycles to compound."""
    if not os.path.isdir(output_dir):
        return []
    found = []
    for name in os.listdir(output_dir):
        if name.startswith("step_") and name.endswith(".pt"):
            try:
                step_num = int(name[len("step_") : -len(".pt")])
            except ValueError:
                continue
            found.append((step_num, os.path.join(output_dir, name)))
    return [path for _, path in sorted(found)]


def _split_train_val_shard_files(shard_files, val_fraction):
    """Reserves roughly val_fraction of shard_files for held-out validation,
    spread across the WHOLE corpus via a fixed stride rather than just the
    tail (see TrainConfig.val_fraction's own docstring for why: glot500's
    per-language corpus sizes span orders of magnitude -- confirmed live,
    from a few hundred to hundreds of thousands of documents -- and
    common.data.corpora._round_robin drops each language once its own data
    is exhausted, so shards written later in a long run skew toward
    whichever languages had the most data; a stride sampled across the
    whole shard sequence avoids that skew).

    Returns (train_files, val_files). Validation is disabled (val_files
    empty, every shard goes to train_files) if val_fraction <= 0 or there
    are too few shards to spare any without leaving training with none --
    a real edge case for a tiny/smoke-test prep run, not just a defensive
    formality. stride is bounded below at 2, so index 1 always survives
    into train_files for any len(shard_files) >= 2 -- train_files can
    never come back empty here."""
    if val_fraction <= 0 or len(shard_files) < 2:
        return list(shard_files), []
    stride = max(round(1 / val_fraction), 2)
    val_files = shard_files[::stride]
    train_files = [f for i, f in enumerate(shard_files) if i % stride != 0]
    return train_files, val_files


@torch.no_grad()
def validate(model, val_loader, device, amp_dtype):
    """Averages cross-entropy loss over val_loader's own fixed set of
    batches (see TrainConfig.eval_iters's own docstring for why this is a
    FIXED set, re-iterated identically on every call, not freshly
    reshuffled). Restores the model's prior train()/eval() mode before
    returning, so this is safe to call mid-training-loop without disturbing
    dropout/etc. state for the next training step (irrelevant at this
    project's own dropout=0.0 default, but not assumed here)."""
    was_training = model.training
    model.eval()
    total_loss = 0.0
    num_batches = 0
    for x, y in val_loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type if device.type != "cpu" else "cpu",
            dtype=amp_dtype,
            enabled=amp_dtype != torch.float32,
        ):
            _, loss = model(x, labels=y)
        total_loss += loss.item()
        num_batches += 1
    if was_training:
        model.train()
    return total_loss / num_batches if num_batches else float("nan")


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

    # Split BEFORE building either dataset: train_shard_files/val_shard_files
    # partition shard_dir's own shard list disjointly (see
    # _split_train_val_shard_files's own docstring) -- val shards are never
    # sampled for training.
    train_shard_files, val_shard_files = _split_train_val_shard_files(
        meta["shard_files"], cfg.val_fraction
    )
    if is_main:
        if val_shard_files:
            print(
                f"validation: {len(val_shard_files)}/{len(meta['shard_files'])} shards reserved, "
                f"evaluating {cfg.eval_iters} batches every {cfg.eval_interval} steps"
            )
        else:
            print("validation: disabled (val_fraction<=0 or too few shards to spare any)")

    samples_per_step = cfg.grad_accum_steps * cfg.per_device_batch_size
    total_samples = cfg.total_steps * samples_per_step
    samples_already_consumed = start_step * samples_per_step
    dataset = ShardedTokenDataset(
        cfg.shard_dir,
        cfg.seq_len,
        num_samples=total_samples - samples_already_consumed,
        seed=cfg.seed + rank,
        index_offset=samples_already_consumed,
        shard_files=train_shard_files,
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg.per_device_batch_size,
        num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda",
    )

    # Validation only ever runs on is_main (see the main loop below) -- one
    # small, fixed-size dataset/loader built once here, not per-rank and not
    # resumed/offset like the training set (a fixed seed with no
    # rank/start_step dependence means every validation pass across the
    # whole run, and across any --resume-from restart, evaluates the exact
    # same samples, so the val-loss curve is directly comparable over time).
    val_loader = None
    if val_shard_files:
        val_dataset = ShardedTokenDataset(
            cfg.shard_dir,
            cfg.seq_len,
            num_samples=cfg.eval_iters * cfg.per_device_batch_size,
            seed=cfg.seed + 999_983,  # arbitrary constant, deliberately NOT
            # rank- or start_step-dependent -- see the paragraph above
            shard_files=val_shard_files,
        )
        val_loader = DataLoader(
            val_dataset, batch_size=cfg.per_device_batch_size, num_workers=0,
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
    # step_{step}.pt paths tracked for rotation, oldest first -- seeded from
    # whatever's ALREADY in output_dir (e.g. from an earlier --resume-from
    # segment), not started empty, so keep_last_n_checkpoints correctly
    # rotates across process restarts too, not just within this one's own
    # lifetime (see _existing_checkpoints's own docstring for why this
    # matters concretely for this project's multi-resubmit training runs).
    saved_checkpoint_paths = _existing_checkpoints(cfg.output_dir) if is_main else []

    while step < cfg.total_steps:
        optimizer.zero_grad(set_to_none=True)
        loss_accum = 0.0
        for micro_step in range(cfg.grad_accum_steps):
            x, y = next(data_iter)
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            is_last_micro_step = micro_step == cfg.grad_accum_steps - 1
            # Under DDP, backward() all-reduces gradients EVERY call by
            # default -- wasteful when grad_accum_steps > 1, since only the
            # LAST micro-step's accumulated gradient actually needs to be
            # synced before optimizer.step(). model.no_sync() (only
            # meaningful on a DistributedDataParallel-wrapped model) skips
            # that all-reduce on every micro-step except the last one --
            # matches lit-llama's own fabric.no_backward_sync(model,
            # enabled=is_accumulating) convention. A no-op contextlib.nullcontext()
            # on the last micro-step or when not distributed (single-GPU
            # has no gradient sync to skip in the first place).
            sync_ctx = (
                model.no_sync()
                if distributed and not is_last_micro_step
                else contextlib.nullcontext()
            )
            with sync_ctx:
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

        # Gated the same way on EVERY rank (not just is_main), even though
        # only is_main actually runs validate()/logs -- the dist.barrier()
        # below must be reached by every rank in lockstep, or ranks that
        # never call it would wait forever for one that does.
        if val_loader is not None and cfg.eval_interval and step % cfg.eval_interval == 0:
            if is_main:
                val_loss = validate(model, val_loader, device, amp_dtype)
                print(f"[step {step:6d}/{cfg.total_steps}] val_loss={val_loss:.4f}")
                if run is not None:
                    run.log({"val/loss": val_loss}, step=step)
            if distributed:
                dist.barrier()  # other ranks wait for is_main's validation
                # pass to finish, rather than racing ahead into the next
                # step's own backward() -- explicit here rather than relying
                # on that next collective to implicitly serve as the wait.

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
