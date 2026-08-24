"""MLM training loop for systems.pretraining.encoder_model's from-scratch
XLM-R-architecture encoder, over the same packed token shards data_prep.py
builds for the decoder (see encoder_data.py -- masking is a batch-time
transform, the shard format itself is unchanged).

Mirrors systems.pretraining.train's shape (a TrainConfig-style dataclass,
DDP via torchrun, checkpoint rotation, wandb logging, a fixed validation
set) but narrower in scope:

  - DDP only, no FSDP. The "large" (XLM-R-large, ~303M transformer-body
    params) encoder preset is small enough that full replication under DDP
    is unproblematic on a single modern GPU -- nowhere near the decoder's
    "7b" preset's own motivation for FSDP (see train.py's module docstring).
    Extending this to FSDP later would follow the same
    torch.distributed.fsdp.fully_shard pattern as train.wrap_fsdp, just
    targeting model.roberta.encoder.layer (HF's module tree) instead of
    model.blocks -- not built here since nothing at this scale needs it yet.
  - No --compile/torch.compile support, no generate_interval (no text-
    generation analogue for an MLM model to qualitatively spot-check), no
    grad_checkpointing knob (HF's model exposes
    model.gradient_checkpointing_enable() directly if a future larger
    preset needs it).

Reuses train.py's is_distributed/setup_distributed and
_split_train_val_shard_files directly (torchrun-detection and the
stride-based train/val shard split are architecture-independent) rather
than duplicating them.
"""

import contextlib
import dataclasses
import functools
import os
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from common.training.lr_schedule import build_lr_scheduler

from .encoder_data import MLMShardedTokenDataset, mlm_collate_fn
from .encoder_model import build_encoder
from .encoder_model_configs import get_preset
from .encoder_tokenizer import RESERVED
from .shard_dataset import load_shard_meta
from .train import _existing_checkpoints, _split_train_val_shard_files, is_distributed, setup_distributed


@dataclasses.dataclass
class EncoderTrainConfig:
    encoder_size: str = "base"  # key into encoder_model_configs.PRESETS
    shard_dir: str = ""  # required -- same shard format data_prep.py builds
    # for the decoder; vocab_size is read from shard_dir/shards_meta.json
    # (mask_id/pad_id are +0/+1 past it -- see encoder_tokenizer.EncoderVocab),
    # not set here, so the embedding table always matches the tokenizer used.
    seq_len: int = 512  # RoBERTa/XLM-R's own default, half the decoder's
    # 1024 default -- MLM's bidirectional context makes shorter sequences
    # less of an information bottleneck than a decoder's causal context
    mlm_probability: float = 0.15  # standard BERT/RoBERTa masking rate
    per_device_batch_size: int = 8
    grad_accum_steps: int = 1  # effective batch size = per_device_batch_size
    # * grad_accum_steps * world_size (world_size=1 outside torchrun)
    total_steps: int = 10_000
    learning_rate: float = 5e-5  # matches Glot500-m's own continued-
    # pretraining LR (modeling/train_base.sh)
    weight_decay: float = 0.01  # HF Trainer's own AdamW default -- unlike
    # the decoder's 0.1 (borrowed from GPT-3/LLaMA practice), this pass
    # follows Glot500/BERT-style MLM training convention throughout
    beta1: float = 0.9
    beta2: float = 0.999  # HF Trainer's/PyTorch's own AdamW default (vs.
    # the decoder's 0.95 -- see train.TrainConfig's own comment on why THAT
    # differs; MLM training doesn't show the same early-training gradient-
    # scale swings large-batch causal LM pretraining does)
    grad_clip: float = 1.0
    warmup_ratio: float = 0.06  # RoBERTa/XLM-R's own published warmup fraction
    lr_scheduler_type: str = "linear"  # HF Trainer's own default, matches
    # Glot500-m's own modeling/run.py (unmodified from upstream run_mlm.py)
    log_steps: int = 10
    val_fraction: float = 0.05  # see train.TrainConfig.val_fraction's own
    # docstring -- identical stride-based reasoning, reused via
    # train._split_train_val_shard_files directly
    eval_interval: int = 1000
    eval_iters: int = 50
    save_steps: int = 1000
    keep_last_n_checkpoints: int = 3
    output_dir: str = "checkpoints/encoder_pretrain"
    resume_from: str = ""  # "" starts fresh; else a checkpoint path saved
    # by this same script (see save_checkpoint/load_checkpoint below)
    seed: int = 0
    device: str = ""  # "" auto-detects cuda if available, else cpu; ignored
    # under torchrun/DDP, which uses the LOCAL_RANK-assigned GPU instead
    dtype: str = "bfloat16"  # or "float32" (CPU/debugging)
    num_workers: int = 4
    use_wandb: bool = False
    wandb_project: str = "pretraining"
    run_name: str = ""


def build_mlm_loader(dataset, batch_size, real_vocab_size, mlm_probability, num_workers, pin_memory):
    """DataLoader with masking baked into collate_fn (functools.partial,
    not a closure -- must be picklable for num_workers > 0's worker
    processes) so it runs IN worker processes when num_workers > 0,
    overlapping with the main process's GPU compute, rather than
    synchronously in the training loop."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=functools.partial(
            mlm_collate_fn, real_vocab_size=real_vocab_size, mlm_probability=mlm_probability
        ),
    )


def _autocast_ctx(device, amp_dtype):
    return torch.autocast(
        device_type=device.type if device.type != "cpu" else "cpu",
        dtype=amp_dtype,
        enabled=amp_dtype != torch.float32,
    )


@torch.no_grad()
def validate(model, val_loader, device, amp_dtype):
    """Averages MLM cross-entropy loss over val_loader's own fixed set of
    windows -- masking is still randomly resampled each call (mlm_collate_fn
    runs inside val_loader's own collate_fn, see build_mlm_loader), so
    consecutive validation passes aren't run over IDENTICAL masked positions
    the way train.py's decoder validate() re-iterates identical (x, y)
    pairs; the windows themselves are fixed (val_loader's own dataset has a
    fixed seed), only which tokens get masked within them varies run to
    run. Restores the model's prior train()/eval() mode before returning."""
    was_training = model.training
    model.eval()
    total_loss = 0.0
    num_batches = 0
    for input_ids, labels in val_loader:
        input_ids, labels = input_ids.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        with _autocast_ctx(device, amp_dtype):
            loss = model(input_ids=input_ids, labels=labels).loss
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


def train(cfg: EncoderTrainConfig):
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

    torch.manual_seed(cfg.seed + rank)

    meta = load_shard_meta(cfg.shard_dir)
    real_vocab_size = meta["vocab_size"]
    encoder_vocab_size = real_vocab_size + RESERVED

    preset = get_preset(cfg.encoder_size)
    preset.max_seq_len = max(preset.max_seq_len, cfg.seq_len)
    model = build_encoder(preset, encoder_vocab_size).to(device)
    model_params = sum(p.numel() for p in model.parameters())
    effective_batch_size = cfg.per_device_batch_size * cfg.grad_accum_steps * world_size
    if is_main:
        print(
            f"encoder_size={cfg.encoder_size} params={model_params:,} "
            f"vocab_size={encoder_vocab_size} (from {cfg.shard_dir}/shards_meta.json, "
            f"+2 for mask/pad) world_size={world_size} effective_batch_size={effective_batch_size}"
        )

    amp_dtype = torch.bfloat16 if cfg.dtype == "bfloat16" else torch.float32
    if distributed:
        model = DistributedDataParallel(model, device_ids=[local_rank] if device.type == "cuda" else None)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        betas=(cfg.beta1, cfg.beta2),
        weight_decay=cfg.weight_decay,
        fused=(device.type == "cuda"),
    )
    scheduler = build_lr_scheduler(optimizer, cfg.total_steps, cfg.warmup_ratio, cfg.lr_scheduler_type)

    start_step = 0
    if cfg.resume_from:
        start_step = load_checkpoint(cfg.resume_from, model, optimizer, scheduler, device)
        if is_main:
            print(f"resumed from {cfg.resume_from} at step {start_step}")

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
    dataset = MLMShardedTokenDataset(
        cfg.shard_dir,
        cfg.seq_len,
        num_samples=total_samples - samples_already_consumed,
        seed=cfg.seed + rank,
        index_offset=samples_already_consumed,
        shard_files=train_shard_files,
    )
    loader = build_mlm_loader(
        dataset, cfg.per_device_batch_size, real_vocab_size, cfg.mlm_probability,
        cfg.num_workers, pin_memory=device.type == "cuda",
    )

    val_loader = None
    if val_shard_files:
        val_dataset = MLMShardedTokenDataset(
            cfg.shard_dir,
            cfg.seq_len,
            num_samples=cfg.eval_iters * cfg.per_device_batch_size,
            seed=cfg.seed + 999_983,  # see train.TrainConfig.val_fraction's
            # own docstring -- arbitrary constant, deliberately rank/step-independent
            shard_files=val_shard_files,
        )
        val_loader = build_mlm_loader(
            val_dataset, cfg.per_device_batch_size, real_vocab_size, cfg.mlm_probability,
            num_workers=0, pin_memory=device.type == "cuda",
        )

    run = None
    if cfg.use_wandb and is_main:
        import wandb

        run = wandb.init(
            project=cfg.wandb_project,
            name=cfg.run_name or None,
            job_type="encoder_train",
            config={
                **dataclasses.asdict(cfg),
                "vocab_size": encoder_vocab_size,
                "model_params": model_params,
                "world_size": world_size,
                "effective_batch_size": effective_batch_size,
            },
        )

    if is_main:
        os.makedirs(cfg.output_dir, exist_ok=True)

    model.train()
    data_iter = iter(loader)
    step = start_step
    t_last_log = time.time()
    tokens_since_log = 0
    overhead_since_log = 0.0  # wall-clock spent in validate()/save_checkpoint()
    # since the last log -- excluded from the tok/s window below. Without
    # this, a validation/save that runs right after a log point (t_last_log
    # resets there, BEFORE eval/save execute) silently bleeds its own wall
    # time into the FOLLOWING window's dt, deflating that window's reported
    # throughput even though no training slowdown actually happened
    # (confirmed against a real decoder run, train.py's own identical bug:
    # tok/s alternated ~90K/~35K in exact lockstep with eval_interval/
    # save_steps boundaries -- the true sustained rate was the high number
    # throughout).
    saved_checkpoint_paths = _existing_checkpoints(cfg.output_dir) if is_main else []

    while step < cfg.total_steps:
        optimizer.zero_grad(set_to_none=True)
        loss_accum = 0.0
        for micro_step in range(cfg.grad_accum_steps):
            input_ids, labels = next(data_iter)
            input_ids, labels = input_ids.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            is_last_micro_step = micro_step == cfg.grad_accum_steps - 1
            sync_ctx = (
                model.no_sync()
                if distributed and not is_last_micro_step
                else contextlib.nullcontext()
            )
            with sync_ctx:
                with _autocast_ctx(device, amp_dtype):
                    loss = model(input_ids=input_ids, labels=labels).loss
                (loss / cfg.grad_accum_steps).backward()
            loss_accum += loss.item() / cfg.grad_accum_steps
            tokens_since_log += input_ids.numel() * world_size

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        scheduler.step()
        step += 1

        if is_main and step % cfg.log_steps == 0:
            dt = time.time() - t_last_log - overhead_since_log
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
            overhead_since_log = 0.0

        if val_loader is not None and cfg.eval_interval and step % cfg.eval_interval == 0 and is_main:
            _overhead_start = time.time()
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            val_loss = validate(raw_model, val_loader, device, amp_dtype)
            print(f"[step {step:6d}/{cfg.total_steps}] val_loss={val_loss:.4f}")
            if run is not None:
                run.log({"val/loss": val_loss}, step=step)
            if distributed:
                dist.barrier()  # other ranks wait for is_main's validation
                # pass to finish, rather than racing ahead into the next
                # step's backward() -- see train.py's identical DDP reasoning
            overhead_since_log += time.time() - _overhead_start
        elif val_loader is not None and cfg.eval_interval and step % cfg.eval_interval == 0 and distributed:
            dist.barrier()

        if cfg.save_steps and step % cfg.save_steps == 0 and is_main:
            _overhead_start = time.time()
            path = os.path.join(cfg.output_dir, f"step_{step}.pt")
            save_checkpoint(path, model, optimizer, scheduler, step, cfg, encoder_vocab_size)
            print(f"saved checkpoint to {path}")
            saved_checkpoint_paths.append(path)
            if cfg.keep_last_n_checkpoints > 0:
                while len(saved_checkpoint_paths) > cfg.keep_last_n_checkpoints:
                    stale_path = saved_checkpoint_paths.pop(0)
                    os.remove(stale_path)
                    print(f"removed older checkpoint {stale_path} (keeping last {cfg.keep_last_n_checkpoints})")
            overhead_since_log += time.time() - _overhead_start

    if is_main:
        final_path = os.path.join(cfg.output_dir, "final.pt")
        save_checkpoint(final_path, model, optimizer, scheduler, step, cfg, encoder_vocab_size)
        print(f"training complete, saved final checkpoint to {final_path}")
        if run is not None:
            run.finish()

    if distributed:
        dist.barrier()
        dist.destroy_process_group()
