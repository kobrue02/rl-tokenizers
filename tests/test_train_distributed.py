"""Regression test for a real DDP deadlock found while auditing 8xA100
readiness: TransformerLM's Attention layers register rope_cos/rope_sin as
non-persistent buffers (model.py), and DistributedDataParallel's default
broadcast_buffers=True fires an unrequested buffer-broadcast collective on
the first forward after any grad-enabled step -- even under torch.no_grad()
-- because its require_forward_param_sync flag doesn't reset until AFTER
that forward runs (confirmed by reading torch's own
torch/nn/parallel/distributed.py: _post_forward only flips it back to False
when the CURRENT forward has grad enabled). train.py's validate() is called
on is_main only while every other rank waits at dist.barrier() -- a
mismatched collective. The fix (train.py): DistributedDataParallel(...,
broadcast_buffers=False) plus unwrapping model.module before validate().

Verified against gloo (CPU) since this sandbox has no GPU -- the underlying
DDP forward-hook logic that causes this is backend-agnostic; nccl is
train.py's own real backend. Empirically confirmed both directions before
writing this as a permanent test: the buggy pattern reliably times out
(~12s, no result written) and the fixed pattern reliably completes (~6s).

Completion is detected by polling for a result file, NOT by
ProcessContext.join(timeout=...)'s return value -- join() waits for process
EXIT, and this environment's process teardown lags behind actual work
completion by several seconds (independent of any deadlock), which made
join()'s return value unusable as a completion signal during development.
"""

import os
import tempfile
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

from systems.pretraining.model import TransformerLM
from systems.pretraining.model_configs import get_preset

WORLD_SIZE = 2
POLL_TIMEOUT = 10.0  # comfortably above the fixed pattern's observed ~6s
# completion time, comfortably below what would be needed to rule out a
# real hang (validated at 12s during development, see module docstring).


def _run_step_and_validate(rank, world_size, sync_file, broadcast_buffers, unwrap_for_validate, result_path):
    dist.init_process_group(
        backend="gloo", init_method=f"file://{sync_file}", rank=rank, world_size=world_size,
    )
    torch.manual_seed(0)
    model_cfg = get_preset("tiny")
    model_cfg.max_seq_len = 16
    model = TransformerLM(model_cfg, vocab_size=64)
    model = DistributedDataParallel(model, broadcast_buffers=broadcast_buffers)

    x = torch.randint(0, 64, (2, 8))
    y = torch.randint(0, 64, (2, 8))
    _, loss = model(x, labels=y)
    loss.backward()  # leaves require_forward_param_sync stale-True, like a real training step

    # train.py's own is_main-only validate() + all-rank dist.barrier() pattern.
    if rank == 0:
        eval_model = model.module if unwrap_for_validate else model
        with torch.no_grad():
            eval_model(x, labels=y)
    dist.barrier()

    if rank == 0:
        with open(result_path, "w") as f:
            f.write("completed")


def _spawn_and_wait(broadcast_buffers, unwrap_for_validate):
    with tempfile.TemporaryDirectory() as d:
        sync_file = os.path.join(d, "sync")
        result_path = os.path.join(d, "result")
        ctx = mp.spawn(
            _run_step_and_validate,
            args=(WORLD_SIZE, sync_file, broadcast_buffers, unwrap_for_validate, result_path),
            nprocs=WORLD_SIZE,
            join=False,
        )
        deadline = time.time() + POLL_TIMEOUT
        completed = False
        while time.time() < deadline:
            if os.path.exists(result_path):
                completed = True
                break
            if not any(p.is_alive() for p in ctx.processes):
                break  # exited without writing the result -- a real failure, not a hang
            time.sleep(0.2)
        for p in ctx.processes:
            if p.is_alive():
                p.terminate()
                p.join(timeout=5.0)
        return completed


def test_ddp_wrapped_validate_with_broadcast_buffers_deadlocks_without_the_fix():
    """The pre-fix pattern: DDP's own broadcast_buffers=True default, with
    the still-DDP-wrapped model passed into validate(). Must NOT complete --
    if this starts completing, either torch's DDP internals changed in a way
    that no longer reproduces the bug, or something masked the regression."""
    completed = _spawn_and_wait(broadcast_buffers=True, unwrap_for_validate=False)
    assert not completed


def test_unwrapped_validate_with_broadcast_buffers_disabled_completes():
    """train.py's actual fixed pattern."""
    completed = _spawn_and_wait(broadcast_buffers=False, unwrap_for_validate=True)
    assert completed
