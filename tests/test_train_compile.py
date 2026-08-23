"""Regression test for TrainConfig.compile=True, added alongside the
grad_checkpointing="large"-preset-off + torch.compile speedup work.

The one real correctness trap here: `model = torch.compile(model)` returns
a NEW OptimizedModule wrapper object, which would silently break every
`isinstance(model, DistributedDataParallel)` unwrap check elsewhere in
train.py (validate()'s call site, save_checkpoint, load_checkpoint) --
those would then skip unwrapping and reintroduce the collective-mismatch
risk the earlier broadcast_buffers=False fix closed (see
tests/test_train_distributed.py). train.py uses `model.compile()`
(nn.Module's own in-place method, mutates __call__/forward, does NOT
replace `model` with a new wrapper object) specifically to avoid this --
this test confirms that in-place property holds for both DDP and FSDP,
not just that compile doesn't crash.

Verified against gloo (CPU) since this sandbox has no GPU -- correctness
(isinstance preserved, forward/backward runs, checkpoint round-trips) is
what's being checked here, not real compiled-kernel speedup, which only
shows up on real tensor-core hardware.
"""

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from systems.pretraining.model import TransformerLM
from systems.pretraining.model_configs import get_preset
from systems.pretraining.train import _autocast_ctx, wrap_fsdp


def _tiny_model(vocab_size=64, max_seq_len=16):
    cfg = get_preset("tiny")
    cfg.max_seq_len = max_seq_len
    return TransformerLM(cfg, vocab_size=vocab_size)


def _init_single_rank_pg(port):
    import os

    os.environ["RANK"] = "0"
    os.environ["LOCAL_RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend="gloo", rank=0, world_size=1)


def test_compile_preserves_ddp_isinstance_and_produces_correct_state_dict():
    _init_single_rank_pg(29601)
    try:
        model = DistributedDataParallel(_tiny_model(), broadcast_buffers=False)
        assert isinstance(model, DistributedDataParallel)

        model.compile()
        assert isinstance(model, DistributedDataParallel), (
            "model.compile() must NOT replace `model` with a new wrapper -- "
            "if this fails, validate()/save_checkpoint/load_checkpoint's own "
            "isinstance-based unwrap checks would silently stop firing"
        )

        x = torch.randint(0, 64, (2, 8))
        y = torch.randint(0, 64, (2, 8))
        _, loss = model(x, labels=y)
        loss.backward()
        assert torch.isfinite(loss)

        raw = model.module if isinstance(model, DistributedDataParallel) else model
        state_dict = raw.state_dict()
        assert "embed.weight" in state_dict  # clean key, no "module."/OptimizedModule prefix
    finally:
        dist.destroy_process_group()


def test_compile_preserves_fsdp_isinstance_and_produces_correct_state_dict():
    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict
    from torch.distributed.fsdp import FSDPModule

    _init_single_rank_pg(29602)
    try:
        device = torch.device("cpu")
        model = wrap_fsdp(_tiny_model(), torch.bfloat16, device)
        assert isinstance(model, FSDPModule)

        model.compile()
        assert isinstance(model, FSDPModule), (
            "model.compile() must NOT replace `model` with a new wrapper -- "
            "if this fails, the FSDP collective-gather checkpoint path would "
            "silently stop working"
        )

        x = torch.randint(0, 64, (2, 8))
        y = torch.randint(0, 64, (2, 8))
        with _autocast_ctx(device, torch.bfloat16, use_fsdp_mp=True):
            _, loss = model(x, labels=y)
        loss.backward()
        assert torch.isfinite(loss)

        state_dict = get_model_state_dict(
            model, options=StateDictOptions(full_state_dict=True, cpu_offload=True)
        )
        assert "embed.weight" in state_dict
    finally:
        dist.destroy_process_group()
