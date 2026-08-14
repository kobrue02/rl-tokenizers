"""Learning-rate scheduling, shared across every trainer in this repo -- matches
HuggingFace Trainer's own warmup_ratio/lr_scheduler_type convention (and default
values: warmup_ratio=0.1, lr_scheduler_type="linear"), implemented directly on
torch.optim.lr_scheduler.LambdaLR rather than depending on `transformers` for one
function's worth of formulas.

Motivation: an early FANTA training run (5 epochs, no scheduler -- flat learning
rate throughout) never reached a stable equilibrium, oscillating between a
character-level-collapse trap and wildly overshot compression rates (as high as
12x its target) for the entire run. A flat, un-decayed learning rate is a classic
contributor to exactly this kind of persistent oscillation late in training, once
the model is already in a reasonable region and large steps just bounce it back
out. Warmup (ramping UP from 0 at the start) additionally guards against early
instability while the model's untrained frontier/boundary predictor is producing
close-to-random signal.
"""

import math

import torch


def build_lr_scheduler(optimizer, total_steps, warmup_ratio=0.1, scheduler_type="linear"):
    """Returns a torch.optim.lr_scheduler.LambdaLR wrapping `optimizer`, scaling
    its base learning_rate by a factor in [0, 1] that:

      1. ramps LINEARLY from 0 up to 1.0 over the first `warmup_ratio *
         total_steps` steps (shared by all three scheduler_type values, matching
         HF's own get_*_schedule_with_warmup family), then
      2. depending on scheduler_type, either:
         - "constant": stays at 1.0 for the rest of training (warmup only, no decay)
         - "linear": decays linearly from 1.0 down to 0.0 by the last step
           (HF Trainer's own default -- get_linear_schedule_with_warmup)
         - "cosine": decays following a half-cosine from 1.0 down to 0.0
           (get_cosine_schedule_with_warmup)

    Caller is responsible for calling scheduler.step() once per optimizer.step()
    (see each trainer's own train() loop) -- LambdaLR doesn't do this on its own.
    """
    warmup_steps = max(1, int(warmup_ratio * total_steps)) if warmup_ratio > 0 else 0

    if scheduler_type not in ("constant", "linear", "cosine"):
        raise ValueError(
            f"unknown lr_scheduler_type {scheduler_type!r} -- expected one of "
            "'constant', 'linear', 'cosine'"
        )

    def lr_lambda(step):
        if warmup_steps and step < warmup_steps:
            return step / warmup_steps
        if scheduler_type == "constant":
            return 1.0
        # steps past warmup, as a [0, 1] progress fraction through the remainder
        remaining = max(1, total_steps - warmup_steps)
        progress = min(1.0, (step - warmup_steps) / remaining)
        if scheduler_type == "linear":
            return max(0.0, 1.0 - progress)
        return 0.5 * (1.0 + math.cos(math.pi * progress))  # "cosine"

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
