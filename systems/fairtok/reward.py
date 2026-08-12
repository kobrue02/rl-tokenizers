"""Reward assembly and GRPO-style group-relative advantage.

Credit assignment: R_predict is dense (per-step, always available). The fairness
penalty is a property of the whole running corpus, so it isn't attributable to a
single byte decision -- it's added once, at the final step, and discounting
propagates that credit backward through the trajectory via the ordinary
reward-to-go computation. (The target-rate penalty that used to live here the
same way has moved out of the reward entirely -- see train.py's direct
rate-consistency loss, adapted from Dauncey & Wattenhofer's mechanism, which
doesn't need this reward-shaping workaround because it's differentiable.)
"""

import numpy as np


def discounted_returns(rewards, gamma):
    returns = np.zeros(len(rewards))
    running = 0.0
    for t in reversed(range(len(rewards))):
        running = rewards[t] + gamma * running
        returns[t] = running
    return returns


def build_rewards(records, fairness_scalar, lambda_fair):
    """R_predict is Dauncey & Wattenhofer's early-exit-relative reward
    (github.com/SamD770/bitter-lesson-tokenization, model/model.py
    `per_token_losses_backbone`, Eq. 15 of the paper): reward = log p_late(byte)
    - log p_early(byte), rather than raw next-byte log-prob. This isolates how
    much the boundary policy itself improved prediction, beyond what a
    boundary-agnostic baseline (see BytePolicy's early-exit head) could already
    predict -- directly targeting the reward-attribution noise problem that
    motivated this change (see conversation: initial REINFORCE loss magnitudes
    in the tens of thousands, dominated by raw byte-predictability, not by
    which boundary decisions were actually good).

    rec.predict_reward is already a plain host float -- batched_sample_rollout
    computes it on-device for the WHOLE batch and pulls it off with a single
    .cpu() call (same fix as actions_np/correct_np there), so this function does
    zero device syncs of its own. It used to do its own torch.stack(...).cpu()
    per call (i.e. once per (group, language) per step) -- on GPU that was still
    ~per_device_train_batch_size * group_sample_size syncs/step, on top of the much larger
    B*T ones already fixed in batched_sample_rollout."""
    rewards = [rec.predict_reward for rec in records]
    fairness_penalty = lambda_fair * fairness_scalar
    rewards[-1] -= fairness_penalty
    return rewards


def group_relative_advantage(returns_by_lang):
    """Group = one parallel sentence's translations across languages (not repeated
    rollouts of one input -- see the redefinition-of-"group" discussion). Baseline
    is the group's mean per-step return, applied as a constant shift per trajectory;
    this is a valid (if not variance-optimal) REINFORCE baseline since it never
    depends on the sampled action itself."""
    means = {lang: float(r.mean()) for lang, r in returns_by_lang.items()}
    baseline = float(np.mean(list(means.values())))
    return {lang: r - baseline for lang, r in returns_by_lang.items()}
