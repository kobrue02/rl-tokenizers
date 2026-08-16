"""Reward assembly and GRPO-style group-relative advantage.

Credit assignment: R_predict is dense (per-step). The fairness penalty is a
property of the whole running corpus, not attributable to one byte decision, so
it's added once at the final step and discounting propagates it backward via the
ordinary reward-to-go computation. (The target-rate penalty used to live here the
same way; it has since moved to train.py's differentiable rate-consistency loss,
which doesn't need this reward-shaping workaround.)
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
    (github.com/SamD770/bitter-lesson-tokenization, Eq. 15): reward =
    log p_late(byte) - log p_early(byte), not raw next-byte log-prob. This isolates
    how much the boundary policy itself improved prediction, beyond what the
    boundary-agnostic early-exit baseline already predicts -- fixes reward-attribution
    noise (initial REINFORCE losses were dominated by raw byte-predictability, not
    boundary quality).

    rec.predict_reward is already a plain host float, pulled off-device once for the
    whole batch in batched_sample_rollout -- this function does zero device syncs
    of its own (it used to, at per-(group, language)-per-step cost)."""
    rewards = [rec.predict_reward for rec in records]
    fairness_penalty = lambda_fair * fairness_scalar
    rewards[-1] -= fairness_penalty
    return rewards


def group_relative_advantage(returns_by_lang):
    """Group = one parallel sentence's translations across languages (not repeated
    rollouts of one input). Baseline is the group's mean per-step return, applied as
    a constant shift per trajectory -- valid (if not variance-optimal) since it never
    depends on the sampled action itself."""
    means = {lang: float(r.mean()) for lang, r in returns_by_lang.items()}
    baseline = float(np.mean(list(means.values())))
    return {lang: r - baseline for lang, r in returns_by_lang.items()}
