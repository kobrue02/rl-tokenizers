"""The boundary-placement policy: a small causal byte-level model.

At every byte position it outputs three things:
  - a boundary logit (sampled via Bernoulli -> the discrete, non-differentiable action
    that REINFORCE is needed for), from the main (boundary-aware) hidden state
  - a next-byte logit from that same main hidden state (dense, differentiable --
    trained directly by gradient descent, not by the score-function estimator)
  - an early-exit next-byte logit from a SEPARATE, boundary-agnostic hidden state
    (see the early-exit baseline note on BytePolicy below)

Context is a GRU hidden state fed by the current byte plus the previous boundary
decision (a window of 1 prior boundary, per Dauncey & Wattenhofer's finding that this
performs comparably to a window of 8 -- keep it small).
"""

from dataclasses import dataclass

import torch
import torch.nn as nn

from common.bytes_utils import bytes_to_tensor, spans_from_boundaries  # noqa: F401 -- re-exported
# for backward-compat call sites (`from fairtok.policy import bytes_to_tensor, ...`); this
# module also uses spans_from_boundaries itself (segment_bytes, below).


class BytePolicy(nn.Module):
    """Early-exit baseline adapted from Dauncey & Wattenhofer, "You Can Learn
    Tokenization End-to-End with Reinforcement Learning"
    (github.com/SamD770/bitter-lesson-tokenization, model/model.py
    `per_token_losses_backbone` / Eq. 14-15 of the paper): a second, SEPARATE
    byte-level path (`early_cell`/`early_byte_head`) that predicts the next byte
    from raw byte content alone, with no access to the boundary/tokenization
    decisions at all. Its prediction quality is a baseline for "how predictable
    is this byte regardless of tokenization" -- subtracting it from the main
    head's prediction quality (done in policy.py's batched_sample_rollout and used as
    the reward in reward.py) isolates how much the boundary policy itself is
    contributing, rather than raw byte predictability the boundary decisions
    had nothing to do with.

    This is an adaptation, not a literal port: D&W's early/late predictions come
    from the SAME multi-layer transformer encoder, diverging only at the final
    unembedding matrix (before vs. after their U-Net's downsample/mid/upsample
    round-trip) -- there's no such shared representation to branch from in our
    single-GRU-cell architecture, so this uses a genuinely separate recurrent
    path instead. The unembedding weights are still tied at initialization
    (matching D&W's initialization trick "to facilitate easy transfer"), then
    allowed to diverge during training. This roughly doubles per-byte compute
    (two GRU cells instead of one).
    """

    def __init__(self, hidden_dim=64, num_layers=2, use_prev_boundary=True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # Diagnostic-only flag (see conversation: "does removing the prev_boundary
        # feedback break the early-exit reward signal?"): when False, the main path's
        # input always looks up boundary_embed(0) -- a constant vector, same shape,
        # zero information -- instead of the real previous action. This isolates
        # exactly that one question, without also changing architecture family
        # (attention/CNN vs recurrence), depth, or receptive field, which a full
        # parallel-architecture rewrite would confound it with. Not meant to be a
        # real training config knob -- default True preserves normal behavior.
        self.use_prev_boundary = use_prev_boundary
        self.byte_embed = nn.Embedding(256, hidden_dim)
        self.boundary_embed = nn.Embedding(2, hidden_dim)

        # Stacked GRU cells, not a single one -- the first layer sees the
        # byte+boundary embedding, every later layer sees the previous layer's
        # hidden output. Compute per byte position scales roughly with
        # num_layers * hidden_dim^2, so widening and deepening together (e.g.
        # the default 32->64, 1->2 bump) costs roughly an order of magnitude
        # more than the original single-cell, hidden_dim=32 policy.
        self.cells = nn.ModuleList([
            nn.GRUCell(hidden_dim * 2 if i == 0 else hidden_dim, hidden_dim)
            for i in range(num_layers)
        ])
        self.boundary_head = nn.Linear(hidden_dim, 1)
        self.byte_head = nn.Linear(hidden_dim, 256)

        self.early_cells = nn.ModuleList([nn.GRUCell(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.early_byte_head = nn.Linear(hidden_dim, 256)
        with torch.no_grad():
            self.early_byte_head.weight.copy_(self.byte_head.weight)
            self.early_byte_head.bias.copy_(self.byte_head.bias)

    def init_state(self, device, batch_size=1):
        hidden = [torch.zeros(batch_size, self.hidden_dim, device=device) for _ in range(self.num_layers)]
        early_hidden = [torch.zeros(batch_size, self.hidden_dim, device=device) for _ in range(self.num_layers)]
        return hidden, early_hidden

    def step(self, byte_id, prev_boundary, hidden, early_hidden):
        byte_x = self.byte_embed(byte_id)
        if not self.use_prev_boundary:
            prev_boundary = torch.zeros_like(prev_boundary)
        x = torch.cat([byte_x, self.boundary_embed(prev_boundary)], dim=-1)

        new_hidden = []
        layer_input = x
        for cell, h_prev in zip(self.cells, hidden):
            h = cell(layer_input, h_prev)
            new_hidden.append(h)
            layer_input = h

        new_early_hidden = []
        early_input = byte_x
        for cell, h_prev in zip(self.early_cells, early_hidden):
            h = cell(early_input, h_prev)
            new_early_hidden.append(h)
            early_input = h

        boundary_logit = self.boundary_head(new_hidden[-1]).squeeze(-1)
        byte_logit = self.byte_head(new_hidden[-1])
        early_byte_logit = self.early_byte_head(new_early_hidden[-1])
        return boundary_logit, byte_logit, early_byte_logit, new_hidden, new_early_hidden


@dataclass
class StepRecord:
    boundary_action: int
    boundary_logit: torch.Tensor  # differentiable, raw (pre-sigmoid) -- used by the
    # direct rate-consistency loss in train.py, not just the sampled action's logprob
    boundary_logprob: torch.Tensor  # differentiable -- score-function weight target
    next_byte_logprob: torch.Tensor  # differentiable -- main head, trained as an aux ML loss
    early_byte_logprob: torch.Tensor  # differentiable -- early-exit head, ALSO trained as
    # its own aux ML loss; (next_byte_logprob - early_byte_logprob) is R_predict (see reward.py)
    byte_correct: bool  # None at the last position (no next byte to predict)
    predict_reward: float  # = next_byte_logprob - early_byte_logprob, already a plain host
    # float -- precomputed and pulled off the device ONCE for the whole batch in
    # batched_sample_rollout (see actions_np/correct_np there), so reward.py's
    # build_rewards no longer needs its own per-(group, language) .cpu() sync


def batched_sample_rollout(policy, byte_seqs, device="cpu", deterministic=False):
    """byte_seqs: list of 1-D LongTensors (variable length, one per sequence in the
    batch). Replaces the old one-sequence-at-a-time sample_rollout: that version had
    terrible GPU utilization (a Python loop calling the GRU with batch size 1, once
    per byte, once per sequence -- kernel-launch overhead dominates on GPU with
    nothing to actually parallelize). This still loops sequentially over TIME (each
    step's input genuinely depends on the previous step's sampled action, so that
    part can't be removed) but processes ALL sequences in the batch TOGETHER at each
    time step via padding + masking, turning sum(len(s) for s in byte_seqs)
    individual small forward passes into max(len(s) for s in byte_seqs) batched ones.

    deterministic: threshold the boundary probability at 0.5 instead of sampling from
    it -- same distinction as segment_bytes's own `deterministic` flag, for the same
    reason (train.py's GRPOTrainer.evaluate wants a reproducible boundary policy when
    scoring a held-out eval set, not stochastic exploration). Everything else about
    the rollout (logprobs, byte predictions) is computed identically either way; eval
    callers just don't read those fields.

    Returns: list of per-sequence lists of StepRecord, same order/length as byte_seqs
    -- downstream code (reward.py, spans_from_boundaries, compression_rate, ...)
    is unchanged, since it only ever consumed per-sequence StepRecord lists anyway.
    """
    B = len(byte_seqs)
    lengths = [int(s.shape[0]) for s in byte_seqs]
    T = max(lengths)

    padded = torch.zeros(B, T, dtype=torch.long, device=device)
    for b, seq in enumerate(byte_seqs):
        padded[b, :lengths[b]] = seq

    hidden, early_hidden = policy.init_state(device, batch_size=B)
    prev_boundary = torch.zeros(B, dtype=torch.long, device=device)
    batch_idx = torch.arange(B, device=device)

    per_step = []  # per_step[t] = dict of (B,)-shaped tensors for time step t
    for t in range(T):
        byte_ids = padded[:, t]
        boundary_logit, byte_logit, early_byte_logit, hidden, early_hidden = policy.step(
            byte_ids, prev_boundary, hidden, early_hidden
        )
        prob = torch.sigmoid(boundary_logit)
        action = (prob > 0.5).long() if deterministic else torch.bernoulli(prob).long()
        boundary_logprob = torch.where(action.bool(), torch.log(prob + 1e-8), torch.log(1 - prob + 1e-8))

        # target byte for t+1, clamped in-bounds; per-sequence validity (has a real
        # next byte) is checked per-sequence below via `lengths`, not here -- this
        # gather is computed for every sequence regardless, garbage values for
        # sequences where it's invalid are simply never read afterward
        next_col = min(t + 1, T - 1)
        target = padded[:, next_col]
        logp_byte = torch.log_softmax(byte_logit, dim=-1)
        next_byte_logprob = logp_byte[batch_idx, target]
        early_logp_byte = torch.log_softmax(early_byte_logit, dim=-1)
        early_byte_logprob = early_logp_byte[batch_idx, target]
        byte_correct = byte_logit.argmax(dim=-1) == target
        # detached: this is R_predict (see reward.py), consumed as a plain reward
        # number, never backpropagated through -- the differentiable copies of these
        # same two logprobs are what train.py's nll_loss/early_nll_loss train on
        predict_reward = (next_byte_logprob - early_byte_logprob).detach()

        per_step.append({
            "action": action, "boundary_logit": boundary_logit, "boundary_logprob": boundary_logprob,
            "next_byte_logprob": next_byte_logprob, "early_byte_logprob": early_byte_logprob,
            "byte_correct": byte_correct, "predict_reward": predict_reward,
        })
        prev_boundary = action

    # Pull `action`/`byte_correct`/`predict_reward` off the device ONCE each (three
    # syncs total), not once per (sequence, position) pair -- on GPU, every individual
    # .item()/.tolist() call is a full host-device synchronization, and B*T (or, for
    # predict_reward before this fix, one per (group, language) in train.py's reward
    # loop) of those previously dominated wall-clock time far more than the actual GRU
    # compute did. Everything still needed as a differentiable tensor (boundary_logit,
    # boundary_logprob, next/early_byte_logprob) stays on-device, untouched.
    actions_np = torch.stack([s["action"] for s in per_step]).cpu().numpy()  # (T, B)
    correct_np = torch.stack([s["byte_correct"] for s in per_step]).cpu().numpy()  # (T, B)
    reward_np = torch.stack([s["predict_reward"] for s in per_step]).cpu().numpy()  # (T, B)

    results = []
    for b in range(B):
        L = lengths[b]
        records = []
        for t in range(L):
            has_next = t + 1 < L
            step = per_step[t]
            records.append(StepRecord(
                int(actions_np[t, b]),
                step["boundary_logit"][b],
                step["boundary_logprob"][b],
                step["next_byte_logprob"][b] if has_next else torch.zeros((), device=device),
                step["early_byte_logprob"][b] if has_next else torch.zeros((), device=device),
                bool(correct_np[t, b]) if has_next else None,
                float(reward_np[t, b]) if has_next else 0.0,
            ))
        results.append(records)
    return results


@torch.no_grad()
def segment_bytes(policy, byte_seq, deterministic=True, device="cpu"):
    """Inference-only rollout for a FROZEN policy: no logprobs, no autograd graph --
    used to harvest a vocabulary from a real corpus after training, not during it.
    `deterministic` thresholds the boundary probability at 0.5 instead of sampling
    from it: training needs stochastic exploration, but building a production
    vocabulary should be reproducible given the same policy and corpus.
    The early-exit head is irrelevant at inference (it only exists to shape the
    reward during training), so its state is threaded through but ignored."""
    hidden, early_hidden = policy.init_state(device)
    prev_boundary = torch.zeros(1, dtype=torch.long, device=device)
    action_tensors = []  # kept on-device; pulled off in ONE sync after the loop below,
    # not via a per-position int(action.item()) -- same fix as batched_sample_rollout's
    # actions_np, applied here since this is still a per-byte sequential loop (the
    # boundary decision at t genuinely depends on the sampled action at t-1) even
    # though it's inference-only and batch-size-1.
    for t in range(byte_seq.shape[0]):
        boundary_logit, _, _, hidden, early_hidden = policy.step(
            byte_seq[t].view(1), prev_boundary, hidden, early_hidden
        )
        prob = torch.sigmoid(boundary_logit)
        action = (prob > 0.5).long() if deterministic else torch.bernoulli(prob).long()
        action_tensors.append(action)
        prev_boundary = action
    actions = torch.cat(action_tensors).tolist() if action_tensors else []
    return spans_from_boundaries(byte_seq, actions)
