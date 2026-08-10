"""The Phase 1 training loop.

The objective, in plain terms: the policy is a "player" that walks through a
sentence byte by byte and decides, at each spot, "cut here to make a new token,
or keep going?" It's scored on three things at once:

  1. Can it predict what comes next? If it's good at guessing the next byte,
     that's a sign the current chunking makes sense (predictable chunks =
     good compression).
  2. Is it cutting about as often as normal BPE would? Not too fine-grained
     (chopping everything into single bytes) and not too coarse (barely
     cutting at all, everything one giant blob).
  3. Is it being roughly equally efficient across languages? If English gets
     nice short tokens but some other language needs way more tokens for the
     same content, that's penalized.

Sentences that are translations of each other (one sentence, many languages)
get processed together as a "group," and a language only gets a good score if
it does better than the group's average that round -- so languages aren't just
judged in isolation, they're judged relative to their siblings.

The fairness part (point 3) isn't checked every single step -- it's checked
periodically, based on everything the model has tokenized so far, not just the
current sentence. So the model doesn't get instant feedback on fairness; it's
more like a report card that updates every so often.

At the very end of training, whatever token pieces the model actually produced
get counted up, and only the most frequently-used ones (up to a fixed budget)
become the final vocabulary -- the model never gets to enforce that budget
while it's playing, only after the fact.

Mechanically: one step = one batch of parallel groups. Every language in a
group is rolled out before any reward is computed, because the fairness term
needs the whole group. The fairness scalar itself is refreshed only every
`fairness_refresh_every` steps, off the running per-language token-frequency
table -- not recomputed from the exact batch being rewarded, which is the
(partial) defense against the policy overfitting to whatever text happens to
be in one batch.
"""

import dataclasses
import math
from collections import Counter, defaultdict

import numpy as np
import torch
from tqdm.auto import tqdm

from .baseline_bpe import encode_with_merges, train_byte_bpe
from .data import LANG_PROFILES, make_synthetic_parallel_groups
from .inference import save_checkpoint
from .metrics import compression_rate, gini_coefficient, renyi_efficiency
from .policy import BytePolicy, batched_sample_rollout, bytes_to_tensor, spans_from_boundaries
from .reward import build_rewards, discounted_returns, group_relative_advantage
from .vocab import top_k_by_frequency, vocab_churn, vocab_snapshot_stats


@dataclasses.dataclass
class Config:
    num_steps: int = 0  # 0 means derive from num_epochs * steps_per_epoch; set explicitly
    # to override with a raw step count instead (bypasses epoch semantics entirely --
    # used by the smoke tests below for a fixed small number of steps regardless of corpus size).
    num_epochs: float = 3.0  # 1 epoch = 1 full pass over every group in train_groups,
    # via a shuffled traversal (see run_training) -- not a fixed step count, since that
    # depends on len(train_groups) and batch_groups, which vary by data source.
    batch_groups: int = 8
    hidden_dim: int = 128  # widened again from 64 (originally 32); GRU cost scales
    # roughly ~hidden_dim^2 per layer, so this alone is ~4x the (64, 2) setup's per-layer cost
    num_layers: int = 3  # stacked GRU cells for both the main and early-exit paths (see
    # BytePolicy) -- one deeper than the (64, 2) setup. Combined with the hidden_dim bump,
    # expect roughly ~6x the compute of the (64, 2) config, and roughly ~50-60x the
    # original (32, 1) single-cell policy this whole project started with.
    gamma: float = 0.99
    lambda_target: float = 2.0  # weight on the rate-consistency loss (see run_training) --
    # adapted from Dauncey & Wattenhofer's `consistency_loss_weight`
    # (github.com/SamD770/bitter-lesson-tokenization, model/model.py
    # `off_policy_flexible_training_step`), whose default of 2.0 this matches. This is no
    # longer a reward-shaping penalty (see reward.py) -- it's a direct, differentiable
    # loss term on the mean boundary logit, replacing the old squared-penalty-in-reward
    # mechanism entirely.
    lambda_early: float = 0.1  # weight on training the early-exit head's own next-byte
    # loss -- matches D&W's paper-reported lambda_early (Eq. 27). Independent of
    # lambda_fair below; this has nothing to do with cross-lingual fairness.
    lambda_fair: float = 1.0
    fairness_refresh_every: int = 20
    vocab_budget: int = 512
    lr: float = 3e-3
    seed: int = 0
    bpe_sample_groups: int = 300  # groups sampled for the plain-BPE target-rate anchor
    bpe_baseline_vocab_size: int = 4000  # the anchor BPE's OWN vocab size -- independent
    # of vocab_budget, since the anchor is just a reference number, not the real vocab.
    # The naive from-scratch trainer is O(vocab_size * corpus_size); if this tracked
    # vocab_budget directly, a vocab_budget=50000 run would need ~49700 merges just to
    # compute the anchor, i.e. the exact freeze this field exists to prevent.
    use_wandb: bool = False
    wandb_project: str = "fairtok"
    wandb_run_name: str = ""
    checkpoint_out: str = "policy.pt"  # empty string to skip; see fairtok.inference to reuse it
    checkpoint_every: int = 100  # 0 to disable periodic saving; overwrites checkpoint_out in
    # place each time (not one file per interval) -- a long run interrupted mid-loop (Ctrl+C
    # or otherwise) still leaves a usable, reasonably fresh checkpoint on disk.
    group_sample_size: int = 24  # cap languages rolled out per group, regardless of how many
    # a group actually has available (oldi_seed/flores_plus can offer 46/212 with langs="all").
    # Each group has its own persistent rotation through its language list (see
    # run_training) -- not independent random draws -- so every language a group has is
    # guaranteed to appear at least once every ceil(num_languages / group_sample_size)
    # visits to that specific group, rather than merely likely to show up eventually.
    device: str = ""  # "" auto-detects cuda if available, else cpu; set explicitly to override.
    # batched_sample_rollout (see policy.py) batches every sequence in a training step
    # together at each time step, so this now has real work to parallelize on GPU --
    # unlike the original one-sequence-at-a-time sample_rollout it replaced.


def _num_tokens(records):
    # spans_from_boundaries always closes a span at the last position regardless
    # of its own action, so tokens = boundary=1 decisions among all but the last, + 1
    return sum(1 for r in records[:-1] if r.boundary_action == 1) + 1


def _plain_bpe_target_rate(train_groups, baseline_vocab_size, sample_groups, group_sample_size, seed):
    """target_rate is only ever used as a soft anchor for the compression-rate
    penalty, not something requiring the full corpus or the real vocab_budget --
    so this trains the naive O(vocab_size * corpus_size) baseline BPE (see
    baseline_bpe.py) on a bounded random sample of groups, at a bounded vocab
    size, instead of everything. Without both caps, a real corpus (tens of
    thousands of sentences) and/or a large vocab_budget makes this take a very
    long time with no feedback (or, for vocab_budget=50000, effectively forever).

    group_sample_size caps languages pooled per group, same as training itself
    does -- without this, langs="all" groups (up to 212 languages each) inflate
    the pooled sample far past what sample_groups alone was meant to bound."""
    rng = np.random.default_rng(seed)
    if sample_groups is not None and sample_groups < len(train_groups):
        idx = rng.choice(len(train_groups), size=sample_groups, replace=False)
        sample = [train_groups[i] for i in idx]
    else:
        sample = train_groups

    pooled = []
    for g in tqdm(sample, desc="pooling BPE-baseline sample", unit="group"):
        langs = list(g.keys())
        if group_sample_size and len(langs) > group_sample_size:
            langs = list(rng.choice(langs, size=group_sample_size, replace=False))
        pooled.extend(bytes_to_tensor(g[lang]).numpy().astype("uint8").tobytes() for lang in langs)
    _, merges = train_byte_bpe(pooled, baseline_vocab_size)
    lengths = [
        len(encode_with_merges(seq, merges))
        for seq in tqdm(pooled, desc="scoring BPE-baseline sample", unit="sentence")
    ]
    return float(np.mean([len(s) for s in pooled]) / np.mean(lengths))


def run_training(cfg: Config, train_groups, target_rate=None):
    """train_groups: list of dicts {lang: text}, text either str (utf-8 encoded on
    the fly) or raw bytes. Languages are read per-group (group.keys()), not from a
    fixed global list, since different sources can contribute differently-sized
    groups (see fairtok.oldi_data)."""
    torch.manual_seed(cfg.seed)
    device = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    if target_rate is None:
        target_rate = _plain_bpe_target_rate(
            train_groups, cfg.bpe_baseline_vocab_size, cfg.bpe_sample_groups, cfg.group_sample_size, cfg.seed
        )

    run = None
    if cfg.use_wandb:
        import wandb

        run = wandb.init(
            project=cfg.wandb_project,
            name=cfg.wandb_run_name or None,
            config={**dataclasses.asdict(cfg), "target_rate": target_rate, "num_train_groups": len(train_groups)},
        )

    policy = BytePolicy(hidden_dim=cfg.hidden_dim, num_layers=cfg.num_layers).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.lr)

    token_freq = defaultdict(Counter)
    compression_trace, fairness_trace = [], []
    fairness_scalar = 0.0
    prev_top_spans = set()

    # Epoch = one shuffled traversal of every group in train_groups, not a fixed step
    # count -- reshuffled at the start of each epoch so batch composition varies across
    # passes while still guaranteeing every group is visited exactly once per epoch
    # (as opposed to independent random sampling with replacement, which can leave some
    # groups unvisited for a very long time -- see the coupon-collector discussion).
    steps_per_epoch = math.ceil(len(train_groups) / cfg.batch_groups)
    total_steps = cfg.num_steps if cfg.num_steps > 0 else math.ceil(cfg.num_epochs * steps_per_epoch)
    print(
        f"corpus={len(train_groups)} groups, batch_groups={cfg.batch_groups} "
        f"-> steps_per_epoch={steps_per_epoch}, total_steps={total_steps} "
        f"({total_steps / steps_per_epoch:.2f} epochs)"
    )

    epoch_rng = np.random.default_rng(cfg.seed)
    epoch_order = None

    # Per-group persistent language rotation: each group gets its own shuffled order
    # over its own language list, advancing through fixed-size blocks on successive
    # visits to that specific group (reshuffling once exhausted) -- this is what
    # guarantees every language in a wide group (up to 212 with langs="all") is shown
    # at least once every ceil(num_languages / group_sample_size) visits, rather than
    # each visit independently re-rolling the dice on which languages get included.
    group_lang_order = {}
    group_lang_pos = {}

    pbar = tqdm(range(total_steps), desc="training", unit="step")
    postfix = {}
    step = -1
    try:
        for step in pbar:
            pos_in_epoch = step % steps_per_epoch
            if pos_in_epoch == 0:
                epoch_order = epoch_rng.permutation(len(train_groups))
            batch_start = pos_in_epoch * cfg.batch_groups
            idx = epoch_order[batch_start:batch_start + cfg.batch_groups]
            actual_batch_size = len(idx)

            step_rng = np.random.default_rng(cfg.seed * 100_003 + step)
            optimizer.zero_grad()
            # kept separate because they behave completely differently: nll_loss is a
            # real bounded loss that should shrink as next-byte prediction improves;
            # reinforce_loss is a policy-gradient surrogate whose magnitude tracks the
            # scale of returns and sequence length, not "how good the policy is" -- it
            # has no reason to trend toward zero. Summing them into one number (as the
            # original version did) hides which one is actually driving any change.
            step_boundary_logits = []  # for the rate-consistency loss, below
            # Collected as plain Python lists and only turned into tensors ONCE, after
            # the loop below -- the previous version did `reinforce_loss = reinforce_loss
            # - float(adv[t]) * rec.boundary_logprob` (etc.) INSIDE the per-position loop,
            # which is a sequential chain of small CUDA kernel launches (subtract, then
            # multiply, up to B*T times per step -- tens of thousands on a real batch).
            # list.append() launches nothing; a single torch.stack + one vectorized
            # multiply-and-sum after the loop does the same math in O(1) kernel launches
            # instead of O(B*T). This was the second bottleneck after the .item()-sync
            # fixes in batched_sample_rollout/build_rewards -- those removed host-device
            # *synchronization* stalls, this removes the (much larger, since B*T can be
            # in the tens of thousands) *kernel-launch* overhead of the loss accumulation.
            step_boundary_logprobs = []
            step_next_byte_logprobs = []
            step_early_byte_logprobs = []
            step_advantages = []  # plain floats, index-aligned with the three lists above
            step_compressions_by_lang = defaultdict(list)
            byte_correct, byte_total = 0, 0

            # Phase 1: decide which (group, language) sequences this step needs, for
            # EVERY group in the batch at once -- rotation logic is unchanged, just no
            # longer interleaved with rollout, so it can all happen before the one big
            # batched forward pass below.
            batch_items = []  # list of (group_idx, lang, byte_seq)
            for i in idx:
                group = train_groups[i]
                group_langs_full = list(group.keys())
                if cfg.group_sample_size and len(group_langs_full) > cfg.group_sample_size:
                    # persistent rotation, not independent re-sampling: advance through
                    # this group's own shuffled language order one block at a time,
                    # reshuffling only once the current cycle is exhausted -- guarantees
                    # every language appears at least once every ceil(L / group_sample_size)
                    # visits to THIS group, rather than merely being likely to eventually
                    order = group_lang_order.get(i)
                    pos = group_lang_pos.get(i, 0)
                    if order is None or pos >= len(order):
                        order = list(step_rng.permutation(group_langs_full))
                        pos = 0
                    group_langs = order[pos:pos + cfg.group_sample_size]
                    group_lang_order[i] = order
                    group_lang_pos[i] = pos + cfg.group_sample_size
                else:
                    group_langs = group_langs_full

                for lang in group_langs:
                    batch_items.append((i, lang, bytes_to_tensor(group[lang], device)))

            # Phase 2: ONE batched rollout across every sequence in this entire step
            # (every group, every language together) -- this is the actual GPU-batching
            # payoff: instead of up to batch_groups * group_sample_size independent
            # single-sequence forward passes, this is a single call processing all of
            # them together, padded/masked, max(len(s)) sequential steps instead of
            # sum(len(s)) of them.
            all_records = batched_sample_rollout(policy, [item[2] for item in batch_items], device)

            # Phase 3: redistribute back into per-group structure -- everything past
            # this point (token_freq, compression, rewards, GRPO baseline, loss
            # accumulation) is UNCHANGED from before batching, since it only ever
            # consumed per-sequence StepRecord lists, never cared how they were produced.
            groups_records = defaultdict(dict)
            for (i, lang, byte_seq), records in zip(batch_items, all_records):
                groups_records[i][lang] = (byte_seq, records)
                token_freq[lang].update(spans_from_boundaries(byte_seq, [r.boundary_action for r in records]))
                byte_correct += sum(1 for r in records if r.byte_correct)
                byte_total += sum(1 for r in records if r.byte_correct is not None)

            # Phase 4: per-group reward/GRPO-baseline/loss accumulation, exactly as before
            for i in idx:
                group_records = groups_records[i]
                returns_by_lang = {}
                for lang, (byte_seq, records) in group_records.items():
                    c_rate = compression_rate(len(byte_seq), _num_tokens(records))
                    step_compressions_by_lang[lang].append(c_rate)
                    rewards = build_rewards(records, fairness_scalar, cfg.lambda_fair)
                    returns_by_lang[lang] = discounted_returns(rewards, cfg.gamma)

                advantages = group_relative_advantage(returns_by_lang)
                for lang, (byte_seq, records) in group_records.items():
                    adv = advantages[lang]
                    for t, rec in enumerate(records):
                        step_boundary_logits.append(rec.boundary_logit)
                        step_boundary_logprobs.append(rec.boundary_logprob)  # score-function term
                        step_next_byte_logprobs.append(rec.next_byte_logprob)  # main head, differentiable
                        step_early_byte_logprobs.append(rec.early_byte_logprob)  # early-exit head, differentiable
                        step_advantages.append(float(adv[t]))

            # One vectorized pass instead of the B*T sequential kernel launches the loop
            # above used to do -- see the comment where these lists were declared.
            boundary_logprob_t = torch.stack(step_boundary_logprobs)
            next_byte_logprob_t = torch.stack(step_next_byte_logprobs)
            early_byte_logprob_t = torch.stack(step_early_byte_logprobs)
            advantage_t = torch.tensor(step_advantages, dtype=boundary_logprob_t.dtype, device=device)
            reinforce_loss = -(advantage_t * boundary_logprob_t).sum()
            nll_loss = -next_byte_logprob_t.sum()
            early_nll_loss = -early_byte_logprob_t.sum()

            # bits-per-byte: the standard metric in the byte-level LM literature (ByT5,
            # MambaByte, D&W's own paper) -- smoother and non-saturating unlike raw
            # top-1 accuracy, and directly comparable across papers/model scales.
            # Captured here, before the batch-size normalization below, since it needs
            # the raw summed nats over all byte positions actually predicted this step
            # (byte_total), not a sum normalized by number of groups.
            bits_per_byte = nll_loss.item() / (byte_total * math.log(2)) if byte_total else 0.0
            early_bits_per_byte = early_nll_loss.item() / (byte_total * math.log(2)) if byte_total else 0.0

            # normalize by the actual batch size, not the configured cfg.batch_groups --
            # the last batch of an epoch is smaller whenever len(train_groups) isn't a
            # multiple of batch_groups
            reinforce_loss = reinforce_loss / actual_batch_size
            nll_loss = nll_loss / actual_batch_size
            early_nll_loss = early_nll_loss / actual_batch_size

            # Rate-consistency loss, adapted from Dauncey & Wattenhofer
            # (github.com/SamD770/bitter-lesson-tokenization, model/model.py
            # `off_policy_flexible_training_step`, Eq. 22-24 of the paper): a direct,
            # differentiable push on the mean boundary logit toward the target rate,
            # rather than a reward-shaping penalty that has to survive being filtered
            # through the noisy REINFORCE pathway (the old mechanism this replaces).
            # `factor` is detached so it only scales the correction, not a gradient path.
            all_logits = torch.stack(step_boundary_logits)
            mean_logit = all_logits.mean()
            mean_prob = torch.sigmoid(all_logits).mean()
            target_downsample_rate = 1.0 / target_rate  # target_rate is bytes/token; D&W's
            # mechanism operates on a target fraction-of-positions-that-are-boundaries
            factor = (mean_prob - target_downsample_rate).detach()
            rate_consistency_loss = cfg.lambda_target * mean_logit * factor

            loss = reinforce_loss + nll_loss + cfg.lambda_early * early_nll_loss + rate_consistency_loss
            loss.backward()
            optimizer.step()

            byte_accuracy = byte_correct / byte_total if byte_total else 0.0
            postfix["loss"] = f"{loss.item():+.3f}"
            postfix["acc"] = f"{byte_accuracy:.3f}"
            postfix["bpb"] = f"{bits_per_byte:.3f}"
            postfix["rate"] = f"{mean_prob.item():.3f}/{target_downsample_rate:.3f}"
            pbar.set_postfix(postfix)

            if run is not None:
                run.log({
                    "train/loss": loss.item(),
                    "train/loss_reinforce": reinforce_loss.item(),
                    "train/loss_nll": nll_loss.item(),
                    "train/loss_early_nll": early_nll_loss.item(),
                    "train/rate_consistency_loss": rate_consistency_loss.item(),
                    "train/mean_downsample_rate": mean_prob.item(),
                    "train/target_downsample_rate": target_downsample_rate,
                    "train/byte_accuracy": byte_accuracy,
                    "train/bits_per_byte": bits_per_byte,
                    "train/early_bits_per_byte": early_bits_per_byte,
                }, step=step)

            if cfg.checkpoint_out and cfg.checkpoint_every and step > 0 and step % cfg.checkpoint_every == 0:
                save_checkpoint(policy, cfg.checkpoint_out)

            if step % cfg.fairness_refresh_every == 0:
                renyi = {lang: renyi_efficiency(list(c.values())) for lang, c in token_freq.items() if c}
                fairness_scalar = float(np.var(list(renyi.values())))
                gini = gini_coefficient(list(renyi.values()))
                # variance/gini shrinking is ambiguous by itself: it can't distinguish
                # low performers catching up (genuine) from high performers being pulled
                # down to match them (hollow -- same aggregate number, worse outcome).
                # Tracking min/max separately makes that visible: min should rise over
                # training; max dropping instead of holding roughly flat is a bad sign.
                min_lang = min(renyi, key=renyi.get)
                max_lang = max(renyi, key=renyi.get)
                renyi_min = renyi[min_lang]
                renyi_max = renyi[max_lang]
                per_lang_compression = {
                    lang: float(np.mean(vals)) for lang, vals in step_compressions_by_lang.items()
                }
                # macro-average across languages, not micro-average over pooled sentences,
                # so a batch skewed toward one language doesn't dominate the reported rate
                avg_compression = float(np.mean(list(per_lang_compression.values()))) if per_lang_compression else 0.0
                fairness_trace.append(fairness_scalar)
                compression_trace.append(avg_compression)

                fairness_improving = len(fairness_trace) > 3 and fairness_trace[-1] < fairness_trace[-4]
                compression_worsening = len(compression_trace) > 3 and compression_trace[-1] < compression_trace[-4]
                gaming_suspected = fairness_improving and compression_worsening
                avg_span_len = _avg_span_length(token_freq)
                top_spans, coverage, cross_lingual_share = vocab_snapshot_stats(token_freq, cfg.vocab_budget)
                churn = vocab_churn(prev_top_spans, top_spans)
                prev_top_spans = top_spans
                warn = " <-- CHECK: fairness up / compression down (possible gaming)" if gaming_suspected else ""
                collapse_warn = ""
                if avg_span_len < 1.2:
                    collapse_warn = " <-- CHECK: drifting toward character-level collapse"
                elif avg_span_len > 40:
                    collapse_warn = " <-- CHECK: drifting toward full-sentence collapse"
                pbar.write(
                    f"[step {step:4d}] loss={loss.item():+.4f} acc={byte_accuracy:.3f} renyi_var={fairness_scalar:.5f} "
                    f"gini={gini:.4f} avg_compression={avg_compression:.2f} target={target_rate:.2f} "
                    f"avg_span={avg_span_len:.2f} renyi_range=[{renyi_min:.3f}({min_lang})-{renyi_max:.3f}({max_lang})] "
                    f"coverage={coverage:.3f} cross_lingual={cross_lingual_share:.3f} churn={churn:.3f}"
                    f"{warn}{collapse_warn}"
                )
                postfix.update(
                    renyi_var=f"{fairness_scalar:.4f}", gini=f"{gini:.3f}",
                    compression=f"{avg_compression:.2f}", span=f"{avg_span_len:.2f}",
                )
                pbar.set_postfix(postfix)

                if run is not None:
                    log_dict = {
                        "fairness/renyi_variance": fairness_scalar,
                        "fairness/gini": gini,
                        "fairness/renyi_min": renyi_min,
                        "fairness/renyi_max": renyi_max,
                        "vocab/coverage": coverage,
                        "vocab/cross_lingual_share": cross_lingual_share,
                        "vocab/churn": churn,
                        "compression/avg_rate_macro": avg_compression,
                        "flags/gaming_suspected": int(gaming_suspected),
                        "vocab/avg_span_length_running": avg_span_len,
                        "vocab/num_distinct_spans_running": sum(len(c) for c in token_freq.values()),
                    }
                    log_dict.update({f"fairness/renyi/{lang}": v for lang, v in renyi.items()})
                    log_dict.update({f"compression/rate/{lang}": v for lang, v in per_lang_compression.items()})
                    run.log(log_dict, step=step)
    except KeyboardInterrupt:
        pbar.close()
        print(f"\ninterrupted at step {step} -- finalizing with training done so far "
              f"(vocab, checkpoint, and wandb summary below reflect this partial run)")

    final_vocab = top_k_by_frequency(token_freq, cfg.vocab_budget)

    if run is not None:
        avg_span_len, vocab_size = _collapse_stats(token_freq, final_vocab)
        run.log({
            "final/vocab_size": vocab_size,
            "final/avg_span_length_bytes": avg_span_len,
            "final/char_collapse": int(avg_span_len < 1.2),
            "final/sentence_collapse": int(avg_span_len > 40),
        })
        run.finish()

    if cfg.checkpoint_out:
        save_checkpoint(policy, cfg.checkpoint_out)
        print(f"saved policy checkpoint to {cfg.checkpoint_out}")

    return policy, token_freq, final_vocab, target_rate


def _avg_span_length(token_freq):
    total_spans = sum(sum(c.values()) for c in token_freq.values())
    total_len = sum(len(s) * n for c in token_freq.values() for s, n in c.items())
    return total_len / total_spans if total_spans else 0.0


def _collapse_stats(token_freq, final_vocab):
    return _avg_span_length(token_freq), len(final_vocab)


def _report_collapse(token_freq, final_vocab):
    avg_span_len, vocab_size = _collapse_stats(token_freq, final_vocab)
    print(f"\nfinal vocab size={vocab_size}  avg span length={avg_span_len:.2f} bytes")
    if avg_span_len < 1.2:
        print("WARNING: near character-level collapse")
    elif avg_span_len > 40:
        print("WARNING: near full-sentence collapse")


def run_smoke_test():
    """The plan's own gate: a small trial run checking the loss moves and the
    policy hasn't collapsed (boundary at every byte, or never) before scaling up.
    Uses synthetic placeholder data -- see run_real_smoke_test for the real corpus."""
    cfg = Config(num_steps=60, batch_groups=4, vocab_budget=384, fairness_refresh_every=10)
    langs = list(LANG_PROFILES)
    train_groups = make_synthetic_parallel_groups(400, langs=langs, seed=cfg.seed)
    policy, token_freq, final_vocab, target_rate = run_training(cfg, train_groups)
    _report_collapse(token_freq, final_vocab)
    return policy, token_freq, final_vocab, target_rate


def run_real_smoke_test(num_groups=60):
    """Same gate, but on a slice of the real OLDI-and-friends training data
    (oldi_seed's first `num_groups` rows) instead of synthetic placeholder data --
    confirms real UTF-8 multi-byte text flows through the whole pipeline correctly.
    Not a full training run: oldi_seed alone has 6193 groups, plus 562 from smol and
    6193 from flores_plus dev -- scaling this up is a separate, much longer run."""
    from .oldi_data import load_oldi_seed

    cfg = Config(num_steps=60, batch_groups=4, vocab_budget=384, fairness_refresh_every=10)
    train_groups = load_oldi_seed()[:num_groups]
    policy, token_freq, final_vocab, target_rate = run_training(cfg, train_groups)
    _report_collapse(token_freq, final_vocab)
    return policy, token_freq, final_vocab, target_rate


if __name__ == "__main__":
    run_smoke_test()
