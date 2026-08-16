"""The Phase 1 training loop.

The policy walks a sentence byte by byte, deciding where to cut a new token, and
is scored on three things: (1) next-byte predictability (good chunking =
predictable = good compression), (2) cutting about as often as normal BPE (not
too fine or too coarse), and (3) roughly equal tokenization efficiency across
languages (no language should need way more tokens for the same content).

Parallel-sentence groups (one sentence, many languages) are rolled out together;
a language's fairness score is relative to its group's average that round, not
judged in isolation. The fairness term (3) is checked only periodically, off
everything tokenized so far, not the current batch -- a report card, not instant
feedback. The final vocabulary is just the most frequently-produced token pieces
(up to a fixed budget), tallied after training ends -- the budget is never
enforced live.

Mechanically: one step = one batch of parallel groups, every language in a group
rolled out before any reward is computed (fairness needs the whole group). The
fairness scalar refreshes every `fairness_refresh_steps` steps off the running
per-language token-frequency table, not the exact batch being rewarded -- partial
defense against overfitting to one batch's text.
"""

import dataclasses
import math
from collections import Counter, defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from common.bytes_utils import bytes_to_tensor, spans_from_boundaries
from common.data.synthetic import LANG_PROFILES, make_synthetic_parallel_groups
from common.training.lr_schedule import build_lr_scheduler
from common.eval.metrics import compression_rate, gini_coefficient, renyi_efficiency
from common.eval.parity import compute_lang_parity_ratios
from common.eval.reporting import avg_span_length, collapse_stats, report_collapse
from common.vocab import top_k_by_frequency, vocab_churn, vocab_snapshot_stats

from .baseline_bpe import encode_with_merges, train_byte_bpe
from .inference import save_checkpoint
from ..base import BaseTokenizerConfig, BaseTokenizerTrainer
from .policy import BytePolicy, batched_sample_rollout
from .reward import build_rewards, discounted_returns, group_relative_advantage


@dataclasses.dataclass
class GRPOConfig(BaseTokenizerConfig):
    """use_wandb/run_name inherited unchanged; vocab_size/output_dir/wandb_project
    are overridden below with fairtok-specific defaults (512, "policy.pt",
    "fairtok"). Named after HF/TRL's Trainer+Config pairing; field names mirror
    transformers.TrainingArguments conventions where a clean equivalent exists, so
    anyone coming from HF tooling recognizes the shape. Fields with no HF
    equivalent (gamma, lambda_*, group_sample_size, bpe_*) are this project's own
    RL/fairness knobs."""

    max_steps: int = 0  # 0 derives from num_train_epochs * steps_per_epoch; set
    # explicitly to override with a raw step count (used by the smoke tests below).
    # Matches HF TrainingArguments.max_steps.
    num_train_epochs: float = 3.0  # 1 epoch = 1 full shuffled traversal of every
    # group in train_dataset (see GRPOTrainer.train) -- not a fixed step count, since
    # that depends on len(train_dataset) and per_device_train_batch_size.
    per_device_train_batch_size: int = 8  # counts parallel-sentence GROUPS, not raw
    # byte sequences -- each group expands to up to group_sample_size sequences
    # (its languages). No multi-device support, so this is simply the step's batch size.
    hidden_size: int = 128  # GRU cost scales ~hidden_size^2 per layer.
    num_hidden_layers: int = 3  # stacked GRU cells for both main and early-exit
    # paths (see BytePolicy).
    gamma: float = 0.99
    lambda_target: float = 2.0  # weight on the rate-consistency loss (see
    # GRPOTrainer.train), adapted from Dauncey & Wattenhofer's `consistency_loss_weight`
    # (github.com/SamD770/bitter-lesson-tokenization) -- matches their default of 2.0.
    # A direct, differentiable loss term on the mean boundary logit, not a
    # reward-shaping penalty (see reward.py).
    lambda_early: float = 0.1  # weight on the early-exit head's own next-byte loss --
    # matches D&W's paper-reported lambda_early (Eq. 27). Independent of lambda_fair;
    # unrelated to cross-lingual fairness.
    lambda_fair: float = 1.0
    fairness_refresh_steps: int = 20
    vocab_size: int = 512
    learning_rate: float = 3e-3
    warmup_ratio: float = 0.1  # matches HF Trainer's field name/default -- see
    # common.training.lr_schedule.build_lr_scheduler.
    lr_scheduler_type: str = "linear"  # "constant" (warmup only), "linear", or
    # "cosine" -- see common.training.lr_schedule.build_lr_scheduler.
    seed: int = 0
    bpe_sample_groups: int = 300  # groups sampled for the plain-BPE target-rate anchor
    bpe_baseline_vocab_size: int = 4000  # the anchor BPE's OWN vocab size, independent
    # of vocab_size (the anchor is a reference number, not the real vocab). The naive
    # trainer is O(vocab_size * corpus_size), so tying this to a large real vocab_size
    # would make computing the anchor itself prohibitively slow.
    anchor_lang: str = "eng"  # pivot language for per-language target-rate scaling
    # (see compute_lang_parity_ratios below) -- matches flexitokens.train.FlexiTokensConfig's
    # anchor_lang. _plain_bpe_target_rate's global target_rate is the ANCHOR's own rate;
    # every other language's target_rate_by_lang entry is that rate scaled by its own
    # parity ratio, per "Compute Optimal Tokenization" (Limisiewicz et al. 2026):
    # languages needing more bytes for the same content warrant a different optimal
    # compression rate, not one global rate for everyone.
    target_rate_floor: float = 1.0  # a target_rate below 1 byte/token is nonsensical;
    # guards a degenerate parity ratio.
    target_rate_ceiling: float = 64.0  # generous upper bound past any realistic
    # compression rate at this project's scale; guards the opposite degenerate case.
    wandb_project: str = "fairtok"
    output_dir: str = "policy.pt"  # empty string to skip; see fairtok.inference to
    # reuse it. A single file path, not numbered checkpoints like HF's output_dir.
    save_steps: int = 100  # 0 disables periodic saving; overwrites output_dir in
    # place each time, so an interrupted run still leaves a usable checkpoint.
    group_sample_size: int = 24  # cap languages rolled out per group (oldi_seed/
    # flores_plus can offer 46/212 with langs="all"). Each group has its own persistent
    # rotation through its language list (see GRPOTrainer.train), not independent random
    # draws, guaranteeing every language appears at least once every
    # ceil(num_languages / group_sample_size) visits to that group.
    device: str = ""  # "" auto-detects cuda if available, else cpu.
    max_eval_samples: int = 20  # cap on eval groups scored per epoch boundary; 0 =
    # every loaded eval group. Kept small since this runs periodically DURING training.
    per_device_eval_batch_size: int = 32  # chunk size for the batched eval rollout --
    # independent of per_device_train_batch_size * group_sample_size, since eval data
    # can be paragraph-length (BOUQuET), and padded-batch memory scales with the
    # longest sequence in it.
    dataloader_num_workers: int = 0  # matches HF TrainingArguments.dataloader_num_workers.
    # > 0 moves GroupLanguageCollator into worker subprocesses for background
    # prefetching, but each worker gets its OWN copy of the per-group rotation state,
    # so the "every language shown at least once every ceil(L / group_sample_size)
    # visits" coverage guarantee (see GroupLanguageCollator) only holds per-worker, not
    # globally, once this is > 0.
    # Held-out evaluation runs automatically at every epoch boundary (see train()) when
    # eval_dataset is given (see fairtok.cli, which loads BOUQuET dev unless
    # --data-source synthetic) -- scores the CURRENT policy deterministically against
    # data it never trains on, unlike fairness_refresh_steps's noisier running stats.


def _num_tokens(records):
    # spans_from_boundaries always closes a span at the last position regardless
    # of its own action, so tokens = boundary=1 decisions among all but the last, + 1
    return sum(1 for r in records[:-1] if r.boundary_action == 1) + 1


def _plain_bpe_target_rate(
    train_groups, baseline_vocab_size, sample_groups, group_sample_size, seed
):
    """target_rate is just a soft anchor for the compression-rate penalty, so this
    trains the naive O(vocab_size * corpus_size) baseline BPE (see baseline_bpe.py)
    on a bounded random sample of groups at a bounded vocab size, not the full
    corpus/vocab_size -- otherwise a real corpus and/or large vocab_size makes this
    take forever.

    group_sample_size caps languages pooled per group, same as training -- without
    it, langs="all" groups (up to 212 languages) inflate the pooled sample far past
    what sample_groups alone bounds."""
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
        pooled.extend(
            bytes_to_tensor(g[lang]).numpy().astype("uint8").tobytes() for lang in langs
        )
    _, merges = train_byte_bpe(pooled, baseline_vocab_size)
    lengths = [
        len(encode_with_merges(seq, merges))
        for seq in tqdm(pooled, desc="scoring BPE-baseline sample", unit="sentence")
    ]
    return float(np.mean([len(s) for s in pooled]) / np.mean(lengths))


class ByteGroupDataset(Dataset):
    """Thin adapter wrapping a plain list of parallel groups (dicts {lang: text}) for
    GRPOTrainer's train_dataset: __getitem__ returns (index, group) rather than just
    group, so GroupLanguageCollator's per-group persistent language rotation can key
    its state by the group's position in the underlying list."""

    def __init__(self, groups):
        self.groups = groups

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        return idx, self.groups[idx]


class GroupLanguageCollator:
    """DataLoader collate_fn: turns a batch of (group_idx, group) pairs (from
    ByteGroupDataset) into the flat (group_idx, lang, byte_seq) list GRPOTrainer.train()
    consumes.

    Stateful: group_lang_order/group_lang_pos persist across __call__s, keyed by
    group_idx -- a persistent rotation through each group's own shuffled language
    order, advancing one block per visit, guaranteeing every language in a wide group
    is shown at least once every ceil(num_languages / group_sample_size) visits to
    THAT group (not merely likely to show up eventually). See
    GRPOConfig.dataloader_num_workers: this guarantee is per-worker, not global,
    once num_workers > 0.

    Also returns each batch's group_idx list alongside batch_items, since downstream
    loss normalization needs the number of GROUPS in the batch, not the (larger)
    flattened (group, lang) count."""

    def __init__(self, group_sample_size, seed, device):
        self.group_sample_size = group_sample_size
        self.device = device
        self.rng = np.random.default_rng(seed)
        self.group_lang_order = {}
        self.group_lang_pos = {}

    def __call__(self, batch):
        batch_items = []
        group_indices = []
        for i, group in batch:
            group_indices.append(i)
            group_langs_full = list(group.keys())
            if (
                self.group_sample_size
                and len(group_langs_full) > self.group_sample_size
            ):
                order = self.group_lang_order.get(i)
                pos = self.group_lang_pos.get(i, 0)
                if order is None or pos >= len(order):
                    order = list(self.rng.permutation(group_langs_full))
                    pos = 0
                group_langs = order[pos : pos + self.group_sample_size]
                self.group_lang_order[i] = order
                self.group_lang_pos[i] = pos + self.group_sample_size
            else:
                group_langs = group_langs_full

            for lang in group_langs:
                batch_items.append((i, lang, bytes_to_tensor(group[lang], self.device)))
        return batch_items, group_indices


class GRPOTrainer(BaseTokenizerTrainer):
    """GRPO-style trainer for the byte-boundary policy ("group" = one parallel
    sentence's translations across languages, not TRL's repeated-rollout sense --
    see reward.py's group_relative_advantage). Shaped after HF Trainer/TRL
    GRPOTrainer: construct with args + train_dataset (+ optional eval_dataset), call
    .train(), then read .model/.token_freq/.vocab/.target_rate off the instance
    (train() also returns them as a tuple). .evaluate() can be called standalone too.

    train_dataset: a plain list of dicts {lang: text} -- auto-wrapped in
    ByteGroupDataset if not already a Dataset. eval_dataset stays a plain list
    (evaluate() doesn't need DataLoader; no rotation state to carry across calls).
    Languages are read per-group, not a fixed global list, since sources can
    contribute differently-sized groups. eval_dataset, if given, is scored by
    .evaluate() at every epoch boundary during .train() -- held out, never trained on.
    """

    def __init__(
        self, args: GRPOConfig, train_dataset, eval_dataset=None, target_rate=None
    ):
        # Passes pre-wrap train_dataset/eval_dataset through as BaseTokenizerTrainer's
        # own train_groups/eval_groups; this class keeps its OWN train_dataset/
        # eval_dataset names for its DataLoader-based train() below.
        super().__init__(args, train_dataset, eval_dataset)
        self.train_dataset = (
            train_dataset
            if isinstance(train_dataset, Dataset)
            else ByteGroupDataset(train_dataset)
        )
        self.eval_dataset = eval_dataset
        self.target_rate = target_rate
        self.device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    def get_train_dataloader(self):
        """Matches HF Trainer.get_train_dataloader's name/role. shuffle=True with a
        seeded generator reshuffles reproducibly at the start of each epoch (a fresh
        permutation each time the loader is re-iterated), while still visiting every
        group exactly once per epoch (permutes, doesn't sample with replacement)."""
        cfg = self.args
        generator = torch.Generator().manual_seed(cfg.seed)
        collate_fn = GroupLanguageCollator(cfg.group_sample_size, cfg.seed, self.device)
        return DataLoader(
            self.train_dataset,
            batch_size=cfg.per_device_train_batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            generator=generator,
            num_workers=cfg.dataloader_num_workers,
        )

    @torch.no_grad()
    def evaluate(self, eval_dataset=None):
        """Score the CURRENT self.model, deterministically and without gradients,
        against a held-out set of parallel groups (BOUQuET dev, via fairtok.cli --
        disjoint from every training source). Distinct from train()'s running
        training-batch stats: real held-out data the policy never trains on, and
        boundary decisions are thresholded not sampled, so repeated calls give the
        same answer.

        Chunks the rollout into per_device_eval_batch_size-sized pieces since
        eval_dataset can be paragraph-length (BOUQuET), and padded-batch memory scales
        with the longest sequence in it.

        Returns {renyi (per-lang), renyi_variance, gini, per_lang_compression,
        avg_compression} -- same quantities train() tracks, computed on eval_dataset."""
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        cfg = self.args
        device = self.device
        policy = self.model

        if cfg.max_eval_samples and len(eval_dataset) > cfg.max_eval_samples:
            rng = np.random.default_rng(cfg.seed)
            idx = rng.choice(
                len(eval_dataset), size=cfg.max_eval_samples, replace=False
            )
            eval_dataset = [eval_dataset[i] for i in idx]

        batch_items = [
            (lang, bytes_to_tensor(text, device))
            for group in eval_dataset
            for lang, text in group.items()
        ]

        was_training = policy.training
        policy.eval()
        token_freq = defaultdict(Counter)
        compressions_by_lang = defaultdict(list)
        for start in range(0, len(batch_items), cfg.per_device_eval_batch_size):
            chunk = batch_items[start : start + cfg.per_device_eval_batch_size]
            records_per_seq = batched_sample_rollout(
                policy, [seq for _, seq in chunk], device, deterministic=True
            )
            for (lang, byte_seq), records in zip(chunk, records_per_seq):
                spans = spans_from_boundaries(
                    byte_seq, [r.boundary_action for r in records]
                )
                token_freq[lang].update(spans)
                compressions_by_lang[lang].append(
                    compression_rate(len(byte_seq), _num_tokens(records))
                )
        policy.train(was_training)

        renyi = {
            lang: renyi_efficiency(list(c.values()))
            for lang, c in token_freq.items()
            if c
        }
        renyi_variance = float(np.var(list(renyi.values()))) if renyi else 0.0
        gini = gini_coefficient(list(renyi.values())) if renyi else 0.0
        per_lang_compression = {
            lang: float(np.mean(vals)) for lang, vals in compressions_by_lang.items()
        }
        avg_compression = (
            float(np.mean(list(per_lang_compression.values())))
            if per_lang_compression
            else 0.0
        )
        return {
            "renyi": renyi,
            "renyi_variance": renyi_variance,
            "gini": gini,
            "per_lang_compression": per_lang_compression,
            "avg_compression": avg_compression,
        }

    def train(self):
        cfg = self.args
        train_dataset = self.train_dataset
        # _plain_bpe_target_rate wants a plain list of {lang: text} dicts, not
        # ByteGroupDataset's (index, group) items -- unwrap for that one call only.
        raw_train_groups = (
            train_dataset.groups
            if isinstance(train_dataset, ByteGroupDataset)
            else train_dataset
        )
        eval_groups = self.eval_dataset
        target_rate = self.target_rate
        device = self.device
        torch.manual_seed(cfg.seed)
        print(f"device={device}")

        if target_rate is None:
            target_rate = _plain_bpe_target_rate(
                raw_train_groups,
                cfg.bpe_baseline_vocab_size,
                cfg.bpe_sample_groups,
                cfg.group_sample_size,
                cfg.seed,
            )

        # Per-language target rate: target_rate above is the single GLOBAL anchor rate;
        # every other language's target_rate_by_lang entry is that rate times its own
        # byte-length parity ratio vs the anchor (see GRPOConfig.anchor_lang, and
        # compute_lang_parity_ratios for the formula). A language with no evidence of
        # disparity (ratio ~= 1.0) just gets target_rate back unchanged.
        parity_ratio_by_lang, parity_anchor = compute_lang_parity_ratios(
            raw_train_groups, cfg.anchor_lang
        )
        if parity_anchor != cfg.anchor_lang:
            print(
                f"[fairtok] anchor language {cfg.anchor_lang!r} not present in this corpus; "
                f"falling back to {parity_anchor!r} for per-language target-rate scaling"
            )
        target_rate_by_lang = {
            lang: max(
                cfg.target_rate_floor,
                min(cfg.target_rate_ceiling, target_rate * ratio),
            )
            for lang, ratio in parity_ratio_by_lang.items()
        }

        run = None
        if cfg.use_wandb:
            import wandb

            run = wandb.init(
                project=cfg.wandb_project,
                name=cfg.run_name or None,
                config={
                    **dataclasses.asdict(cfg),
                    "target_rate": target_rate,
                    "target_rate_by_lang": target_rate_by_lang,
                    "num_train_groups": len(train_dataset),
                },
            )

        policy = BytePolicy(
            hidden_dim=cfg.hidden_size, num_layers=cfg.num_hidden_layers
        ).to(device)
        self.model = policy  # set now, not just at the end, so mid-loop self.evaluate()
        # sees the live, training-in-progress policy, not None
        optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.learning_rate)

        token_freq = defaultdict(Counter)
        compression_trace, fairness_trace = [], []
        fairness_scalar = 0.0
        prev_top_spans = set()

        # Epoch = one shuffled traversal of every group, not a fixed step count --
        # DataLoader(shuffle=True) reshuffles each time its iterator is (re)created,
        # while still visiting every group exactly once per epoch (permutes, doesn't
        # sample with replacement).
        steps_per_epoch = math.ceil(
            len(train_dataset) / cfg.per_device_train_batch_size
        )
        total_steps = (
            cfg.max_steps
            if cfg.max_steps > 0
            else math.ceil(cfg.num_train_epochs * steps_per_epoch)
        )
        print(
            f"corpus={len(train_dataset)} groups, per_device_train_batch_size={cfg.per_device_train_batch_size} "
            f"-> steps_per_epoch={steps_per_epoch}, total_steps={total_steps} "
            f"({total_steps / steps_per_epoch:.2f} epochs)"
        )
        scheduler = build_lr_scheduler(
            optimizer, total_steps, cfg.warmup_ratio, cfg.lr_scheduler_type
        )

        # Accumulated across the WHOLE fairness_refresh_steps window, not reset every
        # step -- with langs="all" (up to 212 languages), a single step's compression
        # reading per language was a mean of one or two sentences: extremely noisy,
        # and the direct cause of wild avg_compression swings seen in earlier runs
        # (e.g. 11.36 -> 4.25 -> 1.75 -> 3.04 -> 4.93) and spurious gaming-suspected
        # flags. Reset only once per window so each reading reflects
        # ~fairness_refresh_steps steps of samples per language.
        window_compressions_by_lang = defaultdict(list)

        train_loader = self.get_train_dataloader()
        data_iter = iter(train_loader)

        pbar = tqdm(range(total_steps), desc="training", unit="step")
        postfix = {}
        step = -1
        try:
            for step in pbar:
                try:
                    batch_items, idx = next(data_iter)
                except StopIteration:
                    # End of an epoch: re-iterating the loader draws a fresh shuffle.
                    data_iter = iter(train_loader)
                    batch_items, idx = next(data_iter)
                actual_batch_size = len(idx)  # number of GROUPS in this batch, not the
                # larger flattened (group, lang) count in batch_items (see
                # GroupLanguageCollator, which returns both).

                optimizer.zero_grad()
                # nll_loss and reinforce_loss are kept separate (not summed early) because
                # they behave differently: nll_loss shrinks as prediction improves,
                # reinforce_loss's magnitude just tracks return/sequence-length scale.
                step_lang_boundary_logits = defaultdict(
                    list
                )  # keyed by language (not one flat list) for the PER-LANGUAGE
                # rate-consistency loss below, so each language's mean boundary logit is
                # pushed toward ITS OWN target_rate_by_lang entry, not a batch-wide pooled
                # rate a dominant language could drag every other language's rate toward.
                # Collected as plain lists, turned into tensors ONCE after the loop --
                # per-position tensor ops here would be a sequential chain of small CUDA
                # kernel launches (up to B*T per step); a single vectorized stack+multiply
                # after the loop does the same math in O(1) launches.
                step_boundary_logprobs = []
                step_next_byte_logprobs = []
                step_early_byte_logprobs = []
                step_advantages = []  # plain floats, index-aligned with the lists above
                byte_correct, byte_total = 0, 0

                # Phase 2: ONE batched rollout across every sequence in this step (every
                # group, every language together) -- the actual GPU-batching payoff:
                # max(len(s)) sequential steps instead of sum(len(s)) independent ones.
                all_records = batched_sample_rollout(
                    policy, [item[2] for item in batch_items], device
                )

                # Phase 3: redistribute back into per-group structure -- unchanged from
                # before batching, since it only ever consumed per-sequence StepRecord lists.
                groups_records = defaultdict(dict)
                for (i, lang, byte_seq), records in zip(batch_items, all_records):
                    groups_records[i][lang] = (byte_seq, records)
                    token_freq[lang].update(
                        spans_from_boundaries(
                            byte_seq, [r.boundary_action for r in records]
                        )
                    )
                    byte_correct += sum(1 for r in records if r.byte_correct)
                    byte_total += sum(1 for r in records if r.byte_correct is not None)

                # Phase 4: per-group reward/GRPO-baseline/loss accumulation
                for i in idx:
                    group_records = groups_records[i]
                    returns_by_lang = {}
                    for lang, (byte_seq, records) in group_records.items():
                        c_rate = compression_rate(len(byte_seq), _num_tokens(records))
                        window_compressions_by_lang[lang].append(c_rate)
                        rewards = build_rewards(
                            records, fairness_scalar, cfg.lambda_fair
                        )
                        returns_by_lang[lang] = discounted_returns(rewards, cfg.gamma)

                    advantages = group_relative_advantage(returns_by_lang)
                    for lang, (_byte_seq, records) in group_records.items():
                        adv = advantages[lang]
                        for t, rec in enumerate(records):
                            step_lang_boundary_logits[lang].append(rec.boundary_logit)
                            step_boundary_logprobs.append(
                                rec.boundary_logprob
                            )  # score-function term
                            step_next_byte_logprobs.append(
                                rec.next_byte_logprob
                            )  # main head, differentiable
                            step_early_byte_logprobs.append(
                                rec.early_byte_logprob
                            )  # early-exit head, differentiable
                            step_advantages.append(float(adv[t]))

                # One vectorized pass instead of B*T sequential kernel launches.
                boundary_logprob_t = torch.stack(step_boundary_logprobs)
                next_byte_logprob_t = torch.stack(step_next_byte_logprobs)
                early_byte_logprob_t = torch.stack(step_early_byte_logprobs)
                advantage_t = torch.tensor(
                    step_advantages, dtype=boundary_logprob_t.dtype, device=device
                )
                reinforce_loss = -(advantage_t * boundary_logprob_t).sum()
                nll_loss = -next_byte_logprob_t.sum()
                early_nll_loss = -early_byte_logprob_t.sum()

                # bits-per-byte: standard metric in byte-level LM literature (ByT5,
                # MambaByte, D&W) -- smoother than top-1 accuracy, comparable across
                # papers/scales. Computed before batch-size normalization below, from the
                # raw summed nats over byte_total (not group-normalized).
                bits_per_byte = (
                    nll_loss.item() / (byte_total * math.log(2)) if byte_total else 0.0
                )
                early_bits_per_byte = (
                    early_nll_loss.item() / (byte_total * math.log(2))
                    if byte_total
                    else 0.0
                )

                # normalize by the actual batch size, not the configured one -- the last
                # batch of an epoch can be smaller.
                reinforce_loss = reinforce_loss / actual_batch_size
                nll_loss = nll_loss / actual_batch_size
                early_nll_loss = early_nll_loss / actual_batch_size

                # Rate-consistency loss, adapted from Dauncey & Wattenhofer
                # (github.com/SamD770/bitter-lesson-tokenization, Eq. 22-24): a direct,
                # differentiable push on the mean boundary logit toward the target rate,
                # rather than a reward-shaping penalty filtered through noisy REINFORCE.
                # `factor` is detached so it only scales the correction, not a gradient path.
                #
                # Computed PER LANGUAGE against that language's own target_rate_by_lang
                # entry, then averaged with equal weight per language present -- avoids a
                # language dominating this step's batch from drowning out every other
                # language's own target in a single pooled mean.
                per_lang_rate_losses = []
                per_lang_mean_probs = {}
                for lang, logits in step_lang_boundary_logits.items():
                    logits_t = torch.stack(logits)
                    mean_logit_lang = logits_t.mean()
                    mean_prob_lang = torch.sigmoid(logits_t).mean()
                    # target_rate_by_lang is bytes/token; D&W's mechanism operates on a
                    # target fraction-of-positions-that-are-boundaries
                    target_downsample_rate_lang = 1.0 / target_rate_by_lang.get(
                        lang, target_rate
                    )
                    factor_lang = (mean_prob_lang - target_downsample_rate_lang).detach()
                    per_lang_rate_losses.append(mean_logit_lang * factor_lang)
                    per_lang_mean_probs[lang] = mean_prob_lang.item()
                rate_consistency_loss = cfg.lambda_target * torch.stack(
                    per_lang_rate_losses
                ).mean()
                # Aggregate scalars for the postfix/wandb logging below only -- the loss
                # itself never pools across languages (see per_lang_rate_losses above).
                mean_prob = float(np.mean(list(per_lang_mean_probs.values())))
                target_downsample_rate = float(
                    np.mean(
                        [
                            1.0 / target_rate_by_lang.get(lang, target_rate)
                            for lang in per_lang_mean_probs
                        ]
                    )
                )

                loss = (
                    reinforce_loss
                    + nll_loss
                    + cfg.lambda_early * early_nll_loss
                    + rate_consistency_loss
                )
                loss.backward()
                optimizer.step()
                scheduler.step()

                byte_accuracy = byte_correct / byte_total if byte_total else 0.0
                postfix["loss"] = f"{loss.item():+.3f}"
                postfix["acc"] = f"{byte_accuracy:.3f}"
                postfix["bpb"] = f"{bits_per_byte:.3f}"
                postfix["rate"] = f"{mean_prob:.3f}/{target_downsample_rate:.3f}"
                pbar.set_postfix(postfix)

                if run is not None:
                    run.log(
                        {
                            "train/loss": loss.item(),
                            "train/loss_reinforce": reinforce_loss.item(),
                            "train/loss_nll": nll_loss.item(),
                            "train/loss_early_nll": early_nll_loss.item(),
                            "train/rate_consistency_loss": rate_consistency_loss.item(),
                            "train/mean_downsample_rate": mean_prob,
                            "train/target_downsample_rate": target_downsample_rate,
                            "train/byte_accuracy": byte_accuracy,
                            "train/bits_per_byte": bits_per_byte,
                            "train/early_bits_per_byte": early_bits_per_byte,
                            "train/learning_rate": scheduler.get_last_lr()[0],
                        },
                        step=step,
                    )

                if (
                    cfg.output_dir
                    and cfg.save_steps
                    and step > 0
                    and step % cfg.save_steps == 0
                ):
                    save_checkpoint(policy, cfg.output_dir)

                if step % cfg.fairness_refresh_steps == 0:
                    renyi = {
                        lang: renyi_efficiency(list(c.values()))
                        for lang, c in token_freq.items()
                        if c
                    }
                    fairness_scalar = float(np.var(list(renyi.values())))
                    gini = gini_coefficient(list(renyi.values()))
                    # variance/gini shrinking alone can't distinguish low performers
                    # catching up (genuine) from high performers being pulled down (hollow).
                    # min/max tracked separately: min should rise, max dropping is a bad sign.
                    min_lang = min(renyi, key=renyi.get)
                    max_lang = max(renyi, key=renyi.get)
                    renyi_min = renyi[min_lang]
                    renyi_max = renyi[max_lang]
                    per_lang_compression = {
                        lang: float(np.mean(vals))
                        for lang, vals in window_compressions_by_lang.items()
                    }
                    window_compressions_by_lang = defaultdict(list)  # reset for next window
                    # macro-average across languages (not micro-average over pooled
                    # sentences), so a batch skewed toward one language doesn't dominate
                    avg_compression = (
                        float(np.mean(list(per_lang_compression.values())))
                        if per_lang_compression
                        else 0.0
                    )
                    fairness_trace.append(fairness_scalar)
                    compression_trace.append(avg_compression)

                    fairness_improving = (
                        len(fairness_trace) > 3
                        and fairness_trace[-1] < fairness_trace[-4]
                    )
                    compression_worsening = (
                        len(compression_trace) > 3
                        and compression_trace[-1] < compression_trace[-4]
                    )
                    gaming_suspected = fairness_improving and compression_worsening
                    avg_span_len = avg_span_length(token_freq)
                    top_spans, coverage, cross_lingual_share = vocab_snapshot_stats(
                        token_freq, cfg.vocab_size
                    )
                    churn = vocab_churn(prev_top_spans, top_spans)
                    prev_top_spans = top_spans
                    warn = (
                        " <-- CHECK: fairness up / compression down (possible gaming)"
                        if gaming_suspected
                        else ""
                    )
                    collapse_warn = ""
                    if avg_span_len < 1.2:
                        collapse_warn = (
                            " <-- CHECK: drifting toward character-level collapse"
                        )
                    elif avg_span_len > 40:
                        collapse_warn = (
                            " <-- CHECK: drifting toward full-sentence collapse"
                        )
                    pbar.write(
                        f"[step {step:4d}] loss={loss.item():+.4f} acc={byte_accuracy:.3f} renyi_var={fairness_scalar:.5f} "
                        f"gini={gini:.4f} avg_compression={avg_compression:.2f} target={target_rate:.2f} "
                        f"avg_span={avg_span_len:.2f} renyi_range=[{renyi_min:.3f}({min_lang})-{renyi_max:.3f}({max_lang})] "
                        f"coverage={coverage:.3f} cross_lingual={cross_lingual_share:.3f} churn={churn:.3f}"
                        f"{warn}{collapse_warn}"
                    )
                    postfix.update(
                        renyi_var=f"{fairness_scalar:.4f}",
                        gini=f"{gini:.3f}",
                        compression=f"{avg_compression:.2f}",
                        span=f"{avg_span_len:.2f}",
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
                            "vocab/num_distinct_spans_running": sum(
                                len(c) for c in token_freq.values()
                            ),
                        }
                        log_dict.update(
                            {f"fairness/renyi/{lang}": v for lang, v in renyi.items()}
                        )
                        log_dict.update(
                            {
                                f"compression/rate/{lang}": v
                                for lang, v in per_lang_compression.items()
                            }
                        )
                        run.log(log_dict, step=step)

                # Held-out evaluation: runs at every epoch boundary, not tied to
                # fairness_refresh_steps -- a full deterministic rollout over
                # eval_groups is much heavier than reading the running token_freq table.
                if eval_groups and step % steps_per_epoch == 0:
                    eval_metrics = self.evaluate(eval_groups)
                    eval_per_lang = " ".join(
                        f"{lang}={v:.2f}"
                        for lang, v in sorted(
                            eval_metrics["per_lang_compression"].items()
                        )
                    )
                    pbar.write(
                        f"[eval  step {step:4d}] avg_compression={eval_metrics['avg_compression']:.2f} "
                        f"renyi_var={eval_metrics['renyi_variance']:.5f} gini={eval_metrics['gini']:.4f} "
                        f"per_lang=[{eval_per_lang}]"
                    )
                    if run is not None:
                        eval_log = {
                            "eval/avg_compression": eval_metrics["avg_compression"],
                            "eval/renyi_variance": eval_metrics["renyi_variance"],
                            "eval/gini": eval_metrics["gini"],
                        }
                        eval_log.update(
                            {
                                f"eval/renyi/{lang}": v
                                for lang, v in eval_metrics["renyi"].items()
                            }
                        )
                        eval_log.update(
                            {
                                f"eval/compression/{lang}": v
                                for lang, v in eval_metrics[
                                    "per_lang_compression"
                                ].items()
                            }
                        )
                        run.log(eval_log, step=step)
        except KeyboardInterrupt:
            pbar.close()
            print(
                f"\ninterrupted at step {step} -- finalizing with training done so far "
                f"(vocab, checkpoint, and wandb summary below reflect this partial run)"
            )

        final_vocab = top_k_by_frequency(token_freq, cfg.vocab_size)

        if run is not None:
            avg_span_len, final_vocab_size = collapse_stats(token_freq, final_vocab)
            run.log(
                {
                    "final/vocab_size": final_vocab_size,
                    "final/avg_span_length_bytes": avg_span_len,
                    "final/char_collapse": int(avg_span_len < 1.2),
                    "final/sentence_collapse": int(avg_span_len > 40),
                }
            )
            run.finish()

        if cfg.output_dir:
            save_checkpoint(policy, cfg.output_dir)
            print(f"saved policy checkpoint to {cfg.output_dir}")

        self.token_freq = token_freq
        self.vocab = final_vocab
        self.target_rate = target_rate
        return policy, token_freq, final_vocab, target_rate


def run_smoke_test():
    """The plan's own gate: a small trial run checking the loss moves and the
    policy hasn't collapsed (boundary at every byte, or never) before scaling up.
    Uses synthetic placeholder data -- see run_real_smoke_test for the real corpus."""
    args = GRPOConfig(
        max_steps=60,
        per_device_train_batch_size=4,
        vocab_size=384,
        fairness_refresh_steps=10,
    )
    langs = list(LANG_PROFILES)
    train_groups = make_synthetic_parallel_groups(400, langs=langs, seed=args.seed)
    policy, token_freq, final_vocab, target_rate = GRPOTrainer(
        args, train_groups
    ).train()
    report_collapse(token_freq, final_vocab)
    return policy, token_freq, final_vocab, target_rate


def run_real_smoke_test(num_groups=60):
    """Same gate, but on a slice of real OLDI-and-friends data (oldi_seed's first
    `num_groups` rows, langs="all") instead of synthetic data -- confirms real UTF-8
    multi-byte text flows through the pipeline correctly. Not a full training run."""
    from common.data.oldi_data import load_oldi_seed

    args = GRPOConfig(
        max_steps=60,
        per_device_train_batch_size=4,
        vocab_size=384,
        fairness_refresh_steps=10,
    )
    train_groups = load_oldi_seed()[:num_groups]
    policy, token_freq, final_vocab, target_rate = GRPOTrainer(
        args, train_groups
    ).train()
    report_collapse(token_freq, final_vocab)
    return policy, token_freq, final_vocab, target_rate


if __name__ == "__main__":
    run_smoke_test()
