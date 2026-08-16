"""Plain backprop training loop for MantaModel.

Much simpler than fairtok.train.GRPOTrainer: no REINFORCE, reward,
per-group baseline, fairness scalar, or rate-consistency loss -- MANTa
genuinely has none of that machinery (manta.model module docstring, point
3). The only loss is next-byte cross-entropy (manta.model.next_byte_loss).
Parallel-sentence "groups" (common.data.oldi_data) are just a convenient
source of multilingual sentences here, not a training-time structure the
way GRPOTrainer's group-relative advantage needs -- this trainer flattens
every group into a flat (lang, text) list up front and samples plain
shuffled minibatches of individual sentences from it.

MantaConfig field names mirror fairtok.train.GRPOConfig's (itself modeled
on HF TrainingArguments) wherever a clean equivalent exists. Note:
GRPOConfig's per_device_train_batch_size counts parallel-sentence groups;
MantaConfig's counts individual byte sequences, since there's no group
structure at batch time here.
"""

import dataclasses
import math
from collections import Counter, defaultdict

import numpy as np
import torch
from tqdm.auto import tqdm

from common.data.synthetic import LANG_PROFILES, make_synthetic_parallel_groups
from common.eval.cross_tokenizer import (
    eval_wandb_log_dict,
    evaluate_on_groups,
    report_eval,
    sample_eval_groups,
)
from common.training.lr_schedule import build_lr_scheduler
from common.eval.metrics import compression_rate, gini_coefficient, renyi_efficiency
from common.bytes_utils import bytes_to_tensor, spans_from_boundaries
from common.eval.reporting import collapse_stats
from common.vocab import top_k_by_frequency

from ..base import BaseTokenizerConfig, BaseTokenizerTrainer
from .model import MantaModel, next_byte_loss
from .segment import boundaries_from_assignment, induce_spans


@dataclasses.dataclass
class MantaConfig(BaseTokenizerConfig):
    """vocab_size/output_dir/use_wandb/wandb_project/run_name inherited from
    BaseTokenizerConfig. Field names mirror fairtok.train.GRPOConfig where a
    clean equivalent exists (see module docstring for the batch-size naming
    caveat)."""

    max_steps: int = 0  # 0 -> derive from num_train_epochs * steps_per_epoch.
    # Matches GRPOConfig's max_steps/num_train_epochs convention; set explicitly
    # for a fixed step count regardless of corpus size (run_smoke_test does this).
    num_train_epochs: float = 3.0  # only used if max_steps == 0. Unlike
    # magnet/flexitokens/fanta, steps_per_epoch here is a real traversal boundary
    # (full reshuffle on exhaustion below), so "N epochs" means N real passes.
    per_device_train_batch_size: int = 8  # individual byte sequences, not
    # parallel-sentence groups (unlike GRPOConfig's same-named field).
    learning_rate: float = 3e-3
    seed: int = 0
    # vocab_size (inherited, default 384): budget for common.vocab.top_k_by_frequency,
    # applied once post-training -- same two-stage design as fairtok's own vocab
    # extraction, for comparable output across systems.
    dim: int = 64  # byte embedding / model width, matching fairtok.policy.BytePolicy's
    # scale (64-128), not the paper's ~200M-param model (model.py docstring point 8).
    window: int = 8  # frontier predictor's local-attention half-window (+/- window)
    num_frontier_layers: int = 2
    num_frontier_heads: int = 4
    block_hidden_size: int = 64
    num_block_layers: int = 1
    max_extra_sigma: float = (
        3.0  # candidate block range truncation, mu_L + this*sigma_L
    )
    max_grad_norm: float = 1.0  # gradient clipping -- cheap insurance since the
    # Gaussian-log-density -> softmax -> matmul chain is more numerically fragile
    # than fairtok's plain GRU cells.
    device: str = ""  # "" auto-detects cuda, else cpu.
    log_steps: int = 10  # progress-printing interval.
    save_steps: int = 0  # 0 disables periodic checkpoint saving.
    wandb_project: str = "manta"

    max_eval_samples: int = 20  # cap on BOUQuET dev groups scored per
    # epoch-boundary eval (0 = score all). Kept small since this runs
    # periodically during training, unlike evaluate.py's full scoring.

    warmup_ratio: float = 0.1  # see common.training.lr_schedule.build_lr_scheduler.
    lr_scheduler_type: str = "linear"  # "constant", "linear", or "cosine".


def _avg_span_length(token_freq):
    total_spans = sum(sum(c.values()) for c in token_freq.values())
    total_len = sum(len(s) * n for c in token_freq.values() for s, n in c.items())
    return total_len / total_spans if total_spans else 0.0


def _pad_batch(tensors, device):
    lengths = torch.tensor(
        [t.shape[0] for t in tensors], dtype=torch.long, device=device
    )
    T = int(lengths.max().item())
    padded = torch.zeros(len(tensors), T, dtype=torch.long, device=device)
    for i, t in enumerate(tensors):
        padded[i, : t.shape[0]] = t
    return padded, lengths


class MantaTrainer(BaseTokenizerTrainer):
    """Construct with args + train_groups (list of {lang: text} dicts, same
    shape as common.data.cli_data.load_groups / make_synthetic_parallel_groups),
    call .train(), then read .model/.token_freq/.vocab off the instance
    (also returned as a tuple, mirroring GRPOTrainer's convention)."""

    def __init__(self, args: MantaConfig, train_groups, eval_groups=None):
        super().__init__(args, train_groups, eval_groups)
        self.device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    def train(self):
        cfg = self.args
        device = self.device
        torch.manual_seed(cfg.seed)
        print(f"device={device}")

        # Flatten every group into (lang, text) pairs once, up front -- no group
        # structure at batch time (unlike GRPOTrainer's GroupLanguageCollator).
        flat_items = [
            (lang, text) for group in self.train_groups for lang, text in group.items()
        ]
        print(
            f"corpus={len(self.train_groups)} groups -> {len(flat_items)} flattened (lang, text) sequences"
        )

        model = MantaModel(
            dim=cfg.dim,
            window=cfg.window,
            num_frontier_layers=cfg.num_frontier_layers,
            num_frontier_heads=cfg.num_frontier_heads,
            block_hidden_size=cfg.block_hidden_size,
            num_block_layers=cfg.num_block_layers,
            max_extra_sigma=cfg.max_extra_sigma,
        ).to(device)
        self.model = model  # set now so a mid-training crash still leaves a usable
        # (partially trained) model on the instance.
        print(f"model parameters: {model.num_parameters():,}")
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

        # Unlike magnet/flexitokens/fanta's random-with-replacement sampling,
        # steps_per_epoch here is an exact epoch boundary (reshuffle-when-exhausted
        # below), not a rough periodic interval.
        steps_per_epoch = max(1, len(flat_items) // cfg.per_device_train_batch_size)
        total_steps = (
            cfg.max_steps
            if cfg.max_steps > 0
            else math.ceil(cfg.num_train_epochs * steps_per_epoch)
        )
        print(
            f"steps_per_epoch={steps_per_epoch} (periodic dev-eval interval) "
            f"-> total_steps={total_steps} ({total_steps / steps_per_epoch:.2f} epochs)"
        )
        scheduler = build_lr_scheduler(
            optimizer, total_steps, cfg.warmup_ratio, cfg.lr_scheduler_type
        )

        # Closure over `model` sees its current weights on every call (Python
        # closures capture the reference, not a snapshot). induce_spans is
        # language-agnostic at inference time (no script arg, unlike magnet's).
        eval_induce_fn_by_lang = None
        if self.eval_groups:
            eval_langs = {lang for group in self.eval_groups for lang in group}
            eval_induce_fn_by_lang = {
                lang: (lambda raw, m=model, d=device: induce_spans(m, raw, d))
                for lang in eval_langs
            }

        run = None
        if cfg.use_wandb:
            import wandb

            run = wandb.init(
                project=cfg.wandb_project,
                name=cfg.run_name or None,
                config={
                    **dataclasses.asdict(cfg),
                    "num_train_groups": len(self.train_groups),
                    "num_flattened_sequences": len(flat_items),
                    "model_parameters": model.num_parameters(),
                    "num_eval_groups": len(self.eval_groups) if self.eval_groups else 0,
                    "steps_per_epoch": steps_per_epoch,
                    "total_steps": total_steps,
                },
            )

        rng = np.random.default_rng(cfg.seed)
        order = rng.permutation(len(flat_items))
        pos = 0

        token_freq = defaultdict(Counter)
        loss_trace = []

        pbar = tqdm(range(total_steps), desc="training", unit="step")
        postfix = {}
        for step in pbar:
            if pos + cfg.per_device_train_batch_size > len(order):
                # Epoch boundary: reshuffle (fresh permutation each pass, matching
                # GRPOTrainer's DataLoader(shuffle=True) semantics).
                order = rng.permutation(len(flat_items))
                pos = 0
            batch_idx = order[pos : pos + cfg.per_device_train_batch_size]
            pos += cfg.per_device_train_batch_size
            batch = [flat_items[i] for i in batch_idx]

            tensors = [bytes_to_tensor(text, device) for _, text in batch]
            padded, lengths = _pad_batch(tensors, device)

            output = model(padded, lengths)
            loss, num_valid, num_correct = next_byte_loss(
                padded, lengths, output.logits
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()
            scheduler.step()

            # Reuses this step's already-computed assignment matrix -- no second
            # forward pass. Detached: bookkeeping, not part of the loss.
            boundaries = boundaries_from_assignment(output.assignment.detach(), lengths)
            for (lang, _), seq, actions in zip(batch, tensors, boundaries):
                token_freq[lang].update(spans_from_boundaries(seq, actions))

            loss_value = loss.item()
            loss_trace.append(loss_value)
            byte_accuracy = num_correct / num_valid if num_valid else 0.0
            postfix["loss"] = f"{loss_value:.3f}"
            postfix["acc"] = f"{byte_accuracy:.3f}"
            pbar.set_postfix(postfix)

            if run is not None:
                run.log(
                    {
                        "train/loss": loss_value,
                        "train/byte_accuracy": byte_accuracy,
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
                torch.save(
                    {
                        "config": dataclasses.asdict(cfg),
                        "state_dict": model.state_dict(),
                    },
                    cfg.output_dir,
                )

            if step % cfg.log_steps == 0:
                avg_span_len = _avg_span_length(token_freq)
                # Mean blocks used per sequence (mu at last real position + 1) --
                # a cheap proxy for segmentation coarseness that's available before
                # any hard-argmax discretization noise (useful early in training).
                last_idx = (lengths - 1).clamp_min(0)
                mean_blocks = (
                    (output.mu.detach().gather(1, last_idx.unsqueeze(1)).squeeze(1) + 1)
                    .mean()
                    .item()
                )
                collapse_warn = ""
                if avg_span_len and avg_span_len < 1.2:
                    collapse_warn = " <-- CHECK: drifting toward single-byte spans"
                elif avg_span_len > 40:
                    collapse_warn = " <-- CHECK: drifting toward whole-sequence spans"
                pbar.write(
                    f"[step {step:4d}] loss={loss_value:.4f} acc={byte_accuracy:.3f} "
                    f"avg_span={avg_span_len:.2f} mean_blocks/seq={mean_blocks:.2f} "
                    f"distinct_spans={sum(len(c) for c in token_freq.values())}{collapse_warn}"
                )
                if run is not None:
                    run.log(
                        {
                            "diagnostics/avg_span_length_running": avg_span_len,
                            "diagnostics/mean_blocks_per_seq": mean_blocks,
                            "diagnostics/num_distinct_spans_running": sum(
                                len(c) for c in token_freq.values()
                            ),
                            "diagnostics/char_collapse": int(
                                bool(avg_span_len) and avg_span_len < 1.2
                            ),
                            "diagnostics/sentence_collapse": int(avg_span_len > 40),
                        },
                        step=step,
                    )

            # Epoch-boundary held-out eval against the still-training model --
            # BOUQuET dev, capped by max_eval_samples (periodic, unlike
            # evaluate.py's one-time full scoring).
            if eval_induce_fn_by_lang and step % steps_per_epoch == 0:
                eval_sample = sample_eval_groups(
                    self.eval_groups, cfg.max_eval_samples, seed=cfg.seed
                )
                eval_results = evaluate_on_groups(eval_induce_fn_by_lang, eval_sample)
                report_eval(eval_results, label=f"manta step {step} dev")
                if run is not None:
                    run.log(eval_wandb_log_dict(eval_results), step=step)

        final_vocab = top_k_by_frequency(token_freq, cfg.vocab_size)
        if cfg.output_dir:
            torch.save(
                {"config": dataclasses.asdict(cfg), "state_dict": model.state_dict()},
                cfg.output_dir,
            )
            print(f"saved model checkpoint to {cfg.output_dir}")

        self.token_freq = token_freq
        self.vocab = final_vocab
        self.loss_trace = loss_trace

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
        return model, token_freq, final_vocab


def _span_length_histogram(token_freq):
    """Merged (across languages) span-length distribution, by frequency --
    feeds the collapse warnings and the smoke test's degeneracy check.
    Returns (hist: {length: count}, total_spans: int)."""
    hist = Counter()
    total = 0
    for counter in token_freq.values():
        for span, n in counter.items():
            hist[len(span)] += n
            total += n
    return hist, total


def _report_smoke_test_metrics(token_freq, final_vocab, loss_trace):
    """Feeds the smoke test's induced vocabulary into fairtok's own metrics
    functions, unmodified -- confirms this model's output is a drop-in match
    for whatever consumes fairtok's tokenizer output."""
    avg_span_len = _avg_span_length(token_freq)
    hist, total_spans = _span_length_histogram(token_freq)
    singleton_frac = hist.get(1, 0) / total_spans if total_spans else 0.0
    print(
        f"\nfinal vocab size={len(final_vocab)}  avg span length={avg_span_len:.2f} bytes"
    )
    print(
        f"span length histogram (top 8 lengths, by share of {total_spans} induced spans): "
        + ", ".join(
            f"{length}b={count / total_spans:.3f}"
            for length, count in sorted(hist.items())[:8]
        )
    )
    if singleton_frac > 0.9:
        print(
            f"NOTE: {singleton_frac:.1%} of induced spans are single bytes -- this is the EXPECTED "
            "consequence of training this bidirectional, boundary-free mechanism with a plain causal "
            "next-byte loss (no rate regularizer to push back against it -- see manta/model.py's module "
            "docstring, point 2, 'EMPIRICAL CONSEQUENCE' for the full explanation: the bidirectional "
            "assignment/pooling pathway can leak next-byte identity, and finer blocks widen that leak). "
            "Not hidden, not a crash -- surfaced here on purpose."
        )
    if avg_span_len > 40:
        print("WARNING: near full-sentence collapse (spans are almost never cut)")

    loss_head = sum(loss_trace[: min(5, len(loss_trace))]) / min(5, len(loss_trace))
    loss_tail = sum(loss_trace[-min(5, len(loss_trace)) :]) / min(5, len(loss_trace))
    print(
        f"\nloss: first-5-step avg={loss_head:.4f}  last-5-step avg={loss_tail:.4f}  (loss[0]={loss_trace[0]:.4f}, loss[-1]={loss_trace[-1]:.4f})"
    )

    print("\nper-language metrics on the induced (discretized) vocabulary:")
    for lang, counter in sorted(token_freq.items()):
        num_bytes = sum(len(span) * n for span, n in counter.items())
        num_tokens = sum(counter.values())
        rate = compression_rate(num_bytes, num_tokens)
        eff = renyi_efficiency(list(counter.values()))
        print(f"  {lang:16s} compression_rate={rate:6.3f}  renyi_efficiency={eff:.4f}")

    renyi_by_lang = {
        lang: renyi_efficiency(list(c.values())) for lang, c in token_freq.items() if c
    }
    gini = gini_coefficient(list(renyi_by_lang.values())) if renyi_by_lang else 0.0
    print(f"\ngini coefficient across per-language renyi efficiency: {gini:.4f}")

    return {
        "avg_span_len": avg_span_len,
        "singleton_frac": singleton_frac,
        "loss_head": loss_head,
        "loss_tail": loss_tail,
    }


def run_smoke_test():
    """Mirrors fairtok.train.run_smoke_test's role: a small trial run on
    synthetic data, gated by three assertions -- (1) no crash end to end,
    (2) loss decreased (mean of first 5 logged steps vs. last 5, since
    per-step loss is noisy), (3) induced spans aren't totally degenerate
    (not literally 100% single-byte, nor collapsed into one giant span per
    sentence). CPU-runnable in a couple minutes. Note: assertion 3 only
    rules out the literal extremes -- it does not rule out the heavy skew
    toward short spans this mechanism produces under a causal LM loss (see
    manta/model.py docstring point 2).
    """
    args = MantaConfig(
        max_steps=80, per_device_train_batch_size=8, vocab_size=384, log_steps=10
    )
    langs = list(LANG_PROFILES)
    train_groups = make_synthetic_parallel_groups(200, langs=langs, seed=args.seed)
    trainer = MantaTrainer(args, train_groups)
    model, token_freq, final_vocab = trainer.train()
    stats = _report_smoke_test_metrics(token_freq, final_vocab, trainer.loss_trace)

    assert stats["loss_tail"] < stats["loss_head"], (
        f"loss did not decrease: first-5-step avg={stats['loss_head']:.4f} "
        f"vs last-5-step avg={stats['loss_tail']:.4f}"
    )
    assert (
        stats["singleton_frac"] < 1.0
    ), "spans collapsed to literally 100% single bytes"
    assert (
        stats["avg_span_len"] < 40
    ), "spans collapsed toward one giant span per sentence"
    print(
        "\nsmoke test PASSED: no crash, loss decreased, spans are not totally degenerate."
    )

    return model, token_freq, final_vocab


if __name__ == "__main__":
    run_smoke_test()
