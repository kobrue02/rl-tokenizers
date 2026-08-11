"""Plain backprop training loop for MantaModel.

Deliberately much simpler than fairtok.train.GRPOTrainer: there is no
REINFORCE, no reward, no per-group baseline, no fairness scalar, and no
rate-consistency loss -- see manta.model's module docstring, point 3, for
why MANTa genuinely has none of that machinery, not just a simplified
version of it. The only loss is next-byte cross-entropy
(manta.model.next_byte_loss), and every byte position contributes to it
independently; "groups" of parallel sentences (common.oldi_data's unit of
organization) are used here purely as a convenient SOURCE of many
sentences across many languages, not as a training-time structure the way
GRPOTrainer's group-relative advantage needs them to be. Concretely: this
trainer flattens every group into a flat list of (lang, text) pairs up
front and samples plain shuffled minibatches of individual sentences from
that flat list -- there is no group-of-languages unit at batch time at all.

Field names in MantaConfig mirror fairtok.train.GRPOConfig's own naming
(itself modeled on HF TrainingArguments) wherever a clean equivalent
exists, for the same reason GRPOConfig does it: anyone coming from either
codebase recognizes max_steps/learning_rate/per_device_train_batch_size/
seed immediately. One naming note worth flagging: GRPOConfig's
per_device_train_batch_size counts parallel-sentence GROUPS (each
expanding to multiple language sequences); MantaConfig's counts individual
byte SEQUENCES directly, since there is no group structure at batch time
here to count instead (see above).
"""

import dataclasses
import math
from collections import Counter, defaultdict

import numpy as np
import torch
from tqdm.auto import tqdm

from common.data import LANG_PROFILES, make_synthetic_parallel_groups
from common.eval_common import (
    eval_wandb_log_dict,
    evaluate_on_groups,
    report_eval,
    sample_eval_groups,
)
from common.lr_schedule import build_lr_scheduler
from common.metrics import compression_rate, gini_coefficient, renyi_efficiency
from common.bytes_utils import bytes_to_tensor, spans_from_boundaries
from common.reporting import collapse_stats
from common.vocab import top_k_by_frequency

from .model import MantaModel, next_byte_loss
from .segment import boundaries_from_assignment, induce_spans


@dataclasses.dataclass
class MantaConfig:
    """See this module's docstring for the naming-convention note vs.
    fairtok.train.GRPOConfig (same spirit, adapted where MANTa's training
    loop is structurally simpler -- no group/fairness/rate-consistency
    machinery to name fields for)."""

    max_steps: int = 0  # 0 means derive from num_train_epochs * steps_per_epoch
    # (see MantaTrainer.train) -- matches fairtok.train.GRPOConfig's own
    # max_steps/num_train_epochs convention; set explicitly to override with a
    # raw step count instead (run_smoke_test below does exactly this, for a
    # fixed, predictable number of updates regardless of corpus size).
    num_train_epochs: float = 3.0  # only takes effect if max_steps == 0. Unlike
    # magnet/flexitokens/fanta, MANTa's steps_per_epoch IS a real traversal
    # boundary (this trainer reshuffles a full permutation of every flattened
    # sequence once it's exhausted -- see the loop below), so "5 epochs" here
    # really does mean 5 full passes over the corpus.
    per_device_train_batch_size: int = 8  # counts individual byte SEQUENCES (see
    # module docstring) -- NOT parallel-sentence groups, unlike GRPOConfig's field
    # of the same name.
    learning_rate: float = 3e-3
    seed: int = 0
    vocab_size: int = (
        384  # final vocab budget passed to common.vocab.top_k_by_frequency,
    )
    # applied once after training -- matches fairtok's own two-stage
    # "never enforce the budget during training, only harvest it after" design
    # (see fairtok/vocab.py's module docstring), even though MANTa has no in-loop
    # reward that a hard budget could distort in the first place; keeping the same
    # two-stage shape just makes the two baselines' vocab outputs directly comparable.
    dim: int = 64  # byte embedding / model width -- see manta.model's module
    # docstring point 8 for why this is sized to match fairtok.policy.BytePolicy's
    # own scale (hidden_dim 64-128), not the original paper's ~200M-param model.
    window: int = 8  # frontier predictor's local-attention half-window (+/- window)
    num_frontier_layers: int = 2
    num_frontier_heads: int = 4
    block_hidden_size: int = 64
    num_block_layers: int = 1
    max_extra_sigma: float = (
        3.0  # candidate block range truncation, mu_L + this*sigma_L
    )
    max_grad_norm: float = 1.0  # gradient clipping -- MANTa's forward pass chains a
    # Gaussian log-density through a softmax through two more matmuls before the
    # loss even starts, more numerically fragile than fairtok's plain GRU cells, so
    # this is cheap insurance against an occasional bad batch producing a huge step.
    device: str = ""  # "" auto-detects cuda if available, else cpu -- same convention
    # as GRPOConfig.device.
    log_steps: int = 10  # progress-printing interval; plays the same role
    # GRPOConfig.fairness_refresh_steps does for periodic console output, but
    # there's no fairness scalar to refresh here, just a print cadence.
    output_dir: str = ""  # "" disables checkpoint saving, matching GRPOConfig's
    # output_dir convention (empty string = skip).
    save_steps: int = 0  # 0 disables periodic saving, same convention as GRPOConfig.
    use_wandb: bool = False  # matches fairtok.train.GRPOConfig's field of the same
    # name/role -- see MantaTrainer.train for the actual wandb.init/run.log calls.
    wandb_project: str = "manta"
    run_name: str = ""

    max_eval_samples: int = 20  # cap on how many BOUQuET dev groups get scored at
    # each epoch-boundary evaluation (see MantaTrainer.train) -- 0 scores every
    # loaded dev group. Kept small since this runs periodically DURING training,
    # not once at the end (see evaluate.py, which always scores everything).

    warmup_ratio: float = 0.1  # matches HF Trainer's own field name/default -- see
    # common.lr_schedule.build_lr_scheduler.
    lr_scheduler_type: str = "linear"  # "constant" (warmup only), "linear", or
    # "cosine" -- see common.lr_schedule.build_lr_scheduler. "linear" matches HF
    # Trainer's own default.


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


class MantaTrainer:
    """Construct with args + train_dataset (a list of dicts {lang: text}, the
    same shape common.oldi_data.load_all_training_groups / common.data's
    make_synthetic_parallel_groups produce -- reused unmodified), call
    .train(), then read .model / .token_freq / .vocab off the instance
    (train() also returns them as a tuple, mirroring GRPOTrainer's
    return-and-store-on-self convention)."""

    def __init__(self, args: MantaConfig, train_dataset, eval_groups=None):
        self.args = args
        self.train_dataset = train_dataset
        self.eval_groups = eval_groups  # BOUQuET dev, or None to skip periodic
        # epoch-boundary evaluation (see common.cli_data.load_bouquet_dev_for_training)
        self.device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.token_freq = None
        self.vocab = None

    def train(self):
        cfg = self.args
        device = self.device
        torch.manual_seed(cfg.seed)
        print(f"device={device}")

        # Flatten every group into a flat list of (lang, text) pairs ONCE, up
        # front -- see module docstring for why there's no group structure at
        # batch time here (unlike GRPOTrainer's GroupLanguageCollator).
        flat_items = [
            (lang, text) for group in self.train_dataset for lang, text in group.items()
        ]
        print(
            f"corpus={len(self.train_dataset)} groups -> {len(flat_items)} flattened (lang, text) sequences"
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
        self.model = model  # set now, not just at the end, so a mid-training crash/
        # interrupt still leaves a usable (if partially trained) model on the instance
        print(f"model parameters: {model.num_parameters():,}")
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

        # Unlike magnet/flexitokens/fanta's random-with-replacement group sampling,
        # MANTa already has a REAL epoch boundary (the reshuffle-when-exhausted
        # logic in the loop below) -- steps_per_epoch here is exact, not a rough
        # periodic-checkpoint interval.
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

        # Built ONCE against the live `model` object -- a closure over `model`
        # keeps seeing its CURRENT weights on every call (Python closures capture
        # the object reference, not a snapshot). MANTa's induce_spans is
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
                    "num_train_groups": len(self.train_dataset),
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
                # Epoch boundary: reshuffle, same "fresh permutation, every group
                # visited exactly once per pass" semantics as GRPOTrainer's
                # DataLoader(shuffle=True) reuses across re-iterations.
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

            # Track induced spans as training progresses, reusing the assignment
            # matrix this step's forward pass ALREADY computed -- no second
            # forward pass needed (see segment.boundaries_from_assignment's
            # docstring). Detached: this is bookkeeping, not part of the loss.
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
                # mean number of blocks actually used, per sequence in THIS batch
                # (mu at the last real position + 1) -- a cheap, differentiable-
                # quantity-derived proxy for "how coarse is the current
                # segmentation," independent of (and available before) any
                # hard-argmax discretization noise; useful for watching whether
                # the frontier predictor is doing anything at all early in
                # training, when the discretized spans themselves can be noisy
                # or near-degenerate (see manta.segment's module docstring).
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

            # Epoch-boundary held-out eval against the CURRENT (still-training)
            # model -- BOUQuET dev, capped by max_eval_samples since this runs
            # periodically, unlike evaluate.py's one-time full scoring.
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
    used both for the collapse warnings below and for the smoke test's own
    "not totally degenerate" pass/fail check. Returns (hist: {length: count},
    total_spans: int)."""
    hist = Counter()
    total = 0
    for counter in token_freq.values():
        for span, n in counter.items():
            hist[len(span)] += n
            total += n
    return hist, total


def _report_smoke_test_metrics(token_freq, final_vocab, loss_trace):
    """Feed the smoke test's induced vocabulary straight into fairtok's own
    metrics functions, UNMODIFIED -- the whole point of reusing
    common.metrics/common.vocab rather than writing MANTa-specific
    versions is to confirm this model's output is a drop-in match for
    whatever consumes fairtok's own tokenizer output."""
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
    synthetic placeholder data (common.data.make_synthetic_parallel_groups,
    reused unmodified), gated by three explicit assertions:

      1. no crash getting here at all (train() and metric computation ran
         end to end over real, padded, variable-length batches);
      2. the loss actually decreased (compares the mean of the first 5
         logged step losses against the mean of the last 5, not just the
         single first/last value, since per-step loss is noisy -- see the
         printed loss trace);
      3. the induced (hard-discretized) spans aren't TOTALLY degenerate:
         not literally 100% single-byte, and not collapsed the other way
         into one giant span per sentence either.

    Small enough to run on CPU in a couple of minutes, same scale as
    fairtok's own smoke test. See _report_smoke_test_metrics's printed
    span-length histogram for the (expected, documented -- see
    manta/model.py) heavy skew toward short spans this mechanism produces
    under a plain causal LM loss; assertion 3 only rules out the LITERAL
    degenerate extremes, not that skew.
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
