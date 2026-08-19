"""FlexiTokens training loop: plain backprop, no REINFORCE/reward machinery
(unlike fairtok.train.GRPOTrainer) -- next-byte CE and the per-language
boundary hinge are both differentiable end-to-end via model.py's
straight-through Gumbel-sigmoid relaxation.

Mirrors GRPOConfig/GRPOTrainer's shape (HF-TrainingArguments-styled config +
Trainer.train()) and field naming where an equivalent exists; FlexiTokens-
specific knobs (d_model, gumbel_temperature, lambda_hinge, margin_lambda,
alpha_anchor, anchor_lang) have no GRPOConfig analogue.

One step = one batch of parallel groups, drawn uniformly at random (with
replacement across steps, not a shuffled epoch traversal -- simpler, fine at
smoke-test scale). Each language in a sampled group contributes one sequence
(capped by group_sample_size); the boundary hinge loss pools observations
per language across the whole step, not per group -- unlike GRPO's
group-relative advantage, there's no "reward relative to this group's
siblings" here.
"""

import dataclasses
import math
import statistics
from collections import Counter, defaultdict

import numpy as np
import torch
from tqdm.auto import tqdm

from common.eval.cross_tokenizer import (
    eval_wandb_log_dict,
    evaluate_on_groups,
    report_eval,
    sample_eval_groups,
)
from common.training.lr_schedule import build_lr_scheduler
from common.eval.metrics import compression_rate, gini_coefficient, renyi_efficiency
from common.bytes_utils import bytes_to_tensor, spans_from_boundaries
from common.eval.parity import compute_lang_parity_ratios
from common.eval.reporting import collapse_stats
from common.vocab import top_k_by_frequency

from ..base import BaseTokenizerConfig, BaseTokenizerTrainer
from .model import FlexiTokensModel, boundary_hinge_loss, next_byte_loss, pad_byte_batch
from .segment import induce_spans


def _byte_len(text):
    return len(text.encode("utf-8")) if isinstance(text, str) else len(text)


def derive_alpha_beta(
    train_groups,
    anchor_lang="eng",
    alpha_anchor=0.25,
    margin_lambda=1.0,
    alpha_floor=0.05,
    alpha_ceiling=0.9,
    beta_floor=0.02,
):
    """Per-language target boundary rate alpha_L and band floor beta_L.

    Neither is pinned down by the paper as a concrete formula -- both are
    JUDGMENT CALLS:

    alpha_L: pick an anchor language (English by default) and scale alpha_L
    inversely to how many more bytes this language needs, on average, to say
    the same thing as the anchor -- using genuinely N-way parallel groups so
    "same content" is literally the same dict entry across languages:

        ratio_L = mean(byte_len(L)) / mean(byte_len(anchor))  [over groups with both]
        alpha_L = clamp(alpha_anchor / ratio_L, alpha_floor, alpha_ceiling)

    (ratio_L via common.eval.parity.compute_lang_parity_ratios, shared with
    fairtok.train's own target-rate derivation.) Rationale: a language
    needing more bytes per equivalent content, under a FIXED boundary rate,
    would get more tokens per sentence than the anchor for the same content
    -- exactly the cross-lingual unfairness fairtok is built to fight.
    Scaling alpha_L down by the same ratio keeps expected TOKEN count, not
    byte count, comparable across languages. alpha_anchor=0.25 (~4
    bytes/token) is a plausible byte-BPE-ish rate, not paper-specified.
    Floor/ceiling guard against a degenerate ratio_L pushing alpha_L outside
    [0, 1].

    beta_L = alpha_L - margin_lambda * sigma_L (the paper's formula), where
    sigma_L is "std of the compression rate for language L" -- but
    compression rate is a property of a TRAINED tokenizer, and alpha_L/beta_L
    must be derived before training (chicken-and-egg). JUDGMENT CALL: proxy
    sigma_L with L's coefficient of variation of raw byte sentence length
    (std/mean), scaled by alpha_L to stay commensurate with it. Intuition: a
    language whose sentence lengths vary more has a less pinned-down
    "reasonable" rate, so it gets a wider band.

    beta_floor (JUDGMENT CALL, not in the paper): without it, beta_L is only
    clamped to >= 0, not >= some positive value. Confirmed live to matter --
    on real BOUQuET training data, 258 of 259 languages' coefficient of
    variation exceeds 1/margin_lambda, driving beta_L to EXACTLY 0 for
    99.6% of languages. boundary_hinge_loss's lower-bound term
    (max(beta - rate, 0)) is then permanently zero for those languages --
    the paper's own anti-collapse safeguard against "compressing less than
    beta_L" (see that function's docstring) silently never fires. A real
    trained checkpoint collapsed to ~0.6% boundary rate (avg 164
    bytes/token, vs. an intended ~4 bytes/token) as a direct result: with no
    floor and CE loss providing little pressure toward frequent
    segmentation on its own, only alpha_L's upper bound remained, and
    nothing stopped the rate from drifting arbitrarily low. beta_floor
    guarantees every language keeps SOME lower-bound pressure regardless of
    how large its computed sigma_L is; 0.02 sits comfortably below
    alpha_floor's own 0.05 (the smallest alpha_L can ever be), so a nonzero
    band survives even for the tightest-alpha language. beta_L is clamped
    to [beta_floor, alpha_L] (the upper clamp still wins if beta_floor would
    otherwise exceed alpha_L, matching the original [0, alpha_L] clamp's own
    intent).

    Returns (alpha_by_lang, beta_by_lang, anchor_used -- may differ from
    `anchor_lang` if that language isn't present in the corpus).
    """
    ratio_by_lang, anchor = compute_lang_parity_ratios(train_groups, anchor_lang)
    if anchor != anchor_lang:
        print(
            f"[flexitokens] anchor language {anchor_lang!r} not present in this corpus; "
            f"falling back to {anchor!r} as the alpha_L anchor"
        )

    lengths_by_lang = defaultdict(list)
    for group in train_groups:
        for lang, text in group.items():
            lengths_by_lang[lang].append(_byte_len(text))

    alpha_by_lang, beta_by_lang = {}, {}
    for lang, all_lens in lengths_by_lang.items():
        ratio = ratio_by_lang[lang]
        alpha = max(alpha_floor, min(alpha_ceiling, alpha_anchor / max(ratio, 1e-6)))

        mean_len = sum(all_lens) / len(all_lens)
        std_len = statistics.pstdev(all_lens) if len(all_lens) > 1 else 0.0
        cv = std_len / mean_len if mean_len > 0 else 0.0
        sigma = alpha * cv
        beta = min(alpha, max(beta_floor, alpha - margin_lambda * sigma))

        alpha_by_lang[lang] = alpha
        beta_by_lang[lang] = beta

    return alpha_by_lang, beta_by_lang, anchor


@dataclasses.dataclass
class FlexiTokensConfig(BaseTokenizerConfig):
    """vocab_size/output_dir/use_wandb/wandb_project/run_name inherited from
    BaseTokenizerConfig (except wandb_project's default). Otherwise mirrors
    GRPOConfig's naming/style; FlexiTokens-specific knobs with no GRPOConfig
    analogue keep their own descriptive names."""

    max_steps: int = 0  # 0 -> derive from num_train_epochs * steps_per_epoch (see
    # FlexiTokensTrainer.train); matches GRPOConfig's convention. Set explicitly
    # to override with a raw step count.
    num_train_epochs: float = 3.0  # only if max_steps == 0. "epoch" = steps_per_
    # epoch steps, a periodic-checkpoint interval -- groups are sampled without
    # replacement per step but visitation isn't tracked across steps (unlike a
    # shuffled DataLoader), so "5 epochs" just means 5 * steps_per_epoch steps.
    per_device_train_batch_size: int = 8  # counts parallel-sentence GROUPS, not raw
    # byte sequences -- same meaning as GRPOConfig's field of the same name.
    learning_rate: float = 3e-3
    seed: int = 0

    d_model: int = 64  # deliberately small -- see model.py's SCALE-DOWN NOTICE.
    nhead: int = 4
    num_pre_layers: int = 2
    num_mid_layers: int = 2
    num_post_layers: int = 2
    gumbel_temperature: float = 0.5  # lower = closer to hard Bernoulli (less
    # biased, higher-variance gradient); higher = smoother/easier early
    # optimization. Common default in the concrete-relaxation literature
    # (Maddison/Jang 2017), not paper-specified.
    grad_clip_norm: float = 5.0  # pragmatic addition, not from the paper --
    # transformer layers + Gumbel-sigmoid noise can spike gradients early on.

    lambda_hinge: float = 1.0  # weight of the boundary-rate hinge loss (loss =
    # ce_loss + lambda_hinge * hinge_loss). Distinct from margin_lambda below,
    # the paper's own band-WIDTH hyperparameter, not a loss weight.
    margin_lambda: float = 1.0
    anchor_lang: str = "eng"
    alpha_anchor: float = 0.25
    alpha_floor: float = 0.05
    alpha_ceiling: float = 0.9
    beta_floor: float = 0.02  # see derive_alpha_beta's own docstring -- without
    # this, beta_L collapses to exactly 0 for ~99.6% of languages on real
    # corpus data (confirmed live), silently disabling the paper's
    # anti-under-segmentation safeguard and letting the boundary predictor
    # drift to a near-zero boundary rate.

    # vocab_size (inherited, default 384).
    group_sample_size: int = 24  # cap languages rolled out per group per step,
    # regardless of how many a group actually offers -- same meaning as GRPOConfig's
    # own field.
    device: str = ""  # "" auto-detects cuda if available, else cpu.
    # output_dir (inherited, default ""): disables checkpointing.
    # use_wandb (inherited, default False) -- see FlexiTokensTrainer.train for
    # the actual wandb.init/run.log calls.
    wandb_project: str = "flexitokens"
    # run_name (inherited, default "").

    max_eval_samples: int = 20  # cap on BOUQuET dev groups scored per
    # epoch-boundary eval (0 = score all). Kept small since this runs
    # periodically during training, unlike evaluate.py's one-time full scoring.

    warmup_ratio: float = 0.1  # matches HF Trainer's own field name/default -- see
    # common.training.lr_schedule.build_lr_scheduler.
    lr_scheduler_type: str = "linear"  # "constant" (warmup only), "linear", or
    # "cosine" -- see common.training.lr_schedule.build_lr_scheduler. "linear" matches HF
    # Trainer's own default.


class FlexiTokensTrainer(BaseTokenizerTrainer):
    """Shaped after fairtok.train.GRPOTrainer: construct with args + train_groups
    (list of {lang: text} dicts), call .train(), then read .model/.token_freq/
    .vocab/.alpha_by_lang/.beta_by_lang/.loss_history/.rate_history off the
    instance. train() also returns (model, token_freq, final_vocab, info)."""

    def __init__(self, args: FlexiTokensConfig, train_groups, eval_groups=None):
        super().__init__(args, train_groups, eval_groups)
        self.device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.alpha_by_lang = None
        self.beta_by_lang = None
        self.loss_history = []
        self.rate_history = defaultdict(list)

    def train(self):
        cfg = self.args
        torch.manual_seed(cfg.seed)
        rng = np.random.default_rng(cfg.seed)
        device = self.device
        print(f"device={device}")

        alpha_by_lang, beta_by_lang, anchor = derive_alpha_beta(
            self.train_groups,
            cfg.anchor_lang,
            cfg.alpha_anchor,
            cfg.margin_lambda,
            cfg.alpha_floor,
            cfg.alpha_ceiling,
            cfg.beta_floor,
        )
        self.alpha_by_lang, self.beta_by_lang = alpha_by_lang, beta_by_lang
        print(f"anchor language: {anchor!r}")
        for lang in sorted(alpha_by_lang):
            print(
                f"  alpha[{lang}]={alpha_by_lang[lang]:.4f}  beta[{lang}]={beta_by_lang[lang]:.4f}"
            )
        default_alpha = float(np.mean(list(alpha_by_lang.values())))
        default_beta = float(np.mean(list(beta_by_lang.values())))

        # "Epoch" = one pass' worth of steps (per_device_train_batch_size counts
        # GROUPS) -- the periodic-checkpoint interval for the dev eval below.
        steps_per_epoch = max(
            1, math.ceil(len(self.train_groups) / cfg.per_device_train_batch_size)
        )
        total_steps = (
            cfg.max_steps
            if cfg.max_steps > 0
            else math.ceil(cfg.num_train_epochs * steps_per_epoch)
        )
        print(
            f"steps_per_epoch={steps_per_epoch} (periodic dev-eval interval) "
            f"-> total_steps={total_steps} ({total_steps / steps_per_epoch:.2f} epochs)"
        )

        run = None
        if cfg.use_wandb:
            import wandb

            run = wandb.init(
                project=cfg.wandb_project,
                name=cfg.run_name or None,
                config={
                    **dataclasses.asdict(cfg),
                    "alpha_by_lang": alpha_by_lang,
                    "beta_by_lang": beta_by_lang,
                    "anchor": anchor,
                    "num_train_groups": len(self.train_groups),
                    "num_eval_groups": len(self.eval_groups) if self.eval_groups else 0,
                    "steps_per_epoch": steps_per_epoch,
                    "total_steps": total_steps,
                },
            )

        model = FlexiTokensModel(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            num_pre_layers=cfg.num_pre_layers,
            num_mid_layers=cfg.num_mid_layers,
            num_post_layers=cfg.num_post_layers,
            gumbel_temperature=cfg.gumbel_temperature,
        ).to(device)
        self.model = (
            model  # set now so anything inspecting mid-training sees the live model
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
        scheduler = build_lr_scheduler(
            optimizer, total_steps, cfg.warmup_ratio, cfg.lr_scheduler_type
        )

        # Built once against the live `model` -- the closure captures the object
        # reference, so it sees current weights without rebuilding at each epoch
        # boundary. FlexiTokens' induce_spans is language-agnostic (no script
        # arg, unlike magnet's).
        eval_induce_fn_by_lang = None
        if self.eval_groups:
            eval_langs = {lang for group in self.eval_groups for lang in group}
            eval_induce_fn_by_lang = {
                lang: (lambda raw, m=model, d=device: induce_spans(m, raw, d))
                for lang in eval_langs
            }

        token_freq = defaultdict(Counter)
        n_groups = len(self.train_groups)

        pbar = tqdm(range(total_steps), desc="training", unit="step")
        for step in pbar:
            batch_size = min(cfg.per_device_train_batch_size, n_groups)
            group_idx = rng.choice(n_groups, size=batch_size, replace=False)
            batch_groups = [self.train_groups[i] for i in group_idx]

            batch_items = []  # (lang, byte_tensor)
            for group in batch_groups:
                langs = list(group.keys())
                if cfg.group_sample_size and len(langs) > cfg.group_sample_size:
                    langs = list(
                        rng.choice(langs, size=cfg.group_sample_size, replace=False)
                    )
                for lang in langs:
                    batch_items.append((lang, bytes_to_tensor(group[lang], device)))
            langs = [lang for lang, _ in batch_items]
            byte_ids, lengths = pad_byte_batch([seq for _, seq in batch_items], device)

            optimizer.zero_grad()
            out = model(byte_ids, lengths, deterministic=False)
            ce_loss, _ = next_byte_loss(out["logits"], byte_ids, out["valid_mask"])
            hinge_loss, per_lang_rate = boundary_hinge_loss(
                out["boundaries"],
                out["valid_mask"],
                langs,
                alpha_by_lang,
                beta_by_lang,
                default_alpha,
                default_beta,
            )
            loss = ce_loss + cfg.lambda_hinge * hinge_loss
            loss.backward()
            if cfg.grad_clip_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            optimizer.step()
            scheduler.step()

            self.loss_history.append(float(loss.item()))
            for lang, rate in per_lang_rate.items():
                self.rate_history[lang].append(rate)

            # Harvest vocabulary from this step's realized (hardened) boundaries --
            # no in-loop vocab-size enforcement, just tallying spans as produced
            # (same philosophy as GRPOTrainer).
            with torch.no_grad():
                hard_boundaries = out["boundaries"].detach().round().long()
            for i, (lang, byte_seq) in enumerate(batch_items):
                length = int(lengths[i].item())
                actions = hard_boundaries[i, :length].tolist()
                token_freq[lang].update(spans_from_boundaries(byte_seq, actions))

            pbar.set_postfix(
                loss=f"{loss.item():.3f}",
                ce=f"{ce_loss.item():.3f}",
                hinge=f"{hinge_loss.item():.4f}",
            )

            if run is not None:
                # Aggregate-only every step (mean rate across whatever languages
                # this batch included) -- logging per_lang_rate as one wandb metric
                # per language every step would mean hundreds of keys/step under
                # --langs all. fairtok avoids this by logging per-language
                # breakdowns only periodically; this trainer has no equivalent
                # mechanism, so per-language logging is omitted rather than done
                # at the wrong cadence.
                run.log(
                    {
                        "train/loss": loss.item(),
                        "train/ce_loss": ce_loss.item(),
                        "train/hinge_loss": hinge_loss.item(),
                        "train/mean_rate": (
                            float(np.mean(list(per_lang_rate.values())))
                            if per_lang_rate
                            else 0.0
                        ),
                        "train/learning_rate": scheduler.get_last_lr()[0],
                    },
                    step=step,
                )

            # Epoch-boundary held-out eval against the current (still-training)
            # model -- BOUQuET dev, capped by max_eval_samples, unlike
            # evaluate.py's one-time full scoring.
            if eval_induce_fn_by_lang and step % steps_per_epoch == 0:
                eval_sample = sample_eval_groups(
                    self.eval_groups, cfg.max_eval_samples, seed=cfg.seed
                )
                eval_results = evaluate_on_groups(eval_induce_fn_by_lang, eval_sample)
                report_eval(eval_results, label=f"flexitokens step {step} dev")
                if run is not None:
                    run.log(eval_wandb_log_dict(eval_results), step=step)

        final_vocab = top_k_by_frequency(token_freq, cfg.vocab_size)
        self.token_freq = token_freq
        self.vocab = final_vocab

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
            torch.save(
                {"state_dict": model.state_dict(), "config": dataclasses.asdict(cfg)},
                cfg.output_dir,
            )
            print(f"saved checkpoint to {cfg.output_dir}")

        info = {
            "alpha": alpha_by_lang,
            "beta": beta_by_lang,
            "anchor": anchor,
            "loss_history": self.loss_history,
            "rate_history": dict(self.rate_history),
        }
        return model, token_freq, final_vocab, info


def _print_vocab_metrics(token_freq):
    """Sanity check: unmodified common.eval.metrics functions consuming this
    module's token_freq output, confirming FlexiTokens' induced vocabulary is
    a drop-in match for fairtok's evaluation pipeline."""
    per_lang_compression, per_lang_renyi = {}, {}
    for lang, counter in token_freq.items():
        if not counter:
            continue
        total_bytes = sum(len(span) * count for span, count in counter.items())
        total_tokens = sum(counter.values())
        per_lang_compression[lang] = compression_rate(total_bytes, total_tokens)
        per_lang_renyi[lang] = renyi_efficiency(list(counter.values()))
    gini = gini_coefficient(list(per_lang_renyi.values())) if per_lang_renyi else 0.0

    print("\ncommon.eval.metrics sanity check on the FlexiTokens-induced vocabulary:")
    for lang in sorted(per_lang_compression):
        print(
            f"  {lang:16s} compression_rate={per_lang_compression[lang]:.3f} bytes/token   "
            f"renyi_efficiency={per_lang_renyi[lang]:.4f}"
        )
    print(f"  gini_coefficient(renyi_efficiency across languages) = {gini:.4f}")
    return per_lang_compression, per_lang_renyi, gini


def run_smoke_test():
    """Mirrors fairtok.train.run_smoke_test's pattern: a small run on synthetic
    placeholder data, checked for (a) no crash, (b) loss decreasing, (c)
    boundary rates for different synthetic "languages" ending up DIFFERENT
    from each other and away from the 0%/100% collapse extremes -- identical
    rates would mean the per-language hinge loss isn't doing anything
    distinguishable.
    Also prints common.eval.metrics on the induced vocabulary as a
    pipeline-compatibility check.
    """
    from common.data.synthetic import LANG_PROFILES, make_synthetic_parallel_groups

    args = FlexiTokensConfig(
        max_steps=80,
        per_device_train_batch_size=6,
        vocab_size=256,
        anchor_lang="high_resource",  # synthetic data has no "eng" key -- name the
        # anchor explicitly rather than relying on derive_alpha_beta's fallback.
    )
    langs = list(LANG_PROFILES)
    train_groups = make_synthetic_parallel_groups(
        300, langs=langs, seed=args.seed, min_len=30, max_len=80
    )

    trainer = FlexiTokensTrainer(args, train_groups)
    model, token_freq, final_vocab, info = trainer.train()

    window = min(10, len(trainer.loss_history) // 2) or 1
    early = float(np.mean(trainer.loss_history[:window]))
    late = float(np.mean(trainer.loss_history[-window:]))
    print(
        f"\nloss: early_avg(first {window})={early:.4f}  late_avg(last {window})={late:.4f}"
    )
    assert late < early, f"loss did not decrease: early={early:.4f} late={late:.4f}"

    final_rates = {
        lang: float(np.mean(v[-window:])) for lang, v in trainer.rate_history.items()
    }
    print(
        f"final per-language boundary rates (avg of last {window} steps): {final_rates}"
    )
    for lang, rate in final_rates.items():
        assert 0.01 < rate < 0.99, f"boundary rate collapsed for {lang!r}: {rate:.4f}"
    rate_spread = max(final_rates.values()) - min(final_rates.values())
    print(f"boundary-rate spread across languages: {rate_spread:.4f}")
    assert (
        rate_spread > 1e-3
    ), "boundary rates identical across languages -- hinge loss had no differentiating effect"

    print(f"final vocab size={len(final_vocab)}")
    _print_vocab_metrics(token_freq)

    return model, token_freq, final_vocab, info


if __name__ == "__main__":
    run_smoke_test()
