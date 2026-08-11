"""The FlexiTokens training loop: plain backprop, no REINFORCE/reward machinery at
all (unlike fairtok.train.GRPOTrainer) -- every piece of the loss (next-byte
cross-entropy, per-language boundary hinge) is differentiable end-to-end thanks to
the straight-through Gumbel-sigmoid boundary relaxation in flexitokens/model.py.

Mirrors fairtok.train.GRPOConfig / GRPOTrainer's general shape (a HF-TrainingArguments
-styled config dataclass + a Trainer class with .train()) and field naming wherever a
clean equivalent exists (max_steps, learning_rate, per_device_train_batch_size, seed,
vocab_size, output_dir) -- see FlexiTokensConfig below for the FlexiTokens-specific
knobs that have no GRPOConfig equivalent (d_model, gumbel_temperature, lambda_hinge,
margin_lambda, alpha_anchor, anchor_lang).

One step = one batch of parallel groups, exactly like GRPOTrainer -- a fixed number
of groups is drawn uniformly at random each step (with replacement across steps, not
a shuffled epoch traversal like GRPOTrainer's DataLoader; simpler, and fine at this
smoke-test scale where max_steps is small relative to corpus size). Every language in
a sampled group contributes one sequence to the batch (capped by group_sample_size,
same meaning as GRPOConfig's own field), and the boundary hinge loss pools
observations across every sequence sharing a language within the step, not per group
-- unlike GRPO's group-relative advantage, FlexiTokens' training signal has no notion
of "reward relative to this group's siblings" at all.
"""

import dataclasses
import math
import statistics
from collections import Counter, defaultdict

import numpy as np
import torch
from tqdm.auto import tqdm

from common.eval_common import (
    eval_wandb_log_dict,
    evaluate_on_groups,
    report_eval,
    sample_eval_groups,
)
from common.lr_schedule import build_lr_scheduler
from common.metrics import compression_rate, gini_coefficient, renyi_efficiency
from common.bytes_utils import bytes_to_tensor, spans_from_boundaries
from common.parity import compute_lang_parity_ratios
from common.reporting import collapse_stats
from common.vocab import top_k_by_frequency

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
):
    """Per-language target boundary rate alpha_L and band floor beta_L.

    Neither quantity is pinned down by the paper at the level of a concrete
    formula -- both are JUDGMENT CALLS, documented here in full:

    alpha_L (target rate, i.e. `k/N` this language "should" be near): pick one
    anchor language (English by default, matching this project's use of English
    as the common.oldi_data reporting pivot), and compute alpha_L PROPORTIONAL
    to how many bytes this language needs, on average, to say the same thing the
    anchor says -- using the genuinely N-way parallel groups from
    common.oldi_data (or common.data's synthetic stand-in) directly, since
    "the same content in language L" and "the same content in the anchor" are
    LITERALLY the same dict entry across languages in one group:

        ratio_L = mean(byte_len(L))  /  mean(byte_len(anchor))
                  over every group containing BOTH L and the anchor

        alpha_L = clamp(alpha_anchor / ratio_L, alpha_floor, alpha_ceiling)

    ratio_L itself is computed by common.parity.compute_lang_parity_ratios, shared
    with fairtok.train's own per-language target-rate derivation (same evidence,
    same formula -- see that module for why a single global rate is a fairness
    problem, not just a simplification).

    Rationale: if L systematically needs MORE bytes than the anchor to express
    the same content (e.g. multi-byte UTF-8 scripts, more verbose morphology),
    a FIXED boundary rate would give L systematically MORE tokens per sentence
    than the anchor gets for equivalent content -- exactly the cross-lingual
    unfairness this whole project's fairtok half is built to fight. Scaling
    alpha_L down by the same ratio (fewer boundaries -> longer average spans)
    is a first-order correction that keeps expected TOKEN count, not byte
    count, roughly comparable across languages. alpha_anchor=0.25 (~4
    bytes/token on average for the anchor) is a plausible byte-level-BPE-ish
    compression rate, not a number the paper specifies anywhere at the
    abstract/equation level. Floor/ceiling guard against a degenerate ratio_L
    (e.g. a near-empty sample) producing a nonsensical alpha_L outside [0, 1].

    beta_L (band floor): beta_L = alpha_L - margin_lambda * sigma_L, per the
    paper's own formula, where sigma_L is described as "the std of the
    compression rate for language L". We cannot compute an actual compression
    rate here -- compression rate is a PROPERTY OF A TRAINED TOKENIZER, and
    deriving alpha_L/beta_L happens before training starts (chicken-and-egg).
    JUDGMENT CALL: proxy sigma_L with that language's own coefficient of
    variation of raw byte sentence length (std/mean, over every sentence
    available for L, not just the anchor-paired subset used for alpha_L),
    scaled into alpha_L's own units by multiplying by alpha_L -- so sigma_L
    stays a dimensionless, rate-like quantity commensurate with alpha_L rather
    than an arbitrary absolute number. Intuition: a language whose sentences
    vary a lot in length in this corpus is a language whose "reasonable"
    compression rate is less pinned-down, so it gets a wider flexibility band.
    beta_L is clamped to [0, alpha_L] (a boundary rate cannot be negative, and
    the band's floor cannot exceed its own ceiling).

    Returns (alpha_by_lang: dict[str,float], beta_by_lang: dict[str,float],
    anchor_used: str -- the language actually used as the anchor, which can
    differ from `anchor_lang` if that language isn't present in the corpus;
    see the fallback below).
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
        beta = max(0.0, min(alpha, alpha - margin_lambda * sigma))

        alpha_by_lang[lang] = alpha
        beta_by_lang[lang] = beta

    return alpha_by_lang, beta_by_lang, anchor


@dataclasses.dataclass
class FlexiTokensConfig:
    """Mirrors fairtok.train.GRPOConfig's naming/style (HF-TrainingArguments-esque
    field names) wherever a clean equivalent exists; FlexiTokens-specific knobs with
    no GRPOConfig analogue keep their own domain-specific names, the same convention
    GRPOConfig itself uses for gamma/lambda_target/lambda_fair."""

    max_steps: int = 0  # 0 means derive from num_train_epochs * steps_per_epoch
    # (see FlexiTokensTrainer.train) -- matches fairtok.train.GRPOConfig's own
    # max_steps/num_train_epochs convention; set explicitly to override with a
    # raw step count instead.
    num_train_epochs: float = 3.0  # only takes effect if max_steps == 0. 1 "epoch"
    # here is steps_per_epoch steps (see FlexiTokensTrainer.train) -- a periodic-
    # checkpoint INTERVAL, not a guaranteed every-group-visited-once traversal
    # (this trainer samples groups randomly WITHOUT replacement per step but
    # doesn't track visitation ACROSS steps, unlike fairtok's shuffled
    # DataLoader) -- so "5 epochs" means "5 * steps_per_epoch steps."
    per_device_train_batch_size: int = 8  # counts parallel-sentence GROUPS, not raw
    # byte sequences -- same meaning as GRPOConfig's field of the same name.
    learning_rate: float = 3e-3
    seed: int = 0

    d_model: int = 64  # deliberately small -- see model.py's SCALE-DOWN NOTICE.
    nhead: int = 4
    num_pre_layers: int = 2
    num_mid_layers: int = 2
    num_post_layers: int = 2
    gumbel_temperature: float = 0.5  # lower = closer to a true hard Bernoulli sample
    # (less biased, higher-variance gradient); higher = smoother/more biased but
    # easier to optimize early in training. 0.5 is a common default in the
    # categorical/binary-concrete relaxation literature (Maddison et al. 2017,
    # Jang et al. 2017), not a value specified by the FlexiTokens paper itself.
    grad_clip_norm: float = 5.0  # pragmatic addition, not from the paper -- transformer
    # layers combined with Gumbel-sigmoid sampling noise can spike gradients early in
    # training, before the boundary predictor has settled into a sensible regime.

    lambda_hinge: float = 1.0  # weight of the boundary-rate hinge loss in the total
    # loss (loss = ce_loss + lambda_hinge * hinge_loss). Distinct from margin_lambda
    # below, which is the PAPER's own per-language band-WIDTH hyperparameter, not a
    # loss weight -- see derive_alpha_beta's docstring for margin_lambda's role.
    margin_lambda: float = 1.0
    anchor_lang: str = "eng"
    alpha_anchor: float = 0.25
    alpha_floor: float = 0.05
    alpha_ceiling: float = 0.9

    vocab_size: int = 384
    group_sample_size: int = 24  # cap languages rolled out per group per step,
    # regardless of how many a group actually offers -- same meaning as GRPOConfig's
    # own field.
    device: str = ""  # "" auto-detects cuda if available, else cpu.
    output_dir: str = ""  # "" disables checkpointing.
    use_wandb: bool = False  # matches fairtok.train.GRPOConfig's field of the same
    # name/role -- see FlexiTokensTrainer.train for the actual wandb.init/run.log calls.
    wandb_project: str = "flexitokens"
    run_name: str = ""

    max_eval_samples: int = 20  # cap on how many BOUQuET dev groups get scored at
    # each epoch-boundary evaluation (see FlexiTokensTrainer.train) -- 0 scores
    # every loaded dev group. Kept small since this runs periodically DURING
    # training, not once at the end (see evaluate.py, which always scores
    # everything).

    warmup_ratio: float = 0.1  # matches HF Trainer's own field name/default -- see
    # common.lr_schedule.build_lr_scheduler.
    lr_scheduler_type: str = "linear"  # "constant" (warmup only), "linear", or
    # "cosine" -- see common.lr_schedule.build_lr_scheduler. "linear" matches HF
    # Trainer's own default.


class FlexiTokensTrainer:
    """Shaped after fairtok.train.GRPOTrainer: construct with args + train_groups
    (a plain list of {lang: text} dicts, see common.oldi_data /
    common.data.make_synthetic_parallel_groups), call .train(), then read
    .model / .token_freq / .vocab / .alpha_by_lang / .beta_by_lang /
    .loss_history / .rate_history off the instance. train() also returns
    (model, token_freq, final_vocab, info) as a tuple for convenience."""

    def __init__(self, args: FlexiTokensConfig, train_groups, eval_groups=None):
        self.args = args
        self.train_groups = train_groups
        self.eval_groups = eval_groups  # BOUQuET dev, or None to skip periodic
        # epoch-boundary evaluation (see common.cli_data.load_bouquet_dev_for_training)
        self.device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.token_freq = None
        self.vocab = None
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
        )
        self.alpha_by_lang, self.beta_by_lang = alpha_by_lang, beta_by_lang
        print(f"anchor language: {anchor!r}")
        for lang in sorted(alpha_by_lang):
            print(
                f"  alpha[{lang}]={alpha_by_lang[lang]:.4f}  beta[{lang}]={beta_by_lang[lang]:.4f}"
            )
        default_alpha = float(np.mean(list(alpha_by_lang.values())))
        default_beta = float(np.mean(list(beta_by_lang.values())))

        # "Epoch" here means one pass' worth of steps (per_device_train_batch_size
        # counts GROUPS, see class docstring) -- used as the periodic-checkpoint
        # INTERVAL for the BOUQuET dev eval below, matching every other trainer's
        # own steps_per_epoch convention in this repo.
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

        # Built ONCE against the live `model` object -- a closure over `model`
        # keeps seeing its CURRENT weights on every call (Python closures capture
        # the object reference, not a snapshot), so this doesn't need rebuilding
        # each time the epoch boundary is hit. FlexiTokens' induce_spans is
        # language-agnostic at inference time, unlike magnet's (no script arg).
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

            # Harvest vocabulary from this step's REALIZED (hardened) boundaries --
            # same "count whatever the model actually produced along the way, apply
            # the fixed vocab budget only at the end" philosophy as
            # fairtok.train.GRPOTrainer (see its module docstring): no in-loop
            # vocab-size enforcement, just tallying spans as they're produced.
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
                # Deliberately aggregate-only every step (mean rate across whatever
                # languages this batch happened to include) -- flattening
                # per_lang_rate into one wandb metric per language, every step,
                # would mean hundreds of keys/step under --langs all. fairtok's own
                # per-language breakdown (fairness/renyi/{lang}) only logs every
                # fairness_refresh_steps for exactly this reason; this trainer has
                # no equivalent periodic-diagnostics mechanism to hang a similar
                # lower-frequency per-language log off of, so it's simply omitted
                # here rather than logged at the wrong (every-step) cadence.
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

            # Epoch-boundary held-out eval against the CURRENT (still-training)
            # model -- BOUQuET dev, capped by max_eval_samples since this runs
            # periodically, unlike evaluate.py's one-time full scoring.
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
    """Sanity check requested alongside the smoke test: common.metrics functions,
    UNMODIFIED, consuming this module's own token_freq output -- confirms
    FlexiTokens' induced vocabulary is a drop-in match for fairtok's existing
    evaluation pipeline, exactly like a fairtok.policy.BytePolicy vocabulary is."""
    per_lang_compression, per_lang_renyi = {}, {}
    for lang, counter in token_freq.items():
        if not counter:
            continue
        total_bytes = sum(len(span) * count for span, count in counter.items())
        total_tokens = sum(counter.values())
        per_lang_compression[lang] = compression_rate(total_bytes, total_tokens)
        per_lang_renyi[lang] = renyi_efficiency(list(counter.values()))
    gini = gini_coefficient(list(per_lang_renyi.values())) if per_lang_renyi else 0.0

    print("\ncommon.metrics sanity check on the FlexiTokens-induced vocabulary:")
    for lang in sorted(per_lang_compression):
        print(
            f"  {lang:16s} compression_rate={per_lang_compression[lang]:.3f} bytes/token   "
            f"renyi_efficiency={per_lang_renyi[lang]:.4f}"
        )
    print(f"  gini_coefficient(renyi_efficiency across languages) = {gini:.4f}")
    return per_lang_compression, per_lang_renyi, gini


def run_smoke_test():
    """Mirrors fairtok.train.run_smoke_test's pattern/gate: a small run on
    synthetic placeholder data (common.data.make_synthetic_parallel_groups),
    checked for:
      (a) no crash end-to-end,
      (b) loss actually decreasing (late-training average < early-training average),
      (c) boundary rates for DIFFERENT synthetic "languages" ending up DIFFERENT
          from each other, and away from the 0%/100% collapse extremes -- the
          entire point of the per-language hinge loss is to NOT force every
          language to the same fixed rate, so identical rates across languages
          would mean the hinge loss isn't doing anything distinguishable.
    Also prints common.metrics (compression_rate, renyi_efficiency,
    gini_coefficient) on the induced vocabulary as a pipeline-compatibility check.
    """
    from common.data import LANG_PROFILES, make_synthetic_parallel_groups

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
