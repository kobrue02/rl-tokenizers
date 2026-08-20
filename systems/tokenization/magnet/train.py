"""MAGNET training loop -- plain backprop, no RL/REINFORCE machinery (unlike
fairtok.train.GRPOTrainer): the boundary predictor is differentiable
end-to-end via model.py's Gumbel-sigmoid + straight-through trick, so this is
just a next-byte CE loss + a boundary-rate loss, summed and backpropagated
like an ordinary supervised loss.

MagnetConfig/MagnetTrainer mirror fairtok.train.GRPOConfig/GRPOTrainer's shape
(HF-TrainingArguments-styled config + Trainer.train()) and field naming where
an equivalent concept exists; fields with no GRPOConfig equivalent (d_model,
boundary_temperature, default_boundary_prior, lambda_boundary) keep their own
descriptive names.
"""

import dataclasses
import math
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from common.bytes_utils import bytes_to_tensor, spans_from_boundaries
from common.eval.cross_tokenizer import (
    eval_wandb_log_dict,
    evaluate_on_groups,
    report_eval,
    sample_eval_groups,
)
from common.training.lr_schedule import build_lr_scheduler
from common.eval.metrics import compression_rate, gini_coefficient, renyi_efficiency
from common.data.oldi_data import LANG_SCRIPT
from common.eval.reporting import collapse_stats
from common.vocab import top_k_by_frequency

from .inference import save_checkpoint
from ..base import BaseTokenizerConfig, BaseTokenizerTrainer
from .model import MagnetModel
from .segment import induce_spans


def lang_to_script(lang):
    """Group languages by SCRIPT (not language) for the per-script boundary
    predictor -- e.g. arz_Arab and kas_Arab share one predictor/target rate.
    common.data.oldi_data.LANG_SCRIPT maps lang -> lang_Script; the part
    after the underscore is the ISO 15924 script code (Arab, Latn, Beng, Nkoo).

    Synthetic placeholder "languages" (e.g. "high_resource") aren't real
    codes and carry no script metadata, so each falls back to being its own
    one-language "script" bucket -- exercises the same code path trivially,
    which is fine since the smoke test only validates the mechanism."""
    if lang in LANG_SCRIPT:
        return LANG_SCRIPT[lang].split("_")[-1]
    return lang


def eval_lang_to_script(lang):
    """Like lang_to_script, but also resolves full lang_Script stems (e.g.
    "arz_Arab") -- what BOUQuET's langs="all" mode returns, vs. training
    data's plain codes. Without this, BOUQuET language keys fail the
    LANG_SCRIPT lookup and fall through unchanged, never matching a real
    script key in model.boundary_predictors, silently producing an empty
    eval dict and skipping held-out eval entirely (confirmed live: a 25-step
    test run's epoch-boundary eval fired 0 of an expected 3 times before
    this fix).

    Falls back to lang_to_script's own whole-string convention if `lang` has
    no underscore. Can't collide with the synthetic-profile fallback (e.g.
    "high_resource") in practice, since eval_groups only ever comes from
    real BOUQuET data or is None -- noted here as the reason this is a
    separate function rather than a change to lang_to_script itself."""
    if lang in LANG_SCRIPT:
        return lang_to_script(lang)
    if "_" in lang:
        return lang.rsplit("_", 1)[-1]
    return lang


@dataclasses.dataclass
class MagnetConfig(BaseTokenizerConfig):
    """vocab_size/output_dir/use_wandb/wandb_project/run_name inherited from
    BaseTokenizerConfig unchanged except wandb_project's default. See module
    docstring for the naming-convention rationale on everything else."""

    max_steps: int = 0  # 0 -> derive from num_train_epochs * steps_per_epoch (see
    # MagnetTrainer.train); matches GRPOConfig's convention. Set explicitly to
    # override with a raw step count (bypasses epoch semantics; run_smoke_test
    # does this).
    num_train_epochs: float = 3.0  # only if max_steps == 0. "epoch" = steps_per_
    # epoch steps, a periodic-checkpoint interval, NOT a full traversal guarantee
    # -- groups are sampled with replacement each step (unlike a shuffled
    # DataLoader), so "5 epochs" just means 5 * steps_per_epoch steps.
    per_device_train_batch_size: int = 8  # counts parallel-sentence GROUPS (like
    # GRPOConfig); expands to one (lang, byte_seq) item per language per group.
    learning_rate: float = 3e-3
    seed: int = 0

    # --- model architecture (deliberately small, same order of magnitude as
    # fairtok.policy.BytePolicy's hidden_size=64-128/num_layers=2-3; see
    # model.py "deliberate simplifications") ---
    d_model: int = 64
    n_heads: int = 4
    n_pre_layers: int = 2
    n_shortened_layers: int = 1
    n_post_layers: int = 1
    boundary_temperature: float = 0.5  # Gumbel-sigmoid temperature -- lower =
    # closer to discrete Bernoulli (sharper, higher-variance gradient), higher =
    # smoother/easier early optimization. Fixed, not annealed (see model.py).

    # --- boundary-rate loss (see model.py module docstring point on Loss) ---
    default_boundary_prior: float = 0.3  # target P(byte is boundary), i.e. an
    # expected segment length of 1/0.3 ~= 3.3 bytes/token -- roughly a real BPE
    # tokenizer's ballpark. A flat, hand-set hyperparameter rather than a
    # per-script measurement off a plain-BPE anchor (contrast
    # fairtok.train._plain_bpe_target_rate) -- an allowed simplification.
    per_script_boundary_prior: dict = dataclasses.field(
        default_factory=dict
    )  # optional {script: prior} overrides -- e.g. denser scripts (Beng, Arab:
    # 2-3 bytes/char vs. Latn's 1) may warrant a lower rate for comparable
    # character-level compression. Empty falls back to default_boundary_prior.
    lambda_boundary: float = 1.0  # weight of the boundary-rate loss vs. next-byte
    # CE. NOT literally "both per-position-normalized means" -- lm_loss divides
    # by n_valid (a true per-position mean), but boundary_loss is a per-example
    # Binomial NLL (over each sequence's real, unpadded length) averaged across
    # the batch, which doesn't scale with sequence length the same way a
    # per-token quantity does. This matches the reference implementation's own
    # padding-aware calc_loss_without_padding exactly (confirmed against
    # github.com/orevaahia/magnet-tokenization/src/magnet.py) -- the reference
    # repo's OTHER variant, calc_loss, does divide by preds.size(-1), but that
    # one assumes no padding at all and isn't the relevant comparison here,
    # since this project's own training loop (unlike that variant) explicitly
    # handles variable-length padded batches. 1.0 is a starting point to tune
    # empirically, not a derived "equal weighting" point. Raise if boundary
    # rate isn't tracking the prior; lower if it dominates.

    # vocab_size (inherited, default 384): final vocab budget, applied once
    # after training by keeping the most frequent distinct byte spans
    # (common.vocab.top_k_by_frequency).
    device: str = ""  # "" auto-detects cuda if available, else cpu.
    log_every: int = 10
    # output_dir (inherited, default ""): empty string to skip; else a path
    # model.state_dict() is saved to.
    # use_wandb (inherited, default False) -- see MagnetTrainer.train for the
    # actual wandb.init/run.log calls.
    wandb_project: str = "magnet"
    # run_name (inherited, default "").

    max_eval_samples: int = 20  # cap on BOUQuET dev groups scored per
    # epoch-boundary eval (0 = score all). Kept small since this runs
    # periodically during training, unlike evaluate.py's one-time full scoring.

    warmup_ratio: float = 0.1  # matches HF Trainer's own field name/default -- see
    # common.training.lr_schedule.build_lr_scheduler.
    lr_scheduler_type: str = "linear"  # "constant" (warmup only), "linear", or
    # "cosine" -- see common.training.lr_schedule.build_lr_scheduler. "linear" matches HF
    # Trainer's own default.


class MagnetTrainer(BaseTokenizerTrainer):
    """Construct with args + train_groups (list of {lang: text} dicts, see
    common.data.cli_data.load_groups). Call .train(), then read .model/
    .token_freq/.vocab off the instance (train() also returns these plus loss
    and boundary-rate traces as a tuple)."""

    def __init__(self, args: MagnetConfig, train_groups, eval_groups=None):
        super().__init__(args, train_groups, eval_groups)
        self.device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    def train(self):
        cfg = self.args
        device = self.device
        torch.manual_seed(cfg.seed)
        rng = np.random.default_rng(cfg.seed)
        print(f"device={device}")

        # Scripts must be known before constructing the model: boundary_predictors
        # is an nn.ModuleDict keyed by script, and a key added later would create
        # parameters the optimizer below never sees.
        all_langs = sorted({lang for group in self.train_groups for lang in group})
        # Precomputed once: lang_to_script is a pure function of `lang`, but is
        # otherwise called once per (group, lang) pair every step -- wasteful
        # since the mapping never changes.
        lang_script = {lang: lang_to_script(lang) for lang in all_langs}
        scripts = sorted(set(lang_script.values()))
        print(f"languages={all_langs}")
        print(f"scripts={scripts}")

        model = MagnetModel(
            scripts=scripts,
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            n_pre_layers=cfg.n_pre_layers,
            n_shortened_layers=cfg.n_shortened_layers,
            n_post_layers=cfg.n_post_layers,
            boundary_temperature=cfg.boundary_temperature,
        ).to(device)
        self.model = model
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

        priors = {
            script: cfg.per_script_boundary_prior.get(
                script, cfg.default_boundary_prior
            )
            for script in scripts
        }
        print(f"boundary priors={priors}")

        # "Epoch" isn't a real traversal (groups are sampled with replacement) --
        # steps_per_epoch is just the periodic-checkpoint interval for the dev eval.
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
        scheduler = build_lr_scheduler(
            optimizer, total_steps, cfg.warmup_ratio, cfg.lr_scheduler_type
        )

        # Built once against the live `model` -- the closure captures the object
        # reference, so it keeps seeing current weights without rebuilding at
        # each epoch boundary. Languages whose script this model never saw are
        # excluded (same policy as evaluate.py). Uses eval_lang_to_script, NOT
        # lang_to_script -- eval_groups' keys are BOUQuET "all"-mode stems (e.g.
        # "arz_Arab"), not the plain codes lang_to_script expects.
        eval_induce_fn_by_lang = None
        if self.eval_groups:
            eval_langs = {lang for group in self.eval_groups for lang in group}
            eval_induce_fn_by_lang = {
                lang: (
                    lambda raw, m=model, s=eval_lang_to_script(lang), d=device: induce_spans(
                        m, raw, s, d
                    )
                )
                for lang in eval_langs
                if eval_lang_to_script(lang) in model.boundary_predictors
            }

        run = None
        if cfg.use_wandb:
            import wandb

            run = wandb.init(
                project=cfg.wandb_project,
                name=cfg.run_name or None,
                config={
                    **dataclasses.asdict(cfg),
                    "scripts": scripts,
                    "boundary_priors": priors,
                    "num_train_groups": len(self.train_groups),
                    "num_eval_groups": len(self.eval_groups) if self.eval_groups else 0,
                    "steps_per_epoch": steps_per_epoch,
                    "total_steps": total_steps,
                },
            )

        token_freq = defaultdict(Counter)
        loss_trace = []
        boundary_rate_trace = []

        pbar = tqdm(range(total_steps), desc="training", unit="step")
        for step in pbar:
            # Random-with-replacement group sampling per step, rather than
            # GRPOTrainer's shuffled-epoch DataLoader -- simpler, since MAGNET
            # needs no group-relative baseline; means "epoch" isn't a meaningful
            # concept here.
            group_idx = rng.integers(
                0, len(self.train_groups), size=cfg.per_device_train_batch_size
            )
            batch_groups = [self.train_groups[i] for i in group_idx]

            # Flatten (lang, text) pairs and bucket by script -- pre/shortened/
            # post stages are shared across scripts; only
            # boundary_predictors[script] differs. All buckets' losses sum into
            # one optimizer step (multi-task accumulation).
            items_by_script = defaultdict(list)
            for group in batch_groups:
                for lang, text in group.items():
                    items_by_script[lang_script[lang]].append(
                        (lang, bytes_to_tensor(text, device))
                    )

            optimizer.zero_grad()
            total_lm_loss = torch.zeros((), device=device)
            total_boundary_loss = torch.zeros((), device=device)
            n_subbatches = 0
            total_real_bytes = 0
            total_boundary_count = 0.0

            for script, items in items_by_script.items():
                langs = [lang for lang, _ in items]
                seqs = [seq for _, seq in items]
                lengths = torch.tensor([s.shape[0] for s in seqs], device=device)
                B = len(seqs)
                T = int(lengths.max().item())
                byte_ids = torch.zeros(B, T, dtype=torch.long, device=device)
                for b, s in enumerate(seqs):
                    byte_ids[b, : s.shape[0]] = s

                logits, boundary_probs, hard_boundaries, pad_mask = model(
                    byte_ids, lengths, script, sample=True
                )

                # Next-byte CE, masked to real (non-pad) positions with an
                # actual next byte (excludes each sequence's last position) --
                # standard shift-by-one LM setup, per script bucket since each
                # has its own padding shape.
                valid = ~pad_mask
                has_next = valid.clone()
                has_next[torch.arange(B, device=device), lengths - 1] = False
                shift_logits = logits[:, :-1, :]
                shift_targets = byte_ids[:, 1:]
                shift_mask = has_next[:, :-1]
                ce = F.cross_entropy(
                    shift_logits.reshape(-1, shift_logits.size(-1)),
                    shift_targets.reshape(-1),
                    reduction="none",
                )
                ce = ce.view(B, T - 1) * shift_mask.float()
                n_valid = shift_mask.sum().clamp_min(1)
                lm_loss = ce.sum() / n_valid

                # Boundary-rate loss: NLL of the observed boundary count under
                # Binomial(real_length, prior) -- computed per real (unpadded)
                # length so padding can't skew the apparent rate.
                prior = priors[script]
                boundary_count = hard_boundaries.sum(dim=1)
                total_count = lengths.to(hard_boundaries.dtype)
                binomial = torch.distributions.Binomial(
                    total_count=total_count,
                    probs=torch.tensor(
                        prior, device=device, dtype=hard_boundaries.dtype
                    ),
                )
                boundary_loss = -binomial.log_prob(boundary_count).mean()

                total_lm_loss = total_lm_loss + lm_loss
                total_boundary_loss = total_boundary_loss + boundary_loss
                n_subbatches += 1
                total_real_bytes += int(lengths.sum().item())
                total_boundary_count += float(boundary_count.sum().item())

                # Harvest the induced vocabulary from this step's hard boundary
                # decisions -- same running-frequency-table role as
                # GRPOTrainer's token_freq.
                for b, (lang, seq) in enumerate(zip(langs, seqs)):
                    L = int(lengths[b].item())
                    boundaries = [
                        int(v) for v in hard_boundaries[b, :L].round().tolist()
                    ]
                    token_freq[lang].update(spans_from_boundaries(seq, boundaries))

            lm_loss_avg = total_lm_loss / n_subbatches
            boundary_loss_avg = total_boundary_loss / n_subbatches
            loss = lm_loss_avg + cfg.lambda_boundary * boundary_loss_avg
            loss.backward()
            optimizer.step()
            scheduler.step()

            boundary_rate = (
                total_boundary_count / total_real_bytes if total_real_bytes else 0.0
            )
            loss_trace.append(float(loss.item()))
            boundary_rate_trace.append(boundary_rate)

            postfix = {
                "loss": f"{loss.item():.3f}",
                "lm": f"{lm_loss_avg.item():.3f}",
                "bnd_loss": f"{boundary_loss_avg.item():.3f}",
                "rate": f"{boundary_rate:.3f}",
            }
            pbar.set_postfix(postfix)
            if step % cfg.log_every == 0:
                pbar.write(
                    f"[step {step:4d}] "
                    + " ".join(f"{k}={v}" for k, v in postfix.items())
                )

            if run is not None:
                run.log(
                    {
                        "train/loss": loss.item(),
                        "train/lm_loss": lm_loss_avg.item(),
                        "train/boundary_loss": boundary_loss_avg.item(),
                        "train/boundary_rate": boundary_rate,
                        "train/learning_rate": scheduler.get_last_lr()[0],
                    },
                    step=step,
                )

            # Epoch-boundary held-out eval against the current (still-training)
            # model -- BOUQuET dev, capped by max_eval_samples. Model only stays
            # in eval mode for the duration of each induce_spans call (see
            # segment.py).
            if eval_induce_fn_by_lang and step % steps_per_epoch == 0:
                eval_sample = sample_eval_groups(
                    self.eval_groups, cfg.max_eval_samples, seed=cfg.seed
                )
                eval_results = evaluate_on_groups(eval_induce_fn_by_lang, eval_sample)
                report_eval(eval_results, label=f"magnet step {step} dev")
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
            # config + scripts alongside state_dict -- load_checkpoint needs
            # both to reconstruct a MagnetModel (scripts sizes the
            # boundary_predictors ModuleDict).
            save_checkpoint(model, cfg, scripts, cfg.output_dir)
            print(f"saved model checkpoint to {cfg.output_dir}")

        return model, token_freq, final_vocab, loss_trace, boundary_rate_trace


def run_smoke_test():
    """Small trial run on synthetic placeholder data (fast, no network) that
    checks (1) no crash, (2) loss trends down, (3) boundary rate hasn't
    collapsed to ~0% (never cuts) or ~100% (cuts every byte, no compression).

    Also prints compression_rate/renyi_efficiency/gini_coefficient (unmodified
    common.eval.metrics) on the induced vocabulary, confirming MAGNET's
    {lang: Counter(span->count)} output is consumable by the fairtok metrics
    pipeline with zero adapter code.
    """
    from common.data.synthetic import LANG_PROFILES, make_synthetic_parallel_groups

    args = MagnetConfig(
        max_steps=80, per_device_train_batch_size=6, vocab_size=256, log_every=10
    )
    langs = list(LANG_PROFILES)
    train_groups = make_synthetic_parallel_groups(200, langs=langs, seed=args.seed)

    trainer = MagnetTrainer(args, train_groups)
    model, token_freq, final_vocab, loss_trace, boundary_rate_trace = trainer.train()

    first_10 = float(np.mean(loss_trace[:10]))
    last_10 = float(np.mean(loss_trace[-10:]))
    final_rate = float(np.mean(boundary_rate_trace[-10:]))
    print(
        f"\nloss: first-10-avg={first_10:.4f}  last-10-avg={last_10:.4f}  (decreased={last_10 < first_10})"
    )
    print(
        f"boundary rate (last-10-avg)={final_rate:.4f}  (collapsed={final_rate < 0.01 or final_rate > 0.99})"
    )
    print(f"final vocab size={len(final_vocab)}")

    print(
        "\nper-language metrics on the induced vocabulary (common.eval.metrics, unmodified):"
    )
    lang_renyi = {}
    for lang, counter in sorted(token_freq.items()):
        num_bytes = sum(len(span) * n for span, n in counter.items())
        num_tokens = sum(counter.values())
        cr = compression_rate(num_bytes, num_tokens)
        renyi = renyi_efficiency(list(counter.values()))
        lang_renyi[lang] = renyi
        print(
            f"  {lang:16s} compression_rate={cr:6.3f} bytes/token  renyi_efficiency={renyi:.4f}  "
            f"distinct_spans={len(counter)}"
        )
    gini = gini_coefficient(list(lang_renyi.values()))
    print(f"gini_coefficient(renyi across languages)={gini:.4f}")

    return model, token_freq, final_vocab, loss_trace, boundary_rate_trace


if __name__ == "__main__":
    run_smoke_test()
