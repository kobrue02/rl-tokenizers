"""MAGNET training loop -- plain backprop, no RL machinery at all (unlike
fairtok.train.GRPOTrainer, whose boundary policy is discrete/REINFORCE-trained;
MAGNET's boundary predictor is differentiable end-to-end via the Gumbel-sigmoid
+ straight-through trick in model.py, so there's no advantage/return/baseline
bookkeeping to do here -- just a next-byte CE loss and a boundary-rate loss,
summed and backpropagated like any ordinary supervised loss).

MagnetConfig/MagnetTrainer mirror fairtok.train.GRPOConfig/GRPOTrainer's shape
(a HF-TrainingArguments-styled config dataclass + a Trainer class with
.train()) and field naming wherever an equivalent concept exists
(max_steps, learning_rate, per_device_train_batch_size, seed, vocab_size,
device, output_dir) -- fields with no GRPOConfig equivalent (d_model,
boundary_temperature, default_boundary_prior, lambda_boundary) are MAGNET's own
architecture/loss hyperparameters and keep descriptive names instead of
force-fitting HF naming that doesn't apply.
"""

import dataclasses
import math
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from common.bytes_utils import bytes_to_tensor, spans_from_boundaries
from common.eval_common import (
    eval_wandb_log_dict,
    evaluate_on_groups,
    report_eval,
    sample_eval_groups,
)
from common.metrics import compression_rate, gini_coefficient, renyi_efficiency
from common.oldi_data import LANG_SCRIPT
from common.reporting import collapse_stats
from common.vocab import top_k_by_frequency

from .inference import save_checkpoint
from .model import MagnetModel
from .segment import induce_spans


def lang_to_script(lang):
    """Group languages by SCRIPT, not by language, for the per-script boundary
    predictor -- e.g. arz_Arab and kas_Arab share ONE predictor and one target
    boundary rate (see model.py's MagnetModel/BoundaryPredictor).
    common.oldi_data.LANG_SCRIPT gives the real lang_Script code (e.g.
    "arz_Arab" -> the part after the underscore is the ISO 15924 script code:
    Arab, Latn, Beng, Nkoo cover this project's 9-language panel.

    Synthetic placeholder "languages" (common.data.make_synthetic_parallel_groups's
    profile names, e.g. "high_resource") aren't real language codes and carry no
    script metadata, so each synthetic profile falls back to being its OWN
    one-language "script" bucket. This still exercises the exact same
    per-script-predictor code path end to end; it's simply a trivial
    (1 language : 1 script) mapping for that placeholder data, which is fine
    since the smoke test's job is to validate the mechanism, not to reproduce
    the real script-sharing structure only real language metadata has."""
    if lang in LANG_SCRIPT:
        return LANG_SCRIPT[lang].split("_")[-1]
    return lang


def eval_lang_to_script(lang):
    """Like lang_to_script, but ALSO resolves language keys that are already
    full lang_Script stems (e.g. "arz_Arab", "aar_Latn") -- exactly what
    common.oldi_data.load_bouquet_dev/load_bouquet_test's langs="all" mode
    returns (see common.cli_data.load_bouquet_dev_for_training, which always
    uses "all"), unlike training data's plain codes (which lang_to_script's own
    LANG_SCRIPT-lookup path already handles). Without this, every BOUQuET
    "all"-mode language silently fails lang_to_script's LANG_SCRIPT lookup,
    falls through to returning the language key UNCHANGED (the whole stem,
    e.g. "arz_Arab"), which then never matches any real script key in
    model.boundary_predictors -- producing an EMPTY (falsy) eval closure dict
    and silently skipping periodic/held-out evaluation entirely. Confirmed as
    a real, not just theoretical, bug: an epoch-boundary eval that should have
    fired 3 times in a 25-step test run fired zero times before this fix.

    Falls back to lang_to_script's own convention (the whole string as its own
    one-off script bucket) if `lang` has no underscore at all -- this never
    actually collides with lang_to_script's SYNTHETIC-placeholder-profile
    fallback (e.g. "high_resource", which DOES contain an underscore but isn't
    a real stem): eval_groups only ever comes from real BOUQuET data or is
    None (see load_bouquet_dev_for_training's synthetic skip), never from
    common.data's synthetic profiles, so that collision risk can't occur in
    practice -- but is called out here since it's the reason this is a
    separate function rather than a change to lang_to_script itself."""
    if lang in LANG_SCRIPT:
        return lang_to_script(lang)
    if "_" in lang:
        return lang.rsplit("_", 1)[-1]
    return lang


@dataclasses.dataclass
class MagnetConfig:
    """See module docstring for the naming-convention rationale."""

    max_steps: int = 100
    per_device_train_batch_size: int = 8  # counts parallel-sentence GROUPS (like
    # GRPOConfig), which then expand to one flattened (lang, byte_seq) item per
    # language in each sampled group -- see MagnetTrainer.train.
    learning_rate: float = 3e-3
    seed: int = 0

    # --- model architecture (kept deliberately small -- see model.py's module
    # docstring "deliberate simplifications" section for the full rationale;
    # these defaults land in the same order of magnitude as
    # fairtok.policy.BytePolicy's own hidden_size=64-128, num_layers=2-3) ---
    d_model: int = 64
    n_heads: int = 4
    n_pre_layers: int = 2
    n_shortened_layers: int = 1
    n_post_layers: int = 1
    boundary_temperature: float = 0.5  # Gumbel-sigmoid relaxation temperature --
    # lower = closer to a true discrete Bernoulli (sharper, higher-variance
    # gradient); higher = smoother relaxation, easier optimization early on. Fixed
    # here rather than annealed (see model.py's "no temperature annealing"
    # simplification).

    # --- boundary-rate loss (see model.py module docstring point on Loss) ---
    default_boundary_prior: float = 0.3  # target P(byte is a boundary), i.e. an
    # expected segment length of 1/0.3 ~= 3.3 bytes/token -- in the ballpark a
    # real BPE tokenizer lands in for Latin-script text. This is deliberately a
    # flat, hand-set hyperparameter rather than a per-script measurement off a
    # plain-BPE anchor (contrast fairtok.train._plain_bpe_target_rate) -- the
    # task spec explicitly allows this simplification ("simpler -- just expose
    # it as a configurable per-script hyperparameter").
    per_script_boundary_prior: dict = dataclasses.field(
        default_factory=dict
    )  # optional
    # {script: prior} overrides for default_boundary_prior -- e.g. a script with
    # denser byte-per-character encoding (e.g. Beng, Arab under UTF-8, which use
    # 2-3 bytes per character vs Latn's 1) might warrant a lower boundary rate
    # (larger byte-segments) to reach a comparable CHARACTER-level compression
    # rate. Left empty (falls back to default_boundary_prior everywhere) unless
    # a caller has a specific reason to differentiate.
    lambda_boundary: float = 1.0  # weight on the boundary-rate loss relative to
    # the next-byte CE loss -- both losses are already per-position-normalized
    # means (see train() below), so a starting point of 1.0 (equal weighting) is
    # a reasonable default; raise it if boundary rate isn't tracking the prior
    # closely enough, lower it if it's dominating and hurting the LM loss.

    vocab_size: int = 384  # final harvested-vocabulary budget, same role as
    # GRPOConfig.vocab_size -- applied once, after training, by keeping the
    # vocab_size most frequent distinct byte spans (see common.vocab.top_k_by_frequency).
    device: str = ""  # "" auto-detects cuda if available, else cpu.
    log_every: int = 10
    output_dir: str = (
        ""  # empty string to skip; else a path model.state_dict() is saved to.
    )
    use_wandb: bool = False  # matches fairtok.train.GRPOConfig's field of the same
    # name/role -- see MagnetTrainer.train for the actual wandb.init/run.log calls.
    wandb_project: str = "magnet"
    run_name: str = ""

    max_eval_samples: int = 20  # cap on how many BOUQuET dev groups get scored at
    # each epoch-boundary evaluation (see MagnetTrainer.train) -- 0 scores every
    # loaded dev group. Kept small by default since this runs periodically DURING
    # training, not once at the end (see evaluate.py, which always scores
    # everything -- that one-time cost is fine; paying it every epoch isn't).


class MagnetTrainer:
    """Construct with args + train_groups (a plain list of dicts {lang: text},
    the same shape common.oldi_data.load_all_training_groups /
    common.data.make_synthetic_parallel_groups both return), call .train(),
    then read .model / .token_freq / .vocab off the instance (train() also
    returns them, plus a per-step loss trace and boundary-rate trace, as a
    tuple for convenience -- see run_smoke_test below for the shape)."""

    def __init__(self, args: MagnetConfig, train_groups, eval_groups=None):
        self.args = args
        self.train_groups = train_groups
        self.eval_groups = eval_groups  # BOUQuET dev, or None to skip periodic
        # epoch-boundary evaluation entirely (see common.cli_data.load_bouquet_dev_for_training)
        self.device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.token_freq = None
        self.vocab = None

    def train(self):
        cfg = self.args
        device = self.device
        torch.manual_seed(cfg.seed)
        rng = np.random.default_rng(cfg.seed)
        print(f"device={device}")

        # The set of scripts must be known BEFORE constructing the model, since
        # MagnetModel.boundary_predictors is an nn.ModuleDict keyed by script --
        # adding a key after construction would create parameters the optimizer
        # below never sees.
        all_langs = sorted({lang for group in self.train_groups for lang in group})
        # Precomputed once: lang_to_script(lang) is a pure function of `lang` alone
        # (a dict lookup + string split), but the loop below calls it once per
        # (group, lang) pair on EVERY training step -- for a fixed, small set of
        # languages that's the same lookup repeated thousands of times over a real
        # run, for no reason, since the mapping never changes after this point.
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

        # "Epoch" isn't a real traversal here either (see train loop's own comment
        # on random-with-replacement sampling) -- steps_per_epoch is just used as a
        # periodic-checkpoint INTERVAL (one epoch's worth of steps) for the BOUQuET
        # dev eval below, not a guarantee every group is visited exactly once.
        steps_per_epoch = max(
            1, math.ceil(len(self.train_groups) / cfg.per_device_train_batch_size)
        )
        print(f"steps_per_epoch={steps_per_epoch} (periodic dev-eval interval)")

        # Built ONCE against the live `model` object -- a closure over `model`
        # keeps seeing its CURRENT weights on every call (Python closures capture
        # the object reference, not a snapshot), so this doesn't need rebuilding
        # each time the epoch boundary is hit. Languages whose script this model
        # never saw during training (eval_lang_to_script(lang) not in
        # model.boundary_predictors) are excluded here, same policy
        # magnet/evaluate.py already applies for its own post-hoc eval. Uses
        # eval_lang_to_script, NOT lang_to_script -- eval_groups' language keys
        # are BOUQuET "all"-mode stems (e.g. "arz_Arab"), not the plain codes
        # lang_to_script's LANG_SCRIPT lookup expects (see eval_lang_to_script's
        # own docstring for why this distinction is load-bearing, not cosmetic).
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
                },
            )

        token_freq = defaultdict(Counter)
        loss_trace = []
        boundary_rate_trace = []

        pbar = tqdm(range(cfg.max_steps), desc="training", unit="step")
        for step in pbar:
            # Plain random-with-replacement group sampling per step, rather than
            # GRPOTrainer's shuffled-epoch DataLoader machinery -- a deliberate
            # simplification for this baseline (no per-group language-rotation
            # state to maintain, since MAGNET doesn't need GRPO's group-relative
            # baseline at all); fine at smoke-test / baseline scale, but means
            # "epoch" isn't a meaningful concept here the way it is in GRPOConfig.
            group_idx = rng.integers(
                0, len(self.train_groups), size=cfg.per_device_train_batch_size
            )
            batch_groups = [self.train_groups[i] for i in group_idx]

            # Flatten every (lang, text) pair in the batch and bucket by script --
            # the pre/shortened/post transformer stages are SHARED nn.Module
            # instances reused across every script's forward call below; only
            # boundary_predictors[script] actually differs per bucket. All
            # buckets' losses are summed into ONE optimizer step (see below),
            # exactly like accumulating multiple task losses in a multi-task batch.
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

                # Next-byte CE, masked to real (b, t) pairs that both (a) aren't
                # padding and (b) actually have a next byte (i.e. t isn't that
                # sequence's last real position) -- standard shift-by-one LM
                # setup, applied per-script-bucket since each bucket has its own
                # (B, T) padding shape.
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

                # Boundary-rate loss: NLL of the observed per-sequence boundary
                # count under Binomial(real_length, prior) -- see model.py's
                # module docstring "Loss" section. Computed per real (unpadded)
                # length, so padding never inflates or deflates a sequence's
                # apparent boundary rate.
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

                # Harvest the induced vocabulary from THIS step's hard boundary
                # decisions -- same running-frequency-table role as
                # fairtok.train.GRPOTrainer's token_freq, and consumed the exact
                # same way by common.vocab.top_k_by_frequency at the end.
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
                    },
                    step=step,
                )

            # Epoch-boundary held-out eval against the CURRENT (still-training)
            # model -- BOUQuET dev, capped by max_eval_samples since this runs
            # periodically, unlike evaluate.py's one-time full scoring. Model is
            # left in eval mode only for the duration of the induce_spans calls
            # inside evaluate_on_groups (each call flips it back via
            # model.train(was_training) internally -- see magnet/segment.py).
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
            # config + scripts alongside state_dict, not state_dict alone -- see
            # magnet.inference.load_checkpoint, which needs both to reconstruct a
            # MagnetModel (scripts sizes its per-script boundary_predictors ModuleDict).
            save_checkpoint(model, cfg, scripts, cfg.output_dir)
            print(f"saved model checkpoint to {cfg.output_dir}")

        return model, token_freq, final_vocab, loss_trace, boundary_rate_trace


def run_smoke_test():
    """The plan's own gate, MAGNET-flavored: a small trial run on
    common.data's synthetic placeholder corpus (fast, no network access
    needed -- see common.data's module docstring for why this stands in for
    real OLDI/FLORES+/SMOL data) that checks (1) no crash, (2) the loss trends
    down, (3) the boundary rate hasn't collapsed to ~0% (never cuts -- the
    hierarchical bottleneck becomes a no-op, see model.py's null_segment
    discussion) or ~100% (cuts every byte -- degenerates to character-level,
    no compression at all).

    Also prints compression_rate / renyi_efficiency / gini_coefficient (all
    from common.metrics, completely unmodified) on the induced per-language
    vocabulary, as the sanity check that MAGNET's {lang: Counter(span->count)}
    output shape is consumable by the rest of the fairtok metrics pipeline with
    zero adapter code -- the same shape fairtok.train.GRPOTrainer.train()
    itself produces.
    """
    from common.data import LANG_PROFILES, make_synthetic_parallel_groups

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
        "\nper-language metrics on the induced vocabulary (common.metrics, unmodified):"
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
