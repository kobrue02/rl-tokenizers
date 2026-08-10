"""FANTA training loop: MantaModel's architecture (unchanged, see fanta/model.py),
trained with next-byte cross-entropy PLUS two added terms: a differentiable
Gini-coefficient penalty over each language's mean compression rate within a batch
(fanta.model.fairness_loss), and a per-language rate ANCHOR
(fanta.model.rate_anchor_loss) pulling each language toward its own target rate.

The anchor term exists because the Gini term alone has a degenerate solution:
every language compressing equally BADLY also scores Gini~=0. Confirmed
empirically (an early FANTA run with no anchor collapsed mean_compression_rate to
~1.0 -- pure character-level segmentation -- within 10 steps, "satisfying"
fairness by refusing to compress at all), not just anticipated as a theoretical
risk. Per-language targets are derived the same way fairtok.train's own
target_rate_by_lang is: common.parity.compute_lang_parity_ratios, motivated by
"Compute Optimal Tokenization" (Limisiewicz et al. 2026) finding that the
compute-optimal compression rate is language-dependent, not one global constant.

Batching is GROUP-based (one parallel {lang: text} group -> up to group_sample_size
language sequences per step), NOT manta.train.MantaTrainer's flat individual-sentence
sampling -- both loss terms need several languages' compression rates in the SAME
forward pass to compare/anchor, and vanilla MANTa's flat sampling gives no guarantee
(or even likelihood) of that. This mirrors flexitokens.train.FlexiTokensTrainer's own
batching convention (per_device_train_batch_size counts GROUPS, group_sample_size
caps languages per group), which already solved the same problem for its own
per-language hinge loss -- see FantaConfig's docstring for the resulting semantic
difference from MantaConfig's field of the same name.
"""

import dataclasses
from collections import Counter, defaultdict

import numpy as np
import torch
from tqdm.auto import tqdm

from common.bytes_utils import bytes_to_tensor, spans_from_boundaries
from common.data import make_synthetic_parallel_groups
from common.metrics import compression_rate, gini_coefficient, renyi_efficiency
from common.parity import compute_lang_parity_ratios
from common.reporting import avg_span_length, collapse_stats, report_collapse
from common.vocab import top_k_by_frequency

from .model import MantaModel, fairness_loss, next_byte_loss, rate_anchor_loss
from .segment import boundaries_from_assignment


@dataclasses.dataclass
class FantaConfig:
    """Mirrors manta.train.MantaConfig's naming wherever a field means the SAME
    thing; see the module docstring for the one field whose MEANING deliberately
    changed (per_device_train_batch_size), and lambda_fair/group_sample_size below
    for FANTA's own additions.

    per_device_train_batch_size: int -- counts parallel-sentence GROUPS, same as
    fairtok.train.GRPOConfig/flexitokens.train.FlexiTokensConfig's field of this
    name -- NOT individual byte sequences, unlike manta.train.MantaConfig's own
    field of the same name. See module docstring for why: a Gini fairness penalty
    needs multiple languages' compression rates in ONE forward pass.
    """

    max_steps: int = 100
    per_device_train_batch_size: int = 8  # counts GROUPS -- see class docstring.
    group_sample_size: int = 24  # cap languages rolled out per group per step,
    # regardless of how many a group actually offers -- same meaning as
    # fairtok.train.GRPOConfig/flexitokens.train.FlexiTokensConfig's own field.
    learning_rate: float = 3e-3
    seed: int = 0
    vocab_size: int = 384  # final vocab budget, same role as MantaConfig's field.

    lambda_fair: float = 1.0  # weight on the Gini fairness penalty relative to the
    # next-byte CE loss -- see fanta.model.fairness_loss. Distinct from fairtok's
    # own lambda_fair (which shapes a REINFORCE reward, not a direct loss term) --
    # same name because it plays the same conceptual role (fairness-term weight),
    # not because the mechanism is the same.

    anchor_lang: str = "eng"  # pivot language for per-language target-rate scaling
    # (see common.parity.compute_lang_parity_ratios) -- matches
    # fairtok.train.GRPOConfig/flexitokens.train.FlexiTokensConfig's own field.
    target_rate_anchor: float = 4.0  # target compression rate (bytes/token) for the
    # ANCHOR language specifically; every other language's target is this scaled by
    # its own byte-length parity ratio vs. the anchor (see rate_anchor_loss). Not
    # derived from a real BPE baseline the way fairtok.train._plain_bpe_target_rate
    # is -- 4.0 is a plausible byte-level-BPE-ish compression rate (same judgment
    # call flexitokens.train.FlexiTokensConfig.alpha_anchor makes, just expressed
    # directly as a rate rather than a boundary probability).
    target_rate_floor: float = 1.0  # a target_rate below 1 byte/token is nonsensical
    # (can't compress below the raw byte stream); guards a degenerate parity ratio.
    target_rate_ceiling: float = 64.0  # generous upper bound; guards the opposite
    # degenerate case (e.g. a near-empty sample making a parity ratio blow up).
    lambda_rate: float = 2.0  # weight on the rate-anchor penalty -- matches
    # fairtok.train.GRPOConfig.lambda_target's own default (2.0, adapted from
    # Dauncey & Wattenhofer's consistency_loss_weight): the empirical collapse
    # this term exists to prevent (see module docstring) happened FAST (within 10
    # steps) and needs real weight from the start of training, not a token gesture.

    dim: int = 64
    window: int = 8
    num_frontier_layers: int = 2
    num_frontier_heads: int = 4
    block_hidden_size: int = 64
    num_block_layers: int = 1
    max_extra_sigma: float = 3.0
    max_grad_norm: float = 1.0

    device: str = ""  # "" auto-detects cuda if available, else cpu.
    log_steps: int = 10
    output_dir: str = ""  # "" disables checkpoint saving.
    save_steps: int = 0  # 0 disables periodic saving.

    use_wandb: bool = False
    wandb_project: str = "fanta"
    run_name: str = ""


def _pad_batch(tensors, device):
    lengths = torch.tensor(
        [t.shape[0] for t in tensors], dtype=torch.long, device=device
    )
    T = int(lengths.max().item())
    padded = torch.zeros(len(tensors), T, dtype=torch.long, device=device)
    for i, t in enumerate(tensors):
        padded[i, : t.shape[0]] = t
    return padded, lengths


class FantaTrainer:
    """Construct with args + train_groups (a plain list of dicts {lang: text}, the
    same shape every other tokenizer's trainer in this repo takes), call .train(),
    then read .model / .token_freq / .vocab off the instance (train() also returns
    them, plus loss/fairness-loss traces, as a tuple for convenience)."""

    def __init__(self, args: FantaConfig, train_groups):
        self.args = args
        self.train_groups = train_groups
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

        model = MantaModel(
            dim=cfg.dim,
            window=cfg.window,
            num_frontier_layers=cfg.num_frontier_layers,
            num_frontier_heads=cfg.num_frontier_heads,
            block_hidden_size=cfg.block_hidden_size,
            num_block_layers=cfg.num_block_layers,
            max_extra_sigma=cfg.max_extra_sigma,
        ).to(device)
        self.model = model
        print(f"model parameters: {model.num_parameters():,}")
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

        # Per-language rate anchor -- see module docstring. Derived ONCE, up front,
        # from the real training corpus (not re-derived per step: the parity ratio
        # between two languages' typical byte lengths doesn't change during
        # training).
        parity_ratio_by_lang, parity_anchor = compute_lang_parity_ratios(
            self.train_groups, cfg.anchor_lang
        )
        if parity_anchor != cfg.anchor_lang:
            print(
                f"[fanta] anchor language {cfg.anchor_lang!r} not present in this corpus; "
                f"falling back to {parity_anchor!r} for per-language target-rate scaling"
            )
        target_rate_by_lang = {
            lang: max(
                cfg.target_rate_floor,
                min(cfg.target_rate_ceiling, cfg.target_rate_anchor * ratio),
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
                    "target_rate_by_lang": target_rate_by_lang,
                    "num_train_groups": len(self.train_groups),
                    "model_parameters": model.num_parameters(),
                },
            )

        token_freq = defaultdict(Counter)
        loss_trace = []
        fairness_loss_trace = []
        n_groups = len(self.train_groups)

        pbar = tqdm(range(cfg.max_steps), desc="training", unit="step")
        postfix = {}
        for step in pbar:
            batch_size = min(cfg.per_device_train_batch_size, n_groups)
            group_idx = rng.choice(n_groups, size=batch_size, replace=False)
            batch_groups = [self.train_groups[i] for i in group_idx]

            batch_items = []  # (lang, byte_tensor)
            for group in batch_groups:
                langs_in_group = list(group.keys())
                if cfg.group_sample_size and len(langs_in_group) > cfg.group_sample_size:
                    langs_in_group = list(
                        rng.choice(langs_in_group, size=cfg.group_sample_size, replace=False)
                    )
                for lang in langs_in_group:
                    batch_items.append((lang, bytes_to_tensor(group[lang], device)))

            langs = [lang for lang, _ in batch_items]
            tensors = [seq for _, seq in batch_items]
            padded, lengths = _pad_batch(tensors, device)

            optimizer.zero_grad()
            output = model(padded, lengths)
            ce_loss, num_valid, num_correct = next_byte_loss(padded, lengths, output.logits)
            gini_loss, per_lang_rate = fairness_loss(langs, lengths, output)
            anchor_loss = rate_anchor_loss(per_lang_rate, target_rate_by_lang)
            loss = ce_loss + cfg.lambda_fair * gini_loss + cfg.lambda_rate * anchor_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()

            # Harvest vocabulary from this step's REALIZED (hardened) boundaries --
            # same "count whatever the model actually produced along the way, apply
            # the fixed vocab budget only at the end" philosophy as every other
            # trainer in this repo. Reuses the assignment matrix this step's
            # forward pass already computed -- see
            # manta.segment.boundaries_from_assignment's docstring.
            boundaries = boundaries_from_assignment(output.assignment.detach(), lengths)
            for (lang, _), seq, actions in zip(batch_items, tensors, boundaries):
                token_freq[lang].update(spans_from_boundaries(seq, actions))

            loss_value = loss.item()
            gini_value = gini_loss.item()
            anchor_value = anchor_loss.item()
            loss_trace.append(loss_value)
            fairness_loss_trace.append(gini_value)
            byte_accuracy = num_correct / num_valid if num_valid else 0.0

            postfix["loss"] = f"{loss_value:.3f}"
            postfix["ce"] = f"{ce_loss.item():.3f}"
            postfix["gini"] = f"{gini_value:.4f}"
            postfix["anchor"] = f"{anchor_value:.4f}"
            postfix["acc"] = f"{byte_accuracy:.3f}"
            pbar.set_postfix(postfix)

            if (
                cfg.output_dir
                and cfg.save_steps
                and step > 0
                and step % cfg.save_steps == 0
            ):
                torch.save(
                    {"config": dataclasses.asdict(cfg), "state_dict": model.state_dict()},
                    cfg.output_dir,
                )

            if run is not None:
                run.log(
                    {
                        "train/loss": loss_value,
                        "train/ce_loss": ce_loss.item(),
                        "train/fairness_loss_gini": gini_value,
                        "train/rate_anchor_loss": anchor_value,
                        "train/byte_accuracy": byte_accuracy,
                        "train/mean_compression_rate": float(
                            np.mean([v.item() for v in per_lang_rate.values()])
                        ),
                        "train/num_langs_this_step": len(per_lang_rate),
                    },
                    step=step,
                )

            if step % cfg.log_steps == 0:
                avg_span_len = avg_span_length(token_freq)
                collapse_warn = ""
                if avg_span_len and avg_span_len < 1.2:
                    collapse_warn = " <-- CHECK: drifting toward single-byte spans"
                elif avg_span_len > 40:
                    collapse_warn = " <-- CHECK: drifting toward whole-sequence spans"
                pbar.write(
                    f"[step {step:4d}] loss={loss_value:.4f} ce={ce_loss.item():.4f} "
                    f"gini={gini_value:.4f} anchor={anchor_value:.4f} acc={byte_accuracy:.3f} "
                    f"avg_span={avg_span_len:.2f} langs_this_step={len(per_lang_rate)}"
                    f"{collapse_warn}"
                )
                if run is not None:
                    run.log(
                        {
                            f"fairness/rate/{lang}": v.item()
                            for lang, v in per_lang_rate.items()
                        },
                        step=step,
                    )

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
        self.fairness_loss_trace = fairness_loss_trace

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
        return model, token_freq, final_vocab, loss_trace, fairness_loss_trace


def _report_smoke_test_metrics(token_freq, final_vocab, loss_trace, fairness_loss_trace):
    """Feed the smoke test's induced vocabulary into common.metrics/common.reporting
    UNMODIFIED, same as every other tokenizer's own smoke test does -- confirms
    FANTA's induced vocabulary is a drop-in match for the rest of this project's
    evaluation pipeline."""
    report_collapse(token_freq, final_vocab)

    loss_head = sum(loss_trace[: min(5, len(loss_trace))]) / min(5, len(loss_trace))
    loss_tail = sum(loss_trace[-min(5, len(loss_trace)) :]) / min(5, len(loss_trace))
    print(
        f"\nloss: first-5-step avg={loss_head:.4f}  last-5-step avg={loss_tail:.4f}"
    )
    gini_head = sum(fairness_loss_trace[: min(5, len(fairness_loss_trace))]) / min(
        5, len(fairness_loss_trace)
    )
    gini_tail = sum(fairness_loss_trace[-min(5, len(fairness_loss_trace)) :]) / min(
        5, len(fairness_loss_trace)
    )
    print(
        f"fairness (gini) loss: first-5-step avg={gini_head:.4f}  last-5-step avg={gini_tail:.4f}"
    )

    print("\nper-language metrics on the induced (discretized) vocabulary:")
    per_lang_renyi = {}
    for lang, counter in sorted(token_freq.items()):
        num_bytes = sum(len(span) * n for span, n in counter.items())
        num_tokens = sum(counter.values())
        rate = compression_rate(num_bytes, num_tokens)
        eff = renyi_efficiency(list(counter.values()))
        per_lang_renyi[lang] = eff
        print(f"  {lang:16s} compression_rate={rate:6.3f}  renyi_efficiency={eff:.4f}")
    gini = gini_coefficient(list(per_lang_renyi.values())) if per_lang_renyi else 0.0
    print(f"\ngini coefficient across per-language renyi efficiency: {gini:.4f}")

    return {"loss_head": loss_head, "loss_tail": loss_tail, "gini_head": gini_head, "gini_tail": gini_tail}


def run_smoke_test():
    """Mirrors manta.train.run_smoke_test's role: a small trial run on synthetic
    placeholder data, gated by two explicit assertions: (1) no crash getting here
    at all -- the group-based batching + Gini loss ran end to end over real,
    padded, variable-length, multi-language batches; (2) the CE loss actually
    decreased (mean of first 5 logged steps vs. last 5). Doesn't assert anything
    about the Gini loss itself trending down -- at this scale (4 synthetic
    placeholder "languages", 80 steps) there's too little signal for that to be a
    meaningful pass/fail gate; see the printed fairness-loss trace instead."""
    args = FantaConfig(
        max_steps=80, per_device_train_batch_size=8, group_sample_size=8, vocab_size=384, log_steps=10
    )
    train_groups = make_synthetic_parallel_groups(200, seed=args.seed)
    trainer = FantaTrainer(args, train_groups)
    model, token_freq, final_vocab, loss_trace, fairness_loss_trace = trainer.train()
    metrics = _report_smoke_test_metrics(token_freq, final_vocab, loss_trace, fairness_loss_trace)

    assert loss_trace, "no steps recorded at all"
    assert metrics["loss_tail"] < metrics["loss_head"], (
        f"CE loss did not decrease: first-5 avg={metrics['loss_head']:.4f}, "
        f"last-5 avg={metrics['loss_tail']:.4f}"
    )
    print("\nFANTA smoke test passed.")
    return model, token_freq, final_vocab


if __name__ == "__main__":
    run_smoke_test()
