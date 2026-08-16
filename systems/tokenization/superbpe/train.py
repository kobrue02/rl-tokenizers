"""Fitting loop for SuperBPEModel -- NOT a gradient-descent trainer: SuperBPE
has no forward pass, optimizer, or per-step loss. "Training" means fitting
the two-stage BPE merge table (superbpe.model.fit_superbpe) once over the
whole corpus in a single call. This module just holds the same
Config/Trainer/.train() shape every other tokenizer's cli.py/jobs script
expects, around that different fitting procedure.

Consequences, all deliberate:
  - no learning_rate/optimizer/num_train_epochs/max_steps/batch_size fields
    -- none apply to a single-shot corpus-statistics fit.
  - no periodic epoch-boundary eval: eval_groups (if given) get scored
    exactly once, right after fit_superbpe returns.
  - no seed field: fit_superbpe is fully deterministic given the same
    train_groups (fixed tiebreak rule, see superbpe.model._fit_merges).
"""

import dataclasses
from collections import Counter, defaultdict

from common.data.synthetic import LANG_PROFILES, make_synthetic_parallel_groups
from common.eval.cross_tokenizer import (
    eval_wandb_log_dict,
    evaluate_on_groups,
    report_eval,
    sample_eval_groups,
)
from common.eval.metrics import compression_rate, gini_coefficient, renyi_efficiency
from common.eval.reporting import collapse_stats
from common.vocab import top_k_by_frequency

from ..base import BaseTokenizerConfig, BaseTokenizerTrainer
from .model import fit_superbpe
from .segment import induce_spans


@dataclasses.dataclass
class SuperBPEConfig(BaseTokenizerConfig):
    """vocab_size/output_dir/use_wandb/wandb_project/run_name inherited from
    BaseTokenizerConfig unchanged except wandb_project's default -- see that
    class's own docstring."""

    wandb_project: str = "superbpe"
    # JUDGMENT CALL (see fit_superbpe docstring), not the paper's reported best config.
    transition_fraction: float = 0.8
    # 0 scores every loaded eval group (sample_eval_groups) -- fires exactly ONCE
    # (unlike neural trainers' periodic checks), so no default reason to subsample;
    # still a knob for a very broad --langs all eval set.
    max_eval_samples: int = 0
    # print merge-fitting progress (fit_superbpe's log_every) -- the only progress
    # signal available at all, since there's no per-step loss/accuracy here.
    verbose: bool = True


class SuperBPETrainer(BaseTokenizerTrainer):
    """Construct with args + train_groups (list of dicts {lang: text}, same
    shape as every other tokenizer's trainer), call .train(), then read
    .model / .token_freq / .vocab off the instance (also returned as a
    tuple). No extra __init__ needed -- SuperBPE needs no device setup."""

    def train(self):
        cfg = self.args
        sentences = [text for group in self.train_groups for text in group.values()]
        print(
            f"corpus={len(self.train_groups)} groups -> {len(sentences)} flattened sentences"
        )

        run = None
        if cfg.use_wandb:
            import wandb

            run = wandb.init(
                project=cfg.wandb_project,
                name=cfg.run_name or None,
                config={
                    **dataclasses.asdict(cfg),
                    "num_train_groups": len(self.train_groups),
                    "num_sentences": len(sentences),
                    "num_eval_groups": len(self.eval_groups) if self.eval_groups else 0,
                },
            )

        model = fit_superbpe(
            sentences, cfg.vocab_size, cfg.transition_fraction, verbose=cfg.verbose
        )
        self.model = model
        print(
            f"learned {len(model.merges)} merges "
            f"({model.num_stage1_merges} stage1 within-word, "
            f"{len(model.merges) - model.num_stage1_merges} stage2 superword)"
        )

        # Harvest realized token frequencies by re-running the fit model over its own
        # training corpus -- same "count what the model produced, apply the vocab
        # budget at the end" philosophy as every other trainer, keeping output shape
        # identical across tokenizers for fair comparison.
        token_freq = defaultdict(Counter)
        for group in self.train_groups:
            for lang, text in group.items():
                token_freq[lang].update(induce_spans(model, text))

        final_vocab = top_k_by_frequency(token_freq, cfg.vocab_size)
        self.token_freq = token_freq
        self.vocab = final_vocab

        if cfg.output_dir:
            import torch

            torch.save(
                {
                    "config": dataclasses.asdict(cfg),
                    "merges": model.merges,
                    "id_to_bytes": model.id_to_bytes,
                    "num_stage1_merges": model.num_stage1_merges,
                },
                cfg.output_dir,
            )
            print(f"saved model checkpoint to {cfg.output_dir}")

        if self.eval_groups:
            eval_sample = sample_eval_groups(self.eval_groups, cfg.max_eval_samples, seed=0)
            induce_fn_by_lang = {
                lang: (lambda raw, m=model: induce_spans(m, raw))
                for lang in {lang for group in eval_sample for lang in group}
            }
            eval_results = evaluate_on_groups(induce_fn_by_lang, eval_sample)
            report_eval(eval_results, label="superbpe (post-fit)")
            if run is not None:
                run.log(eval_wandb_log_dict(eval_results))

        if run is not None:
            avg_span_len, final_vocab_size = collapse_stats(token_freq, final_vocab)
            run.log(
                {
                    "final/vocab_size": final_vocab_size,
                    "final/avg_span_length_bytes": avg_span_len,
                    "final/char_collapse": int(avg_span_len < 1.2),
                    "final/sentence_collapse": int(avg_span_len > 40),
                    "final/num_merges": len(model.merges),
                    "final/num_stage1_merges": model.num_stage1_merges,
                }
            )
            run.finish()

        return model, token_freq, final_vocab


def _report_smoke_test_metrics(model, token_freq, final_vocab):
    """Feed the smoke test's induced vocabulary into common.eval.metrics
    unmodified, same as every other tokenizer's smoke test -- confirms
    SuperBPE's output is a drop-in match for the eval pipeline."""
    avg_span_len = sum(len(s) * n for c in token_freq.values() for s, n in c.items()) / max(
        1, sum(sum(c.values()) for c in token_freq.values())
    )
    print(
        f"\nfinal vocab size={len(final_vocab)}  avg span length={avg_span_len:.2f} bytes  "
        f"merges learned={len(model.merges)} "
        f"({model.num_stage1_merges} stage1, {len(model.merges) - model.num_stage1_merges} stage2)"
    )

    print("\nper-language metrics on the induced vocabulary:")
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

    return {"avg_span_len": avg_span_len, "gini": gini}


def run_smoke_test():
    """Small trial run on synthetic placeholder data, gated by two
    assertions: (1) no crash end to end, (2) merges were actually learned
    (vocab grew past the 256-byte base alphabet) -- no loss-decreased check
    since BPE has no loss.

    Expect noticeably WORSE compression here than neural baselines' smoke
    tests on the same corpus (~1.0-2.1 vs. ~3.9 for FANTA) -- not a bug.
    common.data.synthetic gives each sentence its own randomized repeated
    chunk, so repetition is per-sequence, not corpus-wide; a neural
    predictor can adapt per-sequence, but classical BPE only learns fixed,
    globally-frequent merges from corpus-wide counting, so it has little to
    exploit here. Real language has strong global frequency structure
    (BPE's actual strength) -- this synthetic data just doesn't show it.
    Correctness (priority order, lossless roundtrip, real superword merges)
    is verified separately against real repeated-word text.
    """
    args = SuperBPEConfig(vocab_size=320, verbose=False)
    train_groups = make_synthetic_parallel_groups(200, langs=list(LANG_PROFILES), seed=0)
    trainer = SuperBPETrainer(args, train_groups)
    model, token_freq, final_vocab = trainer.train()
    _report_smoke_test_metrics(model, token_freq, final_vocab)

    assert len(model.merges) > 0, "no merges learned at all"
    print("\nSuperBPE smoke test passed.")
    return model, token_freq, final_vocab


if __name__ == "__main__":
    run_smoke_test()
