"""Fitting loop for SuperBPEModel -- deliberately NOT a gradient-descent
trainer: unlike every other package in this repo, SuperBPE has no forward
pass, no optimizer, no per-step loss. "Training" here means fitting the
two-stage BPE merge table (superbpe.model.fit_superbpe) once over the whole
corpus, in a single call. This module's job is to hold the same
Config/Trainer/.train() shape every other tokenizer's cli.py/jobs script
expects -- so SuperBPE can be a drop-in sixth entry in train.py's dispatcher,
jobs/, etc. -- around that fundamentally different fitting procedure.

Consequences of that difference, all deliberate and documented rather than
faked to look like the neural baselines:
  - no learning_rate/optimizer/num_train_epochs/max_steps/per_device_train_
    batch_size fields -- none of them apply to a single-shot corpus-statistics
    fit.
  - no PERIODIC epoch-boundary eval: there is exactly one "boundary" (the end
    of fitting), so eval_groups (if given) get scored exactly once, right
    after fit_superbpe returns -- not on a recurring schedule the way every
    step-based trainer's own eval_induce_fn_by_lang check is.
  - no seed field: given the same train_groups, fit_superbpe is fully
    deterministic (tie-breaking is a fixed rule -- see
    superbpe.model._fit_merges), so there is nothing a seed would vary.
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
    transition_fraction: float = 0.8  # see superbpe.model.fit_superbpe's own
    # docstring -- JUDGMENT CALL, not a value taken from the paper's single
    # reported best configuration (the paper sweeps this, doesn't fix one).
    max_eval_samples: int = 0  # 0 scores every loaded eval group (see
    # common.eval.cross_tokenizer.sample_eval_groups) -- unlike the neural trainers'
    # periodic in-training checks (small by default since those repeat every
    # epoch), this fires exactly ONCE, so there is no repeated-cost reason to
    # subsample by default; still available as a knob if a very broad
    # --langs all eval set makes even a single pass slow.
    verbose: bool = True  # print merge-fitting progress (see
    # superbpe.model.fit_superbpe's log_every) -- real --vocab-size runs can
    # need tens of thousands of merges, and this is the only progress signal
    # available at all (no per-step loss/accuracy to watch, unlike the neural
    # baselines).


class SuperBPETrainer(BaseTokenizerTrainer):
    """Construct with args + train_groups (list of dicts {lang: text}, same
    shape every other tokenizer's trainer takes), call .train(), then read
    .model / .token_freq / .vocab off the instance (train() also returns them
    as a tuple, matching every other trainer's convention). No extra __init__
    needed beyond BaseTokenizerTrainer's own -- SuperBPE needs no device
    resolution or other per-system setup."""

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

        # Harvest realized token frequencies by re-running the just-fit model
        # over its own training corpus -- same "count whatever the model
        # actually produced, apply the fixed vocab budget only at the end"
        # philosophy as every other trainer in this repo (see common.vocab's
        # module docstring), even though BPE's own merge table already IS a
        # frequency-driven vocabulary; this keeps the harvesting step
        # (and therefore the final vocab_with_stats/save_vocab_json output
        # shape) identical across all six tokenizers for a fair comparison.
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
    """Feed the smoke test's induced vocabulary into common.eval.metrics UNMODIFIED,
    same as every other tokenizer's own smoke test -- confirms SuperBPE's
    output is a drop-in match for the rest of this project's evaluation
    pipeline."""
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
    """Mirrors every other tokenizer's run_smoke_test role: a small trial run
    on synthetic placeholder data, gated by two explicit assertions:
    (1) no crash getting here at all -- fitting + harvesting ran end to end
    over a real multilingual corpus; (2) merges were actually learned (the
    vocabulary grew past the 256-byte base alphabet) -- there is no
    loss-decreased check the way the gradient-based trainers have, since
    there is no loss here at all.

    Expect noticeably WORSE compression numbers here than the neural
    baselines' own smoke tests report on this exact same corpus (confirmed:
    ~1.0-2.1 here vs. ~3.9 for e.g. FANTA) -- not a bug. common.data.synthetic's
    generator gives each SENTENCE its own freshly-randomized 3-byte repeated
    chunk (see make_synthetic_parallel_groups/_gen_sentence), so repetition
    is per-sequence, not corpus-wide. A neural predictor can adapt to
    whatever a given sequence happens to repeat; classical BPE only ever
    learns a handful of FIXED, globally-frequent merge rules from counting
    across the whole corpus, so it has essentially nothing to exploit here.
    Real natural language has strong global frequency structure (common
    words/morphemes recur across many sentences), which is exactly BPE's
    intended regime and famous strength -- this smoke test's synthetic data
    just isn't built to showcase that. Correctness (priority order,
    lossless roundtrip, genuine cross-word "superword" merges) is verified
    separately, against real repeated-word text, not this corpus -- see the
    module's own development notes.
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
