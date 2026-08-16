"""Fitting loop for a standard byte-level BPE tokenizer (bpe.model.fit_bpe,
wrapping HuggingFace's `tokenizers` library directly). Same non-gradient-
descent shape as superbpe/train.py, for the same reasons (no
learning_rate/optimizer/num_train_epochs/max_steps fields, no periodic
epoch-boundary eval, no seed field -- a single-shot corpus-statistics fit).
"""

import dataclasses
from collections import Counter, defaultdict

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
from .model import fit_bpe
from .segment import induce_spans


@dataclasses.dataclass
class BPEConfig(BaseTokenizerConfig):
    """vocab_size/use_wandb/run_name inherited from BaseTokenizerConfig
    unchanged except wandb_project's default. output_dir's MEANING differs
    here: passed straight to tokenizers.Tokenizer.save() -- a self-describing
    JSON file, not a torch.save dict like other systems' checkpoints."""

    wandb_project: str = "bpe"
    # 0 scores every loaded eval group (see SuperBPEConfig's same field) -- fires once, not periodically.
    max_eval_samples: int = 0


class BPETrainer(BaseTokenizerTrainer):
    """Construct with args + train_groups (list of dicts {lang: text}, same
    shape as every other tokenizer's trainer), call .train(), then read
    .model / .token_freq / .vocab off the instance (also returned as a
    tuple). No extra __init__ needed."""

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

        model = fit_bpe(sentences, cfg.vocab_size)
        self.model = model
        print(f"learned vocabulary of {model.num_parameters()} tokens")

        # Harvest realized token frequencies by re-running the fit model over its own
        # training corpus -- see SuperBPETrainer.train's identical step (pipeline
        # consistency across tokenizers).
        token_freq = defaultdict(Counter)
        for group in self.train_groups:
            for lang, text in group.items():
                token_freq[lang].update(induce_spans(model, text))

        final_vocab = top_k_by_frequency(token_freq, cfg.vocab_size)
        self.token_freq = token_freq
        self.vocab = final_vocab

        if cfg.output_dir:
            model.tokenizer.save(cfg.output_dir)
            print(f"saved model checkpoint to {cfg.output_dir}")

        if self.eval_groups:
            eval_sample = sample_eval_groups(self.eval_groups, cfg.max_eval_samples, seed=0)
            induce_fn_by_lang = {
                lang: (lambda raw, m=model: induce_spans(m, raw))
                for lang in {lang for group in eval_sample for lang in group}
            }
            eval_results = evaluate_on_groups(induce_fn_by_lang, eval_sample)
            report_eval(eval_results, label="bpe (post-fit)")
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
                    "final/num_merges": model.num_parameters(),
                }
            )
            run.finish()

        return model, token_freq, final_vocab


def _report_smoke_test_metrics(model, token_freq, final_vocab):
    """Feed the smoke test's induced vocabulary into common.eval.metrics
    unmodified, same as every other tokenizer's smoke test -- confirms this
    baseline's output is a drop-in match for the eval pipeline."""
    avg_span_len = sum(len(s) * n for c in token_freq.values() for s, n in c.items()) / max(
        1, sum(sum(c.values()) for c in token_freq.values())
    )
    print(
        f"\nfinal vocab size={len(final_vocab)}  avg span length={avg_span_len:.2f} bytes  "
        f"learned vocabulary size={model.num_parameters()}"
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


# Small hand-written multilingual corpus, NOT common.data.synthetic's
# make_synthetic_parallel_groups (every other package's smoke-test corpus)
# -- that generator can produce invalid UTF-8, which tokenizers' str-based
# API can't accept without lossy replacement (see bpe.model._to_str).
# Real (if tiny) text in several scripts, so per-language diagnostics still
# have something real to measure.
_SMOKE_TEST_GROUPS = [
    {"eng": "The quick brown fox jumps over the lazy dog.", "deu": "Der schnelle braune Fuchs springt über den faulen Hund."},
    {"eng": "She sells seashells by the seashore.", "deu": "Sie verkauft Muscheln am Meeresufer."},
    {"eng": "A journey of a thousand miles begins with a single step.", "deu": "Eine Reise von tausend Meilen beginnt mit einem einzigen Schritt."},
    {"eng": "Knowledge is power, and power corrupts.", "deu": "Wissen ist Macht, und Macht korrumpiert."},
    {"eng": "The early bird catches the worm.", "deu": "Der frühe Vogel fängt den Wurm."},
] * 20


def run_smoke_test():
    """Small trial run, gated by two assertions: (1) no crash end to end,
    (2) a vocabulary was actually learned beyond the base alphabet -- no
    loss-decreased check since BPE has no loss.

    Uses a small real-text corpus (_SMOKE_TEST_GROUPS above), not
    common.data.synthetic's byte generator that other packages' smoke tests
    reuse -- that generator isn't guaranteed valid UTF-8, and this package
    leans on tokenizers' str-based API, which needs genuinely valid text.
    """
    args = BPEConfig(vocab_size=320)
    trainer = BPETrainer(args, _SMOKE_TEST_GROUPS)
    model, token_freq, final_vocab = trainer.train()
    _report_smoke_test_metrics(model, token_freq, final_vocab)

    assert model.num_parameters() > 256, "no merges learned beyond the base byte alphabet"
    print("\nBPE smoke test passed.")
    return model, token_freq, final_vocab


if __name__ == "__main__":
    run_smoke_test()
