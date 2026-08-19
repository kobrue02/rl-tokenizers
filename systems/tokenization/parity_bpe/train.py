"""Fitting loop for ParityBPEModel (parity_bpe.model.fit_parity_bpe) -- NOT a
gradient-descent trainer, same non-neural shape as superbpe/train.py and
bpe/train.py (no learning_rate/optimizer/num_train_epochs/max_steps fields,
no periodic epoch-boundary eval, no seed field: fit_parity_bpe is fully
deterministic given the same train_groups/eval_groups).

DEV SET IS REQUIRED, unlike every other tokenizer here (where eval_groups is
optional, used only for periodic/post-fit reporting): Parity-aware BPE's
core fairness mechanism needs a per-language compression-rate signal to pick
the worst-compressed language at every merge step (see model.py's module
docstring) -- there is no meaningful "just skip it" fallback the way there
is for e.g. bpe's optional post-fit eval_groups. self.eval_groups (BOUQuET
dev, via common.data.cli_data.load_bouquet_dev_for_training -- already
exactly the small multi-parallel corpus shape the paper recommends) is
reused directly as this parity dev-set, not just for post-fit reporting like
every other trainer's own eval_groups use.
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
from .model import fit_parity_bpe
from .segment import induce_spans


def _sentences_by_lang(groups):
    """[{lang: text}, ...] -> {lang: [text, ...]} -- fit_parity_bpe's own
    per-language input shape, unlike every other trainer here which flattens
    groups into one POOLED sentence list (losing per-language separation
    that parity-aware BPE specifically needs)."""
    out = defaultdict(list)
    for group in groups:
        for lang, text in group.items():
            out[lang].append(text)
    return dict(out)


@dataclasses.dataclass
class ParityBPEConfig(BaseTokenizerConfig):
    """vocab_size/output_dir/use_wandb/run_name inherited from
    BaseTokenizerConfig unchanged except wandb_project's default.
    output_dir's MEANING matches bpe's own (this project's other
    `tokenizers`-backed baseline): passed straight to
    tokenizers.Tokenizer.save() -- a self-describing JSON file, not a
    torch.save dict like the from-scratch systems' checkpoints.

    NO checkpoint_dir field, unlike this project's own from-scratch fitters
    (superbpe) -- reusing the official implementation directly (per explicit
    user instruction) means no mid-fit pause/resume hook exists to expose;
    see model.py's own module docstring for why and what a real timeout at
    this project's --vocab-size scale would mean in practice."""

    wandb_project: str = "parity_bpe"
    # See model.py's module docstring's VARIANTS section.
    num_global_merges: int = 0  # 0 = pure parity-aware from the start ("base");
    # >0 = "hybrid" (first N merges classical global-frequency).
    use_moving_window: bool = False  # "window" variant.
    window_size: int = 100  # paper's own default.
    alpha: float = 2.0  # paper's own default.
    min_frequency: int = 2  # official implementation's own CLI default --
    # stop this language's merge search once its best pair falls below this count.
    # 0 scores every loaded eval group (see bpe/superbpe's own same field) --
    # fires once (this is the POST-FIT report; the SAME eval_groups also
    # drove the fit itself, see module docstring), not periodically.
    max_eval_samples: int = 0


class ParityBPETrainer(BaseTokenizerTrainer):
    """Construct with args + train_groups + eval_groups (REQUIRED here, see
    module docstring -- unlike every other tokenizer's optional eval_groups).
    Call .train(), then read .model / .token_freq / .vocab off the instance
    (also returned as a tuple)."""

    def train(self):
        cfg = self.args
        if not self.eval_groups:
            raise ValueError(
                "ParityBPETrainer needs a non-empty dev set (eval_groups) -- parity-aware BPE's "
                "own fairness mechanism has no meaningful way to run without one (see "
                "parity_bpe.model's module docstring). Pass --data-source something other than "
                "'synthetic' (which has no real BOUQuET dev counterpart), or supply eval_groups "
                "directly if calling this trainer programmatically."
            )

        sentences_by_lang = _sentences_by_lang(self.train_groups)
        dev_sentences_by_lang = _sentences_by_lang(self.eval_groups)
        print(
            f"corpus={len(self.train_groups)} groups -> {sum(len(v) for v in sentences_by_lang.values())} "
            f"flattened sentences across {len(sentences_by_lang)} languages "
            f"(dev: {len(self.eval_groups)} groups -> {len(dev_sentences_by_lang)} languages)"
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
                    "num_eval_groups": len(self.eval_groups),
                    "num_train_langs": len(sentences_by_lang),
                },
            )

        model = fit_parity_bpe(
            sentences_by_lang, dev_sentences_by_lang, cfg.vocab_size,
            num_global_merges=cfg.num_global_merges,
            use_moving_window=cfg.use_moving_window,
            window_size=cfg.window_size, alpha=cfg.alpha,
            min_frequency=cfg.min_frequency,
            verbose=True,
        )
        self.model = model
        print(f"learned vocabulary of {model.num_parameters()} tokens")

        # Harvest realized token frequencies by re-running the fit model over its own
        # training corpus -- same "count what the model produced, apply the vocab
        # budget at the end" philosophy as every other trainer here.
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

        eval_sample = sample_eval_groups(self.eval_groups, cfg.max_eval_samples, seed=0)
        induce_fn_by_lang = {
            lang: (lambda raw, m=model: induce_spans(m, raw))
            for lang in {lang for group in eval_sample for lang in group}
        }
        eval_results = evaluate_on_groups(induce_fn_by_lang, eval_sample)
        report_eval(eval_results, label="parity_bpe (post-fit)")
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
    unmodified, same as every other tokenizer's smoke test -- confirms
    ParityBPE's output is a drop-in match for the eval pipeline."""
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
# make_synthetic_parallel_groups -- same reasoning as bpe/train.py's own
# _SMOKE_TEST_GROUPS (that generator can produce invalid UTF-8; real text in
# several scripts also gives per-language diagnostics something real to
# measure). Reused as BOTH train_groups and eval_groups (self-referential
# dev) -- a reasonable smoke-test simplification, since --data-source
# synthetic has no real BOUQuET dev counterpart at all (see
# ParityBPETrainer.train's own required-dev-set check).
_SMOKE_TEST_GROUPS = [
    {"eng": "The quick brown fox jumps over the lazy dog.", "deu": "Der schnelle braune Fuchs springt über den faulen Hund."},
    {"eng": "She sells seashells by the seashore.", "deu": "Sie verkauft Muscheln am Meeresufer."},
    {"eng": "A journey of a thousand miles begins with a single step.", "deu": "Eine Reise von tausend Meilen beginnt mit einem einzigen Schritt."},
    {"eng": "Knowledge is power, and power corrupts.", "deu": "Wissen ist Macht, und Macht korrumpiert."},
    {"eng": "The early bird catches the worm.", "deu": "Der frühe Vogel fängt den Wurm."},
] * 20


def run_smoke_test():
    """Small trial run on the hand-written corpus above, gated by two
    assertions: (1) no crash end to end, (2) merges were actually learned
    (vocab grew past the 256-byte base alphabet) -- no loss-decreased check
    since BPE has no loss."""
    args = ParityBPEConfig(vocab_size=320)
    trainer = ParityBPETrainer(args, _SMOKE_TEST_GROUPS, eval_groups=_SMOKE_TEST_GROUPS)
    model, token_freq, final_vocab = trainer.train()
    _report_smoke_test_metrics(model, token_freq, final_vocab)

    assert model.num_parameters() > 256, "no merges learned beyond the base byte alphabet"
    print("\nParity-aware BPE smoke test passed.")
    return model, token_freq, final_vocab


if __name__ == "__main__":
    run_smoke_test()
