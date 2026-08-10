"""Shared, cross-tokenizer held-out evaluation.

Runs any trained tokenizer's own existing boundary/span-inducing entry point
(fairtok.policy.segment_bytes, magnet/flexitokens/manta.segment.induce_spans) over a
held-out parallel corpus (BOUQuET dev by default -- see common.oldi_data.
load_bouquet_dev, and fairtok/cli.py's _load_eval_groups docstring for why BOUQuET is
kept disjoint from every training --data-source) and reports the same fairness/
efficiency metrics common.metrics already defines, so all four tokenizers in this
repo can be compared on IDENTICAL held-out data with IDENTICAL scoring code.

Each tokenizer's *.evaluate module supplies an `induce_spans_fn_by_lang` dict (a
`bytes -> list[bytes] spans` callable per language, already bound to that
tokenizer's loaded checkpoint + whatever extra argument its own induce_spans needs,
e.g. magnet's per-script boundary predictor) -- this module stays completely
agnostic to how that callable was built, the same pattern common.stability already
uses for the boundary-stability diagnostic.
"""

from collections import Counter, defaultdict

import numpy as np

from common.metrics import compression_rate, fertility, gini_coefficient, renyi_efficiency
from common.reporting import word_count


def evaluate_on_groups(induce_spans_fn_by_lang, eval_groups):
    """induce_spans_fn_by_lang: dict[lang -> (bytes -> list[bytes] spans)] callable.
    eval_groups: list[dict[lang -> text]] (e.g. common.oldi_data.load_bouquet_dev()'s
    return value).

    Languages present in eval_groups but missing from induce_spans_fn_by_lang are
    silently skipped, not errored on -- a held-out comparison across independently
    trained checkpoints is exactly the situation where language coverage can
    legitimately differ (e.g. a checkpoint trained on fewer languages than BOUQuET
    covers).

    Returns {"token_freq": {lang: Counter}, "renyi": {lang: float}, "gini": float,
    "per_lang_compression": {lang: float}, "avg_compression": float,
    "fertility": {lang: float}}.
    """
    token_freq = defaultdict(Counter)
    compressions_by_lang = defaultdict(list)
    word_counts = defaultdict(int)

    for group in eval_groups:
        for lang, text in group.items():
            induce_fn = induce_spans_fn_by_lang.get(lang)
            if induce_fn is None:
                continue
            raw = text.encode("utf-8") if isinstance(text, str) else bytes(text)
            spans = induce_fn(raw)
            token_freq[lang].update(spans)
            compressions_by_lang[lang].append(compression_rate(len(raw), len(spans)))
            word_counts[lang] += word_count(text)

    renyi = {
        lang: renyi_efficiency(list(c.values())) for lang, c in token_freq.items() if c
    }
    gini = gini_coefficient(list(renyi.values())) if renyi else 0.0
    per_lang_compression = {
        lang: float(np.mean(vals)) for lang, vals in compressions_by_lang.items()
    }
    avg_compression = (
        float(np.mean(list(per_lang_compression.values())))
        if per_lang_compression
        else 0.0
    )
    fertility_by_lang = {
        lang: fertility(sum(counter.values()), word_counts.get(lang, 0))
        for lang, counter in token_freq.items()
    }
    return {
        "token_freq": token_freq,
        "renyi": renyi,
        "gini": gini,
        "per_lang_compression": per_lang_compression,
        "avg_compression": avg_compression,
        "fertility": fertility_by_lang,
    }


def report_eval(results, label=""):
    prefix = f"[{label}] " if label else ""
    print(f"\n{prefix}held-out evaluation:")
    print(f"  avg_compression={results['avg_compression']:.2f}  gini={results['gini']:.4f}")
    print("  per-language compression / renyi efficiency / fertility:")
    for lang in sorted(results["renyi"]):
        print(
            f"    {lang}: compression={results['per_lang_compression'].get(lang, 0.0):.2f}  "
            f"renyi={results['renyi'][lang]:.4f}  "
            f"fertility={results['fertility'].get(lang, 0.0):.2f}"
        )


def sample_eval_groups(eval_groups, max_eval_samples, seed=0):
    """Cap eval_groups to a random sample of max_eval_samples (without
    replacement), for periodic IN-TRAINING dev checks -- scoring hundreds of
    BOUQuET groups every epoch boundary in a long run is wasteful when it's just
    a training-time signal, not the final reported number (see evaluate.py's
    --eval-data-source bouquet_test, which always scores everything).
    max_eval_samples=0, or eval_groups already shorter than it, means score
    everything -- no sampling applied."""
    if not max_eval_samples or len(eval_groups) <= max_eval_samples:
        return eval_groups
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(eval_groups), size=max_eval_samples, replace=False)
    return [eval_groups[i] for i in idx]


def eval_wandb_log_dict(results, prefix="eval"):
    """Flatten an evaluate_on_groups result dict into a wandb.log-able dict,
    matching fairtok.train.GRPOTrainer's own eval/* naming convention for its
    (separately implemented, but equivalent-in-spirit) periodic held-out eval."""
    log_dict = {
        f"{prefix}/avg_compression": results["avg_compression"],
        f"{prefix}/gini": results["gini"],
    }
    log_dict.update(
        {f"{prefix}/renyi/{lang}": v for lang, v in results["renyi"].items()}
    )
    log_dict.update(
        {
            f"{prefix}/compression/{lang}": v
            for lang, v in results["per_lang_compression"].items()
        }
    )
    log_dict.update(
        {f"{prefix}/fertility/{lang}": v for lang, v in results["fertility"].items()}
    )
    return log_dict
