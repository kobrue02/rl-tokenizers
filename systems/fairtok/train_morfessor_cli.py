"""CLI: train per-language unsupervised Morfessor 2.0 models on monolingual text
pooled from Glot500 + MADLAD-400 (see morphology.py for why, and for the per-language
source list). These stand in for the gold morphological data (UniMorph/Universal
Dependencies) that MorphScore normally requires and that most of this project's
9-language panel doesn't have.

This is a separate, one-off preprocessing step -- not part of Phase 1 training. Run it
once (or whenever you want to refresh the models), then load the saved .bin files
wherever a MorphScore-style alignment check is computed.
"""

import argparse
import json
import time
from pathlib import Path

from common.oldi_data import LANGS

from .morphology import MORPH_SOURCES, collect_word_counts, train_morfessor


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Train per-language Morfessor 2.0 models for a MorphScore proxy."
    )
    parser.add_argument(
        "--langs",
        type=str,
        default=None,
        help=f"comma-separated language codes; defaults to the full 9-language panel ({','.join(LANGS)}). "
        "Languages with no configured source (kas, nqo) are skipped with a warning, not an error.",
    )
    parser.add_argument(
        "--max-words-per-source",
        type=int,
        default=2_000_000,
        help="per-language, per-SOURCE word budget -- e.g. a language with 2 sources can use up to 2x this "
        "total. Fetchers stop streaming shards as soon as this is hit (see morphology.py).",
    )
    parser.add_argument(
        "--max-word-types",
        type=int,
        default=300_000,
        help="cap on distinct word types fed into Morfessor after pooling all sources -- bounds training "
        "time for high-resource languages (eng/spa); rarely binds for the low-resource ones.",
    )
    parser.add_argument(
        "--freq-threshold",
        type=int,
        default=1,
        help="discard word types occurring fewer than this many times before training (crawl-noise/typo "
        "filtering) -- kept permissive (1 = keep everything) by default since low-resource languages "
        "can't afford to throw away scarce vocabulary; raise it for high-resource languages if their "
        "output looks noisy.",
    )
    parser.add_argument(
        "--corpusweight",
        type=float,
        default=1.0,
        help="Morfessor's corpus-cost weight -- HIGHER values make it MORE conservative (fewer splits), "
        "not more aggressive; counterintuitive, verified empirically (see morphology.py). Not yet "
        "properly tuned against a hand-checked dev set -- probe this if output looks off.",
    )
    parser.add_argument(
        "--init-rand-split",
        type=float,
        default=0.5,
        help="probability of an initial random split at each character position before training -- NOT "
        "cosmetic: without this, Morfessor's default recursive search converges to zero splits even "
        "on textbook cases (see morphology.py docstring on train_morfessor for why).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out-dir",
        type=str,
        default="morfessor_out",
        help="where to save per-language .bin models + stats",
    )
    parser.add_argument(
        "--preview-count",
        type=int,
        default=20,
        help="segment this many of the most frequent words per language for a quick human sanity-check; 0 to skip",
    )
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    langs = args.langs.split(",") if args.langs else list(LANGS)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    import morfessor

    io = morfessor.MorfessorIO()
    summary = {}

    for lang in langs:
        sources = MORPH_SOURCES.get(lang, [])
        if not sources:
            print(
                f"[{lang}] skipped -- no configured monolingual source (see morphology.MORPH_SOURCES)"
            )
            summary[lang] = {"status": "skipped_no_source"}
            continue

        print(
            f"[{lang}] fetching from {[s for s, _ in sources]} (budget {args.max_words_per_source}/source)..."
        )
        t0 = time.time()
        word_counts = collect_word_counts(
            lang,
            max_words_per_source=args.max_words_per_source,
            max_word_types=args.max_word_types,
        )
        fetch_s = time.time() - t0
        if not word_counts:
            print(
                f"[{lang}] skipped -- sources configured but yielded zero words (unexpected; check access)"
            )
            summary[lang] = {"status": "skipped_empty"}
            continue

        total_words = sum(word_counts.values())
        print(
            f"[{lang}] {total_words} words, {len(word_counts)} distinct types (fetch took {fetch_s:.1f}s) -- training..."
        )
        t0 = time.time()
        model = train_morfessor(
            word_counts,
            freq_threshold=args.freq_threshold,
            corpusweight=args.corpusweight,
            init_rand_split=args.init_rand_split,
            seed=args.seed,
        )
        train_s = time.time() - t0

        model_path = out_dir / f"{lang}.bin"
        io.write_binary_model_file(str(model_path), model)

        preview = []
        if args.preview_count:
            # most_common() alone is dominated by 1-2 character function words/elided
            # clitics (e.g. Ligurian "l'aegua" -> "l" + "aegua" at the apostrophe --
            # arguably a correct morphological split, but useless for a human
            # sanity-check of whether LONGER words are being segmented sensibly) --
            # so preview the most frequent words that are actually long enough to
            # potentially contain an interesting stem+affix split.
            candidates = [w for w in word_counts if len(w) >= 5]
            preview_words = sorted(candidates, key=lambda w: -word_counts[w])[
                : args.preview_count
            ]
            for word in preview_words:
                segments, logprob = model.viterbi_segment(word)
                preview.append(
                    {
                        "word": word,
                        "count": word_counts[word],
                        "segments": segments,
                        "logprob": logprob,
                    }
                )

        stats = {
            "status": "trained",
            "sources": [f"{s}:{c}" for s, c in sources],
            "total_words": total_words,
            "distinct_types": len(word_counts),
            "fetch_seconds": round(fetch_s, 1),
            "train_seconds": round(train_s, 1),
            "model_path": str(model_path),
            "preview": preview,
        }
        summary[lang] = stats
        with open(out_dir / f"{lang}_stats.json", "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        print(f"[{lang}] trained in {train_s:.1f}s, saved to {model_path}")

    print("\n=== summary ===")
    for lang, s in summary.items():
        if s["status"] == "trained":
            print(
                f"  {lang}: {s['total_words']} words, {s['distinct_types']} types -> {s['model_path']}"
            )
        else:
            print(f"  {lang}: {s['status']}")

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return summary


if __name__ == "__main__":
    main()
