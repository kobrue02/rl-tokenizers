"""Command-line entry point for pretraining.contamination: checks whether a
common.data.corpora source (the SAME registry pretraining.data_prep tokenizes)
shares n-gram text spans with a pretraining.benchmarks eval benchmark's own
examples -- see pretraining/contamination.py's own module docstring for the
detection approach, and pretraining/benchmarks.py's CONTAMINATION section
for why this check matters and what it does NOT cover (only checks the
sources you actually point it at; a "0% contaminated" result is a real,
scoped finding about that scan, not a general contamination guarantee).

Usage:
    python3 -m pretraining.cli_contamination \\
        --benchmark xnli --benchmark-langs en,de,fr \\
        --corpus-dataset fineweb_edu --corpus-dataset-config sample-10BT \\
        --max-corpus-docs 1000000 --output results/contamination_xnli_fineweb.json

    python3 -m pretraining.cli_contamination \\
        --benchmark flores_mt --benchmark-lang-pairs eng:spa,eng:arz \\
        --corpus-dataset glot500 --corpus-langs all \\
        --output results/contamination_flores_glot500.json

No SLURM job wraps this deliberately, same reasoning as pretraining.
cli_generate: this is a text-only scan (no tokenization, no GPU) that a
user runs on-demand to check ONE corpus/benchmark pair, not a queued stage
of the regular train->prep->pretrain->eval pipeline -- --max-corpus-docs
bounds a single interactive run; a genuinely full-corpus scan of a huge
source is better submitted as an ad-hoc job on whatever partition/time
limit that scale needs, which varies too much per source to hardcode here.
"""

import argparse
import json

from common.config_file import parse_args_with_config
from common.data.corpora import (
    ALL_SOURCES,
    BITEXT_SOURCES,
    FINEWEB_EDU_CONFIGS,
    MONOLINGUAL_SOURCES,
    OLMO_MIX_CONFIGS,
    stream_groups,
)

from . import benchmarks
from .contamination import build_benchmark_shingle_index, scan_texts_for_contamination, summarize_contamination

_MULTIPLE_CHOICE_TEXT_FIELDS = lambda ex: [ex.context] + list(ex.choices)
_TRANSLATION_TEXT_FIELDS = lambda ex: [ex.source_text, ex.reference_text]

_BENCHMARK_TEXT_FIELDS = {
    "xnli": _MULTIPLE_CHOICE_TEXT_FIELDS,
    "xcopa": _MULTIPLE_CHOICE_TEXT_FIELDS,
    "flores_mt": _TRANSLATION_TEXT_FIELDS,
}


def _corpus_text_iter(corpus_dataset, corpus_langs, corpus_dataset_config, max_corpus_docs):
    if corpus_dataset in ("fineweb_edu", "olmo_mix") or corpus_dataset in BITEXT_SOURCES:
        stream = stream_groups(corpus_dataset, config=corpus_dataset_config)
    else:
        stream = stream_groups(corpus_dataset, langs=corpus_langs)

    docs_yielded = 0
    for group in stream:
        for text in group.values():
            if not text:
                continue
            yield text
            docs_yielded += 1
            if max_corpus_docs and docs_yielded >= max_corpus_docs:
                return


def run_contamination_check(
    examples,
    text_fields_fn,
    corpus_dataset,
    corpus_langs=None,
    corpus_dataset_config=None,
    ngram_size=13,
    max_corpus_docs=None,
):
    """examples: list of pretraining.benchmarks example objects.
    text_fields_fn: see pretraining.contamination.build_benchmark_shingle_index.
    corpus_dataset/corpus_langs/corpus_dataset_config: same meaning as
    pretraining.data_prep's own --dataset/--langs/--dataset-config, since
    this streams from the exact same common.data.corpora.stream_groups. Returns
    pretraining.contamination.summarize_contamination's own dict shape."""
    examples = list(examples)
    index = build_benchmark_shingle_index(examples, text_fields_fn, n=ngram_size)
    text_iter = _corpus_text_iter(corpus_dataset, corpus_langs, corpus_dataset_config, max_corpus_docs)
    hits, docs_scanned = scan_texts_for_contamination(text_iter, index, n=ngram_size)
    return summarize_contamination(examples, hits, docs_scanned)


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Check n-gram text overlap between a pretraining corpus and an eval benchmark's own examples."
    )
    parser.add_argument("--benchmark", choices=sorted(_BENCHMARK_TEXT_FIELDS), required=True)
    parser.add_argument("--benchmark-langs", type=str, default=None, help="comma-separated, for xnli/xcopa")
    parser.add_argument(
        "--benchmark-lang-pairs", type=str, default=None,
        help="comma-separated src:tgt pairs (e.g. eng:spa,deu_Latn:fra_Latn), for flores_mt -- "
        "accepts a short code common.data.oldi_data.LANG_SCRIPT maps OR any of flores_plus's "
        "~227 native lang_Script stems directly, see benchmarks.py",
    )
    parser.add_argument("--benchmark-split", type=str, default=None, help="override the benchmark loader's own default split")
    parser.add_argument("--corpus-dataset", choices=ALL_SOURCES, required=True)
    parser.add_argument(
        "--corpus-langs", type=str, default=None,
        help="comma-separated language codes (or 'all' for glot500; an arbitrary subset for "
        "bible_nlp) -- ignored for fineweb_edu/olmo_mix and every BITEXT_SOURCES entry "
        "(ccmatrix/un_pc/europarl/tatoeba_mt), which use --corpus-dataset-config instead; "
        "see common.data.corpora",
    )
    parser.add_argument(
        "--corpus-dataset-config", type=str, default=None,
        help=f"HF config name for --corpus-dataset fineweb_edu (choices: {FINEWEB_EDU_CONFIGS}) or "
        f"olmo_mix (choices: {OLMO_MIX_CONFIGS}); a native pair name or 'all' for ccmatrix/un_pc/"
        f"europarl; a '{{split}}/{{pair-or-all}}' string or bare 'all' for tatoeba_mt -- see "
        f"common.data.corpora.list_bitext_configs/list_tatoeba_mt_pairs",
    )
    parser.add_argument("--ngram-size", type=int, default=13)
    parser.add_argument(
        "--max-corpus-docs", type=int, default=None,
        help="cap the corpus scan -- a result under a cap only covers that PREFIX of the corpus, "
        "see this module's own docstring",
    )
    parser.add_argument("--output", type=str, default=None, help="write JSON results here (default: print to stdout)")
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument(
        "--wandb-project", type=str, default="pretraining",
        help="SAME default project as every other pretraining.* wandb-logging entry point -- "
        "this run logs job_type='contamination_check'",
    )
    parser.add_argument("--run-name", type=str, default="")
    return parser


def main(argv=None):
    args = parse_args_with_config(build_arg_parser(), argv)

    if (
        args.corpus_dataset in MONOLINGUAL_SOURCES - {"glot500"} or args.corpus_dataset in BITEXT_SOURCES
    ) and args.corpus_langs:
        print(
            f"warning: --corpus-langs is ignored for --corpus-dataset {args.corpus_dataset} "
            "(selected via --corpus-dataset-config instead)"
        )
    corpus_langs = None
    if args.corpus_langs is not None:
        corpus_langs = "all" if args.corpus_langs == "all" else args.corpus_langs.split(",")

    if args.benchmark in ("xnli", "xcopa"):
        benchmark_langs = args.benchmark_langs.split(",") if args.benchmark_langs else None
        loader = benchmarks.load_xnli if args.benchmark == "xnli" else benchmarks.load_xcopa
        kwargs = {"langs": benchmark_langs}
        if args.benchmark_split:
            kwargs["split"] = args.benchmark_split
        examples = list(loader(**kwargs))
    else:
        if not args.benchmark_lang_pairs:
            raise ValueError("--benchmark flores_mt needs --benchmark-lang-pairs (e.g. eng:spa)")
        pairs = [tuple(p.split(":")) for p in args.benchmark_lang_pairs.split(",")]
        kwargs = {"lang_pairs": pairs}
        if args.benchmark_split:
            kwargs["split"] = args.benchmark_split
        examples = list(benchmarks.load_flores_mt(**kwargs))

    result = run_contamination_check(
        examples,
        _BENCHMARK_TEXT_FIELDS[args.benchmark],
        corpus_dataset=args.corpus_dataset,
        corpus_langs=corpus_langs,
        corpus_dataset_config=args.corpus_dataset_config,
        ngram_size=args.ngram_size,
        max_corpus_docs=args.max_corpus_docs,
    )
    result["benchmark"] = args.benchmark
    result["corpus_dataset"] = args.corpus_dataset

    print(f"benchmark={args.benchmark} corpus_dataset={args.corpus_dataset} ngram_size={args.ngram_size}")
    print(f"scanned {result['corpus_docs_scanned']:,} corpus documents")
    print(
        f"{result['num_contaminated']}/{result['num_examples']} benchmark examples "
        f"({result['contamination_rate']:.2%}) share at least one {args.ngram_size}-gram with the corpus"
    )
    if args.max_corpus_docs:
        print(
            f"NOTE: corpus scan capped at --max-corpus-docs={args.max_corpus_docs} -- this result "
            "only covers that prefix of the corpus, not a full-corpus guarantee (see module docstring)"
        )

    payload = json.dumps(result, indent=2)
    if args.output:
        with open(args.output, "w") as f:
            f.write(payload)
        print(f"wrote results to {args.output}")
    print(payload)

    if args.use_wandb:
        import wandb

        run = wandb.init(
            project=args.wandb_project,
            name=args.run_name or None,
            job_type="contamination_check",
            config={
                "benchmark": args.benchmark,
                "benchmark_langs": args.benchmark_langs,
                "benchmark_lang_pairs": args.benchmark_lang_pairs,
                "corpus_dataset": args.corpus_dataset,
                "corpus_langs": corpus_langs,
                "corpus_dataset_config": args.corpus_dataset_config,
                "ngram_size": args.ngram_size,
                "max_corpus_docs": args.max_corpus_docs,
            },
        )
        run.log(
            {
                "num_examples": result["num_examples"],
                "num_contaminated": result["num_contaminated"],
                "contamination_rate": result["contamination_rate"],
                "corpus_docs_scanned": result["corpus_docs_scanned"],
            }
        )
        run.finish()
        print(f"logged contamination check to wandb project={args.wandb_project!r}")


def run_smoke_test():
    """Verifies the overlap-detection mechanism end-to-end against tiny
    synthetic examples/corpus text (no real benchmark/HF network calls,
    unlike a real --corpus-dataset/--benchmark run) -- confirms a
    genuinely shared long span IS detected and an unrelated document is
    NOT flagged, at a small ngram_size so short synthetic sentences still
    produce meaningful shingles (see common/data/dedup.py's own docstring for
    why shingle_size interacts with document length this way)."""
    from .benchmarks import MultipleChoiceExample

    examples = [
        MultipleChoiceExample(
            lang="en",
            context="This is a very specific and distinctive sentence used only for testing",
            choices=[" alpha", " beta"],
            label=0,
        ),
        MultipleChoiceExample(
            lang="en",
            context="Something else entirely unrelated to anything else in this test",
            choices=[" gamma", " delta"],
            label=1,
        ),
    ]
    corpus_texts = [
        "Some unrelated preamble text. This is a very specific and distinctive sentence used only "
        "for testing. And some more text after it that has nothing to do with anything.",
        "A totally different, unrelated corpus document sharing no real vocabulary with either example.",
    ]

    index = build_benchmark_shingle_index(examples, _MULTIPLE_CHOICE_TEXT_FIELDS, n=8)
    hits, docs_scanned = scan_texts_for_contamination(iter(corpus_texts), index, n=8)
    result = summarize_contamination(examples, hits, docs_scanned)

    assert result["corpus_docs_scanned"] == 2
    assert result["num_contaminated"] == 1, "exactly the FIRST example (genuinely shared text) should be flagged"
    assert result["contaminated_examples"][0]["index"] == 0
    assert result["contamination_rate"] == 0.5

    print("pretraining.cli_contamination smoke test passed:")
    print(f"  {result['num_contaminated']}/{result['num_examples']} flagged ({result['contamination_rate']:.1%})")
    print(f"  matched shingle(s): {result['contaminated_examples'][0]['matched_shingles']}")


if __name__ == "__main__":
    main()
