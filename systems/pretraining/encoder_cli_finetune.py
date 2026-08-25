"""Command-line entry point for the encoder finetuning pipeline
(encoder_finetune_tagging.py / encoder_finetune_classification.py /
encoder_finetune_taxi1500.py), evaluating a trained encoder_train.py
checkpoint on the three of Glot500's six tasks that need a finetuned task
head (NER, POS, Taxi1500 -- see encoder_eval.py's own module docstring for
the other three), plus SIB-200 -- a topic-classification benchmark NOT part
of Glot500's own eval suite, added on request. All four finetune on one
source language (English by convention, matching Glot500's own
run_tag.py/zero_shot_train.py protocol) and evaluate zero-shot on a
different target language -- or, for ner/pos/sib200, on MANY target
languages at once (--eval-langs/--eval-configs/--eval-lang-scripts,
comma-separated or "all") without retraining: the finetuned head is trained
ONCE, then evaluated against each target language's own test set via
Trainer's native eval_dataset=dict[str, Dataset] support (see
encoder_finetune_tagging.finetune_tagging's own docstring for why this is
the actually-correct "zero-shot transfer to many languages" protocol,
rather than resubmitting the whole finetune once per language).

Data sources (repo ids/configs/label schemes verified live against the HF
Hub and GitHub as of this writing -- see each function's own docstring for
what was actually checked, not assumed):
  - NER: datasets.load_dataset("unimelb-nlp/wikiann", lang) -- lang is a
    plain 2-letter code ("en", "de", "fr", ...). ner_tags are already
    integer ids in WIKIANN_LABELS' own order.
  - POS: datasets.load_dataset("universal-dependencies/universal_dependencies",
    config) -- config is "{lang}_{treebank}" (e.g. "en_ewt", "de_gsd").
    upos is a list of STRING tags (no baked-in ClassLabel on this dataset),
    remapped here through UPOS_LABELS' own fixed order.
  - Taxi1500: English-only via encoder_finetune_taxi1500 (see its own
    docstring for why -- the HF Hub's only Taxi1500 dataset ships text
    without labels). --taxi1500-eval-tsv lets you point at a non-English
    labeled TSV you've obtained yourself (Taxi1500-c is gated, not
    auto-downloadable); omitting it evaluates on English too.
  - SIB-200: datasets.load_dataset("mteb/sib200", lang_script) -- config is
    a "{lang3}_{Script}" code (e.g. "eng_Latn", "deu_Latn"), 206 total.
    Row: {label: ClassLabel(7 topics), text, lang}. label is already an
    integer id in SIB200_LABELS' own order (confirmed live via the
    dataset's own ClassLabel.names) -- no remapping needed, unlike UPOS.
    Splits: train (701 rows)/validation (99)/test (99), identical across
    every language (SIB-200 is FLORES-200-sentence-aligned).

Usage:
    python3 -m systems.pretraining.encoder_cli_finetune --checkpoint checkpoints/encoder_pretrain/final.pt \\
        --system bpe --tokenizer-checkpoint checkpoints/bpe_12345.json \\
        --task ner --train-lang en --eval-lang de --output-dir finetune_out/ner_de

    python3 -m systems.pretraining.encoder_cli_finetune --checkpoint ... --system bpe --tokenizer-checkpoint ... \\
        --task pos --train-config en_ewt --eval-config de_gsd --output-dir finetune_out/pos_de

    python3 -m systems.pretraining.encoder_cli_finetune --checkpoint ... --system bpe --tokenizer-checkpoint ... \\
        --task taxi1500 --output-dir finetune_out/taxi1500

    python3 -m systems.pretraining.encoder_cli_finetune --checkpoint ... --system bpe --tokenizer-checkpoint ... \\
        --task sib200 --train-lang-script eng_Latn --eval-lang-script deu_Latn --output-dir finetune_out/sib200_deu

    # Full eval: train once on English, evaluate against every SIB-200 language in one run
    python3 -m systems.pretraining.encoder_cli_finetune --checkpoint ... --system bpe --tokenizer-checkpoint ... \\
        --task sib200 --train-lang-script eng_Latn --eval-lang-scripts all --output-dir finetune_out/sib200_full
"""

import argparse
import json

from common.config_file import parse_args_with_config

from .encoder_finetune_classification import TAXI1500_LABELS, finetune_classification
from .encoder_finetune_tagging import finetune_tagging
from .encoder_finetune_taxi1500 import download_taxi1500_split, load_taxi1500_tsv
from .encoder_tokenizer import EncoderVocab
from .tokenizer_adapter import ALL_SYSTEMS

WIKIANN_LABELS = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]
UPOS_LABELS = [
    "ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM",
    "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X",
]
# Confirmed live via mteb/sib200's own ClassLabel(...).names -- do not
# reorder, label ids are positional into this list.
SIB200_LABELS = ["entertainment", "geography", "health", "politics", "science/technology", "sports", "travel"]


def _capped(hf_split, max_examples):
    return hf_split if max_examples is None else hf_split.select(range(min(max_examples, len(hf_split))))


def _load_wikiann_split(lang, split, max_examples):
    from datasets import load_dataset

    return _capped(load_dataset("unimelb-nlp/wikiann", lang, split=split), max_examples)


def _load_ud_raw_split(config, split, max_examples):
    from datasets import load_dataset

    return _capped(load_dataset("universal-dependencies/universal_dependencies", config, split=split), max_examples)


def _remap_ud_upos_to_int(hf_split):
    """Universal Dependencies' own `upos` column is a list of STRINGS (no
    ClassLabel) -- remapped here through UPOS_LABELS' own fixed order into
    plain Python dict rows (see encoder_finetune_tagging.TaggingDataset,
    which just needs len()+index access, not specifically an HF Dataset).
    Split from _load_ud_raw_split so a multi-language sweep can validate
    the raw HF Dataset's own columns (see _build_eval_rows_by_name) BEFORE
    remapping -- remapping a malformed config would raise a much less
    obvious KeyError on `row["upos"]` instead."""
    label_to_id = {label: i for i, label in enumerate(UPOS_LABELS)}
    return [{"tokens": row["tokens"], "upos_ids": [label_to_id[tag] for tag in row["upos"]]} for row in hf_split]


def _load_ud_split_with_int_upos(config, split, max_examples):
    return _remap_ud_upos_to_int(_load_ud_raw_split(config, split, max_examples))


def _load_sib200_split(lang_script, split, max_examples):
    from datasets import load_dataset

    return _capped(load_dataset("mteb/sib200", lang_script, split=split), max_examples)


def _build_eval_rows_by_name(load_raw_fn, eval_names, required_columns, dataset_label, max_eval_examples, remap_fn=None):
    """Builds the eval_rows_by_name dict a multi-language sweep passes to
    finetune_tagging/finetune_classification, skipping (with a printed
    warning, never silently) any language whose loaded split is missing an
    expected column -- not every config of a community-hosted multi-config
    dataset necessarily shares the same schema. Confirmed live: 2 of
    mteb/sib200's first 20 non-English configs (ace_Arab, arb_Arab) ship
    {'index_id', 'category', 'text', 'lang'} instead of the expected
    {'label', 'text', 'lang'} -- this crashed a real 12-hour
    --eval-langs=all finetune job partway through its first epoch
    (Trainer's own eval_strategy="epoch" hit the bad config on the first
    end-of-epoch evaluation, deep enough in that the whole job's progress
    was lost). load_raw_fn(name, "test", max_eval_examples) must return
    the RAW HF Dataset (before any remapping, e.g. UD's own upos-string-
    to-int step) so validation sees the dataset's own real columns;
    remap_fn, if given, is applied only to languages that pass validation.

    Raises if every single requested language fails validation (nothing
    left to evaluate at all) rather than silently returning an empty,
    useless eval_dataset."""
    eval_rows_by_name = {}
    for name in eval_names:
        raw = load_raw_fn(name, "test", max_eval_examples)
        missing = [c for c in required_columns if c not in raw.column_names]
        if missing:
            print(
                f"  WARNING: {dataset_label} config {name!r} is missing column(s) {missing} "
                f"(has {raw.column_names}) -- skipping this language, not the whole run."
            )
            continue
        eval_rows_by_name[name] = (remap_fn(raw) if remap_fn is not None else raw, None)
    if not eval_rows_by_name:
        raise RuntimeError(
            f"{dataset_label}: every requested eval language failed schema validation -- nothing to evaluate"
        )
    return eval_rows_by_name


def _resolve_eval_names(eval_langs_arg, repo_id, train_name, max_eval_langs):
    """eval_langs_arg: None/"" (single-target path, caller falls back to
    --eval-lang/--eval-config/--eval-lang-script), a comma-separated list,
    or the literal "all" -- discovers EVERY config datasets.
    get_dataset_config_names(repo_id) reports for that Hub repo, minus
    train_name itself (no point zero-shot-"transferring" to the language it
    was just finetuned on). max_eval_langs caps an "all" sweep -- capping
    silently would misrepresent what was actually evaluated, so this always
    prints what it resolved to and, when capped, exactly how much was
    dropped (this project's own "no silent caps" convention, see
    Workflow's own guidance this repo's tooling follows elsewhere)."""
    if not eval_langs_arg:
        return None
    if eval_langs_arg == "all":
        from datasets import get_dataset_config_names

        names = [n for n in get_dataset_config_names(repo_id) if n != train_name]
    else:
        names = [n for n in eval_langs_arg.split(",") if n != train_name]
    if max_eval_langs and len(names) > max_eval_langs:
        print(f"  (capping {len(names)} eval languages down to --max-eval-langs={max_eval_langs}: dropping {len(names) - max_eval_langs})")
        names = names[:max_eval_langs]
    print(f"  evaluating on {len(names)} language(s): {names}")
    return names


def run_ner(args, vocab):
    train_rows = _load_wikiann_split(args.train_lang, "train", args.max_train_examples)
    eval_names = _resolve_eval_names(args.eval_langs, "unimelb-nlp/wikiann", args.train_lang, args.max_eval_langs)

    if eval_names is not None:
        print(f"ner: {len(train_rows)} train ({args.train_lang})")
        eval_rows_by_name = _build_eval_rows_by_name(
            _load_wikiann_split, eval_names, ["tokens", "ner_tags"], "wikiann", args.max_eval_examples
        )
        result = finetune_tagging(
            args.checkpoint, train_rows, eval_rows=None, tag_column="ner_tags", label_list=WIKIANN_LABELS,
            vocab=vocab, output_dir=args.output_dir, device=args.device,
            eval_rows_by_name=eval_rows_by_name, use_wandb=args.use_wandb, run_name=args.run_name or None,
        )
        for name in eval_rows_by_name:
            print(f"  {name}: eval_f1={result[f'eval_{name}_f1']:.4f} eval_precision={result[f'eval_{name}_precision']:.4f} eval_recall={result[f'eval_{name}_recall']:.4f}")
        return result

    eval_rows = _load_wikiann_split(args.eval_lang, "test", args.max_eval_examples)
    print(f"ner: {len(train_rows)} train ({args.train_lang}) / {len(eval_rows)} eval ({args.eval_lang}) rows")
    result = finetune_tagging(
        args.checkpoint, train_rows, eval_rows, tag_column="ner_tags", label_list=WIKIANN_LABELS,
        vocab=vocab, output_dir=args.output_dir, device=args.device,
        use_wandb=args.use_wandb, run_name=args.run_name or None,
    )
    print(f"  eval_f1={result['eval_f1']:.4f} eval_precision={result['eval_precision']:.4f} eval_recall={result['eval_recall']:.4f}")
    return result


def run_pos(args, vocab):
    train_rows = _load_ud_split_with_int_upos(args.train_config, "train", args.max_train_examples)
    eval_names = _resolve_eval_names(
        args.eval_configs, "universal-dependencies/universal_dependencies", args.train_config, args.max_eval_langs
    )

    if eval_names is not None:
        print(f"pos: {len(train_rows)} train ({args.train_config})")
        eval_rows_by_name = _build_eval_rows_by_name(
            _load_ud_raw_split, eval_names, ["tokens", "upos"], "universal_dependencies",
            args.max_eval_examples, remap_fn=_remap_ud_upos_to_int,
        )
        result = finetune_tagging(
            args.checkpoint, train_rows, eval_rows=None, tag_column="upos_ids", label_list=UPOS_LABELS,
            scheme="flat", vocab=vocab, output_dir=args.output_dir, device=args.device,
            eval_rows_by_name=eval_rows_by_name, use_wandb=args.use_wandb, run_name=args.run_name or None,
        )
        for name in eval_rows_by_name:
            print(f"  {name}: eval_accuracy={result[f'eval_{name}_accuracy']:.4f}")
        return result

    eval_rows = _load_ud_split_with_int_upos(args.eval_config, "test", args.max_eval_examples)
    print(f"pos: {len(train_rows)} train ({args.train_config}) / {len(eval_rows)} eval ({args.eval_config}) rows")
    result = finetune_tagging(
        args.checkpoint, train_rows, eval_rows, tag_column="upos_ids", label_list=UPOS_LABELS,
        scheme="flat", vocab=vocab, output_dir=args.output_dir, device=args.device,
        use_wandb=args.use_wandb, run_name=args.run_name or None,
    )
    print(f"  eval_accuracy={result['eval_accuracy']:.4f}")
    return result


def run_taxi1500(args, vocab):
    train_path = download_taxi1500_split("train", args.taxi1500_cache_dir)
    train_rows = load_taxi1500_tsv(train_path)
    eval_path = args.taxi1500_eval_tsv or download_taxi1500_split("test", args.taxi1500_cache_dir)
    eval_rows = load_taxi1500_tsv(eval_path)
    if args.max_train_examples:
        train_rows = train_rows[: args.max_train_examples]
    if args.max_eval_examples:
        eval_rows = eval_rows[: args.max_eval_examples]
    print(f"taxi1500: {len(train_rows)} train (eng) / {len(eval_rows)} eval ({eval_path}) rows")
    result = finetune_classification(
        args.checkpoint, train_rows, eval_rows, vocab=vocab, output_dir=args.output_dir,
        label_list=TAXI1500_LABELS, device=args.device,
        use_wandb=args.use_wandb, run_name=args.run_name or None,
    )
    print(f"  eval_macro_f1={result['eval_macro_f1']:.4f}")
    return result


def run_sib200(args, vocab):
    train_rows = _load_sib200_split(args.train_lang_script, "train", args.max_train_examples)
    eval_names = _resolve_eval_names(args.eval_lang_scripts, "mteb/sib200", args.train_lang_script, args.max_eval_langs)

    if eval_names is not None:
        print(f"sib200: {len(train_rows)} train ({args.train_lang_script})")
        eval_rows_by_name = _build_eval_rows_by_name(
            _load_sib200_split, eval_names, ["text", "label"], "sib200", args.max_eval_examples
        )
        result = finetune_classification(
            args.checkpoint, train_rows, eval_rows=None, vocab=vocab, output_dir=args.output_dir,
            label_list=SIB200_LABELS, device=args.device,
            eval_rows_by_name=eval_rows_by_name, use_wandb=args.use_wandb, run_name=args.run_name or None,
        )
        for name in eval_rows_by_name:
            print(f"  {name}: eval_macro_f1={result[f'eval_{name}_macro_f1']:.4f}")
        return result

    eval_rows = _load_sib200_split(args.eval_lang_script, "test", args.max_eval_examples)
    print(f"sib200: {len(train_rows)} train ({args.train_lang_script}) / {len(eval_rows)} eval ({args.eval_lang_script}) rows")
    result = finetune_classification(
        args.checkpoint, train_rows, eval_rows, vocab=vocab, output_dir=args.output_dir,
        label_list=SIB200_LABELS, device=args.device,
        use_wandb=args.use_wandb, run_name=args.run_name or None,
    )
    print(f"  eval_macro_f1={result['eval_macro_f1']:.4f}")
    return result


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Finetune an encoder_train.py checkpoint on NER/POS/Taxi1500/SIB-200.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--system", choices=ALL_SYSTEMS, required=True)
    parser.add_argument("--tokenizer-checkpoint", type=str, required=True, help="systems/ tokenizer checkpoint")
    parser.add_argument("--vocab-json", type=str, default="", help="only needed if the checkpoint's tokenizer system is a span-family system")
    parser.add_argument("--task", choices=["ner", "pos", "taxi1500", "sib200"], required=True)
    parser.add_argument("--train-lang", type=str, default="en", help="--task ner only -- WikiANN config, a 2-letter code")
    parser.add_argument("--eval-lang", type=str, default="de", help="--task ner only -- ignored if --eval-langs is given")
    parser.add_argument(
        "--eval-langs", type=str, default="", help="--task ner only -- comma-separated list of WikiANN "
        "language codes, or the literal 'all' for every WikiANN config (minus --train-lang); evaluates the SAME "
        "finetuned model against each one, no retraining. Overrides --eval-lang when given",
    )
    parser.add_argument("--train-config", type=str, default="en_ewt", help="--task pos only -- '{lang}_{treebank}', see universal-dependencies/universal_dependencies's own config list")
    parser.add_argument("--eval-config", type=str, default="de_gsd", help="--task pos only -- ignored if --eval-configs is given")
    parser.add_argument(
        "--eval-configs", type=str, default="", help="--task pos only -- comma-separated list of "
        "universal-dependencies/universal_dependencies configs, or 'all'. Overrides --eval-config when given",
    )
    parser.add_argument("--taxi1500-cache-dir", type=str, default="data/taxi1500", help="--task taxi1500 only -- local cache dir for the downloaded English TSVs")
    parser.add_argument("--taxi1500-eval-tsv", type=str, default="", help="--task taxi1500 only -- a non-English labeled TSV you've obtained yourself (same 3-column format, see encoder_finetune_taxi1500.py); omit to evaluate on English too")
    parser.add_argument("--train-lang-script", type=str, default="eng_Latn", help="--task sib200 only -- mteb/sib200 config, a '{lang3}_{Script}' code")
    parser.add_argument("--eval-lang-script", type=str, default="deu_Latn", help="--task sib200 only -- ignored if --eval-lang-scripts is given")
    parser.add_argument(
        "--eval-lang-scripts", type=str, default="", help="--task sib200 only -- comma-separated list of "
        "mteb/sib200 '{lang3}_{Script}' configs, or 'all' for every one of its 206 configs (minus "
        "--train-lang-script). Overrides --eval-lang-script when given",
    )
    parser.add_argument(
        "--max-eval-langs", type=int, default=None, help="caps --eval-langs/--eval-configs/--eval-lang-scripts=all "
        "sweeps -- prints exactly how many were dropped, never silently truncates",
    )
    parser.add_argument("--max-train-examples", type=int, default=None)
    parser.add_argument("--max-eval-examples", type=int, default=None, help="applied PER language when evaluating on many")
    parser.add_argument("--output-dir", type=str, required=True, help="Trainer's own output_dir (checkpoints disabled -- save_strategy='no' -- but Trainer still writes logs here)")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--label", type=str, default="", help="name this run compares under -- defaults to --system. "
        "scripts.combine_encoder_results groups records by this key (not --output-dir, which is Trainer's own "
        "unrelated log directory)",
    )
    parser.add_argument("--results-output", type=str, default="", help="write this run's result as JSON here (see scripts.combine_encoder_results); always also printed to stdout")
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument(
        "--wandb-project", type=str, default="pretraining",
        help="same default project as train.py/encoder_train.py/cli_eval.py/encoder_cli_eval.py; this run logs "
        "job_type='finetune' so all share one project, filterable apart in the wandb UI",
    )
    parser.add_argument("--run-name", type=str, default="")
    return parser


def main(argv=None):
    args = parse_args_with_config(build_arg_parser(), argv)
    vocab = EncoderVocab.load(args.system, args.tokenizer_checkpoint, vocab_json_path=args.vocab_json or None)

    # Initialized BEFORE finetune_tagging/finetune_classification build their
    # own Trainer, deliberately: transformers' own WandbCallback.setup()
    # only calls wandb.init() itself when wandb.run is still None (confirmed
    # against this project's installed transformers version's own source),
    # so pre-initializing here is what actually gets OUR project/job_type/
    # config onto the run Trainer logs its own per-step metrics into,
    # instead of Trainer silently falling back to its own WANDB_PROJECT env
    # var / a generic run name.
    run = None
    if args.use_wandb:
        import wandb

        run = wandb.init(
            project=args.wandb_project,
            name=args.run_name or None,
            job_type="finetune",
            config={
                "checkpoint": args.checkpoint,
                "system": args.system,
                "tokenizer_checkpoint": args.tokenizer_checkpoint,
                "task": args.task,
                "train_lang": args.train_lang,
                "eval_lang": args.eval_lang,
                "eval_langs": args.eval_langs,
                "train_config": args.train_config,
                "eval_config": args.eval_config,
                "eval_configs": args.eval_configs,
                "train_lang_script": args.train_lang_script,
                "eval_lang_script": args.eval_lang_script,
                "eval_lang_scripts": args.eval_lang_scripts,
                "max_train_examples": args.max_train_examples,
                "max_eval_examples": args.max_eval_examples,
                "max_eval_langs": args.max_eval_langs,
            },
        )

    if args.task == "ner":
        result = run_ner(args, vocab)
    elif args.task == "pos":
        result = run_pos(args, vocab)
    elif args.task == "taxi1500":
        result = run_taxi1500(args, vocab)
    else:
        result = run_sib200(args, vocab)

    if args.results_output:
        record = {
            "label": args.label or args.system,
            "task": args.task,
            "checkpoint": args.checkpoint,
            "system": args.system,
            "tokenizer_checkpoint": args.tokenizer_checkpoint,
            "config": {
                "train_lang": args.train_lang, "eval_lang": args.eval_lang, "eval_langs": args.eval_langs,
                "train_config": args.train_config, "eval_config": args.eval_config, "eval_configs": args.eval_configs,
                "train_lang_script": args.train_lang_script, "eval_lang_script": args.eval_lang_script,
                "eval_lang_scripts": args.eval_lang_scripts,
            },
            "result": result,
        }
        with open(args.results_output, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, default=str)
        print(f"wrote result to {args.results_output}")

    if run is not None:
        run.log(result)
        run.finish()
        print(f"logged finetune results to wandb project={args.wandb_project!r}")


if __name__ == "__main__":
    main()
