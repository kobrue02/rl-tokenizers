"""Command-line entry point for systems.pretraining.encoder_eval, evaluating
a trained encoder_train.py checkpoint on the three of Glot500's six eval
tasks built here (see encoder_eval.py's own docstring for why NER/POS/
Taxi1500 aren't): pseudoperplexity, sentence retrieval, and roundtrip
alignment. Mirrors cli_eval.py's shape (one script, --benchmark dispatch)
for the decoder's own eval suite.

Usage (--system/--tokenizer-checkpoint/--vocab-json follow the exact same
convention as cli_eval.py's own decoder eval script -- the tokenizer used
to build the checkpoint's training shards is passed explicitly, not
auto-derived from a shard_dir that may not exist anymore or be reachable
from wherever eval runs):
    python3 -m systems.pretraining.encoder_cli_eval --checkpoint checkpoints/encoder_pretrain/final.pt \\
        --system bpe --tokenizer-checkpoint checkpoints/bpe_12345.json \\
        --benchmark retrieval --dataset tatoeba_mt --pair deu-eng --split test

    python3 -m systems.pretraining.encoder_cli_eval --checkpoint ... --system bpe --tokenizer-checkpoint ... \\
        --benchmark roundtrip --cycle-langs eng,fra,deu,eng

    python3 -m systems.pretraining.encoder_cli_eval --checkpoint ... --system bpe --tokenizer-checkpoint ... \\
        --benchmark pppl --dataset tatoeba_mt --pair deu-eng --split test --lang deu

Data comes from common.data.corpora.stream_groups -- retrieval and pppl use
"tatoeba_mt" (config="{split}/{pair}", see list_tatoeba_mt_pairs for valid
pair spellings) or "bible_nlp" (langs=[...], needs a one-time local prep via
common.data.prepare_bible_nlp); roundtrip needs "bible_nlp" specifically,
since only it yields N-way (3+ language) aligned groups in one call.
"""

import argparse
import json

import torch

from common.config_file import parse_args_with_config
from common.data.corpora import stream_groups

from .encoder_eval import corpus_pseudo_perplexity, roundtrip_accuracy, sentence_retrieval
from .encoder_model import build_encoder
from .encoder_model_configs import get_preset
from .encoder_tokenizer import EncoderVocab
from .encoder_train import EncoderTrainConfig
from .tokenizer_adapter import ALL_SYSTEMS


def load_pretrained_encoder(checkpoint_path, device="cpu"):
    """Reconstructs a build_encoder model from an encoder_train.save_checkpoint
    file -- same pattern as cli_eval.load_pretrained_model for the decoder:
    the checkpoint stores EncoderTrainConfig's fields plus vocab_size, not a
    preset directly, so get_preset(cfg.encoder_size) rebuilds the
    architecture the same way encoder_train.train() does."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = EncoderTrainConfig(**ckpt["config"])
    preset = get_preset(cfg.encoder_size)
    preset.max_seq_len = max(preset.max_seq_len, cfg.seq_len)
    model = build_encoder(preset, ckpt["vocab_size"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def _load_vocab(meta_system, meta_checkpoint, vocab_json):
    return EncoderVocab.load(meta_system, meta_checkpoint, vocab_json_path=vocab_json or None)


def run_retrieval(args, model, vocab):
    if args.dataset == "tatoeba_mt":
        rows = list(stream_groups("tatoeba_mt", config=f"{args.split}/{args.pair}"))
        src_lang, tgt_lang = args.pair.split("-")
        if tgt_lang != "eng" and src_lang == "eng":
            src_lang, tgt_lang = tgt_lang, src_lang  # keep English as the retrieval target, matching Glot500
    else:
        rows = list(stream_groups("bible_nlp", langs=[args.lang, "eng"]))
        src_lang, tgt_lang = args.lang, "eng"
    source_texts = [r[src_lang] for r in rows]
    target_texts = [r[tgt_lang] for r in rows]
    print(f"retrieval: {len(rows)} {src_lang}-{tgt_lang} pairs from {args.dataset}")
    result = sentence_retrieval(model, vocab, source_texts, target_texts, device=args.device, layer=args.layer)
    for k, acc in sorted(result.items()):
        print(f"  top{k}_accuracy={acc:.4f}")
    return {f"top{k}_accuracy": acc for k, acc in result.items()}


def run_roundtrip(args, model, vocab):
    cycle_langs = args.cycle_langs.split(",")
    rows = list(stream_groups("bible_nlp", langs=list(set(cycle_langs))))
    print(f"roundtrip: {len(rows)} verse groups, cycle={cycle_langs}")
    acc = roundtrip_accuracy(model, vocab, rows, cycle_langs, device=args.device, layer=args.layer)
    print(f"  roundtrip_accuracy={acc:.4f}")
    return {"roundtrip_accuracy": acc}


def run_pppl(args, model, vocab):
    if args.dataset == "tatoeba_mt":
        rows = list(stream_groups("tatoeba_mt", config=f"{args.split}/{args.pair}"))
        texts = [r[args.lang] for r in rows]
    else:
        rows = list(stream_groups("bible_nlp", langs=[args.lang]))
        texts = [r[args.lang] for r in rows]
    print(f"pppl: {len(texts)} {args.lang} sentences from {args.dataset}")
    ppl = corpus_pseudo_perplexity(model, vocab, texts, langs=[args.lang] * len(texts), device=args.device)
    print(f"  pseudoperplexity={ppl:.4f}")
    return {"pseudoperplexity": ppl}


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Evaluate an encoder_train.py checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--system", choices=ALL_SYSTEMS, required=True)
    parser.add_argument("--tokenizer-checkpoint", type=str, required=True, help="systems/ tokenizer checkpoint")
    parser.add_argument("--benchmark", choices=["retrieval", "roundtrip", "pppl"], required=True)
    parser.add_argument(
        "--dataset", choices=["tatoeba_mt", "bible_nlp"], default="tatoeba_mt",
        help="data source for retrieval/pppl (see common.data.corpora); roundtrip always uses bible_nlp",
    )
    parser.add_argument("--pair", type=str, default=None, help="e.g. 'deu-eng' -- see list_tatoeba_mt_pairs")
    parser.add_argument("--split", type=str, default="test", help="tatoeba_mt split (dev/test)")
    parser.add_argument("--lang", type=str, default=None, help="target language for --dataset bible_nlp / --benchmark pppl")
    parser.add_argument("--cycle-langs", type=str, default="eng,fra,deu,eng", help="comma-separated, must start and end on the same language -- --benchmark roundtrip only")
    parser.add_argument("--layer", type=int, default=8, help="hidden_states index to pool/align from -- Glot500's own default")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--vocab-json", type=str, default="", help="only needed if the checkpoint's tokenizer system is a span-family system")
    parser.add_argument(
        "--label", type=str, default="", help="name this run compares under -- defaults to --system. "
        "scripts.combine_encoder_results groups records by this key (e.g. run bpe and fanta checkpoints "
        "through this same --benchmark with --label bpe / --label fanta so both land under distinct keys "
        "even if --system alone would collide, e.g. two fanta runs at different vocab sizes)",
    )
    parser.add_argument("--output", type=str, default="", help="write this run's result as JSON here (see scripts.combine_encoder_results); always also printed to stdout")
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument(
        "--wandb-project", type=str, default="pretraining",
        help="same default project as train.py/encoder_train.py/cli_eval.py; this run logs "
        "job_type='encoder_eval' so all share one project, filterable apart in the wandb UI",
    )
    parser.add_argument("--run-name", type=str, default="")
    return parser


def main(argv=None):
    args = parse_args_with_config(build_arg_parser(), argv)
    device = torch.device(args.device)
    model = load_pretrained_encoder(args.checkpoint, device=device)
    vocab = _load_vocab(args.system, args.tokenizer_checkpoint, args.vocab_json)

    if args.benchmark == "retrieval":
        result = run_retrieval(args, model, vocab)
    elif args.benchmark == "roundtrip":
        result = run_roundtrip(args, model, vocab)
    else:
        result = run_pppl(args, model, vocab)

    if args.output:
        record = {
            "label": args.label or args.system,
            "benchmark": args.benchmark,
            "checkpoint": args.checkpoint,
            "system": args.system,
            "tokenizer_checkpoint": args.tokenizer_checkpoint,
            "config": {
                "dataset": args.dataset, "pair": args.pair, "split": args.split,
                "lang": args.lang, "cycle_langs": args.cycle_langs, "layer": args.layer,
            },
            "result": result,
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, default=str)
        print(f"wrote result to {args.output}")

    if args.use_wandb:
        import wandb

        run = wandb.init(
            project=args.wandb_project,
            name=args.run_name or None,
            job_type="encoder_eval",
            config={
                "checkpoint": args.checkpoint,
                "system": args.system,
                "tokenizer_checkpoint": args.tokenizer_checkpoint,
                "benchmark": args.benchmark,
                "dataset": args.dataset,
                "pair": args.pair,
                "split": args.split,
                "lang": args.lang,
                "cycle_langs": args.cycle_langs,
                "layer": args.layer,
            },
        )
        run.log(result)
        run.finish()
        print(f"logged eval results to wandb project={args.wandb_project!r}")


if __name__ == "__main__":
    main()
