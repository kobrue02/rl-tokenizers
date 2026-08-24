"""Command-line entry point for systems.pretraining.encoder_train.train,
mirroring cli.py's shape (every EncoderTrainConfig field becomes a `--flag`,
generated from the dataclass itself) and pretraining.cli's own field-to-flag
generation.

Usage:
    python3 -m systems.pretraining.encoder_cli --shard-dir vocab_out/prep_run --encoder-size base --total-steps 20000
    torchrun --nproc_per_node=4 -m systems.pretraining.encoder_cli --shard-dir ... --encoder-size large ...

(data prep is shared with the decoder pipeline -- see
`python3 -m systems.pretraining.data_prep --help`, which builds the packed
token shards --shard-dir here points at; nothing about shard construction
differs between training a decoder or this encoder over the same corpus/tokenizer.)
"""

import argparse
import dataclasses

from common.config_file import parse_args_with_config

from .encoder_train import EncoderTrainConfig, train


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Train an XLM-R-architecture MLM encoder (see encoder_model.py) over packed "
        "token shards (see systems.pretraining.data_prep)."
    )
    for field in dataclasses.fields(EncoderTrainConfig):
        flag = "--" + field.name.replace("_", "-")
        help_text = f"(EncoderTrainConfig.{field.name}, default: {field.default})"
        if field.type is bool:
            parser.add_argument(
                flag, action=argparse.BooleanOptionalAction, default=field.default, help=help_text
            )
        else:
            parser.add_argument(flag, type=field.type, default=field.default, help=help_text)
    return parser


def _config_from_args(args):
    field_names = {f.name for f in dataclasses.fields(EncoderTrainConfig)}
    return EncoderTrainConfig(**{k: v for k, v in vars(args).items() if k in field_names})


def main(argv=None):
    args = parse_args_with_config(build_arg_parser(), argv)
    cfg = _config_from_args(args)
    if not cfg.shard_dir:
        raise SystemExit("--shard-dir is required (output of a prior data_prep.py run)")
    train(cfg)


if __name__ == "__main__":
    main()
