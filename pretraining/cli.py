"""Command-line entry point for pretraining.train.train, mirroring every
systems/*/cli.py's shape: every TrainConfig field becomes a `--flag`,
generated from the dataclass itself.

Usage:
    python3 -m pretraining.cli --shard-dir vocab_out/prep_run --model-size small --total-steps 20000
    torchrun --nproc_per_node=4 -m pretraining.cli --shard-dir ... --model-size large ...

(data prep is a separate entry point -- see `python3 -m pretraining.data_prep --help`,
which builds the packed token shards --shard-dir here points at.)
"""

import argparse
import dataclasses

from common.config_file import parse_args_with_config

from .train import TrainConfig, train


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Pretrain a TransformerLM over packed token shards (see pretraining.data_prep)."
    )
    for field in dataclasses.fields(TrainConfig):
        flag = "--" + field.name.replace("_", "-")
        help_text = f"(TrainConfig.{field.name}, default: {field.default})"
        if field.type is bool:
            parser.add_argument(
                flag, action=argparse.BooleanOptionalAction, default=field.default, help=help_text
            )
        else:
            parser.add_argument(flag, type=field.type, default=field.default, help=help_text)
    return parser


def main(argv=None):
    args = parse_args_with_config(build_arg_parser(), argv)
    cfg = TrainConfig(**vars(args))
    if not cfg.shard_dir:
        raise SystemExit("--shard-dir is required (output of a prior data_prep.py run)")
    train(cfg)


if __name__ == "__main__":
    main()
