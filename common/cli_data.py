"""Shared --data-source/--langs/--num-groups CLI data-loading logic, used
identically by every tokenizer's CLI in this repo (fairtok.cli, magnet.cli,
flexitokens.cli, manta.cli) -- extracted here so it's implemented exactly once
instead of copy-pasted per tokenizer, which is exactly the kind of drift risk
("did I update all four when the data sources changed?") a shared module exists
to remove.
"""

from .data import LANG_PROFILES, make_synthetic_parallel_groups
from .oldi_data import (
    LANGS,
    load_all_training_groups,
    load_flores_plus,
    load_oldi_seed,
    load_smol_groups,
)

DATA_SOURCES = ["synthetic", "oldi_seed", "flores_dev", "smol", "all"]


def load_groups(args):
    """args: an argparse.Namespace (or anything with the same attributes) with
    `.langs`, `.data_source`, `.seed`, and `.num_groups` -- every tokenizer's CLI
    in this repo adds these same four flags (see e.g. fairtok/cli.py's
    build_arg_parser), so this function works unmodified against any of them."""
    if args.langs is None:
        langs = None
    elif args.langs == "all":
        langs = "all"
    else:
        langs = args.langs.split(",")

    if langs == "all" and args.data_source in ("synthetic", "smol"):
        raise ValueError(
            f"--langs all isn't supported for --data-source {args.data_source} "
            "(synthetic has a fixed toy panel; smol needs a per-language-file rescan -- see oldi_data.py)"
        )

    if args.data_source == "synthetic":
        groups = make_synthetic_parallel_groups(
            400, langs=langs or list(LANG_PROFILES), seed=args.seed
        )
    elif args.data_source == "oldi_seed":
        groups = load_oldi_seed(langs=langs or LANGS)
    elif args.data_source == "flores_dev":
        groups = load_flores_plus(split="dev", langs=langs or LANGS)
    elif args.data_source == "smol":
        groups = load_smol_groups(langs=langs or [l for l in LANGS if l != "eng"])
    elif args.data_source == "all":
        groups = load_all_training_groups(langs=langs or LANGS)
    else:
        raise ValueError(f"unknown data source: {args.data_source}")

    if args.num_groups is not None:
        groups = groups[: args.num_groups]
    return groups
