"""Shared --data-source/--langs/--num-groups CLI data-loading logic, used
identically by every tokenizer's CLI in this repo (fairtok.cli, magnet.cli,
flexitokens.cli, manta.cli, fanta.cli, superbpe.cli, bpe.cli) -- extracted
here so it's implemented exactly once instead of copy-pasted per tokenizer.

Every named source except "all" now goes through common.data.corpora.stream_groups
-- the SAME registry pretraining.data_prep draws on for LLM pretraining, not
a separate tokenizer-training-only list. See that module's own docstring for
which sources are genuinely cross-lingual PARALLEL (oldi_seed/flores_dev/smol)
vs. single-language MONOLINGUAL (glot500/fineweb_edu/olmo_mix) -- the latter
now trains a tokenizer just fine through the plain next-byte CE loss every
trainer has, it just contributes nothing to any fairness loss term that needs
genuinely parallel content within a group. "all" stays its own case here
(common.data.oldi_data.load_all_training_groups, pooling only the three parallel
sources) rather than folding the monolingual sources into it -- that keeps
"all"'s existing meaning (and every past run's reproducibility) unchanged.
"""

import itertools

from .corpora import ALL_SOURCES, BITEXT_SOURCES, MONOLINGUAL_SOURCES, stream_groups
from .oldi_data import LANGS, load_all_training_groups

# Derived from common.data.corpora.ALL_SOURCES (+ "all", this module's own pooling
# special-case -- see module docstring) rather than a second hardcoded list:
# a new source registered in corpora.py becomes selectable here automatically,
# with no separate list to remember to update in sync.
DATA_SOURCES = ALL_SOURCES + ["all"]

# A monolingual source's own stream is lazy/effectively unbounded (a live
# HF Hub stream, not a small fixed file the way oldi_seed/flores_dev/smol
# are) -- materializing one without SOME bound would try to download an
# unbounded amount of data. Used only when --num-groups isn't given.
_DEFAULT_MONOLINGUAL_GROUPS_PER_LANG = 2000

_DEFAULT_DATA_SOURCE_HELP = "'all' pools oldi_seed+flores_dev+smol (default); 'synthetic' is the placeholder corpus"
_DEFAULT_LANGS_HELP = (
    "comma-separated language codes; 'all' to use every language oldi_seed/flores_dev natively "
    "offer; defaults to the 9-language panel for the chosen data source"
)


def add_data_source_args(parser, data_source_help=None, langs_help=None):
    """--data-source/--num-groups/--langs, the three flags every systems/*/
    cli.py's build_arg_parser already added identically (confirmed live:
    five of seven byte-identical; bpe adds an extra UTF-8 caveat sentence to
    --data-source's help, fairtok adds an extra clarifying clause to
    --langs's -- both passed through via the optional *_help params rather
    than silently overwritten with the plain default text)."""
    parser.add_argument(
        "--data-source", choices=DATA_SOURCES, default="all", help=data_source_help or _DEFAULT_DATA_SOURCE_HELP
    )
    parser.add_argument(
        "--num-groups", type=int, default=None,
        help="cap the number of parallel groups loaded (real sources are large; omit for the full set)",
    )
    parser.add_argument("--langs", type=str, default=None, help=langs_help or _DEFAULT_LANGS_HELP)


def load_groups(args):
    """args: an argparse.Namespace (or anything with the same attributes) with
    `.langs`, `.data_source`, `.seed`, and `.num_groups` -- every tokenizer's CLI
    in this repo adds these same four flags (see e.g. fairtok/cli.py's
    build_arg_parser), so this function works unmodified against any of them.
    An optional `.dataset_config` attribute (HF config name) is read too, for
    --data-source fineweb_edu/olmo_mix and every BITEXT_SOURCES entry
    (ccmatrix/un_pc/europarl/tatoeba_mt) -- see common.data.corpora.stream_groups;
    every tokenizer's cli.py that wants to expose those sources adds that
    flag, the others simply never read it.
    """
    if args.langs is None:
        langs = None
    elif args.langs == "all":
        langs = "all"
    else:
        langs = args.langs.split(",")

    if langs == "all" and (
        args.data_source in ("synthetic", "smol", "fineweb_edu", "olmo_mix") or args.data_source in BITEXT_SOURCES
    ):
        raise ValueError(
            f"--langs all isn't supported for --data-source {args.data_source} "
            "(synthetic has a fixed toy panel; smol needs a per-language-file rescan; "
            "fineweb_edu/olmo_mix and every BITEXT_SOURCES entry (ccmatrix/un_pc/europarl/"
            "tatoeba_mt) are selected via --dataset-config, not --langs -- see common.data.corpora)"
        )

    if args.data_source == "all":
        groups = load_all_training_groups(langs=langs or LANGS)
    else:
        stream = stream_groups(
            args.data_source,
            langs=langs,
            config=getattr(args, "dataset_config", None),
            seed=args.seed,
        )
        if args.data_source in MONOLINGUAL_SOURCES or args.data_source in BITEXT_SOURCES:
            # Lazy/effectively-unbounded source -- bound materialization
            # explicitly rather than draining a live stream to exhaustion.
            limit = args.num_groups or _DEFAULT_MONOLINGUAL_GROUPS_PER_LANG * max(
                1, len(langs) if isinstance(langs, list) else 1
            )
            groups = list(itertools.islice(stream, limit))
        else:
            groups = list(stream)

    if args.num_groups is not None:
        groups = groups[: args.num_groups]
    return groups


def load_bouquet_dev_for_training(args):
    """BOUQuET dev (see common.data.oldi_data.load_bouquet_dev) -- disjoint from every
    --data-source load_groups above trains on, used for periodic in-training
    evaluation at epoch boundaries (see each tokenizer's own Trainer.train()).
    Skipped (returns None) for --data-source synthetic, which has no real
    BOUQuET counterpart. Loads EVERY language BOUQuET covers ("all"), not just
    this project's 9-language panel -- common.eval.cross_tokenizer.evaluate_on_groups
    already skips languages the training model has no entry for.

    Reserve BOUQuET's TEST split (common.data.oldi_data.load_bouquet_test) for final
    reported numbers, via each tokenizer's own evaluate.py
    --eval-data-source bouquet_test, run once training is done -- using dev
    here, repeatedly, across an entire training run's worth of epoch checks, is
    exactly the exploratory/tuning use dev exists for.
    """
    if args.data_source == "synthetic":
        print(
            "warning: --data-source synthetic has no real BOUQuET counterpart -- "
            "skipping periodic epoch-boundary evaluation"
        )
        return None
    from .oldi_data import load_bouquet_dev

    eval_groups = load_bouquet_dev("all")
    print(
        f"loaded {len(eval_groups)} BOUQuET dev groups for periodic epoch-boundary evaluation"
    )
    return eval_groups
