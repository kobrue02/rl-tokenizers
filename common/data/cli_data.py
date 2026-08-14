"""Shared --data-source/--langs/--num-groups CLI data-loading logic, used
identically by every tokenizer's CLI in this repo (fairtok.cli, magnet.cli,
flexitokens.cli, manta.cli, fanta.cli, superbpe.cli, bpe.cli) -- extracted
here so it's implemented exactly once instead of copy-pasted per tokenizer.

Every named source goes through common.data.corpora.stream_groups -- the SAME
registry pretraining.data_prep draws on for LLM pretraining, not a separate
tokenizer-training-only list. See that module's own docstring for which
sources are genuinely cross-lingual PARALLEL (oldi_seed/flores_dev, and
bible_nlp once prepared) vs. BITEXT (smol/ccmatrix/un_pc/europarl/tatoeba_mt,
one language pair per group) vs. single-language MONOLINGUAL (glot500/
fineweb_edu/olmo_mix) -- the latter still trains a tokenizer fine through the
plain next-byte CE loss every trainer has, it just contributes nothing to any
fairness loss term that needs genuinely parallel content within a group.

Every source now defaults to loading EVERY language it natively offers (no
built-in curated panel any more -- bible_nlp is the one deliberate exception,
which has no default at all and always requires an explicit --langs, given
the real cost of scanning it -- see common.data.corpora's own docstring).
Config files decide WHICH corpora feed a given training run: --data-source
takes either one source name, the literal "all" (the original oldi_seed+
flores_dev+smol pool, kept for backward compatibility with existing configs
-- each of the three now loads every language it has, not the old 9-panel),
or a comma-separated list of several source names to pool together for one
run (e.g. "oldi_seed,ccmatrix,europarl") -- --langs/--dataset-config aren't
supported alongside a multi-source list (they aren't source-specific), so
override either by training on a single source at a time instead.
"""

import itertools

from .corpora import ALL_SOURCES, BITEXT_SOURCES, MONOLINGUAL_SOURCES, stream_groups

# Derived from common.data.corpora.ALL_SOURCES (+ "all", this module's own pooling
# special-case -- see module docstring) rather than a second hardcoded list:
# a new source registered in corpora.py becomes selectable here automatically,
# with no separate list to remember to update in sync.
DATA_SOURCES = ALL_SOURCES + ["all"]

# The legacy "all" meta-source, predating general multi-source pooling --
# kept as sugar for it (expanded to these 3 names, see _expand_data_sources)
# rather than its own separate pooling code path, so both go through the
# exact same per-source loading logic in _load_one_source below.
_ALL_META_SOURCES = ["oldi_seed", "flores_dev", "smol"]

# A monolingual/bitext source's own stream is lazy/effectively unbounded (a
# live HF Hub stream, not a small fixed file the way oldi_seed/flores_dev
# are) -- materializing one without SOME bound would try to download an
# unbounded amount of data. Used only when --num-groups isn't given.
_DEFAULT_MONOLINGUAL_GROUPS_PER_LANG = 2000

_DEFAULT_DATA_SOURCE_HELP = (
    "'all' pools oldi_seed+flores_dev+smol (default); 'synthetic' is the placeholder corpus; "
    "a comma-separated list (e.g. 'oldi_seed,ccmatrix') pools several sources for one run"
)
_DEFAULT_LANGS_HELP = (
    "comma-separated language codes, or 'all' (the default for every source now, so this "
    "flag is rarely needed) -- ignored for fineweb_edu/olmo_mix and every BITEXT_SOURCES "
    "entry (smol/ccmatrix/un_pc/europarl/tatoeba_mt), which use --dataset-config instead; "
    "unsupported when --data-source names more than one source"
)


def add_data_source_args(parser, data_source_help=None, langs_help=None):
    """--data-source/--num-groups/--langs, the three flags every systems/*/
    cli.py's build_arg_parser already added identically (confirmed live:
    five of seven byte-identical; bpe adds an extra UTF-8 caveat sentence to
    --data-source's help, fairtok adds an extra clarifying clause to
    --langs's -- both passed through via the optional *_help params rather
    than silently overwritten with the plain default text). --data-source has
    no `choices=` constraint (unlike before) since it now also accepts a
    comma-separated list of several DATA_SOURCES names -- load_groups
    validates each individual name itself and raises a clear error on an
    unknown one, rather than argparse rejecting the whole multi-name string."""
    parser.add_argument("--data-source", type=str, default="all", help=data_source_help or _DEFAULT_DATA_SOURCE_HELP)
    parser.add_argument(
        "--num-groups", type=int, default=None,
        help="cap the number of parallel groups loaded (real sources are large; omit for the full set)",
    )
    parser.add_argument("--langs", type=str, default=None, help=langs_help or _DEFAULT_LANGS_HELP)


def _expand_data_sources(data_source_str):
    """Splits --data-source on commas, expanding any literal "all" entry into
    its own fixed 3-source legacy pool (_ALL_META_SOURCES) -- see that
    constant's own comment."""
    names = []
    for name in data_source_str.split(","):
        names.extend(_ALL_META_SOURCES if name == "all" else [name])
    return names


def _load_one_source(name, langs, dataset_config, seed, num_groups):
    """One named source's own bounded group list -- shared by load_groups'
    single- and multi-source paths. `name` is never the literal "all" here --
    _expand_data_sources already expanded it before this is called."""
    if name not in ALL_SOURCES:
        raise ValueError(f"unknown --data-source {name!r} -- choose from {DATA_SOURCES}")
    stream = stream_groups(name, langs=langs, config=dataset_config, seed=seed)
    if name in MONOLINGUAL_SOURCES or name in BITEXT_SOURCES:
        # Lazy/effectively-unbounded source -- bound materialization
        # explicitly rather than draining a live stream to exhaustion.
        limit = num_groups or _DEFAULT_MONOLINGUAL_GROUPS_PER_LANG * max(
            1, len(langs) if isinstance(langs, list) else 1
        )
        return list(itertools.islice(stream, limit))
    return list(stream)


def load_groups(args):
    """args: an argparse.Namespace (or anything with the same attributes) with
    `.langs`, `.data_source`, `.seed`, and `.num_groups` -- every tokenizer's CLI
    in this repo adds these same four flags (see e.g. fairtok/cli.py's
    build_arg_parser), so this function works unmodified against any of them.
    An optional `.dataset_config` attribute (HF config name) is read too, for
    --data-source fineweb_edu/olmo_mix and every BITEXT_SOURCES entry
    (smol/ccmatrix/un_pc/europarl/tatoeba_mt) -- see common.data.corpora.
    stream_groups; every tokenizer's cli.py that wants to expose those sources
    adds that flag, the others simply never read it.

    --data-source may name more than one source (comma-separated) to pool
    them for this run -- see module docstring. --langs/--dataset-config
    aren't supported in that case (raises ValueError) since they aren't
    source-specific; each pooled source just uses its own default.
    """
    if args.langs is None:
        langs = None
    elif args.langs == "all":
        langs = "all"
    else:
        langs = args.langs.split(",")

    dataset_config = getattr(args, "dataset_config", None)
    data_sources = _expand_data_sources(args.data_source)

    if len(data_sources) > 1:
        # "all" (langs="all", dataset_config in (None,"all")) is a no-op here --
        # every pooled source already defaults to "all" languages/pairs on its
        # own now, so an explicit `langs: all` in an existing config (e.g.
        # configs/train_bpe_50k.yml's `data_source: all, langs: all`, predating
        # this uniform default) stays valid rather than breaking. A genuinely
        # RESTRICTIVE override (a specific language list, or a specific
        # --dataset-config value) isn't source-specific, so that still errors.
        if (langs not in (None, "all")) or (dataset_config not in (None, "all")):
            raise ValueError(
                f"--langs/--dataset-config aren't supported when --data-source names more than "
                f"one source ({args.data_source!r}) -- each pooled source uses its own default "
                "('all' languages, uniformly); train on a single --data-source instead to override"
            )
        if "synthetic" in data_sources:
            raise ValueError(
                "--data-source synthetic can't be pooled with other sources -- it's a placeholder "
                "corpus, not real data, so mixing it in wouldn't mean anything"
            )
        groups = []
        for name in data_sources:
            groups.extend(_load_one_source(name, langs=None, dataset_config=None, seed=args.seed, num_groups=None))
        if args.num_groups is not None:
            groups = groups[: args.num_groups]
        return groups

    data_source = data_sources[0]

    if langs == "all" and (
        data_source in ("synthetic", "fineweb_edu", "olmo_mix") or data_source in BITEXT_SOURCES
    ):
        raise ValueError(
            f"--langs all isn't supported for --data-source {data_source} "
            "(synthetic has a fixed toy panel; fineweb_edu/olmo_mix and every BITEXT_SOURCES entry "
            "(smol/ccmatrix/un_pc/europarl/tatoeba_mt) are selected via --dataset-config, not "
            "--langs -- see common.data.corpora)"
        )

    groups = _load_one_source(data_source, langs, dataset_config, args.seed, args.num_groups)
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
