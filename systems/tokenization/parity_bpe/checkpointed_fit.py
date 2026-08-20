"""Checkpoint-aware reimplementation of the OFFICIAL learn_bpe/
learn_bpe_moving_window OUTER LOOP -- reuses their exact sub-functions
(preprocess_input_data, prune_stats, replace_pair, update_pair_statistics,
replace_pair_dict, select_language_index) completely unmodified, straight
from .vendor.parity_aware_learn_bpe, adding ONLY periodic save/resume of the
loop's own live state.

WHY: learn_bpe/learn_bpe_moving_window are single monolithic calls with no
pause/resume hook, and the state a byte-identical resume needs (the live
per-language pair-count arrays, pruning threshold, step counter) only
exists inside their loop bodies -- observing/injecting it means either
modifying vendored code, or writing the loop ourselves reusing their
sub-functions as building blocks. This does the latter, keeping vendor/
untouched; confirmed line-by-line that every operation below is the same
operation in the same order, just with save/load points added (the
`ratio`-based variant is omitted -- this project always has a real dev
set, BOUQuET). model.py's fit_parity_bpe calls the vendored functions
directly when checkpoint_dir is None, and fit_checkpointed here only when
checkpointing is requested; tests/test_parity_bpe.py confirms both paths
produce identical merges given the same input.

PERSISTED STATE (see _save_checkpoint): stats/big_stats (per-pair
per-language frequency counts), threshold (resumed, not recomputed, which
is what makes this byte-identical rather than merely "comparably fair"),
i (the step counter threshold's growth schedule is relative to),
sorted_vocab/dev_vocab/indices (the working structures mutated in place
every step), lengths (per-language dev-vocab token-count estimate),
merge_lines, and selected_indices (moving-window variant only).

PICKLING: stats/big_stats/dev_vocab/indices are defaultdicts built by the
vendored preprocess_input_data with a LAMBDA default_factory, which pickle
can't serialize. An earlier version converted to a plain dict and back
around every save/load -- CONFIRMED a real OOM on a cluster run, since
rebuilding a 75-million-pair `indices` structure (345 languages/515k
sentences) allocates a brand-new dict for every pair at every checkpoint,
sitting in memory alongside the original. Fixed by swapping each
defaultdict's `default_factory` ATTRIBUTE once, right after
preprocess_input_data returns, to a functools.partial/builtin equivalent
pickle can serialize -- an O(1) assignment, not a rebuild; existing data is
untouched and every new key (including via copy.deepcopy mid-loop,
confirmed to preserve a swapped factory) uses the new factory.
"""

import copy
import functools
import os
import pickle
from collections import defaultdict, deque

import numpy

from .vendor import parity_aware_learn_bpe as vendor


def _make_stats_picklable(d, width):
    """Swaps `d`'s default_factory (from vendor.preprocess_input_data, which
    this project doesn't control) for a picklable equivalent -- see module
    docstring. Mutates `d` in place and returns it for chaining."""
    d.default_factory = functools.partial(numpy.zeros, width, dtype=int)
    return d


def _make_indices_picklable(indices):
    """Same fix for `indices` (a NESTED defaultdict(lambda: defaultdict(int)))
    -- only the OUTER factory needs swapping; each INNER defaultdict(int) is
    already directly picklable (int is a builtin, not a lambda)."""
    indices.default_factory = functools.partial(defaultdict, int)
    return indices


def _save_checkpoint(path, state):
    """Atomic write (temp file + os.replace), same convention as every
    other checkpoint in this project (see e.g.
    systems.tokenization.superbpe.model._save_merge_checkpoint)."""
    tmp_path = path + ".tmp"
    with open(tmp_path, "wb") as f:
        pickle.dump(state, f)
    os.replace(tmp_path, path)


def _load_checkpoint(path):
    if not path or not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def fit_checkpointed(
    infiles, devfiles, num_symbols,
    min_frequency=2, verbose=False, num_global=0,
    use_moving_window=False, window_size=100, alpha=2,
    checkpoint_path=None, checkpoint_every=500,
):
    """Learns num_symbols Parity-aware BPE merges. Functionally equivalent
    to calling vendor.learn_bpe (use_moving_window=False) or
    vendor.learn_bpe_moving_window (True) directly -- same algorithm, same
    sub-function calls, in the same order (see module docstring) -- but
    checkpointed: if checkpoint_path already holds a checkpoint from a
    previous call with the SAME infiles/devfiles/config, resumes from
    EXACTLY the state that call left, rather than restarting.

    infiles/devfiles: lists of file-like objects (io.StringIO in this
    project's own usage -- see model.py's fit_parity_bpe). vendor.pre_tokenizer
    must already be set by the caller, same requirement as calling
    vendor.learn_bpe/learn_bpe_moving_window directly.

    Returns list[str] of learned merge lines ("tok1 tok2" per line, NOT
    including a "#version:" header), in learned/priority order.
    """
    num_train_files = len(infiles)
    num_dev_files = len(devfiles)
    config_key = dict(
        num_symbols=num_symbols, num_train_files=num_train_files, num_dev_files=num_dev_files,
        num_global=num_global, use_moving_window=use_moving_window, min_frequency=min_frequency,
        window_size=window_size if use_moving_window else None, alpha=alpha if use_moving_window else None,
    )

    resumed = _load_checkpoint(checkpoint_path)
    if resumed is not None:
        if resumed["config_key"] != config_key:
            raise ValueError(
                f"{checkpoint_path}: checkpoint doesn't match this run "
                f"(saved config {resumed['config_key']!r} != requested {config_key!r}) -- looks "
                "like it belongs to a different corpus or configuration; delete it if you're "
                "intentionally starting a different experiment."
            )
        array_length = resumed["array_length"]
        dev_width = resumed["dev_width"]
        # Already picklable defaultdicts (see module docstring) -- loaded
        # directly, no reconstruction needed, and no duplicate copy either.
        stats = resumed["stats"]
        big_stats = resumed["big_stats"]
        threshold = resumed["threshold"]
        lengths = resumed["lengths"]
        sorted_vocab = resumed["sorted_vocab"]
        dev_vocab = resumed["dev_vocab"]
        indices = resumed["indices"]
        start_i = resumed["i"]
        merge_lines = list(resumed["merge_lines"])
        selected_indices = deque(resumed["selected_indices"], maxlen=window_size) if use_moving_window else None
        if verbose:
            print(f"[parity_bpe] resuming from checkpoint at merge {start_i}/{num_symbols}")
    else:
        dev_vocab, sorted_vocab, stats, indices, big_stats, threshold, lengths, array_length = \
            vendor.preprocess_input_data(
                infiles, devfiles, is_dict=False, total_symbols=False,
                num_global=num_global, num_workers=1, bpe_file=None,
            )
        dev_width = len(lengths)
        _make_stats_picklable(stats, array_length)
        _make_stats_picklable(big_stats, array_length)
        _make_stats_picklable(dev_vocab, dev_width)
        _make_indices_picklable(indices)
        start_i = 0
        merge_lines = []
        selected_indices = deque(maxlen=window_size) if use_moving_window else None

    selection_threshold = (alpha * 1.0 / len(threshold)) if use_moving_window else None

    for i in range(start_i, num_symbols):
        if stats:
            if i < num_global:
                max_index = -1
            elif use_moving_window:
                max_index = vendor.select_language_index(lengths, selected_indices, selection_threshold, window_size)
                selected_indices.append(max_index)
            else:
                max_index = max(range(len(lengths)), key=lambda idx: lengths[idx])
            most_frequent = max(stats, key=lambda x: (stats[x][max_index], x))

        if not stats or (i and stats[most_frequent][max_index] < threshold[max_index]):
            vendor.prune_stats(stats, big_stats, threshold, full_sync=True)
            stats = copy.deepcopy(big_stats)
            most_frequent = max(stats, key=lambda x: (stats[x][max_index], x))
            for l in range(array_length):
                threshold[l] = stats[max(stats, key=lambda x: (stats[x][l], x))][l] * i / (i + 10000.0)
            vendor.prune_stats(stats, big_stats, threshold)

        if stats[most_frequent][max_index] < min_frequency:
            if verbose:
                print(f"[parity_bpe] no pair has frequency >= {min_frequency} -- stopping for language {max_index}")
            break

        if verbose:
            print(f"[parity_bpe] pair {i}: {most_frequent[0]} {most_frequent[1]} -> {''.join(most_frequent)}")

        merge_lines.append("{0} {1}".format(*most_frequent))

        changes = vendor.replace_pair(most_frequent, sorted_vocab, indices)
        length_change = vendor.replace_pair_dict(most_frequent, dev_vocab)
        lengths -= length_change

        vendor.update_pair_statistics(most_frequent, changes, stats, indices)

        if not i % 100:
            vendor.prune_stats(stats, big_stats, threshold)

        stats[most_frequent] = numpy.zeros(array_length, dtype=int)

        # Saved AFTER this step's merge is fully applied -- otherwise a resume
        # would load an `i` one step ahead of what stats/sorted_vocab/etc.
        # actually reflect, corrupting every merge after that (the exact bug
        # class superbpe.model's own checkpoint fix addressed earlier in this
        # project -- avoided here from the start).
        next_i = i + 1
        if checkpoint_path and checkpoint_every and next_i % checkpoint_every == 0:
            # stats/big_stats/dev_vocab/indices pickled DIRECTLY -- already
            # picklable (see module docstring), no plain-dict duplication.
            _save_checkpoint(checkpoint_path, {
                "config_key": config_key,
                "array_length": array_length, "dev_width": dev_width,
                "stats": stats, "big_stats": big_stats,
                "threshold": threshold, "lengths": lengths, "sorted_vocab": sorted_vocab,
                "dev_vocab": dev_vocab, "indices": indices,
                "i": next_i, "merge_lines": merge_lines,
                "selected_indices": list(selected_indices) if use_moving_window else None,
            })

    # Loop ended (full range or an early min_frequency break) -- either way
    # this fit is DONE, nothing left a future resume could continue (same
    # "remove on success" convention as superbpe.model's own checkpoint).
    if checkpoint_path and os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    return merge_lines
