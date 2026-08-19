"""Checkpoint-aware reimplementation of the OFFICIAL learn_bpe/
learn_bpe_moving_window OUTER LOOP -- reuses their exact sub-functions
(preprocess_input_data, prune_stats, replace_pair, update_pair_statistics,
replace_pair_dict, select_language_index) completely unmodified, straight
from .vendor.parity_aware_learn_bpe, adding ONLY periodic save/resume of the
loop's own live state.

WHY THIS EXISTS, RATHER THAN A MODIFICATION TO vendor/parity_aware_learn_bpe.py:
learn_bpe/learn_bpe_moving_window are single monolithic calls with no
pause/resume hook of any kind (see that module's own docstring). The state
a genuinely BYTE-IDENTICAL resume needs -- the live per-language pair-count
arrays (stats/big_stats), the pruning threshold, and the loop's own step
counter -- only exists INSIDE their loop bodies; there is no way to
observe or inject it from outside without either (a) adding save/load hooks
inside their function bodies, or (b) writing the loop ourselves, reusing
their existing sub-functions as building blocks. (a) would mean modifying
vendored code; this file does (b) instead, keeping vendor/ completely
untouched. Confirmed line-by-line against learn_bpe/learn_bpe_moving_window
themselves: every operation below is the SAME operation in the SAME order,
just with save/load points added and (per model.py's own fit_parity_bpe
docstring) the `ratio`-based variant omitted entirely, since this project
never uses it (a real dev set, BOUQuET, is always available).

model.py's fit_parity_bpe calls .vendor.parity_aware_learn_bpe.learn_bpe/
learn_bpe_moving_window DIRECTLY (maximum reuse of the official
implementation) whenever checkpoint_dir is None; it calls fit_checkpointed
here ONLY when checkpointing is actually requested. tests/test_parity_bpe.py
confirms both paths produce IDENTICAL merges given the same input, so this
reimplementation is a faithful equivalent, not an approximation.

WHAT GETS PERSISTED, AND WHY EACH PIECE IS NEEDED (see _save_checkpoint):
  - stats, big_stats: per-pair, per-language running frequency counts
    (numpy arrays) -- the actual counting state a resume must continue from.
  - threshold: current per-language pruning threshold -- resuming this
    (rather than recomputing it fresh, which would use a wrong, chunk-local
    step count) is exactly what makes this resume byte-identical rather
    than merely "comparably fair" (see checkpointed_fit design discussion).
  - i: the loop's own step counter, since threshold's own growth schedule
    (see learn_bpe's "* i/(i+10000.0)") is relative to it.
  - sorted_vocab, dev_vocab, indices: the working per-word/per-pair
    structures replace_pair/update_pair_statistics/replace_pair_dict mutate
    in place every step.
  - lengths: current per-language dev-vocab token-count estimate (what
    picks the "worst-compressed" language every step).
  - merge_lines: every merge learned so far, in order.
  - selected_indices (moving-window variant only): the recent-language-
    selection history the window-exclusion rule depends on.

Several of these are defaultdicts with a LAMBDA default factory
(stats/big_stats/dev_vocab: lambda: numpy.zeros(N); indices: lambda:
defaultdict(int)) -- Python's pickle can't serialize a lambda, so
_to_plain/_from_plain convert to/from plain dicts around every save/load,
reconstructing the same factory (with the right dimensionality, itself
persisted) on load.
"""

import copy
import os
import pickle
from collections import defaultdict, deque

import numpy

from .vendor import parity_aware_learn_bpe as vendor


def _stats_to_plain(d):
    return dict(d)


def _stats_from_plain(plain, width):
    d = defaultdict(lambda: numpy.zeros(width, dtype=int))
    d.update(plain)
    return d


def _indices_to_plain(indices):
    return {pair: dict(inner) for pair, inner in indices.items()}


def _indices_from_plain(plain):
    d = defaultdict(lambda: defaultdict(int))
    for pair, inner in plain.items():
        inner_d = defaultdict(int)
        inner_d.update(inner)
        d[pair] = inner_d
    return d


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
        stats = _stats_from_plain(resumed["stats"], array_length)
        big_stats = _stats_from_plain(resumed["big_stats"], array_length)
        threshold = resumed["threshold"]
        lengths = resumed["lengths"]
        sorted_vocab = resumed["sorted_vocab"]
        dev_vocab = _stats_from_plain(resumed["dev_vocab"], dev_width)
        indices = _indices_from_plain(resumed["indices"])
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
            _save_checkpoint(checkpoint_path, {
                "config_key": config_key,
                "array_length": array_length, "dev_width": dev_width,
                "stats": _stats_to_plain(stats), "big_stats": _stats_to_plain(big_stats),
                "threshold": threshold, "lengths": lengths, "sorted_vocab": sorted_vocab,
                "dev_vocab": _stats_to_plain(dev_vocab), "indices": _indices_to_plain(indices),
                "i": next_i, "merge_lines": merge_lines,
                "selected_indices": list(selected_indices) if use_moving_window else None,
            })

    # Loop ended (full range or an early min_frequency break) -- either way
    # this fit is DONE, nothing left a future resume could continue (same
    # "remove on success" convention as superbpe.model's own checkpoint).
    if checkpoint_path and os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    return merge_lines
