"""Tests for systems.tokenization.parity_bpe.checkpointed_fit -- the
reimplemented OUTER LOOP that makes Parity-aware BPE fitting resumable (see
that module's own docstring for why this exists rather than modifying the
vendored learn_bpe/learn_bpe_moving_window directly). Mirrors
tests/test_superbpe_resume.py's own crash-simulation style for that same
class of feature: a mid-run crash is simulated by monkeypatching
_save_checkpoint to raise right after a real checkpoint write, leaving
exactly the state a real SLURM time-limit kill leaves. Every resumed run is
asserted to produce the EXACT SAME merges as an uninterrupted baseline --
not merely a comparably fair result -- since that's the whole point of this
reimplementation over the simpler chunk-and-reload-via-bpe_file approach."""

import io
import os

import pytest
from tokenizers.pre_tokenizers import ByteLevel, Sequence, Whitespace

import systems.tokenization.parity_bpe.checkpointed_fit as ckpt_fit
from systems.tokenization.parity_bpe.model import fit_parity_bpe
from systems.tokenization.parity_bpe.vendor import parity_aware_learn_bpe as vendor

_TRAIN = {
    "eng": ["the quick brown fox jumps over the lazy dog"] * 20,
    "deu": ["der schnelle braune fuchs springt ueber den faulen hund"] * 20,
}
_DEV = {
    "eng": ["the quick brown fox"] * 10,
    "deu": ["der schnelle braune fuchs"] * 10,
}
_LANGS = sorted(_TRAIN)


def _make_files(d):
    return [io.StringIO("\n".join(d[l]) + "\n") for l in _LANGS]


@pytest.fixture(autouse=True)
def _set_pre_tokenizer():
    vendor.pre_tokenizer = Sequence([Whitespace(), ByteLevel(use_regex=False)])


@pytest.fixture
def baseline():
    return ckpt_fit.fit_checkpointed(_make_files(_TRAIN), _make_files(_DEV), 40, min_frequency=2)


def test_make_picklable_helpers_mutate_in_place_not_copy():
    """Regression test for a real, confirmed OOM: an earlier version of this
    module converted stats/big_stats/dev_vocab/indices to brand-new plain
    dicts at every checkpoint save (to work around pickle's own inability to
    serialize a lambda-based defaultdict factory) -- at real corpus scale
    (hundreds of thousands of pairs), that duplicated the entire structure
    into memory at every checkpoint interval. The fix swaps each
    defaultdict's OWN default_factory attribute in place instead -- confirms
    that here directly: the returned object must be the SAME object
    (identical id()), never a copy, and must still round-trip through
    pickle correctly."""
    import pickle
    from collections import defaultdict

    stats = defaultdict(lambda: __import__("numpy").zeros(3, dtype=int))
    stats[(1, 2)] += 1
    result = ckpt_fit._make_stats_picklable(stats, 3)
    assert result is stats  # same object, not a copy
    pickle.loads(pickle.dumps(stats))  # must not raise

    indices = defaultdict(lambda: defaultdict(int))
    indices[(1, 2)][0] += 1
    result = ckpt_fit._make_indices_picklable(indices)
    assert result is indices
    pickle.loads(pickle.dumps(indices))


def test_fit_checkpointed_matches_vendor_learn_bpe_directly():
    """Proves the reimplemented loop is a faithful equivalent, not an
    approximation: given identical input, it must produce EXACTLY the same
    merges as calling the official vendor.learn_bpe directly."""
    outfile = io.StringIO()
    vendor.learn_bpe(_make_files(_TRAIN), outfile, _make_files(_DEV), 40, min_frequency=2)
    direct = [line for line in outfile.getvalue().splitlines() if line and not line.startswith("#version")]

    mine = ckpt_fit.fit_checkpointed(_make_files(_TRAIN), _make_files(_DEV), 40, min_frequency=2)
    assert mine == direct


def test_fit_checkpointed_moving_window_matches_vendor_directly():
    outfile = io.StringIO()
    vendor.learn_bpe_moving_window(_make_files(_TRAIN), outfile, _make_files(_DEV), 40, window_size=5, alpha=1, min_frequency=2)
    direct = [line for line in outfile.getvalue().splitlines() if line and not line.startswith("#version")]

    mine = ckpt_fit.fit_checkpointed(
        _make_files(_TRAIN), _make_files(_DEV), 40, min_frequency=2,
        use_moving_window=True, window_size=5, alpha=1,
    )
    assert mine == direct


def test_uninterrupted_checkpointed_run_matches_baseline_and_cleans_up(tmp_path, baseline):
    ckpt_path = str(tmp_path / "ckpt.pkl")
    result = ckpt_fit.fit_checkpointed(
        _make_files(_TRAIN), _make_files(_DEV), 40, min_frequency=2,
        checkpoint_path=ckpt_path, checkpoint_every=5,
    )
    assert result == baseline
    assert not os.path.exists(ckpt_path)


def _crash_after_n_saves(n):
    """Monkeypatches _save_checkpoint to raise right after the n-th real
    write -- simulates a real SLURM time-limit kill (the process dies AFTER
    a checkpoint write completes, not mid-write)."""
    real_save = ckpt_fit._save_checkpoint
    count = {"n": 0}

    def flaky_save(path, state):
        real_save(path, state)
        count["n"] += 1
        if count["n"] == n:
            raise RuntimeError("simulated crash")

    def patch():
        ckpt_fit._save_checkpoint = flaky_save

    def restore():
        ckpt_fit._save_checkpoint = real_save

    return patch, restore


def test_resume_after_crash_matches_baseline(tmp_path, baseline):
    ckpt_path = str(tmp_path / "ckpt.pkl")
    patch, restore = _crash_after_n_saves(2)
    patch()
    try:
        with pytest.raises(RuntimeError, match="simulated crash"):
            ckpt_fit.fit_checkpointed(
                _make_files(_TRAIN), _make_files(_DEV), 40, min_frequency=2,
                checkpoint_path=ckpt_path, checkpoint_every=5,
            )
    finally:
        restore()
    assert os.path.exists(ckpt_path)

    resumed = ckpt_fit.fit_checkpointed(
        _make_files(_TRAIN), _make_files(_DEV), 40, min_frequency=2,
        checkpoint_path=ckpt_path, checkpoint_every=5,
    )
    assert not os.path.exists(ckpt_path)
    assert resumed == baseline


def test_repeated_crash_resume_cycles_still_match_baseline(tmp_path, baseline):
    """Mirrors test_superbpe_resume.py's own multi-crash test -- several
    separate crashes at different points, same as a job resubmitted across
    several real time-limit kills."""
    ckpt_path = str(tmp_path / "ckpt.pkl")
    for n in (3, 2, 4):
        patch, restore = _crash_after_n_saves(n)
        patch()
        try:
            with pytest.raises(RuntimeError, match="simulated crash"):
                ckpt_fit.fit_checkpointed(
                    _make_files(_TRAIN), _make_files(_DEV), 40, min_frequency=2,
                    checkpoint_path=ckpt_path, checkpoint_every=3,
                )
        finally:
            restore()
        assert os.path.exists(ckpt_path)

    final = ckpt_fit.fit_checkpointed(
        _make_files(_TRAIN), _make_files(_DEV), 40, min_frequency=2,
        checkpoint_path=ckpt_path, checkpoint_every=3,
    )
    assert not os.path.exists(ckpt_path)
    assert final == baseline


def test_moving_window_resume_matches_baseline(tmp_path):
    """The moving-window variant's own extra state (selected_indices, the
    recent-language-selection deque) must ALSO survive a crash/resume --
    otherwise a resumed run could re-select a language the window would
    have excluded had it never been interrupted."""
    baseline_window = ckpt_fit.fit_checkpointed(
        _make_files(_TRAIN), _make_files(_DEV), 40, min_frequency=2,
        use_moving_window=True, window_size=5, alpha=1,
    )

    ckpt_path = str(tmp_path / "ckpt.pkl")
    patch, restore = _crash_after_n_saves(2)
    patch()
    try:
        with pytest.raises(RuntimeError, match="simulated crash"):
            ckpt_fit.fit_checkpointed(
                _make_files(_TRAIN), _make_files(_DEV), 40, min_frequency=2,
                use_moving_window=True, window_size=5, alpha=1,
                checkpoint_path=ckpt_path, checkpoint_every=5,
            )
    finally:
        restore()
    assert os.path.exists(ckpt_path)

    resumed = ckpt_fit.fit_checkpointed(
        _make_files(_TRAIN), _make_files(_DEV), 40, min_frequency=2,
        use_moving_window=True, window_size=5, alpha=1,
        checkpoint_path=ckpt_path, checkpoint_every=5,
    )
    assert resumed == baseline_window


def test_mismatched_checkpoint_raises_instead_of_silently_corrupting(tmp_path):
    """Reusing a checkpoint_dir across two different corpora/configs must
    fail loudly, not silently fit garbage from a corpus that no longer
    matches -- same convention as superbpe's own checkpoint mismatch check."""
    other_train = {"fra": ["une phrase completement differente avec ses propres mots uniques"] * 5}
    other_dev = {"fra": ["une phrase completement differente"] * 5}
    other_langs = sorted(other_train)

    def make_other_files(d):
        return [io.StringIO("\n".join(d[l]) + "\n") for l in other_langs]

    ckpt_path = str(tmp_path / "ckpt.pkl")
    patch, restore = _crash_after_n_saves(1)
    patch()
    try:
        with pytest.raises(RuntimeError, match="simulated crash"):
            ckpt_fit.fit_checkpointed(
                make_other_files(other_train), make_other_files(other_dev), 20, min_frequency=2,
                checkpoint_path=ckpt_path, checkpoint_every=1,
            )
    finally:
        restore()
    assert os.path.exists(ckpt_path)

    with pytest.raises(ValueError, match="doesn't match this run"):
        ckpt_fit.fit_checkpointed(
            _make_files(_TRAIN), _make_files(_DEV), 40, min_frequency=2,
            checkpoint_path=ckpt_path, checkpoint_every=5,
        )


def test_fit_parity_bpe_with_checkpoint_dir_matches_without(tmp_path):
    """End-to-end (through the dict-of-sentences fit_parity_bpe API, not
    checkpointed_fit directly): a checkpointed fit must produce a model
    functionally equivalent to the default (non-checkpointed, calls
    vendor.learn_bpe directly) path -- same merges learned, same encoding
    behavior. NOT compared via raw tokenizer.get_vocab() equality: HF's own
    ByteLevel.alphabet() returns its 256 characters in a NON-deterministic
    order every call (confirmed live -- two calls in the same process gave
    different orderings), so build_vocab_from_merges assigns a different,
    but internally self-consistent, byte<->id mapping every fit_parity_bpe
    call regardless of checkpointing. Harmless for correctness (encode_spans
    decodes via common.vocab.BYTE_TO_UNICODE's own FIXED table, never the
    tokenizer's internal ids), but means comparing get_vocab() dicts across
    two separate fits is the wrong test -- compare functional encoding
    behavior and merge count instead."""
    model_default = fit_parity_bpe(_TRAIN, _DEV, vocab_size=296, min_frequency=2)
    model_checkpointed = fit_parity_bpe(
        _TRAIN, _DEV, vocab_size=296, min_frequency=2,
        checkpoint_dir=str(tmp_path), checkpoint_every=5,
    )
    assert model_default.num_parameters() == model_checkpointed.num_parameters()
    for text in ["the quick brown fox", "der schnelle braune fuchs", "a totally unseen sentence"]:
        assert model_default.encode_spans(text) == model_checkpointed.encode_spans(text)
    assert not os.listdir(tmp_path)
