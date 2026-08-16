"""Resume correctness for systems.tokenization.superbpe.model.fit_superbpe/_fit_merges.
A mid-run crash is simulated by monkeypatching _save_merge_checkpoint to raise right
after a real checkpoint write, leaving sequences+merges-so-far on disk exactly the
state a real SLURM time-limit kill leaves (the process dies before _fit_merges' own
loop-completion cleanup ever runs). Every resumed run is asserted BYTE-FOR-BYTE
identical to an uninterrupted baseline -- the whole point of a deterministic fit."""

import os

import pytest

import systems.tokenization.superbpe.model as sbpe

_SENTENCES = [
    "the quick brown fox jumps over the lazy dog",
    "a small tokenizer trained only for this test",
    "resume support should not lose or duplicate any tokens",
    "hello world this is another sentence for training",
    "byte pair encoding merges the most frequent pairs first",
] * 20
_VOCAB_SIZE = 320


@pytest.fixture
def baseline():
    return sbpe.fit_superbpe(_SENTENCES, _VOCAB_SIZE, verbose=False)


def _crash_after_n_saves(n, path_filter=None):
    """Monkeypatches _save_merge_checkpoint to raise right after the n-th real
    write (optionally only counting writes to a path containing `path_filter`,
    e.g. "stage2" -- lets a test crash specifically mid-stage2 rather than
    stage1). Returns (patch, restore) -- caller must restore in a finally block."""
    real_save = sbpe._save_merge_checkpoint
    count = {"n": 0}

    def flaky_save(path, sequences, merges):
        real_save(path, sequences, merges)
        if path_filter is None or path_filter in path:
            count["n"] += 1
            if count["n"] == n:
                raise RuntimeError("simulated crash")

    def patch():
        sbpe._save_merge_checkpoint = flaky_save

    def restore():
        sbpe._save_merge_checkpoint = real_save

    return patch, restore


def test_uninterrupted_checkpointed_run_matches_baseline_and_cleans_up(tmp_path, baseline):
    result = sbpe.fit_superbpe(_SENTENCES, _VOCAB_SIZE, verbose=False, checkpoint_dir=str(tmp_path), checkpoint_every=5)
    assert result.merges == baseline.merges
    assert result.id_to_bytes == baseline.id_to_bytes
    assert os.listdir(tmp_path) == []


def test_resume_after_crash_mid_stage1_matches_baseline(tmp_path, baseline):
    patch, restore = _crash_after_n_saves(4)
    patch()
    try:
        with pytest.raises(RuntimeError, match="simulated crash"):
            sbpe.fit_superbpe(_SENTENCES, _VOCAB_SIZE, verbose=False, checkpoint_dir=str(tmp_path), checkpoint_every=5)
    finally:
        restore()
    assert "stage1_checkpoint.json" in os.listdir(tmp_path)

    resumed = sbpe.fit_superbpe(_SENTENCES, _VOCAB_SIZE, verbose=False, checkpoint_dir=str(tmp_path), checkpoint_every=5)
    assert os.listdir(tmp_path) == []
    assert resumed.merges == baseline.merges
    assert resumed.id_to_bytes == baseline.id_to_bytes
    assert resumed.num_stage1_merges == baseline.num_stage1_merges


def test_resume_after_crash_mid_stage2_matches_baseline(tmp_path, baseline):
    patch, restore = _crash_after_n_saves(2, path_filter="stage2")
    patch()
    try:
        with pytest.raises(RuntimeError, match="simulated crash"):
            sbpe.fit_superbpe(_SENTENCES, _VOCAB_SIZE, verbose=False, checkpoint_dir=str(tmp_path), checkpoint_every=3)
    finally:
        restore()
    # Stage 1 must have finished (and cleaned up its own checkpoint) before stage 2
    # ever starts -- only stage2's checkpoint should survive the crash.
    assert os.listdir(tmp_path) == ["stage2_checkpoint.json"]

    resumed = sbpe.fit_superbpe(_SENTENCES, _VOCAB_SIZE, verbose=False, checkpoint_dir=str(tmp_path), checkpoint_every=3)
    assert os.listdir(tmp_path) == []
    assert resumed.merges == baseline.merges
    assert resumed.id_to_bytes == baseline.id_to_bytes


def test_mismatched_checkpoint_raises_instead_of_silently_corrupting(tmp_path):
    """Reusing a --checkpoint-dir across two different corpora/vocab-sizes must
    fail loudly, not silently fit garbage from a corpus that no longer matches."""
    other_sentences = ["a completely different corpus with its own unique words"] * 5
    sbpe.fit_superbpe(other_sentences, 300, verbose=False, checkpoint_dir=str(tmp_path), checkpoint_every=1000)  # no-op: finishes clean, nothing left to collide

    patch, restore = _crash_after_n_saves(2)
    patch()
    try:
        with pytest.raises(RuntimeError, match="simulated crash"):
            sbpe.fit_superbpe(other_sentences, 300, verbose=False, checkpoint_dir=str(tmp_path), checkpoint_every=2)
    finally:
        restore()
    assert os.listdir(tmp_path) != []

    with pytest.raises(ValueError, match="doesn't match this run"):
        sbpe.fit_superbpe(_SENTENCES, _VOCAB_SIZE, verbose=False, checkpoint_dir=str(tmp_path), checkpoint_every=5)


def test_repeated_crash_resume_cycles_still_match_baseline(tmp_path, baseline):
    """Mirrors test_data_prep.py's own multi-crash test -- several separate
    crashes at different points, not just one, same as a job resubmitted across
    several real time-limit kills."""
    for n in (3, 2, 4):
        patch, restore = _crash_after_n_saves(n)
        patch()
        try:
            with pytest.raises(RuntimeError, match="simulated crash"):
                sbpe.fit_superbpe(_SENTENCES, _VOCAB_SIZE, verbose=False, checkpoint_dir=str(tmp_path), checkpoint_every=4)
        finally:
            restore()
        assert os.listdir(tmp_path) != []

    final = sbpe.fit_superbpe(_SENTENCES, _VOCAB_SIZE, verbose=False, checkpoint_dir=str(tmp_path), checkpoint_every=4)
    assert os.listdir(tmp_path) == []
    assert final.merges == baseline.merges
    assert final.id_to_bytes == baseline.id_to_bytes
