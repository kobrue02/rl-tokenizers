"""End-to-end resume correctness for pretraining.data_prep.prep_dataset. Uses
the deterministic in-memory "synthetic" source and a tiny bpe checkpoint built
on the fly, so this runs fast and offline. A mid-run crash is simulated by
monkeypatching TokenizerAdapter.encode_batch to raise partway through; the run
should leave shards_meta.json absent and prep_checkpoint.json present, exactly
the state a real SLURM time-limit kill leaves."""

import json
import os

import pytest

from pretraining.data_prep import prep_dataset
from pretraining import tokenizer_adapter as tokenizer_adapter_module
from systems.bpe.model import fit_bpe


@pytest.fixture
def bpe_checkpoint(tmp_path):
    sentences = [
        "the quick brown fox jumps over the lazy dog",
        "a small tokenizer trained only for this test",
        "resume support should not lose or duplicate any tokens",
    ]
    model = fit_bpe(sentences, vocab_size=300)
    path = tmp_path / "bpe_checkpoint.json"
    model.tokenizer.save(str(path))
    return str(path)


def _run(output_dir, bpe_checkpoint, max_docs=200):
    return prep_dataset(
        dataset_name="synthetic",
        system="bpe",
        checkpoint_path=bpe_checkpoint,
        output_dir=str(output_dir),
        max_docs=max_docs,
        dedup=False,  # keep assertions exact -- dedup state isn't preserved across a resume boundary
        shard_size=500,
        encode_batch_size=8,
        bucket_pool_multiplier=1,
    )


def test_uninterrupted_run_has_no_leftover_checkpoint(tmp_path, bpe_checkpoint):
    out_dir = tmp_path / "clean"
    meta = _run(out_dir, bpe_checkpoint)
    assert os.path.exists(out_dir / "shards_meta.json")
    assert not os.path.exists(out_dir / "prep_checkpoint.json")
    assert meta["num_docs"] == 200


def test_resume_after_simulated_crash_matches_uninterrupted_run(tmp_path, bpe_checkpoint, monkeypatch):
    meta_full = _run(tmp_path / "full", bpe_checkpoint)

    resumed_dir = tmp_path / "resumed"
    call_count = {"n": 0}
    real_encode_batch = tokenizer_adapter_module.TokenizerAdapter.encode_batch

    def flaky_encode_batch(self, *args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 5:
            raise RuntimeError("simulated crash")
        return real_encode_batch(self, *args, **kwargs)

    with monkeypatch.context() as m:
        m.setattr(tokenizer_adapter_module.TokenizerAdapter, "encode_batch", flaky_encode_batch)
        with pytest.raises(RuntimeError, match="simulated crash"):
            _run(resumed_dir, bpe_checkpoint)

    assert not os.path.exists(resumed_dir / "shards_meta.json")
    assert os.path.exists(resumed_dir / "prep_checkpoint.json")
    with open(resumed_dir / "prep_checkpoint.json") as f:
        ckpt = json.load(f)
    assert 0 < ckpt["total_tokens"] < meta_full["total_tokens"]

    meta_resumed = _run(resumed_dir, bpe_checkpoint)  # no patch this time -- resumes and finishes

    assert os.path.exists(resumed_dir / "shards_meta.json")
    assert not os.path.exists(resumed_dir / "prep_checkpoint.json")
    assert meta_resumed["num_docs"] == meta_full["num_docs"]
    assert meta_resumed["total_tokens"] == meta_full["total_tokens"]
    assert meta_resumed["lang_counts"] == meta_full["lang_counts"]
    for name in meta_resumed["shard_files"]:
        assert os.path.exists(resumed_dir / name)


def test_repeated_crash_resume_cycles_still_match(tmp_path, bpe_checkpoint, monkeypatch):
    """Mirrors the real multi-day scenario (a SLURM job resubmitted across
    several time-limit kills) -- three separate crashes at different points,
    not just one."""
    meta_full = _run(tmp_path / "full", bpe_checkpoint)

    target_dir = tmp_path / "multi_resume"
    real_encode_batch = tokenizer_adapter_module.TokenizerAdapter.encode_batch

    def make_flaky(crash_at_call):
        call_count = {"n": 0}

        def flaky(self, *args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == crash_at_call:
                raise RuntimeError("simulated crash")
            return real_encode_batch(self, *args, **kwargs)

        return flaky

    # Call counts chosen past the first shard flush, so a checkpoint always exists
    # to resume from (an earlier crash would just restart from scratch -- a
    # different, already-correct case, not what this test exercises).
    for crash_at_call in (6, 5, 9):
        with monkeypatch.context() as m:
            m.setattr(tokenizer_adapter_module.TokenizerAdapter, "encode_batch", make_flaky(crash_at_call))
            with pytest.raises(RuntimeError, match="simulated crash"):
                _run(target_dir, bpe_checkpoint)
        assert not os.path.exists(target_dir / "shards_meta.json")
        assert os.path.exists(target_dir / "prep_checkpoint.json")

    meta_final = _run(target_dir, bpe_checkpoint)  # no patch -- finishes for good

    assert os.path.exists(target_dir / "shards_meta.json")
    assert not os.path.exists(target_dir / "prep_checkpoint.json")
    assert not os.path.exists(target_dir / "prep_checkpoint.json.buffer.bin")
    assert meta_final["num_docs"] == meta_full["num_docs"]
    assert meta_final["total_tokens"] == meta_full["total_tokens"]
    assert meta_final["lang_counts"] == meta_full["lang_counts"]
