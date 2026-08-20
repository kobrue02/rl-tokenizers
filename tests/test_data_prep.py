"""End-to-end resume correctness for systems.pretraining.data_prep.prep_dataset. Uses
the deterministic in-memory "synthetic" source and a tiny bpe checkpoint built
on the fly, so this runs fast and offline. A mid-run crash is simulated by
monkeypatching TokenizerAdapter.encode_batch to raise partway through; the run
should leave shards_meta.json absent and prep_checkpoint.json present, exactly
the state a real SLURM time-limit kill leaves."""

import json
import os

import pytest

from systems.pretraining.data_prep import prep_dataset
from systems.pretraining import tokenizer_adapter as tokenizer_adapter_module
from systems.tokenization.bpe.model import fit_bpe


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


def _run(output_dir, bpe_checkpoint, max_docs=200, prefetch=False, prefetch_queue_size=None):
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
        prefetch=prefetch,
        prefetch_queue_size=prefetch_queue_size,
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


def test_prefetch_matches_non_prefetch_output(tmp_path, bpe_checkpoint):
    """--prefetch changes ONLY when documents arrive at the consumer, never
    their order or content (see PREFETCH docstring section in data_prep.py)
    -- the two runs must produce byte-identical accounting."""
    meta_plain = _run(tmp_path / "plain", bpe_checkpoint)
    meta_prefetch = _run(tmp_path / "prefetch", bpe_checkpoint, prefetch=True, prefetch_queue_size=3)

    assert meta_prefetch["total_tokens"] == meta_plain["total_tokens"]
    assert meta_prefetch["num_docs"] == meta_plain["num_docs"]
    assert meta_prefetch["lang_counts"] == meta_plain["lang_counts"]
    assert meta_prefetch["lang_token_gini"] == meta_plain["lang_token_gini"]
    assert meta_prefetch["shard_files"] == meta_plain["shard_files"]
    for name in meta_prefetch["shard_files"]:
        with open(tmp_path / "plain" / name, "rb") as f_plain, open(tmp_path / "prefetch" / name, "rb") as f_pre:
            assert f_plain.read() == f_pre.read()


def test_prefetch_resume_after_simulated_crash_matches_uninterrupted_run(tmp_path, bpe_checkpoint, monkeypatch):
    """Mirrors test_resume_after_simulated_crash_matches_uninterrupted_run
    above, with --prefetch on -- the producer thread must not disturb the
    RESUME fast-forward invariant (same stream order, same
    stream_docs_consumed semantics -- see _document_source's docstring)."""
    meta_full = _run(tmp_path / "full", bpe_checkpoint, prefetch=True)

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
            _run(resumed_dir, bpe_checkpoint, prefetch=True)

    assert not os.path.exists(resumed_dir / "shards_meta.json")
    assert os.path.exists(resumed_dir / "prep_checkpoint.json")

    meta_resumed = _run(resumed_dir, bpe_checkpoint, prefetch=True)  # no patch -- resumes and finishes

    assert os.path.exists(resumed_dir / "shards_meta.json")
    assert not os.path.exists(resumed_dir / "prep_checkpoint.json")
    assert meta_resumed["num_docs"] == meta_full["num_docs"]
    assert meta_resumed["total_tokens"] == meta_full["total_tokens"]
    assert meta_resumed["lang_counts"] == meta_full["lang_counts"]
    for name in meta_resumed["shard_files"]:
        assert os.path.exists(resumed_dir / name)


def test_prefetch_stream_exception_propagates(tmp_path, bpe_checkpoint, monkeypatch):
    """A producer-thread exception (e.g. the underlying stream itself
    raising, not just encode_batch) must surface in the main thread as a
    real exception, not vanish silently or hang -- see
    _prefetch_source/_produce_stream's own docstrings."""
    from systems.pretraining import data_prep as data_prep_module

    def broken_stream(*args, **kwargs):
        yield {"en": "first ok document one two three"}
        yield {"en": "second ok document four five six"}
        raise RuntimeError("simulated stream failure")

    monkeypatch.setattr(data_prep_module, "stream_groups", lambda *a, **k: broken_stream())

    with pytest.raises(RuntimeError, match="simulated stream failure"):
        prep_dataset(
            dataset_name="synthetic",
            system="bpe",
            checkpoint_path=bpe_checkpoint,
            output_dir=str(tmp_path / "err"),
            dedup=False,
            shard_size=500,
            encode_batch_size=8,
            bucket_pool_multiplier=1,
            prefetch=True,
        )


def test_prep_dataset_reads_glot500_from_local_cache(tmp_path, bpe_checkpoint):
    """Integration check for the glot500 local-disk-cache read path (see
    common/data/prepare_glot500.py and common/data/corpora.py's rewritten
    glot500 branch in stream_groups) -- prep_dataset's own --dataset-config
    (dataset_config) must reach stream_groups as the local cache dir
    override, entirely offline (no network, no live HF streaming)."""
    cache_dir = tmp_path / "glot500_cache"
    cache_dir.mkdir()
    with open(cache_dir / "eng_Latn.jsonl", "w", encoding="utf-8") as f:
        for text in ["the quick brown fox", "jumps over the lazy dog", "one more english document"]:
            f.write(json.dumps({"eng_Latn": text}) + "\n")
    with open(cache_dir / "fra_Latn.jsonl", "w", encoding="utf-8") as f:
        f.write(json.dumps({"fra_Latn": "bonjour le monde entier"}) + "\n")

    meta = prep_dataset(
        dataset_name="glot500",
        langs=["eng_Latn", "fra_Latn"],
        dataset_config=str(cache_dir),
        system="bpe",
        checkpoint_path=bpe_checkpoint,
        output_dir=str(tmp_path / "out"),
        dedup=False,
        shard_size=500,
        encode_batch_size=8,
        bucket_pool_multiplier=1,
    )

    assert set(meta["lang_counts"]) == {"eng_Latn", "fra_Latn"}
    assert meta["lang_counts"]["eng_Latn"]["docs"] == 3
    assert meta["lang_counts"]["fra_Latn"]["docs"] == 1
    assert meta["num_docs"] == 4
