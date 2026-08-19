"""Tests for systems.tokenization.parity_bpe.model, the adapter layer around
the OFFICIAL Parity-aware BPE implementation (vendored in .vendor -- see
that module's own docstring). Since the actual fitting algorithm is reused
directly rather than reimplemented, these tests focus on what this project's
OWN code is responsible for: converting {lang: text} group data into the
vendored functions' expected in-memory file-object shape and back into a
usable ParityBPEModel, correctly handling train/dev language mismatches, and
wiring config knobs (hybrid/window/min_frequency) through to the underlying
calls -- not re-deriving the paper's own fairness algorithm's correctness,
which is the vendored code's own responsibility."""

import io

import pytest
from tokenizers.pre_tokenizers import ByteLevel, Sequence, Whitespace

import systems.tokenization.parity_bpe.model as pbm
from systems.tokenization.parity_bpe.model import ParityBPEModel, fit_parity_bpe
from systems.tokenization.parity_bpe.train import ParityBPEConfig, ParityBPETrainer
from systems.tokenization.parity_bpe.vendor import parity_aware_learn_bpe as vendor


def _learn_merges(train_by_lang, dev_by_lang, num_symbols, **kwargs):
    """Calls the vendored learn_bpe DIRECTLY (bypassing fit_parity_bpe) so a
    test can inspect the raw learned merge order -- mirrors exactly what
    model.fit_parity_bpe itself does, used here to confirm OUR data-shaping
    reaches the official function correctly, not to re-verify its own
    algorithm's fairness guarantees (that's the vendored code's own concern)."""
    vendor.pre_tokenizer = Sequence([Whitespace(), ByteLevel(use_regex=False)])
    langs = sorted(train_by_lang)
    infiles = [io.StringIO("\n".join(train_by_lang[l]) + "\n") for l in langs]
    devfiles = [io.StringIO("\n".join(dev_by_lang[l]) + "\n") for l in langs]
    outfile = io.StringIO()
    vendor.learn_bpe(infiles, outfile, devfiles, num_symbols, **kwargs)
    lines = [line for line in outfile.getvalue().splitlines() if line and not line.startswith("#version")]
    return lines, langs


def test_normalize_lang_code_handles_both_conventions():
    """Regression test for a real bug hit on a live cluster run:
    --data-source all pools oldi_seed+flores_dev (BOUQuET-convention
    lang_Script codes) with smol (short/hyphenated codes) -- without
    normalization, 195 of 345 training languages were silently excluded
    from the fit entirely because "en" != "eng_Latn" as raw dict keys, even
    though they're the same language."""
    cases = [
        ("en", "eng"), ("eng_Latn", "eng"),
        ("ar-MA", "ara"), ("pa-Arab", "pan"), ("ks-Deva", "kas"),
        ("apc_Arab_nort3139", "apc"), ("twi_Latn_akua1239", "twi"),
        ("aar_Latn", "aar"),
    ]
    for raw, expected in cases:
        assert pbm._normalize_lang_code(raw) == expected, f"{raw!r} should normalize to {expected!r}"


def test_normalize_lang_code_falls_back_to_raw_on_parse_failure():
    """langcodes is lenient about underscore/hyphen-separated ALPHANUMERIC
    garbage (it just treats the first segment as the language subtag and
    normalizes that, e.g. "not_a_real_..." -> "not"), but genuinely
    malformed strings (containing non-alphanumeric characters, empty, or
    all-numeric) raise LanguageTagError -- confirmed live. The fallback
    exists for exactly those, so this project's own code-normalization step
    can never crash the whole fit over one weird language code."""
    for garbage in ("???", "", "12345"):
        assert pbm._normalize_lang_code(garbage) == garbage


def test_short_code_matches_lang_script_dev_code(capsys):
    """The exact scenario from the real cluster bug: train uses smol's own
    short code ("en"), dev uses BOUQuET's lang_Script convention
    ("eng_Latn") -- these must be recognized as the same language and BOTH
    used, not dropped as unrelated train-only/dev-only languages."""
    train = {"en": ["hello world hello world"] * 20}
    dev = {"eng_Latn": ["hello world"] * 5}
    model = fit_parity_bpe(train, dev, vocab_size=280)
    assert model.num_parameters() > 256
    out = capsys.readouterr().out
    assert "have no corresponding training data" not in out
    assert "have no dev-set compression" not in out


def test_merged_raw_codes_pool_sentences_together(capsys):
    """If a corpus somehow has BOTH a short code and a lang_Script code for
    the same language as separate keys, their sentences must be POOLED
    together under the normalized key, not arbitrarily picking one and
    discarding the other's data."""
    train = {"en": ["hello world"] * 10, "eng_Latn": ["goodbye world"] * 10}
    dev = {"eng_Latn": ["hello world"] * 5}
    model = fit_parity_bpe(train, dev, vocab_size=280)
    assert model.num_parameters() > 256
    out = capsys.readouterr().out
    assert "language-code normalization merged multiple raw codes" in out
    assert "train" in out


def test_vendored_learn_bpe_is_reachable_and_fairness_restricted():
    """Sanity check that our data-shaping (io.StringIO per language, sorted
    order, pre_tokenizer set) correctly reaches the real vendored learn_bpe:
    'rich' dominates raw frequency but 'poor' is worse-compressed in dev, so
    the very first learned merge should come from 'poor's own vocabulary,
    not 'rich's globally-dominant one -- confirms the official fair-max
    objective is actually engaged, not silently falling back to plain
    frequency counting due to a data-shape mismatch on our end."""
    train = {"rich": ["aaaa aaaa aaaa aaaa"] * 50, "poor": ["xyxy yzyz zxzx"] * 5}
    dev = {"rich": ["aaaa aaaa"] * 5, "poor": ["xyxy yzyz"] * 5}
    lines, _ = _learn_merges(train, dev, num_symbols=1)
    assert len(lines) == 1
    tok1, tok2 = lines[0].split()
    assert "a" not in tok1 and "a" not in tok2, (
        f"expected the first merge to come from 'poor's own data, got {lines[0]!r} -- "
        "looks like classical global frequency won instead of the parity-aware objective"
    )


def test_hybrid_num_global_merges_uses_classical_frequency_first():
    train = {"rich": ["aaaa aaaa aaaa aaaa"] * 50, "poor": ["xyxy yzyz zxzx"] * 5}
    dev = {"rich": ["aaaa aaaa"] * 5, "poor": ["xyxy yzyz"] * 5}
    lines, _ = _learn_merges(train, dev, num_symbols=1, num_global=1)
    tok1, tok2 = lines[0].split()
    assert tok1 == "a" and tok2 == "a"


def test_fit_parity_bpe_returns_working_model():
    train = {"eng": ["hello world hello world"] * 20, "deu": ["hallo welt hallo welt"] * 20}
    dev = {"eng": ["hello world"] * 5, "deu": ["hallo welt"] * 5}
    model = fit_parity_bpe(train, dev, vocab_size=280)
    assert isinstance(model, ParityBPEModel)
    assert model.num_parameters() >= 256
    spans = model.encode_spans("hello world")
    assert b"".join(spans) == b"hello world"


def test_fit_parity_bpe_hybrid_and_window_flags_do_not_crash():
    train = {"eng": ["hello world hello world"] * 20, "deu": ["hallo welt hallo welt"] * 20}
    dev = {"eng": ["hello world"] * 5, "deu": ["hallo welt"] * 5}
    model_hybrid = fit_parity_bpe(train, dev, vocab_size=270, num_global_merges=5)
    model_window = fit_parity_bpe(train, dev, vocab_size=270, use_moving_window=True, window_size=10, alpha=1)
    assert model_hybrid.num_parameters() >= 256
    assert model_window.num_parameters() >= 256


def test_dev_only_language_is_dropped_not_crashing(capsys):
    train = {"eng": ["hello world"] * 10}
    dev = {"eng": ["hello world"] * 5, "unrelated": ["zzzz zzzz"] * 5}
    model = fit_parity_bpe(train, dev, vocab_size=270)
    assert model.num_parameters() >= 256
    assert "unrelated" in capsys.readouterr().out


def test_train_only_language_is_dropped_not_crashing(capsys):
    train = {"eng": ["hello world"] * 10, "deu": ["hallo welt"] * 10}
    dev = {"eng": ["hello world"] * 5}  # 'deu' has no dev signal at all
    model = fit_parity_bpe(train, dev, vocab_size=270)
    assert model.num_parameters() >= 256
    assert "deu" in capsys.readouterr().out


def test_raises_when_train_and_dev_share_no_language():
    with pytest.raises(ValueError, match="no language appears in BOTH"):
        fit_parity_bpe({"eng": ["hello"] * 5}, {"deu": ["hallo"] * 5}, vocab_size=270)


def test_parity_bpe_model_round_trips_bytes():
    train = {"eng": ["the quick brown fox jumps over the lazy dog"] * 20}
    dev = {"eng": ["the quick brown fox"] * 5}
    model = fit_parity_bpe(train, dev, vocab_size=300)
    for text in ["the quick brown fox", "a completely unseen sentence with new words"]:
        spans = model.encode_spans(text)
        assert b"".join(spans) == text.encode("utf-8")


def test_trainer_requires_nonempty_eval_groups():
    """ParityBPETrainer's dev set drives the fit ITSELF (see the module's own
    docstring), unlike every other tokenizer's optional eval_groups -- must
    raise a clear, actionable error rather than crash deep inside
    fit_parity_bpe or silently proceed with no fairness signal at all."""
    cfg = ParityBPEConfig(vocab_size=280)
    train_groups = [{"eng": "hello world"}]
    trainer = ParityBPETrainer(cfg, train_groups, eval_groups=None)
    with pytest.raises(ValueError, match="non-empty dev set"):
        trainer.train()

    trainer_empty = ParityBPETrainer(cfg, train_groups, eval_groups=[])
    with pytest.raises(ValueError, match="non-empty dev set"):
        trainer_empty.train()


def test_trainer_end_to_end_sets_model_token_freq_vocab():
    cfg = ParityBPEConfig(vocab_size=280)
    groups = [
        {"eng": "the quick brown fox", "deu": "der schnelle braune fuchs"},
    ] * 20
    trainer = ParityBPETrainer(cfg, groups, eval_groups=groups)
    model, token_freq, final_vocab = trainer.train()
    assert trainer.model is model
    assert trainer.token_freq is token_freq
    assert trainer.vocab is final_vocab
    assert set(token_freq) == {"eng", "deu"}
    assert len(final_vocab) > 0


def test_trainer_saves_and_reloads_via_tokenizer_json(tmp_path):
    """output_dir goes straight to tokenizers.Tokenizer.save() -- same
    convention as bpe/train.py's own output_dir handling, NOT a torch.save
    dict (there's no torch involved anywhere in this tokenizer at all)."""
    from systems.tokenization.parity_bpe.inference import load_checkpoint

    out_path = str(tmp_path / "model.json")
    cfg = ParityBPEConfig(vocab_size=280, output_dir=out_path)
    groups = [{"eng": "hello world", "deu": "hallo welt"}] * 20
    trainer = ParityBPETrainer(cfg, groups, eval_groups=groups)
    model, _, _ = trainer.train()

    reloaded = load_checkpoint(out_path)
    assert reloaded.num_parameters() == model.num_parameters()
    assert reloaded.encode_spans("hello world") == model.encode_spans("hello world")
