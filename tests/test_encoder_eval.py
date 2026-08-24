"""Tests for systems.pretraining.encoder_eval: the pure-math pieces
(topk_retrieval_accuracy, word_align, roundtrip_accuracy) get exact-answer
unit tests on hand-crafted embeddings; mean_pooled_embeddings/
sentence_retrieval/pseudo_perplexity get integration tests against a real
(tiny) encoder + EncoderVocab to confirm the whole wiring works and returns
sane ranges, not exact values (a randomly-initialized encoder has no
"correct" retrieval/perplexity value to assert against)."""

import math

import torch

from systems.pretraining.encoder_eval import (
    corpus_pseudo_perplexity,
    mean_pooled_embeddings,
    pseudo_perplexity,
    roundtrip_accuracy,
    sentence_retrieval,
    topk_retrieval_accuracy,
    word_align,
)
from systems.pretraining.encoder_model import build_encoder
from systems.pretraining.encoder_model_configs import get_preset
from systems.pretraining.encoder_tokenizer import EncoderVocab
from systems.pretraining.tokenizer_adapter import TokenizerAdapter
from systems.tokenization.bpe.model import fit_bpe


def test_topk_retrieval_accuracy_perfect_when_rows_are_identical():
    embeds = torch.randn(20, 8)
    result = topk_retrieval_accuracy(embeds, embeds, k=(1, 5, 10))
    assert result[1] == result[5] == result[10] == 1.0


def test_topk_retrieval_accuracy_finds_correct_row_even_with_similar_neighbors():
    """Ground truth is row-POSITION alignment (query[i] should retrieve
    index[i]), not content -- query_embeds[i] and index_embeds[i] are each
    a different, distinctly-directioned base plus small independent noise,
    so top-1 must match by position rather than falling for a
    similar-but-wrong neighboring row."""
    torch.manual_seed(0)
    bases = torch.eye(6) * 10  # 6 well-separated directions
    query_embeds = bases + torch.randn(6, 6) * 0.01
    index_embeds = bases + torch.randn(6, 6) * 0.01

    result = topk_retrieval_accuracy(query_embeds, index_embeds, k=(1,))

    assert result[1] == 1.0


def test_topk_retrieval_accuracy_top1_worse_than_top10_when_ambiguous():
    torch.manual_seed(0)
    n = 30
    embeds = torch.randn(n, 4)
    noisy = embeds + torch.randn(n, 4) * 3.0  # heavy noise -- top-1 should
    # be considerably harder than "is it anywhere in the top 10"
    result = topk_retrieval_accuracy(embeds, noisy, k=(1, 10))
    assert result[10] >= result[1]


def test_word_align_finds_mutual_nearest_neighbors():
    # 3 well-separated directions -- alignment should recover the identity
    # permutation (token i in a matches token i in b) exactly.
    a = torch.eye(3) * 10
    b = torch.eye(3) * 10
    assert word_align(a, b) == {(0, 0), (1, 1), (2, 2)}


def test_word_align_excludes_non_mutual_pairs():
    # b has only 2 rows; a's 3rd row's best match (row 0) is NOT mutual
    # (row 0 of b prefers a's row 0), so (2, 0) must be excluded.
    a = torch.tensor([[10.0, 0.0], [0.0, 10.0], [9.0, 1.0]])
    b = torch.tensor([[10.0, 0.0], [0.0, 10.0]])
    alignment = word_align(a, b)
    assert (0, 0) in alignment
    assert (1, 1) in alignment
    assert not any(i == 2 for i, j in alignment)


def test_word_align_empty_when_either_side_is_empty():
    assert word_align(torch.zeros((0, 4)), torch.randn(3, 4)) == set()
    assert word_align(torch.randn(3, 4), torch.zeros((0, 4))) == set()


def test_roundtrip_accuracy_all_succeed_with_identity_alignments(monkeypatch):
    """Stubs _token_embeddings so every language's tokens are IDENTICAL
    (identity alignment at every hop) -- every start_idx must round-trip
    back to itself, giving accuracy exactly 1.0. Isolates roundtrip_accuracy's
    own frontier-tracking logic from the neural model entirely."""
    import systems.pretraining.encoder_eval as encoder_eval_module

    fixed_embeds = torch.eye(4) * 10  # 4 "words", well-separated

    def fake_token_embeddings(model, vocab, text, lang, device, layer, max_len):
        return fixed_embeds

    monkeypatch.setattr(encoder_eval_module, "_token_embeddings", fake_token_embeddings)

    verse_groups = [{"eng": "a b c d", "fra": "a b c d", "deu": "a b c d"}]
    acc = roundtrip_accuracy(
        model=None, vocab=None, verse_groups=verse_groups, cycle_langs=["eng", "fra", "deu", "eng"]
    )
    assert acc == 1.0


def test_roundtrip_accuracy_skips_verses_missing_a_cycle_language():
    verse_groups = [{"eng": "a b c", "fra": "a b c"}]  # no "deu"
    acc = roundtrip_accuracy(
        model=None, vocab=None, verse_groups=verse_groups, cycle_langs=["eng", "fra", "deu", "eng"]
    )
    assert math.isnan(acc)  # total stayed 0 -- nothing scoreable


def test_roundtrip_accuracy_rejects_a_cycle_that_doesnt_close():
    try:
        roundtrip_accuracy(None, None, [], cycle_langs=["eng", "fra", "deu"])
        assert False, "expected ValueError"
    except ValueError:
        pass


# --- Integration: real tiny encoder + real bpe tokenizer ------------------


def _tiny_vocab_and_model():
    sentences = [
        "the quick brown fox jumps over the lazy dog",
        "a small tokenizer trained only for this test",
        "held out validation should never leak into training samples",
    ]
    bpe_model = fit_bpe(sentences, vocab_size=300)
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        bpe_model.tokenizer.save(f.name)
        checkpoint_path = f.name
    adapter = TokenizerAdapter.load("bpe", checkpoint_path)
    vocab = EncoderVocab(adapter)
    preset = get_preset("tiny")
    model = build_encoder(preset, vocab.vocab_size)
    return vocab, model


def test_mean_pooled_embeddings_shape_and_finiteness():
    vocab, model = _tiny_vocab_and_model()
    texts = ["the quick brown fox", "a lazy dog", "tokenizer test sentence"]

    embeds = mean_pooled_embeddings(model, vocab, texts, device="cpu", layer=1)

    assert embeds.shape == (3, model.config.hidden_size)
    assert torch.isfinite(embeds).all()


def test_sentence_retrieval_returns_accuracies_in_valid_range():
    vocab, model = _tiny_vocab_and_model()
    source = ["the quick brown fox", "a lazy dog sleeps", "testing the tokenizer"]
    target = ["le renard brun rapide", "un chien paresseux dort", "tester le tokenizer"]

    result = sentence_retrieval(model, vocab, source, target, device="cpu", layer=1, k=(1, 3))

    assert set(result.keys()) == {1, 3}
    for acc in result.values():
        assert 0.0 <= acc <= 1.0
    assert result[1] <= result[3]  # top-3 accuracy can never be lower than top-1


def test_pseudo_perplexity_is_positive_and_finite():
    vocab, model = _tiny_vocab_and_model()
    ppl = pseudo_perplexity(model, vocab, "the quick brown fox jumps", device="cpu")
    assert math.isfinite(ppl)
    assert ppl > 0


def test_pseudo_perplexity_empty_text_is_nan():
    vocab, model = _tiny_vocab_and_model()
    assert math.isnan(pseudo_perplexity(model, vocab, "", device="cpu"))


def test_corpus_pseudo_perplexity_averages_finite_values():
    vocab, model = _tiny_vocab_and_model()
    ppl = corpus_pseudo_perplexity(model, vocab, ["the fox", "a dog", "testing"], device="cpu")
    assert math.isfinite(ppl)
    assert ppl > 0
