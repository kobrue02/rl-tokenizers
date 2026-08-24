"""Evaluation for systems.pretraining.encoder_train's MLM encoder, covering
the three of Glot500's six published eval tasks that don't need a
finetune-a-classification-head harness (NER/POS/Taxi1500 all do, and aren't
built here -- see this project's own comparison of Glot500's eval suite
against what exists in systems.pretraining.benchmarks/eval_harness):
pseudoperplexity, sentence retrieval (Tatoeba/Bible), and roundtrip alignment.

Sentence/word embeddings come directly from transformers.AutoModelForMaskedLM's
own `output_hidden_states=True`, from a fixed hidden layer -- Glot500's own
retrieval/roundtrip scripts (XTREME's evaluate_retrieval_*.py, the vendored
SimAlign package) do the identical thing. Reimplemented directly here on top
of this project's own EncoderVocab (this project's tokenizers aren't HF
tokenizers) using plain torch cosine-similarity + topk/argmax instead of
faiss/scikit-learn -- no approximate search is needed at these eval sizes
(at most a few thousand sentences, a handful of tokens per sentence for
roundtrip alignment), so an exact O(N^2) search is instant and adds no
dependency this project doesn't already have.

Typical data source: common.data.corpora.stream_groups("tatoeba_mt",
config=f"{split}/{pair}") for retrieval, or stream_groups("bible_nlp",
langs=cycle_langs) for roundtrip alignment (bible_nlp's own multi-way
verse-id intersection already yields exactly the N-way-aligned groups
roundtrip needs in one call -- see corpora.py's own bible_nlp docstring).
"""

import math

import torch
import torch.nn.functional as F

from .encoder_tokenizer import MASK_ID, PAD_ID


def _encode_padded_batch(vocab, texts, langs, device, max_len):
    """texts: list[str]. Returns (input_ids, attention_mask), right-padded
    with PAD_ID to the batch's own longest (truncated) sequence."""
    langs = langs if langs is not None else [None] * len(texts)
    id_lists = [vocab.encode(t, lang=l)[:max_len] for t, l in zip(texts, langs)]
    longest = max((len(ids) for ids in id_lists), default=1)
    input_ids = torch.full((len(id_lists), longest), PAD_ID, dtype=torch.long)
    attention_mask = torch.zeros((len(id_lists), longest), dtype=torch.long)
    for i, ids in enumerate(id_lists):
        if ids:
            input_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            attention_mask[i, : len(ids)] = 1
    return input_ids.to(device), attention_mask.to(device)


@torch.no_grad()
def mean_pooled_embeddings(model, vocab, texts, langs=None, device="cpu", layer=8, batch_size=32, max_len=256):
    """Returns a (len(texts), hidden_size) tensor: mean-pool
    hidden_states[layer] over non-pad positions, matching Glot500's own
    retrieval script's default pool_type='mean' at a fixed layer (layer=8
    there too -- model(...).hidden_states[0] is the embedding output,
    [1..num_layers] each transformer layer's output, identical indexing
    convention here)."""
    was_training = model.training
    model.eval()
    all_embeds = []
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        batch_langs = langs[start : start + batch_size] if langs is not None else None
        input_ids, attention_mask = _encode_padded_batch(vocab, batch_texts, batch_langs, device, max_len)
        hidden = model(
            input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True
        ).hidden_states[layer]
        mask = attention_mask.unsqueeze(-1).float()
        summed = (hidden * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1.0)
        all_embeds.append((summed / counts).cpu())
    model.train(was_training)
    return torch.cat(all_embeds, dim=0)


def topk_retrieval_accuracy(query_embeds, index_embeds, k=(1, 5, 10)):
    """query_embeds[i] should retrieve index_embeds[i] (row-aligned parallel
    pairs) via cosine-similarity nearest neighbor. Returns {k: accuracy}."""
    q = F.normalize(query_embeds, dim=-1)
    idx = F.normalize(index_embeds, dim=-1)
    sims = q @ idx.t()  # (N, N)
    n = sims.shape[0]
    top = sims.topk(min(max(k), n), dim=-1).indices  # (N, max_k)
    correct = torch.arange(n).unsqueeze(1)
    hits = top == correct
    return {kk: hits[:, :kk].any(dim=1).float().mean().item() for kk in k}


def sentence_retrieval(model, vocab, source_texts, target_texts, source_langs=None, device="cpu", layer=8, k=(1, 5, 10)):
    """source_texts[i]/target_texts[i]: row-aligned parallel sentences.
    Matches Glot500's own tgt_lang='eng' convention: pass the non-English
    side as source_texts and English as target_texts to measure "retrieve
    English from this language" accuracy."""
    source_embeds = mean_pooled_embeddings(model, vocab, source_texts, source_langs, device, layer)
    target_embeds = mean_pooled_embeddings(model, vocab, target_texts, None, device, layer)
    return topk_retrieval_accuracy(source_embeds, target_embeds, k)


@torch.no_grad()
def _token_embeddings(model, vocab, text, lang, device, layer, max_len):
    ids = vocab.encode(text, lang=lang)[:max_len]
    if not ids:
        return torch.zeros((0, model.config.hidden_size))
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    was_training = model.training
    model.eval()
    hidden = model(input_ids=input_ids, output_hidden_states=True).hidden_states[layer][0]
    model.train(was_training)
    return hidden.cpu()


def word_align(embeds_a, embeds_b):
    """embeds_a: (len_a, H), embeds_b: (len_b, H) -- per-token embeddings
    for two sentences. Returns the set of (i, j) index pairs aligned via
    SimAlign's default "Argmax"/intersection method: keep a pair only if j
    is i's row-argmax (i's best match in b) AND i is j's column-argmax (j's
    best match in a) -- i.e. mutual nearest neighbors."""
    if embeds_a.shape[0] == 0 or embeds_b.shape[0] == 0:
        return set()
    sims = F.normalize(embeds_a, dim=-1) @ F.normalize(embeds_b, dim=-1).t()  # (len_a, len_b)
    fwd = sims.argmax(dim=1)  # each token in a -> its best match in b
    bwd = sims.argmax(dim=0)  # each token in b -> its best match in a
    return {(i, j.item()) for i, j in enumerate(fwd) if bwd[j.item()].item() == i}


def roundtrip_accuracy(model, vocab, verse_groups, cycle_langs, device="cpu", layer=8, max_len=64, max_branch=5):
    """verse_groups: list of dicts {lang: text}, one per aligned verse --
    e.g. from common.data.corpora.stream_groups("bible_nlp",
    langs=list(set(cycle_langs))), which already yields exactly this N-way
    aligned shape (see module docstring). cycle_langs: e.g.
    ["eng", "fra", "deu", "eng"] -- a word in the first language is aligned
    forward hop-by-hop through the rest of the cycle; success = landing
    back on the SAME word index it started from (Dufter & Schutze 2018's
    roundtrip evaluation, reused by Glot500's own paper). max_branch bounds
    how many candidate word indices are carried forward per hop (a word can
    align to more than one word; unbounded branching blows up over a long
    cycle) -- matches Glot500's own max_branch_thresh=5 default."""
    if cycle_langs[0] != cycle_langs[-1]:
        raise ValueError("cycle_langs must start and end on the same language")
    successes = 0
    total = 0
    for verse in verse_groups:
        if any(lang not in verse for lang in cycle_langs):
            continue
        token_embeds = {}
        skip = False
        for lang in set(cycle_langs):
            embeds = _token_embeddings(model, vocab, verse[lang], lang, device, layer, max_len)
            if embeds.shape[0] == 0:
                skip = True
                break
            token_embeds[lang] = embeds
        if skip:
            continue

        # One alignment per consecutive hop, computed once per verse (not
        # once per starting word index below).
        hop_alignments = [
            word_align(token_embeds[cycle_langs[hop]], token_embeds[cycle_langs[hop + 1]])
            for hop in range(len(cycle_langs) - 1)
        ]

        first_len = token_embeds[cycle_langs[0]].shape[0]
        for start_idx in range(first_len):
            total += 1
            frontier = {start_idx}
            for alignment in hop_alignments:
                next_frontier = {j for i, j in alignment if i in frontier}
                if len(next_frontier) > max_branch:
                    next_frontier = set(sorted(next_frontier)[:max_branch])
                frontier = next_frontier
                if not frontier:
                    break
            if start_idx in frontier:
                successes += 1
    return successes / total if total else float("nan")


@torch.no_grad()
def pseudo_perplexity(model, vocab, text, lang=None, device="cpu", max_len=256):
    """Standard PLL (pseudo-log-likelihood) recipe (Salazar et al. 2020, the
    metric Glot500's own paper cites for its PPPL numbers -- its actual
    scoring script was never published, see this project's own research
    into the Glot500 repo, so this reimplements the published recipe from
    scratch): mask each non-special token ONE AT A TIME, sum
    log P(true_token | rest of sentence, that one token masked), normalize
    by token count, exponentiate. Batches all n masked copies of the
    sentence into one forward call (not one call per token) but is still
    O(n) forward-pass cost in n=len(ids) -- expensive but simple, matching
    this project's own eval_harness.py's preference for straightforward
    over optimized (see e.g. TransformerLM.generate's own no-KV-cache note)."""
    ids = vocab.encode(text, lang=lang)[:max_len]
    n = len(ids)
    if n == 0:
        return float("nan")
    input_ids = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0).repeat(n, 1)
    positions = torch.arange(n, device=device)
    input_ids[positions, positions] = MASK_ID
    was_training = model.training
    model.eval()
    logits = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids)).logits  # (n, n, V)
    model.train(was_training)
    log_probs = F.log_softmax(logits, dim=-1)
    target_ids = torch.tensor(ids, dtype=torch.long, device=device)
    token_log_probs = log_probs[positions, positions, target_ids]  # (n,)
    return math.exp(-token_log_probs.mean().item())


def corpus_pseudo_perplexity(model, vocab, texts, langs=None, device="cpu", max_len=256):
    """Average PPPL over multiple sentences -- reported per language-script,
    matching Glot500's Table 4/5 convention of one PPPL number per language."""
    langs = langs if langs is not None else [None] * len(texts)
    values = [pseudo_perplexity(model, vocab, t, l, device, max_len) for t, l in zip(texts, langs)]
    values = [v for v in values if not math.isnan(v)]
    return sum(values) / len(values) if values else float("nan")
