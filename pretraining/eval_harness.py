"""Scoring primitives for the benchmarks in pretraining.benchmarks, run
against an already-pretrained TransformerLM checkpoint through its matching
TokenizerAdapter.

Two example shapes, two evaluators:
  - MultipleChoiceExample (XNLI, XCOPA) -> evaluate_multiple_choice, scored
    via loglikelihood (a forward pass, no sampling -- exact and cheap).
  - TranslationExample (FLORES MT) -> evaluate_translation, scored via
    TransformerLM.generate (sampling) + sacrebleu BLEU/chrF.

Infrastructure only: this module is unit-tested against a tiny from-scratch
model (see cli_eval.py's smoke test), not run against a real pretrained
checkpoint -- an untrained model has no reason to score above chance, so
such a number would just be noise. Real numbers are the user's to produce
once an actual pretraining run exists.
"""

import collections

import torch
import torch.nn.functional as F


def _encode_tensor(adapter, text, lang, device):
    ids = adapter.encode(text, lang=lang)
    return ids, torch.tensor([ids], dtype=torch.long, device=device)


def _common_prefix_len(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def loglikelihood(model, adapter, context, continuation, lang=None, device="cpu"):
    """Sum log P(continuation token | context, earlier continuation tokens)
    under `model`, in one forward pass. Returns (sum_logprob, num_tokens).

    context/continuation are encoded jointly (encode(context+continuation)),
    not as two separately-encoded id lists, since a token can straddle the
    boundary (notably for systems.superbpe, whose merges cross word/
    whitespace boundaries) -- joint encoding is the only way to score the
    same tokenization the model would actually see there. The split point
    is the longest common prefix of encode(context) and encode(context+
    continuation), not assumed equal to len(encode(context)): when the
    boundary tokenizes differently jointly, a few trailing context tokens
    can get folded into the scored region. Still a coherent likelihood,
    just not guaranteed to isolate exactly the continuation's own bytes --
    accepted rather than building a byte-span re-alignment.
    """
    context_ids, _ = _encode_tensor(adapter, context, lang, device)
    full_ids, _ = _encode_tensor(adapter, context + continuation, lang, device)
    split = _common_prefix_len(context_ids, full_ids)
    if split >= len(full_ids) - 1:
        raise ValueError(
            f"continuation {continuation!r} tokenized to nothing new past the "
            f"context boundary (split={split}, len(full_ids)={len(full_ids)}) -- "
            "cannot score an empty continuation"
        )

    max_seq_len = model.cfg.max_seq_len
    if len(full_ids) > max_seq_len:
        # Truncate context from the left, keeping the continuation intact;
        # if the continuation alone doesn't fit, raise rather than silently
        # truncating the thing being scored.
        drop = len(full_ids) - max_seq_len
        if drop >= split:
            raise ValueError(
                f"continuation {continuation!r} alone (+ boundary tokens) needs "
                f"{len(full_ids) - split} tokens, which doesn't fit in this "
                f"model's max_seq_len={max_seq_len}"
            )
        full_ids = full_ids[drop:]
        split -= drop

    ids_tensor = torch.tensor([full_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        logits, _ = model(ids_tensor)
    logprobs = F.log_softmax(logits[0, split - 1 : -1].float(), dim=-1)
    target = ids_tensor[0, split:]
    token_logprobs = logprobs.gather(1, target.unsqueeze(1)).squeeze(1)
    return token_logprobs.sum().item(), target.numel()


def evaluate_multiple_choice(model, adapter, examples, device="cpu", length_normalize=False):
    """examples: iterable of benchmarks.MultipleChoiceExample. Scores every
    candidate in ex.choices via loglikelihood, predicts argmax, compares to
    ex.label. length_normalize divides each candidate's score by its token
    count -- off by default (raw sum-loglikelihood, the standard "loglikelihood"
    request type); on trades exact ranking for robustness to candidates of
    very different lengths.

    Returns {"accuracy": float, "n": int, "per_language": {lang: {"accuracy":
    float, "n": int}}}.
    """
    model.eval()
    correct = 0
    total = 0
    per_lang = collections.defaultdict(lambda: [0, 0])  # lang -> [correct, n]

    for ex in examples:
        scores = []
        for choice in ex.choices:
            total_lp, n_tok = loglikelihood(model, adapter, ex.context, choice, ex.lang, device)
            scores.append(total_lp / n_tok if length_normalize and n_tok else total_lp)
        pred = max(range(len(scores)), key=lambda i: scores[i])
        is_correct = int(pred == ex.label)
        correct += is_correct
        total += 1
        per_lang[ex.lang][0] += is_correct
        per_lang[ex.lang][1] += 1

    return {
        "accuracy": correct / total if total else 0.0,
        "n": total,
        "per_language": {
            lang: {"accuracy": c / n if n else 0.0, "n": n} for lang, (c, n) in per_lang.items()
        },
    }


def evaluate_translation(
    model,
    adapter,
    examples,
    device="cpu",
    max_new_tokens=128,
    temperature=1.0,
    prompt_template=None,
    max_samples_per_pair=5,
):
    """examples: iterable of benchmarks.TranslationExample. Generates a
    translation via model.generate (no beam search / KV cache -- fine for
    infrastructure verification, a real large-scale eval would want a
    faster decode path) and scores against ex.reference_text with sacrebleu
    BLEU + chrF, aggregated per (source_lang, target_lang) pair.

    prompt_template(ex) -> str builds the generation prompt; default is a
    minimal English-worded instruction ("Translate {source_lang} to
    {target_lang}:\\n{source_text}\\n{target_lang}:") since pretraining.train
    produces a plain base LM, not an instruction-tuned one -- a real
    few-shot-example bank per language pair is future work.

    max_samples_per_pair: how many raw (source, hypothesis, reference)
    triples to keep verbatim per pair for qualitative inspection, capped
    since a real eval run can have thousands of examples per pair
    (corpus-level BLEU/chrF is still computed over the full set).

    Returns {"bleu": float, "chrf": float, "n": int, "per_pair": {"src->tgt":
    {"bleu": float, "chrf": float, "n": int, "samples": [{"source":...,
    "hypothesis":..., "reference":...}, ...]}}}. Pair keys are "src->tgt"
    strings, not tuples, since tuple keys aren't valid JSON and this result
    is written straight to disk via json.dumps in cli_eval.main.
    """
    import sacrebleu

    template = prompt_template or (
        lambda ex: f"Translate {ex.source_lang} to {ex.target_lang}:\n{ex.source_text}\n{ex.target_lang}:"
    )

    model.eval()
    by_pair = collections.defaultdict(lambda: {"srcs": [], "hyps": [], "refs": []})

    for ex in examples:
        prompt = template(ex)
        ids, ids_tensor = _encode_tensor(adapter, prompt, ex.source_lang, device)
        generated = model.generate(
            ids_tensor, max_new_tokens=max_new_tokens, temperature=temperature
        )
        new_ids = generated[0, len(ids) :].tolist()
        hyp_text = adapter.decode(new_ids).decode("utf-8", errors="replace")
        pair_key = f"{ex.source_lang}->{ex.target_lang}"
        by_pair[pair_key]["srcs"].append(ex.source_text)
        by_pair[pair_key]["hyps"].append(hyp_text)
        by_pair[pair_key]["refs"].append(ex.reference_text)

    per_pair = {}
    all_hyps, all_refs = [], []
    for pair_key, bucket in by_pair.items():
        bleu = sacrebleu.corpus_bleu(bucket["hyps"], [bucket["refs"]])
        chrf = sacrebleu.corpus_chrf(bucket["hyps"], [bucket["refs"]])
        if len(bucket["hyps"]) > max_samples_per_pair:
            print(
                f"[eval_harness] {pair_key}: keeping {max_samples_per_pair} of "
                f"{len(bucket['hyps'])} generated samples in results (full set "
                "still scored in bleu/chrf above, just not stored verbatim)"
            )
        samples = [
            {"source": s, "hypothesis": h, "reference": r}
            for s, h, r in zip(
                bucket["srcs"][:max_samples_per_pair],
                bucket["hyps"][:max_samples_per_pair],
                bucket["refs"][:max_samples_per_pair],
            )
        ]
        per_pair[pair_key] = {
            "bleu": bleu.score,
            "chrf": chrf.score,
            "n": len(bucket["hyps"]),
            "samples": samples,
        }
        all_hyps.extend(bucket["hyps"])
        all_refs.extend(bucket["refs"])

    overall_bleu = sacrebleu.corpus_bleu(all_hyps, [all_refs]) if all_hyps else None
    overall_chrf = sacrebleu.corpus_chrf(all_hyps, [all_refs]) if all_hyps else None
    return {
        "bleu": overall_bleu.score if overall_bleu else 0.0,
        "chrf": overall_chrf.score if overall_chrf else 0.0,
        "n": len(all_hyps),
        "per_pair": per_pair,
    }
