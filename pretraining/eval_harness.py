"""Scoring primitives for the benchmarks in pretraining.benchmarks, run
against an already-pretrained pretraining.model.TransformerLM checkpoint
through its matching pretraining.tokenizer_adapter.TokenizerAdapter.

Two example shapes, two evaluators:
  - MultipleChoiceExample (XNLI, XCOPA) -> evaluate_multiple_choice, scored
    via loglikelihood (a forward pass, no sampling -- exact and cheap).
  - TranslationExample (FLORES MT) -> evaluate_translation, scored via
    TransformerLM.generate (sampling -- this is what finally gives
    generate() a real caller; see model.py's own docstring, which had
    flagged it as correctly-implemented-but-previously-unreachable) +
    sacrebleu BLEU/chrF.

Infrastructure only, per explicit scope: this module is unit-tested against
a tiny from-scratch model in this same PR (see the smoke test at the bottom
of pretraining/cli_eval.py), NOT run against a real pretrained checkpoint --
a from-scratch, randomly-initialized/undertrained model has no reason to
score above chance on any of this, and reporting such a number would be
noise dressed up as a result. Real numbers are the user's own to produce,
once an actual pretraining run exists to evaluate.
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

    context/continuation are ENCODED JOINTLY (adapter.encode(context +
    continuation)), not concatenated as two separately-encoded id lists --
    a token can straddle the context/continuation boundary (this matters
    especially for systems.superbpe, whose whole design point is merges
    that cross word/whitespace boundaries), and re-splitting after joint
    encoding is the only way to score the SAME tokenization the model would
    actually see at that boundary. The split point is found as the longest
    common prefix between encode(context) and encode(context+continuation)
    rather than assumed equal to len(encode(context)) -- a JUDGMENT CALL:
    when the boundary tokenizes differently jointly than alone (again, most
    likely for superbpe), a few trailing context tokens can get folded into
    the scored region. That's still a coherent likelihood (every token
    scored is still conditioned on everything before it), just not
    guaranteed to isolate EXACTLY the continuation string's own bytes --
    accepted here rather than building a byte-span-based re-alignment this
    project's other tokenizers don't need.
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
        # Truncate CONTEXT from the left (drop earliest tokens), keeping the
        # continuation intact -- if the continuation alone doesn't fit, that's
        # a genuinely different problem (the model can't score it at all at
        # this max_seq_len) and we raise rather than silently truncating the
        # very thing being scored.
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
    ex.label. length_normalize divides each candidate's score by its own
    token count first -- off by default (raw sum-loglikelihood, matching
    the plain "loglikelihood" request type most base-model MC evals report),
    on trades exact-log-likelihood ranking for robustness to candidates of
    very different lengths (e.g. XCOPA's choice1/choice2 are usually similar
    length, so this rarely matters there; expose it anyway since it's a
    one-line difference and the standard alternative).

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
    translation via model.generate (no beam search / KV cache -- see that
    method's own docstring; fine for infrastructure verification, a real
    large-scale MT eval would want a faster decode path) and scores against
    ex.reference_text with sacrebleu BLEU + chrF, aggregated per (source_lang,
    target_lang) pair.

    prompt_template(ex) -> str builds the string handed to the model as the
    generation prompt; default is a minimal, English-worded instruction
    ("Translate {source_lang} to {target_lang}:\\n{source_text}\\n{target_lang}:")
    -- a JUDGMENT CALL, same spirit as benchmarks.py's XNLI/XCOPA templates:
    a genuinely instruction-tuned/few-shot-primed setup would do better, but
    this project's pretraining.train produces a plain base LM, and building
    a real few-shot-example bank per language pair is future work, not
    infrastructure this module should silently fake.

    max_samples_per_pair: how many raw (source, hypothesis, reference)
    triples to keep verbatim per pair for qualitative inspection (the
    corpus-level BLEU/chrF numbers alone don't tell you WHAT the model
    actually generated) -- capped, not all of them, since a real eval run
    can have thousands of examples per pair; a print() notes when a pair's
    samples were truncated, so a small --max-samples-per-pair isn't mistaken
    for "the model only translated this many sentences."

    Returns {"bleu": float, "chrf": float, "n": int, "per_pair": {"src->tgt":
    {"bleu": float, "chrf": float, "n": int, "samples": [{"source":...,
    "hypothesis":..., "reference":...}, ...]}}}. Pair keys are "src->tgt"
    STRINGS, not (src, tgt) tuples -- tuple dict keys aren't valid JSON, and
    this result is written straight to disk via json.dumps in
    pretraining.cli_eval.main (confirmed directly: json.dumps on a
    tuple-keyed dict raises TypeError -- this was caught before it could
    crash a real eval run, since cli_eval's own smoke test only ever
    inspected this dict in-memory, never serialized it).
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
