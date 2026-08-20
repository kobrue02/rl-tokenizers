"""Scoring primitives for the benchmarks in systems.pretraining.benchmarks, run
against an already-pretrained TransformerLM checkpoint through its matching
TokenizerAdapter.

Four example shapes, four evaluators:
  - MultipleChoiceExample (XNLI, XCOPA, BLiMP) -> evaluate_multiple_choice,
    scored via loglikelihood (a forward pass, no sampling -- exact and cheap).
  - TranslationExample (FLORES MT) -> evaluate_translation, scored via
    TransformerLM.generate (sampling) + sacrebleu BLEU/chrF.
  - CoLAExample (CoLA) -> evaluate_cola, scored via loglikelihood +
    threshold calibration + MCC (see that function's own docstring for why
    this needs a genuinely different scoring shape than evaluate_multiple_choice).
  - QAExample (SQuAD) -> evaluate_qa, scored via TransformerLM.generate +
    official SQuAD exact-match/F1.

Infrastructure only: this module is unit-tested against a tiny from-scratch
model (see cli_eval.py's smoke test), not run against a real pretrained
checkpoint -- an untrained model has no reason to score above chance, so
such a number would just be noise. Real numbers are the user's to produce
once an actual pretraining run exists.
"""

import collections
import re
import string

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
    boundary (notably for systems.tokenization.superbpe, whose merges cross word/
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

    if split == 0:
        # No context token precedes the continuation's own first token, so
        # there's no position to condition its prediction on (this model has
        # no trained BOS). adapter.eos_id is a real token the model HAS seen
        # in exactly this role: data_prep.py packs documents back-to-back
        # separated by eos_id, so "eos_id, first real token" is a genuine
        # part of its training distribution, not an invented stand-in.
        full_ids = [adapter.eos_id] + full_ids
        split = 1

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


def _mcc(tp, tn, fp, fn):
    """Matthews correlation coefficient over a 2x2 confusion matrix -- CoLA's
    own official GLUE metric (not accuracy, which its class imbalance makes
    a poor discriminator). Returns 0.0 for a degenerate matrix (zero
    denominator, e.g. every prediction the same class) rather than raising."""
    denom_sq = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    if denom_sq == 0:
        return 0.0
    return (tp * tn - fp * fn) / (denom_sq**0.5)


def _sweep_best_threshold(labeled_scores):
    """labeled_scores: list[(label, score)]. Returns (threshold, mcc) for
    the threshold (predict label=1 iff score >= threshold) that maximizes
    MCC. O(n log n): sorts once, then sweeps every split point with
    prefix-sum confusion-matrix counts instead of a naive O(n^2) re-scan per
    candidate threshold -- matters at CoLA train's real scale (8551 rows)."""
    n = len(labeled_scores)
    if n == 0:
        return 0.0, 0.0
    ordered = sorted(labeled_scores, key=lambda p: p[1])
    total_pos = sum(1 for label, _ in ordered if label == 1)
    total_neg = n - total_pos

    prefix_pos = [0] * (n + 1)  # prefix_pos[i] = positives among ordered[:i]
    for i, (label, _) in enumerate(ordered):
        prefix_pos[i + 1] = prefix_pos[i] + (1 if label == 1 else 0)

    best_mcc = float("-inf")
    best_threshold = ordered[0][1]
    for i in range(n + 1):
        # Candidate threshold predicts ordered[i:] positive, ordered[:i]
        # negative (i == n means "predict everyone negative").
        fn_ = prefix_pos[i]
        tp = total_pos - fn_
        fp = (n - i) - tp
        tn = total_neg - fp
        mcc = _mcc(tp, tn, fp, fn_)
        threshold = ordered[i][1] if i < n else ordered[-1][1] + 1.0
        if mcc > best_mcc:
            best_mcc = mcc
            best_threshold = threshold
    return best_threshold, best_mcc


def evaluate_cola(model, adapter, examples, calibration_examples, device="cpu"):
    """examples: iterable of benchmarks.CoLAExample -- the real, reported
    eval set (glue/cola's "validation" split). calibration_examples: a
    SEPARATE CoLAExample iterable (glue/cola's own "train" split) used ONLY
    to pick a decision threshold, never itself scored/reported -- picking a
    threshold against the same data being reported would leak, same
    principle as not tuning hyperparameters on a test set.

    Score = each sentence's own UNCONDITIONAL (context="") log-likelihood,
    LENGTH-NORMALIZED (sum_logprob / num_tokens). Unlike
    evaluate_multiple_choice's within-item comparison, this score is
    compared against one GLOBAL threshold across the whole corpus, so raw
    sum-loglikelihood (which scales with sentence length regardless of
    grammaticality) isn't usable directly.

    Threshold: chosen by _sweep_best_threshold to maximize MCC on
    calibration_examples, then applied unchanged to `examples`.

    Returns {"mcc": float, "accuracy": float, "n": int, "threshold": float,
    "n_calibration": int}.
    """
    model.eval()

    def _score(ex):
        total_lp, n_tok = loglikelihood(model, adapter, "", ex.sentence, ex.lang, device)
        return total_lp / n_tok if n_tok else total_lp

    calibration = [(ex.label, _score(ex)) for ex in calibration_examples]
    if not calibration:
        raise ValueError("evaluate_cola needs at least one calibration example to pick a threshold")
    threshold, _ = _sweep_best_threshold(calibration)

    examples = list(examples)
    tp = tn = fp = fn = 0
    correct = 0
    for ex in examples:
        pred = 1 if _score(ex) >= threshold else 0
        correct += int(pred == ex.label)
        if pred == 1 and ex.label == 1:
            tp += 1
        elif pred == 0 and ex.label == 0:
            tn += 1
        elif pred == 1 and ex.label == 0:
            fp += 1
        else:
            fn += 1

    return {
        "mcc": _mcc(tp, tn, fp, fn),
        "accuracy": correct / len(examples) if examples else 0.0,
        "n": len(examples),
        "threshold": threshold,
        "n_calibration": len(calibration),
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
    {target_lang}:\\n{source_text}\\n{target_lang}:") since systems.pretraining.train
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


_ARTICLES_RE = re.compile(r"\b(a|an|the)\b")


def _normalize_answer(text):
    """Official SQuAD normalization (Rajpurkar et al. 2016's own eval
    script): lowercase, strip punctuation, remove articles, collapse
    whitespace -- so "The Denver Broncos." and "denver broncos" match."""
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = _ARTICLES_RE.sub(" ", text)
    return " ".join(text.split())


def _exact_match(prediction, reference):
    return int(_normalize_answer(prediction) == _normalize_answer(reference))


def _f1(prediction, reference):
    pred_tokens = _normalize_answer(prediction).split()
    ref_tokens = _normalize_answer(reference).split()
    if not pred_tokens or not ref_tokens:
        return float(pred_tokens == ref_tokens)
    common = collections.Counter(pred_tokens) & collections.Counter(ref_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def _best_over_references(score_fn, prediction, references):
    """Official SQuAD protocol: a question can have multiple acceptable
    reference answers, and the reported score per-question is the BEST
    match over all of them, not an average."""
    return max((score_fn(prediction, ref) for ref in references), default=0.0)


def evaluate_qa(
    model, adapter, examples, device="cpu", max_new_tokens=32, temperature=1.0,
    prompt_template=None, max_samples=20,
):
    """examples: iterable of benchmarks.QAExample. Generates an answer via
    model.generate (same unbatched/no-KV-cache path evaluate_translation
    uses) and scores against ex.answers with the official SQuAD exact-match/
    F1 metrics, taking the best score over every acceptable reference
    per question (see _best_over_references).

    prompt_template(ex) -> str builds the generation prompt; default is a
    minimal "Context: ...\\nQuestion: ...\\nAnswer:" instruction -- same
    base-LM-not-instruction-tuned caveat as evaluate_translation's own default.
    Generation is cut at the first newline (no stop token exists for "end
    of answer" without instruction-tuning, so anything after is the model
    continuing past the answer, not part of it).

    SKIPS (and counts, doesn't crash on) any example whose encoded prompt
    plus max_new_tokens would exceed model.cfg.max_seq_len -- byte-level
    tokenization over a paragraph-length SQuAD context can genuinely not
    fit, especially at smaller presets; truncating the context instead
    risks silently cutting the answer span out from under the question.

    max_samples: how many raw (question, prediction, references) triples to
    keep verbatim in the result for qualitative inspection -- aggregate EM/F1
    is still computed over every scored example, not just these.

    Returns {"exact_match": float, "f1": float, "n": int,
    "n_skipped_too_long": int, "samples": [...]}.
    """
    template = prompt_template or (
        lambda ex: f"Context: {ex.context}\nQuestion: {ex.question}\nAnswer:"
    )

    model.eval()
    max_seq_len = model.cfg.max_seq_len
    em_total = 0.0
    f1_total = 0.0
    n_scored = 0
    n_skipped = 0
    samples = []

    for ex in examples:
        prompt = template(ex)
        ids, ids_tensor = _encode_tensor(adapter, prompt, ex.lang, device)
        if len(ids) + max_new_tokens > max_seq_len:
            n_skipped += 1
            continue
        generated = model.generate(ids_tensor, max_new_tokens=max_new_tokens, temperature=temperature)
        new_ids = generated[0, len(ids) :].tolist()
        prediction = adapter.decode(new_ids).decode("utf-8", errors="replace").split("\n", 1)[0]

        em_total += _best_over_references(_exact_match, prediction, ex.answers)
        f1_total += _best_over_references(_f1, prediction, ex.answers)
        n_scored += 1
        if len(samples) < max_samples:
            samples.append({"question": ex.question, "prediction": prediction, "references": ex.answers})

    return {
        "exact_match": em_total / n_scored if n_scored else 0.0,
        "f1": f1_total / n_scored if n_scored else 0.0,
        "n": n_scored,
        "n_skipped_too_long": n_skipped,
        "samples": samples,
    }
