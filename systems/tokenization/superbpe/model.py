"""SuperBPE: two-stage byte-level BPE that lifts the whitespace-pretokenization
wall partway through training, letting later merges span multiple words
("superwords").

Paper: Liu, Hayase, Hofmann, Oh, Smith & Choi, "SuperBPE: Space Travel for
Language Models" (COLM 2025, arxiv.org/abs/2503.13423). Official code:
github.com/PythonNut/superbpe (MIT, reused here with attribution).

NOT a port of that release's training code, which uses a custom Rust fork of
huggingface/tokenizers -- out of place in this project's pure Python/PyTorch
byte-level stack. This instead reimplements the ALGORITHM (paper Sec 2):
ordinary BPE up to a transition point `t * vocab_size`, then continue
training WITHOUT the whitespace-pretokenization constraint so later merges
can bridge former word boundaries. Where the paper's prose doesn't pin down
an exact mechanism, search for "JUDGMENT CALL" for this file's choice.

Like MANTa, SuperBPE has no gradient descent/forward pass/fairness
mechanism -- a fairness-agnostic control. But unlike MANTa it's a
COMPLETELY different construction paradigm (classical greedy BPE merge
counting, not a trained boundary predictor) -- tests whether fairness
properties differ by tokenizer FAMILY, not just by neural mechanism.

SCALE-DOWN NOTICE (matches every other baseline): paper trains on
~26M-330M-byte corpora, vocab up to 200K, Rust-backed. This runs at
fairtok's compute scale (thousands of sentences, vocab in the hundreds for
smoke tests / tens of thousands for real runs) in pure Python -- FITTING
(_fit_merges) uses the same complexity class as a real trainer (touches
only merge-affected sequences per step); APPLYING (bpe_encode) favors a
simple, obviously-correct implementation over a heap-optimized one. Slower
per-operation, not a different algorithm -- an accepted tradeoff.
"""

import re
from collections import Counter, defaultdict

# Byte-level pretokenizer: a run of whitespace attaches to the WORD that
# FOLLOWS it (e.g. b" cat"), and trailing whitespace with nothing after it
# is its own pretoken -- the GPT-2/byte-level-BPE convention; this is the
# "wall" Stage 1 respects and Stage 2 lifts. Matches only ASCII whitespace
# bytes (\s never matches multi-byte UTF-8, since continuation/lead bytes
# are >= 0x80) -- JUDGMENT CALL: non-ASCII space separators (e.g. U+00A0)
# aren't treated as walls, same simplification GPT-2's regex makes.
_PRETOKEN_RE = re.compile(rb"\s*\S+|\s+")


def _to_bytes(text):
    return text.encode("utf-8") if isinstance(text, str) else bytes(text)


def pretokenize(raw):
    """str/bytes -> list[bytes], splitting on the whitespace wall above.
    Deterministic; used both for Stage 1's word boundaries and to
    reconstruct them when re-applying Stage 1's merges in Stage 2 setup."""
    return _PRETOKEN_RE.findall(_to_bytes(raw))


def _apply_one_merge(seq, pair, new_id):
    """Merges every non-overlapping occurrence of `pair` in `seq`, left to
    right, in ONE pass -- sufficient because a new (a, b) pair could only
    reappear if removed elements resurfaced elsewhere, which substitution
    can't do."""
    a, b = pair
    out = []
    i = 0
    n = len(seq)
    while i < n:
        if i < n - 1 and seq[i] == a and seq[i + 1] == b:
            out.append(new_id)
            i += 2
        else:
            out.append(seq[i])
            i += 1
    return out


def bpe_encode(seq, merge_info):
    """Reference BPE application (Sennrich et al. 2016): repeatedly find the
    LOWEST-RANK (earliest-learned = highest-priority) adjacent pair present
    in `seq`, merge every occurrence via _apply_one_merge, repeat until no
    listed pair remains. Priority order matters once more than one merge
    rule exists (i.e. always past the first) -- used everywhere merges get
    applied, not a naive single-pass substitution.

    merge_info: dict[(int,int) -> (rank:int, new_id:int)]. `seq`: list[int].
    O(T) per round, <= T-1 rounds -- not the heap-based asymptotically
    faster version (see module docstring's SCALE-DOWN NOTICE).
    """
    seq = list(seq)
    while True:
        best_rank = None
        best_pair = None
        for a, b in zip(seq, seq[1:]):
            info = merge_info.get((a, b))
            if info is not None and (best_rank is None or info[0] < best_rank):
                best_rank = info[0]
                best_pair = (a, b)
        if best_pair is None:
            return seq
        new_id = merge_info[best_pair][1]
        seq = _apply_one_merge(seq, best_pair, new_id)


def _fit_merges(sequences, weights, num_merges, id_to_bytes, next_id, log_prefix=None, log_every=500):
    """Shared incremental BPE-fitting loop, used by both Stage 1 (deduped
    unique pretokens) and Stage 2 (deduped unique sentences) -- see
    fit_superbpe. `sequences`: list[list[int]], MUTATED IN PLACE. `weights`:
    parallel list[int] of occurrence counts (dedup + weighting avoids
    reprocessing the same string thousands of times). `id_to_bytes`:
    dict[int, bytes], MUTATED to add each newly-merged symbol's bytes.

    Incremental, not "recount the whole corpus after every merge": maintains
    pair_counts (weighted pair frequency) and pair_seqs (which sequences
    contain each pair) together, so one merge only touches affected
    sequences -- same complexity class as a real trainer, in pure Python.
    Each step applies only ONE pair, so priority order (unlike bpe_encode)
    is a non-issue here.

    Returns list[((int,int), int)] merges in learned order (== priority
    order at encode time -- see SuperBPEModel).
    """

    def seq_pairs(seq):
        return list(zip(seq, seq[1:]))

    pair_counts = Counter()
    pair_seqs = defaultdict(set)
    for idx, (seq, w) in enumerate(zip(sequences, weights)):
        for pair in seq_pairs(seq):
            pair_counts[pair] += w
            pair_seqs[pair].add(idx)

    merges = []
    for step in range(num_merges):
        if log_prefix and log_every and step % log_every == 0:
            print(f"[superbpe] {log_prefix}: merge {step}/{num_merges}")
        # Drop pairs whose count decayed to <= 0 (fully consumed) before
        # picking the max -- keeps pair_counts from growing unboundedly.
        pair_counts = Counter({p: c for p, c in pair_counts.items() if c > 0})
        if not pair_counts:
            break  # corpus exhausted before vocab_size reached
        best_count = max(pair_counts.values())
        # Deterministic tiebreak (lexicographically smallest) -- reproducible
        # given the same corpus, no seed needed.
        pair = min(p for p, c in pair_counts.items() if c == best_count)

        new_id = next_id
        next_id += 1
        a, b = pair
        id_to_bytes[new_id] = id_to_bytes[a] + id_to_bytes[b]
        merges.append((pair, new_id))

        affected = list(pair_seqs.pop(pair, ()))
        for idx in affected:
            seq = sequences[idx]
            w = weights[idx]
            for p in seq_pairs(seq):
                pair_counts[p] -= w
                pair_seqs[p].discard(idx)
            new_seq = _apply_one_merge(seq, pair, new_id)
            sequences[idx] = new_seq
            for p in seq_pairs(new_seq):
                pair_counts[p] += w
                pair_seqs[p].add(idx)

    return merges


def fit_superbpe(sentences, vocab_size, transition_fraction=0.8, verbose=False):
    """Fits a SuperBPEModel from a flat list of raw sentences (str or bytes,
    languages mixed -- BPE has no notion of language).

    Stage 1 (within-word): dedupe into unique PRETOKENS (pretokenize) with
    frequency weights, learn merges via _fit_merges up to
    `transition_fraction * (vocab_size - 256)` merges -- ordinary BPE, since
    pretoken chunks never cross whitespace.

    Stage 2 (superwords): rebuild each sentence by pretokenizing it, applying
    Stage 1's merges in PRIORITY order (bpe_encode) within each chunk to
    mirror inference-time encoding, then CONCATENATING chunks into one flat
    per-sentence sequence -- the wall is gone, so continuing _fit_merges on
    these flat sequences lets a merge bridge two former chunks. Deduped by
    full-sentence text (weaker than Stage 1's, but still free when it hits).

    transition_fraction: JUDGMENT CALL -- the paper sweeps this rather than
    fixing one value; 0.8 is a plausible middle-of-the-literature choice,
    not a reported best configuration.

    Returns a SuperBPEModel.
    """
    id_to_bytes = {i: bytes([i]) for i in range(256)}
    total_merges = max(0, vocab_size - 256)
    stage1_merges = round(transition_fraction * total_merges)
    stage2_merges = total_merges - stage1_merges

    word_counts = Counter()
    for sent in sentences:
        word_counts.update(pretokenize(sent))
    words = list(word_counts.keys())
    word_weights = [word_counts[w] for w in words]
    word_seqs = [list(w) for w in words]

    stage1_merges_list = _fit_merges(
        word_seqs, word_weights, stage1_merges, id_to_bytes, next_id=256,
        log_prefix="stage1 (within-word)" if verbose else None,
    )
    stage1_merge_info = {
        pair: (rank, new_id) for rank, (pair, new_id) in enumerate(stage1_merges_list)
    }

    sentence_counts = Counter(_to_bytes(s) for s in sentences)
    sentences_unique = list(sentence_counts.keys())
    sentence_weights = [sentence_counts[s] for s in sentences_unique]
    sentence_seqs = []
    for sent in sentences_unique:
        flat = []
        for chunk in pretokenize(sent):
            flat.extend(bpe_encode(list(chunk), stage1_merge_info))
        sentence_seqs.append(flat)

    next_id = 256 + len(stage1_merges_list)
    stage2_merges_list = _fit_merges(
        sentence_seqs, sentence_weights, stage2_merges, id_to_bytes, next_id=next_id,
        log_prefix="stage2 (superword)" if verbose else None,
    )

    return SuperBPEModel(
        merges=stage1_merges_list + stage2_merges_list,
        id_to_bytes=id_to_bytes,
        num_stage1_merges=len(stage1_merges_list),
    )


class SuperBPEModel:
    """Trained artifact: an ORDERED list of merges (pair -> new id, learned
    order == priority order at encode time -- the standard BPE invariant)
    plus the byte string each symbol id represents. No neural weights, so
    num_parameters() reports merge-table size instead."""

    def __init__(self, merges, id_to_bytes, num_stage1_merges=0):
        self.merges = merges  # list[((int,int), int)], learned order == priority order
        self.id_to_bytes = id_to_bytes  # dict[int, bytes]
        self.num_stage1_merges = num_stage1_merges  # for reporting only
        self._merge_info = {
            pair: (rank, new_id) for rank, (pair, new_id) in enumerate(merges)
        }

    def num_parameters(self):
        return len(self.merges)

    def encode_ids(self, raw):
        """str/bytes -> list[int] symbol ids, applying every learned merge in
        priority order (see bpe_encode)."""
        return bpe_encode(list(_to_bytes(raw)), self._merge_info)

    def encode_spans(self, raw):
        """str/bytes -> list[bytes] spans -- what common.eval.cross_tokenizer/common.vocab
        expect from every tokenizer's induce_spans (see segment.py)."""
        return [self.id_to_bytes[i] for i in self.encode_ids(raw)]
