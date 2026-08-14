"""SuperBPE: two-stage byte-level BPE that lifts the whitespace-pretokenization
wall partway through training, letting later merges span multiple words
("superwords").

Paper: Liu, Hayase, Hofmann, Oh, Smith & Choi, "SuperBPE: Space Travel for
Language Models" (COLM 2025, arxiv.org/abs/2503.13423). Official code release:
github.com/PythonNut/superbpe (MIT licensed, reused here with attribution per
the project owner's confirmation that this is public/research use).

NOT a port of that release's training code: it trains via a custom Rust fork of
huggingface/tokenizers (github.com/alisawuffles/tokenizers-superbpe), which has
no place in this project's pure Python/PyTorch, byte-level stack -- every other
tokenizer here (fairtok/magnet/flexitokens/manta/fanta) is plain Python/PyTorch,
none of them touch the `tokenizers` library or Rust at all. This instead
reimplements the ALGORITHM the paper describes (Sec 2): ordinary BPE up to a
transition point `t * vocab_size`, then continue training WITHOUT the
whitespace-pretokenization constraint, so later merges can bridge what were
previously separate words. Everywhere the paper's prose doesn't pin down an
exact mechanism, this file makes an explicit choice -- search for "JUDGMENT
CALL".

Unlike every neural baseline in this repo, SuperBPE has no gradient descent, no
forward pass, and no fairness mechanism of its own -- like MANTa, it's a
fairness-agnostic control. The point of including it is different from MANTa's
though: it's not just another neural architecture without a fairness term, it's
a COMPLETELY different tokenizer construction paradigm (classical greedy BPE
merge counting over a static corpus, not a trained boundary predictor) --
testing whether cross-lingual fairness properties differ by tokenizer FAMILY,
not just by which neural mechanism is used within this project's own family.

SCALE-DOWN NOTICE, matching every other baseline's own module docstring: the
paper trains on ~26M-330M-byte corpora, vocab sizes up to 200K, with a
Rust-backed trainer. This project runs at fairtok's own compute scale (a few
thousand training sentences, vocab sizes in the hundreds for smoke tests, up to
the tens of thousands for real runs) with a pure-Python trainer -- FITTING
merges (_fit_merges) uses the same complexity class a real BPE trainer does
(touches only merge-affected sequences per step, not a full corpus rescan);
APPLYING merges (bpe_encode) favors a simple, obviously-correct reference
implementation over a heap-optimized one. Both are slower per-operation than a
Rust implementation, not a different algorithm -- a very large --vocab-size
real run will take noticeably longer wall-clock than the neural baselines' own
GPU training. Accepted, documented tradeoff, not an oversight.
"""

import re
from collections import Counter, defaultdict

# Byte-level pretokenizer: a run of whitespace bytes attaches to the WORD that
# FOLLOWS it (so an ordinary mid-sentence word carries its own preceding space,
# e.g. b" cat"), and a run of TRAILING whitespace with nothing after it is its
# own pretoken -- the GPT-2/byte-level-BPE convention. This is the "wall" Stage
# 1 respects and Stage 2 lifts (see module docstring). Matches only ASCII
# whitespace bytes (\s in a bytes pattern never matches a multi-byte UTF-8
# sequence, since every continuation/lead byte is >= 0x80) -- JUDGMENT CALL:
# non-ASCII Unicode space separators (e.g. U+00A0) are not treated as
# pretokenization walls, the same simplification GPT-2's own byte-level BPE
# regex makes.
_PRETOKEN_RE = re.compile(rb"\s*\S+|\s+")


def _to_bytes(text):
    return text.encode("utf-8") if isinstance(text, str) else bytes(text)


def pretokenize(raw):
    """str/bytes -> list[bytes], splitting on the whitespace-pretokenization
    wall described above. Deterministic, no learned state -- used identically
    at training time (Stage 1's word boundaries) and to reconstruct those same
    boundaries when re-applying Stage 1's merges during Stage 2 setup (see
    fit_superbpe below)."""
    return _PRETOKEN_RE.findall(_to_bytes(raw))


def _apply_one_merge(seq, pair, new_id):
    """Merges every non-overlapping occurrence of exactly `pair` in `seq`,
    left to right, in a SINGLE pass. Provably sufficient in one pass (not
    "loop until no change"): the only way a new (a, b) pair could appear
    after merging some occurrences of (a, b) into a brand-new symbol id
    would be for the removed elements to reappear elsewhere, which a pure
    substitution can't do -- so a single pass resolves every occurrence."""
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
    """The reference BPE application algorithm (Sennrich et al. 2016):
    repeatedly find the LOWEST-RANK (earliest-learned = highest-priority)
    adjacent pair actually present in `seq`, merge every occurrence of that
    exact pair (via _apply_one_merge), and repeat until no listed pair
    remains. This is priority order, not "whichever pair happens to appear
    first left-to-right" -- that distinction matters once more than one
    merge rule exists (i.e. always, past the very first learned merge),
    which is why this is used everywhere merges get APPLIED (final
    encoding, and Stage 2's setup re-application of Stage 1's rules) rather
    than a naive single-dict-pass substitution.

    merge_info: dict[(int,int) -> (rank:int, new_id:int)]. `seq`: list[int].
    O(T) per round, at most T-1 rounds for a length-T sequence (each round
    merges at least one pair) -- see module docstring's SCALE-DOWN NOTICE
    for why this isn't the asymptotically-faster heap-based version.
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
    """The shared incremental BPE-fitting loop, used by both Stage 1 (over
    deduped unique pretokens) and Stage 2 (over deduped unique sentences) --
    see fit_superbpe. `sequences`: list[list[int]], MUTATED IN PLACE as merges
    are applied. `weights`: parallel list[int], how many times each sequence
    occurs in the real corpus (deduping identical sequences into one entry
    with a frequency weight is what keeps this from re-processing the same
    string thousands of times). `id_to_bytes`: dict[int, bytes], MUTATED to
    add every newly-merged symbol's underlying byte string.

    Incremental, not "recount the whole corpus after every merge": maintains
    pair_counts (weighted adjacent-pair frequency) and pair_seqs (which
    sequence indices currently contain each pair) together, so applying one
    merge only touches the sequences that actually contained it -- the same
    complexity class a real BPE trainer uses, just in pure Python (see module
    docstring's SCALE-DOWN NOTICE). Each step only ever applies ONE pair
    (via _apply_one_merge, single-pass-sufficient -- see that function's own
    docstring), so priority order is a non-issue here: unlike bpe_encode,
    there is never more than one active rule per call.

    Returns list[((int,int), int)] merges in the order they were learned
    (this order IS priority order at encode time -- see SuperBPEModel).
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
        # Drop any pair whose count has decayed to <= 0 (fully consumed by
        # earlier merges) before picking the max -- keeps pair_counts from
        # growing unboundedly with dead entries.
        pair_counts = Counter({p: c for p, c in pair_counts.items() if c > 0})
        if not pair_counts:
            break  # corpus exhausted (no adjacent pairs left) before vocab_size reached
        best_count = max(pair_counts.values())
        # Deterministic tiebreak (lexicographically smallest pair) -- makes a
        # run reproducible given the same corpus, with no seed needed (see
        # SuperBPEConfig's own docstring).
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
    any language mixed together -- BPE itself has no notion of language, it
    just counts byte-pair frequencies over whatever text it's given).

    Stage 1 (within-word): dedupe the corpus into unique PRETOKENS (see
    pretokenize) with frequency weights, and learn merges via _fit_merges up
    to `transition_fraction * (vocab_size - 256)` merges -- ordinary BPE,
    since pretoken chunks never cross whitespace.

    Stage 2 (superwords): rebuild each sentence by pretokenizing it, applying
    Stage 1's learned merges (in PRIORITY order, via bpe_encode -- not a
    naive single-pass substitution, since more than one Stage 1 rule can
    apply within one chunk) within each chunk to mirror exactly what
    encoding will do at inference time up to this point, then CONCATENATING
    the chunks back into one flat per-sentence sequence -- the wall is gone,
    so continuing _fit_merges on these flat sequences lets a merge bridge
    two former chunks. Deduped by unique full-sentence text (weaker dedup
    than Stage 1's, since exact duplicate sentences are rarer than exact
    duplicate words, but still free when they occur).

    transition_fraction: JUDGMENT CALL. The paper sweeps this as a
    hyperparameter rather than fixing one value; 0.8 (80% of the merge
    budget spent on ordinary within-word BPE before superwords begin) is a
    plausible middle-of-the-literature choice, not a number taken from a
    specific reported best-configuration in the paper.

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
    """The trained artifact: an ORDERED list of merges (pair -> new id, in the
    order they were learned -- that order IS priority order at encode time,
    the standard BPE invariant) plus the byte string each symbol id
    ultimately represents. No neural weights, so num_parameters() reports the
    merge-table size instead -- the closest analog to "how big is this
    system" every other package's own num_parameters() answers."""

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
