"""Synthetic multilingual parallel corpus -- a placeholder for FLORES+/OLDI
Seed/SMOL until the real dataset/language lists are confirmed. Each
"language" is a byte-generation profile: high-resource profiles reuse a
small, repetitive alphabet (cheap to compress); low-resource profiles draw
from a larger, less repetitive range -- so cross-language compression
disparity is a real signal even on toy data, giving fairness-aware
losses/rewards something to push against. Used by every tokenizer's smoke
tests in this repo (fairtok, magnet, flexitokens, manta).

TODO(real data): replace with a loader over FLORES+/OLDI Seed/SMOL once the
Tier-1/Tier-2 language lists are confirmed.
"""

import random

LANG_PROFILES = {
    "high_resource": dict(alphabet=list(range(97, 107)), repeat_bias=0.7),
    "mid_resource": dict(alphabet=list(range(97, 123)), repeat_bias=0.4),
    "low_resource_a": dict(
        alphabet=list(range(48, 58)) + list(range(200, 240)), repeat_bias=0.15
    ),
    "low_resource_b": dict(
        alphabet=list(range(1, 40)) + list(range(240, 256)), repeat_bias=0.05
    ),
}


def _gen_sentence(profile, min_len, max_len, rng):
    alphabet = profile["alphabet"]
    bias = profile["repeat_bias"]
    length = rng.randint(min_len, max_len)
    common_chunk = [rng.choice(alphabet) for _ in range(3)]
    seq = []
    while len(seq) < length:
        if rng.random() < bias:
            seq.extend(common_chunk)
        else:
            seq.append(rng.choice(alphabet))
    return bytes(seq[:length])


def make_synthetic_parallel_groups(
    num_groups, langs=None, min_len=20, max_len=60, seed=0
):
    """Each group is one "parallel sentence": a dict {lang: byte_seq}, the
    same shape real data (common.data.oldi_data) produces."""
    rng = random.Random(seed)
    langs = langs or list(LANG_PROFILES)
    return [
        {
            lang: _gen_sentence(LANG_PROFILES[lang], min_len, max_len, rng)
            for lang in langs
        }
        for _ in range(num_groups)
    ]
