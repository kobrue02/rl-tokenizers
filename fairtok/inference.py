"""Apply a FROZEN, already-trained policy to a corpus to harvest the final
vocabulary. This is the step that turns "a policy that's good at fair
byte-boundary placement" (learned on the curated Phase 1 parallel data) into
"an actual tokenizer vocabulary" for the real pretraining corpus -- the
behavior transfers, not the Phase 1 data itself.

Distinct from fairtok.train: no gradients, no reward, no fairness scalar,
no parallel groups required -- just segment whatever text you give it and
count what comes out.
"""

from collections import Counter, defaultdict

import torch

from common.bytes_utils import bytes_to_tensor
from common.vocab import save_vocab_json, save_vocab_stats, vocab_with_stats

from .policy import BytePolicy, segment_bytes


def save_checkpoint(policy, path):
    # Read hidden_dim/num_layers off the policy itself rather than requiring the
    # caller to track and pass them separately -- BytePolicy already stores both.
    torch.save(
        {"hidden_dim": policy.hidden_dim, "num_layers": policy.num_layers, "state_dict": policy.state_dict()},
        path,
    )


def load_checkpoint(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    policy = BytePolicy(hidden_dim=ckpt["hidden_dim"], num_layers=ckpt.get("num_layers", 1))
    policy.load_state_dict(ckpt["state_dict"])
    policy.eval()
    return policy


def build_vocab_from_corpus(policy, texts, vocab_size, deterministic=True, progress=None):
    """texts: either a flat iterable of str/bytes documents (a single unlabeled
    corpus), or a dict {label: iterable_of_documents} if the corpus has
    language/source labels worth keeping in the per-entry breakdown. This
    doesn't need parallel groups across languages the way Phase 1 training
    data does -- the fairness objective already shaped the policy; this step
    is just applying it.

    progress: optional callable(iterable, desc) -> iterable, e.g. tqdm.auto.tqdm,
    to show progress without hard-coding a UI dependency into this function.
    """
    if not isinstance(texts, dict):
        texts = {"corpus": texts}
    wrap = progress if progress is not None else (lambda it, desc: it)

    token_freq = defaultdict(Counter)
    for label, docs in texts.items():
        for text in wrap(docs, f"segmenting [{label}]"):
            byte_seq = bytes_to_tensor(text)
            spans = segment_bytes(policy, byte_seq, deterministic=deterministic)
            token_freq[label].update(spans)

    entries = vocab_with_stats(token_freq, vocab_size)
    return token_freq, entries


def build_and_save_vocab(policy, texts, vocab_size, out_json, out_stats, deterministic=True, progress=None):
    token_freq, entries = build_vocab_from_corpus(policy, texts, vocab_size, deterministic, progress)
    if out_json:
        save_vocab_json(entries, out_json)
    if out_stats:
        save_vocab_stats(entries, out_stats)
    return token_freq, entries
