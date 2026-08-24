"""Taxi1500-specific data loading for encoder_finetune_classification.py --
a separate module rather than folded into that generic one because this
project's own research into the Glot500 repo found the HF Hub's only
Taxi1500 dataset (cis-lmu/Taxi1500-RawData) ships text WITHOUT labels or
verse ids (confirmed live -- its own dataset card says so). The actual
label==Recommendation/Faith/Violence/Grace/Sin/Description data
zero_shot_train.py trains on only exists as plain TSV files in the
cisnlp/Taxi1500 GitHub repo's eng_data/ directory (verified live:
{split}.tsv, tab-separated verse_id/label/verse_text, no header) --
English only. Non-English labeled data needs the gated Taxi1500-c corpus
(a Google Form request per that repo's own README -- the same kind of gate
common/data/prepare_bible_nlp.py's own local-cache convention already deals
with for a different source) and isn't auto-downloadable here.

DOWNLOAD (download_taxi1500_split, one-time, mirrors common/data/
prepare_bible_nlp.py's own local-cache convention): fetches
eng_data/{split}.tsv from GitHub raw content into a local cache file, a
no-op if that file already exists.

LOADING (load_taxi1500_tsv): parses that same 3-column format from ANY
local path -- including a non-English file obtained yourself from the
gated Taxi1500-c corpus and formatted identically -- into the
list[{"text": ..., "label": ...}] rows encoder_finetune_classification.
ClassificationDataset expects, with label already mapped through
TAXI1500_LABELS' own id order.
"""

import os

import requests

from .encoder_finetune_classification import TAXI1500_LABELS

TAXI1500_GITHUB_RAW = "https://raw.githubusercontent.com/cisnlp/Taxi1500/main/eng_data/{split}.tsv"


def download_taxi1500_split(split, output_dir, timeout=60):
    """split: "train"/"dev"/"test" -- English only (see module docstring).
    Downloads once into output_dir/eng_{split}.tsv; a no-op if that file
    already exists (matches this project's own local-cache scripts'
    idempotent-rerun convention)."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"eng_{split}.tsv")
    if os.path.exists(path):
        return path
    resp = requests.get(TAXI1500_GITHUB_RAW.format(split=split), timeout=timeout)
    resp.raise_for_status()
    with open(path, "w", encoding="utf-8") as f:
        f.write(resp.text)
    return path


def load_taxi1500_tsv(path, label_list=TAXI1500_LABELS):
    """Parses a Taxi1500-format TSV (verse_id\\tlabel\\tverse_text, no
    header) into [{"text": verse_text, "label": int}, ...], label mapped
    through label_list's own id order (TAXI1500_LABELS by default, matching
    zero_shot_train.py's own {'Recommendation': 0, ...} scheme -- confirmed
    verbatim against that file). Raises a clear error on a malformed line
    or unrecognized label string rather than silently dropping the row or
    crashing on a KeyError deep inside a DataLoader worker."""
    label_to_id = {label: i for i, label in enumerate(label_list)}
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 3:
                raise ValueError(f"{path}:{line_num}: expected 3 tab-separated fields, got {len(parts)}")
            _verse_id, label, text = parts
            if label not in label_to_id:
                raise ValueError(
                    f"{path}:{line_num}: unrecognized label {label!r} -- expected one of {label_list}"
                )
            rows.append({"text": text, "label": label_to_id[label]})
    return rows
