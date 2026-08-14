"""One-time local prep for bible-nlp/biblenlp-corpus.

common.data.corpora's own bible_nlp loader used to stream the live, ~5.2GB
corpus.json over HTTP on EVERY training run (via ijson, no local download) --
correct, but expensive: even a 2-language request scans however much of the
file precedes those languages, and a missing/mistyped language code forces a
full sequential scan before it can even raise an error (confirmed live, see
common.data.corpora._bible_nlp_canonical_entries's own docstring). This
script does that expensive scan ONCE, for every language the corpus has, and
writes the result to local disk in the exact same {"id": ..., "text": ...}
per-line JSONL shape common.data.oldi_data's own oldi_seed/flores_plus files
use -- so afterward, common.data.corpora's bible_nlp loader is an ordinary,
fast, local N-way intersect-by-id join, the same shape as those two sources,
not a special live-streaming case any more.

Usage (run once, ideally on a machine/connection you don't mind tying up for
a while -- this downloads the entire 5.2GB file in one pass):

    python -m common.data.prepare_bible_nlp --output-dir data/bible_nlp

Then --data-source bible_nlp (or common.data.corpora.stream_groups directly)
reads from that directory. Re-run this script (safe to overwrite) if the
upstream corpus is ever updated.

CANONICAL TRANSLATION PER LANGUAGE: some languages have multiple distinct
Bible translations mixed together under one corpus.json key (confirmed live,
e.g. English alone spans 43 distinct "file" values across ~1.09M verse-
entries) -- this picks whichever single translation has the MOST verse
entries as that language's canonical text, the same kind of one-canonical-
choice judgment call common.data.oldi_data.LANG_SCRIPT already makes for
scripts. metadata.json records what got picked and what didn't, per
language, so this choice stays inspectable rather than silent:
"file" (the chosen translation's own filename), "num_verses", "license",
"copyright" (as recorded on the chosen translation's own entries -- NOT
verified/normalized further, just passed through), "num_alternative_
translations" (how many OTHER translations existed for this language but
weren't used), and "alternative_files" (their filenames, for anyone who
wants to go pick a different one deliberately).

VERSE-KEY IDS: a bible_nlp entry's "verses" field is a list (usually one
ref, sometimes a range, e.g. ["GEN 1:1", "GEN 1:2"]) -- joined here with
";;" into a single string id (verse refs themselves never contain ";;",
confirmed by inspection, so this is a safe, reversible-in-practice
delimiter) to match oldi_seed/flores_plus's own flat string `id` convention.
Two translations that segment the same underlying content into different
verse RANGES won't share an id even if they cover the same verses -- a
known, accepted limitation (same one common.data.corpora's own former
live-streaming loader already documented), not something this script papers
over.
"""

import argparse
import json
import os

import ijson
import requests
from huggingface_hub import hf_hub_url
from tqdm.auto import tqdm

from .corpora import BIBLE_NLP_REPO


def _verse_key_to_id(verses):
    return ";;".join(verses)


def prepare_bible_nlp(output_dir, limit=None, request_timeout=1800):
    """Streams corpus.json ONCE (no local download of the raw file -- same
    hf_hub_url + requests.get(stream=True) + ijson.kvitems technique
    common.data.corpora._bible_nlp_canonical_entries already used, just run
    to completion instead of stopping early), writing one
    "{output_dir}/{lang}.jsonl" per language (canonical translation only) and
    a combined "{output_dir}/metadata.json" -- see module docstring for both
    shapes. limit: process at most this many languages (for a quick test
    run); None (the default) processes every language the corpus has, which
    takes a while -- this is a genuinely large one-time download+scan, not a
    quick command.

    Returns the metadata dict (also written to disk).
    """
    os.makedirs(output_dir, exist_ok=True)
    url = hf_hub_url(BIBLE_NLP_REPO, "corpus.json", repo_type="dataset")
    resp = requests.get(url, stream=True, timeout=request_timeout)
    resp.raise_for_status()

    metadata = {}
    try:
        for lang, entries in tqdm(ijson.kvitems(resp.raw, ""), desc="bible_nlp languages", unit="lang"):
            by_file = {}
            for entry in entries:
                by_file.setdefault(entry["file"], []).append(entry)
            best_file = max(by_file, key=lambda f: len(by_file[f]))
            chosen = by_file[best_file]

            rows = [
                {"id": _verse_key_to_id(entry["verses"]), "text": entry["text"]}
                for entry in chosen
                if entry.get("text")
            ]
            if not rows:
                continue

            with open(os.path.join(output_dir, f"{lang}.jsonl"), "w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")

            metadata[lang] = {
                "file": best_file,
                "num_verses": len(rows),
                "license": chosen[0].get("license", ""),
                "copyright": chosen[0].get("copyright", ""),
                "num_alternative_translations": len(by_file) - 1,
                "alternative_files": sorted(f for f in by_file if f != best_file),
            }
            if limit and len(metadata) >= limit:
                break
    finally:
        resp.close()

    with open(os.path.join(output_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"wrote {len(metadata)} languages to {output_dir!r} (see metadata.json for per-language detail)")
    return metadata


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="One-time download+convert of bible-nlp/biblenlp-corpus into this project's own local JSONL schema."
    )
    parser.add_argument("--output-dir", type=str, default="data/bible_nlp")
    parser.add_argument(
        "--limit", type=int, default=None,
        help="process at most this many languages -- for a quick test run; omit to process every language "
        "the corpus has (the real, intended use, but a much longer run)",
    )
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    prepare_bible_nlp(args.output_dir, limit=args.limit)


if __name__ == "__main__":
    main()
