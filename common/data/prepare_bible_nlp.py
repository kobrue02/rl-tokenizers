"""One-time local prep for bible-nlp/biblenlp-corpus.

common.data.corpora's bible_nlp loader used to stream the live ~5.2GB
corpus.json over HTTP on every training run -- correct but expensive (a
missing/mistyped language code forces a full sequential scan before even
raising an error). This script does that scan ONCE for every language and
writes the result to local disk in the same {"id": ..., "text": ...} JSONL
shape common.data.oldi_data's oldi_seed/flores_plus files use, so
afterward the bible_nlp loader is an ordinary, fast, local N-way
intersect-by-id join, not a special live-streaming case.

Usage (run once, ideally on a connection you don't mind tying up -- this
downloads the entire 5.2GB file in one pass):

    python -m common.data.prepare_bible_nlp --output-dir data/bible_nlp

Then --data-source bible_nlp reads from that directory. Safe to re-run if
the upstream corpus updates.

CANONICAL TRANSLATION PER LANGUAGE: some languages have multiple distinct
Bible translations mixed under one corpus.json key (e.g. English alone
spans 43 "file" values across ~1.09M verse-entries) -- this picks whichever
single translation has the MOST verse entries as canonical. metadata.json
records what got picked and what didn't per language: "file", "num_verses",
"license", "copyright" (passed through as-is), "num_alternative_
translations", and "alternative_files" (for anyone who wants a different
one).

VERSE-KEY IDS: a bible_nlp entry's "verses" field is a list (e.g. ["GEN
1:1", "GEN 1:2"]) -- joined here with ";;" into a single string id (verse
refs never contain ";;") to match oldi_seed/flores_plus's flat string `id`
convention. Two translations that segment the same content into different
verse RANGES won't share an id even if they cover the same verses -- a
known, accepted limitation.
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
    """Streams corpus.json ONCE (no local download of the raw file), writing
    one "{output_dir}/{lang}.jsonl" per language (canonical translation
    only) and a combined "{output_dir}/metadata.json" (see module
    docstring). limit: process at most this many languages (quick test
    run); None (default) processes every language, which takes a while.

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
