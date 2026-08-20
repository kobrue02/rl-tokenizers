"""One-time local prep for cis-lmu/Glot500.

common.data.corpora's glot500 loader used to stream all ~411 per-language
HF Hub configs LIVE on every training run (~308GB total) -- correct but
extremely expensive: a real glot500-scale pretraining data prep took
multiple days and several SLURM resubmits, and confirmed live that EVERY
resume re-streamed from scratch over the network just to fast-forward past
already-processed documents, and every separate prep run (bpe/fanta x
50k/large, four total) re-downloaded the identical corpus independently.
This script does that download ONCE per language and writes the result to
local disk in the exact {lang: text} shape common.data.corpora's glot500
loader already yields, so afterward it's an ordinary fast local read, not a
live streaming case -- same pattern as common.data.prepare_bible_nlp /
prepare_indigenous_panel, see common.data.corpora's own module docstring.

Usage (run once; the full ~411-language, ~308GB run will take a while --
try a small --limit/--max-workers smoke test first):

    python -m common.data.prepare_glot500 --output-dir data/glot500 --limit 5 --max-workers 4
    python -m common.data.prepare_glot500 --output-dir data/glot500

Then --dataset glot500 (with --dataset-config pointing at the same
--output-dir, if not the default) reads from that directory instead of
live HF streaming -- see common.data.corpora.stream_groups's own glot500
branch. No live fallback.

RESUMABLE, UNLIKE prepare_bible_nlp/prepare_indigenous_panel: 308GB across
~411 languages will not finish inside one SLURM job's time limit. Each
language is written to a sibling ".jsonl.tmp" file and only atomically
os.replace()'d to its final "{lang}.jsonl" name once that language's
stream is fully exhausted -- a killed run never leaves a half-written
".jsonl", so "does {lang}.jsonl already exist" is a reliable per-language
completion marker. Rerunning the identical command skips every language
already done (pass --force to redo everything anyway).

PARALLELIZED, UNLIKE THE OTHER TWO: --max-workers (default 8) runs that
many languages concurrently via a ThreadPoolExecutor -- genuinely I/O-bound
(network streaming per language), matching this project's only other
concurrency precedent (systems/tokenization/claude_tokenizer/evaluate.py's
ThreadPoolExecutor). One language's failure doesn't abort the others (same
per-item error isolation as scripts/evaluate_own_tokenizers_indigenous_panel.py),
recorded under metadata.json's own "_failed" key.

KEYED BY THE EXACT SAME STRING common.data.corpora's (now-retired) live
glot500 loader used to yield under: whatever string was actually iterated
(a native Glot500 config name like "eng_Latn" when using --langs all/
list_glot500_configs, or the literal --langs entry a caller passed
otherwise) -- preserves every downstream consumer's existing key
expectations (lang_counts, dedup, encode()'s lang hint) exactly.

metadata.json is non-authoritative and rewritten fresh on every invocation
from whatever THIS run's own languages did (skipped vs. freshly written,
with doc/byte counts for the latter only -- an already-skipped language's
file isn't re-read just to report a count, since that would burn real I/O
for no functional purpose); common.data.corpora.list_glot500_local_langs
(a directory listing) is what actually determines what's usable, not this
file.
"""

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm.auto import tqdm

from .corpora import GLOT500_LOCAL_DIR, _resolve_glot500_config, _stream_hf, list_glot500_configs


def _prepare_one_language(lang, output_dir, force):
    """Downloads one language's full glot500 stream and writes it to
    "{output_dir}/{lang}.jsonl" (atomically, via a sibling ".tmp" file --
    see module docstring's RESUMABLE section). Returns a dict describing
    what happened; never raises -- a per-language failure is reported, not
    propagated, so it doesn't abort the ThreadPoolExecutor's other
    in-flight languages."""
    path = os.path.join(output_dir, f"{lang}.jsonl")
    if not force and os.path.exists(path):
        return {"lang": lang, "status": "skipped_already_exists"}

    tmp_path = path + ".tmp"
    num_docs = 0
    num_bytes = 0
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            for row in _stream_hf("cis-lmu/Glot500", _resolve_glot500_config(lang)):
                text = row.get("text")
                if not text:
                    continue
                f.write(json.dumps({lang: text}, ensure_ascii=False) + "\n")
                num_docs += 1
                num_bytes += len(text.encode("utf-8"))
        os.replace(tmp_path, path)
    except Exception as e:
        # Leaves the ".tmp" file behind on failure -- harmless: nothing
        # ever reads it (only the final "{lang}.jsonl" name is), and a
        # rerun overwrites it from scratch since the final file still
        # doesn't exist.
        return {"lang": lang, "status": "failed", "error": str(e)}

    return {"lang": lang, "status": "written", "num_docs": num_docs, "num_bytes": num_bytes}


def prepare_glot500(output_dir, langs=None, max_workers=8, limit=None, force=False):
    """langs: None/"all" processes every language list_glot500_configs()
    returns; an explicit list processes only those (in whatever form the
    caller supplies -- native config name or this project's own short code,
    see _resolve_glot500_config). limit: process at most this many
    languages this invocation (quick smoke test); None (default) processes
    every requested language. max_workers: concurrent per-language
    downloads (see module docstring's PARALLELIZED section). force:
    redownload even languages whose ".jsonl" already exists.

    Returns the metadata dict (also written to "{output_dir}/metadata.json").
    """
    os.makedirs(output_dir, exist_ok=True)
    lang_list = list_glot500_configs() if langs in (None, "all") else list(langs)
    if limit:
        lang_list = lang_list[:limit]

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_prepare_one_language, lang, output_dir, force): lang for lang in lang_list}
        for future in tqdm(as_completed(futures), total=len(futures), desc="glot500 languages", unit="lang"):
            results.append(future.result())

    metadata = {r["lang"]: {k: v for k, v in r.items() if k != "lang"} for r in results}
    failed = {lang: m["error"] for lang, m in metadata.items() if m["status"] == "failed"}
    if failed:
        metadata["_failed"] = failed

    with open(os.path.join(output_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    num_written = sum(1 for r in results if r["status"] == "written")
    num_skipped = sum(1 for r in results if r["status"] == "skipped_already_exists")
    print(
        f"wrote {num_written} language(s), skipped {num_skipped} already-prepared, "
        f"{len(failed)} failed, out of {len(lang_list)} requested -- see metadata.json"
    )
    return metadata


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="One-time download of cis-lmu/Glot500 into this project's own local per-language JSONL cache."
    )
    parser.add_argument("--output-dir", type=str, default=GLOT500_LOCAL_DIR)
    parser.add_argument(
        "--langs", type=str, default="all",
        help="comma-separated language codes, or 'all' (default) for every language "
        "list_glot500_configs() returns",
    )
    parser.add_argument(
        "--max-workers", type=int, default=8,
        help="concurrent per-language downloads -- see module docstring's PARALLELIZED section",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="process at most this many languages -- for a quick test run; omit to process every "
        "requested language (the real, intended use, but a much longer run)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="redownload every requested language even if its .jsonl already exists "
        "(default: skip and reuse it -- see module docstring's RESUMABLE section)",
    )
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    langs = None if args.langs == "all" else args.langs.split(",")
    prepare_glot500(
        args.output_dir, langs=langs, max_workers=args.max_workers, limit=args.limit, force=args.force,
    )


if __name__ == "__main__":
    main()
