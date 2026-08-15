"""One-time local prep for common.data.indigenous_panel's curated pair
manifest -- see that module's own docstring for the panel itself and each
pair's real, verified provenance. Each pair has its own bespoke access
method (an HF dataset, a direct-download archive, individual files fetched
straight from a GitHub repo), unlike bible_nlp's single homogeneous source,
so this script is three small loaders sharing one CLI, not one generic
downloader.

Usage (run once):

    python -m common.data.prepare_indigenous_panel --output-dir data/indigenous_panel

Writes one "{output_dir}/{pair_code}.jsonl" per pair -- each row already the
2-key {lang_a: text, lang_b: text} shape common.data.corpora.
_stream_indigenous_panel_single reads directly (no per-language join
needed at read time: every source here already row-aligns its own two
languages, unlike bible_nlp's cross-translation verse IDs) -- and a
combined "{output_dir}/metadata.json" (per-pair row count, source,
license, family, morphology tag). Re-run (safe to overwrite) if any
upstream source updates; --pairs restricts to a subset for a quicker test
run.
"""

import argparse
import json
import os
import tarfile
import tempfile

import requests
from datasets import load_dataset
from tqdm.auto import tqdm

from .indigenous_panel import (
    AMERICASNLP_BRANCH,
    AMERICASNLP_REPO,
    HF_CREE_REPO,
    NRC_HANSARD_ARCHIVE_ROOT,
    NRC_HANSARD_URL,
    PAIRS,
)


def _write_pairs_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _prepare_cree(output_dir):
    """KonradBRG/plains-cree-figurative -- both "gold" (228, human-verified)
    and "silver" (10,619, LLM-labeled) splits combined; the figurative-
    language annotation columns (label/rationale/footnote_en) are unused
    here, only text_cree/text_en matter for tokenizer fairness."""
    rows = []
    for config in ("gold", "silver"):
        ds = load_dataset(HF_CREE_REPO, config)
        split = next(iter(ds.values()))  # one split per config, named after it
        for r in split:
            if r["text_cree"] and r["text_en"]:
                rows.append({"crk": r["text_cree"], "en": r["text_en"]})
    _write_pairs_jsonl(os.path.join(output_dir, "crk-en.jsonl"), rows)
    return {"num_pairs": len(rows), "source": f"hf:{HF_CREE_REPO}", "license": "CC-BY-4.0"}


def _extract_nrc_hansard_test_split(tgz_path):
    """Split out so a test can point this at an already-downloaded local
    .tgz instead of re-fetching ~202MB over the network every run -- the
    extraction logic (what actually needs verifying) is identical either
    way."""
    en_member = f"{NRC_HANSARD_ARCHIVE_ROOT}/split/test.en"
    iu_member = f"{NRC_HANSARD_ARCHIVE_ROOT}/split/test.iu"
    with tarfile.open(tgz_path, "r:gz") as tar:
        en_text = tar.extractfile(en_member).read().decode("utf-8").splitlines()
        iu_text = tar.extractfile(iu_member).read().decode("utf-8").splitlines()
    if len(en_text) != len(iu_text):
        raise ValueError(
            f"Nunavut Hansard test split line count mismatch: {len(en_text)} en vs "
            f"{len(iu_text)} iu -- expected these to be line-aligned 1:1"
        )
    return [{"iu": iu, "en": en} for en, iu in zip(en_text, iu_text) if en and iu]


def _prepare_inuktitut(output_dir, request_timeout=1800):
    """Nunavut Hansard Inuktitut-English Parallel Corpus 3.0.1 -- downloads
    the full ~202MB archive to a temp file (its own Accept-Ranges support
    isn't useful here: tar has no central directory, so extracting two
    specific members still needs the whole stream read through), extracts
    only the corpus's OWN held-out split/test.{en,iu} (13,082 pairs -- see
    indigenous_panel's own docstring for why the test split, not the full
    ~1.3M-pair training-scale corpus), then removes the temp archive."""
    with tempfile.NamedTemporaryFile(suffix=".tgz", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        resp = requests.get(NRC_HANSARD_URL, stream=True, timeout=request_timeout)
        resp.raise_for_status()
        total_mb = int(resp.headers.get("content-length", 0)) // (1 << 20)
        with open(tmp_path, "wb") as f:
            for chunk in tqdm(
                resp.iter_content(chunk_size=1 << 20),
                desc="downloading Nunavut Hansard", unit="MiB", total=total_mb or None,
            ):
                f.write(chunk)
        rows = _extract_nrc_hansard_test_split(tmp_path)
    finally:
        os.remove(tmp_path)

    _write_pairs_jsonl(os.path.join(output_dir, "iu-en.jsonl"), rows)
    return {"num_pairs": len(rows), "source": NRC_HANSARD_URL, "license": "CC-BY-4.0"}


def _prepare_americasnlp_pair(pair_code, meta, output_dir, request_timeout=60):
    """One AmericasNLP 2021 shared-task language, fetched directly via
    raw.githubusercontent.com (no git clone -- this repo also ships a large
    PDF and per-language ipynb/csv files this project has no use for; a
    full clone would pull all of that for nothing). train.{code}/train.es
    are line-aligned -- confirmed by direct inspection of all nine of this
    shared task's language dirs, same line count, same file naming
    convention throughout."""
    base = f"https://raw.githubusercontent.com/{AMERICASNLP_REPO}/{AMERICASNLP_BRANCH}/data/{meta['dir']}"
    indigenous_resp = requests.get(f"{base}/train.{meta['code']}", timeout=request_timeout)
    indigenous_resp.raise_for_status()
    spanish_resp = requests.get(f"{base}/train.es", timeout=request_timeout)
    spanish_resp.raise_for_status()
    indigenous_lines = indigenous_resp.text.splitlines()
    spanish_lines = spanish_resp.text.splitlines()
    if len(indigenous_lines) != len(spanish_lines):
        raise ValueError(
            f"{pair_code}: line count mismatch ({len(indigenous_lines)} {meta['code']} vs "
            f"{len(spanish_lines)} es) -- expected train.{meta['code']}/train.es to be line-aligned 1:1"
        )
    rows = [
        {meta["code"]: ind, "es": es}
        for ind, es in zip(indigenous_lines, spanish_lines)
        if ind and es
    ]
    _write_pairs_jsonl(os.path.join(output_dir, f"{pair_code}.jsonl"), rows)
    return {
        "num_pairs": len(rows),
        "source": f"{base}/train.{{{meta['code']},es}}",
        "license": "see AmericasNLP 2021 shared task's own per-language README",
    }


def prepare_indigenous_panel(output_dir, pairs=None):
    """pairs: subset of common.data.indigenous_panel.PAIRS's own keys to
    prepare (default: every pair in PAIRS). Returns the metadata dict this
    also writes to output_dir/metadata.json.
    """
    os.makedirs(output_dir, exist_ok=True)
    pair_codes = list(pairs) if pairs else list(PAIRS)
    unknown = set(pair_codes) - set(PAIRS)
    if unknown:
        raise ValueError(f"unknown pair(s) {sorted(unknown)} -- choose from {sorted(PAIRS)}")

    metadata = {}
    for pair_code in tqdm(pair_codes, desc="indigenous_panel pairs", unit="pair"):
        meta = PAIRS[pair_code]
        if meta["loader"] == "hf_cree":
            info = _prepare_cree(output_dir)
        elif meta["loader"] == "nrc_hansard":
            info = _prepare_inuktitut(output_dir)
        elif meta["loader"] == "americasnlp":
            info = _prepare_americasnlp_pair(pair_code, meta, output_dir)
        else:
            raise ValueError(f"{pair_code}: unknown loader {meta['loader']!r}")
        metadata[pair_code] = {
            **info,
            "language": meta["language"],
            "anchor": meta["anchor"],
            "family": meta["family"],
            "morphology": meta["morphology"],
        }
        print(f"  {pair_code}: {info['num_pairs']:,} pairs")

    with open(os.path.join(output_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"wrote {len(metadata)} pair(s) to {output_dir!r} (see metadata.json for per-pair detail)")
    return metadata


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="One-time download+convert of common.data.indigenous_panel's curated pairs into local JSONL."
    )
    parser.add_argument("--output-dir", type=str, default="data/indigenous_panel")
    parser.add_argument(
        "--pairs", type=str, default=None,
        help=f"comma-separated subset of pair codes to prepare (default: all of {sorted(PAIRS)})",
    )
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    pairs = args.pairs.split(",") if args.pairs else None
    prepare_indigenous_panel(args.output_dir, pairs=pairs)


if __name__ == "__main__":
    main()
