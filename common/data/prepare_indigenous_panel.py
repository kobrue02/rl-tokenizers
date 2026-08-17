"""One-time local prep for common.data.indigenous_panel's curated pair
manifest (see that module for the panel and its provenance). Each pair has
its own bespoke access method (HF dataset, direct-download archive, plain
CSV, or aligned/transcript files fetched from GitHub), unlike bible_nlp's
single homogeneous source -- so this is several small loaders sharing one
CLI, not one generic downloader.

Usage (run once):

    python -m common.data.prepare_indigenous_panel --output-dir data/indigenous_panel

Writes one "{output_dir}/{pair_code}.jsonl" per pair (already the 2-key
{lang_a: text, lang_b: text} shape common.data.corpora reads directly --
every source here row-aligns its own two languages already, unlike
bible_nlp's cross-translation verse IDs) and a combined
"{output_dir}/metadata.json" (per-pair row count, source, license, family,
morphology tag). Safe to re-run; --pairs restricts to a subset for a
quicker test run.
"""

import argparse
import csv
import io
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
    CHREN_BRANCH,
    CHREN_REPO,
    HF_CREE_REPO,
    MAORI_HF_REPO,
    MAPUDUNGUN_BRANCH,
    MAPUDUNGUN_REPO,
    NRC_HANSARD_ARCHIVE_ROOT,
    NRC_HANSARD_URL,
    PAIRS,
)


def _write_pairs_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _prepare_cree(output_dir):
    """KonradBRG/plains-cree-figurative -- combines "gold" (228,
    human-verified) and "silver" (10,619, LLM-labeled) splits. Only
    text_cree/text_en are used; the figurative-language annotation columns
    are ignored."""
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
    .tgz instead of re-fetching ~202MB over the network every run."""
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
    the full ~202MB archive to a temp file (tar has no central directory, so
    extracting two members still requires reading the whole stream),
    extracts only the held-out split/test.{en,iu} (13,082 pairs, not the
    full ~1.3M-pair training-scale corpus), then removes the temp
    archive."""
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


def _fetch_raw_github_lines(repo, branch, path, request_timeout=60):
    """One raw.githubusercontent.com file -> list[str] of its lines, no git
    clone (shared by every GitHub-raw-hosted panel source below -- each repo
    also ships large PDFs/notebooks/audio not needed here)."""
    url = f"https://raw.githubusercontent.com/{repo}/{branch}/{path}"
    resp = requests.get(url, timeout=request_timeout)
    resp.raise_for_status()
    return resp.text.splitlines()


def _zip_aligned_lines(pair_code, code_a, lines_a, code_b, lines_b, label):
    """Two line-aligned file's contents -> row dicts, dropping any row where
    either side is empty. Raises if the two sides don't line up 1:1 --
    a real mismatch here means the wrong file pair or a source that changed
    shape, not something to silently truncate/pad around."""
    if len(lines_a) != len(lines_b):
        raise ValueError(
            f"{pair_code}: line count mismatch ({len(lines_a)} {code_a} vs "
            f"{len(lines_b)} {code_b}) -- expected {label} to be line-aligned 1:1"
        )
    return [{code_a: a, code_b: b} for a, b in zip(lines_a, lines_b) if a and b]


def _prepare_americasnlp_pair(pair_code, meta, output_dir, request_timeout=60):
    """One AmericasNLP 2021 shared-task language. train.{code}/train.es are
    line-aligned (verified across all nine of this shared task's language
    dirs)."""
    base = f"data/{meta['dir']}"
    indigenous_lines = _fetch_raw_github_lines(
        AMERICASNLP_REPO, AMERICASNLP_BRANCH, f"{base}/train.{meta['code']}", request_timeout
    )
    spanish_lines = _fetch_raw_github_lines(AMERICASNLP_REPO, AMERICASNLP_BRANCH, f"{base}/train.es", request_timeout)
    rows = _zip_aligned_lines(
        pair_code, meta["code"], indigenous_lines, "es", spanish_lines, f"train.{meta['code']}/train.es"
    )
    _write_pairs_jsonl(os.path.join(output_dir, f"{pair_code}.jsonl"), rows)
    return {
        "num_pairs": len(rows),
        "source": f"https://raw.githubusercontent.com/{AMERICASNLP_REPO}/{AMERICASNLP_BRANCH}/{base}/train.{{{meta['code']},es}}",
        "license": "see AmericasNLP 2021 shared task's own per-language README",
    }


def _prepare_chren(output_dir, request_timeout=60):
    """ChrEn (Zhang, Frey & Bansal, EMNLP 2020) -- data/parallel/{split}.{chr,en}
    line-aligned files per split. Combines train+dev+test (NOT out_dev/out_test,
    the paper's separate out-of-domain eval split -- a deliberately different
    distribution, not part of this panel's core bitext), same "combine what's
    available into one pair" approach as crk-en's gold+silver."""
    rows = []
    for split in ("train", "dev", "test"):
        base = f"data/parallel/{split}"
        chr_lines = _fetch_raw_github_lines(CHREN_REPO, CHREN_BRANCH, f"{base}.chr", request_timeout)
        en_lines = _fetch_raw_github_lines(CHREN_REPO, CHREN_BRANCH, f"{base}.en", request_timeout)
        rows.extend(_zip_aligned_lines("chr-en", "chr", chr_lines, "en", en_lines, f"{split}.chr/{split}.en"))
    _write_pairs_jsonl(os.path.join(output_dir, "chr-en.jsonl"), rows)
    return {
        "num_pairs": len(rows),
        "source": (
            f"https://raw.githubusercontent.com/{CHREN_REPO}/{CHREN_BRANCH}/"
            "data/parallel/{train,dev,test}.{chr,en}"
        ),
        "license": "none declared in source repo -- academic research release, cite Zhang et al. 2020",
    }


def _parse_maori_csv(text):
    """teara-en-mi.csv's raw text (columns en/mi) -> list[{"mi":..., "en":...}],
    dropping any row where either column is empty."""
    reader = csv.DictReader(io.StringIO(text))
    return [{"mi": r["mi"], "en": r["en"]} for r in reader if r["mi"] and r["en"]]


def _prepare_maori(output_dir, request_timeout=60):
    """jinglishi0206/Maori_English_New_Zealand (HF) -- a single teara-en-mi.csv,
    scraped from Te Ara (the Encyclopedia of New Zealand). Fetched as a plain
    CSV rather than through the `datasets` library -- one file, no configs to
    manage, same no-extra-dependency style as the AmericasNLP/ChrEn loaders.
    CC-BY-NC-3.0, research-use-only per the dataset card."""
    url = f"https://huggingface.co/datasets/{MAORI_HF_REPO}/resolve/main/teara-en-mi.csv"
    resp = requests.get(url, timeout=request_timeout)
    resp.raise_for_status()
    rows = _parse_maori_csv(resp.text)
    _write_pairs_jsonl(os.path.join(output_dir, "mi-en.jsonl"), rows)
    return {
        "num_pairs": len(rows),
        "source": url,
        "license": "CC-BY-NC-3.0 (research use only, per dataset card)",
    }


def _parse_mapudungun_transcript(lines):
    """One translation-clean/*.txt transcript's lines -> list[{"arn":...,
    "es":...}]. Format (AVENUE Mapudungun corpus, Duan et al. 2019): a
    ";"-prefixed comment header, then utterance blocks separated by blank
    lines -- each block an utterance-ID line followed by one or more
    "M: <mapudungun>" lines immediately followed by the same number of
    "C: <castellano>" lines (a block can split one utterance across several
    M:/C: line pairs, confirmed by inspecting real transcripts rather than
    assumed from the README alone). Utterance-ID lines are otherwise
    ignored -- neither language's text nor an alignment key we need."""
    rows = []
    m_lines, c_lines = [], []

    def flush():
        if m_lines and len(m_lines) == len(c_lines):
            rows.append({"arn": " ".join(m_lines), "es": " ".join(c_lines)})
        m_lines.clear()
        c_lines.clear()

    for line in lines:
        stripped = line.strip()
        if not stripped:
            flush()
        elif stripped.startswith(";"):
            continue
        elif stripped.startswith("M:"):
            m_lines.append(stripped[2:].strip())
        elif stripped.startswith("C:"):
            c_lines.append(stripped[2:].strip())
        # else: an utterance-ID header line, e.g. "nmdch-nmdch_x_0000_nmdch_00:"
    flush()
    return [r for r in rows if r["arn"] and r["es"]]


def _list_github_dir_filenames(repo, path, branch, request_timeout=60):
    """Case-correct filenames actually present in one directory of a GitHub
    repo, via the Contents API (raw.githubusercontent.com can't list a
    directory) -- unauthenticated, rate-limited to 60 req/hr, but this needs
    exactly one call per prep run."""
    url = f"https://api.github.com/repos/{repo}/contents/{path}?ref={branch}"
    resp = requests.get(url, timeout=request_timeout)
    resp.raise_for_status()
    return [item["name"] for item in resp.json() if item["type"] == "file"]


def _normalize_mapudungun_filename(name):
    """Matches dataset_splits/mt/*_files.txt entries against actual
    translation-clean/ filenames despite the source repo's own naming
    inconsistencies -- confirmed live: ~194 of 343 listed names differ from
    the real file only by case (e.g. listed "nmlch-lfch1" vs actual file
    "nmLch-lfch1"), and a few by embedded whitespace (listed "nmlch-nmjm1"
    vs actual "nmlch- nmjm1"). Stripping whitespace and lowercasing resolves
    all of those; ~7 remaining names have a genuine typo (transposed
    prefix, missing digit/letter) that can't be auto-corrected without
    guessing -- see _prepare_mapudungun's own handling of those."""
    return name.replace(" ", "").lower()


def _prepare_mapudungun(output_dir, request_timeout=60):
    """mingjund/mapudungun-corpus (AVENUE project: CMU, Chilean Ministry of
    Education, Instituto de Estudios Indígenas at Universidad de La
    Frontera) -- translation-clean/*.txt per-recording transcripts (see
    _parse_mapudungun_transcript). Combines the training/dev/test file
    lists under dataset_splits/mt/ into one pair, same "combine what's
    available" approach as crk-en/chr-en. A handful of listed filenames
    have no real match even after normalization (source repo typos, see
    _normalize_mapudungun_filename) -- skipped with a printed warning
    rather than guessed at or allowed to crash the whole prep run.
    CC-BY-NC-SA-3.0 -- research use only; ShareAlike applies to any
    redistribution of derived data, moot here since data/ is gitignored
    (see this module's own docstring)."""
    actual_files = _list_github_dir_filenames(MAPUDUNGUN_REPO, "translation-clean", MAPUDUNGUN_BRANCH, request_timeout)
    by_normalized = {
        _normalize_mapudungun_filename(name[:-4]): name for name in actual_files if name.endswith(".txt")
    }

    filenames = []
    for split_list in ("training_files.txt", "dev_files.txt", "test_files.txt"):
        lines = _fetch_raw_github_lines(
            MAPUDUNGUN_REPO, MAPUDUNGUN_BRANCH, f"dataset_splits/mt/{split_list}", request_timeout
        )
        filenames.extend(name.strip() for name in lines if name.strip())

    rows = []
    unresolved = []
    for name in filenames:
        actual_name = by_normalized.get(_normalize_mapudungun_filename(name))
        if actual_name is None:
            unresolved.append(name)
            continue
        lines = _fetch_raw_github_lines(
            MAPUDUNGUN_REPO, MAPUDUNGUN_BRANCH, f"translation-clean/{actual_name}", request_timeout
        )
        rows.extend(_parse_mapudungun_transcript(lines))
    if unresolved:
        print(
            f"  arn-es: {len(unresolved)} of {len(filenames)} listed transcripts have no matching file in "
            f"translation-clean/ even after normalization -- skipped (likely source-repo typos): {unresolved}"
        )

    _write_pairs_jsonl(os.path.join(output_dir, "arn-es.jsonl"), rows)
    return {
        "num_pairs": len(rows),
        "source": (
            f"https://raw.githubusercontent.com/{MAPUDUNGUN_REPO}/{MAPUDUNGUN_BRANCH}/"
            f"translation-clean/ ({len(filenames) - len(unresolved)} of {len(filenames)} files listed "
            "under dataset_splits/mt/ resolved)"
        ),
        "license": "CC-BY-NC-SA-3.0 (research use only, ShareAlike -- see source repo's License.txt)",
    }


def prepare_indigenous_panel(output_dir, pairs=None):
    """pairs: subset of common.data.indigenous_panel.PAIRS's keys to prepare
    (default: every pair). Returns the metadata dict this also writes to
    output_dir/metadata.json.
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
        elif meta["loader"] == "chren":
            info = _prepare_chren(output_dir)
        elif meta["loader"] == "hf_csv":
            info = _prepare_maori(output_dir)
        elif meta["loader"] == "mapudungun":
            info = _prepare_mapudungun(output_dir)
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
