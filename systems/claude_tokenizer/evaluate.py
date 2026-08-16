"""Held-out evaluation for one or more Claude models' token counts (--model,
via Anthropic's public count_tokens API -- see model.py for why this is a
narrower integration than hf_frontier's: no local tokenizer, no byte spans,
one network call per document, and only compression/fertility/token parity
are reportable, NOT Rényi efficiency or Gini (need per-token identities the
public API doesn't expose).

--model takes a COMMA-SEPARATED list -- every model shares ONE RateLimiter
for the whole run, not one per model, since Anthropic's RPM limits are
account-wide, not per-model.

One model failing (bad name, exhausted retries, auth failure) does NOT
abort the rest of the list -- same per-repo error isolation as
hf_frontier/evaluate.py, doubly important here since each entry can involve
tens of thousands of real network calls.

--checkpoint-dir (or its --output-derived default) makes a run resumable:
every completed (group, lang, count) call is appended to a per-model JSONL
file, and a resumed run (same command) skips what's already recorded. This
matters because an org's real rate limit can be far below its tier's
published number (seen: 429s at --rpm 2000 where the account's actual cap
was 100/min) -- at a low real rpm, scoring the full ~272k-pair BOUQuET test
split can outlast a single job's time limit, so the intended workflow is:
submit, let it run until killed/timed out, resubmit the same command as
many times as it takes.
"""

import json
import os
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from common.config_file import parse_args_with_config
from common.data.oldi_data import load_bouquet_dev, load_bouquet_test
from common.eval.metrics import compression_rate, fertility
from common.eval.parity import _find_anchor_key, anchor_invariant_parity
from common.eval.reporting import word_count

from systems.bpe.train import _SMOKE_TEST_GROUPS

from .model import ClaudeTokenCounter, RateLimiter


def build_arg_parser():
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate one or more Claude models' token counts on held-out data "
        "via Anthropic's public count_tokens API (no model weights or local tokenizer involved)."
    )
    parser.add_argument(
        "--model", type=str, required=True,
        help="comma-separated Claude model names, e.g. claude-opus-5,claude-sonnet-5 -- "
        "every model shares one rate limiter for the whole run, see module docstring",
    )
    parser.add_argument(
        "--api-key", type=str, default=None,
        help="explicit Anthropic API key; falls back to the ANTHROPIC_API_KEY environment "
        "variable if not given (the anthropic SDK's own default lookup)",
    )
    parser.add_argument(
        "--rpm", type=int, default=2000,
        help="requests-per-minute cap shared across every model/thread -- match to your "
        "Anthropic usage tier (Start=2000, Build=4000, Scale=8000 at time of writing). "
        "WARNING: an org's real limit can be LOWER than its tier's published default -- if "
        "you see repeated 429s, lower this to match the limit the error message reports",
    )
    parser.add_argument(
        "--max-workers", type=int, default=25,
        help="concurrent request threads -- count_tokens has no batch endpoint, so this is "
        "what lets a run approach --rpm (a sequential loop is latency-bound, not rate-limit-"
        "bound, well under even a 2000/min cap)",
    )
    parser.add_argument(
        "--eval-data-source",
        choices=["bouquet", "bouquet_test", "synthetic", "indigenous_panel"],
        default="bouquet",
        help="'bouquet' (default): BOUQuET DEV, for tuning/exploratory comparisons; "
        "'bouquet_test': BOUQuET TEST, the held-out split -- reserve for final reported numbers; "
        "'synthetic': small real-text placeholder (systems.bpe.train's _SMOKE_TEST_GROUPS, not "
        "common.data.synthetic's byte generator which isn't guaranteed valid UTF-8), for a quick "
        "sanity check with minimal API usage; "
        "'indigenous_panel': common.data.indigenous_panel's curated panel (needs a one-time "
        "common.data.prepare_indigenous_panel run first) -- scored via "
        "evaluate_claude_on_indigenous_panel (mixed-anchor), not evaluate_claude_on_groups; "
        "results have a different shape, not comparable to a bouquet/bouquet_test/synthetic run's",
    )
    parser.add_argument(
        "--num-groups", type=int, default=None,
        help="cap the number of held-out groups scored; omit for the full set -- the full "
        "BOUQuET test split means ~272k individual API calls, uncapped by default",
    )
    parser.add_argument("--output", type=str, default=None, help="write combined JSON results here (default: print to stdout)")
    parser.add_argument(
        "--checkpoint-dir", type=str, default=None,
        help="directory to store one <model>.jsonl checkpoint file per model, appending every "
        "successfully completed (group, lang, count) call -- lets a killed/preempted/timed-out "
        "job resume via the exact same command without re-querying calls already paid for (a "
        "failed call is never checkpointed, so it's retried, not skipped forever). Defaults to "
        "'<output>.checkpoint/' if --output is given; if neither is set, checkpointing is "
        "disabled and an interrupted run has no resume safety net",
    )
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument(
        "--wandb-project", type=str, default="claude_tokenizer",
        help="own project, separate from hf_frontier's and every systems/*/train.py's "
        "per-system project -- this measures an external API's token counts, not a model "
        "or tokenizer trained/loaded locally",
    )
    parser.add_argument("--run-name", type=str, default="")
    return parser


def _load_eval_groups(args):
    if args.eval_data_source == "synthetic":
        groups = _SMOKE_TEST_GROUPS
        return groups[: args.num_groups] if args.num_groups else groups
    if args.eval_data_source == "indigenous_panel":
        from common.data.corpora import stream_groups

        groups = list(stream_groups("indigenous_panel", config="all"))
        return groups[: args.num_groups] if args.num_groups else groups
    loader = load_bouquet_test if args.eval_data_source == "bouquet_test" else load_bouquet_dev
    groups = loader("all")
    if args.num_groups:
        groups = groups[: args.num_groups]
    return groups


def _load_checkpoint(checkpoint_path):
    counts = {}
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        return counts
    with open(checkpoint_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                counts[(row["group"], row["lang"])] = row["count"]
            except (json.JSONDecodeError, KeyError):
                # A killed job can leave its last line half-written -- skip
                # it rather than fail the whole resume over one truncated row.
                continue
    return counts


def evaluate_claude_on_groups(
    eval_groups, count_fn, anchor_lang="eng", max_workers=25, progress_every=1000, checkpoint_path=None
):
    """count_fn: (text) -> int token count (e.g. ClaudeTokenCounter.count,
    bound to one model + rate limiter). Mirrors common.eval.cross_tokenizer.
    evaluate_on_groups's compression/fertility/token_parity computation,
    but WITHOUT renyi/gini (see module docstring for why).

    Every (group, language) call is dispatched to a ThreadPoolExecutor so
    the run can approach max_workers-many requests in flight rather than
    being latency-bound one call at a time. A call that ultimately fails
    (exhausted retries) is dropped from aggregate stats, not fatal to the
    whole run -- printed immediately as it happens.

    checkpoint_path: optional JSONL file that every SUCCESSFUL (group,
    lang, count) triple is appended to as it completes. If the file already
    has content (a prior run got killed/preempted/timed out), those pairs
    are loaded up front and never re-queried -- lets a multi-day run
    survive being resubmitted without re-paying for completed calls. Failed
    calls are never checkpointed, so they're retried next run.

    Returns {"per_lang_compression": {...}, "avg_compression": float,
    "fertility": {...}, "token_parity": {...}, "token_parity_anchor": str,
    "token_parity_gm": {...}, "token_parity_spread": float, "renyi": {}
    (always empty), "gini": None (always), "num_failed_calls": int,
    "num_total_calls": int, "num_skipped_via_checkpoint": int}.
    token_parity_gm/token_parity_spread are anchor-invariant (see
    common.eval.parity.anchor_invariant_parity) -- computed identically to
    evaluate_on_groups's, from the same token_parity dict, at zero extra
    API cost.
    """
    tasks_all = [(gi, lang, text) for gi, group in enumerate(eval_groups) for lang, text in group.items()]

    counts = _load_checkpoint(checkpoint_path)
    if counts:
        print(f"  resuming from checkpoint: {len(counts)} (group, lang) calls already done, skipping those")

    tasks = [(gi, lang, text) for gi, lang, text in tasks_all if (gi, lang) not in counts]
    skipped = len(tasks_all) - len(tasks)

    if checkpoint_path:
        os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)
    checkpoint_file = open(checkpoint_path, "a", encoding="utf-8") if checkpoint_path else None
    checkpoint_lock = threading.Lock()

    errors = []
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            future_to_task = {ex.submit(count_fn, text): (gi, lang) for gi, lang, text in tasks}
            done = 0
            for fut in as_completed(future_to_task):
                gi, lang = future_to_task[fut]
                try:
                    count = fut.result()
                    counts[(gi, lang)] = count
                    if checkpoint_file is not None:
                        with checkpoint_lock:
                            checkpoint_file.write(json.dumps({"group": gi, "lang": lang, "count": count}) + "\n")
                            checkpoint_file.flush()
                except Exception as e:
                    cause = e.__cause__
                    detail = f"{type(e).__name__}: {e}"
                    if cause is not None:
                        # count()'s RuntimeError wraps the real reason via `raise ... from
                        # last_exc` -- unwrap it or every failure just says "failed after 5 attempts".
                        detail += f" (caused by {type(cause).__name__}: {cause})"
                    errors.append((gi, lang, detail))
                    print(f"  [FAILED] group={gi} lang={lang}: {detail}", flush=True)
                done += 1
                if progress_every and done % progress_every == 0:
                    print(
                        f"  ...{done}/{len(tasks)} count_tokens calls done this run "
                        f"({len(errors)} failed so far, {skipped} already skipped via checkpoint)"
                    )
    finally:
        if checkpoint_file is not None:
            checkpoint_file.close()

    if errors:
        print(
            f"  {len(errors)} of {len(tasks)} count_tokens calls failed and were dropped from aggregate stats"
        )

    compressions_by_lang = defaultdict(list)
    word_counts = defaultdict(int)
    token_counts_by_lang = defaultdict(int)
    paired_anchor_counts = defaultdict(list)
    paired_lang_counts = defaultdict(list)
    anchor_found = False

    for gi, group in enumerate(eval_groups):
        num_tokens_this_group = {}
        for lang, text in group.items():
            count = counts.get((gi, lang))
            if count is None:
                continue
            raw = text.encode("utf-8") if isinstance(text, str) else bytes(text)
            compressions_by_lang[lang].append(compression_rate(len(raw), count))
            word_counts[lang] += word_count(text)
            token_counts_by_lang[lang] += count
            num_tokens_this_group[lang] = count

        anchor_key = _find_anchor_key(num_tokens_this_group, anchor_lang)
        if anchor_key is not None:
            anchor_found = True
            anchor_count = num_tokens_this_group[anchor_key]
            for lang, count in num_tokens_this_group.items():
                paired_anchor_counts[lang].append(anchor_count)
                paired_lang_counts[lang].append(count)

    per_lang_compression = {lang: float(np.mean(v)) for lang, v in compressions_by_lang.items()}
    avg_compression = float(np.mean(list(per_lang_compression.values()))) if per_lang_compression else 0.0
    fertility_by_lang = {
        lang: fertility(token_counts_by_lang[lang], word_counts.get(lang, 0)) for lang in token_counts_by_lang
    }
    token_parity_anchor = anchor_lang if anchor_found else next(iter(token_counts_by_lang), anchor_lang)
    token_parity = {}
    for lang in token_counts_by_lang:
        a_counts = paired_anchor_counts.get(lang, [])
        l_counts = paired_lang_counts.get(lang, [])
        if a_counts and sum(a_counts) > 0:
            token_parity[lang] = (sum(l_counts) / len(l_counts)) / (sum(a_counts) / len(a_counts))
        else:
            token_parity[lang] = 1.0
    token_parity_gm, token_parity_spread = anchor_invariant_parity(token_parity)

    return {
        "per_lang_compression": per_lang_compression,
        "avg_compression": avg_compression,
        "fertility": fertility_by_lang,
        "token_parity": token_parity,
        "token_parity_anchor": token_parity_anchor,
        "token_parity_gm": token_parity_gm,
        "token_parity_spread": token_parity_spread,
        "renyi": {},
        "gini": None,
        "num_failed_calls": len(errors),
        "num_total_calls": len(tasks_all),
        "num_skipped_via_checkpoint": skipped,
    }


def evaluate_claude_on_indigenous_panel(
    eval_groups, count_fn, max_workers=25, progress_every=1000, checkpoint_path=None
):
    """Claude-specific analog of common.eval.cross_tokenizer.
    evaluate_on_indigenous_panel, built around evaluate_claude_on_groups the
    same way. This panel needs dedicated handling because it's mixed-anchor
    (English for crk-en/iu-en, Spanish for the ten AmericasNLP pairs -- see
    common.data.indigenous_panel) -- merging ratios computed against
    different anchors would silently reintroduce anchor bias.

    checkpoint_path: unlike the single-file case in evaluate_claude_on_groups,
    this runs TWO internal evaluate_claude_on_groups calls (one per anchor),
    each with its own checkpoint file ("{checkpoint_path}.{anchor}.jsonl")
    so a resumed run resumes each anchor's calls independently.

    Returns {"combined": {"avg_compression", "per_lang_compression",
    "fertility", "renyi": {} (always empty), "gini": None (always) --
    pooled across both anchor subgroups, safe since these are per-language,
    anchor-free}, "token_parity_by_anchor": {anchor: <that subgroup's
    evaluate_claude_on_groups result>}, "morphology_spread":
    {"fertility_spread", "compression_spread"} (max/min across the whole
    panel), "num_total_calls"/"num_failed_calls"/"num_skipped_via_checkpoint"
    (summed across both anchor subgroups)}.
    """
    from common.data.indigenous_panel import PAIRS

    known_anchors = {meta["anchor"] for meta in PAIRS.values()}
    groups_by_anchor = defaultdict(list)
    for group in eval_groups:
        anchors_present = known_anchors & set(group)
        if not anchors_present:
            raise ValueError(
                f"indigenous_panel group has none of this panel's known anchor languages "
                f"({sorted(known_anchors)}) as a key: {sorted(group)} -- see "
                "common.data.indigenous_panel.PAIRS"
            )
        for anchor in anchors_present:
            groups_by_anchor[anchor].append(group)

    token_parity_by_anchor = {}
    per_lang_compression = {}
    fertility_by_lang = {}
    total_calls = total_failed = total_skipped = 0
    for anchor, anchor_groups in groups_by_anchor.items():
        anchor_ckpt = f"{checkpoint_path}.{anchor}.jsonl" if checkpoint_path else None
        anchor_results = evaluate_claude_on_groups(
            anchor_groups, count_fn, anchor_lang=anchor, max_workers=max_workers,
            progress_every=progress_every, checkpoint_path=anchor_ckpt,
        )
        token_parity_by_anchor[anchor] = anchor_results
        per_lang_compression.update(anchor_results["per_lang_compression"])
        fertility_by_lang.update(anchor_results["fertility"])
        total_calls += anchor_results["num_total_calls"]
        total_failed += anchor_results["num_failed_calls"]
        total_skipped += anchor_results["num_skipped_via_checkpoint"]

    avg_compression = float(np.mean(list(per_lang_compression.values()))) if per_lang_compression else 0.0
    fertility_vals = [v for v in fertility_by_lang.values() if v > 0]
    compression_vals = [v for v in per_lang_compression.values() if v > 0]
    morphology_spread = {
        "fertility_spread": (max(fertility_vals) / min(fertility_vals)) if fertility_vals else 1.0,
        "compression_spread": (max(compression_vals) / min(compression_vals)) if compression_vals else 1.0,
    }

    return {
        "combined": {
            "avg_compression": avg_compression,
            "per_lang_compression": per_lang_compression,
            "fertility": fertility_by_lang,
            "renyi": {},
            "gini": None,
        },
        "token_parity_by_anchor": token_parity_by_anchor,
        "morphology_spread": morphology_spread,
        "num_total_calls": total_calls,
        "num_failed_calls": total_failed,
        "num_skipped_via_checkpoint": total_skipped,
    }


def report_claude_indigenous_panel_eval(results, label=""):
    prefix = f"[{label}] " if label else ""
    print(f"\n{prefix}indigenous_panel evaluation (mixed-anchor, compression/fertility only -- "
          f"see evaluate_claude_on_indigenous_panel's own docstring):")
    skipped = results.get("num_skipped_via_checkpoint", 0)
    skipped_note = f", {skipped} resumed from checkpoint" if skipped else ""
    print(f"  count_tokens calls: {results['num_total_calls']} total, {results['num_failed_calls']} failed{skipped_note}")
    print(
        f"  panel-wide fertility_spread={results['morphology_spread']['fertility_spread']:.3f}  "
        f"compression_spread={results['morphology_spread']['compression_spread']:.3f}"
    )
    combined = results["combined"]
    print(f"  avg_compression={combined['avg_compression']:.2f}")
    print("  per-language compression / fertility (anchor-free, comparable across the whole panel):")
    for lang in sorted(combined["per_lang_compression"]):
        print(
            f"    {lang}: compression={combined['per_lang_compression'][lang]:.2f}  "
            f"fertility={combined['fertility'].get(lang, 0.0):.2f}"
        )
    for anchor, anchor_results in sorted(results["token_parity_by_anchor"].items()):
        print(f"  token_parity vs anchor={anchor!r} (only comparable within this anchor's own languages):")
        token_parity = anchor_results["token_parity"]
        token_parity_gm = anchor_results["token_parity_gm"]
        for lang in sorted(token_parity):
            if lang == anchor:
                continue
            print(
                f"    {lang}: token_parity={token_parity[lang]:.3f}  "
                f"token_parity_gm={token_parity_gm.get(lang, 1.0):.3f}"
            )


def report_claude_eval(results, label=""):
    prefix = f"[{label}] " if label else ""
    print(f"\n{prefix}held-out evaluation (compression / fertility / token parity only -- "
          f"no renyi/gini, see systems/claude_tokenizer/model.py's own docstring for why):")
    print(f"  avg_compression={results['avg_compression']:.2f}")
    if results.get("num_total_calls"):
        skipped = results.get("num_skipped_via_checkpoint", 0)
        skipped_note = f", {skipped} resumed from checkpoint" if skipped else ""
        print(f"  count_tokens calls: {results['num_total_calls']} total, {results['num_failed_calls']} failed{skipped_note}")
    if "token_parity_spread" in results:
        print(f"  token_parity_spread (anchor-invariant, max/min across languages)={results['token_parity_spread']:.3f}")
    anchor = results.get("token_parity_anchor", "eng")
    print(
        f"  per-language compression / fertility / token parity vs {anchor}=1.0 / "
        f"anchor-invariant token parity vs the geometric mean=1.0:"
    )
    token_parity_gm = results.get("token_parity_gm", {})
    for lang in sorted(results["per_lang_compression"]):
        print(
            f"    {lang}: compression={results['per_lang_compression'][lang]:.2f}  "
            f"fertility={results['fertility'].get(lang, 0.0):.2f}  "
            f"token_parity={results['token_parity'].get(lang, 1.0):.3f}  "
            f"token_parity_gm={token_parity_gm.get(lang, 1.0):.3f}"
        )


def claude_wandb_log_dict(results, prefix="eval"):
    log_dict = {f"{prefix}/avg_compression": results["avg_compression"]}
    if "token_parity_spread" in results:
        log_dict[f"{prefix}/token_parity_spread"] = results["token_parity_spread"]
    log_dict.update({f"{prefix}/compression/{lang}": v for lang, v in results["per_lang_compression"].items()})
    log_dict.update({f"{prefix}/fertility/{lang}": v for lang, v in results["fertility"].items()})
    log_dict.update({f"{prefix}/token_parity/{lang}": v for lang, v in results["token_parity"].items()})
    log_dict.update({f"{prefix}/token_parity_gm/{lang}": v for lang, v in results.get("token_parity_gm", {}).items()})
    return log_dict


def main(argv=None):
    args = parse_args_with_config(build_arg_parser(), argv)
    models = [m.strip() for m in args.model.split(",")]

    eval_groups = _load_eval_groups(args)
    print(
        f"eval_data_source={args.eval_data_source} groups={len(eval_groups)} models={models} "
        f"rpm={args.rpm} max_workers={args.max_workers}"
    )

    # ONE shared rate limiter across every model in this run -- see module docstring for why.
    rate_limiter = RateLimiter(max_calls=args.rpm, period=60.0)

    checkpoint_dir = args.checkpoint_dir or (f"{args.output}.checkpoint" if args.output else None)
    if checkpoint_dir:
        print(f"checkpointing enabled: {checkpoint_dir}/<model>.jsonl (resumes automatically if this run is interrupted)")
    else:
        print("no --output/--checkpoint-dir given -- checkpointing disabled, an interrupted run can't resume")

    is_indigenous_panel = args.eval_data_source == "indigenous_panel"
    all_results = {}
    failed = {}
    for model in models:
        checkpoint_path = os.path.join(checkpoint_dir, f"{model}.jsonl") if checkpoint_dir else None
        try:
            counter = ClaudeTokenCounter(model, rate_limiter, api_key=args.api_key)
            if is_indigenous_panel:
                results = evaluate_claude_on_indigenous_panel(
                    eval_groups, counter.count, max_workers=args.max_workers, checkpoint_path=checkpoint_path
                )
            else:
                results = evaluate_claude_on_groups(
                    eval_groups, counter.count, max_workers=args.max_workers, checkpoint_path=checkpoint_path
                )
        except Exception as e:
            print(f"\n[{model}] FAILED: {type(e).__name__}: {e}")
            failed[model] = f"{type(e).__name__}: {e}"
            continue
        if is_indigenous_panel:
            report_claude_indigenous_panel_eval(results, label=model)
        else:
            report_claude_eval(results, label=model)
        all_results[model] = results

    if failed:
        print(f"\n{len(failed)}/{len(models)} model(s) failed: {list(failed)} -- see FAILED lines above for why")
        all_results["_failed"] = failed

    payload = json.dumps(all_results, indent=2)
    if args.output:
        with open(args.output, "w") as f:
            f.write(payload)
        print(f"\nwrote combined results to {args.output}")

    if args.use_wandb:
        import wandb

        run = wandb.init(
            project=args.wandb_project,
            name=args.run_name or None,
            job_type="eval",
            config={
                "models": models,
                "eval_data_source": args.eval_data_source,
                "num_groups": args.num_groups,
                "rpm": args.rpm,
                "max_workers": args.max_workers,
                "checkpoint_dir": checkpoint_dir,
                "failed_models": list(failed),
            },
        )
        successful = {m: r for m, r in all_results.items() if m != "_failed"}
        # indigenous_panel nests avg_compression under "combined";
        # num_total_calls/num_failed_calls stay top-level either way.
        avg_compression_of = (lambda r: r["combined"]["avg_compression"]) if is_indigenous_panel else (lambda r: r["avg_compression"])
        summary_rows = [
            [m, avg_compression_of(r), r["num_total_calls"], r["num_failed_calls"]] for m, r in successful.items()
        ]
        run.log(
            {
                "comparison": wandb.Table(
                    columns=["model", "avg_compression", "num_total_calls", "num_failed_calls"],
                    data=summary_rows,
                ),
            }
        )
        for model, r in successful.items():
            if is_indigenous_panel:
                log_dict = {
                    f"{model}/avg_compression": r["combined"]["avg_compression"],
                    **{f"{model}/compression/{lang}": v for lang, v in r["combined"]["per_lang_compression"].items()},
                    **{f"{model}/fertility/{lang}": v for lang, v in r["combined"]["fertility"].items()},
                    **{f"{model}/morphology_spread/{k}": v for k, v in r["morphology_spread"].items()},
                }
                for anchor, anchor_results in r["token_parity_by_anchor"].items():
                    log_dict.update(
                        {
                            f"{model}/token_parity_vs_{anchor}/{lang}": v
                            for lang, v in anchor_results["token_parity"].items()
                            if lang != anchor
                        }
                    )
                run.log(log_dict)
            else:
                run.log(claude_wandb_log_dict(r, prefix=model))
        run.finish()
        print(f"logged comparison to wandb project={args.wandb_project!r}")

    return all_results


def run_smoke_test():
    """This module's real substance (calls to a paid/authenticated API)
    can't be exercised without a live ANTHROPIC_API_KEY, so this checks
    three things independent of that:

    1. RateLimiter's pacing -- fill a tiny window to capacity, confirm a
       further acquire() actually blocks until the window rolls over.
    2. evaluate_claude_on_groups's aggregation math (compression/fertility/
       token_parity) against a FAKE count_fn (word-count stand-in), so a
       regression fails loudly here regardless of API reachability.
    3. Checkpoint resume -- pre-seed one (group, lang) as already "done",
       confirm it's never re-queried (spy count_fn), reflected in
       num_skipped_via_checkpoint, and that aggregate stats exactly match
       an uninterrupted full run.

    If ANTHROPIC_API_KEY is set, also makes one real count_tokens call as a
    bonus check; skipped (not failed) if no key is set."""
    import tempfile
    import time

    from .model import RateLimiter

    # 1. RateLimiter pacing.
    limiter = RateLimiter(max_calls=3, period=0.5)
    t0 = time.monotonic()
    for _ in range(3):
        limiter.acquire()  # should return immediately, capacity not yet hit
    assert time.monotonic() - t0 < 0.3, "first 3 acquires should not have blocked"
    t1 = time.monotonic()
    limiter.acquire()  # capacity hit -- must wait out the rest of the 0.5s window
    waited = time.monotonic() - t1
    assert waited > 0.2, f"4th acquire should have blocked for the window to roll over, only waited {waited:.3f}s"

    # 2. Aggregation math against a fake, non-network count function.
    fake_groups = [
        {"eng": "the quick brown fox", "deu": "der schnelle braune fuchs springt"},
        {"eng": "hello world", "deu": "hallo welt"},
    ]

    def fake_count(text):
        return len(text.split())  # deterministic, no network

    results = evaluate_claude_on_groups(fake_groups, fake_count, max_workers=4, progress_every=0)
    assert results["renyi"] == {}, "renyi must always be empty -- not available from a bare count"
    assert results["gini"] is None, "gini must always be None -- not available from a bare count"
    assert set(results["per_lang_compression"]) == {"eng", "deu"}
    assert results["token_parity"]["eng"] == 1.0, "anchor's own parity must always be exactly 1.0"
    assert results["num_total_calls"] == 4 and results["num_failed_calls"] == 0
    assert results["num_skipped_via_checkpoint"] == 0, "no checkpoint given -- nothing should be skipped"
    assert set(results["token_parity_gm"]) == {"eng", "deu"}
    assert results["token_parity_spread"] >= 1.0
    json.dumps(results, default=str)  # confirms the result dict is actually JSON-serializable

    # Anchor-invariance: re-run with anchor_lang="deu" instead of "eng" and confirm
    # token_parity_gm is identical (see common.eval.parity.anchor_invariant_parity).
    deu_anchored = evaluate_claude_on_groups(fake_groups, fake_count, anchor_lang="deu", max_workers=4, progress_every=0)
    for lang in results["token_parity_gm"]:
        assert abs(results["token_parity_gm"][lang] - deu_anchored["token_parity_gm"][lang]) < 1e-9, (
            f"token_parity_gm[{lang!r}] must be anchor-invariant"
        )
    assert abs(results["token_parity_spread"] - deu_anchored["token_parity_spread"]) < 1e-9

    # 3. Checkpoint resume, against the exact same fake_groups/fake_count.
    with tempfile.TemporaryDirectory() as d:
        ckpt = os.path.join(d, "ckpt.jsonl")
        # Pre-seed as if (group=0, lang="eng") already completed in a prior, killed run.
        with open(ckpt, "w", encoding="utf-8") as f:
            f.write(json.dumps({"group": 0, "lang": "eng", "count": fake_count(fake_groups[0]["eng"])}) + "\n")

        queried = []

        def spy_count(text):
            queried.append(text)
            return fake_count(text)

        resumed = evaluate_claude_on_groups(fake_groups, spy_count, max_workers=4, progress_every=0, checkpoint_path=ckpt)
        assert fake_groups[0]["eng"] not in queried, "a checkpoint-resumed (group, lang) must not be re-queried"
        assert resumed["num_skipped_via_checkpoint"] == 1
        assert resumed["num_total_calls"] == 4, "total is the full dataset size, not just this run's remaining tasks"
        for key in ("avg_compression", "per_lang_compression", "fertility", "token_parity"):
            assert resumed[key] == results[key], f"resumed[{key!r}] must exactly match an uninterrupted full run"

        with open(ckpt, encoding="utf-8") as f:
            lines = [json.loads(line) for line in f if line.strip()]
        assert len(lines) == 4, "checkpoint file should end up recording all 4 (group, lang) pairs"

    print("\nclaude_tokenizer smoke test passed (RateLimiter pacing + aggregation math + checkpoint resume, no real API call).")

    if os.environ.get("ANTHROPIC_API_KEY"):
        from .model import ClaudeTokenCounter

        counter = ClaudeTokenCounter("claude-haiku-4-5-20251001", RateLimiter(max_calls=10, period=60.0))
        n = counter.count("Hello, Claude")
        assert isinstance(n, int) and n > 0
        print(f"  bonus: a real count_tokens call succeeded (input_tokens={n}).")
    else:
        print("  ANTHROPIC_API_KEY not set -- skipping the real-API bonus check (expected in most dev environments).")

    return results


if __name__ == "__main__":
    main()
