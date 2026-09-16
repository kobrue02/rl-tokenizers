"""Wraps this project's own TransformerLM + TokenizerAdapter as an
lm_eval.api.model.LM, so EleutherAI/lm-evaluation-harness's own registered
tasks -- confirmed usable here (see their own task directories on
github.com/EleutherAI/lm-evaluation-harness for the exact templates/scoring
shape each reuses, rather than a hand-rolled equivalent):

  - xnli, xcopa, blimp -- real per-language LOCALIZED templates for
    xnli/xcopa (not English scaffolding reused across languages, this
    project's own benchmarks.py's known limitation), confirmed directly
    against their yaml configs.
  - xstorycloze -- 2-way narrative-continuation commonsense, same
    difficulty class/output_type as xcopa. 11 languages.
  - lambada_openai_mt (lambada_multilingual group) -- predict-the-final-
    word, output_type=loglikelihood directly (no multiple-choice framing
    at all -- the simplest possible integration). 5 languages.
  - global_piqa (nonparallel_cloze / parallel_cloze configs) -- 2-way
    physical-commonsense reasoning; ITS OWN README explicitly recommends
    the cloze config for base/small non-instruction-tuned models,
    confirmed by reading it directly. ~130+ language/dialect configs,
    keyed by FLORES-style lang_Script stems.

-- can be run against any of this project's tokenizer systems' pretrained
checkpoints, through the SAME TokenizerAdapter interface every one of them
already shares. One wrapper class covers all ~10 systems (bpe/superbpe/
fanta/magnet/flexitokens/manta/fairtok/parity_bpe/blt/hf_frontier), not one
per system.

WHY these specifically, not this project's own flores_mt/cola/squad:
confirmed via a direct scan of the harness's own task registry that
FLORES-MT has no equivalent task there at all (stays this project's own
custom eval_harness.evaluate_translation); cola/squad exist in some form
but haven't been verified to fit a non-instruction-tuned base LM better
than what's already implemented, so they're left as this project's own
code for now; mgsm/tydiqa/xquad/mlqa/multiblimp/afrixnli were surveyed and
deprioritized (math is floor-bound at this model scale regardless of
tokenizer; the QA tasks are generate_until, the same decoding difficulty
already identified as a weak point for this project's own squad/flores_mt;
the last two need real per-config curation effort before they're usable).
BOUQuET-based intrinsic tokenizer eval, the indigenous language panel, and
contamination checking are project-specific and have no general-purpose-
harness equivalent at all.

Reuses eval_harness._loglikelihood_full's own joint-encoding logic (handles
a token straddling the context/continuation boundary correctly, e.g.
superbpe's whitespace-crossing merges; already deals with the empty-context
eos_id-prepend case and max_seq_len left-truncation) rather than
reimplementing any of it -- this module only adds the two things that
function doesn't: generate_until's stop-string cutting (needed for any
harness task using free-form generation, not multiple-choice scoring) and
the create_from_arg_string/register_model plumbing lm_eval's own model
registry expects.

UNBATCHED, matching every other evaluator in eval_harness.py (loglikelihood/
evaluate_multiple_choice/evaluate_cola/evaluate_qa are all one-example-at-
a-time already) -- HFLM's own real batching is a throughput optimization
this project's evaluation code has never had, not a correctness
requirement; adding it here would be new, unverified complexity for a
speed win only, not attempted.

See cli_lm_eval.py for the actual command-line entry point that constructs
one of these and calls lm_eval.simple_evaluate with it.
"""

import torch
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from lm_eval.models.utils import handle_stop_sequences, normalize_gen_kwargs
from tqdm.auto import tqdm

from .eval_harness import _encode_tensor, _loglikelihood_full

# task_name prefixes this project's TokenizerAdapter.encode() can extract a
# real per-language `lang` hint from (used by MAGNET's per-script boundary
# predictor; harmless no-op for every other system, see TokenizerAdapter.
# encode's own docstring). Anything else (e.g. blimp's paradigm-named
# tasks, "adjunct_island" etc. -- English-only, no per-script need) falls
# back to lang=None. Extend this tuple as more harness tasks get adopted --
# confirmed each of these by fetching the harness's own real per-language
# yaml configs directly (not assumed from the task family's name alone):
#   xnli_<lang>, xcopa_<lang> -- e.g. "xnli_sw" -> "sw"
#   xstorycloze_<lang> -- e.g. "xstorycloze_ar" -> "ar"
#   lambada_openai_mt_<lang> -- e.g. "lambada_openai_mt_de" -> "de"
#   global_piqa_{nonparallel,parallel}_cloze_<lang_Script> -- e.g.
#     "global_piqa_nonparallel_cloze_als_latn" -> "als_latn", a FLORES-style
#     lang_Script stem (lowercase script suffix; TokenizerAdapter's own
#     script resolver -- see tokenizer_adapter.py's own docstring -- expects
#     this shape already, "eng_Latn"-style, just case-insensitively here).
#     NOTE: a handful of global_piqa's dialectal-Arabic configs use a
#     THREE-part stem with a region suffix (e.g. "apc_arab_jord") -- the
#     naive `rsplit("_", 1)` in tokenizer_adapter.py's own script resolver
#     would extract "jord" as the "script" for those, not "arab", a
#     pre-existing narrow limitation of that resolver (not new here,
#     affects any 3-part lang_Script-like string) that only matters for
#     --system magnet specifically; every other tokenizer system ignores
#     the lang hint's exact shape entirely.
_LANG_PREFIXES = (
    "xnli_",
    "xcopa_",
    "xstorycloze_",
    "lambada_openai_mt_",
    "global_piqa_nonparallel_cloze_",
    "global_piqa_parallel_cloze_",
)


def _lang_from_task_name(task_name):
    if not task_name:
        return None
    for prefix in _LANG_PREFIXES:
        if task_name.startswith(prefix):
            return task_name[len(prefix) :]
    return None


@register_model("thesis_lm")
class ThesisLM(LM):
    """model_args (lm-eval's own bare CLI, via --model_args "k=v,k=v")
    accepts the same four fields cli_eval.py's own --checkpoint/--system/
    --tokenizer-checkpoint/--vocab-json/--device flags take: checkpoint=,
    system=, tokenizer_checkpoint=, vocab_json= (optional), device=
    (default cpu). See create_from_arg_string.

    Prefer constructing directly with an already-loaded (model, adapter)
    pair (from_pretrained below) when calling from this project's own
    cli_lm_eval.py -- avoids re-parsing a string just to immediately
    reconstruct what cli_lm_eval.py's own argparse flags already parsed.
    """

    def __init__(self, model=None, adapter=None, device="cpu"):
        super().__init__()
        if model is None or adapter is None:
            raise ValueError(
                "ThesisLM needs both model and adapter -- use ThesisLM.from_pretrained(...) "
                "or create_from_arg_string(...), not the bare constructor directly"
            )
        self.model = model
        self.adapter = adapter
        self._device = device
        self._rank = 0
        self._world_size = 1

    @classmethod
    def from_pretrained(cls, checkpoint, system, tokenizer_checkpoint, vocab_json=None, device="cpu"):
        """The path cli_lm_eval.py's own main() uses -- identical
        model/adapter construction to cli_eval.load_pretrained_model +
        TokenizerAdapter.load, just wrapped as a classmethod here so both
        entry points build the exact same objects the exact same way."""
        from .cli_eval import load_pretrained_model
        from .tokenizer_adapter import TokenizerAdapter

        model = load_pretrained_model(checkpoint, device=device)
        adapter = TokenizerAdapter.load(system, tokenizer_checkpoint, vocab_json_path=vocab_json, device=device)
        return cls(model=model, adapter=adapter, device=device)

    @classmethod
    def create_from_arg_string(cls, arg_string, additional_config=None):
        """Supports lm-eval's own bare `lm_eval --model thesis_lm --model_args
        checkpoint=...,system=...,tokenizer_checkpoint=...` CLI invocation
        (arg_string is a comma-separated k=v list) -- NOT the path
        cli_lm_eval.py itself uses (see from_pretrained), but kept working
        so this model is usable through lm-eval's own generic CLI too, not
        just this project's own wrapper script."""
        kwargs = {}
        for pair in arg_string.split(","):
            pair = pair.strip()
            if not pair:
                continue
            key, _, value = pair.partition("=")
            kwargs[key] = value
        kwargs.update(additional_config or {})
        return cls.from_pretrained(**kwargs)

    def loglikelihood(self, requests, disable_tqdm=False):
        """requests: list[Instance], each .args == (context, continuation).
        Returns list[(logprob, is_greedy)] -- see eval_harness.
        _loglikelihood_full's own docstring for exactly what is_greedy
        means and why it lives there, not duplicated here."""
        results = []
        for req in tqdm(requests, disable=disable_tqdm, desc="loglikelihood"):
            context, continuation = req.args
            lang = _lang_from_task_name(req.task_name)
            total_lp, _, is_greedy = _loglikelihood_full(
                self.model, self.adapter, context, continuation, lang, self._device
            )
            results.append((total_lp, is_greedy))
        return results

    def loglikelihood_rolling(self, requests, disable_tqdm=False):
        """requests: list[Instance], each .args == (text,) -- the string's
        own full log-likelihood, unconditional (empty context), for
        perplexity-style tasks. None of xnli/xcopa/blimp (this module's
        actual integration target) use this output_type; implemented for
        ABC completeness with the simple whole-string case, NOT the
        max_seq_len-chunked-and-averaged version LM.loglikelihood_rolling's
        own docstring describes for documents longer than one context
        window -- eval_harness._loglikelihood_full raises for a
        continuation that can't fit in one window, so a future task
        actually needing this would surface that clearly rather than
        silently truncating."""
        results = []
        for req in tqdm(requests, disable=disable_tqdm, desc="loglikelihood_rolling"):
            (text,) = req.args
            total_lp, _, _ = _loglikelihood_full(self.model, self.adapter, "", text, None, self._device)
            results.append(total_lp)
        return results

    def generate_until(self, requests, disable_tqdm=False):
        """requests: list[Instance], each .args == (context, gen_kwargs).
        gen_kwargs (normalized via lm_eval's own normalize_gen_kwargs, the
        same helper HFLM uses) carries do_sample/until/max_gen_toks/
        temperature. do_sample=False (the common case: "generate the most
        likely completion") maps to top_k=1 on TransformerLM.generate --
        restricting multinomial sampling to a single nonzero-probability
        candidate makes it deterministic/argmax-equivalent without needing
        a separate greedy code path in model.py. `until` stop strings are
        applied by STRING-truncating the full max_gen_toks-token decode at
        the EARLIEST occurrence of any of them (TransformerLM.generate has
        no native stopping-criterion support -- same post-hoc-cut approach
        eval_harness.evaluate_translation/evaluate_qa already use for their
        own single hardcoded "\\n" stop, generalized here to an arbitrary
        stop-string list)."""
        results = []
        for req in tqdm(requests, disable=disable_tqdm, desc="generate_until"):
            context, raw_gen_kwargs = req.args
            gen_kwargs = normalize_gen_kwargs(raw_gen_kwargs or {}, default_max_gen_toks=128)
            until = handle_stop_sequences(gen_kwargs.get("until"), eos=None)
            max_gen_toks = gen_kwargs["max_gen_toks"]
            do_sample = gen_kwargs.get("do_sample", False)
            temperature = gen_kwargs.get("temperature", 1.0) if do_sample else 1.0
            top_k = None if do_sample else 1

            lang = _lang_from_task_name(req.task_name)
            ids, ids_tensor = _encode_tensor(self.adapter, context, lang, self._device)
            generated = self.model.generate(
                ids_tensor, max_new_tokens=max_gen_toks, temperature=temperature, top_k=top_k
            )
            new_ids = generated[0, len(ids) :].tolist()
            text = self.adapter.decode(new_ids).decode("utf-8", errors="replace")

            cut = len(text)
            for stop in until:
                idx = text.find(stop)
                if idx != -1:
                    cut = min(cut, idx)
            results.append(text[:cut])
        return results
