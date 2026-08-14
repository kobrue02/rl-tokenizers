"""Not a trained tokenizer this project fits itself -- a thin wrapper over
an ARBITRARY HuggingFace model repo's own tokenizer (deepseek-ai/DeepSeek-V4-Pro,
moonshotai/Kimi-K3, meta-llama/Llama-3.1-8B-Instruct, or any other repo with
a fast/Rust-backed tokenizer -- see model.py's own docstring), loaded
TOKENIZER-ONLY (confirmed directly: AutoTokenizer.from_pretrained never
requests a repo's model weight files), so a real frontier model's own
tokenizer can be scored on IDENTICAL held-out data (BOUQuET) with the SAME
fairness/efficiency metrics (common.eval.cross_tokenizer) every from-scratch
tokenizer in systems/ already reports -- a direct, apples-to-apples
comparison against fairtok/magnet/flexitokens/manta/fanta/superbpe/bpe, not
a separate benchmark.

No train.py/cli.py here (unlike every other systems/ package) -- there's
nothing to fit, only something to load and evaluate. See evaluate.py for
the entry point, model.py for the loading + span-reconstruction mechanism.
"""
