"""Thin wrapper over an ARBITRARY HuggingFace model repo's own tokenizer
(e.g. deepseek-ai/DeepSeek-V4-Pro, moonshotai/Kimi-K3, meta-llama/Llama-3.1-
8B-Instruct), loaded TOKENIZER-ONLY (AutoTokenizer.from_pretrained never
fetches model weights), so a real frontier tokenizer can be scored on the
same held-out data (BOUQuET) with the same metrics (common.eval.cross_tokenizer)
as every from-scratch tokenizer in systems/.

No train.py/cli.py here -- there's nothing to fit, only something to load
and evaluate. See evaluate.py for the entry point, model.py for loading +
span reconstruction.
"""
