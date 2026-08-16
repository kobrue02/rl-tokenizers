"""Pretrain an actual language model using any of the seven systems/
tokenizers, then evaluate it on downstream benchmarks. Four pieces, run in
order:

  1. systems.pretraining.data_prep -- stream a corpus from common.data.corpora's
     shared registry (the same one tokenizer training uses), tokenize with
     a chosen systems/ checkpoint (systems.pretraining.tokenizer_adapter), pack
     into token shards on disk.
  2. systems.pretraining.cli / systems.pretraining.train -- train a LLaMA-style
     TransformerLM (systems.pretraining.model) over those shards, at any named
     size in systems.pretraining.model_configs (tiny through 7b).
  3. See train.py's own docstring for what's verified (single/multi-GPU
     DDP) vs. not yet built (FSDP/sharding, needed for the 7b preset).
  4. systems.pretraining.cli_eval / systems.pretraining.eval_harness -- evaluate a trained
     checkpoint on downstream benchmarks (systems.pretraining.benchmarks: XNLI,
     XCOPA, FLORES-MT) through the same tokenizer_adapter. Infrastructure
     verified via cli_eval.run_smoke_test against a tiny freshly-
     initialized model, not against real pretraining-run numbers.

This package never trains a tokenizer itself -- only consumes an
already-trained systems/ checkpoint through tokenizer_adapter.
"""
