"""Pretrain an actual language model using any of the seven systems/
tokenizers, then evaluate it on downstream benchmarks. Four pieces, run in
order:

  1. pretraining.data_prep -- stream a corpus from common.data.corpora's shared
     registry (the SAME registry common.data.cli_data.load_groups uses for
     tokenizer training -- oldi_seed/flores_dev/smol/glot500/fineweb_edu/
     olmo_mix/ccmatrix/un_pc/europarl/tatoeba_mt/bible_nlp, no separate
     pretraining-only source list), tokenize with a
     chosen systems/ checkpoint (pretraining.tokenizer_adapter), pack into
     token shards on disk.
  2. pretraining.cli / pretraining.train -- train a LLaMA-style
     TransformerLM (pretraining.model) over those shards, at any of the
     named sizes in pretraining.model_configs (tiny through 7b).
  3. See train.py's own module docstring for what's verified (single- and
     multi-GPU DDP) vs. explicitly not yet built (FSDP/sharding, needed to
     actually fit the 7b preset across multiple GPUs' memory).
  4. pretraining.cli_eval / pretraining.eval_harness -- evaluate a trained
     checkpoint on downstream benchmarks (pretraining.benchmarks: XNLI,
     XCOPA, FLORES-MT) through the same tokenizer_adapter used at both
     prior stages. Infrastructure verified via cli_eval.run_smoke_test
     against a tiny freshly-initialized model, not against real numbers
     from an actual pretraining run -- see eval_harness.py's own docstring.

Independent of systems/ in one important sense: this package never trains a
tokenizer itself, only ever CONSUMES an already-trained systems/ checkpoint
through tokenizer_adapter's unified interface.
"""
