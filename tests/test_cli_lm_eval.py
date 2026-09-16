"""Wraps systems.pretraining.cli_lm_eval.run_smoke_test as a pytest test,
matching test_cli_generate.py's own convention. Needs network access
(downloads blimp's real dataset via lm-evaluation-harness, cached after
the first run) -- unlike most of this project's own smoke tests."""

from systems.pretraining import cli_lm_eval


def test_lm_eval_adapter_smoke_test_passes():
    cli_lm_eval.run_smoke_test()
