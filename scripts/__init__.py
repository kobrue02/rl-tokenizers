"""Standalone utility scripts (figure generation, result-file merging, one-off
backfills) that aren't part of the train.py/evaluate.py dispatch pipeline. Run as
modules from the repo root, e.g. `python3 -m scripts.generate_tikz_figures`, so
their own `from common...`/`from systems...` imports resolve correctly.
"""
