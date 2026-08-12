"""FlexiTokens baseline: a clean-room reimplementation of the differentiable,
boundary-predictor-based byte-level tokenizer described in the FlexiTokens paper
(ACL Findings 2026, https://aclanthology.org/2026.findings-acl.848.pdf), sized and
adapted for comparison against this project's own fairtok/ package. See
flexitokens/model.py's module docstring for the full architecture description,
every scale-down relative to the paper, and every judgment call made where the
paper's description doesn't fully pin down a mechanism.

Left empty otherwise, matching fairtok/__init__.py's own convention -- submodules
are imported directly (`from flexitokens.model import FlexiTokensModel`,
`from flexitokens.train import FlexiTokensConfig, FlexiTokensTrainer`, etc.), not
re-exported through this file.
"""
