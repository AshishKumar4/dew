"""Reinforcement learning as array math, one module per function.

`advantage` turns rewards into advantages, `surrogate` turns log-probabilities
and advantages into a scalar loss. Neither knows about a model, an objective,
the trainer or a batch dict, so an objective composes them and this package
stays testable on fixed tensors.

The import arrow points one way. `dew.rl` may read `dew`; nothing under `dew`
outside `dew.rl` and `dew.objectives.rl` may read `dew.rl`, which is what keeps
the split into a separate distribution a directory move
(`docs/design/plan.md`, section 5.1). tests/test_rl_imports.py walks the tree
and fails when either half of that stops being true.

`advantage` and `surrogate` port Apache-2.0 code from Tunix and verl and carry
their notice; the rest of Dew is MIT.
"""

from .advantage import gae, group_advantage, masked_mean, masked_whiten, rloo_advantage
from .surrogate import (
    clipped_surrogate, k3_kl, sequence_log_ratio, token_log_ratio, token_mean,
)

__all__ = [
    "gae",
    "group_advantage",
    "masked_mean",
    "masked_whiten",
    "rloo_advantage",
    "clipped_surrogate",
    "k3_kl",
    "sequence_log_ratio",
    "token_log_ratio",
    "token_mean",
]
