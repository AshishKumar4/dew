"""Reinforcement-learning math on plain arrays, with advantages and losses in separate modules.

`advantage` turns rewards into advantages, and `surrogate` turns
log-probabilities and advantages into a loss. Neither knows about a model,
an objective, the trainer or a batch dict. An objective combines them, and
the package can be tested on fixed tensors.

Imports go one way. `dew.rl` may import from `dew`, but nothing in `dew`
outside `dew.rl` and `dew.objectives.rl` may import `dew.rl`. That way,
splitting it into a separate distribution only means moving a directory
(`docs/design/plan.md`, section 5.1).

`sandbox` runs untrusted programs and tool sessions in bounded subprocesses
or containers, and a verifiable reward uses it to score completions. It
imports the episode types from `dew.objectives.rl`, so you import it as
`dew.rl.sandbox`; this package does not re-export it.

`advantage` and `surrogate` port Apache-2.0 code from Tunix and verl and
include their notice. The rest of Dew is MIT.
"""

from .advantage import gae, group_advantage, masked_mean, masked_whiten, rloo_advantage
from .surrogate import (
    behavior_importance_weights,
    clipped_surrogate_terms,
    clipped_value_loss_terms,
    k3_kl,
    preference_logsigmoid_terms,
    sequence_log_ratio,
    token_log_ratio,
)

__all__ = [
    "behavior_importance_weights",
    "clipped_surrogate_terms",
    "clipped_value_loss_terms",
    "gae",
    "group_advantage",
    "k3_kl",
    "masked_mean",
    "masked_whiten",
    "preference_logsigmoid_terms",
    "rloo_advantage",
    "sequence_log_ratio",
    "token_log_ratio",
]
