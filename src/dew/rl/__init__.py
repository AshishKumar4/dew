"""Reinforcement learning on plain arrays and pure functions: its math, its
environments, their physics and renderer, a replay and simulators.

The math is `advantage`, which turns rewards into advantages, `surrogate`,
which turns log-probabilities and advantages into a loss, and
`distributions`, DreamerV3's output transforms and categoricals.
`advantage` and `surrogate` know nothing of a model, an objective, the
trainer or a batch dict. An objective combines them, and the package can be
tested on fixed tensors.

An environment is an `env.Env`, a reset and a step, both jittable over a
state, with its batching, its time limit and `env_rollout`, the one scan
that acts in it. `playground` makes MuJoCo Playground's environments Envs,
`simulator` is the boundary for a simulator outside JAX, stepped from the
host, and `replay` keeps environment steps on the device. A first-person
drone race is built from them. `quadrotor` holds a racing drone's physics in
JAX, `flight` races it through gates as an `Env` (`FlightEnv`) and, seen
through a camera, as a `Simulator` (`GateRace`), and `hyperlux` draws that
camera with a ray tracer. This package re-exports the names of `advantage`
and `surrogate`; every other module is imported by its own name, as
`dew.rl.env`.

Imports go one way. `dew.rl` may import from `dew`, but not from
`dew.training`, and nothing in `dew` outside `dew.rl` and
`dew.objectives.rl` may import `dew.rl`. That way, splitting it into a
separate distribution only means moving a directory.

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
