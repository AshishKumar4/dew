"""The named RL methods as compositions of Dew's parts: an advantage estimator, a ratio and its
clipping (the policy loss), a reference penalty in the loss or in the reward, and a reduction, with
DAPO's dynamic sampling and overlong punishment where it reads them. Each method is its paper's
objective, held to it by tests/test_rl_methods.py; a run names one (`PromptRollout.method`) and may
set only what the method leaves free: its group size, its reference penalty's weight where it has one
(`beta` in the loss, `reward_kl` in the reward), and a clip range its paper tunes.

| method | estimator | policy loss | clip (low, high) | reference penalty | reduction |
|---|---|---|---|---|---|
| grpo (arXiv:2402.03300) | group | ppo | 0.2, 0.2 | k3 in the loss, beta | session-mean |
| dapo (arXiv:2503.14476) | group | ppo | 0.2, 0.28 | none | token-mean; varied groups; overlong |
| dr_grpo (arXiv:2503.20783) | mean | ppo | 0.2, 0.2 | none | constant (the budget) |
| gspo (arXiv:2507.18071) | group | gspo | the run's | none | session-mean |
| cispo (arXiv:2506.13585) | group | cispo | none below, the run's above | none | token-mean |
| gmpo (arXiv:2507.20673) | group | gmpo | e^-0.4, e^0.4 | none | session-mean |
| reinforce_pp (arXiv:2501.03262) | global | ppo | 0.2, 0.2 | k2 in the loss, beta | session-mean |
| rloo (arXiv:2402.14740) | rloo | reinforce | none | k1 in the reward, reward_kl | constant (one token) |
"""

import dataclasses
import math
from collections.abc import Mapping
from typing import TypedDict

from .grpo import GRPOObjective


class Rollout(TypedDict, total=False):
    """A method's scheduler settings (`RolloutScheduler`): its estimator, DAPO's dynamic sampling, and
    its overlong punishment's budget and cache where it has one."""

    estimator: str
    varied: bool
    overlong: tuple[int, int]


_NO_PENALTY = {"beta": 0.0, "reward_kl": 0.0}


@dataclasses.dataclass(frozen=True)
class Method:
    """One named method: the `GRPOObjective` settings it fixes (`objective`, a penalty it has none of
    fixed at 0) and its rollout's (the estimator, `RolloutScheduler.varied` and `.overlong`). `budget`
    is whether its reduction divides by the response budget (Dr. GRPO's constant)."""

    objective: Mapping[str, object]
    estimator: str
    varied: bool = False
    overlong: bool = False
    budget: bool = False

    def settings(self, max_new_tokens: int) -> Mapping[str, object]:
        """The objective settings the method fixes for a response budget of `max_new_tokens`."""
        settings = dict(self.objective)
        if self.budget:
            settings["aggregation_tokens"] = max_new_tokens
        return settings

    def rollout(self, max_new_tokens: int, overlong_cache: int | None = None) -> Rollout:
        """The scheduler's settings for a response budget of `max_new_tokens`: the estimator, DAPO's
        dynamic sampling, and its overlong punishment over the last `overlong_cache` tokens of the
        budget (a quarter of it by default)."""
        settings = Rollout(estimator=self.estimator, varied=self.varied)
        if self.overlong:
            settings["overlong"] = (max_new_tokens, overlong_cache or max(1, max_new_tokens // 4))
        return settings

    def build(self, model, seq_len: int, *, max_new_tokens: int, overlong_cache: int | None = None,
              **options) -> tuple[GRPOObjective, Rollout]:
        """The method's objective over `model` and its rollout's settings (`rollout`); `options` set what
        the method leaves free, and one it fixes is refused."""
        settings = self.settings(max_new_tokens)
        fixed = sorted(set(settings) & set(options))
        if fixed:
            raise ValueError(f"the method fixes {fixed}")
        return (GRPOObjective(model, seq_len, **settings, **options),
                self.rollout(max_new_tokens, overlong_cache))


METHODS: dict[str, Method] = {
    "grpo": Method({"policy_loss": "ppo", "aggregation": "session-mean", "dual_clip": None, "kl_term": "k3",
                    "reward_kl": 0.0}, "group"),
    "dapo": Method({"policy_loss": "ppo", "aggregation": "token-mean", "epsilon_low": 0.2,
                    "epsilon_high": 0.28, "dual_clip": None, **_NO_PENALTY}, "group",
                   varied=True, overlong=True),
    "dr_grpo": Method({"policy_loss": "ppo", "aggregation": "constant", "dual_clip": None, **_NO_PENALTY},
                      "mean", budget=True),
    "gspo": Method({"policy_loss": "gspo", "aggregation": "session-mean", **_NO_PENALTY}, "group"),
    "cispo": Method({"policy_loss": "cispo", "aggregation": "token-mean", "epsilon_low": 1.0, **_NO_PENALTY},
                    "group"),
    "gmpo": Method({"policy_loss": "gmpo", "aggregation": "session-mean",
                    "epsilon_low": 1 - math.exp(-0.4), "epsilon_high": math.exp(0.4) - 1, **_NO_PENALTY},
                   "group"),
    "reinforce_pp": Method({"policy_loss": "ppo", "aggregation": "session-mean", "kl_term": "k2",
                            "dual_clip": None, "reward_kl": 0.0}, "global"),
    "rloo": Method({"policy_loss": "reinforce", "aggregation": "constant", "aggregation_tokens": 1,
                    "beta": 0.0}, "rloo"),
}
"""Each named method's composition, the module docstring's table."""


def method(name: str) -> Method:
    """The method `name` names, refused by name where Dew has none."""
    if name not in METHODS:
        raise ValueError(f"no method {name!r}; Dew composes {sorted(METHODS)}")
    return METHODS[name]
