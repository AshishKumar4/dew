"""The named RL methods as compositions of Dew's parts: an advantage estimator, a ratio and its
clipping (the policy loss), a reference penalty and a reduction, with DAPO's dynamic sampling and
overlong punishment where it reads them. Each method is its paper's objective, held to it by
tests/test_rl_methods.py; a run names one (`method`) and may override a part only where the method
leaves it free (its group size, its KL weight where it has one, a clip range its paper tunes).

| method | estimator | policy loss | clip (low, high) | reference penalty | reduction |
|---|---|---|---|---|---|
| grpo (arXiv:2402.03300) | group | ppo | 0.2, 0.2 | k3, beta | session-mean |
| dapo (arXiv:2503.14476) | group | ppo | 0.2, 0.28 | none | token-mean; varied groups; overlong |
| dr_grpo (arXiv:2503.20783) | mean | ppo | 0.2, 0.2 | none | constant (the budget) |
| gspo (arXiv:2507.18071) | group | gspo | the run's | none | session-mean |
| cispo (arXiv:2506.13585) | group | cispo | none below, the run's above | none | token-mean |
| gmpo (arXiv:2507.20673) | group | gmpo | e^-0.4, e^0.4 | none | session-mean |
| reinforce_pp (arXiv:2501.03262) | global | ppo | 0.2, 0.2 | squared, beta | session-mean |
| rloo (arXiv:2402.14740) | rloo | reinforce | none | none | constant (one token) |
"""

import dataclasses
import math
from typing import Any

from .grpo import GRPOObjective


@dataclasses.dataclass(frozen=True)
class Method:
    """One named method: its objective's settings (`GRPOObjective`) and its rollout's (the estimator,
    `RolloutScheduler.varied` and `.overlong`). `kl` is whether it penalizes a reference, whose weight a
    run sets; `budget` whether its reduction or its overlong punishment reads the response budget."""

    objective: dict[str, Any]
    estimator: str
    kl: bool = False
    varied: bool = False
    overlong: bool = False
    budget: bool = False

    def build(self, model, seq_len: int, *, beta: float = 0.0, max_new_tokens: int | None = None,
              overlong_cache: int | None = None, **options) -> tuple[GRPOObjective, dict[str, Any]]:
        """The method's objective over `model` and its rollout's settings, the reference's weight
        `beta` where it penalizes one, its budget `max_new_tokens` where it reads one, and DAPO's
        overlong cache. A setting the method fixes is refused in `options`."""
        fixed = set(self.objective) & set(options)
        if fixed:
            raise ValueError(f"the method fixes {sorted(fixed)}")
        if beta and not self.kl:
            raise ValueError("this method penalizes no reference, so it takes no beta")
        if self.budget and max_new_tokens is None:
            raise ValueError("this method reads the response budget: give max_new_tokens")
        settings = dict(self.objective)
        if settings.get("aggregation") == "constant" and "aggregation_tokens" not in settings:
            settings["aggregation_tokens"] = max_new_tokens
        rollout: dict[str, Any] = {"estimator": self.estimator, "varied": self.varied}
        if self.overlong:
            rollout["overlong"] = (max_new_tokens, overlong_cache or max(1, max_new_tokens // 4))
        return GRPOObjective(model, seq_len, beta=beta, **settings, **options), rollout


METHODS: dict[str, Method] = {
    "grpo": Method({"policy_loss": "ppo", "aggregation": "session-mean", "dual_clip": None}, "group",
                   kl=True),
    "dapo": Method({"policy_loss": "ppo", "aggregation": "token-mean", "epsilon_low": 0.2,
                    "epsilon_high": 0.28, "dual_clip": None}, "group",
                   varied=True, overlong=True, budget=True),
    "dr_grpo": Method({"policy_loss": "ppo", "aggregation": "constant", "dual_clip": None}, "mean",
                      budget=True),
    "gspo": Method({"policy_loss": "gspo", "aggregation": "session-mean"}, "group"),
    "cispo": Method({"policy_loss": "cispo", "aggregation": "token-mean", "epsilon_low": 1.0}, "group"),
    "gmpo": Method({"policy_loss": "gmpo", "aggregation": "session-mean",
                    "epsilon_low": 1 - math.exp(-0.4), "epsilon_high": math.exp(0.4) - 1}, "group"),
    "reinforce_pp": Method({"policy_loss": "ppo", "aggregation": "session-mean", "kl_term": "squared",
                            "dual_clip": None}, "global", kl=True),
    "rloo": Method({"policy_loss": "reinforce", "aggregation": "constant", "aggregation_tokens": 1}, "rloo"),
}
"""Each named method's composition, the module docstring's table."""


def method(name: str) -> Method:
    """The method `name` names, refused by name where Dew has none."""
    if name not in METHODS:
        raise ValueError(f"no method {name!r}; Dew composes {sorted(METHODS)}")
    return METHODS[name]
