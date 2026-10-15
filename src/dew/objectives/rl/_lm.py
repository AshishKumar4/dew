"""LM training terms unsupported by policy and preference losses."""

import inspect
from collections.abc import Mapping

from ..lm import LMObjective

TERMS = ("balance_rate", "aux_loss_alpha", "mtp_weight", "z_loss", "router_z_loss", "indexer")
"""The `LMObjective` training terms a policy or preference loss never applies."""


def check_lm_terms(options: Mapping[str, object]) -> None:
    """Refuse any of `TERMS` asked other than at `LMObjective`'s own default."""
    defaults = inspect.signature(LMObjective.__init__).parameters
    asked = [name for name in TERMS if name in options and options[name] != defaults[name].default]
    if asked:
        raise ValueError(f"policy and preference losses do not apply LM training terms: {', '.join(asked)}")
