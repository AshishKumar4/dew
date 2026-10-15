"""LM training terms unsupported by policy and preference losses."""

from collections.abc import Mapping


def check_lm_terms(options: Mapping[str, object]) -> None:
    """Refuse auxiliary LM terms these losses never apply."""
    defaults = {"balance_rate": None, "aux_loss_alpha": None, "mtp_weight": None,
                "z_loss": 0.0, "router_z_loss": 0.0, "indexer": None}
    asked = [name for name, default in defaults.items() if options.get(name, default) != default]
    if asked:
        raise ValueError(f"policy and preference losses do not apply LM training terms: {', '.join(asked)}")
