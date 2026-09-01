from typing import Any

__all__ = ["DiffusionObjective"]


def __getattr__(name: str) -> Any:
    """Resolve DiffusionObjective on first use.

    objective.py reaches the samplers for its validation step, and the samplers
    import this package's schedules and transforms - importing it eagerly here
    would close that loop and break `import dew.sampling.ddim` on a cold
    interpreter. Deferring it keeps `from dew.objectives.diffusion import
    DiffusionObjective` working without the cycle.
    """
    if name == "DiffusionObjective":
        from .objective import DiffusionObjective
        return DiffusionObjective
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
