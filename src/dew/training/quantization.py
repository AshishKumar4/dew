"""The training step that quantizes the modules an objective trains (`dew.quantization`)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dew.objectives.base import Objective
    from dew.quantization import Quantization


def quantize_trunk[LossT, EffectsT](objective: Objective[LossT, EffectsT], spec: Quantization) -> None:
    """Quantize the trunk matmuls of the modules `objective` trains, in place.

    This is `RunConfig.train`'s step, not a user's. `Quantization.apply`
    wraps a module before an objective is built, which is what a recipe or
    a script that builds its own model does. A run that names
    `--trainer.quantization` has handed `RunConfig.train` the objective
    already, so the wrap lands on the modules the objective runs instead
    (`Objective.substitute`), before anything has initialised or traced
    them; each wrapped module is a copy of the same class, so what the
    objective read off it at construction still holds. A frozen teacher or
    reference keeps its own numerics.

    An objective that trains no module has nothing to wrap and is refused by name.
    """
    entries = objective.program_key()
    if not any(entry.trained for entry in entries):
        raise ValueError(
            f"--trainer.quantization quantizes the modules an objective trains, and "
            f"{type(objective).__name__} trains none; train an objective that holds a model, "
            f"or leave the quantization unset")
    objective.substitute([spec.apply(entry.module) if entry.trained else entry.module for entry in entries])


__all__ = ["quantize_trunk"]
