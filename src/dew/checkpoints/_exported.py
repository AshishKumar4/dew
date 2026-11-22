"""The model a step exports for readers without Dew (`Checkpoints.export_model`)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import jax
import numpy as np

from dew._coordination import spans_part_of_pool

if TYPE_CHECKING:
    from dew.state import TrainState

EXPORTED = "exported"
"""The Orbax item a step exported for readers without Dew holds (`Checkpoints.export_model`): the model's
variables and EMA whole, in Orbax's standard layout."""


def exported(state: TrainState) -> Mapping[str, object]:
    """`state`'s model as `EXPORTED` holds it: its variables, every collection among them, and its EMA where
    it keeps one, each its own arrays. Refused where an array lies on part of the pool (a pipeline's row on
    its stage's processes alone), which Orbax's standard layout writes none of (`export_refusal`, which a
    run checks as it starts)."""
    from dew.checkpoints.submesh import apart

    model = {"variables": state.variables, **({} if state.ema is None else {"ema": state.ema})}
    if any(apart(leaf) for leaf in jax.tree.leaves(model)):
        raise ValueError(EXPORT_APART)
    return model


EXPORT_APART = ("export_model writes the model in Orbax's standard layout, which holds no array on part of "
                "the pool; this pipeline's stages each hold their rows on their own processes")


def export_refusal(mesh: jax.sharding.Mesh, stage_axis: str) -> str | None:
    """Why a run over `mesh` exports no model (`exported`): its stages, along `stage_axis`, each on processes
    of the pool and not every one, where a pipeline's rows lie apart; None where every stage spans the
    pool (one process, or a pipeline within each process)."""
    if stage_axis not in mesh.axis_names:
        return None
    devices = np.moveaxis(mesh.devices, mesh.axis_names.index(stage_axis), 0)
    return EXPORT_APART if any(spans_part_of_pool(stage.ravel()) for stage in devices) else None
