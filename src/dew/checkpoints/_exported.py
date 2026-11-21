"""The model a step exports for readers without Dew (`Checkpoints.export_model`)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dew.state import TrainState

EXPORTED = "exported"
"""The Orbax item a step exported for readers without Dew holds (`Checkpoints.export_model`): the model's
variables and EMA whole, in Orbax's standard layout."""


def exported(state: TrainState, held_apart: bool) -> Mapping[str, object]:
    """`state`'s model as `EXPORTED` holds it: its variables, every collection among them, and its EMA where
    it keeps one, each its own arrays. Refused for a pipeline's rows each on its own stage's devices
    (`held_apart`), which Orbax's standard layout writes no part-of-pool array of."""
    if held_apart:
        raise ValueError("export_model writes the model in Orbax's standard layout, which holds no array on "
                         "part of the pool; this pipeline's stages each hold their own rows")
    return {"variables": state.variables, **({} if state.ema is None else {"ema": state.ema})}
