"""A run's release metadata beside an export's config (`RELEASE_FILE`): the run's own task declaration, the
numerics its tasks trace under and the dtype its model computes in, which `Pretrained.save` writes and
`Pretrained.load` reads back in place of a family's defaults (ledger 4314). Its record is the release
contract's task (`dew.config.Release`); generation_config.json, which every other reader reads, is derived
from the same declaration.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Literal, NamedTuple

import jax
import jax.numpy as jnp
from flax import linen as nn
from jax.typing import DTypeLike

from dew import _records
from dew.batch import Variables, unboxed
from dew.interop.weights import AUTO
from dew.nn.precision import Numerics
from dew.nn.protocols import ComputesIn
from dew.registry import DtypeName, dtype_name, from_record, to_record

if TYPE_CHECKING:
    from dew.inference.tasks import TaskSettings

RELEASE_FILE = "dew_release.json"
"""The file an export of a run writes its release metadata in, beside its config."""


def release_record(settings: TaskSettings, numerics: Numerics | None, model: nn.Module,
                   variables: Variables) -> Mapping[str, object]:
    """The release metadata of a run's export: its task settings' record, its numerics and the dtype its
    model computes in, the one it states (`ComputesIn`), else its parameters' (`variables`), which a
    module stating none computes in; the parameters' own dtypes are the export's."""
    from dew.inference.tasks import TaskSettings

    stated = model.dtype if isinstance(model, ComputesIn) else None
    held = jnp.result_type(*(leaf.dtype for leaf in jax.tree.leaves(unboxed(variables))
                             if jnp.issubdtype(leaf.dtype, jnp.floating)))
    compute = held if stated is None else stated
    return {"task": {"settings": to_record(settings, TaskSettings),
                     "numerics": None if numerics is None else to_record(numerics, Numerics),
                     "dtypes": {"compute": dtype_name(compute)}}}


def released(directory: Path) -> tuple[TaskSettings | None, Numerics | None, DtypeName | None]:
    """The run's own task settings, numerics and compute dtype an export of it wrote (`release_record`),
    or None for each where the directory holds none: a source with no run behind it builds its tasks from
    its own files, in the load's dtypes."""
    from dew.inference.tasks import TaskSettings

    path = directory / RELEASE_FILE
    if not path.is_file():
        return None, None, None
    task = _records.record(_records.record(json.loads(path.read_text()), RELEASE_FILE)["task"], "task")
    numerics, dtypes = task.get("numerics"), _records.record(task["dtypes"], "dtypes")
    return (from_record(TaskSettings, _records.record(task["settings"], "settings"), dtypes=False),
            None if numerics is None else from_record(Numerics, _records.record(numerics, "numerics"),
                                                      dtypes=False),
            dtype_name(_records.text(dtypes["compute"], "compute")))


class LoadSettings(NamedTuple):
    """What a load of a source takes from it and from what the load states (`load_settings`)."""

    settings: TaskSettings | None
    numerics: Numerics | None
    dtype: DtypeName
    param_dtype: DtypeName | Literal["auto"]


def load_settings(directory: Path, numerics: Numerics | None, dtype: DTypeLike | None,
                  param_dtype: DTypeLike | Literal["auto"] | None) -> LoadSettings:
    """A load of the source in `directory`: the run's declaration an export of it carries (`released`),
    the numerics its tasks trace under (the load's, else the run's), and the compute and storage dtypes
    it loads in, by name: its numerics' dtypes, else each one it states, and each it leaves unstated
    (None) its source's: the run's compute dtype over its weights as stored, or BF16 compute over FP32
    master weights."""
    settings, run_numerics, run_compute = released(directory)
    if numerics is not None:
        return LoadSettings(settings, numerics, dtype_name(numerics.dtypes.compute),
                            dtype_name(numerics.dtypes.params))
    compute = dtype_name(dtype) if dtype is not None else run_compute or "bfloat16"
    params: DtypeName | Literal["auto"] = (
        AUTO if param_dtype == AUTO or (param_dtype is None and run_compute is not None)
        else "float32" if param_dtype is None else dtype_name(param_dtype))
    return LoadSettings(settings, run_numerics, compute, params)
