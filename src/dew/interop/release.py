"""The release metadata of an export of a causal decoder's run (`dew.config.Release`'s task): the run's own
task declaration, the numerics its tasks trace under and the dtype its model computes in, which
`Pretrained.save` writes and `Pretrained.load` reads back in place of a family's defaults (ledger 4314).
generation_config.json, which every other reader reads, is derived from the same declaration. The record, its
file and its other fields (a decision release's layout) are the release contract's, `dew.config.release`.
"""

from __future__ import annotations

import dataclasses
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
    from dew.config.release import Release
    from dew.inference.tasks import TaskSettings


def release_record(settings: TaskSettings, numerics: Numerics | None, model: nn.Module,
                   variables: Variables) -> Release:
    """The release of a run's export: its task settings' record, its numerics and the dtype its
    model computes in (`computed_in`); the parameters' own dtypes are the export's."""
    from dew.config.release import Release
    from dew.inference.tasks import TaskSettings

    return Release(task={"settings": to_record(settings, TaskSettings),
                         "numerics": None if numerics is None else to_record(numerics, Numerics),
                         "dtypes": {"compute": computed_in(numerics, model, variables)},
                         "rounds": rounding(model)})


ROUNDING = ("scale_after_cast", "force_fp32_for_softmax")
"""A model's fields that order its rounding (a norm's scale against its cast to the compute dtype, a softmax's
width), which no family's config states and which change what it computes below float32 alone
(`dew.interop.hf_decoders._RUNTIME_FIELDS`)."""


def rounding(model: nn.Module) -> dict[str, bool]:
    """The fields of `ROUNDING` `model` declares, as it holds them: what an export's release states so that
    its load rounds as its run did (`with_rounding`)."""
    declared = {field.name for field in dataclasses.fields(model)}
    return {name: getattr(model, name) for name in ROUNDING if name in declared}


def with_rounding[M: nn.Module](model: M, rounds: Mapping[str, bool]) -> M:
    """`model` rounding as a release states it (`rounding`): a field the release states and the model
    declares none of is refused, as the release then describes another model."""
    declared = {field.name for field in dataclasses.fields(model)}
    unread = sorted(set(rounds) - declared)
    if unread:
        raise ValueError(f"the release states {unread} of how its model rounds, which a "
                         f"{type(model).__name__} does not declare: it describes another model")
    return dataclasses.replace(model, **rounds) if rounds else model


def computed_in(numerics: Numerics | None, model: nn.Module, variables: Variables) -> DtypeName:
    """The dtype a released model computes in, by name: the one the model states (`ComputesIn`), else its
    numerics' compute where its task traces under them (a run's precision, which a model stating no dtype
    of its own computes under), else its parameters' (`variables`), which a module stating none computes
    in. Every export's release states it, so its load computes as its run did (`load_settings`)."""
    stated = model.dtype if isinstance(model, ComputesIn) else None
    if stated is not None:
        return dtype_name(stated)
    if numerics is not None:
        return dtype_name(numerics.dtypes.compute)
    return dtype_name(jnp.result_type(*(leaf.dtype for leaf in jax.tree.leaves(unboxed(variables))
                                        if jnp.issubdtype(leaf.dtype, jnp.floating))))


def released(directory: Path) -> tuple[TaskSettings | None, Numerics | None, DtypeName | None,
                                       Mapping[str, bool]]:
    """The run's own task settings, numerics, compute dtype and rounding an export of it wrote
    (`release_record`), or None for each (no rounding) where the directory holds none: a source with no run
    behind it builds its tasks from its own files, in the load's dtypes."""
    from dew.config.release import Release
    from dew.inference.tasks import TaskSettings

    release = Release.load(directory)
    if release is None or release.task is None:
        return None, None, None, {}
    task = release.task
    # A decision release states its task's numerics alone (`dew.decision.release.stated_release`): its own
    # files declare its task.
    settings, numerics = task.get("settings"), task.get("numerics")
    dtypes = _records.record(task["dtypes"], "dtypes")
    return (None if settings is None else from_record(TaskSettings, _records.record(settings, "settings"),
                                                      dtypes=False),
            None if numerics is None else from_record(Numerics, _records.record(numerics, "numerics"),
                                                      dtypes=False),
            dtype_name(_records.text(dtypes["compute"], "compute")),
            {name: _records.boolean(value, name)
             for name, value in _records.record(task.get("rounds", {}), "rounds").items()})


class LoadSettings(NamedTuple):
    """What a load of a source takes from it and from what the load states (`load_settings`)."""

    settings: TaskSettings | None
    numerics: Numerics | None
    dtype: DtypeName
    param_dtype: DtypeName | Literal["auto"]
    rounds: Mapping[str, bool]
    """How its model rounds, as its release states it (`rounding`); none for a source that states none."""


def load_settings(directory: Path, numerics: Numerics | None, dtype: DTypeLike | None,
                  param_dtype: DTypeLike | Literal["auto"] | None) -> LoadSettings:
    """A load of the source in `directory`: the run's declaration an export of it carries (`released`),
    the numerics its tasks trace under (the load's, else the run's), and the compute and storage dtypes
    it loads in, by name: its numerics' dtypes, else each one it states, and each it leaves unstated
    (None) its source's: the run's compute dtype over its weights as stored, or BF16 compute over FP32
    master weights."""
    settings, run_numerics, run_compute, rounds = released(directory)
    if numerics is not None:
        return LoadSettings(settings, numerics, dtype_name(numerics.dtypes.compute),
                            dtype_name(numerics.dtypes.params), rounds)
    compute = dtype_name(dtype) if dtype is not None else run_compute or "bfloat16"
    params: DtypeName | Literal["auto"] = (
        AUTO if param_dtype == AUTO or (param_dtype is None and run_compute is not None)
        else "float32" if param_dtype is None else dtype_name(param_dtype))
    return LoadSettings(settings, run_numerics, compute, params, rounds)
