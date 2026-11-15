"""The release metadata of an export of a causal decoder's run (`dew.config.Release`'s task): the run's own
task declaration, the numerics its tasks trace under and the dtype its model computes in, which
`Pretrained.save` writes and `Pretrained.load` reads back in place of a family's defaults (ledger 4314).
generation_config.json, which every other reader reads, is derived from the same declaration. The record, its
file and its other fields (a decision release's layout) are the release contract's,
`dew.interop.release_record`.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, NamedTuple

import jax
import jax.numpy as jnp
from flax import linen as nn
from jax.typing import DTypeLike

from dew import _records
from dew._records import JSON
from dew.batch import Variables, unboxed
from dew.inference.tasks import TaskSettings
from dew.interop.weights import AUTO
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.precision import Numerics
from dew.nn.protocols import Bounded, ComputesIn
from dew.registry import DtypeName, dtype_name, from_record, record_fields

if TYPE_CHECKING:
    from dew.interop.release_record import Release


@dataclass(frozen=True)
class ReleaseDtypes:
    """The dtype a released model computes in, by name (`computed_in`)."""

    compute: DtypeName


@dataclass(frozen=True)
class ReleaseTask:
    """A release's task declaration (`dew.config.Release.task`), one record written (`record`) and read
    (`read`) through the registry: the run's task settings, the numerics its tasks trace under, the dtype its
    model computes in, how it rounds (`rounding`) and the tokens its cache holds (`capacity`), each None
    where the release states none (a decision release states its numerics and dtype alone,
    `dew.decision.release.decision_envelope`); a release with no task states none of them."""

    settings: TaskSettings | None = None
    numerics: Numerics | None = None
    dtypes: ReleaseDtypes | None = None
    rounds: Mapping[str, bool] | None = None
    max_seq_len: int | None = None

    def __post_init__(self) -> None:
        if self.rounds is not None:
            object.__setattr__(self, "rounds", {name: _records.boolean(value, name)
                                                for name, value in self.rounds.items()})
        if self.max_seq_len is not None:
            _records.integer(self.max_seq_len, "max_seq_len")

    def record(self) -> dict[str, JSON]:
        """The task's record (`to_record`), the fields it states none of left out."""
        return {name: value for name, value in record_fields(self, ReleaseTask).items() if value is not None}

    @classmethod
    def read(cls, record: Mapping[str, object]) -> ReleaseTask:
        """The task a release's record states (`record`'s inverse)."""
        return from_record(cls, _records.record(record, "task"), dtypes=False)

    @property
    def compute(self) -> DtypeName | None:
        """The dtype its model computes in, None where the release states none."""
        return None if self.dtypes is None else self.dtypes.compute


def release_record(settings: TaskSettings, numerics: Numerics | None, model: nn.Module,
                   variables: Variables) -> Release:
    """The release of a run's export: its task settings, its numerics, the dtype its model computes in
    (`computed_in`), how it rounds (`rounding`) and the cache it holds (`capacity`); the parameters' own
    dtypes are the export's."""
    from dew.interop.release_record import Release

    dtypes = ReleaseDtypes(computed_in(numerics, model, variables))
    return Release(task=ReleaseTask(settings, numerics, dtypes, rounding(model), capacity(model)).record())


def capacity(model: nn.Module) -> int | None:
    """The tokens `model`'s cache holds (`Bounded.max_seq_len`), where it states a count: a load-time
    capacity no family's config states beside its context (`max_position_embeddings`), which a request no
    shape bucket holds pads its cache to, so a release states it for its load (`released`) to build the
    model holding it, as `Pretrained.load(max_seq_len=...)` does."""
    return model.max_seq_len if isinstance(model, Bounded) else None


ROUNDING = tuple(field.name for field in dataclasses.fields(CausalTransformer)
                 if field.metadata.get(_records.ROUNDS))
"""A model's fields that order its rounding (a norm's scale against its cast to the compute dtype, a softmax's
width), which no family's config states and which change what it computes below float32 alone: the
decoder's that say so (`dew._records.ROUNDS`), by name in any model that declares them."""


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


def released(directory: Path) -> ReleaseTask:
    """The run's own task settings, numerics, compute dtype, rounding and cache capacity an export of it
    wrote (`release_record`), or none of them where the directory holds none: a source with no run behind
    it builds its tasks from its own files, in the load's dtypes."""
    from dew.interop.release_record import Release

    release = Release.load(directory)
    return ReleaseTask() if release is None or release.task is None else ReleaseTask.read(release.task)


class LoadSettings(NamedTuple):
    """What a load of a source takes from it and from what the load states (`load_settings`)."""

    settings: TaskSettings | None
    numerics: Numerics | None
    dtype: DtypeName
    param_dtype: DtypeName | Literal["auto"]
    rounds: Mapping[str, bool]
    """How its model rounds, as its release states it (`rounding`); none for a source that states none."""
    max_seq_len: int | None
    """The tokens its model's cache holds: the load's, else its release's (`capacity`); None for its
    source's own."""


def load_settings(directory: Path, numerics: Numerics | None, dtype: DTypeLike | None,
                  param_dtype: DTypeLike | Literal["auto"] | None,
                  max_seq_len: int | None = None) -> LoadSettings:
    """A load of the source in `directory`: the run's declaration an export of it carries (`released`),
    the numerics its tasks trace under (the load's, else the run's), the compute and storage dtypes
    it loads in, by name: its numerics' dtypes, else each one it states, and each it leaves unstated
    (None) its source's: the run's compute dtype over its weights as stored, or BF16 compute over FP32
    master weights; and its cache's capacity, the load's `max_seq_len`, else the run's."""
    run = released(directory)
    held = run.max_seq_len if max_seq_len is None else max_seq_len
    rounds = run.rounds or {}
    if numerics is not None:
        return LoadSettings(run.settings, numerics, dtype_name(numerics.dtypes.compute),
                            dtype_name(numerics.dtypes.params), rounds, held)
    compute = dtype_name(dtype) if dtype is not None else run.compute or "bfloat16"
    params: DtypeName | Literal["auto"] = (
        AUTO if param_dtype == AUTO or (param_dtype is None and run.compute is not None)
        else "float32" if param_dtype is None else dtype_name(param_dtype))
    return LoadSettings(run.settings, run.numerics, compute, params, rounds, held)
