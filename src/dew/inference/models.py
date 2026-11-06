"""The native release of a trained model: any objective's model, its variables and what its loader cannot
read off them (row 4313).

An objective that declares a task of its own (`TextGeneration`, `Decide`, `TextToImage`...) loads as that
task. One that trains a model with no inference task in Dew (a JEPA encoder, an environment policy, a world
model) had no release: its checkpoints recorded no declaration, `SavedRun.at` refused them, and the weights
could only be read raw (`Checkpoints.variables`) by a reader that rebuilt the model itself. `SavedModel` is
the task such an objective's run loads as: the model its record names (`dew.registry.ModelConfig`), the
variables the objective selects of its trained tree (`Objective.model_tree`), the settings its record keeps
beside them (`ModelSettings`) and the numerics the run traced under. It is a `ReleasedTask`, so `dew export`
writes it as a release of its own, the weights as safetensors and the rest in the release record
(`dew.config.Release`), and `dew.pipeline` of that directory loads it back, to the model and weights the run
trained.

An objective takes this release by mixing in `ReleasesModel`; the model it records is
`Objective.recorded_model` and the tree it releases `Objective.model_tree`, which an objective that nests its
model beside other networks (a JEPA encoder beside its predictor) selects.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path, PurePath
from typing import TYPE_CHECKING, ClassVar, Self

import jax
import jax.numpy as jnp
from flax import linen as nn

from dew._records import integer, json_value, record as named_fields
from dew.batch import PARAMETER_ROOTS, Variables, under_roots
from dew.inference.tasks import SavedRun, SavedRunTask, saved_model, saved_settings
from dew.interop.safetensors_io import MAX_SHARD_SIZE, flatten, read_weights, save_sharded, unflatten
from dew.nn.precision import Numerics

if TYPE_CHECKING:
    from _typeshed import DataclassInstance
    from etils import epath
    from jax.sharding import Mesh
    from jax.typing import DTypeLike

    from dew.distributed.placement import Layout, MeshSpec
    from dew.objectives.base import Objective, SavedTask


@dataclass(frozen=True, kw_only=True)
class ModelSettings:
    """What a native release records of a model beside its record: the facts its loader cannot read off the
    model, a field each (`dew.objectives.rl.control.ControlSettings`). Most models state theirs in their own
    fields, and record none."""


@dataclass(frozen=True)
class SavedModel(SavedRunTask):
    """A trained model as its run released it: the module its record names, the variables its objective
    selects (`Objective.model_tree`) at the step and average loaded, the settings it records
    (`ModelSettings`) and the numerics the run traced under, with the objective class that trained it.

    `dew export` writes it (`save_pretrained`) and `dew.pipeline` loads it back from that directory or from
    the run (`from_pretrained`, `from_saved`), either as the model and weights the run trained; its
    `model.apply(variables, ...)` is what the objective's own loss applied, with the objective's run
    configuration (`settings`) beside it."""

    model: nn.Module
    variables: Variables
    settings: ModelSettings
    objective: type[Objective]
    numerics: Numerics | None = None

    @classmethod
    def from_saved(cls, run: SavedRun, *, ema: bool | None = None, mesh: MeshSpec | Mesh | None = None,
                   layout: Layout | None = None, dtype: DTypeLike | None = None,
                   param_dtype: DTypeLike | None = None) -> Self:
        """The model `run` records, over the variables its objective selects of the step pinned, placed on
        `mesh` under `layout`, its average where `ema` (`SavedRun.variables`), computing in `dtype` and
        stored in `param_dtype` where given."""
        objective = run.objective
        settings = objective.settings
        if settings is None or not issubclass(settings, ModelSettings):
            raise TypeError(f"{objective.__name__} records {settings!r} as its task settings, not a "
                            "ModelSettings: it declares a task of its own")
        model = saved_model(run.record, dtype).build()
        variables = run.variables(ema=ema, mesh=mesh, layout=layout, param_dtype=param_dtype)
        return cls(model, variables, saved_settings(run.record, settings), objective, run.precision)

    @classmethod
    def publishes(cls, run: SavedRun) -> bool:
        """Its own release states every model: it is its run's native release."""
        return True

    @classmethod
    def from_pretrained(cls, directory: str | Path, *, mesh: MeshSpec | Mesh | None = None,
                        layout: Layout | None = None, dtype: DTypeLike | None = None,
                        param_dtype: DTypeLike | None = None) -> Self:
        """The model the release in `directory` states (`save_pretrained`), its weights read from the
        safetensors beside it and placed on `mesh` under `layout`, computing in `dtype` and stored in
        `param_dtype` where given: its run's step as the release holds it (`ReleasedRun`)."""
        return cls.from_saved(ReleasedRun.of(directory), mesh=mesh, layout=layout, dtype=dtype,
                              param_dtype=param_dtype)

    def save_pretrained(self, directory: str | Path, *, max_shard_size: int | str = MAX_SHARD_SIZE) -> None:
        """Write this model as its run's native release (`write_run_release`): its variables as safetensors
        and, in the release record, the inference declaration a run of its objective records of it (its
        model, its settings and the numerics it traced under), which `from_pretrained` rebuilds it from."""
        from dew.registry import ModelConfig, import_path, record_fields, to_record

        record = {"objective": import_path(self.objective),
                  "model": to_record(ModelConfig.from_model(self.model), ModelConfig),
                  **record_fields(self.settings, type(self.settings)),
                  **({} if self.numerics is None else {"precision": to_record(self.numerics, Numerics)})}
        written(self.variables, record, 0, (), directory, max_shard_size=max_shard_size)


@dataclass(frozen=True)
class ReleasedRun(SavedRun):
    """A run's step as its native release holds it (`write_run_release`): the step's inference declaration
    and the files its task keeps by the step, read from the release's directory, and its variables from the
    safetensors beside them, the average its export chose. Every task loads it as it loads its run
    (`SavedRunTask.from_saved`), so a release reads as the run it was written of."""

    average: bool = False
    """Whether its variables are the run's average (`SavedRun.averaged`)."""

    @classmethod
    def of(cls, directory: str | Path, trust: Sequence[str] = ()) -> Self:
        """The run step the release in `directory` holds, the packages `trust` names that its record imports
        imported first (`dew.registry.import_trusted`), as a run's are (`SavedRun.at`); a directory that
        holds none is refused. A model's release written before it stated its run (its objective, model,
        settings and numerics) reads as the declaration its run would have recorded, at step 0."""
        from dew.config.release import Release
        from dew.inference.tasks import earlier_shapes
        from dew.registry import import_trusted

        release = Release.load(directory)
        task = None if release is None else release.task
        if task is None or ("run" not in task and "objective" not in task):
            raise ValueError(f"{directory} holds no native release: its release record states no run "
                             "(`dew export` of a run writes one)")
        record = (named_fields(task["run"], "run") if "run" in task else
                  {"objective": task["objective"], "model": task["model"],
                   **named_fields(task.get("settings", {}), "settings"),
                   **({} if task.get("numerics") is None else {"precision": task["numerics"]})})
        import_trusted(record, trust)
        held = cls.declared(str(directory), earlier_shapes(record), integer(task.get("step", 0), "step"))
        return replace(held, average=task.get("averaged") is True)

    def averaged(self, ema: bool | None) -> bool:
        """Whether its variables are the run's average, as its export wrote them; an `ema` asked is
        refused (`variables`)."""
        if ema is not None:
            raise ValueError("a release holds the weights its export chose (`dew export --ema`); load the "
                             "run itself to choose its average")
        return self.average

    def variables(self, *, ema: bool | None, mesh: MeshSpec | Mesh | None = None,
                  layout: Layout | None = None, param_dtype: DTypeLike | None = None,
                  parameter_roots: tuple[tuple[str, ...], ...] = PARAMETER_ROOTS) -> Variables:
        """The release's variables, placed on `mesh` under `layout`, the collections `parameter_roots` names
        stored in `param_dtype` where given and the others as written, as a run's load stores them
        (`Checkpoints.materialized`). It holds the one average its export chose: an `ema` asked is
        refused."""
        from dew.inference.loading import place
        from dew.registry import resolve_dtype

        if ema is not None:
            raise ValueError("a release holds the weights its export chose (`dew export --ema`); load the "
                             "run itself to choose its average")
        variables = unflatten(read_weights(self.directory))
        if param_dtype is not None:
            stored = jnp.dtype(resolve_dtype(param_dtype))
            variables = jax.tree_util.tree_map_with_path(
                lambda path, leaf: leaf.astype(stored) if jnp.issubdtype(leaf.dtype, jnp.floating)
                and under_roots(path, parameter_roots) else leaf, variables)
        return place(variables, mesh, layout)


def write_run_release(run: SavedRun, destination: str | Path, *, ema: bool | None,
                      mesh: MeshSpec | Mesh | None, layout: Layout | None) -> None:
    """Write `run`'s step as its native release in `destination`: the variables its tasks read
    (`SavedRun.variables`, its average where `ema`) as safetensors, the files its task keeps by the step
    (`step_records`) at their places, and its inference declaration in the release record, which
    `ReleasedRun` reads back. The export of a run whose model no family's source layout states
    (`dew.interop.saved_sources.source_publishes`), a model's own (`SavedModel.save_pretrained`) among
    them."""
    from dew.interop.saved_sources import step_records

    records = step_records(json_value(run.record, "artifact"), run.directory, run.step)
    written(run.variables(ema=ema, mesh=mesh, layout=layout), run.record, run.step,
            [(path, path.relative_to(run.directory)) for path in records], destination,
            averaged=run.averaged(ema))


def written(variables: Variables, record: Mapping[str, object], step: int,
            records: Sequence[tuple[epath.Path, PurePath]], destination: str | Path, *,
            averaged: bool = False, max_shard_size: int | str = MAX_SHARD_SIZE) -> None:
    """A native release in `destination`: `variables` as safetensors, `<collection>/<path>` the name of each
    leaf, read home a tensor at a time (`save_sharded`), each of `records` copied to its place, and the
    release record of the run `record` declares at `step` (0 for a model released outside a run), whose
    model states the dtype it computes in, and whether `variables` are its average."""
    from dew.config.release import Release

    root = Path(destination)
    root.mkdir(parents=True, exist_ok=True)
    save_sharded(flatten(variables), root, max_shard_size)
    for source, place in records:
        (root / place).parent.mkdir(parents=True, exist_ok=True)
        (root / place).write_bytes(source.read_bytes())
    Release(task={"run": json_value(record, "run"), "step": step, "averaged": averaged}).write(root)


class ReleasesModel:
    """An objective that releases its model as a `SavedModel`: it records its `recorded_model` and
    `task_settings` (`ModelSettings`, which an objective with facts of its own subclasses) and loads a saved
    run as the native release. Mixed in before `Objective`: `class JepaObjective(ReleasesModel,
    Objective[Ratio])`."""

    saved_task: ClassVar[type[SavedTask] | None] = SavedModel
    settings: ClassVar[type[DataclassInstance] | None] = ModelSettings

    def task_settings(self) -> ModelSettings:
        """The facts a loader cannot read off the model: none, unless the objective states some."""
        return ModelSettings()

__all__ = ["ModelSettings", "ReleasedRun", "ReleasesModel", "SavedModel", "write_run_release"]
