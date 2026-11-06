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

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Self

import jax
import jax.numpy as jnp
from flax import linen as nn

from dew._records import json_value, record as named_fields, text as named
from dew.batch import PARAMETER_ROOTS, Variables
from dew.inference.tasks import SavedRunTask, saved_model, saved_settings
from dew.interop.safetensors_io import MAX_SHARD_SIZE, flatten, read_weights, save_sharded, unflatten
from dew.nn.precision import Numerics
from dew.nn.protocols import ComputesIn

if TYPE_CHECKING:
    from _typeshed import DataclassInstance
    from jax.sharding import Mesh
    from jax.typing import DTypeLike

    from dew.distributed.placement import Layout, MeshSpec
    from dew.inference.tasks import SavedRun
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
    def from_pretrained(cls, directory: str | Path, *, mesh: MeshSpec | Mesh | None = None,
                        layout: Layout | None = None, dtype: DTypeLike | None = None,
                        param_dtype: DTypeLike | None = None) -> Self:
        """The model the release in `directory` states (`save_pretrained`), its weights read from the
        safetensors beside it and placed on `mesh` under `layout`, computing in `dtype` and stored in
        `param_dtype` where given."""
        from dew.config.release import Release
        from dew.inference.loading import place
        from dew.registry import ModelConfig, dtype_name, from_record, objectives, resolve_dtype

        release = Release.load(directory)
        task = None if release is None else release.task
        if task is None or "objective" not in task:
            raise ValueError(f"{directory} holds no native model release: its release record names no "
                             "objective (`SavedModel.save_pretrained` writes one)")
        objective = objectives[named(task["objective"], "objective")]
        settings = objective.settings
        if settings is None or not issubclass(settings, ModelSettings):
            raise TypeError(f"{objective.__name__} records {settings!r} as its task settings, not a "
                            "ModelSettings: it declares a task of its own")
        model = ModelConfig.from_dict(named_fields(task["model"], "model")).with_dtype(
            dtype_name(resolve_dtype(dtype))).build()
        stored = None if param_dtype is None else jnp.dtype(resolve_dtype(param_dtype))
        variables = unflatten(read_weights(directory))
        if stored is not None:
            # Its parameters stored as asked, its other state as written, as a run's load stores them
            # (`PARAMETER_ROOTS`, `Checkpoints.materialized`).
            variables = {name: jax.tree.map(lambda leaf: leaf.astype(stored) if jnp.issubdtype(
                leaf.dtype, jnp.floating) else leaf, part) if (name,) in PARAMETER_ROOTS else part
                         for name, part in variables.items()}
        numerics = None if task.get("numerics") is None else from_record(
            Numerics, named_fields(task["numerics"], "numerics"), dtypes=False)
        return cls(model, place(variables, mesh, layout),
                   from_record(settings, named_fields(task.get("settings", {}), "settings"), dtypes=False),
                   objective, numerics)

    def save_pretrained(self, directory: str | Path, *, max_shard_size: int | str = MAX_SHARD_SIZE) -> None:
        """Write this model as a release: its variables as safetensors, `<collection>/<path>` the name of
        each leaf, read home a tensor at a time (`save_sharded`), and the release record beside them
        (`dew.config.Release`): the objective that trained it, its model's record and settings, and the
        numerics and the dtype it states it computes in, which `from_pretrained` rebuilds it from."""
        from dew.config.release import Release
        from dew.registry import ModelConfig, dtype_name, import_path, record_fields, to_record

        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        # The dtype the model states it computes in, which its record holds too; one that states none
        # computes in what its inputs and parameters make, which no record of its storage says.
        compute = self.model.dtype if isinstance(self.model, ComputesIn) else None
        save_sharded(flatten(self.variables), root, max_shard_size)
        Release(task={
            "objective": import_path(self.objective),
            "model": json_value(to_record(ModelConfig.from_model(self.model), ModelConfig), "model"),
            "settings": json_value(record_fields(self.settings, type(self.settings)), "settings"),
            "numerics": None if self.numerics is None else to_record(self.numerics, Numerics),
            "dtypes": {} if compute is None else {"compute": dtype_name(compute)}}).write(root)


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

__all__ = ["ModelSettings", "ReleasesModel", "SavedModel"]
