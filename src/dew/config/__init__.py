"""Typed configuration for a training run.

One dataclass tree describes a run: what to build, what to feed it, how to
optimize it, and how the trainer runs. Recipes parse it with tyro, build the
objective and the data it names, and hand both to `RunConfig.train`, so
`to_dict()` is a full record of a run and `from_dict()` puts it back together.

Model kwargs stay an opaque JSON dict. The registry already knows which
architecture takes which fields, and mirroring them here would be a second
place to keep in sync. A dataset is the registered spec itself, which tyro
turns into a subcommand (`data:token-windows --data.path ...`).

The resolved config is the run's spec. A recipe writes it to `run.json` next
to the checkpoints with `save`, and `load` reads it back into the same class,
so inference rebuilds a run from what training was built from. A field the
class does not have, or one the file lacks, raises.
"""

import dataclasses
import json
import os
import re
import types
import typing
from collections.abc import Sequence
from typing import (TYPE_CHECKING, Annotated, Any, Literal, Mapping, Optional, Self,
                    TypeAlias, Union)

import jax
import tyro

from etils import epath

import dew.data  # noqa: F401  registers the datasets a config names
import dew.io
from dew.checkpoints import RUN_FILE, Checkpoints
from dew.data import Dataset, DatasetSpec
import dew.nn.backbones  # noqa: F401  registers the models a config names
from dew import registry
from dew.objectives.base import Metric, Objective
from dew.registry import Registry, datasets, models, with_precision
from dew.telemetry.instrumentation import default_compilation_cache_dir
from dew.training.distributed import Layout, MeshSpec
from dew.training.optim import build_optimizer
from dew.training.state import TrainState
from dew.training.tracker import WandbTracker
from dew.training.trainer import Profile, Trainer

JsonDict = Annotated[
    dict[str, Any],
    tyro.constructors.PrimitiveConstructorSpec(
        nargs=1,
        metavar="JSON",
        instance_from_str=lambda args: json.loads(args[0]),
        is_instance=lambda value: isinstance(value, dict),
        str_from_instance=lambda value: [json.dumps(value)],
    ),
]
"""A dict, written as a single JSON string on the command line."""

if TYPE_CHECKING:
    # tyro reads the runtime annotation, a Union of the registered specs, and a
    # type checker cannot read a variable in a type expression. Both get what
    # they need: the base class statically, the union at runtime.
    DataSpec: TypeAlias = DatasetSpec
else:
    DataSpec = datasets.union

REGISTRIES: tuple[Registry[Any], ...] = (registry.models, registry.presets, registry.samplers, registry.datasets,
              registry.encoders, registry.metrics, registry.objectives)


@dataclasses.dataclass(frozen=True)
class ModelConfig:
    """Architecture name and the fields `models.build` receives."""

    architecture: str = "simple_dit"
    config: JsonDict = dataclasses.field(default_factory=dict)
    dtype: Literal["float32", "bfloat16"] = "bfloat16"
    """Compute dtype; params stay float32."""
    attention_impl: Literal["auto", "reference", "xla", "cudnn", "tpu"] = "auto"
    """Attention kernel; 'auto' is cudnn on a GPU for the shapes cudnn
    supports and xla for the rest, xla on any other backend."""

    def fields(self) -> dict[str, Any]:
        """The model's fields with the run's precision settings in them."""
        return with_precision(self.architecture, self.config,
                              dtype=self.dtype, attention_impl=self.attention_impl)

    def build(self):
        return models.build(self.architecture, **self.fields())


@dataclasses.dataclass(frozen=True)
class OptimConfig:
    """Optimizer, learning-rate schedule and gradient clipping."""

    optimizer: Literal["adam", "adamw", "lamb", "muon"] = "adamw"
    optimizer_opts: JsonDict = dataclasses.field(default_factory=dict)
    learning_rate: float = 2.7e-4
    learning_rate_schedule: Optional[Literal["cosine"]] = None
    learning_rate_peak: float = 3e-4
    learning_rate_end: float = 2e-4
    learning_rate_warmup_steps: int = 10000
    learning_rate_decay_steps: Optional[int] = None
    """Steps the cosine decays over; unset decays over the run."""
    weight_decay: Optional[float] = None
    clip_grads: float = 0.0


@dataclasses.dataclass(frozen=True)
class Wandb:
    """Where a run reports to. Its presence is what turns tracking on: the
    entity and the offline switch mean nothing without a project, and an
    unset project used to stand in for running without a tracker."""

    project: str
    entity: Optional[str] = None
    offline: bool = False


@dataclasses.dataclass(frozen=True)
class TrainerConfig:
    """Run length, checkpointing, sharding and run tracking."""

    name: Optional[str] = None
    checkpoint_dir: str = "./checkpoints"
    keep: int = 2
    """Latest checkpoints kept, besides the best one."""
    batch_size: int = 32
    """Global batch, over every process."""
    seed: int = 0
    """Seed of the run key: parameter init and every per-step draw."""
    steps: Optional[int] = None
    epochs: Optional[int] = None
    """Run length as passes over the data; `steps` names it directly instead."""
    log_every: int = 100
    eval_every: Union[int, Literal["epoch"], None] = "epoch"
    """Steps between validation passes: a number of steps, "epoch" for one
    pass over the data, None to never validate. "epoch" over a stream that
    reports no record count is refused by name, since it has no pass."""
    checkpoint_every: Union[int, Literal["epoch"], None] = "epoch"
    """Steps between checkpoints, the same three answers. None is what a
    stream whose iterator cannot report a read position trains with; the
    trainer refuses any other answer for one."""
    accumulation: int = 1
    """Micro-batches per optimizer update."""
    dynamic_scale: bool = False
    mesh: MeshSpec = MeshSpec()
    layout: Layout = Layout()
    profile: Optional[Profile] = None
    """One profiler window: the steps to trace, the warmup before it and the
    directory it is written to. Unset traces nothing."""
    compilation_cache_dir: Optional[str] = dataclasses.field(
        default_factory=default_compilation_cache_dir)
    """Persisted XLA cache, so a restart skips recompiling the step. None
    compiles from scratch every run."""
    wandb: Optional[Wandb] = None
    """Unset runs without a tracker."""
    multi_host: Optional[bool] = None
    """Join the JAX process pool. None asks and continues alone only when no cluster is configured; True requires the pool; False never asks."""
    xla_flags: Optional[str] = None
    """Extra XLA_FLAGS for this run, appended to the environment by
    `prepare_process` before JAX opens a backend. Library users set XLA_FLAGS
    themselves; see docs/performance.md for what was measured."""

    def __post_init__(self):
        if self.steps is not None and self.epochs is not None:
            raise ValueError("steps and epochs both name the run length; set one")

    def total_steps(self, data: Dataset) -> int:
        """The run's length in steps, from `steps` or from `epochs` over `data`."""
        if self.steps is not None:
            return self.steps
        if self.epochs is None:
            raise ValueError("the run length is --trainer.steps or --trainer.epochs")
        if data.steps_per_epoch is None:
            raise ValueError(
                "epochs need a dataset with a record count; this one streams without "
                "one, so give the run length as --trainer.steps")
        return self.epochs * data.steps_per_epoch

    def eval_interval(self, data: Dataset) -> Optional[int]:
        """Steps between validation passes over `data`, or None for never."""
        return self._interval(self.eval_every, data, "eval-every")

    def checkpoint_interval(self, data: Dataset) -> Optional[int]:
        """Steps between checkpoints over `data`, or None for never."""
        return self._interval(self.checkpoint_every, data, "checkpoint-every")

    @staticmethod
    def _interval(value, data: Dataset, flag: str) -> Optional[int]:
        if value is None or isinstance(value, int):
            return value
        if data.steps_per_epoch is None:
            raise ValueError(
                f"--trainer.{flag} epoch needs a dataset with a record count; this "
                f"one streams without one, so give the interval in steps or None")
        return data.steps_per_epoch


def _registry_for(annotation):
    """The registry whose members the annotation names, or None."""
    members = typing.get_args(annotation) or (annotation,)
    for held in REGISTRIES:
        if all(any(member is m for m in held.values()) for member in members):
            return held
    return None


def _to_json(value) -> Any:
    """`value` as JSON: a dict, a list, or a scalar json.dump can write."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        held = _registry_for(type(value))
        fields = {f.name: _to_json(getattr(value, f.name))
                  for f in dataclasses.fields(value)}
        return {"name": held.name_of(type(value)), "fields": fields} if held else fields
    if isinstance(value, (list, tuple)):
        return [_to_json(item) for item in value]
    if isinstance(value, Mapping):
        return {key: _to_json(item) for key, item in value.items()}
    return value


def _fields(cls, values):
    hints = typing.get_type_hints(cls)
    declared = [f.name for f in dataclasses.fields(cls) if f.init]
    unknown = sorted(set(values) - set(declared))
    missing = [name for name in declared if name not in values]
    if unknown or missing:
        raise ValueError(
            f"{cls.__name__} does not match the record: unknown fields {unknown}, "
            f"missing fields {missing}")
    return {name: _rebuild(hints[name], values[name]) for name in declared}


def _rebuild(annotation, value) -> Any:
    """The value `annotation` asks for, built out of a record. Any is the
    truth here: what comes back is whatever type the field declares."""
    held = _registry_for(annotation)
    if held is not None:
        member = held[value["name"]]
        return member(**_fields(member, value["fields"]))
    if isinstance(annotation, type) and dataclasses.is_dataclass(annotation):
        return annotation(**_fields(annotation, value))
    if typing.get_origin(annotation) in (typing.Union, types.UnionType):
        inner = [m for m in typing.get_args(annotation) if m is not type(None)]
        if value is None or len(inner) != 1:
            return value
        return _rebuild(inner[0], value)
    if isinstance(value, list):
        # JSON writes every sequence as a list; the field says which are tuples.
        entries = registry.entry_types(annotation, len(value))
        items = [_rebuild(entry, item) for entry, item in zip(entries, value)]
        return tuple(items) if registry.wants_tuple(annotation) else items
    return value


@dataclasses.dataclass(frozen=True)
class RunConfig:
    """A whole run. Recipes add their objective's knobs by subclassing this."""

    model: ModelConfig = dataclasses.field(default_factory=ModelConfig)
    data: DataSpec = dataclasses.field(
        default_factory=lambda: datasets["oxford_flowers102"]())
    optim: OptimConfig = dataclasses.field(default_factory=OptimConfig)
    trainer: TrainerConfig = dataclasses.field(default_factory=TrainerConfig)

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe record of the run; a registered member is written as its
        name and fields."""
        return {field.name: _to_json(getattr(self, field.name))
                for field in dataclasses.fields(self)}

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> Self:
        """Inverse of `to_dict`, for subclasses too; an unknown or a missing
        field raises."""
        return _rebuild(cls, values)

    def save(self, directory: str) -> str:
        """Write this config as `run.json` in `directory` and return the path.

        The path goes through `epath`, the same filesystem layer orbax writes
        the checkpoints with, so a `gs://` run directory takes the record too
        instead of failing a pod run after the training succeeded.
        """
        path = epath.Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        target = path / RUN_FILE
        target.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True))
        return str(target)

    @classmethod
    def load(cls, directory: str) -> Self:
        """The config a run in `directory` was built from, as this class."""
        return cls.from_dict(json.loads((epath.Path(directory) / RUN_FILE).read_text()))

    def train(self, objective: Objective, data: Dataset, *, name: str,
              metrics: Sequence[Metric] = (),
              summary: Mapping[str, object] | None = None) -> TrainState:
        """Train `objective` on `data` as this run says, which is what every
        recipe does once it has built both.

        The run lives under `name` in `trainer.checkpoint_dir`, and process
        zero writes the record there before anything trains. A `trainer.wandb`
        opens a tracker under the same name, with the record, `summary` (the
        recipe's own view of the run) and the step count as its config, and
        the checkpoint the run ends on is published to the registry under the
        name with slashes and spaces replaced, since an artifact name allows
        neither. Without one the run trains and checkpoints alone.
        """
        trainer = self.trainer
        steps = trainer.total_steps(data)
        print("Experiment_Name:", name)
        tracker = None
        if trainer.wandb is not None:
            tracker = WandbTracker(
                trainer.wandb.project, name, entity=trainer.wandb.entity,
                offline=trainer.wandb.offline,
                config={"run_config": self.to_dict(), **(summary or {}), "steps": steps})
        checkpoints = Checkpoints(os.path.join(trainer.checkpoint_dir, name), keep=trainer.keep)
        if jax.process_index() == 0:
            self.save(checkpoints.directory)
        state = Trainer(
            objective, build_optimizer(self.optim, steps),
            key=jax.random.key(trainer.seed),
            mesh=trainer.mesh,
            layout=trainer.layout,
            accumulation=trainer.accumulation,
            dynamic_scale=trainer.dynamic_scale,
            checkpoints=checkpoints,
            tracker=tracker,
            profile=trainer.profile,
        ).fit(
            data, steps=steps,
            log_every=trainer.log_every,
            eval_every=trainer.eval_interval(data),
            checkpoint_every=trainer.checkpoint_interval(data),
            metrics=metrics,
        )
        if tracker is not None:
            # fit wrote the step it ended on, so that checkpoint is the run's.
            dew.io.publish(checkpoints.path(int(state.step)), re.sub(r"[^\w.-]", "-", name),
                           tracker=tracker)
        return state
