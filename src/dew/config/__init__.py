"""Typed configuration for a training run.

One dataclass tree describes a run: what to build, what to feed it, how to
optimize it, and how the trainer runs. Recipes parse it with tyro and hand the
pieces to the trainer, so `to_dict()` is a full record of a run and
`from_dict()` puts it back together.

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
import types
import typing
from typing import Annotated, Any, Literal, Mapping, Optional, Self, Union

import tyro

from etils import epath

import dew.data  # noqa: F401  registers the datasets a config names
import dew.nn.backbones  # noqa: F401  registers the models a config names
from dew import registry
from dew.registry import datasets, models, with_precision
from dew.telemetry.instrumentation import default_compilation_cache_dir
from dew.training.distributed import Layout, MeshSpec
from dew.training.trainer import Profile

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

RUN_FILE = "run.json"

REGISTRIES = (registry.models, registry.presets, registry.samplers, registry.datasets,
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

    def total_steps(self, data) -> int:
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

    def eval_interval(self, data) -> Optional[int]:
        """Steps between validation passes over `data`, or None for never."""
        return self._interval(self.eval_every, data, "eval-every")

    def checkpoint_interval(self, data) -> Optional[int]:
        """Steps between checkpoints over `data`, or None for never."""
        return self._interval(self.checkpoint_every, data, "checkpoint-every")

    @staticmethod
    def _interval(value, data, flag: str) -> Optional[int]:
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


def _to_json(value):
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


def _rebuild(annotation, value):
    held = _registry_for(annotation)
    if held is not None:
        member = held[value["name"]]
        return member(**_fields(member, value["fields"]))
    if dataclasses.is_dataclass(annotation):
        return annotation(**_fields(annotation, value))
    if typing.get_origin(annotation) in (typing.Union, types.UnionType):
        inner = [m for m in typing.get_args(annotation) if m is not type(None)]
        if value is None or len(inner) != 1:
            return value
        return _rebuild(inner[0], value)
    return value


@dataclasses.dataclass(frozen=True)
class RunConfig:
    """A whole run. Recipes add their objective's knobs by subclassing this."""

    model: ModelConfig = dataclasses.field(default_factory=ModelConfig)
    data: datasets.union = dataclasses.field(
        default_factory=lambda: datasets["oxford_flowers102"]())
    optim: OptimConfig = dataclasses.field(default_factory=OptimConfig)
    trainer: TrainerConfig = dataclasses.field(default_factory=TrainerConfig)

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe record of the run; a registered member is written as its
        name and fields."""
        return _to_json(self)

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
