"""Typed configuration for a training run.

One dataclass tree describes a run: what to build, what to feed it, how to
optimize it, and how the trainer runs. Recipes parse it with tyro, build the
objective and the data it names, and hand both to `RunConfig.train`, so
`to_dict()` is a full record of a run and `from_dict()` puts it back together.

Model kwargs are an opaque JSON dict; the registry knows which architecture
takes which fields. A dataset is the registered spec itself, which tyro
turns into a subcommand (`data:token-windows --data.path ...`).

The resolved config is the run's spec. A recipe writes it to `run.json` next
to the checkpoints with `save`, and `load` reads it back into the same class,
so inference rebuilds a run from what training was built from. A field the
class does not have, or one the file lacks, raises.
"""

import dataclasses
import hashlib
import json
import os
import re
import sys
import types
import typing
from collections.abc import Sequence
from typing import TYPE_CHECKING, Annotated, Any, Literal, Mapping, Self

import jax
import tyro
from etils import epath

import dew.data  # registers the datasets a config names
import dew.io
import dew.nn.backbones  # registers the models a config names
from dew import registry
from dew.artifacts import agree_process_phase
from dew.checkpoints import RUN_FILE, Checkpoints
from dew.data import Dataset, DatasetSpec, Ramp, ramped
from dew.nn.attention import AttentionImpl
from dew.objectives.base import Effects, Loss, Metric, Objective
from dew.records import JSON
from dew.registry import REGISTRIES, _declared_type, datasets, models, with_precision
from dew.telemetry.instrumentation import default_compilation_cache_dir
from dew.telemetry.records import RunRecord, json_value, packages_installed
from dew.training.distributed import Layout, MeshSpec
from dew.training.optim import build_optimizer
from dew.training.quantization import Quantization, quantize
from dew.training.state import TrainState
from dew.training.tracker import LocalTracker, Trackers, WandbTracker
from dew.training.trainer import ProfileWindow, Trainer

JsonDict = Annotated[
    Mapping[str, object],
    tyro.constructors.PrimitiveConstructorSpec(
        nargs=1,
        metavar="JSON",
        instance_from_str=lambda args: json.loads(args[0]),
        is_instance=lambda value: isinstance(value, dict),
        str_from_instance=lambda value: [json.dumps(value)],
    ),
]
"""The fields a model is built from, written as a single JSON string on the
command line. The registry knows which architecture takes which field and
narrows each one where it builds it, so the values are read there."""

if TYPE_CHECKING:
    # tyro reads the runtime annotation, a Union of the registered specs, and a
    # type checker cannot read a variable in a type expression. Both get what
    # they need: the base class statically, the union at runtime.
    type DataSpec = DatasetSpec
else:
    DataSpec = datasets.union


@dataclasses.dataclass(frozen=True)
class ModelConfig:
    """Architecture name and the fields `models.build` receives."""

    architecture: str = "simple_dit"
    config: JsonDict = dataclasses.field(default_factory=dict)
    dtype: registry.DtypeName = "bfloat16"
    """Compute dtype; parameter storage is independent."""
    param_dtype: registry.DtypeName | None = None
    """Parameter storage, where the model declares the field. Unset stores
    float32, which is what a model's own default is and what the optimizer
    and the checkpoint have always held."""
    matmul_precision: Literal["default", "high", "highest"] | None = None
    """What every matmul of the model asks XLA for, where the model declares
    a `precision` field: `default` is the backend's fastest algorithm,
    `high` and `highest` trade throughput for mantissa bits (on Ampere and
    later, tf32 and fp32 against bf16x3). Unset leaves the model's own."""
    attention_impl: AttentionImpl = "auto"
    """Attention kernel; 'auto' is cudnn on a GPU for the shapes cudnn
    supports and xla for the rest, xla on any other backend."""

    def fields(self) -> Mapping[str, object]:
        """The model's fields with the run's precision settings in them."""
        return with_precision(self.architecture, self.config,
                              dtype=self.dtype, attention_impl=self.attention_impl,
                              param_dtype=self.param_dtype,
                              matmul_precision=self.matmul_precision)

    def precision_settings(self) -> frozenset[str]:
        """The names `fields()` writes that `config` did not carry: the run's
        precision settings, as this architecture takes them. A resolved
        record leaves them out, since this value writes them again every
        time it builds."""
        return frozenset(self.fields()) - frozenset(self.config)

    @classmethod
    def from_dict(cls, values: Mapping[str, object]) -> Self:
        """Inverse of the record `RunConfig.to_dict` writes for this field."""
        return _rebuild(cls, values)

    def build(self):
        return models.build(self.architecture, **self.fields())


@dataclasses.dataclass(frozen=True)
class OptimConfig:
    """Optimizer, learning-rate schedule and gradient clipping."""

    optimizer: Literal["adam", "adamw", "lamb", "muon", "muonclip"] = "adamw"
    optimizer_opts: JsonDict = dataclasses.field(default_factory=dict)
    learning_rate: float = 2.7e-4
    learning_rate_schedule: Literal["cosine"] | None = None
    learning_rate_peak: float = 3e-4
    learning_rate_end: float = 2e-4
    learning_rate_warmup_steps: int = 10000
    learning_rate_decay_steps: int | None = None
    """Optimizer updates the cosine decays over; unset uses the training step target."""
    weight_decay: float | None = None
    clip_grads: float = 0.0


@dataclasses.dataclass(frozen=True)
class Wandb:
    """Where a run reports to. Setting it turns tracking on; the entity and
    the offline switch mean nothing without a project."""

    project: str
    entity: str | None = None
    offline: bool = False


@dataclasses.dataclass(frozen=True)
class TrainerConfig:
    """Run length, checkpointing, sharding and run tracking."""

    name: str | None = None
    checkpoint_dir: str = "./checkpoints"
    keep: int = 2
    """Latest checkpoints kept, besides the best one."""
    batch_size: int = 32
    """Global batch, over every process."""
    seed: int = 0
    """Seed of the run key: parameter init and every per-step draw."""
    steps: int | None = None
    epochs: int | None = None
    """Run length as passes over the data; `steps` names it directly instead."""
    log_every: int = 100
    eval_every: int | Literal["epoch"] | None = "epoch"
    """Steps between validation passes: a number of steps, "epoch" for one
    pass over the data, None to never validate. "epoch" over a stream that
    reports no record count raises a ValueError, since it has no pass."""
    checkpoint_every: int | Literal["epoch"] | None = "epoch"
    """Steps between checkpoints, the same three answers. None is what a
    stream whose iterator cannot report a read position trains with; the
    trainer refuses any other answer for one."""
    accumulation: int = 1
    """Micro-batches per optimizer update."""
    batch_ramp: Ramp | None = None
    """Grow `batch_size` over the run's first records instead of starting
    there: the global batch the run starts at, what a stage adds and the
    records the whole ramp spans. Unset trains at `batch_size` throughout.
    One optimizer update a step either way, and the compiled step is traced
    once per stage."""
    dynamic_scale: bool = False
    mesh: MeshSpec = MeshSpec()
    layout: Layout = Layout()
    profile: ProfileWindow | None = None
    """One profiler window: the steps to trace, the warmup before it and the
    directory it is written to. Unset traces nothing."""
    compilation_cache_dir: str | None = dataclasses.field(
        default_factory=default_compilation_cache_dir)
    """Persisted XLA cache, so a restart skips recompiling the step. None
    compiles from scratch every run."""
    wandb: Wandb | None = None
    """Optional W&B sink in addition to the local tracking journal."""
    multi_host: bool | None = None
    """Join the JAX process pool. None asks and continues alone only when no cluster is configured; True requires the pool; False never asks."""
    xla_flags: str | None = None
    """Extra XLA_FLAGS for this run, appended to the environment by
    `prepare_process` before JAX opens a backend. Library users set XLA_FLAGS
    themselves; see docs/performance.md for what was measured."""
    quantization: Quantization | None = None
    """Quantized-training spec, wrapped around the module the objective
    trains before the run initialises it; unset trains in the compute dtype.
    `dew.training.quantization` says what the wrap does and what it keeps."""

    def __post_init__(self):
        if self.steps is not None and self.epochs is not None:
            raise ValueError("steps and epochs both name the run length; set one")

    def total_steps(self, dataset: Dataset) -> int:
        """The run's length in steps, from `steps` or from `epochs` over `data`."""
        if self.steps is not None:
            return self.steps
        if self.epochs is None:
            raise ValueError("the run length is --trainer.steps or --trainer.epochs")
        if dataset.steps_per_epoch is None:
            raise ValueError(
                "epochs need a dataset with a record count; this one streams without "
                "one, so give the run length as --trainer.steps")
        return self.epochs * dataset.steps_per_epoch

    def eval_interval(self, dataset: Dataset) -> int | None:
        """Steps between validation passes over `data`, or None for never."""
        return self._interval(self.eval_every, dataset, "eval-every")

    def checkpoint_interval(self, dataset: Dataset) -> int | None:
        """Steps between checkpoints over `data`, or None for never."""
        return self._interval(self.checkpoint_every, dataset, "checkpoint-every")

    @staticmethod
    def _interval(value, dataset: Dataset, field_name: str) -> int | None:
        if value is None or isinstance(value, int):
            return value
        if dataset.steps_per_epoch is None:
            raise ValueError(
                f"--trainer.{field_name} epoch needs a dataset with a record count; this "
                f"one streams without one, so give the interval in steps or None")
        return dataset.steps_per_epoch


def _registry_for(annotation):
    """The registry whose members the annotation names, or None."""
    members = typing.get_args(annotation) or (annotation,)
    for held in REGISTRIES:
        if all(any(member is m for m in held.values()) for member in members):
            return held
    return None


def _to_json(value, annotation) -> JSON:
    """`value` as JSON: a dict, a list, or a scalar json.dump can write.
    `annotation` is the declared field type, so the write side names the same
    registry and member types the read side rebuilds from."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        held = _registry_for(annotation)
        if held is not None and not any(type(value) is member
                                      for member in held.values()):
            raise ValueError(
                f"{type(value).__qualname__} is not a registered {held.kind}; a "
                f"run config can only record members that load back, register "
                f"it with @{held.kind}s(...)")
        if held is None:
            held = _registry_for(type(value))
        fields = {f.name: _to_json(getattr(value, f.name), _declared_type(type(value), f.name))
                  for f in dataclasses.fields(value)}
        if held is None:
            return fields
        name = held.name_of(type(value))
        return ({"kind": name, **fields} if held.record == "kind"
                else {"name": name, "fields": fields})
    if isinstance(value, (list, tuple)):
        entries = registry.entry_types(annotation, len(value))
        return [_to_json(entry_value, entry)
                for entry_value, entry in zip(value, entries, strict=True)]
    if isinstance(value, Mapping):
        entries = registry.entry_types(annotation, len(value))
        return {str(key): _to_json(entry_value, entry)
                for (key, entry_value), entry in zip(value.items(), entries, strict=True)}
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(
        f"{type(value).__name__} is not something a run record can carry; a "
        f"config field holds JSON scalars, sequences, mappings, and the "
        f"registered values this writes as their name and fields")


def _fields(cls, values):
    declared = [f.name for f in dataclasses.fields(cls) if f.init]
    unknown = sorted(set(values) - set(declared))
    missing = [name for name in declared if name not in values]
    if unknown or missing:
        raise ValueError(
            f"{cls.__name__} does not match the record: unknown fields {unknown}, "
            f"missing fields {missing}")
    return {name: _rebuild(_declared_type(cls, name), values[name]) for name in declared}


def _rebuild(annotation, value) -> Any:
    """The value `annotation` asks for, built out of a record. It returns
    whatever type the field declares, so the annotation is Any."""
    annotation = registry.resolve_alias(annotation)
    held = _registry_for(annotation)
    if held is not None:
        if held.record == "kind":
            member = held[value["kind"]]
            fields = {name: entry for name, entry in value.items() if name != "kind"}
        else:
            member, fields = held[value["name"]], value["fields"]
        return member(**_fields(member, fields))
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
        rebuilt = [_rebuild(entry, record)
                   for entry, record in zip(entries, value, strict=True)]
        return tuple(rebuilt) if registry.wants_tuple(annotation) else rebuilt
    return value


@dataclasses.dataclass(frozen=True)
class RunConfig:
    """A whole run. Recipes add their objective's knobs by subclassing this."""

    model: ModelConfig = dataclasses.field(default_factory=ModelConfig)
    data: DataSpec = dataclasses.field(
        default_factory=lambda: datasets["oxford_flowers102"]())
    optim: OptimConfig = dataclasses.field(default_factory=OptimConfig)
    trainer: TrainerConfig = dataclasses.field(default_factory=TrainerConfig)
    objective: str | None = None

    def to_dict(self) -> dict[str, JSON]:
        """JSON-safe record of the run; a registered member is written as its
        name and fields."""
        return {field.name: _to_json(getattr(self, field.name),
                                     _declared_type(type(self), field.name))
                for field in dataclasses.fields(self)}

    @classmethod
    def from_dict(cls, values: Mapping[str, object]) -> Self:
        """Inverse of `to_dict`, for subclasses too; an unknown or a missing
        field raises."""
        return _rebuild(cls, values)

    def save(self, directory: str) -> str:
        """Write this config as `run.json` in `directory` and return the path.

        The path goes through `epath`, the same filesystem layer orbax writes
        the checkpoints with, so a `gs://` run directory takes the record too.
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

    def train(self, objective: Objective[Loss, Effects], dataset: Dataset, *, name: str,
              metrics: Sequence[Metric] = (),
              summary: Mapping[str, object] | None = None) -> TrainState:
        """Train `objective` on `data` as this run says; every recipe calls
        this once it has built both.

        The run lives under `name` in `trainer.checkpoint_dir`, and process
        zero writes the record there before anything trains. A `trainer.wandb`
        opens a tracker under the same name, with the record, `summary` (the
        recipe's own view of the run) and the step count as its config, and
        the checkpoint the run ends on is published to the registry under the
        name with slashes and spaces replaced, since an artifact name allows
        neither. Local tracking journals live in a separate tracking directory.

        `dataset` is the one a recipe loaded at `trainer.batch_size`, and a
        dataset at any other batch is refused here: the record, the ramp and
        the run's reported throughput all name the configured number, while
        every step reads the dataset's, so the two disagreeing is a run that
        trains at a batch it does not report.

        A `trainer.quantization` wraps the module `objective` trains before
        anything initialises it, so the quantized forward is what the run
        learns through.
        """
        if dataset.batch != self.trainer.batch_size:
            raise ValueError(
                f"--trainer.batch-size is {self.trainer.batch_size} and this dataset "
                f"reads {dataset.batch} records a step; load it with "
                f"load(batch={self.trainer.batch_size})")
        if self.trainer.quantization is not None:
            quantize(objective, self.trainer.quantization)
        objective_type = type(objective)
        kind = (registry.objectives.name_of(objective_type) if objective_type in registry.objectives.values()
                else f"{objective_type.__module__}.{objective_type.__qualname__}")
        self = dataclasses.replace(self, objective=kind)
        trainer = self.trainer
        # Before the run length, since a ramp reads fewer records a step early
        # and a pass over the data is that many steps longer.
        dataset = dataset if trainer.batch_ramp is None else ramped(dataset, trainer.batch_ramp)
        steps = trainer.total_steps(dataset)
        tracker = None
        try:
            wandb_tracker = None
            if trainer.wandb is not None:
                wandb_tracker = WandbTracker(
                    trainer.wandb.project, name, entity=trainer.wandb.entity,
                    offline=trainer.wandb.offline)
            checkpoints = Checkpoints(os.path.join(trainer.checkpoint_dir, name), keep=trainer.keep)
            local = LocalTracker(os.path.join(checkpoints.directory, "tracking"))
            if "://" in checkpoints.directory:
                local = LocalTracker(os.path.join(os.path.expanduser("~/.cache/dew/tracking"),
                                                 re.sub(r"[^\w.-]", "-", name) + "-"
                                                 + hashlib.sha256(checkpoints.directory.encode()).hexdigest()[:12]))
            tracker = Trackers(local, *(() if wandb_tracker is None else (wandb_tracker,)))
            error = None
            try:
                if jax.process_index() == 0:
                    print("Experiment_Name:", name)
                    print(f"Local tracking: {local.directory}")
                    self.save(checkpoints.directory)
                    tracker.artifact(RunRecord(name, json_value(self.to_dict()),
                        json_value(summary or {}), steps, packages_installed()), 0)
            except BaseException as failure:
                error = failure
            agree_process_phase(error, phase="run metadata")
            state = Trainer.from_config(
                trainer, objective, build_optimizer(self.optim, steps),
                key=jax.random.key(trainer.seed),
                checkpoints=checkpoints,
                tracker=tracker,
            ).fit(
                dataset, steps=steps,
                log_every=trainer.log_every,
                eval_every=trainer.eval_interval(dataset),
                checkpoint_every=trainer.checkpoint_interval(dataset),
                metrics=metrics, preview=trainer.wandb is not None,
            )
            error = None
            try:
                if wandb_tracker is not None:
                    # fit wrote the step it ended on, so that checkpoint is the run's.
                    dew.io.publish(checkpoints.path(int(state.step)), re.sub(r"[^\w.-]", "-", name),
                                   tracker=wandb_tracker)
            except BaseException as failure:
                error = failure
            agree_process_phase(error, phase="checkpoint publishing")
            return state

        finally:
            primary = sys.exception()
            error = None
            try:
                if tracker is not None:
                    tracker.__exit__(type(primary), primary, None)
            except BaseException as failure:
                error = failure
            try:
                agree_process_phase(error, phase="tracker close")
            except BaseException as failure:
                if primary is None:
                    raise
                primary.add_note(f"Tracker close failed: {failure!r}")
