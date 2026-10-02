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
file lacks takes its declared default, and a field the class does not have
raises.
"""

import dataclasses
import datetime
import functools
import hashlib
import json
import operator
import os
import re
import sys
import types
import typing
from collections.abc import Callable, Mapping, Mapping as MappingABC, MutableMapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal, Self

import jax
import optax
import tyro
from etils import epath

import dew.data  # registers the datasets a config names
import dew.io
import dew.nn.backbones  # registers the models a config names
from dew import registry
from dew.artifacts import agree_process_phase, agreed
from dew.checkpoints import RUN_FILE, Checkpoints, Keep
from dew.config.sweep import Search, Space, _read, _write, override, random_search
from dew.data import Dataset, DatasetSpec, Ramp
from dew.data.dataset import json_list_argument, ramped
from dew.lora import LoRA, _attach
from dew.nn.attention import AttentionImpl
from dew.objectives.base import Effects, Loss, Metric, Objective
from dew.records import JSON, duration, recorded_duration
from dew.registry import REGISTRIES, _declared_type, datasets, models, schedules, with_precision
from dew.telemetry.instrumentation import default_compilation_cache_dir, dew_cache_dir
from dew.telemetry.records import RunRecord, TrialFinished, json_value, packages_installed
from dew.training.display import TrainingDisplay
from dew.training.distributed import Layout, MeshSpec
from dew.training.optim import (
    BF16_STATE_OPTIMIZERS,
    OPTIMIZER_MAP,
    ParamGroup,
    ScheduleBase,
    learning_rate_schedule,
    param_labels,
    power_profiles,
)
from dew.training.quantization import Quantization, _quantize
from dew.training.selection import Best
from dew.training.state import TrainState
from dew.training.tracker import LocalTracker, Tracker, Trackers, WandbTracker
from dew.training.trainer import ProfileWindow, Rollout, Trainer

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
    type ScheduleSpec = ScheduleBase
else:
    DataSpec = datasets.union
    ScheduleSpec = schedules.union


@dataclasses.dataclass(frozen=True)
class ModelConfig:
    """Holds the architecture name and the fields `models.build` receives."""

    architecture: str = "simple_dit"
    config: JsonDict = dataclasses.field(default_factory=dict)
    dtype: registry.DtypeName | None = "bfloat16"
    """Compute dtype; parameter storage is independent."""
    param_dtype: registry.DtypeName | None = None
    """Parameter storage, where the model declares the field. Unset stores
    float32, which is the model's own default."""
    matmul_precision: Literal["default", "high", "highest"] | None = None
    """What every matmul of the model asks XLA for, where the model declares
    a `precision` field: `default` is the backend's fastest algorithm,
    `high` and `highest` trade throughput for mantissa bits (on Ampere and
    later, tf32 and fp32 against bf16x3). Unset leaves the model's own, the
    default. Under bf16 compute a decoder's vocabulary head at the default
    rounds its logits and their gradient to bf16, as torch autocast does;
    `high` and `highest` keep that head fp32, the setting for comparing
    parallel layouts in bf16 (`dew.nn.precision.head_product`)."""
    attention_impl: AttentionImpl = "auto"
    """Attention kernel; 'auto' is cudnn on a GPU for the shapes cudnn
    supports and xla for the rest, xla on any other backend."""

    def fields(self) -> Mapping[str, object]:
        """Return the model's fields with the run's precision settings in them."""
        return with_precision(self.architecture, self.config,
                              dtype=self.dtype, attention_impl=self.attention_impl,
                              param_dtype=self.param_dtype,
                              matmul_precision=self.matmul_precision)

    def precision_settings(self) -> frozenset[str]:
        """Return the names `fields()` writes that `config` did not carry: the run's
        precision settings, as this architecture takes them. A resolved
        record leaves them out, since this value writes them again every
        time it builds."""
        return frozenset(self.fields()) - frozenset(self.config)

    @classmethod
    def from_dict(cls, values: Mapping[str, object]) -> Self:
        """Read back the record `RunConfig.to_dict` writes for this field."""
        return _built(cls, values)

    @classmethod
    def from_model(cls, model) -> Self:
        """The registered module's constructor fields, with its actual compute settings."""
        architecture = models.name_of(type(model))
        fields = {}
        compute, storage, attention = None, None, 'auto'
        precision: Literal['default', 'high', 'highest'] | None = None
        for field in dataclasses.fields(model):
            if not field.init or field.name in ('parent', 'name'):
                continue
            value = getattr(model, field.name)
            if field.name == 'dtype':
                compute = registry.dtype_name(value)
            elif field.name == 'param_dtype':
                storage = registry.dtype_name(value)
            elif field.name == 'precision':
                if value is not None:
                    setting = str(value).lower()
                    if setting == 'default':
                        precision = 'default'
                    elif setting == 'high':
                        precision = 'high'
                    elif setting == 'highest':
                        precision = 'highest'
                    else:
                        raise ValueError(f'model precision {value!r} has no recorded counterpart')
            elif field.name == 'attention_impl':
                attention = value
            elif callable(value) and value is field.default:
                continue
            else:
                fields[field.name] = _to_json(value, _declared_type(type(model), field.name))
        return cls(architecture, fields, dtype=compute, param_dtype=storage,
                   matmul_precision=precision, attention_impl=attention)

    def build(self):
        return models.build(self.architecture, self.fields())


@dataclasses.dataclass(frozen=True)
class OptimConfig:
    """Holds the optimizer, learning-rate schedule and gradient clipping."""

    optimizer: Literal["adam", "adamw", "lamb", "muon", "muonclip"] = "adamw"
    optimizer_opts: JsonDict = dataclasses.field(default_factory=dict)
    learning_rate: float = 2.7e-4
    """The constant rate, when no schedule is named."""
    schedule: ScheduleSpec | None = None
    """The learning-rate schedule, one typed record per kind
    (`dew.training.optim`): cosine, power (lm-engine's power law with an
    optional linear tail) or linear; each holds only its own fields."""
    weight_decay: float | None = None
    param_groups: Annotated[tuple[ParamGroup, ...], json_list_argument(ParamGroup)] = ()
    """Per-group learning-rate multipliers and weight decay, first match wins;
    empty moves every parameter alike. `ParamGroup.mup` is lm-engine's muP
    split."""
    clip_grads: float = 0.0
    state_dtype: Literal["float32", "bfloat16"] = "float32"
    """Adam's moments in memory. bfloat16 stores both stochastically rounded
    (`dew.training.optim.bf16_moments`), for adam and adamw
    only: half the optimizer state and less of the update's memory traffic."""
    forced_weight_normalization: bool = False
    """Renormalize every magnitude-preserving weight (`dew.nn.mp.MPConv`)
    after each update, EDM2's forced weight normalization, which its
    `edm2_unet` trains with (`dew.nn.mp.forced_weight_normalization`)."""
    ema_profiles: tuple[float, ...] = ()
    """Relative standard deviations of the power-function EMAs a run keeps
    for post-hoc EMA (`dew.training.optim.power_profiles`), such as Karras
    et al.'s (0.05, 0.10); every checkpoint save snapshots them, and
    `Checkpoints.posthoc_ema` builds an average of any other
    relative standard deviation from the snapshots. Empty keeps none."""

    def build(self, steps: int) -> optax.GradientTransformation:
        """Build the solver this config describes, with its schedule, parameter
        groups and clipping.

        `steps` is the run's length, which a schedule decays over unless the
        config names its own end. `param_groups` runs one solver per group under
        `optax.multi_transform`, each on the schedule times its multiplier and
        with its own weight decay; the global-norm clip still reads every
        gradient together, before the groups split them."""
        learning_rate = learning_rate_schedule(self, steps)
        opts = dict(self.optimizer_opts)
        if self.weight_decay is not None:
            opts['weight_decay'] = self.weight_decay
            if self.optimizer in ('muon', 'muonclip'):
                # Muon's weight_decay does not cover the AdamW group's norm scales.
                opts.setdefault('adam_weight_decay', self.weight_decay)
        make = OPTIMIZER_MAP[self.optimizer]
        if self.state_dtype == 'bfloat16':
            if self.optimizer not in BF16_STATE_OPTIMIZERS:
                raise ValueError(
                    f"state_dtype='bfloat16' stores Adam's moments in bf16, which "
                    f"{sorted(BF16_STATE_OPTIMIZERS)} have; {self.optimizer!r} does not")
            make = BF16_STATE_OPTIMIZERS[self.optimizer]
        if self.param_groups:
            names = [group.name for group in self.param_groups]
            if len(set(names)) != len(names):
                raise ValueError(f"param group names repeat: {names}")
            solvers = {}
            for group in self.param_groups:
                group_opts = dict(opts)
                if group.weight_decay is not None:
                    group_opts['weight_decay'] = group.weight_decay
                    if self.optimizer in ('muon', 'muonclip'):
                        group_opts['adam_weight_decay'] = group.weight_decay
                solvers[group.name] = make(
                    _scaled(learning_rate, group.learning_rate_multiplier), **group_opts)
            solver = optax.multi_transform(solvers, param_labels(self.param_groups))
        else:
            solver = make(learning_rate, **opts)

        if self.clip_grads > 0:
            solver = optax.chain(optax.clip_by_global_norm(self.clip_grads), solver)
        if self.forced_weight_normalization:
            from dew.nn.mp import forced_weight_normalization

            solver = optax.chain(solver, forced_weight_normalization())
        if self.ema_profiles:
            solver = power_profiles(solver, self.ema_profiles)
        return solver


def _scaled(learning_rate: float | optax.Schedule, multiplier: float) -> float | optax.Schedule:
    if multiplier == 1.0:
        return learning_rate
    if callable(learning_rate):
        schedule = learning_rate
        return lambda count: multiplier * schedule(count)
    return multiplier * learning_rate


@dataclasses.dataclass(frozen=True)
class Wandb:
    """Says where a run reports to. Setting it turns tracking on; the entity and
    the offline switch mean nothing without a project."""

    project: str
    entity: str | None = None
    offline: bool = False


def _best_argument():
    """A metric name or JSON policies through one CLI argument."""
    def read(given):
        text = given[0]
        if text == 'None':
            return None
        if text.startswith(('{', '[')):
            policies = json.loads(text)
            return (
                tuple(Best(**policy) for policy in policies)
                if isinstance(policies, list)
                else Best(**policies)
            )
        return Best(text)

    def write(policy):
        if policy is None:
            return ['None']
        if isinstance(policy, str):
            return [policy]
        return [
            json.dumps(
                [_to_json(entry, Best) for entry in policy]
                if isinstance(policy, tuple)
                else _to_json(policy, Best)
            )
        ]

    return tyro.constructors.PrimitiveConstructorSpec(
        nargs=1, metavar='METRIC|JSON', instance_from_str=read,
        is_instance=lambda policy: policy is None or isinstance(policy, (str, Best, tuple)),
        str_from_instance=write)


def _keep_argument():
    return tyro.constructors.PrimitiveConstructorSpec(
        nargs=1,
        metavar="LATEST|JSON",
        instance_from_str=lambda given: Keep(**json.loads(given[0]))
        if given[0].startswith("{")
        else int(given[0]),
        is_instance=lambda keep: isinstance(keep, (int, Keep)),
        str_from_instance=lambda keep: [
            json.dumps(_to_json(keep, Keep)) if isinstance(keep, Keep) else str(keep)
        ],
    )


def _cadence_argument():
    def read(given):
        value = given[0]
        if value == 'None':
            return None
        if value == 'epoch':
            return value
        return int(value) if value.isdecimal() else duration(value)
    return tyro.constructors.PrimitiveConstructorSpec(
        nargs=1,
        metavar="STEPS|DURATION|epoch",
        instance_from_str=read,
        is_instance=lambda value: value is None or isinstance(value, (int, str, datetime.timedelta)),
        str_from_instance=lambda value: [
            recorded_duration(value) if isinstance(value, datetime.timedelta) else str(value)
        ],
    )


@dataclasses.dataclass(frozen=True)
class TrainerConfig:
    """Holds the run length, checkpointing, sharding and run tracking."""

    name: str | None = None
    checkpoint_dir: str = "./checkpoints"
    keep: Annotated[int | Keep, _keep_argument()] = 2
    """Latest checkpoints kept, besides the best one."""
    best: Annotated[str | Best | tuple[Best, ...] | None, _best_argument()] = None
    """Metric name and ranking policy; None selects validation loss or training loss."""
    batch_size: int = 32
    """Global batch, over every process."""
    key: int = 0
    """Seed of the run key: parameter init and every per-step draw."""
    steps: int | None = None
    epochs: int | None = None
    """Run length as passes over the data; `steps` names it directly instead."""
    log_every: int = 100
    eval_every: int | Literal["epoch"] | None = "epoch"
    """Steps between validation passes: a number of steps, "epoch" for one
    pass over the data, None to never validate. "epoch" over a stream that
    reports no record count raises a ValueError, since it has no pass."""
    checkpoint_every: Annotated[int | str | datetime.timedelta | None, _cadence_argument()] = "epoch"
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
    mesh: MeshSpec = dataclasses.field(default_factory=MeshSpec)
    layout: Layout = dataclasses.field(default_factory=Layout)
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
    """Join the JAX process pool. None asks and continues alone only when no
    cluster is configured; True requires the pool; False never asks."""
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
        if isinstance(self.checkpoint_every, str) and self.checkpoint_every != 'epoch':
            object.__setattr__(self, 'checkpoint_every', duration(self.checkpoint_every))
        if isinstance(self.keep, Mapping):
            object.__setattr__(self, 'keep', _built(Keep, self.keep))
        if isinstance(self.keep, Keep) and self.keep.where is not None:
            raise TypeError("Keep.where is code-only; a recorded retention policy contains no callable")
        if self.best is not None:
            choices = self.best if isinstance(self.best, (tuple, list)) else (self.best,)
            rebuilt = []
            for choice in choices:
                if isinstance(choice, str):
                    choice = Best(choice)
                elif isinstance(choice, Mapping):
                    choice = _built(Best, choice)
                if not isinstance(choice, Best) or choice._source is not None:
                    raise TypeError("a recorded best selector names metrics; callable scores are code-only")
                rebuilt.append(choice)
            object.__setattr__(
                self, "best", tuple(rebuilt) if isinstance(self.best, (tuple, list)) else rebuilt[0]
            )

    def best_policies(self):
        if self.best is None:
            return None
        return (Best(self.best),) if isinstance(self.best, str) else (
            self.best if isinstance(self.best, tuple) else (self.best,))

    def total_steps(self, dataset: Dataset) -> int:
        """Return the run's length in steps, from `steps` or from `epochs` over `data`."""
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
        """Return the steps between validation passes over `data`, or None for never."""
        return self._interval(self.eval_every, dataset, "eval-every")

    def checkpoint_interval(self, dataset: Dataset) -> int | datetime.timedelta | None:
        """Steps, or a recorded duration such as 30m, between checkpoints."""
        value = self.checkpoint_every
        if isinstance(value, datetime.timedelta):
            if value.total_seconds() <= 0:
                raise ValueError("checkpoint_every duration must be positive")
            return value
        if isinstance(value, str) and value != 'epoch':
            return duration(value)
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


def _local_tracker(directory: str, name: str):
    """Return the journal directory for a run, beside its checkpoints where it can be.

    A bucket URI is not a directory this process can write journals into, so a
    remote run journals under Dew's cache (`dew_cache_dir`), keyed by the run name and
    a digest of the bucket path.
    """
    if "://" not in directory:
        return LocalTracker(os.path.join(directory, "tracking"))
    return LocalTracker(os.path.join(dew_cache_dir(), "tracking",
                                     _artifact_name(name) + "-"
                                     + hashlib.sha256(directory.encode()).hexdigest()[:12]))


def _closed(tracker, primary: BaseException | None) -> None:
    """Close `tracker` on every rank and agree on the outcome.

    `primary` is the exception the run is already unwinding with, if any. A
    close that fails is noted on it rather than replacing it, since the run's
    own failure is the one a caller wants.
    """
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


def _artifact_name(name: str) -> str:
    """Return `name` with every character a path or a tracker id cannot hold replaced.

    A run name is the caller's and may hold spaces or slashes; a directory
    entry and a tracker artifact id take neither.
    """
    return re.sub(r"[^\w.-]", "-", name)


def _registry_for(annotation):
    """Return the registry whose members the annotation names, or None."""
    members = typing.get_args(annotation) or (annotation,)
    for held in REGISTRIES:
        if all(any(member is m for m in held.values()) for member in members):
            return held
    return None


def _to_json(value, annotation) -> JSON:
    """Return `value` as JSON: a dict, a list, or a scalar json.dump can write.
    `annotation` is the declared field type, so the write side names the same
    registry and member types the read side rebuilds from."""
    if isinstance(value, datetime.timedelta):
        return recorded_duration(value)
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
                  for f in dataclasses.fields(value) if _recorded(f)}
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
        return {_key(key): _to_json(entry_value, entry)
                for (key, entry_value), entry in zip(value.items(), entries, strict=True)}
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(
        f"{type(value).__name__} is not something a run record can carry; a "
        f"config field holds JSON scalars, sequences, mappings, and the "
        f"registered values this writes as their name and fields")


def _key(key: object) -> str:
    """Return one mapping key as JSON names it, since JSON has only string keys.

    A tree path is its parts joined the way every message in this tree joins
    them, `params/layers_0/self_attn/q_proj`, which is the spelling
    `_rebuild` splits back into the tuple the field declares. A key that is
    neither a name nor a path of them has no spelling a record reads back,
    so it is refused here rather than written as its repr.
    """
    if isinstance(key, tuple):
        return "/".join(_key(part) for part in key)
    if not isinstance(key, (str, int, float)):
        raise TypeError(
            f"{key!r} is not a key a run record can carry; a config mapping is keyed "
            f"by a name, a number, or a tree path of names")
    return str(key)


def _rebuild_key(annotation: registry.Annotation, key: str) -> str | tuple[str, ...]:
    """Return one record key as the field declares it: a name, or the joined path."""
    return tuple(key.split("/")) if registry.wants_tuple(annotation) else key


def _instantiate(member: Callable, fields: Mapping[str, registry.Configured]) -> registry.Configured:
    """Call the member with the record's fields, as a value a config carries.

    The call is written here, behind `Callable[...]`, because a record names
    its fields at runtime and the class it builds cannot check them at type
    time; `_fields` has already refused any the class does not declare.
    """
    return registry.configured(member(**fields))


def _recorded(field: dataclasses.Field) -> bool:
    """Return whether a field is part of the run record.

    A field marked `metadata={"record": False}` is a binding the value picked
    up at runtime, not something the record describes: an adapter's source
    names, say, which follow from the source the record already names. It is
    neither written nor required, and a rebuilt value takes its default.
    """
    return field.init and field.metadata.get("record", True)


def _has_default(field: dataclasses.Field) -> bool:
    return field.default is not dataclasses.MISSING or field.default_factory is not dataclasses.MISSING


def _fields(cls: type, values: registry.Configured) -> dict[str, registry.Configured]:
    """The record's fields as `cls` declares them. A field the record lacks
    takes its declared default; a field `cls` does not declare, or a
    required one the record lacks, raises."""
    if not isinstance(values, Mapping):
        raise ValueError(f"{cls.__name__} is built from a record of its fields, not {values!r}")
    declared = [f for f in dataclasses.fields(cls) if _recorded(f)]
    unknown = sorted(set(values) - {f.name for f in declared})
    missing = [f.name for f in declared if f.name not in values and not _has_default(f)]
    if unknown or missing:
        raise ValueError(
            f"{cls.__name__} does not match the record: unknown fields {unknown}, "
            f"missing fields {missing}")
    return {f.name: _rebuild(_declared_type(cls, f.name), registry.configured(values[f.name]))
            for f in declared if f.name in values}


def _built[ValueT](cls: type[ValueT], values: Mapping[str, object]) -> ValueT:
    """Build one record into the class it describes, or raise naming what it built."""
    rebuilt = _rebuild(cls, values)
    if not isinstance(rebuilt, cls):
        raise ValueError(f"{values!r} builds a {type(rebuilt).__name__}, not a {cls.__name__}")
    return rebuilt


_MAPPINGS = (dict, MappingABC, MutableMapping)


def _rebuild(annotation: registry.Annotation, value: registry.Configured) -> registry.Configured:
    """Build the value `annotation` asks for, out of a record.

    It hands back what the field declares, which only the annotation knows,
    so the width here is what a config field can carry. `_built` is the same
    walk for a caller that holds the class and reads a value of it back.

    `dew.registry._rebuilt` is the sibling walk over a module field. That one
    resolves a `dtype` entry and walks a record with no value class behind it;
    this one reads registered members and honours `record: False`.
    """
    annotation = registry.resolve_alias(annotation)
    held = _registry_for(annotation)
    if held is not None:
        if not isinstance(value, Mapping):
            raise ValueError(f"a {held.kind} is the record that names it, not {value!r}")
        named = value["kind" if held.record == "kind" else "name"]
        if not isinstance(named, str):
            raise ValueError(f"a {held.kind} names a registered member, not {named!r}")
        member = held[named]
        if not isinstance(member, type):
            raise ValueError(f"the {held.kind} {named!r} is a function, and a record "
                             f"names the fields of a class")
        fields = ({name: entry for name, entry in value.items() if name != "kind"}
                  if held.record == "kind" else value["fields"])
        return _instantiate(member, _fields(member, registry.configured(fields)))
    if isinstance(annotation, type) and dataclasses.is_dataclass(annotation):
        return _instantiate(annotation, _fields(annotation, value))
    if typing.get_origin(annotation) in (typing.Union, types.UnionType):
        inner = [m for m in typing.get_args(annotation) if m is not type(None)]
        if value is None:
            return value
        if len(inner) == 1:
            return _rebuild(inner[0], value)
        # An optional registry union (a schedule or None) rebuilds through
        # its registry; any other union holds a JSON value as it is.
        members = functools.reduce(operator.or_, inner)
        return _rebuild(members, value) if _registry_for(members) is not None else value
    if typing.get_origin(annotation) in _MAPPINGS and isinstance(value, Mapping):
        # A mapping's own annotation names its keys and its values, and the
        # record carries neither: JSON keys are strings and JSON values are
        # the scalars and lists below. Both go back through this walk.
        keys, values = typing.get_args(annotation)
        return {_rebuild_key(keys, str(name)): _rebuild(values, registry.configured(entry))
                for name, entry in value.items()}
    if isinstance(value, list):
        # JSON writes every sequence as a list; the field says which are tuples.
        entries = registry.entry_types(annotation, len(value))
        rebuilt = [_rebuild(entry, record)
                   for entry, record in zip(entries, value, strict=True)]
        return tuple(rebuilt) if registry.wants_tuple(annotation) else rebuilt
    return value


@dataclasses.dataclass(frozen=True)
class RunConfig:
    """Describes a whole run. Recipes add their objective's knobs by subclassing this."""

    model: ModelConfig = dataclasses.field(default_factory=ModelConfig)
    data: DataSpec = dataclasses.field(default_factory=lambda: datasets["tfds_images"]())
    optim: OptimConfig = dataclasses.field(default_factory=OptimConfig)
    trainer: TrainerConfig = dataclasses.field(default_factory=TrainerConfig)
    objective: str | None = None
    lora: LoRA | None = None
    """The low-rank adapter the run trains instead of the whole model. The
    targets are the module paths under `params` a delta sits on; `train`
    adapts the objective's module and freezes every other leaf."""

    def to_dict(self) -> dict[str, JSON]:
        """Return a JSON-safe record of the run.

        A registered member is written as its name and its fields.
        """
        return {field.name: _to_json(getattr(self, field.name),
                                     _declared_type(type(self), field.name))
                for field in dataclasses.fields(self)}

    @classmethod
    def from_dict(cls, values: Mapping[str, object]) -> Self:
        """Read back what `to_dict` wrote, for subclasses too. A field the
        record lacks takes its default; an unknown field raises."""
        return _built(cls, values)

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
        """Read the config a run in `directory` was built from, as this class."""
        return cls.from_dict(json.loads((epath.Path(directory) / RUN_FILE).read_text()))

    def _naming(self, objective: Objective[Loss, Effects]) -> Self:
        """Return this config with `objective` named the way the record spells it.

        A registered objective is written under its registry name and anything
        else under its import path, so a record always says what was trained.
        """
        objective_type = type(objective)
        kind = (registry.objectives.name_of(objective_type) if objective_type in registry.objectives.values()
                else f"{objective_type.__module__}.{objective_type.__qualname__}")
        return dataclasses.replace(self, objective=kind)

    def train(self, objective: Objective[Loss, Effects], dataset: Dataset, *, name: str,
              metrics: Sequence[Metric] = (), rollout: Rollout | None = None,
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
        learns through, and a `lora` adapts the same module the same way,
        so the run traces the adapted forward and moves only its factors.
        `rollout` is the trainer's, which turns each prefetched batch into the
        one the step trains on, as an on-policy objective samples it.
        """
        if dataset.batch != self.trainer.batch_size:
            raise ValueError(
                f"--trainer.batch-size is {self.trainer.batch_size} and this dataset "
                f"reads {dataset.batch} records a step; load it with "
                f"load(batch={self.trainer.batch_size})")
        if self.trainer.quantization is not None:
            _quantize(objective, self.trainer.quantization)
        if self.lora is not None:
            _attach(objective, self.lora)
        self = self._naming(objective)
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
            local = _local_tracker(checkpoints.directory, name)
            tracker = Trackers(local, *(() if wandb_tracker is None else (wandb_tracker,)))
            def record_run() -> None:
                """Write the run's record and name where it is tracked, on rank zero."""
                if jax.process_index() == 0:
                    display = TrainingDisplay()
                    display.note(f"Experiment_Name: {name}")
                    display.note(f"Local tracking: {local.directory}")
                    self.save(checkpoints.directory)
                    tracker.artifact(RunRecord(name, json_value(self.to_dict()),
                        json_value(summary or {}), steps, packages_installed()), 0)

            agreed("run metadata", record_run)
            # The trainer holds mesh, layout, accumulation, dynamic_scale and
            # profile; prepare_process read the process fields, and the rest
            # are fit's arguments or built the checkpoints and the tracker.
            state = Trainer(
                objective, self.optim.build(steps), key=trainer.key,
                mesh=trainer.mesh, layout=trainer.layout, accumulation=trainer.accumulation,
                dynamic_scale=trainer.dynamic_scale, checkpoints=checkpoints, tracker=tracker,
                rollout=rollout, profile=trainer.profile,
            ).fit(
                dataset, steps=steps,
                log_every=trainer.log_every,
                eval_every=trainer.eval_interval(dataset),
                checkpoint_every=trainer.checkpoint_interval(dataset),
                metrics=metrics, preview=trainer.wandb is not None,
                best=trainer.best_policies(),
            )
            def publish_checkpoint() -> None:
                """Upload the checkpoint the run ended on, where a tracker takes one."""
                if wandb_tracker is not None:
                    # fit wrote the step it ended on, so that checkpoint is the run's.
                    dew.io.publish(checkpoints.path(int(state.step)), _artifact_name(name),
                                   tracker=wandb_tracker)

            agreed("checkpoint publishing", publish_checkpoint)
            return state

        finally:
            _closed(tracker, sys.exception())

    def sweep(self, space: Space, *, train: Callable[[Self], float], trials: int, ledger: str | Path,
              tracker: Tracker, search: Search = random_search, seed: int = 0) -> list[TrialFinished]:
        """Train `trials` trials of this config over `space` and return the ledger.

        Each trial draws a point from `space`, trains under the run name
        `<trainer.name>/trial-<index>` so trials keep their own checkpoints and
        tracking, and records the score `train` returns for it. `tracker`
        receives that score as `sweep/value` at the trial's number and the
        trial's `TrialFinished` record. A trial reaches `ledger` before it is
        reported, so rerunning the same call continues an interrupted sweep.
        """
        path = Path(ledger)
        if self.trainer.name is None:
            raise ValueError("a sweep needs trainer.name: every trial trains under "
                             "<trainer.name>/trial-<index>, and trials sharing one name would "
                             "resume from each other")
        finished = _read(path, space)
        for index in range(len(finished), trials):
            point = search(space, finished, seed)
            name = f"{self.trainer.name}/trial-{index}"
            value = train(override(self, {**point, "trainer.name": name}))
            trial = TrialFinished(index, name, point, value)
            finished.append(trial)
            _write(path, space, finished)
            tracker.log({"sweep/value": value}, index)
            tracker.artifact(trial, index)
        return finished
