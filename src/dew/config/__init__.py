"""Typed configuration for a training run.

One dataclass tree describes a run: what to build, what to feed it, how to
optimize it, and how the trainer runs. Recipes parse it with tyro, build the
objective and the data it names, and pass both to `RunConfig.train`, so
`to_dict()` is a full record of a run and `from_dict()` rebuilds it.

Model kwargs are an opaque JSON dict, and the registry knows which
architecture takes which fields. A dataset is the registered spec itself,
which tyro turns into a subcommand (`data:token-windows --data.path ...`).

The resolved config is the run's spec. A recipe writes it to `run.json` next
to the checkpoints with `save`, and `load` reads it back into the same class,
so inference rebuilds a run from what training was built from. A field the
file lacks takes its declared default, and a field the class does not have
raises an error.
"""

import dataclasses
import datetime
import hashlib
import json
import os
import re
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, ClassVar, Literal, Self

import jax
import optax
import tyro
from etils import epath

import dew.data  # registers the datasets a config names
import dew.io
import dew.nn.backbones  # registers the models a config names
from dew import registry
from dew.artifacts import agree_process_phase, agreed
from dew.cache import default_compilation_cache_dir, dew_cache_dir
from dew.checkpoints import RUN_FILE, Checkpoints, Keep
from dew.config.sweep import Search, Space, _read, _write, override, random_search
from dew.data import Dataset, DatasetSpec, Ramp
from dew.data.dataset import json_list_argument, ramped
from dew.lora import LoRA, _Adapted, adapted
from dew.nn.attention import AttentionImpl
from dew.objectives.base import Effects, Loss, Metric, Objective
from dew.records import JSON, duration, recorded_duration
from dew.registry import _declared_type, datasets, from_record, models, schedules, to_record, with_precision
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
from dew.training.quantization import Quantization, _quantize, _Quantized
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
    adapter: JsonDict | None = None
    """The bound LoRA record; its factors are stored with the checkpoint variables."""
    quantization: Quantization | None = None
    """The quantized training the model was wrapped in, which `build` wraps it in again."""
    dtype: registry.DtypeName | None = "bfloat16"
    """The compute dtype; parameter storage is set separately."""
    param_dtype: registry.DtypeName | None = None
    """The parameter storage dtype, where the model declares that field.

    Unset stores float32, the model's own default.
    """
    matmul_precision: Literal["default", "high", "highest"] | None = None
    """The precision every matmul in the model asks XLA for, where the model declares a `precision` field.

    `default` is the backend's fastest algorithm; `high` and `highest` trade
    throughput for mantissa bits (on Ampere and later, tf32 and fp32 against
    bf16x3). Unset keeps the model's own setting, `default`. Under bf16 compute, a
    decoder's vocabulary head at `default` rounds its logits and their gradient to
    bf16, as torch autocast does. `high` and `highest` keep that head in fp32,
    which is the setting to use when comparing parallel layouts in bf16
    (`dew.nn.precision.head_product`).
    """
    attention_impl: AttentionImpl = "auto"
    """The attention kernel.

    `dew.nn.attention` documents the kernels and how 'auto' chooses among them for
    each call.
    """

    def fields(self) -> Mapping[str, object]:
        """Return the model's fields with the run's precision settings in them."""
        return with_precision(self.architecture, self.config,
                              dtype=self.dtype, attention_impl=self.attention_impl,
                              param_dtype=self.param_dtype,
                              matmul_precision=self.matmul_precision)

    def precision_settings(self) -> frozenset[str]:
        """Return the names `fields()` writes that `config` did not have.

        These are the run's precision settings, as this architecture takes them. A
        resolved record leaves them out, because this value writes them again every
        time it builds the model.
        """
        return frozenset(self.fields()) - frozenset(self.config)

    @classmethod
    def from_dict(cls, values: Mapping[str, object]) -> Self:
        """Read back the record `RunConfig.to_dict` writes for this field."""
        return from_record(cls, values, dtypes=False)

    @classmethod
    def from_model(cls, model) -> Self:
        """Return the module's constructor fields, with its actual compute settings.

        A registered class is recorded under its registered name. A class that no
        registry names is recorded under its own name in lower case, and a loader asks
        the user to register it under that name: `build` needs the name, but training,
        checkpointing and resuming do not.
        """
        model_type = type(model)
        adapter, quantization = None, None
        if isinstance(model_type, _Adapted):
            adapter = model_type._dew_lora_record()
            model_type = model_type._dew_lora_base
        if isinstance(model_type, _Quantized):
            quantization = model_type._dew_quantization
            model_type = model_type._unquantized_type
        architecture = _architecture(model_type)
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
                fields[field.name] = to_record(value, _declared_type(model_type, field.name))
        return cls(architecture, fields, adapter=adapter, quantization=quantization, dtype=compute,
                   param_dtype=storage, matmul_precision=precision, attention_impl=attention)

    def build(self):
        model = models.build(self.architecture, self.fields())
        if self.quantization is not None:
            model = self.quantization.apply(model)
        return model if self.adapter is None else adapted(model, self.adapter)


def _architecture(model_type: type) -> str:
    """The name a model class is recorded under: its registered name, or its
    own name in lower case for a class no registry names yet."""
    for name, member in models.items():
        if member is model_type:
            return name
    name = model_type.__name__.lower()
    if name in models:
        raise ValueError(f"{model_type.__qualname__} is unregistered and {name!r} names "
                         f"{models[name].__qualname__}; register it under a name of its own")
    return name



@dataclasses.dataclass(frozen=True)
class OptimConfig:
    """Holds the optimizer, learning-rate schedule and gradient clipping."""

    optimizer: Literal["adam", "adamw", "lamb", "muon", "muonclip"] = "adamw"
    optimizer_opts: JsonDict = dataclasses.field(default_factory=dict)
    learning_rate: float = 2.7e-4
    """The constant rate, when no schedule is named."""
    schedule: ScheduleSpec | None = None
    """The learning-rate schedule, as one typed record per kind (`dew.training.optim`).

    The kinds are cosine, power (lm-engine's power law with an optional linear
    tail), linear, one_cycle (torch's `OneCycleLR`) and exponential; each record
    holds only its own fields, and `every` steps any of them once per that many
    updates.
    """
    weight_decay: float | None = None
    """The weight decay passed to the optimizer; for 'adam' it is torch's coupled L2
    penalty, added to the gradient before the moments read it, for 'adamw' the
    decoupled decay."""
    param_groups: Annotated[tuple[ParamGroup, ...], json_list_argument(ParamGroup)] = ()
    """Per-group learning rates, momentum schedules, weight decay and bounds; the first
    matching group wins.

    Empty treats every parameter alike. `ParamGroup.mup` is lm-engine's muP split.
    """
    clip_grads: float = 0.0
    state_dtype: Literal["float32", "bfloat16"] = "float32"
    """The dtype of Adam's moments in memory.

    bfloat16 stores both with stochastic rounding (`dew.training.optim.bf16_moments`),
    for adam and adamw only. That halves the optimizer state and reduces the
    update's memory traffic.
    """
    forced_weight_normalization: bool = False
    """Whether to renormalize every magnitude-preserving weight (`dew.nn.mp.MPConv`) after each update.

    This is EDM2's forced weight normalization, which its `edm2_unet` trains with
    (`dew.nn.mp.forced_weight_normalization`).
    """
    ema_profiles: tuple[float, ...] = ()
    """Relative standard deviations of the power-function EMAs a run keeps for post-hoc EMA.

    See `dew.training.optim.power_profiles`; Karras et al. use (0.05, 0.10). Every
    checkpoint save snapshots these averages, and `Checkpoints.posthoc_ema` builds
    an average with any other relative standard deviation from the snapshots.
    Empty keeps none.
    """

    def build(self, steps: int) -> optax.GradientTransformation:
        """Build the optimizer this config describes, with its schedule, parameter groups and clipping.

        `steps` is the run's length, which the schedule decays over unless the config
        names its own end. With `param_groups`, one optimizer runs per group under
        `optax.multi_transform`, each on its own schedule (the config's when it names
        none) times its multiplier, with its own weight decay, `b1` schedule and bounds
        (`ParamGroup.solver`); the global-norm clip still reads every gradient
        together, before the groups split them.
        """
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
                rate = learning_rate if group.schedule is None else group.schedule.schedule(steps)
                solvers[group.name] = group.solver(
                    make, _scaled(rate, group.learning_rate_multiplier), steps, group_opts)
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
    """Where a run reports to Weights & Biases.

    Setting it turns tracking on; the entity and the offline switch have no effect
    without a project.
    """

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
                [to_record(entry, Best) for entry in policy]
                if isinstance(policy, tuple)
                else to_record(policy, Best)
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
            json.dumps(to_record(keep, Keep)) if isinstance(keep, Keep) else str(keep)
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
    """The number of latest checkpoints kept, besides the best one."""
    best: Annotated[str | Best | tuple[Best, ...] | None, _best_argument()] = None
    """The metric name and ranking policy; None selects validation loss or training loss."""
    batch_size: int = 32
    """The global batch size, over every process."""
    key: int = 0
    """The seed of the run key, which sets parameter init and every per-step draw."""
    steps: int | None = None
    epochs: int | None = None
    """The run length as passes over the data; `steps` sets it directly instead."""
    log_every: int = 100
    eval_every: int | Literal["epoch"] | None = "epoch"
    """The steps between validation passes.

    It is a number of steps, "epoch" for one pass over the data, or None to never
    validate. "epoch" over a stream that reports no record count raises a
    ValueError, since such a stream has no pass.
    """
    checkpoint_every: Annotated[int | str | datetime.timedelta | None, _cadence_argument()] = "epoch"
    """The steps between checkpoints, with the same three choices as `eval_every`.

    A stream whose iterator cannot report its read position must train with None;
    the trainer refuses any other value for it.
    """
    accumulation: int = 1
    """The number of micro-batches per optimizer update."""
    batch_ramp: Ramp | None = None
    """A ramp that grows `batch_size` over the run's first records instead of starting there.

    It sets the global batch the run starts at, what each stage adds, and the
    records the whole ramp spans. Unset trains at `batch_size` throughout. Either
    way there is one optimizer update a step, and the compiled step is traced
    once per stage.
    """
    dynamic_scale: bool = False
    mesh: MeshSpec = dataclasses.field(default_factory=MeshSpec)
    layout: Layout = dataclasses.field(default_factory=Layout)
    profile: ProfileWindow | None = None
    """One profiler window: the steps to trace, the warmup before them and the output directory.

    Unset traces nothing.
    """
    compilation_cache_dir: str | None = dataclasses.field(
        default_factory=default_compilation_cache_dir)
    """The directory for a persistent XLA cache, so a restart skips recompiling the step.

    None compiles from scratch every run.
    """
    wandb: Wandb | None = None
    """An optional W&B sink, in addition to the local tracking journal."""
    multi_host: bool | None = None
    """Whether to join the JAX process pool.

    None tries, and continues alone only when no cluster is configured; True
    requires the pool; False never joins.
    """
    xla_flags: str | None = None
    """Extra XLA_FLAGS for this run, appended to the environment by `prepare_process`.

    `prepare_process` appends them before JAX opens a backend. Library users set
    XLA_FLAGS themselves; docs/performance.md records what was measured.
    """
    quantization: Quantization | None = None
    """The quantized-training spec, wrapped around the module the objective trains.

    The wrapper is applied before the run initializes the module. Unset trains in
    the compute dtype. `dew.training.quantization` describes what the wrapper does
    and what it keeps.
    """

    def __post_init__(self):
        if self.steps is not None and self.epochs is not None:
            raise ValueError("steps and epochs both name the run length; set one")
        if isinstance(self.checkpoint_every, str) and self.checkpoint_every != 'epoch':
            object.__setattr__(self, 'checkpoint_every', duration(self.checkpoint_every))
        if isinstance(self.keep, Mapping):
            object.__setattr__(self, 'keep', from_record(Keep, self.keep, dtypes=False))
        if isinstance(self.keep, Keep) and self.keep.where is not None:
            raise TypeError("Keep.where is code-only; a recorded retention policy contains no callable")
        if self.best is not None:
            choices = self.best if isinstance(self.best, (tuple, list)) else (self.best,)
            rebuilt = []
            for choice in choices:
                if isinstance(choice, str):
                    choice = Best(choice)
                elif isinstance(choice, Mapping):
                    choice = from_record(Best, choice, dtypes=False)
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
        """Return the steps, or a recorded duration such as 30m, between checkpoints."""
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


@dataclasses.dataclass(frozen=True)
class RunConfig:
    """A whole run's configuration; recipes subclass it to add their objective's settings."""

    _FLAG_SELECTED: ClassVar[Mapping[str, str]] = {"lora": "lora"}
    """The optional settings `cli` turns on by their own flags, each with the
    subcommand its flags imply."""

    model: ModelConfig = dataclasses.field(default_factory=ModelConfig)
    data: DataSpec = dataclasses.field(default_factory=lambda: datasets["tfds_images"]())
    optim: OptimConfig = dataclasses.field(default_factory=OptimConfig)
    trainer: TrainerConfig = dataclasses.field(default_factory=TrainerConfig)
    objective: str | None = None
    lora: Annotated[LoRA, tyro.conf.subcommand("lora")] | None = None
    """The low-rank adapter the run trains instead of the whole model.

    On the command line, set it with `--lora.rank 16 --lora.modules q_proj v_proj`
    (see `cli`). A recipe attaches it to the source that `--pretrained` loads
    (`Pretrained.adapt`), or, when training from scratch, to a model freshly
    initialized from the run's key. The objective then trains only the adapter's
    factors, and the run's record stores the model with the adapter attached.
    """

    def to_dict(self) -> dict[str, JSON]:
        """Return a JSON-safe record of the run.

        A registered member is written as its name and its fields.
        """
        return {field.name: to_record(getattr(self, field.name),
                                     _declared_type(type(self), field.name))
                for field in dataclasses.fields(self)}

    @classmethod
    def from_dict(cls, values: Mapping[str, object]) -> Self:
        """Read back what `to_dict` wrote, for subclasses too. A field the
        record lacks takes its default; an unknown field raises."""
        return from_record(cls, values, dtypes=False)

    def save(self, directory: str) -> str:
        """Write this config as `run.json` in `directory` and return the path.

        The path goes through `epath`, the same filesystem layer Orbax writes the
        checkpoints with, so a `gs://` run directory gets the record too.
        """
        path = epath.Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        target = path / RUN_FILE
        target.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True))
        return str(target)

    @classmethod
    def cli(cls, args: Sequence[str] | None = None, *, default: Self | None = None) -> Self:
        """Parse a run from `args`, or from the process's own command line when `args` is None.

        An optional setting such as `lora` is turned on by its own flags. For
        example, `--lora.rank 16 --lora.modules q_proj` stands for
        `lora:lora --lora.rank 16 ...`, and with no `--lora.` flag the run
        trains without an adapter.
        """
        given = list(sys.argv[1:] if args is None else args)
        for field, subcommand in cls._FLAG_SELECTED.items():
            flags = [index for index, arg in enumerate(given) if arg.startswith(f"--{field}.")]
            if flags and not any(arg.startswith(f"{field}:") for arg in given):
                given.insert(flags[0], f"{field}:{subcommand}")
        parser = tyro.conf.CascadeSubcommandArgs[cls]
        if default is None:
            return tyro.cli(parser, args=given)
        return tyro.cli(parser, args=given, default=default)

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
        """Train `objective` on `data` as this config describes; every recipe calls this after building both.

        The run lives under `name` in `trainer.checkpoint_dir`, and process zero writes
        the record there before training starts. A `trainer.wandb` opens a tracker
        under the same name, with the record, `summary` (the recipe's own view of the
        run) and the step count as its config. The final checkpoint is published to
        the W&B registry under the name with slashes and spaces replaced, because an
        artifact name allows neither. Local tracking journals live in a separate
        tracking directory.

        `dataset` must be the one the recipe loaded at `trainer.batch_size`; a dataset
        at any other batch size is refused here. The record, the ramp and the reported
        throughput all use the configured batch size while each step reads the
        dataset's, so a mismatch would train at a batch size the run does not report.

        A `trainer.quantization` wraps the module `objective` trains before anything
        initializes it, so the run learns through the quantized forward pass. A
        `lora` is not applied here: the recipe attaches it to the model and variables
        it builds the objective from. `rollout` is the trainer's rollout, which turns
        each prefetched batch into the one the step trains on, as an on-policy
        objective samples it.
        """
        if dataset.batch != self.trainer.batch_size:
            raise ValueError(
                f"--trainer.batch-size is {self.trainer.batch_size} and this dataset "
                f"reads {dataset.batch} records a step; load it with "
                f"load(batch={self.trainer.batch_size})")
        if self.trainer.quantization is not None:
            _quantize(objective, self.trainer.quantization)
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

        Each trial draws a point from `space` and trains under the run name
        `<trainer.name>/trial-<index>`, so trials keep their own checkpoints and
        tracking, and the score `train` returns is recorded for it. `tracker` receives
        that score as `sweep/value` at the trial's number, along with the trial's
        `TrialFinished` record. A trial is written to `ledger` before it is reported,
        so rerunning the same call continues an interrupted sweep.
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


__all__ = ["JsonDict", "ModelConfig", "OptimConfig", "RunConfig", "TrainerConfig", "Wandb"]
