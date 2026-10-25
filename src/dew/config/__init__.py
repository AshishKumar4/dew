"""Typed configuration for a training run.

One dataclass tree describes a run: what to build, what to feed it, how to
optimize it, and how the trainer runs. Recipes parse it with tyro, build the
objective and the data it names, and pass both to `RunConfig.train`, so
`to_dict()` is a full record of a run and `from_dict()` rebuilds it.

The model is its class and the fields the run gives it: `--model
hybrid_dit` (an alias or an import path) picks the class, and each of its
own fields is a flag typed by its annotation (`--model.num_layers 12`,
`--model.dtype bfloat16`). The data is a chain (`dew.data`), which the record writes through
the data's codec (`dew.data.records`) and the command line sets one setting of at a time:
`--set data.<call>.<setting>=...` for a call's (`data.token_windows.seq_len=512`) and
`--set data.source.<field>=...` for its store's.

The resolved config is the run's spec. A recipe writes it to `run.json` next
to the checkpoints with `save`, and `load` reads it back into the same class,
so inference rebuilds a run from what training was built from. A field the
file lacks takes its declared default, and a field the class does not have
raises an error. The model and the objective record their classes' defaults
too (`ModelConfig.defaults`), so a default changed since leaves the record
building what it built.
"""

import copy
import dataclasses
import datetime
import functools
import hashlib
import json
import os
import re
import shlex
import sys
import types
import typing
from collections.abc import Callable, Mapping, Sequence
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, ClassVar, Literal, Self, TypedDict, Union

import jax
import tyro
from etils import epath

import dew.io
from dew import registry
from dew._coordination import agree_process_phase, agreed
from dew._records import JSON, duration, record, recorded_duration
from dew.cache import dew_cache_dir
from dew.checkpoints import RUN_FILE, Checkpoints, Keep
from dew.config.sweep import RandomSearch, Search, Space, read_ledger, write_ledger
from dew.data import records
from dew.data.compile import fitted
from dew.data.dataset import Loader, Ramp, Reader, ramped
from dew.data.pipeline import Dataset
from dew.files import write_atomically
from dew.inputs import Field
from dew.lora import LoRA
from dew.objectives.base import Effects, Loss, Metric, Objective
from dew.optim import OptimConfig
from dew.quantization import Quantization
from dew.registry import (
    Annotation,
    ModelConfig,
    Registered,
    declared_type,
    default_record,
    from_record,
    member_fields,
    metrics,
    models,
    objectives,
    parameters,
    to_record,
    wants_tuple,
)
from dew.state import TrainState
from dew.telemetry.records import RunRecord, TrialFinished, json_value, packages_installed
from dew.training.display import TrainingDisplay
from dew.training.distributed import Layout, MeshSpec
from dew.training.profiling import ProfileWindow
from dew.training.quantization import quantize_trunk
from dew.training.selection import Best
from dew.training.tracker import LocalTracker, Tracker, Trackers, Wandb, WandbTracker
from dew.training.trainer import Rollout, Trainer, run_length, run_steps

if TYPE_CHECKING:
    from _typeshed import DataclassInstance


@dataclasses.dataclass(frozen=True)
class ObjectiveConfig(Registered):
    """An objective's class and the arguments the run gives it.

    `name` is an alias or an import path, held as the path. `fields` are the
    constructor arguments the run states, held as their records
    (`registry.argument_records`), which the command line sets one flag each
    (`--objective.ema_decay 0.999`); what the objective is built around, the
    model and what a recipe reads off its data, `build` takes from the caller.
    `defaults` holds every other argument the class defaults, as `ModelConfig`
    holds a model's, so an EMA decay, a loss weighting or a sampling step count
    the class defaults differently later leaves a recorded run as it was.
    """

    name: str

    @staticmethod
    def _member(name: str) -> type:
        return objectives[name]

    def build(self, **derived) -> Objective:
        """Build the objective from its arguments and those the caller
        `derived`, which replace a recorded default; an argument the run
        states is refused there, since the caller's would silently replace
        it."""
        stated = sorted(set(self.fields) & set(derived))
        if stated:
            raise ValueError(f"the run derives {stated} for {self.name}; leave them out of its arguments")
        return objectives.build(self.name, self.arguments, **derived)


@dataclasses.dataclass(frozen=True)
class MetricConfig(Registered):
    """A metric's class and the arguments the run gives it.

    `name` is an alias of `dew.registry.metrics` or an import path, held as
    the path. `fields` are the constructor arguments the run states, as
    `ObjectiveConfig` holds an objective's: a CLIP metric's `modelname`, an
    FID's `weights`. `defaults` holds every other argument the class defaults,
    so a metric whose default model changes later scores a recorded run as it
    did.
    """

    name: str

    @staticmethod
    def _member(name: str) -> type:
        return _metric_class(name)

    def build(self, **derived) -> Metric:
        """Build the metric from its arguments and those the run `derived`
        where it states none of its own (the field a video run's sample
        is read from, say)."""
        unstated = {name: value for name, value in derived.items() if name not in self.fields}
        return metrics.build(self.name, self.arguments, **unstated)


def _metric_class(name: str) -> type:
    """The metric class `name` names, an alias or import path: a run's metric is a class, whose
    arguments its record holds."""
    member = metrics[name]
    if not isinstance(member, type):
        raise TypeError(f"{name!r} names {member!r}, which is no metric class")
    return member


def _member_flags(member: type, given: Mapping[str, object]) -> tuple[type, Mapping[str, object]]:
    """A dataclass of `member`'s own fields, one flag each, defaulting to
    `given` over the member's defaults, and the value each flag starts at.

    A model's fields are its dataclass fields. An objective's are its
    constructor's arguments, but for those it takes positionally without a
    default: those are what its caller builds and passes, the model first.
    Scalars, literals and tuples of them are typed as declared; dtypes read
    their names and precision its level. Any other field (a sequence of
    records, a nested model) takes one JSON value, the record `build` reads.
    A field with no default also takes None, for leaving it to the recipe. A
    field whose default no record carries (a callable, say) gets no flag and
    keeps that default."""
    flags, started = [], {}
    # One record: a module two defaults share is written once.
    with registry.recording():
        for name, declared, annotation in member_fields(member):
            declared = None if declared is dataclasses.MISSING else declared
            annotation = registry.resolve_alias(annotation)
            if name in ("dtype", "param_dtype"):
                typed = registry.DtypeName | None
                declared = default_record(name, declared, annotation)
            elif name == "precision":
                typed = Literal["default", "high", "highest"] | None
                declared = default_record(name, declared, annotation)
            elif annotation is not None and _scalar(annotation):
                typed = annotation if declared is not None else annotation | None
            else:
                typed = Annotated[JSON, _JSON_FLAG]
                try:
                    declared = default_record(name, declared, annotation)
                except (TypeError, ValueError):
                    continue  # a field no record carries keeps its class default
            started[name] = given.get(name, declared)
            copied = functools.partial(copy.deepcopy, started[name])
            flags.append((name, typed, dataclasses.field(default_factory=copied)))
    return dataclasses.make_dataclass(f"{member.__name__}Fields", flags, frozen=True), started


def _chosen(given: list[str], field: str) -> str | None:
    """Pop `--<field> <name>` or `--<field>=<name>` off `given` and return the name, or None."""
    index = next((index for index, arg in enumerate(given)
                  if arg == f"--{field}" or arg.startswith(f"--{field}=")), None)
    if index is None:
        return None
    return given.pop(index).removeprefix(f"--{field}").removeprefix("=") or given.pop(index)


def _given[C: (ModelConfig, ObjectiveConfig)](start: C, parsed: "DataclassInstance",
                                              started: Mapping[str, object]) -> C:
    """`start` with the fields its flags `parsed`: those it started from, and
    those the command line set to something other than the flag `started`
    at. A field it started from that takes no flag is kept as it was, and so
    is a recorded default the command line left alone. A tuple is held as
    the list its record reads back as."""
    def listed(value):
        return [listed(entry) for entry in value] if isinstance(value, tuple) else value

    chosen = {key: listed(value) for key, value in vars(parsed).items()
              if key in start.fields or listed(value) != listed(started[key])}
    return dataclasses.replace(start, fields={**start.fields, **chosen})


def _scalar(annotation: Annotation) -> bool:
    """Whether tyro reads `annotation` as one typed flag: a bool, number,
    string, literal or enum, a tuple of those, or an optional one."""
    origin = typing.get_origin(annotation)
    if origin in (Union, types.UnionType):
        return all(_scalar(member) for member in typing.get_args(annotation) if member is not type(None))
    if origin is Literal:
        return True
    if origin in (tuple, Sequence):
        return all(_scalar(registry.resolve_alias(member)) for member in typing.get_args(annotation)
                   if member is not Ellipsis)
    return annotation in (bool, int, float, str) or (isinstance(annotation, type)
                                                      and issubclass(annotation, Enum))


_JSON_FLAG = tyro.constructors.PrimitiveConstructorSpec(
    nargs=1, metavar="JSON", instance_from_str=lambda given: _flag_json(given[0]),
    is_instance=lambda given: True, str_from_instance=lambda given: [json.dumps(given)])
"""A model field tyro cannot type, written as the JSON record `build` reads."""


def _flag_json(text: str) -> JSON:
    """A JSON flag's value: its JSON, or None for `None`, as every optional flag spells it."""
    return None if text == "None" else json.loads(text)


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


class SourcePlacement(TypedDict, total=False):
    """Where a run reads the source it starts from: `Pretrained.load`'s keywords
    (`TrainerConfig.source_placement`)."""

    weights: bool
    on_host: bool
    mesh: MeshSpec
    layout: Layout


@dataclasses.dataclass(frozen=True)
class TrainerConfig:
    """Holds the run length, checkpointing, sharding and run tracking."""

    name: str | None = None
    checkpoint_dir: str = "./checkpoints"
    keep: Annotated[int | Keep, _keep_argument()] = Keep.latest
    """The number of latest checkpoints kept, besides the best one (`Keep`'s own default)."""
    best: Annotated[str | Best | tuple[Best, ...] | None, _best_argument()] = None
    """The metric name and ranking policy; None selects validation loss or training loss."""
    update_tokens: int | None = None
    """The optimizer update's token budget, a whole number of the data's token batches (its batch stage's
    `TokenBudget`), which sets the run's accumulation (`RunConfig.train`)."""
    key: int = 0
    """The seed of the run key, which sets parameter init and every per-step draw."""
    steps: int | None = None
    epochs: int | None = None
    """The run length as passes over the data; `steps` sets it directly instead."""
    tokens: int | None = None
    """Stop after the loss has scored this many targets across all microbatches, including before a
    resume.

    Uses the counter reported as `train/tokens`, without padding. The final
    microbatch may cross the budget. With `steps` or `epochs`, the first cap
    reached ends the run. A token-only run needs a constant learning rate or
    a schedule with its own explicit step horizon.
    """
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
    checkpoint_final: bool = True
    """Write the final state when training ends, independently of `checkpoint_every`.

    False disables this save; cadenced, ranked and preemption saves keep their own policy.
    """
    accumulation: int = 1
    """The number of steps per optimizer update."""
    split: int | None = None
    """The microbatches each step's batch is read in, its rows split evenly.

    None lets the trainer plan it from the compiled step's memory: the
    fewest that fit the devices before it recomputes anything, at the least
    recomputation that fits (`Trainer.compile`). A step's tokens and the
    update's stay as configured either way; only the microbatch shrinks.
    """
    batch_ramp: Ramp | None = None
    """A ramp that grows `batch_size` over the run's first records instead of starting there.

    It sets the global batch the run starts at, what each stage adds, and the
    records the whole ramp spans. Unset trains at `batch_size` throughout. Either
    way there is one optimizer update a step, and the compiled step is traced
    once per stage.
    """
    dynamic_scale: bool = False
    mesh: MeshSpec = dataclasses.field(default_factory=MeshSpec)
    microbatches: int | None = None
    """Microbatches a pipelined step feeds through the mesh's stages, a
    multiple of `mesh.stage`; unset feeds one a stage."""
    layout: Layout = dataclasses.field(default_factory=Layout)
    profile: ProfileWindow | None = None
    """One profiler window: the steps to trace, the warmup before them and the output directory.

    Unset traces nothing.
    """
    compilation_cache_dir: str | None = None
    """The directory of JAX's persistent compilation cache, so a restart skips recompiling the step.

    None caches under the user's cache directory wherever the run's processes share it
    (`dew.training.runtime.ready_process`); a pool across hosts without shared storage names one
    they share. JAX_ENABLE_COMPILATION_CACHE=false compiles from scratch every run.
    """
    wandb: Wandb | None = None
    """An optional W&B telemetry sink, in addition to the local tracking journal.

    Scalars, records and objective previews are tracked; checkpoints are not uploaded.
    """
    publish: bool = False
    """Publish the final checkpoint to W&B; requires `wandb` and `checkpoint_final`."""
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
    the compute dtype. `dew.quantization` describes what the wrapper does
    and what it keeps.
    """
    lower_for: str | None = None
    """Check the run's step for a GPU on any host, a CPU too, then exit without training.

    `--trainer.lower-for sm80` (or sm75, sm86, sm89, sm90, sm120) builds the run
    as it would, traces its first step as on a GPU of that CUDA generation, with
    every kernel choice and refusal that GPU makes, and lowers it for CUDA,
    each of its Pallas kernels to Triton. A kernel that would fail the GPU's
    first compile there fails here, in seconds and without the GPU; XLA's
    compile of the whole step still needs one. The program exits with status
    0 once the step lowers, and opens no W&B run.
    """
    tune_kernels: bool = False
    """Measure each Pallas launch of the step that has no config measured on this GPU.

    Only a process with the GPU to itself measures, and the picks persist beside
    the compilation cache for every later run. Unset, such a launch runs the
    nearest measured shape's config or its heuristic, and the run says how many.
    """

    def accumulating(self, dataset: Loader) -> Self:
        """This config with the accumulation its token budget sets over `dataset`'s token batches:
        `update_tokens` a whole number of them. Raises ValueError for a budget over data batched in
        rows, one no whole number of them, or an accumulation set otherwise."""
        if self.update_tokens is None:
            return self
        if dataset.batch_tokens is None or self.update_tokens % dataset.batch_tokens:
            raise ValueError(f"--trainer.update-tokens {self.update_tokens} is no whole number of the data's "
                             f"token batches ({dataset.batch_tokens}); batch the data by tokens "
                             "(Dataset.batch(tokens=)) dividing it")
        accumulation = self.update_tokens // dataset.batch_tokens
        if self.accumulation not in (1, accumulation):
            raise ValueError(f"--trainer.update-tokens sets the accumulation to {accumulation}; omit "
                             "--trainer.accumulation or match it")
        return dataclasses.replace(self, accumulation=accumulation)

    def __post_init__(self):
        if self.update_tokens is not None and self.update_tokens < 1:
            raise ValueError("--trainer.update-tokens must be positive")
        if self.update_tokens is not None and self.batch_ramp is not None:
            raise ValueError("--trainer.batch-ramp grows row batches and cannot use a token budget")
        if self.split is not None and self.split < 1:
            raise ValueError("--trainer.split reads each step's batch in that many microbatches, at least 1")
        if self.publish and (self.wandb is None or not self.checkpoint_final):
            raise ValueError("trainer.publish requires trainer.wandb and trainer.checkpoint_final")
        run_length(self.steps, self.epochs, self.tokens, required=False)
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

    def resumes(self) -> bool:
        """Whether this run continues from a checkpoint of its own: a named run whose directory holds a
        step a resume reads (`Checkpoints.latest`). Its weights then come from that step, so the source
        it started from need not be read (`Pretrained.load`'s `weights`)."""
        if self.name is None:
            return False
        with Checkpoints(os.path.join(self.checkpoint_dir, self.name), keep=self.keep) as checkpoints:
            return checkpoints.latest is not None

    def source_placement(self, *, frozen: bool) -> SourcePlacement:
        """Where a run that starts from a source (`Pretrained.load`) reads it: the load's keywords,
        `frozen` where the run keeps the source as it is (an adapter trained over it). A run that
        resumes reads no weights (`resumes`), as its checkpoint holds them. A frozen source is read onto
        this run's mesh, sharded as its layout shards the state, each process reading its devices'
        shards, and the start takes those arrays as they are (`Trainer.place`): the run and its
        objective hold one copy. A source the run trains stays on the host, as its start copies the
        arrays it trains, and so does one whose state lives on the host (`Layout.host` holds the
        variables) or keeps its frozen tree whole (`frozen_whole`), which its start places itself."""
        if self.resumes():
            return {"weights": False}
        if not frozen or "variables" in self.layout.host or self.layout.frozen_whole:
            return {"on_host": True}
        return {"mesh": self.mesh, "layout": self.layout}

    def total_steps(self, dataset: Loader) -> int | None:
        """Return the step cap from `steps` or `epochs`, None for a token-only run (`run_steps`, the
        one policy a direct fit applies too)."""
        return run_steps(dataset, self.steps, self.epochs, self.tokens, self.accumulation)

    def total_updates(self, dataset: Loader) -> int | None:
        """Return the optimizer updates the run takes over `dataset` as `RunConfig.train` finalizes it,
        its token budget's accumulation and its batch ramp applied: what an average that moves once an
        update ramps over. None for a token-only run."""
        trainer = self.accumulating(dataset)
        ramp = trainer.batch_ramp
        steps = trainer.total_steps(dataset if ramp is None else ramped(dataset, ramp))
        # A window the run ends inside is never committed, as the trainer counts its updates.
        return None if steps is None else steps // trainer.accumulation

    def eval_interval(self, dataset: Loader) -> int | None:
        """Return the steps between validation passes over `data`, or None for never."""
        return self._interval(self.eval_every, dataset, "eval-every")

    def checkpoint_interval(self, dataset: Loader) -> int | datetime.timedelta | None:
        """Return the steps, or a recorded duration such as 30m, between checkpoints."""
        value = self.checkpoint_every
        if isinstance(value, datetime.timedelta):
            if value.total_seconds() <= 0:
                raise ValueError("checkpoint_every duration must be positive")
            return value
        return self._interval(self.checkpoint_every, dataset, "checkpoint-every")

    def _interval(self, value, dataset: Loader, field_name: str) -> int | None:
        if value is None or isinstance(value, int):
            return value
        if dataset.steps_per_epoch is None:
            raise ValueError(
                f"--trainer.{field_name} epoch needs a dataset with a record count; this "
                f"one streams without one, so give the interval in steps or None")
        return dataset.epoch_steps(window=self.accumulation)


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


def _closed(tracker, checkpoints: Checkpoints | None, primary: BaseException | None) -> None:
    """Close `tracker` on every rank and agree on the outcome, then close `checkpoints`, even where
    the tracker's close failed.

    `primary` is the exception the run is already unwinding with, if any. A
    close that fails is noted on it rather than replacing it, since the run's
    own failure is the one a caller wants; without one, the first failure is
    raised with the other noted on it.
    """
    error = None
    try:
        if tracker is not None:
            tracker.__exit__(type(primary), primary, None)
    except BaseException as failure:
        error = failure
    failed = None
    try:
        agree_process_phase(error, phase="tracker close")
    except BaseException as failure:
        if primary is None:
            failed = failure
        else:
            primary.add_note(f"Tracker close failed: {failure!r}")
    raised = primary if primary is not None else failed
    if checkpoints is not None:
        checkpoints.__exit__(None if raised is None else type(raised), raised, None)
    if failed is not None:
        raise failed


def _artifact_name(name: str) -> str:
    """Return `name` with every character a path or a tracker id cannot hold replaced.

    A run name is the caller's and may hold spaces or slashes; a directory
    entry and a tracker artifact id take neither.
    """
    return re.sub(r"[^\w.-]", "-", name)


@dataclasses.dataclass(frozen=True)
class Prepared:
    """What a run trains, as its class builds it (`RunConfig.prepare`).

    `run` is the run as it records itself, with what building resolved (a
    pinned source, the model's fields as built). `train` trains it under a
    name: `run.train` over the objective and the data the class built.
    """

    run: "RunConfig"
    train: Callable[[str], TrainState]


def media_field(chain: Dataset, run: str) -> Field:
    """The batch field a run over images or clips reads, at the size its chain makes it
    (`Dataset.facts`): a clip's video, else an image. Raises ValueError, naming the `run`, for a chain
    of neither, or one whose size varies."""
    fields = record(chain.facts()["fields"], "the chain's fields")
    name = "video" if "video" in fields else "image"
    shape = fields.get(name)
    if not isinstance(shape, list | tuple) or not all(isinstance(size, int) for size in shape):
        raise ValueError(f"a {run} run trains on images or clips of one size (dew.data.presets.images or "
                         f"clips); this chain's {name} is {shape!r}")
    return Field(name, tuple(shape))


def flag_selected(given: Sequence[str], selected: Mapping[str, str]) -> list[str]:
    """`given` command-line arguments with each optional setting of `selected`, a field's name and its
    subcommand's, turned on by its own flags: `--lora.rank 16` stands for `lora:lora --lora.rank 16`,
    where no `lora:` choice is given. A setting none of whose flags is given stays off."""
    given = list(given)
    for field, subcommand in selected.items():
        flags = [index for index, arg in enumerate(given) if arg.startswith(f"--{field}.")]
        if flags and not any(arg.startswith(f"{field}:") for arg in given):
            given.insert(flags[0], f"{field}:{subcommand}")
    return given


def _set_arguments(given: list[str]) -> tuple[list[str], list[str]]:
    """`given` without its `--set path=value` pairs, and the pairs' assignments, in order."""
    rest, settings, index = [], [], 0
    while index < len(given):
        if given[index] == "--set" and index + 1 < len(given):
            settings.append(given[index + 1])
            index += 2
        elif given[index].startswith("--set="):
            settings.append(given[index].removeprefix("--set="))
            index += 1
        else:
            rest.append(given[index])
            index += 1
    return rest, settings


def _chain_label(chain: Dataset | None) -> str:
    """A run name's word for its data: the last preset its recorded chain calls (`lm`, `images`), or
    `data` where it calls none or was not recorded."""
    encoded = None if chain is None else records.encode(chain)
    stages = encoded.get("stages") if isinstance(encoded, Mapping) else None
    stages = stages if isinstance(stages, list) else []
    calls = [stage["pipe"] for stage in stages if isinstance(stage, Mapping) and "pipe" in stage]
    return str(calls[-1]).rpartition(":")[2].rpartition(".")[2] if calls else "data"


_CHAINS = ("data", "validation")
"""A run's fields that hold chains, which its record writes through the data's codec (`_chain_record`)."""


def _chain_record(value: Dataset | Mapping[str, Dataset] | None) -> JSON:
    """The record of a chain field: a chain's record (`dew.data.records.encode`), `{"unrecorded":
    reason}` where the codec cannot describe it, `{"splits": [[name, record], ...]}` for named splits
    in their order (the first is the one a run ranks and calibrates on, which a sorted JSON object
    would not keep), or None."""
    if value is None:
        return None
    if isinstance(value, Mapping):
        return {"splits": [[name, _chain_record(split)] for name, split in value.items()]}
    encoded = records.encode(value)
    return {"unrecorded": encoded.reason} if isinstance(encoded, records.Unrecorded) else encoded


def _unrecorded(value: Dataset | Mapping[str, Dataset] | None) -> list[str]:
    """Why each chain of a chain field is not recorded (`_chain_record`), none where all are."""
    if value is None:
        return []
    if isinstance(value, Mapping):
        return [reason for split in value.values() for reason in _unrecorded(split)]
    encoded = records.encode(value)
    return [encoded.reason] if isinstance(encoded, records.Unrecorded) else []


def _read_chain(name: str, written: object) -> Dataset | dict[str, Dataset] | None:
    """The chain field `name` its record holds (`_chain_record`). Raises ValueError for a chain that was
    not recorded, naming why, and for a recipe's record, naming it: a run.json written before runs
    recorded chains trains again under the Dew release that wrote it."""
    if written is None:
        return None
    if not isinstance(written, Mapping):
        raise ValueError(f"run.json's {name} is {written!r}, not a chain's record")
    if "unrecorded" in written:
        raise ValueError(f"this run's {name} was built in Python and not recorded ({written['unrecorded']}); "
                         "train it again from the Python that built it")
    if "class" in written:
        raise ValueError(f"run.json's {name} is a {written['class']} recipe, which this Dew no longer reads: "
                         "train it under the Dew release that wrote it, or write its data as a chain "
                         "(`dew.data.presets`) and train a new run")
    if "version" in written:
        return records.decode(written)
    if "splits" not in written:
        raise ValueError(f"run.json's {name} is no chain's record and no named splits: {sorted(written)}")
    pairs = written["splits"]
    if not isinstance(pairs, list) or not all(isinstance(pair, list) and len(pair) == 2
                                              and isinstance(pair[0], str) for pair in pairs):
        raise ValueError(f"run.json's {name}.splits is {pairs!r}, not a list of [name, chain] pairs")
    splits = {}
    for split, value in pairs:
        chain = _read_chain(f"{name}.{split}", value)
        if not isinstance(chain, Dataset):
            raise ValueError(f"run.json's {name}.{split} is no chain's record")
        splits[split] = chain
    return splits


def _overridden(chains: Mapping[str, object], point: Mapping[str, JSON]) -> Mapping[str, object]:
    """A run's chain fields `chains` with the settings of their recorded chains in `point` set
    (`dew.data.records.override`): each path is `data.<call>.<setting>`, or
    `validation[.<split>].<call>.<setting>`. A chain's settings are set together on its record and it
    is read back once, and no run is built here, so the caller builds the run once with every field
    of the point in place."""
    targets: dict[tuple[str, str | None], dict[str, JSON]] = {}
    for path, value in point.items():
        name, _, rest = path.partition(".")
        held = chains[name]
        split = None
        if isinstance(held, Mapping):
            split, _, rest = rest.partition(".")
            if split not in held:
                raise KeyError(f"{path} names no split of the run's {name}")
        elif held is None:
            raise KeyError(f"{path} sets the run's {name}, which it does not hold")
        targets.setdefault((name, split), {})[rest] = value
    changed = dict(chains)
    for (name, split), settings in targets.items():
        held = changed[name]
        if isinstance(held, Mapping) and split is not None:
            changed[name] = {**held, split: _overridden_chain(f"{name}.{split}", held[split], settings)}
        elif isinstance(held, Dataset):
            changed[name] = _overridden_chain(name, held, settings)
    return changed


def _overridden_chain(name: str, chain: Dataset, settings: Mapping[str, JSON]) -> Dataset:
    encoded = records.encode(chain)
    if isinstance(encoded, records.Unrecorded):
        raise ValueError(f"{name}'s settings set a chain that was not recorded ({encoded.reason}); set it in "
                         "Python")
    for setting, value in settings.items():
        encoded = records.override(encoded, setting, value)
    return records.decode(encoded)


def _flag_json_or_text(text: str) -> JSON:
    """`text` as JSON, or the text itself where it is no JSON, as a chain's setting reads it."""
    try:
        return _flag_json(text)
    except json.JSONDecodeError:
        return text


def _within(node: dict[str, JSON], key: str) -> dict[str, JSON]:
    """The record `node` holds under `key`. A field declaring a base or a
    union records its value's class beside its fields, and this is the fields."""
    held = node[key]
    held = held["fields"] if isinstance(held, dict) and "class" in held else held
    if not isinstance(held, dict):
        raise KeyError(f"{key} holds {held!r}, not a record of fields")
    return held


def _placed(config: "RunConfig", path: str) -> tuple[list[str], Annotation]:
    """The keys of `path`'s value in `config`'s record, and the annotation it is read by."""
    *groups, field = path.split(".")
    held: object = config
    for group in groups:
        if not (dataclasses.is_dataclass(held) and group in _names(held)):
            raise KeyError(f"{path} names no group of the run record")
        held = getattr(held, group)
    if isinstance(held, ModelConfig | ObjectiveConfig) and field not in _names(held):
        member = models[held.name] if isinstance(held, ModelConfig) else objectives[held.name]
        # A flag's annotation types the value, as `--model.<field>` reads it;
        # an argument without a flag (the loss a `Supervised` is given) takes JSON.
        annotation = declared_type(_member_flags(member, held.fields)[0], field)
        if annotation is None and field not in parameters(member)[0]:
            raise KeyError(f"{path} names no argument of {held.name}")
        return [*groups, "fields", field], annotation
    if not (dataclasses.is_dataclass(held) and field in _names(held)):
        raise KeyError(f"{path} names no field of the run record")
    return [*groups, field], declared_type(type(held), field)


def _names(held: "DataclassInstance | type[DataclassInstance]") -> set[str]:
    """The fields a dataclass declares."""
    return {field.name for field in dataclasses.fields(held)}


def _parsed(path: str, text: str, annotation: Annotation) -> JSON:
    """`text` as the flag of `annotation` reads it (`RunConfig.assigned`), a
    tuple as the list its record holds."""
    if annotation is not None and _scalar(annotation):
        holder = dataclasses.make_dataclass("Set", [("value", tyro.conf.Positional[annotation])])
        given = shlex.split(text) if wants_tuple(annotation) else [text]
        return json.loads(json.dumps(tyro.cli(holder, args=given, prog=f"--set {path}").value))
    try:
        return _flag_json(text)
    except json.JSONDecodeError:
        return text


_RANDOM_SEARCH = RandomSearch()


@dataclasses.dataclass(frozen=True)
class RunConfig:
    """A whole run's configuration. A kind of run subclasses it with its own
    settings and builds what it trains from them (`prepare`); `run` trains it.
    """

    _FLAG_SELECTED: ClassVar[Mapping[str, str]] = {"lora": "lora"}
    """The optional settings `cli` turns on by their own flags, each with the
    subcommand its flags imply."""
    RESUMABLE: ClassVar[tuple[str, ...]] = ("trainer", "optim", "data", "validation", "data_seed")
    """The fields a run resumed under its name may change: how it trains and optimizes, what it reads,
    and, in a kind of run that adds them, how it evaluates and previews. Every other field says what
    it trains, which the resume must keep (`check_resume`), and so does `trainer.quantization`, which
    changes the forward pass the weights train through."""

    model: ModelConfig = dataclasses.field(default_factory=ModelConfig)
    data: Annotated[Dataset | None, tyro.conf.Suppress] = None
    """The data the run trains on: a chain ending in its preset (`dew.data.presets`), whose batch stage
    is the run's batch, read and loaded as it says. The record writes it as its chain
    (`dew.data.records`); on the command line, `--set data.<preset>.<setting>=` sets one setting of a
    recorded chain's call."""
    validation: Annotated[Dataset | Mapping[str, Dataset] | None, tyro.conf.Suppress] = None
    """The splits each evaluation scores, one dataset or several by name, each batched by its own batch
    stage or the training batch."""
    data_seed: int = 0
    """The seed of the data's order and of its random steps."""
    optim: OptimConfig = dataclasses.field(default_factory=OptimConfig)
    trainer: TrainerConfig = dataclasses.field(default_factory=TrainerConfig)
    objective: ObjectiveConfig | None = None
    """The objective's class and the arguments the run states, which `train`
    names from the objective it trains when the run leaves it unset."""
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

        A value of a subclass of the field's type is written as its class's
        import path and its fields. The data and its validation splits are
        their chains' records (`dew.data.records.encode`), and a chain the
        codec cannot describe is `{"unrecorded": reason}`.
        """
        return {**self._fields_record(), **{name: _chain_record(getattr(self, name)) for name in _CHAINS}}

    def unrecorded(self) -> list[str]:
        """Why each of the run's chains is not recorded, none where its record can rebuild them all:
        a run built in Python with a step the record cannot describe trains as built, and its
        run.json, which says why, refuses to rebuild it."""
        return [reason for name in _CHAINS for reason in _unrecorded(getattr(self, name))]

    def _fields_record(self) -> dict[str, JSON]:
        """The record of every field but the chains (`to_dict`), which an override of the others
        writes without encoding the chains."""
        with registry.recording():
            return {field.name: to_record(getattr(self, field.name), declared_type(type(self), field.name))
                    for field in dataclasses.fields(self) if field.name not in _CHAINS}

    @classmethod
    def from_dict(cls, values: Mapping[str, object]) -> Self:
        """Read back what `to_dict` wrote, for subclasses too. A field the
        record lacks takes its default; an unknown field raises. A chain
        recorded as unrecorded, or a recipe's record written before runs
        recorded chains, is refused by name."""
        rest = {name: value for name, value in values.items() if name not in _CHAINS}
        chains = {name: _read_chain(name, values[name]) for name in _CHAINS if name in values}
        return dataclasses.replace(from_record(cls, rest, dtypes=False), **chains)

    def override(self, point: Mapping[str, JSON]) -> Self:
        """This config with each dotted path in `point` replaced, through its record.

        A path walks the run's fields and the fields of the values they hold.
        Under `model` or `objective`, a name the config does not declare is an
        argument of the class it names, as `--model.num_layers` is on the
        command line. A path the run does not declare raises, rather than
        training the unchanged config. The paths are set together and the run
        is built once, so fields valid only together (a token budget with no
        row batch) are one point; each path is placed in this config, so a path
        under `model` names an argument of the model this config holds.
        """
        chained = {path: value for path, value in point.items() if path.partition(".")[0] in _CHAINS}
        record = self._fields_record()
        for path, value in point.items():
            if path in chained:
                continue
            *groups, field = _placed(self, path)[0]
            node = record
            for group in groups:
                node = _within(node, group)
            node[field] = value
        chains = _overridden({name: getattr(self, name) for name in _CHAINS}, chained)
        # Built once, every field of the point in place, so its checks read the run it describes.
        return dataclasses.replace(type(self).from_dict(record), **chains)

    def assigned(self, assignments: Sequence[str]) -> Self:
        """This config with each `path=value` in `assignments` set (`override`).

        The value is read as the field's annotation reads a flag: a number, a
        string, a literal or a tuple of them as its type parses it on the
        command line, None for an optional field, and anything else (a record,
        a list of records) as JSON, or as the text itself when it is not JSON.
        """
        point: dict[str, JSON] = {}
        for assignment in assignments:
            path, equals, text = assignment.partition("=")
            if not equals:
                raise ValueError(f"{assignment!r} sets no value; write path=value, as trainer.steps=2000")
            # A chain's setting is read against its call's signature (`dew.data.records.override`).
            point[path] = (_flag_json_or_text(text) if path.partition(".")[0] in _CHAINS
                           else _parsed(path, text, _placed(self, path)[1]))
        return self.override(point)

    def save(self, directory: str) -> str:
        """Write this config as `run.json` in `directory` and return the path.

        The file is the run's class record, its class's import path and its
        fields (`to_dict`), so `load` rebuilds the run as the class it is. The
        path goes through `epath`, the same filesystem layer Orbax writes the
        checkpoints with, so a `gs://` run directory gets the record too.
        """
        path = epath.Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        target = path / RUN_FILE
        # Whole or not at all: `load` reads it while a run is resumed.
        write_atomically(target, json.dumps(self.record(), indent=2, sort_keys=True))
        return str(target)

    def record(self) -> dict[str, JSON]:
        """The run's class record: its class by its alias in `dew.registry.runs`, or by its import path
        where the table holds none, and its fields (`to_dict`)."""
        return {"class": registry.runs.name_of(type(self)), "fields": self.to_dict()}

    def check_resume(self, saved: Mapping[str, object]) -> None:
        """Refuse to resume the run whose record is `saved` as this one where they train different
        things: another class, or a field outside `RESUMABLE` that differs. A changed `rope_theta`
        keeps every shape, so it would otherwise continue the old weights under new semantics. The
        saved fields are read back as this class reads them (`from_dict`), so a field the record
        predates takes its default."""
        named = registry.class_record(saved)
        if named is None or registry.runs[named[0]] is not type(self):
            raise ValueError(f"the run to resume is no {type(self).__name__}; resume it as the run it "
                             "records, or train under another name")
        recorded = type(self).from_dict({name: value for name, value in named[1].items()
                                         if name not in _CHAINS})
        before, now = recorded._fields_record(), self._fields_record()
        changed = sorted(name for name in now.keys() | before.keys()
                         if name not in self.RESUMABLE and now.get(name) != before.get(name))
        if recorded.trainer.quantization != self.trainer.quantization:
            changed.append("trainer.quantization")
        if changed:
            raise ValueError(f"the run to resume trained other {', '.join(changed)}; resume it as it "
                             "recorded them, or train under another name")

    @classmethod
    def cli(cls, args: Sequence[str] | None = None, *, default: Self | None = None) -> Self:
        """Parse a run from `args`, or from the process's own command line when `args` is None.

        An optional setting such as `lora` is turned on by its own flags. For
        example, `--lora.rank 16 --lora.modules q_proj` stands for
        `lora:lora --lora.rank 16 ...`, and with no `--lora.` flag the run
        trains without an adapter.

        `--model <alias or import path>` picks the model's class, and every
        field the class declares is a flag, `--model.<field>`, typed by its
        annotation (`ModelConfig`). Naming the class the run defaults to keeps
        the default's fields; naming another starts from that class's own and
        the default's compute dtype. `--objective` picks the objective's class
        the same way, each argument it takes a flag, `--objective.<argument>`
        (`ObjectiveConfig`). `--set path=value`, once a setting, sets a field at its dotted path
        after the flags, as `dew train` does (`assigned`): the data's settings are its chain's calls'
        (`--set data.lm.seq_len=512`), which have no flags of their own.
        """
        given, settings = _set_arguments(list(sys.argv[1:] if args is None else args))
        given = flag_selected(given, cls._FLAG_SELECTED)
        model, objective = cls._started(default, "model"), cls._started(default, "objective")
        if not isinstance(model, ModelConfig) or not isinstance(objective, ObjectiveConfig | None):
            raise TypeError(f"{cls.__name__} starts from the model {model!r} and the objective "
                            f"{objective!r}, not a ModelConfig and an ObjectiveConfig")
        name = _chosen(given, "model")
        if name is not None and registry.import_path(models[name]) != model.name:
            # Another class starts from its own fields and the run's
            # compute dtype, where it declares one.
            kept = {"dtype": model.fields["dtype"]} if "dtype" in model.fields and "dtype" in {
                field.name for field in dataclasses.fields(models[name])} else {}
            model = ModelConfig(name, kept)
        name = _chosen(given, "objective")
        if name is not None and (objective is None
                                 or registry.import_path(objectives[name]) != objective.name):
            objective = ObjectiveConfig(name)
        model_flags, model_started = _member_flags(models[model.name], model.arguments)
        if objective is None:
            objective_flags, objective_started = tyro.conf.Suppress[None], {}
            objective_default = dataclasses.field(default=None)
        else:
            objective_flags, objective_started = _member_flags(objectives[objective.name],
                                                               objective.arguments)
            objective_default = dataclasses.field(default_factory=objective_flags)
        # The run parses with its model and objective as those flags, and
        # validates once it holds the classes they name.
        parser = dataclasses.make_dataclass(
            cls.__name__, [("model", model_flags, dataclasses.field(default_factory=model_flags)),
                           ("objective", objective_flags, objective_default)],
            bases=(cls,), frozen=True, namespace={"__post_init__": lambda self: None})
        held = None if default is None else parser(
            **{field.name: getattr(default, field.name) for field in dataclasses.fields(cls)
               if field.name not in ("model", "objective")})
        parsed = tyro.cli(tyro.conf.CascadeSubcommandArgs[parser], args=given, default=held)
        if parsed is None:
            raise ValueError("the command line built no run config")
        built = cls(**{field.name: getattr(parsed, field.name) for field in dataclasses.fields(cls)
                       if field.name not in ("model", "objective")},
                    model=_given(model, parsed.model, model_started),
                    objective=None if objective is None else _given(objective, parsed.objective,
                                                                    objective_started))
        return built.assigned(settings) if settings else built

    @classmethod
    def _started(cls, default: Self | None, field: str) -> ModelConfig | ObjectiveConfig | None:
        """The value the command line starts `field` from: `default`'s, or the class's default."""
        declared = cls.__dataclass_fields__[field]
        value = (getattr(default, field) if default is not None else
                 declared.default_factory() if callable(declared.default_factory) else declared.default)
        if not isinstance(value, ModelConfig | ObjectiveConfig | None):
            raise TypeError(f"{cls.__name__}.{field} starts from {value!r}, not a ModelConfig or an "
                            "ObjectiveConfig")
        return value

    @classmethod
    def load(cls, directory: str, *, trust: Sequence[str] = ()) -> Self:
        """Read the run a directory's `run.json` records, or the file itself,
        as the class it names, which must be this one or derive from it.
        `trust` names the packages outside Dew the record may import."""
        path = epath.Path(directory)
        record = (path if str(directory).endswith(".json") else path / RUN_FILE).read_text()
        return cls.read(json.loads(record), trust=trust)

    @classmethod
    def read(cls, record: Mapping[str, object], *, trust: Sequence[str] = ()) -> Self:
        """The run a class record (`record`) names, as its class, which must be
        this one or derive from it."""
        if record.get("rebuildable") is False:
            raise ValueError("this run was defined in Python, so no record rebuilds it; it loads for "
                             "inference through dew.pipeline or a task's from_run")
        registry.import_trusted(record, trust)
        named = registry.class_record(record)
        if named is None:
            raise ValueError('a run record is its class and its fields, {"class": ..., "fields": ...}')
        run_class = registry.runs[named[0]]
        if not (isinstance(run_class, type) and issubclass(run_class, cls)):
            raise ValueError(f"the record names {named[0]}, which is no {cls.__name__}")
        return run_class.from_dict(named[1])

    def prepare(self) -> Prepared:
        """Build what this run trains: the objective its record names around
        its model, on its data. A kind of run with settings of its own builds
        from them in its own `prepare`."""
        if self.objective is None or self.lora is not None:
            raise ValueError(f"{type(self).__name__} builds the objective it names around its whole model; "
                             "name an objective, and train an adapter as a kind of run that attaches one")
        objective = self.objective.build(model=self.model.build())
        return Prepared(self, lambda name: self.train(objective, self.chain(), name=name,
                                                      validation=self.validation))

    def chain(self) -> Dataset:
        """The run's `data`. Raises ValueError for a run that names none: a run trains on the chain
        it states (`dew.data.presets`), in Python or in its run.json."""
        if self.data is None:
            raise ValueError(f"{type(self).__name__} names no data: set data= to a chain ending in its "
                             "preset (dew.data.presets), in Python or in run.json")
        return self.data

    def run(self) -> TrainState:
        """Prepare the process, build what this run trains (`prepare`) and train it.

        The run trains under `trainer.name`, or `<objective>-<data>/date-<time>`."""
        from dew.training import prepare_process, run_timestamp

        trainer = self.trainer
        prepare_process(trainer.wandb, trainer.multi_host, trainer.xla_flags, trainer.compilation_cache_dir,
                        layout=trainer.layout)
        prepared = self.prepare()
        stated = prepared.run.objective
        name = trainer.name or (f"{'run' if stated is None else objectives.label(stated.name)}-"
                                f"{_chain_label(prepared.run.data)}/date-{run_timestamp()}")
        return prepared.train(name)

    def recorded(self, objective: Objective[Loss, Effects]) -> Self:
        """Return this config as the record of the run that trains `objective`.

        The record names the class by its import path, so it says what was
        trained and loads in a process that imported nothing. Arguments the
        run states belong to the class it names, so a run that states them
        for another class than the one it trains is refused. Each argument
        the objective resolved from what it was built around
        (`Objective.resolved`) and the run leaves unstated is recorded as
        the objective holds it, so the record builds what the run sampled
        with.
        """
        trained = registry.import_path(type(objective))
        stated = ObjectiveConfig(trained) if self.objective is None else self.objective
        if stated.name != trained:
            if stated.fields:
                raise ValueError(f"the run states the arguments of {stated.name} and trains {trained}")
            stated = ObjectiveConfig(trained)
        missing = [name for name in objective.resolved if not hasattr(objective, name)]
        if missing:
            raise ValueError(f"{trained} resolves {missing} and holds no attribute of that name to record")
        resolved = registry.argument_records(type(objective), {
            name: getattr(objective, name) for name in objective.resolved if name not in stated.fields})
        return dataclasses.replace(self, objective=dataclasses.replace(
            stated, defaults={**stated.defaults, **resolved}))

    def train(self, objective: Objective[Loss, Effects], dataset: Dataset | Loader, *, name: str,
              metrics: Sequence[Metric] = (), rollout: Rollout | None = None,
              summary: Mapping[str, object] | None = None,
              validation: Dataset | Mapping[str, Dataset | Reader] | None = None,
              after: Callable[[TrainState, str], None] | None = None) -> TrainState:
        """Train `objective` on `dataset` as this config describes; every kind of run calls this after
        building both.

        The run lives under `name` in `trainer.checkpoint_dir`, and process zero writes
        the record there before training starts. A `trainer.wandb` opens a tracker
        under the same name, with the record, `summary` (the recipe's own view of the
        run) and the step count as its config. Only `trainer.publish` publishes the
        final checkpoint to the W&B registry, under the name with slashes and spaces
        replaced because an artifact name allows neither. Local tracking journals
        live in a separate tracking directory.

        `dataset` is compiled once with the objective's own stages (`dew.data.compile.fitted`), and its
        validation splits with it; the run's length, its ramp, its cadences and its recorded plan are
        read off that one compiled loader, which `fit` trains on. Its batch stage is the run's batch, and
        a token budget's accumulation is `trainer.update_tokens` over its token batch.

        A `trainer.quantization` wraps the module `objective` trains before anything
        initializes it, so the run learns through the quantized forward pass. A
        `lora` is not applied here: the recipe attaches it to the model and variables
        it builds the objective from. `rollout` is the trainer's rollout, which turns
        each prefetched batch into the one the step trains on, as an on-policy
        objective samples it. `validation` names the splits each evaluation
        scores in place of `dataset.val`, as `Trainer.fit` takes them: a held-out
        split and a test set are logged as `val/...` and `test/...`. With no
        `trainer.best`, the first split's loss ranks the checkpoints; a `Best`
        over a metric names its split. `after` runs on the state training ends
        on and the run's directory before the checkpoint is published, as a
        decision run fits its calibration there, so what it writes is published.
        """
        dataset, validation = fitted(dataset, validation, objective=objective.data(), seed=self.data_seed)
        self = dataclasses.replace(self, trainer=self.trainer.accumulating(dataset))
        if self.trainer.quantization is not None:
            quantize_trunk(objective, self.trainer.quantization)
        # The record names the run's directory, so its run.json trains the same run again.
        self = dataclasses.replace(self.recorded(objective),
                                   trainer=dataclasses.replace(self.trainer, name=name))
        trainer = self.trainer
        # Before the run length, since a ramp reads fewer records a step early
        # and a pass over the data is that many steps longer.
        dataset = dataset if trainer.batch_ramp is None else ramped(dataset, trainer.batch_ramp)
        if dataset.plan is not None:
            summary = {**(summary or {}), "data_plan": dataset.plan.record()}
        steps = trainer.total_steps(dataset)
        tracker = checkpoints = None
        try:
            wandb_tracker = None
            if trainer.wandb is not None and trainer.lower_for is None:
                # Shown under its own name where one is given, kept under the run's either way.
                wandb_tracker = WandbTracker(
                    trainer.wandb.project, name if trainer.wandb.name is None else trainer.wandb.name,
                    entity=trainer.wandb.entity, offline=trainer.wandb.offline)
            checkpoints = Checkpoints(os.path.join(trainer.checkpoint_dir, name), keep=trainer.keep)
            local = _local_tracker(checkpoints.directory, name)
            tracker = Trackers(local, *(() if wandb_tracker is None else (wandb_tracker,)))
            def check_resumed() -> None:
                """Refuse a checkpoint of another run under the name, on every rank."""
                saved = epath.Path(checkpoints.directory) / RUN_FILE
                if checkpoints.latest is not None and saved.exists():
                    self.check_resume(json.loads(saved.read_text()))

            def record_run() -> None:
                """Write the run's record and name where it is tracked, on rank zero."""
                if jax.process_index() == 0:
                    display = TrainingDisplay()
                    display.note(f"Experiment_Name: {name}")
                    display.note(f"Local tracking: {local.directory}")
                    self.save(checkpoints.directory)
                    tracker.artifact(RunRecord(name, json_value(self.to_dict()),
                        json_value(summary or {}), steps, packages_installed(),
                        device_kind=jax.devices()[0].device_kind, devices=jax.device_count(),
                        processes=jax.process_count()), 0)

            # The whole pool accepts the resume before its record is replaced or its state restored.
            agreed("run resume", check_resumed)
            agreed("run metadata", record_run)
            # The trainer holds mesh, microbatches, layout, accumulation, split,
            # dynamic_scale, profile and tune_kernels; prepare_process read the
            # process fields, and the rest are fit's arguments or built the
            # checkpoints and the tracker.
            state = Trainer(
                objective, self.optim, key=trainer.key, mesh=trainer.mesh,
                microbatches=trainer.microbatches, layout=trainer.layout, accumulation=trainer.accumulation,
                split=trainer.split, dynamic_scale=trainer.dynamic_scale, checkpoints=checkpoints,
                tracker=tracker, rollout=rollout, profile=trainer.profile, lower_for=trainer.lower_for,
                tune_kernels=trainer.tune_kernels,
            ).fit(
                dataset, steps=steps, tokens=trainer.tokens,
                log_every=trainer.log_every,
                eval_every=trainer.eval_interval(dataset),
                checkpoint_every=trainer.checkpoint_interval(dataset),
                checkpoint_final=trainer.checkpoint_final,
                metrics=metrics, preview=trainer.wandb is not None,
                best=trainer.best_policies(), validation=validation,
            )
            if after is not None:
                after(state, checkpoints.directory)
            def publish_checkpoint() -> None:
                """Upload the checkpoint the run ended on, where a tracker takes one."""
                if trainer.publish and wandb_tracker is not None:
                    # fit wrote the step it ended on, so that checkpoint is the run's.
                    dew.io.publish(checkpoints.path(int(state.step)), _artifact_name(name),
                                   tracker=wandb_tracker)

            agreed("checkpoint publishing", publish_checkpoint)
            return state

        finally:
            _closed(tracker, checkpoints, sys.exception())

    def sweep(self, space: Space, *, train: Callable[[Self], float], trials: int, ledger: str | Path,
              tracker: Tracker, search: Search = _RANDOM_SEARCH) -> list[TrialFinished]:
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
        finished = read_ledger(path, space)
        for index in range(len(finished), trials):
            point = search(space, finished)
            name = f"{self.trainer.name}/trial-{index}"
            value = train(self.override({**point, "trainer.name": name}))
            trial = TrialFinished(index, name, point, value)
            finished.append(trial)
            write_ledger(path, space, finished)
            tracker.log({"sweep/value": value}, index)
            tracker.artifact(trial, index)
        return finished


__all__ = ["MetricConfig", "ObjectiveConfig", "Prepared", "RunConfig", "TrainerConfig", "flag_selected"]
