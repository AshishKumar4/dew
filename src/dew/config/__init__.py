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
from dew import registry, training
from dew._coordination import agree_process_phase, agreed
from dew._records import JSON, duration, is_share, recorded_duration
from dew.cache import dew_cache_dir
from dew.checkpoints import RUN_FILE, Checkpoints, Keep, Resume
from dew.config.sweep import RandomSearch, Search, Space, read_ledger, write_ledger
from dew.data import records
from dew.data.compile import fitted
from dew.data.dataset import Loader, Ramp, Reader, ramped
from dew.data.pipeline import Dataset, ObjectiveData
from dew.distributed.placement import Layout, MeshSpec
from dew.files import write_atomically
from dew.lora import LoRA
from dew.nn.kernels import generation
from dew.nn.kernels.templates import attention_route
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
    record_argument,
    to_record,
    wants_tuple,
)
from dew.state import TrainState
from dew.telemetry.records import RunRecord, TrialFinished, json_value, packages_installed
from dew.training.ahead import AheadOfTime, AheadTarget, CompiledAhead, compile_only_gpus
from dew.training.budget import Pause, Plateau, budget_tokens, run_length, run_steps
from dew.training.cooldown import Cooldowns
from dew.training.display import TrainingDisplay
from dew.training.evaluation import EvalSuite, Summary
from dew.training.profiling import ProfileReport, ProfileWindow
from dew.training.quantization import quantize_trunk
from dew.training.rollout import Rollout
from dew.training.selection import Best
from dew.training.tracker import LocalTracker, Tracker, Trackers, Wandb, WandbTracker

if TYPE_CHECKING:
    from _typeshed import DataclassInstance

    from dew.interop import Pretrained
    from dew.training.trainer import Trainer


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
    did. `split` is the validation split it scores, every split where None.
    """

    name: str
    split: str | None = None

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


def _cadence_argument(*, timed: bool = True):
    """A cadence's command-line form: steps, `epoch`, a share of the run such as `5%`, None, and with
    `timed` a duration such as `30m`."""
    def read(given):
        value = given[0]
        if value == 'None':
            return None
        if value == 'epoch' or is_share(value):
            return value
        return int(value) if value.isdecimal() or not timed else duration(value)
    return tyro.constructors.PrimitiveConstructorSpec(
        nargs=1,
        metavar="STEPS|DURATION|SHARE%|epoch" if timed else "STEPS|SHARE%|epoch",
        instance_from_str=read,
        is_instance=lambda value: value is None or isinstance(value, (int, str, datetime.timedelta)),
        str_from_instance=lambda value: [
            recorded_duration(value) if isinstance(value, datetime.timedelta) else str(value)
        ],
    )


@dataclasses.dataclass(frozen=True)
class SourceOptions:
    """What a run builds of the source it starts from: `Pretrained.load`'s keywords for its parts,
    beside where it reads them (`TrainerConfig.source_placement`), every run kind's alike
    (`RunConfig.source_keywords`)."""

    text_only: bool = False
    """A multimodal wrapper's language model alone: no tower, projector or audio encoder built, nor
    their tensors read (`Pretrained.load`'s `text_only`)."""
    prediction_depths: int | None = None
    """The first that many multi-token prediction depths, 0 for none; None the checkpoint's own
    (`Pretrained.load`'s `prediction_depths`)."""
    attention: dict[str, Literal["causal", "block_bidirectional"]] | None = None
    """How each layer kind's attention reads, by kind (`full_attention`): causal, or
    block_bidirectional, each of a row's blocks read whole, which a run's rows then lie in
    (`Pretrained.load`'s `modeling_overrides`); None the checkpoint's own:
    `--source.attention full_attention block_bidirectional`."""

    def overrides(self) -> Mapping[str, object] | None:
        """The `modeling_overrides` these options state, None where they state none."""
        return None if self.attention is None else {"attention": dict(self.attention)}

    def keywords(self) -> "SourceOptionsKeywords":
        """`Pretrained.load`'s keywords for what a run builds of its source. With `weights=False` beside
        them a load is the run's base as the run builds it, its config and its variables' shapes, read
        from the source's config and safetensors headers alone, no weight downloaded (a release of
        PyTorch pickles alone, whose shapes no header states, is read whole): a check that a base
        loads under a run's options, on a disk too small for its weights."""
        return {"text_only": self.text_only, "prediction_depths": self.prediction_depths,
                "modeling_overrides": self.overrides()}


class SourcePlacement(TypedDict, total=False):
    """Where a run reads the source it starts from: `Pretrained.load`'s keywords
    (`TrainerConfig.source_placement`)."""

    weights: bool
    on_host: bool
    mesh: MeshSpec
    layout: Layout


class SourceOptionsKeywords(SourcePlacement, total=False):
    """`Pretrained.load`'s keywords for a run's source: what it builds of it (`SourceOptions`) and
    where it reads it (`SourcePlacement`, `RunConfig.source_keywords`)."""

    text_only: bool
    prediction_depths: int | None
    modeling_overrides: Mapping[str, object] | None


@dataclasses.dataclass(frozen=True)
class TrainerConfig:
    """Holds the run length, checkpointing, sharding and run tracking."""

    name: str | None = None
    checkpoint_dir: str = "./checkpoints"
    keep: Annotated[int | Keep, _keep_argument()] = Keep.latest
    """The number of latest checkpoints kept, besides the best one (`Keep`'s own default)."""
    resume: Resume = "auto"
    """Whether the run continues its directory's latest checkpoint: where there is one, "require" for a
    job that must not start afresh, or "never" for one that must not continue another (`Trainer.fit`)."""
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
    microbatch may cross the budget. With `steps`, `epochs` or `examples`, the
    first cap reached ends the run. A token-only run needs a constant learning
    rate or a schedule with its own explicit step horizon.
    """
    examples: int | None = None
    """The run length as the first this many source examples of the data, each read once with every
    row it makes: the records the chain makes in its own order, counted before they are packed and
    before an objective makes several rows of one (a decision layout's row a question)
    (`Trainer.fit`'s `examples`); `steps` or `epochs` set it instead."""
    stop: (Annotated[Pause, tyro.conf.subcommand("pause")]
           | Annotated[Plateau, tyro.conf.subcommand("plateau")] | None) = None
    """End the run at a fraction of its length, to resume it later as if it had never stopped
    (`Pause`): `--trainer.stop.fraction 0.5`; or once a metric, by its recorded name
    as `best` names one, stops improving (`Plateau`): `trainer.stop:plateau --trainer.stop.metric
    val/loss --trainer.stop.evals 3`."""
    restore_best: bool = False
    """End on the model the run ranked best (`Trainer.fit`'s `restore_best`): the state the run hands
    on (`RunConfig.train`'s `after`) holds the best step's weights, EMA and counters, with the final
    step's optimizer state. The checkpoints stay as written; `dew export --step best` exports the
    best one."""
    cooldowns: Annotated[Cooldowns, tyro.conf.subcommand("cooldowns")] | None = None
    """Branch the run into cooldowns, each scored by its validation and discarded (`Cooldowns`):
    `--trainer.cooldowns.at 20% 40% --trainer.cooldowns.length 20%`."""
    log_every: Annotated[int | str, _cadence_argument(timed=False)] = 100
    """The steps between logs, or a share of the run such as "1%" (`eval_every`)."""
    eval_every: Annotated[int | str | None, _cadence_argument(timed=False)] = "epoch"
    """The steps between validation passes.

    It is a number of steps, "epoch" for one pass over the data, a share of the
    run such as "5%", resolved once against its planned length in steps, epochs,
    examples or tokens (`dew.training.Trainer.fit`), or None to never validate.
    "epoch" over a stream that reports no record count raises a ValueError, since
    such a stream has no pass.
    """
    checkpoint_every: Annotated[int | str | datetime.timedelta | None, _cadence_argument()] = "epoch"
    """The steps between checkpoints, with the same choices as `eval_every`, or a duration such as 30m.

    A stream whose iterator cannot report its read position must train with None;
    the trainer refuses any other value for it.
    """
    checkpoint_final: bool = True
    """Write the final state when training ends, independently of `checkpoint_every`.

    False disables this save; cadenced, ranked and preemption saves keep their own policy.
    """
    local_directory: str | None = None
    """A path on every host's own disk where the run also writes checkpoints every `local_every`
    steps, under its name as in `checkpoint_dir`, cheaper than the persistent ones and lost with the
    host (`Checkpoints`); a resume reads the newer of the two. Set with `local_every`, or neither."""
    local_every: int | str | None = None
    """The steps between local checkpoints, or a share of the run such as "5%" (`local_directory`)."""
    scored_to: str | None = None
    """A directory that takes every validation batch's scoring artifacts at each evaluation, a decision
    run's decisions among them, as `<scored_to>/step-<step>/<split>-<batch>.npz`, for pooling a pass's
    metrics offline (`Trainer.fit`'s `scored_to`); None writes none."""
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
    attention_form: Literal["fast", "accurate"] = "fast"
    """The form the attention template computes in (`dew.nn.kernels.templates.attention_route`):
    "accurate" takes its accurate products, at their cost, for every attention the run traces, the
    step and its evaluations, and a resumed run traces in the form it records."""
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
    compile_ahead: Annotated[AheadTarget, tyro.conf.subcommand("ahead")] | None = None
    """Compile the run's first step for GPUs this host lacks, save it, then exit without training.

    `--trainer.compile-ahead.generation sm90 --trainer.compile-ahead.devices 8
    --trainer.compile-ahead.path step.dew-aot` builds the run as it would over eight compile-only
    H100s (`dew.training.ahead`), compiles its first step there with XLA's CUDA compiler, which needs
    the CUDA plugin and no GPU, and writes it with what it costs: each device's peak bytes and the
    step's FLOPs, read before booking the GPUs. The program exits with status 0, and writes no
    checkpoint or tracking.
    """
    ahead: str | None = None
    """Run the step `compile_ahead` saved at this path as the run's first step, in place of compiling it.

    The run lowers its first step as it would and runs the saved executable where that is the same
    program, on a host of the topology it was compiled for; another program, from another config,
    data shape, Dew or a plan recorded for another rung, is refused before the run trains.
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
        if self.attention_form not in attention_route.FORMS:
            raise ValueError(f"--trainer.attention-form is one of the template's forms "
                             f"{attention_route.FORMS}, not {self.attention_form!r}")
        run_length(self.steps, self.epochs, self.tokens, self.examples, required=False)
        if isinstance(self.log_every, str) and not is_share(self.log_every):
            raise ValueError(f"--trainer.log-every is steps or a share such as 1%, not {self.log_every!r}")
        if isinstance(self.eval_every, str) and self.eval_every != 'epoch' and not is_share(self.eval_every):
            raise ValueError(f"--trainer.eval-every is steps, epoch or a share such as 5%, not "
                             f"{self.eval_every!r}")
        if isinstance(self.checkpoint_every, str) and self.checkpoint_every != 'epoch' and not is_share(
                self.checkpoint_every):
            object.__setattr__(self, 'checkpoint_every', duration(self.checkpoint_every))
        if isinstance(self.keep, Mapping):
            object.__setattr__(self, 'keep', from_record(Keep, self.keep, dtypes=False))
        if isinstance(self.stop, Mapping) and "class" not in self.stop:
            # A record written while `stop` held a Pause alone names no class: its fields are a Pause's.
            object.__setattr__(self, 'stop', from_record(Pause, self.stop, dtypes=False))
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
        with self.checkpoints(self.name) as checkpoints:
            return checkpoints.latest is not None

    def checkpoints(self, name: str) -> Checkpoints:
        """The checkpoints of the run `name` in `checkpoint_dir`, keeping `keep`, with its local tier
        under `name` in `local_directory`, so runs sharing a local disk resume each only its own."""
        return Checkpoints(os.path.join(self.checkpoint_dir, name), keep=self.keep,
                           local_directory=None if self.local_directory is None else os.path.join(
                               self.local_directory, name), local_every=self.local_every)

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

    def resolved(self, dataset: Loader) -> tuple[Self, Loader, int | None]:
        """This config and `dataset` as a run trains on them, and the run's steps: the accumulation
        its token budget sets (`accumulating`), the dataset its batch ramp reads, and the step cap over
        that (`total_steps`), before which the ramp goes, since it reads fewer records a step early
        and a pass over the data is that many steps longer. Its updates are the steps over the
        accumulation."""
        trainer = self.accumulating(dataset)
        dataset = dataset if trainer.batch_ramp is None else ramped(dataset, trainer.batch_ramp)
        return trainer, dataset, trainer.total_steps(dataset)

    def total_steps(self, dataset: Loader) -> int | None:
        """Return the step cap from `steps`, `epochs` or `examples`, None for a run its tokens bound
        alone, its own or where its data's schedule ends (`budget_tokens`; `run_steps`, the one policy
        a direct fit applies too, which refuses a schedule to an objective counting no tokens)."""
        tokens = budget_tokens(dataset, self.tokens)
        return run_steps(dataset, self.steps, self.epochs, tokens, self.accumulation, self.examples)

    def total_updates(self, dataset: Loader) -> int | None:
        """Return the optimizer updates the run takes over `dataset` as `RunConfig.train` finalizes it,
        its token budget's accumulation and its batch ramp applied: what an average that moves once an
        update ramps over. None for a token-only run."""
        trainer, _, steps = self.resolved(dataset)
        # A window the run ends inside is never committed, as the trainer counts its updates.
        return None if steps is None else steps // trainer.accumulation

    def eval_interval(self, dataset: Loader) -> int | str | None:
        """Return the steps between validation passes over `data`, or None for never."""
        return self._interval(self.eval_every, dataset, "eval-every")

    def checkpoint_interval(self, dataset: Loader) -> int | str | datetime.timedelta | None:
        """Return the steps, or a recorded duration such as 30m, between checkpoints."""
        value = self.checkpoint_every
        if isinstance(value, datetime.timedelta):
            if value.total_seconds() <= 0:
                raise ValueError("checkpoint_every duration must be positive")
            return value
        return self._interval(self.checkpoint_every, dataset, "checkpoint-every")

    def _interval(self, value, dataset: Loader, field_name: str) -> int | str | None:
        """Steps for `epoch`; steps, a share of the run (which `fit` resolves) or None as given."""
        if value is None or isinstance(value, int) or is_share(value):
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

    _FLAG_SELECTED: ClassVar[Mapping[str, str]] = {"lora": "lora", "trainer.stop": "pause",
                                                   "trainer.cooldowns": "cooldowns",
                                                   "trainer.compile-ahead": "ahead"}
    """The optional settings `cli` turns on by their own flags, each with the
    subcommand its flags imply."""
    RESUMABLE: ClassVar[tuple[str, ...]] = ("trainer", "optim", "data", "validation", "data_seed",
                                            "val_metrics", "val_every", "val_summaries")
    """The fields a run resumed under its name may change: how it trains and optimizes, what it reads
    and scores, and, in a kind of run that adds them, how it previews. Every other field says what
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
    source: SourceOptions = dataclasses.field(default_factory=SourceOptions)
    """What the run builds of the source it starts from (`--source.text-only`,
    `--source.prediction-depths`), where it starts from one (`pretrained`)."""
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
    val_metrics: (Annotated[tuple[MetricConfig, ...], record_argument(tuple[MetricConfig, ...])]
                  | None) = None
    """The metrics each validation split scores, each with the arguments the run gives it (a CLIP
    metric's model, an FID's weights) and the split it scores, every split where it names none
    (`MetricConfig.split`), one JSON list on the command line: `--val-metrics '[{"name": "accuracy"},
    {"name": "brier", "split": "test"}]'` scores accuracy on every split and the Brier score on `test`.
    A split no metric names scores none; `[]` scores none anywhere. None scores the kind of run's own
    on every split (`default_metrics`): perplexity for a language model, a decision run's accuracy,
    calibration, risk-coverage and log loss, a JEPA run's probes, CLIP for an image run. A metric
    given is asked for whatever the data holds: a split the run does not validate, or data with no
    validation split, refuses it.
    """
    val_every: Annotated[Mapping[str, int | str], record_argument(Mapping[str, int | str])] = (
        dataclasses.field(default_factory=dict))
    """A split's own validation cadence, as `trainer.eval_every` spells one (`EvalSuite.every`), one
    JSON object on the command line: `--val-every '{"test": "25%"}'` scores `test` every quarter of the
    run and every other split at `trainer.eval_every`. A split it names must be one the run validates."""
    val_summaries: Annotated[tuple[Summary, ...], record_argument(tuple[Summary, ...])] = ()
    """Scalars weighed from each evaluation's split scores, a board's weighted index, each formed at a
    step that scored every split it weighs and logged as `evaluation/summary/<name>`, which
    `--trainer.best evaluation/summary/<name>` ranks by (`dew.training.evaluation.Summary`):
    `--val-summaries '[{"name": "board", "weights": {"val/accuracy": 1, "test/accuracy": 2},
    "better": "higher"}]'`.
    """

    def __post_init__(self) -> None:
        # A record carries every sequence as a JSON list and a command line writes one too; the field
        # is a tuple, so the value is one.
        if self.val_metrics is not None:
            object.__setattr__(self, "val_metrics", tuple(self.val_metrics))
            stray = [metric for metric in self.val_metrics if not isinstance(metric, MetricConfig)]
            if stray:
                raise TypeError(f"val_metrics holds metrics' records, {{'name': ..., 'fields': {{...}}}}, "
                                f"not {stray!r}")

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

    def source_keywords(self, *, frozen: bool) -> SourceOptionsKeywords:
        """`Pretrained.load`'s keywords for the source the run starts from: what it builds of it
        (`source`) and where it reads it (`TrainerConfig.source_placement`, `frozen` where the run
        keeps the source as it is). Every run kind that starts from a source loads it with these."""
        return {**self.source.keywords(), **self.trainer.source_placement(frozen=frozen)}

    def built_fresh(self) -> None:
        """Refuse `source` in a run whose model is built from `model`, loading no source: what it says
        `Pretrained.load` builds of a source (`source_keywords`) reaches nothing such a run builds."""
        if self.source != SourceOptions():
            raise ValueError(f"--source says what a run builds of the source it loads, and this "
                             f"{type(self).__name__} loads none: its model is built from --model, whose "
                             "own fields say it; or start the run from a source")

    def sourced_model(self) -> None:
        """Refuse a `model` name in a run whose source decides its architecture: such a run reads its
        model's fields' few choices alone, and a name other than its kind's own would be dropped."""
        (declared,) = (field for field in dataclasses.fields(self) if field.name == "model")
        default = (declared.default if declared.default_factory is dataclasses.MISSING
                   else declared.default_factory())
        if self.model.name != default.name:
            raise ValueError(f"--model names {self.model.name}, and the source this {type(self).__name__} "
                             "starts from decides its architecture; name none")

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
            reasons = record.get("reasons", [])
            stated = "".join(f"; {reason}" for reason in reasons) if isinstance(reasons, list) else ""
            raise ValueError("this run was defined in Python without a recorded RunConfig, so no "
                             f"record rebuilds it{stated}. It resumes through the Python that built it, and "
                             "loads for inference through dew.pipeline or a task's from_run; a run built as "
                             "a RunConfig in Python and trained with run() writes a run.json that rebuilds "
                             "it (docs/recipes.md, Python experiments)")
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
        self.built_fresh()
        objective = self.objective.build(model=self.model.build())
        scored = self.validation_metrics(validates=self.validation is not None)
        return Prepared(self, lambda name: self.train(objective, self.chain(), name=name, metrics=scored,
                                                      validation=self.validation))

    def default_metrics(self, *, validates: bool) -> tuple[MetricConfig, ...]:
        """The metrics a validation pass scores where `val_metrics` names none, by what the run's data
        `validates`: a kind of run's own, none for a run that names its objective alone."""
        del validates
        return ()

    def metric(self, config: MetricConfig) -> Metric:
        """`config` built with what this kind of run derives for it where it states none of its own."""
        return config.build()

    def validation_metrics(self, *, validates: bool) -> tuple[Metric, ...]:
        """The kind of run's own metrics (`default_metrics`), built (`metric`), by what the run's data
        `validates`: what each split scores where `val_metrics` names none (`suites`)."""
        return tuple(self.metric(config) for config in self.default_metrics(validates=validates))

    def suites(self, dataset: Loader, validation: Mapping[str, Reader] | None, metrics: Sequence[Metric] = (),
               ) -> Mapping[str, EvalSuite] | None:
        """Each split the run validates, `validation`'s or `dataset.val` as `val`, with the metrics
        `val_metrics` names for it built (`metric`), else the kind's own `metrics`, at its cadence in
        `val_every`, else the run's; None where neither names any, so each scores the kind's own at
        the run's. A metric or a cadence for a split the run does not validate is refused."""
        if self.val_metrics is None and not self.val_every:
            return None
        splits = dict(validation) if validation is not None else (
            {"val": dataset.val} if dataset.val is not None else {})
        named = {config.split for config in self.val_metrics or () if config.split is not None}
        if (self.val_metrics and not splits) or not named <= set(splits):
            raise ValueError(f"val_metrics score {sorted(named) or 'every split'}, and the run validates "
                             f"{sorted(splits) or 'none'}")
        if not set(self.val_every) <= set(splits):
            raise ValueError(f"val_every times {sorted(self.val_every)}, and the run validates "
                             f"{sorted(splits) or 'none'}")
        return {split: EvalSuite(reader, tuple(metrics) if self.val_metrics is None else tuple(
                    self.metric(config) for config in self.val_metrics if config.split in (None, split)),
                                 self.val_every.get(split))
                for split, reader in splits.items()}

    def chain(self) -> Dataset:
        """The run's `data`. Raises ValueError for a run that names none: a run trains on the chain
        it states (`dew.data.presets`), in Python or in its run.json."""
        if self.data is None:
            raise ValueError(f"{type(self).__name__} names no data: set data= to a chain ending in its "
                             "preset (dew.data.presets), in Python or in run.json")
        return self.data

    def compiled(self, objective: ObjectiveData, dataset: Dataset | Loader,
                 validation: Dataset | Mapping[str, Dataset | Reader] | None,
                 ) -> tuple[Loader, Mapping[str, Reader] | None]:
        """`dataset` and `validation` as the run trains on them (`fitted`): compiled with `objective`'s
        stages and `data_seed`, the training read bounded to `trainer.examples` where the run counts
        them, so its length is read off the loader this returns."""
        return fitted(dataset, validation, objective=objective, seed=self.data_seed,
                      examples=self.trainer.examples)

    def run(self) -> TrainState:
        """Prepare the process, build what this run trains (`prepare`) and train it.

        The run trains under `trainer.name`, or `<objective>-<data>/date-<time>`."""
        from dew.training import prepare_process, run_timestamp

        trainer = self.trainer
        prepare_process(trainer.wandb, trainer.multi_host, trainer.xla_flags, trainer.compilation_cache_dir)
        prepared = self.prepare()
        stated = prepared.run.objective
        name = trainer.name or (f"{'run' if stated is None else objectives.label(stated.name)}-"
                                f"{_chain_label(prepared.run.data)}/date-{run_timestamp()}")
        return prepared.train(name)

    def pinned_adapter(self) -> Self:
        """This run with its imported adapter's Hub source named at its commit (`LoRA.resolved`), so the
        record names the factors the run started from: every kind's `prepare` prepares this run, which
        loads the factors it records. Itself where nothing is left to pin."""
        lora = None if self.lora is None else self.lora.resolved()
        return self if lora == self.lora else dataclasses.replace(self, lora=lora)

    def profile(self, directory: str | None = None, *, warmup: int = 20, steps: int = 10,
                capture: ProfileWindow | None = None) -> ProfileReport | None:
        """Train this run afresh under `directory` for `warmup` steps, then a window of `steps` more traced
        (`ProfileWindow`), and report where the window's steps spent their time (`ProfileReport`), written
        as `report.json` beside XProf's own reports. The warmup is whole windows, at least one, so the step
        compiles before the window and the window is one log interval. The run keeps its own settings but
        those that would read or write outside the window, which a profile drops together: its length,
        stop, checkpoints local or persistent, the checkpoint ranking (`best`) and the best model it
        would end on, its cooldowns, the validation splits it names and their metrics, cadences and
        summaries (a split its chain holds out scores after the window), its W&B sink, and a lowering, a
        compile ahead or a step compiled ahead for its own length, which the profile trains past.
        `directory` unset is profiles/<the run's name>/<the time it starts>, read once the process is
        prepared, as `run` reads a run's name. The run's own directory is left as it is. `capture` gives the
        capture's options (its `host_tracer_level`, `python_tracer_level` and `activity_cap`; its
        directory and steps are the profile's). Raises ValueError for a directory that holds this run's
        profile already. The report is process zero's, which logs the run; None on the others."""
        from dew.training import prepare_process, run_timestamp

        name = self.trainer.name or "profile"
        warmup = max(1, -(-warmup // steps)) * steps
        trainer = dataclasses.replace(
            self.trainer, name=name, resume="never", steps=warmup + steps, epochs=None, tokens=None,
            examples=None, stop=None, best=None, restore_best=False, cooldowns=None, log_every=steps,
            eval_every=None, checkpoint_every=None, checkpoint_final=False, local_directory=None,
            local_every=None, wandb=None, publish=False, lower_for=None, compile_ahead=None, ahead=None)
        prepare_process(None, trainer.multi_host, trainer.xla_flags, trainer.compilation_cache_dir)
        # The report names its paths whole, wherever it is read from.
        directory = os.path.abspath(directory or os.path.join("profiles", name, run_timestamp()))
        if (Path(directory) / name / "tracking").exists():
            raise ValueError(f"{directory} holds a profile of {name} already; give the profile a directory "
                             "of its own")
        window = os.path.join(directory, "capture")
        captured = dataclasses.replace(capture or ProfileWindow(window, steps), directory=window, steps=steps,
                                       warmup=warmup)
        trainer = dataclasses.replace(trainer, checkpoint_dir=directory, profile=captured)
        profiled = dataclasses.replace(self, validation=None, val_metrics=None, val_every={},
                                       val_summaries=(), trainer=trainer)
        profiled.prepare().train(name)
        if jax.process_index() != 0:
            return None
        report = ProfileReport.read(Path(directory), name, Path(window), warmup=warmup, steps=steps)
        Path(directory, "report.json").write_text(json.dumps(dataclasses.asdict(report), indent=2) + "\n")
        return report

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

    def check_process(self) -> None:
        """Refuse to train in a process that does not carry the run's process options
        (`dew.training.runtime.unprepared`): `run()` and `dew train` set them up before JAX opens a
        backend (`prepare_process`), and a run trained after `prepare()` in a process nothing set up
        with them would drop them."""
        from dew.training.runtime import unprepared

        process = self.trainer
        missed = unprepared(process.multi_host, process.xla_flags, process.compilation_cache_dir)
        if missed:
            raise ValueError(f"trainer.{', trainer.'.join(missed)} set the process up before JAX opens a "
                             "backend, as `run()` and `dew train` do (`prepare_process`), and this process "
                             "carries none of it: train the run through `run()`, or call prepare_process "
                             "with them before anything calls JAX")

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
        split and a test set are logged as `val/...` and `test/...`. `metrics` are
        the kind's own, which each split scores where `val_metrics` names none. With no
        `trainer.best`, the first split's loss ranks the checkpoints; a `Best`
        over a metric names its split. `after` runs on the state training ends
        on and the run's directory before the checkpoint is published, as a
        decision run fits its calibration there, so what it writes is published.
        """
        self.check_process()
        dataset, validation = self.compiled(objective.data(), dataset, validation)
        # Each split scores the metrics `val_metrics` names for it, or else `metrics`, the kind's.
        suites = self.suites(dataset, validation, metrics)
        sized, dataset, steps = self.trainer.resolved(dataset)
        self = dataclasses.replace(self, trainer=sized)
        if self.trainer.quantization is not None:
            quantize_trunk(objective, self.trainer.quantization)
        # The record names the run's directory, so its run.json trains the same run again.
        self = dataclasses.replace(self.recorded(objective),
                                   trainer=dataclasses.replace(self.trainer, name=name))
        trainer = self.trainer
        if dataset.plan is not None:
            summary = {**(summary or {}), "data_plan": dataset.plan.record()}
        if trainer.compile_ahead is not None:
            raise self.compiled_ahead(objective, dataset, steps, rollout)
        tracker = checkpoints = None
        try:
            wandb_tracker = None
            if trainer.wandb is not None and trainer.lower_for is None:
                # Shown under its own name where one is given, kept under the run's either way.
                wandb_tracker = WandbTracker(
                    trainer.wandb.project, name if trainer.wandb.name is None else trainer.wandb.name,
                    entity=trainer.wandb.entity, offline=trainer.wandb.offline)
            checkpoints = trainer.checkpoints(name)
            local = _local_tracker(checkpoints.directory, name)
            tracker = Trackers(local, *(() if wandb_tracker is None else (wandb_tracker,)))
            def check_resumed() -> None:
                """Refuse a checkpoint of another run under the name, or a directory `resume` refuses
                (`Checkpoints.resumed`), on every rank, before the record is written over it."""
                saved = epath.Path(checkpoints.directory) / RUN_FILE
                if checkpoints.resumed(trainer.resume) is not None and saved.exists():
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
            # The trainer (`built`) holds mesh, microbatches, layout, accumulation, split, dynamic_scale,
            # profile, tune_kernels and any step compiled ahead; prepare_process read the process fields,
            # and the rest are fit's arguments or built the checkpoints and the tracker.
            ahead = None if trainer.ahead is None else AheadOfTime.load(trainer.ahead)
            # Every attention the run traces takes its recorded form.
            with generation.chosen(attention_form=trainer.attention_form):
                state = self.built(objective, trainer.mesh, rollout, checkpoints=checkpoints, tracker=tracker,
                                   profile=trainer.profile, lower_for=trainer.lower_for, ahead=ahead).fit(
                    dataset, steps=steps, tokens=trainer.tokens, log_every=trainer.log_every,
                    eval_every=trainer.eval_interval(dataset),
                    checkpoint_every=trainer.checkpoint_interval(dataset),
                    checkpoint_final=trainer.checkpoint_final, scored_to=trainer.scored_to,
                    metrics=metrics if suites is None else (), preview=trainer.wandb is not None,
                    best=trainer.best_policies(), stop=trainer.stop, restore_best=trainer.restore_best,
                    cooldowns=trainer.cooldowns, resume=trainer.resume,
                    # A run that names no metrics and validates no split has no suite to score.
                    validation=validation if suites is None else suites or None,
                    summaries=self.val_summaries)
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

    def compiled_ahead[LossT, EffectsT](self, objective: Objective[LossT, EffectsT], dataset: Loader,
                                         steps: int | None, rollout: Rollout | None) -> CompiledAhead:
        """The run's first step compiled for its `compile_ahead` target and saved
        (`AheadTarget.compiled`), its trainer built as `train` builds it, over that target's
        compile-only GPUs, with no checkpoints or tracker: what ends a run that compiles ahead."""
        target = self.trainer.compile_ahead
        assert target is not None
        mesh = self.trainer.mesh.build(compile_only_gpus(target.generation, target.devices))
        with generation.chosen(attention_form=self.trainer.attention_form):
            return target.compiled(self.built(objective, mesh, rollout), dataset, steps)

    def built[LossT, EffectsT](self, objective: Objective[LossT, EffectsT],
                               mesh: MeshSpec | jax.sharding.Mesh, rollout: Rollout | None,
                               **resources) -> "Trainer[LossT, EffectsT]":
        """The run's trainer over `mesh`: how it steps, every numeric choice its `TrainerConfig`
        holds (key, microbatches, layout, accumulation, split, loss scale, kernel tuning), and the
        `resources` its mode adds (checkpoints, tracker, profile, a lowering or a step compiled
        ahead), so a run that trains and one that compiles ahead build the one trainer. The trainer is
        the training package's lazy export (`dew.training`), read where a run trains or compiles
        ahead, so a config read alone reads none of it."""
        trainer = self.trainer
        return training.Trainer(
            objective, self.optim, key=trainer.key, mesh=mesh, microbatches=trainer.microbatches,
            layout=trainer.layout, accumulation=trainer.accumulation, split=trainer.split,
            dynamic_scale=trainer.dynamic_scale, rollout=rollout, tune_kernels=trainer.tune_kernels,
            **resources)

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


@dataclasses.dataclass(frozen=True)
class DecoderRunConfig(RunConfig):
    """A run over a loaded decoder, which may run a block of its layers in passes: the pass count, the
    block, the passes trained through and their recomputation (`CausalTransformer.with_passes`), each
    the run's choice over the checkpoint's own. The language-model and the decision runs take it."""

    loop_steps: int | None = None
    """The passes a looped checkpoint runs its block (`CausalTransformer.loop`), such
    as Ouro's at 1, 2 or 4; None keeps the checkpoint's own."""
    loop_layers: tuple[int, int] | None = None
    """The block of layers [first, end) that `loop_steps` repeats: a pretrained
    decoder with no loop gains one over its own layers, as Qwen 3.5's middle block
    repeated twice; None keeps a looped checkpoint's block."""
    loop_backprop_steps: int | None = None
    """Train through the last k passes of the loop alone, with earlier passes
    cut from backpropagation; None keeps the loop's own choice."""
    loop_remat: bool = False
    """Checkpoint every pass of the loop, including the first, and recompute
    it in the backward pass; requires a loop and leaves decoding unchanged.
    False keeps a checkpoint's own choice."""

    @property
    def loops(self) -> bool:
        """Whether the run asks anything of a loop (`looped`)."""
        return (self.loop_steps is not None or self.loop_layers is not None
                or self.loop_backprop_steps is not None or self.loop_remat)

    def looped(self, start: "Pretrained") -> "Pretrained":
        """`start` with its block run as the run asks (`looping`), as loaded where it asks nothing."""
        if not self.loops:
            return start
        return looping(start, self.loop_steps, self.loop_layers, backprop_steps=self.loop_backprop_steps,
                       remat=self.loop_remat or None)


def looping(start: "Pretrained", steps: int | None, layers: tuple[int, int] | None, *,
            backprop_steps: int | None = None, remat: bool | None = None) -> "Pretrained":
    """A loaded checkpoint whose block runs `steps` passes: a looped one's own block, or `layers`,
    which gives a decoder with no loop one over its own layers. Refused where neither names a block,
    or a new block's passes are not stated. `backprop_steps` and `remat` choose
    the loop's backward memory; None keeps a looped checkpoint's own value."""
    from dew.nn.protocols import Looping

    looped = (start.model.with_passes(steps, layers, backprop_steps=backprop_steps, remat=remat)
              if isinstance(start.model, Looping) else None)
    if looped is None:
        raise ValueError(f"this {type(start.model).__name__} has no loop: --loop-layers names the block of "
                         "its layers to repeat and --loop-steps how many times")
    return dataclasses.replace(start, model=looped)


__all__ = ["DecoderRunConfig", "MetricConfig", "ObjectiveConfig", "Prepared", "RunConfig", "SourceOptions",
           "TrainerConfig",
           "flag_selected", "looping"]
