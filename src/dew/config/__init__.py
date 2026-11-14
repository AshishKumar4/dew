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

import contextlib
import copy
import dataclasses
import functools
import json
import os
import shlex
import sys
import types
import typing
from collections.abc import Callable, Iterator, Mapping, Sequence
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, ClassVar, Literal, Self, Union

import jax
import tyro
from etils import epath
from tyro.conf._confstruct import _SubcommandConfig

import dew.io
from dew import _records, registry, training
from dew._cleanup import attempted
from dew._coordination import agree_process_phase, agreed, agreed_same
from dew._files import write_atomically
from dew._records import JSON
from dew.checkpoints import RUN_FILE, Checkpoints
from dew.config.earlier import earlier_option_writing, earlier_run
from dew.config.release import Release
from dew.config.sweep import RandomSearch, Search, Space, read_ledger, write_ledger
from dew.config.trainer import (
    SourceOptions as SourceOptions,
    SourceOptionsKeywords,
    TrainerConfig as TrainerConfig,
    _artifact_name,
    local_records,
)
from dew.data import records
from dew.data.compile import fitted
from dew.data.dataset import Closeable, Loader, Reader
from dew.data.pipeline import Dataset, ObjectiveData
from dew.data.plan import PLAN_VERSIONS, SCHEDULE, PlanOrderChanged, read_by
from dew.distributed.placement import MeshSpec
from dew.interop.weights import AUTO
from dew.lora import LoRA
from dew.nn.kernels import generation
from dew.objectives.base import Effects, Loss, Metric, Objective
from dew.optim import OptimConfig, ParamGroup
from dew.registry import (
    Annotation,
    Configured,
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
    trusted_packages,
    wants_tuple,
)
from dew.state import TrainState
from dew.telemetry.profile import CaptureOptions
from dew.telemetry.records import RunRecord, TrialFinished, packages_installed, tracker_json
from dew.training._fit import FitArguments
from dew.training.ahead import AheadOfTime, CompiledAhead, LaterPhases, fit_plan
from dew.training.budget import Pause
from dew.training.display import TrainingDisplay
from dew.training.evaluation import EvalSuite, Split, Summary
from dew.training.profiling import ProfileReport, ProfileWindow
from dew.training.quantization import quantize_trunk
from dew.training.rollout import Rollout
from dew.training.tracker import LocalTracker, Tracker, Trackers, WandbTracker

if TYPE_CHECKING:
    from _typeshed import DataclassInstance

    from dew.interop import Pretrained
    from dew.nn.attention import AttentionImpl
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


def _member_flags(member: type | Callable[..., Configured],
                  given: Mapping[str, object]) -> tuple[type, Mapping[str, object]]:
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


def recorded_param_groups(saved: epath.Path) -> tuple[ParamGroup, ...] | None:
    """The param groups the run whose record is `saved` trained in, read alone: the rest of the record,
    a data chain only its Python holds among it, may not rebuild here. () for a record written before
    runs took groups; None where the record holds no optimizer, a run defined in Python without one."""
    fields = json.loads(saved.read_text()).get("fields")
    optim = fields.get("optim") if isinstance(fields, Mapping) else None
    if not isinstance(optim, Mapping):
        return None
    held = optim.get("fields", optim)
    groups = held.get("param_groups", []) if isinstance(held, Mapping) else []
    return from_record(tuple[ParamGroup, ...], groups)


def partitioned(groups: Sequence[ParamGroup]) -> tuple[tuple[str, tuple[str, ...], bool], ...]:
    """What of `groups` shapes a run's optimizer state: each group's name and patterns, in order (a
    parameter belongs to the first group it matches), and whether it is bounded, its clamp a state of its
    own (`dew.optim.BoundsState`); a bounded group's interval is not, which a resume sets anew
    (`dew.optim.with_bounds`)."""
    return tuple((group.name, group.patterns, group.bounds is not None) for group in groups)


def trained_modules(objective: Objective[Loss, Effects],
                    modules: Sequence[str]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Those of `modules`, paths under `params`, that hold a parameter `objective` trains, as its tree is
    drawn (`Objective.init`, its shapes alone), and those that are one (a parameter a loaded module
    gained, `Pretrained.with_model`): a tied readout is drawn with none."""
    if not modules:
        return (), ()
    from dew.optim import parameter_name

    trained = objective.drawn_shapes()["params"]
    names = [parameter_name(path) for path, _ in jax.tree_util.tree_leaves_with_path(trained)]
    return (tuple(module for module in modules if any(name.startswith(f"{module}/") for name in names)),
            tuple(module for module in modules if module in names))


def _local_tracker(directory: str, name: str):
    """Return the journal of a run, kept where its local records are (`local_records`)."""
    return LocalTracker(local_records(directory, name, "tracking"))


def _closed(tracker, checkpoints: Checkpoints | None, primary: BaseException | None,
            rollout: Closeable | None = None) -> None:
    """Close `rollout` (what it runs beside the trainer: a served policy's engine, a reward pool), then
    `tracker` on every rank and agree on the outcome, then `checkpoints`, each even where one before it
    failed.

    `primary` is the exception the run is already unwinding with, if any. A
    close that fails is noted on it rather than replacing it, since the run's
    own failure is the one a caller wants; without one, the first failure is
    raised with the others noted on it.
    """
    closes = [*([] if rollout is None else [("closing the rollout", rollout.close)]),
              *([] if tracker is None else
                [("closing the tracker", functools.partial(tracker.__exit__, type(primary), primary, None))])]
    error = None
    try:
        attempted(closes)
    except BaseException as failure:
        error = failure
    failed = None
    try:
        agree_process_phase(error, phase="run close")
    except BaseException as failure:
        if primary is None:
            failed = failure
        else:
            primary.add_note(f"Closing the run failed: {failure!r}")
            if error is not None and failure is error:
                # This process's own: the closes it came from, as `attempted` noted each with its step.
                for note in error.__notes__:
                    primary.add_note(note)
    raised = primary if primary is not None else failed
    if checkpoints is not None:
        checkpoints.__exit__(None if raised is None else type(raised), raised, None)
    if failed is not None:
        raise failed


@dataclasses.dataclass(frozen=True)
class Prepared[LossT, EffectsT]:
    """What a run trains, as its class builds it (`RunConfig.prepare`).

    `run` is the run as it records itself, with what building resolved (a
    pinned source, the model's fields as built); `objective` and `dataset`
    what it trains on, and the rest what `RunConfig.train` takes beside them.
    `train` trains it under a name, and `inspected` views one sampled update
    of it there (`RunConfig.viewed`).
    """

    run: "RunConfig"
    objective: Objective[LossT, EffectsT]
    dataset: Dataset | Loader
    metrics: Sequence[Metric] = ()
    rollout: Rollout | None = None
    summary: Mapping[str, object] | None = None
    validation: Dataset | Mapping[str, Dataset | Split] | None = None
    after: Callable[[TrainState, str], None] | None = None

    def train(self, name: str) -> TrainState:
        return self.run.train(self.objective, self.dataset, name=name, metrics=self.metrics,
                              rollout=self.rollout, summary=self.summary, validation=self.validation,
                              after=self.after)

    def inspected(self, name: str) -> dict[str, float]:
        return self.run.viewed(self.objective, self.dataset, name=name, rollout=self.rollout,
                               validation=self.validation)


_SUBCOMMAND = _SubcommandConfig
"""The marker a field's annotation declares a subcommand with (`tyro.conf.subcommand`, which states it as
an object of this class)."""


def flag_subcommands(config: type, prefix: str = "") -> dict[str, str]:
    """The optional settings `config` declares that `cli` turns on by their own flags (`flag_selected`),
    by flag path, each with the subcommand its flags imply: a field annotated as one or more subcommands or
    None (`tyro.conf.subcommand`), its first; and those within each such setting's settings and within a
    nested run setting (a `dew.config` dataclass, the trainer's), parents before their own."""
    found: dict[str, str] = {}
    hints = typing.get_type_hints(config, include_extras=True)
    for field in dataclasses.fields(config):
        path, hint = f"{prefix}{field.name.replace('_', '-')}", hints[field.name]
        options = typing.get_args(hint) if typing.get_origin(hint) in (Union, types.UnionType) else (hint,)
        given = [option for option in options if option is not type(None)]
        marked = [(typing.get_args(option)[0], marker.name) for option in given
                  if typing.get_origin(option) is Annotated
                  for marker in typing.get_args(option)[1:]
                  if isinstance(marker, _SUBCOMMAND) and marker.name is not None]
        if marked and len(marked) == len(given) < len(options):
            found[path] = marked[0][1]
            nested = [kind for kind, _ in marked]
        else:
            nested = given if len(given) == 1 else []
        for kind in nested:
            if (isinstance(kind, type) and dataclasses.is_dataclass(kind)
                    and kind.__module__.startswith("dew.config")):
                found.update(flag_subcommands(kind, f"{path}."))
    return found


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


def takes_dtype(model: ModelConfig) -> bool:
    """Whether `model`'s class takes a compute `dtype`, as a Flax module does."""
    from dew.registry import imported, member_fields, models

    owner = models[model.name] if model.name in models.paths else imported(model.name)
    return isinstance(owner, type) and any(name == "dtype" for name, _, _ in member_fields(owner))


def _learns_alike(before: ObjectiveConfig | None, now: ObjectiveConfig | None) -> bool:
    """Whether an objective `now` trains what `before` trained: one class, every argument the same but
    those its class lets a resume change (`dew.objectives.base.Objective.resumable`), a stated argument
    the same as the default it states again."""
    if before is None or now is None or before.name != now.name:
        return False
    resumable = objectives[now.name].resumable

    def kept(config: ObjectiveConfig) -> dict:
        return {name: value for name, value in {**config.defaults, **config.fields}.items()
                if name not in resumable}

    return kept(before) == kept(now)


@contextlib.contextmanager
def _routed(saved: Mapping[str, object], unseeded: str) -> Iterator[None]:
    """Within the block, a refusal of a resumed run's changed chain (`PlanOrderChanged`) names the record
    `saved` of the run it resumes as a route where that record rebuilds its chains (`_reads_back`), as a
    run.json written of a chain the codec could not describe does not (`{"unrecorded": reason}`), and
    the schedule's route as the run's kind spells it (`unseeded`, `RunConfig.UNSEEDED`)."""
    try:
        yield
    except PlanOrderChanged as refused:
        named = registry.class_record(saved)
        recorded = named is not None and _reads_back(named[1])
        raise refused.routed(recorded=recorded, unseeded=unseeded) from None


CHAINS = ("data", "validation")
"""A run's fields that hold chains, which its record writes through the data's codec (`_chain_record`)."""


def _reads_back(fields: Mapping[str, object]) -> bool:
    """Whether a saved run's record `fields` holds its data and rebuilds every chain it records, as
    `RunConfig.load` reads them (`_read_chain`): the record route of a refusal (`RECORD`) names such a
    record alone, where an unrecorded chain, a recipe's record or an unrecorded split would refuse it
    again (row 3913)."""
    if fields.get("data") is None:
        return False
    try:
        for name in CHAINS:
            _read_chain(name, fields.get(name))
    except ValueError:
        return False
    return True


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


def _holds(config: "RunConfig", path: str) -> bool:
    """Whether `config` holds every record along the dotted `path` up to its field: false past an optional
    record it holds none of (a teacher of None)."""
    held: object = config
    for group in path.split(".")[:-1]:
        held = getattr(held, group, None)
        if held is None:
            return False
    return True


def _recorded_at(config: "RunConfig", path: str) -> JSON:
    """The value `config`'s record holds at the dotted `path` (`_placed`), as `override` writes it."""
    *groups, field = _placed(config, path)[0]
    node = config._fields_record()
    for group in groups:
        node = _within(node, group)
    return node.get(field)


def _placed(config: "RunConfig", path: str) -> tuple[list[str], Annotation]:
    """The keys of `path`'s value in `config`'s record, and the annotation it is read by."""
    *groups, field = path.split(".")
    held: object = config
    for group in groups:
        if not (dataclasses.is_dataclass(held) and group in _names(held)):
            raise KeyError(f"{path} names no group of the run record")
        held = getattr(held, group)
    if isinstance(held, Registered) and field not in _names(held):
        member = held.member
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
    settings and builds what it trains from them (`build_run`, which `prepare` calls
    on the run with its Hub sources pinned); `run` trains it.
    """

    RESUMABLE: ClassVar[tuple[str, ...]] = ("trainer", "optim", "data", "validation", "data_seed",
                                            "val_metrics", "val_every", "val_summaries", "kernels")
    """The fields a run resumed under its name may change: how it trains and optimizes, what it reads
    and scores, the kernels it runs (`kernels`, each on whatever hardware the resume lands on), and, in a
    kind of run that adds them, how it previews. Every other field says what
    it trains, which the resume must keep (`check_resume`), and so does `trainer.quantization`, which
    changes the forward pass the weights train through."""

    EXECUTED: ClassVar[tuple[str, ...]] = ("scan_layers", "loop_remat")
    """The model and source fields, by name, that say how a model's layers execute and not what they
    compute or store: a decoder's runs under one scan or a layer at a time hold the same layers on disk
    (`layers_N`) and compute the same step, so a run resumes under either (`check_resume`), as it does
    with a looped block's passes recomputed in its backward or held (`loop_remat`), memory alone. A source's
    experts packed or decoded (`SourceOptions.packed_experts`) are not: packed, the experts' banks are no
    trainable parameters and hold no optimizer moments, so a resume across them is refused by name."""

    BUILT: ClassVar[tuple[str, ...]] = ("source.param_dtype", "source.step", "data_plan_version")
    """The fields, by their dotted paths, a run may leave None for its build or its pinning to resolve,
    whose resolution a later Dew, or a later state of what it reads, may change: the storage of the
    source it reads (`SourceOptions.param_dtype`), the step of a saved Dew run it starts from, which
    that run's later checkpoints move (`with_pinned_run`), the data plan version its order is read by
    (`data_plan_version`), and in a kind of run that adds them its own
    (a decision run's option text). A run that leaves one None and resumes a saved run under its name
    takes the saved run's value (`resumed`), so running the same command again resumes what it trained;
    one it sets is its own, and a resume refuses it where it differs (`check_resume`). A saved run whose
    record holds one None left it to its build unresolved, which nothing it saved resolves (each earlier
    shape's translation resolves what it can, `earlier_run`), so the value this build resolves is its
    value."""

    UNSEEDED: ClassVar[str] = SCHEDULE
    """How a run of this kind reads a mix saved unseeded and seeded now, as the refusal of its resume names
    it (`PlanOrderChanged.routed`): the data layer's `mix(..., seed=None)`; a decision run's mixture's
    order (`dew.decision.data.SCHEDULE_MIXTURE`)."""

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
    kernels: dict[str, str] = dataclasses.field(default_factory=dict)
    """The kernel the run runs for each op it names (`dew.kernels.use`, ops `dew.kernels.choices`), for
    every program it traces, its steps', evaluations' and samples' (`traced`): `--kernels attention xla`.
    None named, each op's pick for its call (row 4649)."""
    data_plan_version: int | None = None
    """The data plan version the run's positions are saved under (`dew.data.plan.PLAN_VERSIONS`), which
    `resumed` resolves on every path a run trains by and its record holds (`BUILT`): this Dew's, where the
    run resumes none or one of a version it reads; a version it reads none of is refused resumed exactly
    and started again relaxed. A saved run that recorded none read by version 8, and one that recorded a
    `data_plan` of None by 9 (`earlier_data_plan`)."""
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
        return {**self._fields_record(), **{name: _chain_record(getattr(self, name)) for name in CHAINS}}

    def unrecorded(self) -> list[str]:
        """Why each of the run's chains is not recorded, none where its record can rebuild them all:
        a run built in Python with a step the record cannot describe trains as built, and its
        run.json, which says why, refuses to rebuild it."""
        return [reason for name in CHAINS for reason in _unrecorded(getattr(self, name))]

    def _fields_record(self) -> dict[str, JSON]:
        """The record of every field but the chains (`to_dict`), which an override of the others
        writes without encoding the chains."""
        with registry.recording():
            return {field.name: to_record(getattr(self, field.name), declared_type(type(self), field.name))
                    for field in dataclasses.fields(self) if field.name not in CHAINS}

    @classmethod
    def from_dict(cls, values: Mapping[str, object]) -> Self:
        """Read back what `to_dict` wrote, for subclasses too. A field the
        record lacks takes its default; an unknown field raises. A chain
        recorded as unrecorded, or a recipe's record written before runs
        recorded chains, is refused by name. A model recorded with an earlier
        main's fields reads as its checkpoint's record does."""
        from dew.inference.tasks import earlier_model

        rest = {name: value for name, value in values.items() if name not in CHAINS}
        trainer = rest.get("trainer")
        fields = trainer.get("fields", trainer) if isinstance(trainer, Mapping) else {}
        stop = fields.get("stop")
        if isinstance(trainer, Mapping) and isinstance(stop, Mapping) and "class" not in stop:
            # A record written while `stop` held a Pause alone names no class: its fields are a Pause's.
            # Carried until 1.0 (ledger 1552).
            fields = {**fields, "stop": {"class": f"{Pause.__module__}:Pause", "fields": stop}}
            rest = {**rest, "trainer": {**trainer, "fields": fields} if "class" in trainer else fields}
        # A model recorded before its components' fields changed reads as today's, as its checkpoint's
        # record does (`earlier_model`, ledger 2423).
        if isinstance(rest.get("model"), Mapping):
            rest["model"] = earlier_model({"model": rest["model"]})["model"]
        chains = {name: _read_chain(name, values[name]) for name in CHAINS if name in values}
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
        chained = {path: value for path, value in point.items() if path.partition(".")[0] in CHAINS}
        record = self._fields_record()
        for path, value in point.items():
            if path in chained:
                continue
            *groups, field = _placed(self, path)[0]
            node = record
            for group in groups:
                node = _within(node, group)
            node[field] = value
        chains = _overridden({name: getattr(self, name) for name in CHAINS}, chained)
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
            point[path] = (_flag_json_or_text(text) if path.partition(".")[0] in CHAINS
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
        (`source`), where it reads it (`TrainerConfig.source_placement`, `frozen` where the run
        keeps the source as it is) and what it stores its parameters in. Every run kind that starts
        from a source loads it with these.

        A source the run trains is stored in its numerics' `dtypes.params`, the master weights its
        updates land in. One it keeps frozen is stored as its checkpoint stores it
        (`param_dtype='auto'`), since no update reaches it: a float32 copy of a bf16 checkpoint doubles
        its memory and is cast back to bf16 every pass for nothing, while a cast to the compute dtype
        would round the float32 tensors a bf16 checkpoint keeps (Mamba's `A_log`). A `dtypes.params`
        narrower than float32 stores it so too. A storage the run's source options state wins
        (`SourceOptions.param_dtype`)."""
        params = self.trainer.precision.dtypes.params
        stored = (self.source.param_dtype if self.source.param_dtype is not None
                  else AUTO if frozen and jax.numpy.dtype(params) == jax.numpy.float32 else params)
        return {**self.source.keywords(), **self.trainer.source_placement(frozen=frozen),
                "param_dtype": stored}

    def source_attention(self) -> "AttentionImpl":
        """The attention implementation a run that starts from a source loads it with: its model's
        `attention_impl` field, auto unless set, refused unless it names one (`AttentionImpl`). Every run
        kind that loads a source reads it here."""
        from dew.nn.attention import attention_implementation

        return attention_implementation(str(self.model.fields.get("attention_impl", "auto")))

    def built_fresh(self) -> None:
        """Refuse `source` in a run whose model is built from `model`, loading no source: what it says
        `Pretrained.load` builds of a source (`source_keywords`), how it scans its layers among it, reaches
        nothing such a run builds, whose own fields say it (`--model.scan_layers`)."""
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
        predates takes its default, and one it left to its build unresolved takes this build's."""
        recorded = self._saved_as(saved)
        if recorded is None:
            raise ValueError(f"the run to resume is no {type(self).__name__}; resume it as the run it "
                             "records, or train under another name")
        before, now = (_records.without_fields(record, self.EXECUTED)
                       for record in (recorded._fields_record(), self._fields_record()))
        # A field the saved run left to its build and its record holds unresolved, which no checkpoint
        # states either (`earlier_option_writing`), is the one this build resolves (`BUILT`).
        unresolved = {path for path in self.BUILT if "." not in path and before.get(path) is None}
        changed = sorted(name for name in now.keys() | before.keys()
                         if name not in self.RESUMABLE and name not in unresolved
                         and now.get(name) != before.get(name))
        if "objective" in changed and _learns_alike(recorded.objective, self.objective):
            changed.remove("objective")
        # A relaxed resume reads by this Dew's data plan where the saved run's is none it reads, its data
        # started again (`resumed`), and any resume of a version it reads reads it as its own (`read_by`):
        # its state resumes as it trained.
        if "data_plan_version" in changed and (self.trainer.data_resume == "relaxed" or (
                recorded.data_plan_version in PLAN_VERSIONS and self.data_plan_version in PLAN_VERSIONS)):
            changed.remove("data_plan_version")
        if recorded.trainer.quantization != self.trainer.quantization:
            changed.append("trainer.quantization")
        if recorded.trainer.precision != self.trainer.precision:
            changed.append("trainer.precision")
        if changed:
            raise ValueError(f"the run to resume trained other {', '.join(changed)}; resume it as it "
                             "recorded them, or train under another name")

    def _saved_as(self, saved: Mapping[str, object]) -> Self | None:
        """The saved run whose record is `saved` read as this class reads it (`from_dict`), a field the
        record predates at its default and its chains apart; None where it records another class."""
        named = registry.class_record(saved)
        if named is None or registry.runs[named[0]] is not type(self):
            return None
        return type(self).from_dict({name: value for name, value in named[1].items() if name not in CHAINS})

    @staticmethod
    def _saved_record(directory: str, checkpoints: Checkpoints) -> Mapping[str, object] | None:
        """The record of the saved run in its checkpoints' `directory`, read as every earlier shape of a
        saved run is (`earlier_run`); None where none was saved."""
        saved = epath.Path(directory) / RUN_FILE
        if not saved.exists():
            return None
        return earlier_option_writing(earlier_run(json.loads(saved.read_text())), checkpoints)

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
        given = flag_selected(given, flag_subcommands(cls))
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
        named_file = str(directory).endswith(".json")
        record = json.loads((path if named_file else path / RUN_FILE).read_text())
        # The packages trusted first, so the record's earlier shapes are read with its classes imported.
        registry.import_trusted(record, trust)
        # A field its record left to its checkpoints (`earlier_option_writing`) is read where the run resumes
        # (`resumed`, `check_resumed`), its process prepared: `dew train run.json` reads the run first, before
        # a pool of processes is joined, which no device opened before may be (train 589's review).
        return cls.read(earlier_run(record), trust=trust)

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
        """Build what this run trains, as `build_run` builds it from the run with the fields it leaves to
        its build at the run it resumes (`resumed`) and its Hub sources pinned (`pinned`), so the record
        names what it started from and is what it loads."""
        return self.resumed().pinned().build_run()

    def resumed(self, paths: Sequence[str] | None = None) -> Self:
        """This run with each field it leaves None for its build to resolve (`BUILT`, or those of them
        `paths` names) at the value the saved run it resumes under its name recorded, read as every
        earlier shape of a saved run is (`earlier_run`), and its data plan version this Dew's
        (`PLAN_VERSIONS`), where it resumes no run that recorded one, one that recorded a version this Dew
        reads, which it reads as its own (`dew.data.plan.read_by`), or, relaxed
        (`TrainerConfig.data_resume`), any: the one place a run's version is resolved,
        on each path a run trains by (`prepare`, and `readied`, which a run trained in Python reaches
        alone), before its data compiles or its record is written; and, resolving every field (no
        `paths`), as `prepare` does before it builds them, each class it names, its model's and its
        objective's, with every argument it leaves unstated as the saved run recorded it
        (`_defaulted_as`), so a default changed since resumes the run as it trained (row 2938); an
        objective built already is compared as built (`check_resume`). Its other fields as they are where
        it resumes no run (`TrainerConfig.resumes`)."""
        trainer = self.trainer
        built = self.BUILT if paths is None else paths
        unset = [path for path in built if _holds(self, path) and _recorded_at(self, path) is None]
        run = self
        # The classes' defaults are resolved where the run builds its objective and model from them
        # (`prepare`, every field), never where they are built already (`readied`'s data plan version).
        defaulting = paths is None
        if ((unset or defaulting) and trainer.name is not None and trainer.resume != "never"
                and trainer.resumes()):
            with trainer.checkpoints(trainer.name) as checkpoints:
                saved = self._saved_record(trainer.directory(trainer.name), checkpoints)
            recorded = None if saved is None else self._saved_as(saved)
            # A field of a record the saved run held none of (a teacher it had not) takes nothing.
            if recorded is not None:
                run = self.override({path: _recorded_at(recorded, path) for path in unset
                                     if _holds(recorded, path)})
                run = run._defaulted_as(recorded) if defaulting else run
        # A relaxed resume of a run saved under a version this Dew reads none of reads by this Dew's, its
        # data started again (`dew.training.Trainer._data_resumed`); an exact one is refused by it.
        # A version this Dew reads it reads as its own (`read_by`), so the run's record states the version its
        # positions are saved under.
        if "data_plan_version" in built and (run.data_plan_version in (None, *PLAN_VERSIONS) or (
                trainer.data_resume == "relaxed")):
            run = dataclasses.replace(run, data_plan_version=PLAN_VERSIONS[-1])
        return run

    def _defaulted_as(self, recorded: Self) -> Self:
        """This run with each class it names (`Registered`: its model, its objective) that `recorded`, the
        saved run it resumes, names too, defaulting every argument it leaves unstated as `recorded` did:
        the defaults a run records are the ones it trained under (`Registered.defaults`), and an argument
        the class gained since keeps the default it takes now."""
        changed = {}
        for field in dataclasses.fields(self):
            mine, theirs = getattr(self, field.name), getattr(recorded, field.name, None)
            if isinstance(mine, Registered) and isinstance(theirs, Registered) and mine.name == theirs.name:
                kept = {name: value for name, value in theirs.defaults.items() if name not in mine.fields}
                changed[field.name] = dataclasses.replace(mine, defaults={**mine.defaults, **kept})
        return dataclasses.replace(self, **changed) if changed else self

    def build_run(self) -> Prepared:
        """Build what this run trains, the run as it stands, its sources already pinned (`prepare`
        pins them first): the objective its record names around its model, on its data. A kind of
        run with settings of its own builds from them in its own `build_run`."""
        if self.objective is None or self.lora is not None:
            raise ValueError(f"{type(self).__name__} builds the objective it names around its whole model; "
                             "name an objective, and train an adapter as a kind of run that attaches one")
        self.built_fresh()
        objective = self.objective.build(model=self.model.build(**self.numerics_fields()))
        scored = self.validation_metrics(validates=self.validation is not None)
        return Prepared(self, objective, self.chain(), metrics=scored, validation=self.validation)

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

    def suites(self, dataset: Loader, validation: Mapping[str, Split] | None, metrics: Sequence[Metric] = (),
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

    def compiled[Entry: Split](self, objective: ObjectiveData, dataset: Dataset | Loader,
                               validation: Dataset | Mapping[str, Dataset | Entry] | None,
                               ) -> tuple[Loader, Mapping[str, Reader | Entry] | None]:
        """`dataset` and `validation` as the run trains on them (`fitted`): compiled with `objective`'s
        stages, `data_seed` and `data_plan_version`, the training read bounded to `trainer.examples` where
        the run counts them, or to `trainer.epochs` passes over its source examples where its read has no
        record count, so its length is read off the loader this returns; its loading workers sized for
        the run's length, its `trainer.steps`, `epochs` or `tokens` (`dew.data.Loading.resolved`)."""
        # At the version `resumed` resolves (itself where `prepare` or `readied` resolved it; a run built
        # alone, `build_run`, resolves it here), which it always states.
        planned = self.resumed(("data_plan_version",)).data_plan_version
        assert planned is not None
        # A run of a version this Dew reads none of, resumed exactly, is refused before its data compiles.
        read_by(planned)
        trainer = self.trainer
        return fitted(dataset, validation, objective=objective, seed=self.data_seed,
                      resume=trainer.data_resume,
                      examples=trainer.examples, epochs=trainer.epochs, steps=trainer.steps,
                      tokens=trainer.tokens)

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

    def pinned(self) -> Self:
        """This run with its Hub sources named at the commits they resolve to now, as its record names
        them and `prepare` loads them: here an imported adapter's (`LoRA.resolved`); a kind of run pins
        its own sources beside it. Itself where nothing is left to pin."""
        lora = None if self.lora is None else self.lora.resolved()
        return self if lora == self.lora else dataclasses.replace(self, lora=lora)

    def with_pinned_run(self, pretrained: str | None) -> Self:
        """This run with the checkpoint it reads of the saved Dew run it starts from (`pretrained`, a
        run's directory) pinned to the step its selection resolves to now (`SourceOptions.step`:
        `best`, `best:<metric>` or its task's default), so its record names the run and the step it
        read and a resume reads them again. Itself where it starts from no saved run, or names a step."""
        from dew.inference.tasks import SavedRun
        from dew.interop.saved_sources import is_saved_run

        if pretrained is None or not is_saved_run(pretrained) or isinstance(self.source.step, int):
            return self
        step = SavedRun.pinned(pretrained, self.source.step).step
        return dataclasses.replace(self, source=dataclasses.replace(self.source, step=step))

    def profile(self, directory: str | None = None, *, warmup: int = 20, steps: int = 10,
                capture: CaptureOptions | None = None) -> ProfileReport | None:
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
        prepared, as `run` reads a run's name. The run's own directory is left as it is. `capture` is what
        the capture records (`CaptureOptions`), its own by default. Raises ValueError for a directory that
        holds this run's profile already. The report is process zero's, which logs the run; None on the
        others."""
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
        captured = ProfileWindow(window, steps, warmup, capture or CaptureOptions())
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

    def readied(self, objective: Objective[Loss, Effects], dataset: Dataset | Loader,
                validation: Dataset | Mapping[str, Dataset | Split] | None, metrics: Sequence[Metric],
                name: str) -> tuple[Self, Loader, Mapping[str, Split] | None, Mapping[str, EvalSuite] | None,
                                    int | None]:
        """This run as it trains `objective` under `name`, and what it trains on, before anything is
        placed (`train`, `viewed`): its data compiled with the objective's own stages and its validation
        splits with it, the suites they score (`suites`), sized and ramped (`TrainerConfig.resolved`), with
        its steps; its trunk quantized where it asks; its record naming the run's directory and its
        optimizer's groups (`grouped`)."""
        self.check_process()
        # Its data plan version resolved before its data compiles and its record is written, a run trained
        # in Python reaching `readied` alone, as the run it trains under `name` reads it (train 589's
        # review); a loader compiled already reads as every version this Dew reads does
        # (`dew.data.plan.read_by`).
        named = dataclasses.replace(self, trainer=dataclasses.replace(self.trainer, name=name))
        planned = named.resumed(("data_plan_version",))
        dataset, held = planned.compiled(objective.data(), dataset, validation)
        # Each split scores the metrics `val_metrics` names for it, or else `metrics`, the kind's.
        suites = planned.suites(dataset, held, metrics)
        sized, dataset, steps = planned.trainer.resolved(dataset)
        run = dataclasses.replace(planned, trainer=sized)
        if run.trainer.quantization is not None:
            quantize_trunk(objective, run.trainer.quantization)
        # The record names the run's directory, so its run.json trains the same run again.
        run = dataclasses.replace(run.recorded(objective),
                                  trainer=dataclasses.replace(run.trainer, name=name))
        return dataclasses.replace(run, optim=run.grouped(objective, name)), dataset, held, suites, steps

    def train(self, objective: Objective[Loss, Effects], dataset: Dataset | Loader, *, name: str,
              metrics: Sequence[Metric] = (), rollout: Rollout | None = None,
              summary: Mapping[str, object] | None = None,
              validation: Dataset | Mapping[str, Dataset | Split] | None = None,
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
        self, dataset, validation, suites, steps = self.readied(objective, dataset, validation, metrics,
                                                                name)
        trainer = self.trainer
        if dataset.plan is not None:
            summary = {**(summary or {}), "data_plan": dataset.plan.record()}
        # What `fit` is asked for but its resume, which a run compiled ahead compiles its evaluations from.
        fitted = FitArguments(
            # A run bounded in examples that counts no steps ahead keeps its clock (`Trainer.fit`).
            steps=steps, tokens=trainer.tokens, log_every=trainer.log_every,
            examples=dataset.examples if steps is None else None,
            eval_every=trainer.eval_interval(dataset), checkpoint_every=trainer.checkpoint_interval(dataset),
            checkpoint_final=trainer.checkpoint_final, scored_to=trainer.scored_to,
            metrics=metrics if suites is None else (), preview=trainer.wandb is not None,
            best=trainer.best_policies(), stop=trainer.stop, restore_best=trainer.restore_best,
            cooldowns=trainer.cooldowns,
            # A run that names no metrics and validates no split has no suite to score.
            validation=validation if suites is None else suites or None, summaries=self.val_summaries)
        if trainer.compile_ahead is not None:
            raise self.compiled_ahead(objective, dataset, steps, rollout, name, fitted)
        tracker = checkpoints = None
        try:
            wandb_tracker = None
            if trainer.wandb is not None and trainer.lower_for is None:
                # Shown under its own name where one is given, kept under the run's either way.
                wandb_tracker = WandbTracker(
                    trainer.wandb.project, name if trainer.wandb.name is None else trainer.wandb.name,
                    entity=trainer.wandb.entity, offline=trainer.wandb.offline, shared=trainer.wandb.shared)
            checkpoints = trainer.checkpoints(name)
            local = _local_tracker(checkpoints.directory, name)
            tracker = Trackers(local, *(() if wandb_tracker is None else (wandb_tracker,)))
            # The trainer (`built`) holds mesh, microbatches, layout, accumulation, split, dynamic_scale,
            # profile, tune_kernels and any step compiled ahead; prepare_process read the process fields,
            # and the rest are fit's arguments or built the checkpoints and the tracker.
            ahead = None if trainer.ahead is None else AheadOfTime.load(trainer.ahead)
            built = self.built(objective, trainer.mesh, rollout, checkpoints=checkpoints, tracker=tracker,
                               profile=trainer.window(checkpoints.directory, name),
                               lower_for=trainer.lower_for, ahead=ahead)

            def record_run() -> None:
                """Write the run's record and name where it is tracked, on rank zero."""
                if jax.process_index() == 0:
                    display = TrainingDisplay()
                    display.note(f"Experiment_Name: {name}")
                    display.note(f"Local tracking: {local.directory}")
                    self.save(checkpoints.directory)
                    tracker.artifact(RunRecord(name, tracker_json(self.to_dict()),
                        tracker_json(summary or {}), steps, packages_installed(),
                        device_kind=jax.devices()[0].device_kind, devices=jax.device_count(),
                        processes=jax.process_count()), 0)

            # The whole pool accepts the resume, its data among it, before its record is replaced or its
            # state restored.
            self._agree_resume(checkpoints, built, dataset, steps)
            agreed("run metadata", record_run)
            # Its later phases' steps compile ahead in a run of its record (`LaterPhases`), where one
            # rebuilds it.
            if not self.unrecorded():
                built.later_phases = LaterPhases(self, trusted_packages())
            with self.traced():
                state = built.fit(dataset, resume=trainer.resume, **fitted)
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
            _closed(tracker, checkpoints, sys.exception(),
                    rollout if isinstance(rollout, Closeable) else None)

    @contextlib.contextmanager
    def traced(self) -> Iterator[None]:
        """The block every program the run traces is traced within (its steps', evaluations' and samples'):
        its recorded precision policy (`generation.run_precision`) and its kernel choices
        (`dew.kernels.use`), which JAX keys its trace caches on."""
        import dew.kernels

        with generation.run_precision(self.trainer.precision), dew.kernels.use(**self.kernels):
            yield

    def compiled_ahead[LossT, EffectsT](self, objective: Objective[LossT, EffectsT], dataset: Loader,
                                         steps: int | None, rollout: Rollout | None, name: str,
                                         fitted: FitArguments | None = None) -> CompiledAhead:
        """The run's first step compiled for its `compile_ahead` target and saved
        (`AheadTarget.compiled`), its trainer built as `train` builds it, its links the run's own, over
        that target's compile-only devices (`AheadTarget.compile_only`), with no tracker: what ends a run
        that compiles ahead, with its evaluations' programs where `fitted` is what `train` hands `fit`.
        Its checkpointer is the run `name`'s own, as its fit is planned with one (a `Pause` its
        checkpoints resume, a validation loss that ranks them): a directory that holds checkpoints is
        read, the step its resume takes compiled, and one that holds none is asked and not created
        (`Checkpoints.latest`)."""
        target = self.trainer.compile_ahead
        assert target is not None
        mesh = self.trainer.mesh.build(target.compile_only())
        trainer = self.built(objective, mesh, rollout, checkpoints=self.trainer.checkpoints(name))
        with self.traced():
            plan = None if fitted is None else fit_plan(trainer, dataset, fitted)
            # The run's bound in examples as its data was compiled to it (`viewed`).
            return target.compiled(trainer, dataset, steps, dataset.examples if steps is None else None,
                                   self.trainer.resume, plan)

    def viewed(self, objective: Objective[Loss, Effects], dataset: Dataset | Loader, *, name: str,
               rollout: Rollout | None = None,
               validation: Dataset | Mapping[str, Dataset | Split] | None = None) -> dict[str, float]:
        """The run's inspection of one sampled update over the batch its first step takes
        (`dew.training.inspected`, `first_step_batch`), the run readied as `train` readies it
        (`readied`) and its trainer built as `train` builds it, from the checkpoints its directory `name`
        holds, their latest, else its start, with no tracker: what `inspected` returns."""
        from dew.training import inspected
        from dew.training.ahead import first_step_batch

        run, loader, _, _, steps = self.readied(objective, dataset, validation, (), name)
        held = epath.Path(run.trainer.directory(name)).is_dir()
        trainer = run.built(objective, run.trainer.mesh, rollout,
                            checkpoints=run.trainer.checkpoints(name) if held else None)
        with run.traced():
            # The run's bound in examples as its data was compiled to it: its `examples`, or its `epochs`'
            # passes over a read with no record count (`fitted`).
            batch = first_step_batch(trainer, loader, steps, loader.examples if steps is None else None)
            return inspected(trainer, batch, steps=steps)

    def inspected(self) -> dict[str, float]:
        """What the run's inspection (`TrainerConfig.inspection`) logs of one sampled update over its training
        data's first batch, from its checkpoints' latest, else its start, offline (`viewed`): the view its
        training logs, for a saved run, which this trains and writes nothing of (`dew inspect`)."""
        from dew.training import prepare_process

        trainer = self.trainer
        if trainer.inspection is None:
            raise ValueError("the run names no inspection to view; name one, as --set trainer.inspection=...")
        prepare_process(None, trainer.multi_host, trainer.xla_flags, trainer.compilation_cache_dir)
        return self.prepare().inspected(trainer.name or "inspected")

    def grouped(self, objective: Objective[Loss, Effects], name: str) -> OptimConfig:
        """The run's optimizer as it trains under `name`: the groups that claim what it draws fresh beside a
        loaded base, each matching those that hold a parameter (`ParamGroup.fresh`, `trained_modules`): its
        own, or its named ones with the fresh group beside them, or none named, the default groups
        (`OptimConfig.fresh_groups`). A resumed run that names no groups trains in those its record names,
        as its optimizer state holds them (none for a record written before runs took them); one that
        names some trains in its own, their rates and a bounded group's interval as it states them,
        refused where their names, patterns or which of them are bounded are not the record's, which
        shape the optimizer state it resumes (`partitioned`). Every process
        agrees on the groups before any records them."""
        def resolved() -> JSON:
            named = self.optim.param_groups
            checkpoints = self.trainer.checkpoints(name)
            saved = epath.Path(checkpoints.directory) / RUN_FILE
            recorded = recorded_param_groups(saved) if checkpoints.resumed(
                self.trainer.resume) is not None and saved.exists() else None
            if recorded is not None and not named:
                return to_record(recorded, tuple[ParamGroup, ...])
            if recorded is not None and partitioned(named) == partitioned(recorded):
                # The record's own groups named again (a saved run's config), fresh claims and all.
                return to_record(named, tuple[ParamGroup, ...])
            modules, leaves = trained_modules(objective, objective.fresh_modules())
            # A group named fresh claims what the run draws, or is refused where it draws none (`with_fresh`).
            groups = self.optim.fresh_groups() if modules or leaves else named
            claimed = dataclasses.replace(self.optim, param_groups=groups).with_fresh(modules, leaves)
            if recorded is not None and partitioned(claimed.param_groups) != partitioned(recorded):
                raise ValueError(f"run {name!r} resumes an optimizer state partitioned by the groups "
                                 f"{partitioned(recorded)} (names, patterns and whether each is bounded), "
                                 f"and names {partitioned(claimed.param_groups)}: a group's bounds added or "
                                 "dropped change the state it holds; change only a group's rate, decay, "
                                 "schedule or a bounded group's interval on a resume, or start a new run")
            return to_record(claimed.param_groups, tuple[ParamGroup, ...])
        groups = from_record(tuple[ParamGroup, ...], agreed_same("run param groups", resolved))
        return self.optim if groups == self.optim.param_groups else dataclasses.replace(
            self.optim, param_groups=groups)

    def _agree_resume[LossT, EffectsT](self, checkpoints: Checkpoints,
                                       built: "Trainer[LossT, EffectsT]", dataset: Loader,
                                       steps: int | None) -> None:
        """Refuse, on every rank and before the run's record is written over what it resumes, a checkpoint
        of another run under the name or a directory `resume` refuses (`Checkpoints.resumed`,
        `check_resume`); then, in an agreement of its own, so every rank enters the position's restore
        only once all accept the record, a saved data position the run's data does not read on from
        (`resume_admitted`), a changed chain's refusal naming the record it resumes where that holds its
        chain (`_routed`)."""
        resume = self.trainer.resume
        # The record of the run this one resumes, as each rank read it once the pool agreed it.
        resumed: dict[str, Mapping[str, object]] = {}

        def check_resumed() -> None:
            if checkpoints.resumed(resume) is not None and (
                    record := self._saved_record(checkpoints.directory, checkpoints)) is not None:
                self.check_resume(record)
                resumed["record"] = record

        def check_data() -> None:
            from dew.training.startup import resume_admitted

            if "record" in resumed:
                with _routed(resumed["record"], self.UNSEEDED):
                    resume_admitted(built, dataset, resume)

        agreed("run resume", check_resumed)
        agreed("run data", check_data)

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
            layout=trainer.layout, accumulation=trainer.accumulation,
            split=trainer.split,
            dynamic_scale=(self.compute_dtype() == "float16" if trainer.dynamic_scale is None
                           else trainer.dynamic_scale), rollout=rollout,
            tune_kernels=trainer.tune_kernels, param_dtype=self.drawn_dtype(), inspection=trainer.inspection,
            diloco=trainer.diloco, data_resume=trainer.data_resume, **{"links": trainer.links, **resources})

    def compute_dtype(self) -> str:
        """The dtype the run's model computes in: the one its record states, else its numerics'
        (`TrainerConfig.precision`)."""
        stated = self.model.fields.get("dtype")
        return str(stated) if stated is not None else self.trainer.precision.dtypes.compute

    def drawn_dtype(self) -> str | None:
        """The dtype the run stores the parameters it draws fresh in (`Trainer`'s `param_dtype`): its
        numerics' `dtypes.params` where that is not Flax's own float32 and its model's record states
        none; None leaves each module's own."""
        params = self.trainer.precision.dtypes.params
        return None if "param_dtype" in self.model.fields or jax.numpy.dtype(params) == jax.numpy.float32 \
            else params

    def numerics_fields(self) -> dict[str, str]:
        """The fields the run's numerics derive for its model (`ModelConfig.build`): its compute dtype,
        where the model takes one and its record states none."""
        if "dtype" in self.model.fields or not takes_dtype(self.model):
            return {}
        return {"dtype": self.trainer.precision.dtypes.compute}

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

    loop_steps: int | tuple[int, ...] | None = None
    """The passes a looped checkpoint runs its block (`CausalTransformer.loop`), such
    as Ouro's at 1, 2 or 4; or one count a phase of the run's data (`dew.data.phases`,
    `Trainer`'s `phase_passes`), a pass count that grows over the run, each phase's step
    compiled with its own and the run handing on the last; None keeps the checkpoint's own."""
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
    loop_implicit: bool = False
    """Relax each later pass toward the block's input by a per-channel θ drawn
    at zero (`Loop.implicit`), so the looped decoder starts as the one loaded
    computes; False keeps a checkpoint's own choice."""
    loop_exit_gate: bool = False
    """Weigh each pass's exit by a gate drawn at zeros over its states (`Loop.exit_gate`), which the
    objective's `exit_entropy` trains as Ouro does, at `EXIT_ENTROPY` where the run states none
    (`exit_settings`); False keeps a checkpoint's own choice."""
    loop_exit_threshold: float | None = None
    """Exit each token at the first pass whose cumulative exit probability reaches this
    (`Loop.exit_threshold`); None keeps a checkpoint's own choice."""

    def __post_init__(self) -> None:
        super().__post_init__()
        if isinstance(self.loop_steps, tuple) and self.data is not None:
            # Refused before anything loads: a count a phase the data reads in.
            phases = self.data.phase_count
            if len(self.loop_steps) != phases:
                raise ValueError(f"loop_steps names {len(self.loop_steps)} pass counts, one a phase, and the "
                                 f"run's data reads {phases} phase(s)")

    @property
    def handed_passes(self) -> int | None:
        """The pass count the run's model holds and hands on: the last phase's, where the run names one a
        phase (`loop_steps`), else `loop_steps`."""
        return self.loop_steps[-1] if isinstance(self.loop_steps, tuple) else self.loop_steps

    def built[LossT, EffectsT](self, objective: Objective[LossT, EffectsT],
                               mesh: MeshSpec | jax.sharding.Mesh, rollout: Rollout | None,
                               **resources) -> "Trainer[LossT, EffectsT]":
        """`RunConfig.built`, its steps run each phase's pass count where the run names one a phase."""
        phased = {"phase_passes": self.loop_steps} if isinstance(self.loop_steps, tuple) else {}
        return super().built(objective, mesh, rollout, **phased, **resources)

    @property
    def loops(self) -> bool:
        """Whether the run asks anything of a loop (`looped`)."""
        return (self.loop_steps is not None or self.loop_layers is not None
                or self.loop_backprop_steps is not None or self.loop_remat or self.loop_implicit
                or self.loop_exit_gate or self.loop_exit_threshold is not None)

    def exit_settings(self) -> dict[str, float]:
        """The exit objective's weight an objective that trains a gate the run draws (`loop_exit_gate`)
        takes: `EXIT_ENTROPY` (`dew.nn.backbones.looping`) where the run's objective states none, none
        derived where it states one; refused where it states None, which would leave the gate at its draw
        (a hazard of one half every pass), trained by nothing. None where the run draws no gate."""
        from dew.nn.backbones.looping import EXIT_ENTROPY

        if not self.loop_exit_gate:
            return {}
        stated = {} if self.objective is None else self.objective.fields
        if "exit_entropy" not in stated:
            return {"exit_entropy": EXIT_ENTROPY}
        if stated["exit_entropy"] is None:
            raise ValueError("--loop-exit-gate draws a gate the objective's exit_entropy trains, and the run "
                             "sets it to None, which trains nothing: state a weight, or leave it out for "
                             f"{EXIT_ENTROPY}")
        return {}

    def looped(self, start: "Pretrained") -> "Pretrained":
        """`start` with its block run as the run asks (`looping`), as loaded where it asks nothing."""
        if not self.loops:
            return start
        return looping(start, self.handed_passes, self.loop_layers, backprop_steps=self.loop_backprop_steps,
                       remat=self.loop_remat or None, implicit=self.loop_implicit or None,
                       exit_gate=self.loop_exit_gate or None, exit_threshold=self.loop_exit_threshold,
                       key=self.trainer.key)


def looping(start: "Pretrained", steps: int | None, layers: tuple[int, int] | None, *,
            backprop_steps: int | None = None, remat: bool | None = None,
            implicit: bool | None = None, exit_gate: bool | None = None,
            exit_threshold: float | None = None, key: int | jax.Array = 0) -> "Pretrained":
    """A loaded checkpoint whose block runs `steps` passes: a looped one's own block, or `layers`,
    which gives a decoder with no loop one over its own layers. Refused where neither names a block,
    or a new block's passes are not stated. `backprop_steps` and `remat` choose
    the loop's backward memory, `implicit` its later passes' rule (`Loop.implicit`), and `exit_gate` and
    `exit_threshold` its exits (`Loop.exit_gate`); None keeps a looped checkpoint's own value. A loop
    that gives its layers parameters of their own (an implicit loop's θ, an exit gate) draws them from
    `key` beside the loaded weights, fresh (`Pretrained.with_model`)."""
    from dew.nn.protocols import Looping

    looped = (start.model.with_passes(steps, layers, backprop_steps=backprop_steps, remat=remat,
                                      implicit=implicit, exit_gate=exit_gate,
                                      exit_threshold=exit_threshold)
              if isinstance(start.model, Looping) else None)
    if looped is None:
        raise ValueError(f"this {type(start.model).__name__} has no loop: --loop-layers names the block of "
                         "its layers to repeat and --loop-steps how many times")
    return start.with_model(looped, key=key)


__all__ = ["DecoderRunConfig", "MetricConfig", "ObjectiveConfig", "Prepared", "Release", "RunConfig",
           "SourceOptions", "TrainerConfig",
           "flag_selected", "looping"]
