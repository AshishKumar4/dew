"""Typed configuration for a training run.

One dataclass tree describes a run: what to build, what to feed it, how to
optimize it, and how the trainer runs. Recipes parse it with tyro, build the
objective and the data it names, and pass both to `RunConfig.train`, so
`to_dict()` is a full record of a run and `from_dict()` rebuilds it.

The model is its class and the fields the run gives it (`ModelConfig`), and the data a chain
(`dew.data`), which the record writes through the data's codec (`dew.data.records`); both are built
in Python, where a recipe builds its run, and the command line sets the run's own settings alone
(`RunConfig.cli`).

The resolved config is the run's spec. A recipe writes it to `run.json` next
to the checkpoints with `save`, and `load` reads it back into the same class,
so inference rebuilds a run from what training was built from. A field the
file lacks takes its declared default, and a field the class does not have
raises an error. The model and the objective record their classes' defaults
too (`ModelConfig.defaults`), so a default changed since leaves the record
building what it built.
"""

import contextlib
import dataclasses
import functools
import json
import os
import sys
import types
import typing
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, ClassVar, NoReturn, Self, Union

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
from dew.checkpoints._step_records import saved_under
from dew.config.earlier import (
    ADDITIVE_UNSETTABLE,
    KERNELS_SPLIT,
    LAYOUT_UNSETTABLE,
    MARKERS,
    MASKED_DECAY_MARK,
    TYPED_OPTIMIZER,
    _LoopChange,
    earlier_run,
    unmarked,
)
from dew.config.release import Release
from dew.config.sweep import RandomSearch, Search, Space, read_ledger, write_ledger
from dew.config.trainer import (
    PASSES,
    Diagnostics as Diagnostics,
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
from dew.nn.backbones.decoder_stack import Loop
from dew.nn.kernels import generation
from dew.nn.precision import Numerics
from dew.objectives.base import Effects, Loss, Metric, Objective
from dew.optim import OptimConfig, ParamGroup
from dew.registry import (
    Configured,
    ModelConfig,
    Registered,
    declared_type,
    from_record,
    implied_subcommand,
    member_fields,
    metrics,
    models,
    objectives,
    parameters,
    record_argument,
    to_record,
)
from dew.scores import Summary
from dew.state import TrainState
from dew.telemetry.profile import CaptureOptions
from dew.telemetry.records import RunRecord, TrialFinished, packages_installed, tracker_json
from dew.training._fit import FitArguments, validation_suites
from dew.training.ahead import AheadOfTime, CompiledAhead, fit_plan
from dew.training.display import TrainingDisplay
from dew.training.evaluation import EvalSuite, Split, episodic
from dew.training.phase_passes import PassPolicy, SampledPasses
from dew.training.profiling import ProfileReport, ProfileWindow
from dew.training.quantization import quantize_trunk
from dew.training.rollout import Rollout
from dew.training.targets import AheadTarget, LowerFor
from dew.training.tracker import LocalTracker, Tracker, Trackers, WandbTracker

if TYPE_CHECKING:
    from _typeshed import DataclassInstance

    from dew.interop import Pretrained
    from dew.training._fit import Resumed
    from dew.training.trainer import Trainer


type LowerTarget = AheadTarget | LowerFor
"""What a run compiles or lowers its first step for in place of training (`RunConfig.lower`): GPUs or a TPU
topology it compiles ahead for and saves (`AheadTarget`), or a CUDA generation it lowers for
(`LowerFor`)."""


@dataclasses.dataclass(frozen=True)
class ObjectiveConfig(Registered):
    """An objective's class and the arguments the run gives it.

    `name` is an alias or an import path, held as the path. `fields` are the
    constructor arguments the run states, held as their records
    (`registry.argument_records`); what the objective is built around, the
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
    did. The split it scores is its suite's (`Suite`, `RunConfig.suites`).
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


@dataclasses.dataclass(frozen=True)
class Suite:
    """What one validation split scores and how often (`RunConfig.suites`): its `metrics`, each with the
    arguments the run gives it, the run's where None (`RunConfig.val_metrics`), none where empty; and its
    cadence, as `trainer.eval_every` spells one, the run's where None."""

    metrics: tuple[MetricConfig, ...] | None = None
    every: int | str | None = None

    def __post_init__(self):
        # A record carries every sequence as a JSON list; the field is a tuple of metrics' configs.
        if self.metrics is not None:
            object.__setattr__(self, "metrics", tuple(self.metrics))
            stray = [metric for metric in self.metrics if not isinstance(metric, MetricConfig)]
            if stray:
                raise TypeError(f"a suite's metrics are metrics' configs, {{'name': ..., 'fields': "
                                f"{{...}}}}, not {stray!r}")


def _metric_class(name: str) -> type:
    """The metric class `name` names, an alias or import path: a run's metric is a class, whose
    arguments its record holds."""
    member = metrics[name]
    if not isinstance(member, type):
        raise TypeError(f"{name!r} names {member!r}, which is no metric class")
    return member


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

    def lower(self, name: str, target: LowerTarget) -> NoReturn:
        """Build the run under `name` as `train` would and compile (`AheadTarget`) or lower (`LowerFor`) its
        first step for `target`, training nothing: it ends by raising what says it did (`CompiledAhead`,
        `dew.training._lowering.Lowered`), a `SystemExit` of status 0."""
        self.run.train(self.objective, self.dataset, name=name, metrics=self.metrics, rollout=self.rollout,
                       summary=self.summary, validation=self.validation, after=self.after, lowering=target)
        raise AssertionError("a run lowered for a target ends by raising")

    def inspected(self, name: str) -> dict[str, float]:
        return self.run.viewed(self.objective, self.dataset, name=name, rollout=self.rollout,
                               validation=self.validation)


_SUBCOMMAND = _SubcommandConfig
"""The marker a field's annotation declares a subcommand with (`tyro.conf.subcommand`, which states it as
an object of this class)."""


def flag_subcommands(config: type, prefix: str = "", held: Configured = None) -> dict[str, str]:
    """The optional settings `config` declares that `cli` turns on by their own flags (`flag_selected`),
    by flag path, each with the subcommand its flags imply: a field annotated as one or more subcommands or
    None (`tyro.conf.subcommand`), its first; and those within each such setting's settings and within a
    nested run setting (a `dew.config` dataclass, the trainer's), parents before their own. A field's
    default is `held`'s, the run the command line parses over (`cli`'s `default`, a recipe's), where given,
    else the class's own, so a flag of a recipe's optimizer implies that optimizer, never the class's
    default one (review of API03)."""
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
        default = (vars(held)[field.name] if held is not None
                   else field.default_factory() if callable(field.default_factory)
                   else None if field.default is dataclasses.MISSING else field.default)
        # A default's own subcommand, which a flag of its fields given alone implies: a chosen record's
        # (`dew.registry.implied_subcommand`), or the one member of its own class: `--optim.optimizer.b2 0.95`
        # sets the default AdamW's. Members of one class apart by their defaults (`trainer.precision`'s
        # presets) imply none: the command line names one.
        implied = implied_subcommand(default) if default is not None else None
        own = [name for kind, name in marked if type(default) is kind]
        if implied is None and len(own) == 1:
            implied = own[0]
        if marked and len(marked) == len(given) < len(options):
            found[path] = marked[0][1]
            nested = [kind for kind, _ in marked]
        else:
            if implied is not None:
                found[path] = implied
            nested = given if len(given) == 1 else []
        for kind in nested:
            if (isinstance(kind, type) and dataclasses.is_dataclass(kind)
                    and kind.__module__.startswith(("dew.config", "dew.optim"))):
                found.update(flag_subcommands(kind, f"{path}.",
                                              default if isinstance(default, kind) else None))
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
    from dew.registry import imported

    owner = models[model.name] if model.name in models.paths else imported(model.name)
    return isinstance(owner, type) and any(name == "dtype" for name, _, _ in member_fields(owner))


def _model_changes(saved: ModelConfig, before: JSON, now: JSON) -> list[str]:
    """What a resume of a run whose model record was `before` (`saved` read back) changes of what it trains
    under `now`, by name: its class or wrappers (`model`), or a field that says what the model computes or
    the numbers it computes in, or what it trains toward (`registry.computing_differences`, `NUMBERS`,
    `ROUNDS` and `TRAINS` kept), as `model.<field>`. A field that says how it runs or trains alone
    (`dew._records.RUNS`: remat, dropout, kernels), which the memory plan or the devices a resume lands on
    choose, changes freely."""
    records = [_records.record(record, "model") for record in (before, now)]
    named = [{key: value for key, value in record.items() if key not in ("fields", "defaults")}
             for record in records]
    if named[0] != named[1]:
        return ["model"]
    stated = [{**_records.record(record.get("defaults") or {}, "model.defaults"),
               **_records.record(record.get("fields") or {}, "model.fields")} for record in records]
    return [f"model.{name}" for name in registry.computing_differences(
        saved.member, *stated, kept=(_records.NUMBERS, _records.ROUNDS, _records.TRAINS))]


def _learns_alike(before: ObjectiveConfig | None, now: ObjectiveConfig | None) -> bool:
    """Whether an objective `now` trains what `before` trained: one class, every argument the same but
    those its class lets a resume change (`dew.objectives.base.Objective.resumed_freely`), a stated argument
    the same as the default it states again."""
    if before is None or now is None or before.name != now.name:
        return False
    resumable = objectives[now.name].resumed_freely()

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


def _within(node: dict[str, JSON], key: str, unset: JSON = None) -> dict[str, JSON]:
    """The record `node` holds under `key`. A field declaring a base or a
    union records its value's class beside its fields, and this is the fields.
    An optional record `node` holds none of is set to `unset` first, its class's
    default (`_unset_default`), where it has one."""
    if node.get(key) is None and unset is not None:
        node[key] = unset
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
    keys, unset = _placed(config, path)
    *groups, field = keys
    node = config._fields_record()
    for group in groups:
        node = _within(node, group, unset.get(group))
    return node.get(field)


def _unset_default(owner: "RunConfig | DataclassInstance | type[DataclassInstance]", group: str
                   ) -> "DataclassInstance | None":
    """The value an optional record field `group` of `owner` takes where it holds none and a path steps into
    it (`trainer.layout.min_shard` of an unset `TrainerConfig.layout`): its one dataclass's defaults; None
    for a field of no such type (row 9635), or of one with no default to take, a field it needs stated (a
    decision run's `CrossContext.layers`)."""
    annotation = registry.declared_type(owner if isinstance(owner, type) else type(owner), group)
    kinds = [kind for kind in typing.get_args(annotation)
             if isinstance(kind, type) and dataclasses.is_dataclass(kind)]
    if len(kinds) != 1 or type(None) not in typing.get_args(annotation):
        return None
    stated = [field.name for field in dataclasses.fields(kinds[0]) if field.init
              and field.default is dataclasses.MISSING and field.default_factory is dataclasses.MISSING]
    return None if stated else kinds[0]()


def _placed(config: "RunConfig | DataclassInstance", path: str, record: Mapping[str, JSON] | None = None
            ) -> tuple[list[str], dict[str, JSON]]:
    """The keys of `path`'s value in `config`'s record, and the record of the default each unset optional
    record along it takes (`_unset_default`), by its key: each group stepped through the run where it holds
    it; where it holds a section unset that `record` (the record an override writes) sets, a shallower path
    of the same point, that section read off `record` by the one dataclass its field declares beside None,
    building no run (row 12908); KeyError for a path naming no field, or a section of no one such class."""
    *groups, field = path.split(".")
    held: object = config
    kind: type = type(config)
    node: object = record
    unset: dict[str, JSON] = {}
    for group in groups:
        # A record section read off `record` steps by its declared dataclass; any other mapping (a run's
        # `TrainerConfig.links`) holds no group a path names.
        owner: DataclassInstance | type[DataclassInstance]
        if isinstance(held, Mapping):
            if not (dataclasses.is_dataclass(kind) and group in _names(kind)):
                raise KeyError(f"{path} names no group of the run record")
            owner, value = kind, held.get(group)
        else:
            if not (dataclasses.is_dataclass(held) and group in _names(held)):
                raise KeyError(f"{path} names no group of the run record")
            owner, value = held, getattr(held, group)
        stated = node.get(group) if isinstance(node, Mapping) else None
        if value is None and isinstance(stated, Mapping):
            declared = registry.declared_type(kind, group)
            kinds = [member for member in (declared, *typing.get_args(declared))
                     if isinstance(member, type) and dataclasses.is_dataclass(member)]
            if len(kinds) != 1:
                raise KeyError(f"{path} names no group of the run record")
            held, kind = stated, kinds[0]
        elif value is None and (default := _unset_default(owner, group)) is not None:
            held, unset[group] = default, registry.to_record(default, type(default))
            kind = type(default)
        else:
            held, kind = value, type(value)
        node = stated
    if isinstance(held, Registered):
        # A command line's subcommand (`registry.record_flags`) is the record it makes: its plain flags are
        # that record's `fields` entries.
        held = held.recorded()
    if isinstance(held, Registered) and field not in _names(held):
        if field not in parameters(held.member)[0]:
            raise KeyError(f"{path} names no argument of {held.name}")
        return [*groups, "fields", field], unset
    named = kind if isinstance(held, Mapping) else held
    if not (dataclasses.is_dataclass(named) and field in _names(named)):
        raise KeyError(f"{path} names no field of the run record")
    return [*groups, field], unset


def _names(held: "DataclassInstance | type[DataclassInstance]") -> set[str]:
    """The fields a dataclass declares."""
    return {field.name for field in dataclasses.fields(held)}


_RANDOM_SEARCH = RandomSearch()


def _declared(field: dataclasses.Field) -> Configured:
    """A dataclass field's declared default, its factory's value where it has one."""
    if field.default_factory is not dataclasses.MISSING:
        return field.default_factory()
    return field.default


def _flagged(value: Configured) -> Configured:
    """`value` as a command line's parser holds it: a record of a kind chosen by subcommands its
    subcommand's (`Registered.flagged`), any other value as it is."""
    return value.flagged() if isinstance(value, Registered) else value



@dataclasses.dataclass(frozen=True)
class RunConfig:
    """A whole run's configuration. A kind of run subclasses it with its own
    settings and builds what it trains from them (`build_run`, which `prepare` calls
    on the run with its Hub sources pinned); `run` trains it.
    """

    RESUMABLE: ClassVar[tuple[str, ...]] = ("trainer", "optim", "data", "validation", "data_seed",
                                            "val_metrics", "suites", "val_summaries", "diagnostics")
    """The fields a run resumed under its name may change: how it trains and optimizes, what it reads
    and scores, the kernels it runs (`kernels`, each on whatever hardware the resume lands on), and, in a
    kind of run that adds them, how it previews. Every other field says what
    it trains, which the resume must keep (`check_resume`), and so does `trainer.quantization`, which
    changes the forward pass the weights train through."""

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
    (`dew.data.records`)."""
    validation: Annotated[Dataset | Mapping[str, Dataset] | None, tyro.conf.Suppress] = None
    """The splits each evaluation scores, one dataset or several by name, each batched by its own batch
    stage or the training batch."""
    data_seed: int = 0
    """The seed of the data's order and of its random steps."""
    diagnostics: Diagnostics = dataclasses.field(default_factory=Diagnostics)
    """What the run pins of how its programs run where it diagnoses or measures (`Diagnostics`): each op's
    kernel, its axes' links, the measuring of its launches and extra XLA flags; each derived unset."""
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
    """The metrics a validation split scores where its suite names none (`suites`), each with the
    arguments the run gives it (a CLIP metric's model, an FID's weights), one JSON list on the command
    line: `--val-metrics '[{"name": "accuracy"}]'`; `[]` scores none. None scores the kind of run's own
    (`default_metrics`): perplexity for a language model, a decision run's accuracy, calibration,
    risk-coverage and log loss, a JEPA run's probes, CLIP for an image run."""
    suites: Annotated[Mapping[str, Suite], record_argument(Mapping[str, Suite])] = (
        dataclasses.field(default_factory=dict))
    """What a validation split scores and how often where it differs from the run's (`val_metrics`,
    `trainer.eval_every`), by the split's name (`Suite`), one JSON object on the command line: `--suites
    '{"test": {"metrics": [{"name": "brier"}], "every": "25%"}}'` scores the Brier score on `test` every
    quarter of the run, and every other split as the run says. A split it names must be one the run
    validates."""
    val_summaries: Annotated[tuple[Summary, ...], record_argument(tuple[Summary, ...])] = ()
    """Scalars weighed from each evaluation's split scores, a board's weighted index, each formed at a
    step that scored every split it weighs and logged as `evaluation/summary/<name>`, which
    `--trainer.best.metric evaluation/summary/<name>` ranks by (`dew.scores.Summary`):
    `--val-summaries '[{"name": "board", "weights": {"val/accuracy": 1, "test/accuracy": 2},
    "better": "higher"}]'`.
    """

    def __post_init__(self) -> None:
        # A record carries every sequence as a JSON list and a command line writes one too; the field
        # is a tuple, so the value is one.
        if self.val_metrics is not None:
            object.__setattr__(self, "val_metrics", tuple(self.val_metrics))
            stray_metrics = [metric for metric in self.val_metrics if not isinstance(metric, MetricConfig)]
            if stray_metrics:
                raise TypeError(f"val_metrics holds metrics' records, {{'name': ..., 'fields': {{...}}}}, "
                                f"not {stray_metrics!r}")
        stray = {split: suite for split, suite in self.suites.items() if not isinstance(suite, Suite)}
        if stray:
            raise TypeError(f"suites hold each split's Suite, {{'metrics': [...], 'every': ...}}, not "
                            f"{stray!r}")

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
        # A private field (a saved record's translation, `_earlier_loop`) recorded where it holds one alone.
        with registry.recording():
            return {field.name: to_record(getattr(self, field.name), declared_type(type(self), field.name))
                    for field in dataclasses.fields(self) if field.name not in CHAINS
                    and not (field.name.startswith("_") and getattr(self, field.name) == field.default)}

    @classmethod
    def from_dict(cls, values: Mapping[str, object]) -> Self:
        """Read back what `to_dict` wrote, for subclasses too. A field the
        record lacks takes its default; an unknown field raises. A chain
        recorded as unrecorded, or a recipe's record written before runs
        recorded chains, is refused by name. A model recorded with an earlier
        main's fields reads as its checkpoint's record does."""
        from dew.inference.tasks import earlier_model

        # A `stop` naming no class reads through the registry's one rule (`registry._unnamed`): a record
        # written while it held a Pause alone (ledger 1552) a Pause, a plateau's fields a Plateau (row 12115).
        rest = {name: value for name, value in values.items() if name not in CHAINS}
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
        argument of the class it names. A path the run does not declare raises, rather than
        training the unchanged config. The paths are set together and the run
        is built once, so fields valid only together (a token budget with no
        row batch) are one point; each path is placed in this config, so a path
        under `model` names an argument of the model this config holds.
        """
        chained = {path: value for path, value in point.items() if path.partition(".")[0] in CHAINS}
        record = self._fields_record()
        # The shallower paths first: a path under a section no run holds unset that the same point sets
        # (`context`, then `context.encoder`) is placed in the run as the point sets that section.
        for path in sorted(point, key=lambda path: path.count(".")):
            value = point[path]
            if path in chained:
                continue
            # A section the run holds unset that a shallower path of the point set is placed by its declared
            # class, the run built once, from the completed record (`_placed`'s `record`).
            keys, unset = _placed(self, path, record)
            *groups, field = keys
            node = record
            for group in groups:
                node = _within(node, group, unset.get(group))
            node[field] = value
        chains = _overridden({name: getattr(self, name) for name in CHAINS}, chained)
        # Built once, every field of the point in place, so its checks read the run it describes.
        return dataclasses.replace(type(self).from_dict(record), **chains)

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
        # The record written is the run's whole account from here on: an intermediate Dew's mark beside
        # it, read where this run was loaded (`load`), is decided and read no more.
        (path / MASKED_DECAY_MARK).unlink(missing_ok=True)
        return str(target)

    def record(self) -> dict[str, JSON]:
        """The run's class record: its class by its alias in `dew.registry.runs`, or by its import path
        where the table holds none, and its fields (`to_dict`); beside them, that it states its trainer's
        layout and its messages' additive term as this Dew does (`LAYOUT_UNSETTABLE`, `ADDITIVE_UNSETTABLE`),
        and names SYRK's kernel apart from the grouped products' (`KERNELS_SPLIT`),
        which `earlier_run` reads and takes off."""
        return {"class": registry.runs.name_of(type(self)), "fields": self.to_dict(), LAYOUT_UNSETTABLE: True,
                ADDITIVE_UNSETTABLE: True, KERNELS_SPLIT: True}

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

    def built_fresh(self) -> None:
        """Refuse `source` in a run whose model is built from `model`, loading no source: what it says
        `Pretrained.load` builds of a source (`source_keywords`), reaches nothing such a run builds, whose
        own fields say it."""
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
            raise ValueError(f"the run's model names {self.model.name}, and the source this "
                             f"{type(self).__name__} starts from decides its architecture; name none")

    def check_resume(self, saved: Mapping[str, object], *, numerics_stated: bool = False) -> None:
        """Refuse to resume the run whose record is `saved` as this one where they train different
        things: another class, or a field outside `RESUMABLE` that differs. A changed `rope_theta`
        keeps every shape, so it would otherwise continue the old weights under new semantics. The
        saved fields are read back as this class reads them (`from_dict`), so a field the record
        predates takes its default, and one it left to its build unresolved takes this build's. Its
        `trainer.precision` is held to this one's where the step it resumes states no numerics of its
        own (`numerics_stated`), which the trainer holds to what they computed instead
        (`dew.training.startup.numerics_differs`, row 13708)."""
        recorded = self._saved_as(saved)
        if recorded is None:
            raise ValueError(f"the run to resume is no {type(self).__name__}; resume it as the run it "
                             "records, or train under another name")
        before, now = (run._resumed_as()._fields_record() for run in (recorded, self))
        # A field the saved run left to its build and its record holds unresolved, which no checkpoint
        # states either (`earlier_checkpointed`), is the one this build resolves (`BUILT`).
        unresolved = {path for path in self.BUILT if "." not in path and before.get(path) is None}
        changed = sorted(name for name in now.keys() | before.keys()
                         if name not in self.RESUMABLE and name not in unresolved and name != "model"
                         and now.get(name) != before.get(name))
        changed += _model_changes(recorded.model, before.get("model"), now.get("model"))
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
        if not numerics_stated and recorded.trainer.precision != self.trainer.precision:
            changed.append("trainer.precision")
        if changed:
            raise ValueError(f"the run to resume trained other {', '.join(changed)}; resume it as it "
                             "recorded them, or train under another name")

    def _resumed_as(self) -> Self:
        """This run as a resume compares it (`check_resume`): with the fields that say how its layers execute
        and not what they compute or store read as their defaults, a subclass's; memory alone, they compute
        the same step, so a run resumes under either. A model's own fields that say how it runs, its
        recompute (`remat`) among them, are its class's to mark (`dew._records.RUNS`), which a resume reads by
        its class (`registry.computing_fields`). A source's weights packed or decoded
        (`SourceOptions.packed_weights`) are not: packed, its banks are no trainable parameters and hold no
        optimizer moments, so a resume across them is refused by name."""
        return self

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
        saved run is (`earlier_run`), and as the run class it names reads its own from its checkpoints
        (`earlier_checkpointed`); None where none was saved."""
        saved = epath.Path(directory) / RUN_FILE
        if not saved.exists():
            return None
        record = earlier_run(json.loads(saved.read_text()))
        named = registry.class_record(record)
        run = None if named is None else registry.runs[named[0]]
        if not (isinstance(run, type) and issubclass(run, RunConfig)):
            return record
        return run.earlier_checkpointed(record, checkpoints)

    @classmethod
    def earlier_checkpointed(cls, record: Mapping[str, object], checkpoints: Checkpoints
                             ) -> Mapping[str, object]:
        """A saved run's `record` of this class (`earlier_run`'s) with what an earlier Dew left to its
        `checkpoints` stated in it, so a resume builds as the run trained; as it is for a class that left
        none (a decision run's option writing does, `dew.decision.DecisionRunConfig`)."""
        return record

    @classmethod
    def cli(cls, args: Sequence[str] | None = None, *, default: Self | None = None) -> Self:
        """Parse a run from `args`, or from the process's own command line when `args` is None: a flag for
        each of the run's own settings, `--trainer.steps 2000` or `--optim.lr 3e-4`, over
        `default` (a recipe's run) or the class's own.

        An optional setting such as `lora` is turned on by its own flags: `--lora.rank 16 --lora.modules
        q_proj` stands for `lora:lora --lora.rank 16 ...`, and with no `--lora.` flag the run trains
        without an adapter. What the run trains, its model, objective and data, is built in Python, as
        `default` holds them, and takes no flag (`dew train` reads a run a Python file builds).
        """
        given = flag_selected(list(sys.argv[1:] if args is None else args),
                              flag_subcommands(cls, held=default))
        # The run parses with its model and objective held as `default` holds them, or as the class
        # declares them; every other field from `default`, or from its own declared default, which a value
        # a field's default resolves (an optimizer's own rate) follows where its flags move it.
        hints = typing.get_type_hints(cls)
        declared = cls.__dataclass_fields__
        held_fields = [(name, tyro.conf.Suppress[hints[name]],
                        dataclasses.field(default=getattr(default, name)) if default is not None else
                        dataclasses.field(default=declared[name].default,
                                          default_factory=declared[name].default_factory))
                       for name in ("model", "objective")]
        parser = dataclasses.make_dataclass(
            cls.__name__, held_fields,
            bases=(cls,), frozen=True, namespace={"__post_init__": lambda self: None})
        # A kind its command line chooses by subcommands holds its default as that subcommand's record
        # (`Registered.flagged`), which tyro reads a union's default by; the run it parses holds each as
        # its kind's own record again (`Registered.recorded`), its subcommands the parser's alone.
        held = parser(**{field.name: _flagged(getattr(default, field.name) if default is not None
                                              else _declared(declared[field.name]))
                         for field in dataclasses.fields(cls)})
        parsed = tyro.cli(tyro.conf.CascadeSubcommandArgs[parser], args=given, default=held)
        if parsed is None:
            raise ValueError("the command line built no run config")
        values = {field.name: vars(parsed)[field.name] for field in dataclasses.fields(cls)}
        for name, value in values.items():
            if isinstance(value, Registered):
                values[name] = value.recorded()
        # A rate the run left to its optimizer (`OptimizerSpec.OWN_RATE`) follows the optimizer its flags
        # choose, unless a flag names the rate: Gauss-Newton's whole step, never the default AdamW's.
        start = default.optim if default is not None else OptimConfig()
        optim = values["optim"]
        if (isinstance(optim, OptimConfig) and start.lr == start.optimizer.OWN_RATE
                and optim.lr == start.lr and not any(arg.startswith("--optim.lr") for arg in given)):
            values["optim"] = dataclasses.replace(optim, lr=None)
        return cls(**values)

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
        # A field its record left to its checkpoints (`earlier_checkpointed`) is read where the run resumes
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
        # Read in this Dew's shapes, whatever the record says of them beside its fields (`earlier_run`).
        named = registry.class_record({key: value for key, value in record.items() if key not in MARKERS})
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
        run = self._marked() if trainer.name is not None and trainer.resume != "never" else self
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
                run = run.override({path: _recorded_at(recorded, path) for path in unset
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

    def _marked(self) -> Self:
        """This run as its earlier run trained where an intermediate Dew rewrote its record beside its mark
        (`MASKED_DECAY_MARK`), decided by the step its resume restores (`Checkpoints.latest`), here, where
        its process has joined its pool (`resumed`, after `prepare_process`): its optimizer's decay masked off
        as that run's (`unmarked`), or as it is where that step is the intermediate Dew's own
        (`TYPED_OPTIMIZER`), saved under the record it rewrote."""
        name = self.trainer.name
        assert name is not None, "a run resumes under its name"
        mark = epath.Path(self.trainer.directory(name)) / MASKED_DECAY_MARK
        # Read alike by every process, or refused on all, before the checkpoints' choice agrees across them.
        marked = agreed_same("a run's masked-decay mark",
                             lambda: json.loads(mark.read_text()) if mark.exists() else None)
        if marked is None:
            return self
        with self.trainer.checkpoints(name) as checkpoints:
            step = checkpoints.latest
            if step is not None and saved_under(checkpoints, step, TYPED_OPTIMIZER) is True:
                return self
        earlier = unmarked(self.record(), _records.record(marked, MASKED_DECAY_MARK))
        fields = _records.record(earlier["fields"], "run fields")
        return dataclasses.replace(self, optim=from_record(OptimConfig, _records.json_value(fields["optim"],
                                                                                           "optim")))

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
        """The metrics a validation pass scores where the run names none (`val_metrics`, `suites`), by what
        the run's data
        `validates`: a kind of run's own, none for a run that names its objective alone."""
        del validates
        return ()

    def metric(self, config: MetricConfig) -> Metric:
        """`config` built with what this kind of run derives for it where it states none of its own."""
        return config.build()

    def validation_metrics(self, *, validates: bool) -> tuple[Metric, ...]:
        """The kind of run's own metrics (`default_metrics`), built (`metric`), by what the run's data
        `validates`: what each split scores where its suite names none (`evaluation_suites`)."""
        return tuple(self.metric(config) for config in self.default_metrics(validates=validates))

    def evaluation_suites(self, dataset: Loader, validation: Mapping[str, Split] | None,
                          metrics: Sequence[Metric] = ()) -> Mapping[str, EvalSuite] | None:
        """Each split the run validates, `validation`'s or `dataset.val` as `val`, scoring the metrics its
        suite names, else the run's `val_metrics`, built (`metric`), else the kind's own `metrics`, at its
        suite's cadence, else the run's; None where the run names neither, so each scores the kind's own at
        the run's. A suite of a split the run does not validate is refused, as are metrics the run names
        (`val_metrics`) where it validates no split, which would score nothing."""
        if self.val_metrics is None and not self.suites:
            return None
        splits = dict(validation) if validation is not None else (
            {"val": dataset.val} if dataset.val is not None else {})
        if not set(self.suites) <= set(splits):
            raise ValueError(f"suites name {sorted(self.suites)}, and the run validates "
                             f"{sorted(splits) or 'none'}")
        if self.val_metrics and not splits:
            raise ValueError(f"val_metrics names {[metric.name for metric in self.val_metrics]}, and the run "
                             "validates no split: give it a validation split, or name none")
        built = {}
        for split, reader in splits.items():
            suite = self.suites.get(split, Suite())
            named = self.val_metrics if suite.metrics is None else suite.metrics
            scored = tuple(metrics) if named is None else tuple(map(self.metric, named))
            # A split's cadence is read as the run's is (`TrainerConfig.interval`): `epoch` and `passes` too.
            every = (None if suite.every is None else
                     self.trainer.interval(suite.every, dataset, f"--suites {split}.every"))
            built[split] = EvalSuite(reader, scored, every)
        return built

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
        stages, `data_seed` and `data_plan_version`, the training read bounded to its horizon's examples where
        the run counts them (`trainer.examples`), or to its epochs' passes over its source examples where its
        read has no record count, so its length is read off the loader this returns; its loading workers sized
        for the run's length, its horizon's steps, epochs or tokens (`dew.data.Loading.resolved`)."""
        # At the version `resumed` resolves (itself where `prepare` or `readied` resolved it; a run built
        # alone, `build_run`, resolves it here), which it always states.
        planned = self.resumed(("data_plan_version",)).data_plan_version
        assert planned is not None
        # A run of a version this Dew reads none of, resumed exactly, is refused before its data compiles.
        read_by(planned)
        trainer = self.trainer
        return fitted(dataset, validation, objective=objective, seed=self.data_seed,
                      resume=trainer.data_resume,
                      examples=trainer.examples, epochs=trainer.epochs,
                      steps=trainer.steps, tokens=trainer.tokens,
                      count_passes=self.reads_passes())

    def reads_passes(self) -> bool:
        """Whether a cadence of the run reads its data's passes (`epoch`, or the default `passes`): its
        evaluation's, its checkpoints' or a split's (`Suite.every`), which a run bounded otherwise than in
        epochs or examples counts (`dew.data.compile`'s `count_passes`); a run whose cadences read none
        compiles its data as it would without them."""
        trainer = self.trainer
        return any(value in ("epoch", PASSES) for value in (
            trainer.eval_every, trainer.checkpoint_every,
            *(suite.every for suite in self.suites.values())))

    def run(self) -> TrainState:
        """Prepare the process, build what this run trains (`prepare`) and train it.

        The run trains under `trainer.name`, or `<objective>-<data>/date-<time>`."""
        from dew.training import prepare_process

        trainer = self.trainer
        prepare_process(trainer.wandb, trainer.multi_host, self.diagnostics.xla_flags,
                        trainer.compilation_cache_dir)
        prepared = self.prepare()
        return prepared.train(prepared.run.run_name())

    def run_name(self) -> str:
        """The name the run trains under: `trainer.name`, or `<objective>-<data>/date-<time>`."""
        from dew.training import run_timestamp

        stated = self.objective
        return self.trainer.name or (f"{'run' if stated is None else objectives.label(stated.name)}-"
                                     f"{_chain_label(self.data)}/date-{run_timestamp()}")

    def lower(self, target: LowerTarget) -> NoReturn:
        """Prepare the process, build what this run trains (`prepare`) and compile or lower its first
        step for `target` (`Prepared.lower`) under its name (`run_name`), training nothing."""
        from dew.training import prepare_process

        prepare_process(None, self.trainer.multi_host, self.diagnostics.xla_flags,
                        self.trainer.compilation_cache_dir)
        prepared = self.prepare()
        prepared.lower(prepared.run.run_name(), target)

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
        # The step alone, from the run's records: the loader writes what it reads, where it places it.
        step = SavedRun.selected(pretrained, self.source.step).step
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
            self.trainer, name=name, resume="never", steps=warmup + steps, stop=None,
            best=None, restore_best=False, cooldowns=None, log_every=steps, eval_every=None,
            checkpoint_every=None, checkpoint_final=False, local_directory=None, local_every=None, wandb=None,
            ahead=None)
        prepare_process(None, trainer.multi_host, self.diagnostics.xla_flags, trainer.compilation_cache_dir)
        # The report names its paths whole, wherever it is read from.
        directory = os.path.abspath(directory or os.path.join("profiles", name, run_timestamp()))
        if (Path(directory) / name / "tracking").exists():
            raise ValueError(f"{directory} holds a profile of {name} already; give the profile a directory "
                             "of its own")
        window = os.path.join(directory, "capture")
        captured = ProfileWindow(window, steps, warmup, capture or CaptureOptions())
        trainer = dataclasses.replace(trainer, checkpoint_dir=directory, profile=captured)
        profiled = dataclasses.replace(self, validation=None, val_metrics=None, suites={}, val_summaries=(),
                                       trainer=trainer)
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
        missed = unprepared(process.multi_host, self.diagnostics.xla_flags, process.compilation_cache_dir)
        if missed:
            named = ", ".join(f"diagnostics.{name}" if name == "xla_flags" else f"trainer.{name}"
                              for name in missed)
            raise ValueError(f"{named} set the process up before JAX opens a "
                             "backend, as `run()` and `dew train` do (`prepare_process`), and this process "
                             "carries none of it: train the run through `run()`, or call prepare_process "
                             "with them before anything calls JAX")

    def skipped_preview(self, objective: Objective[Loss, Effects]) -> str | None:
        """Why the run previews its model at no validation where it would by default, a tracker's run: its
        objective cannot sample (`Objective.preview_refusal`, a model that refuses generation), where a
        preview asked of `Trainer.fit` is refused. The run records it in its summary (`train`); None where
        it previews, or tracks nothing to preview to."""
        return None if self.trainer.wandb is None else objective.preview_refusal()

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
        run = named.resumed(("data_plan_version",))
        dataset, held = run.compiled(objective.data(), dataset, validation)
        dataset, steps = run.trainer.resolved(dataset)
        # Each split scores the metrics its suite or `val_metrics` names, or else `metrics`, the kind's, at
        # its cadence read as the run's is: over the sized trainer and the loader its batch ramp reads.
        suites = run.evaluation_suites(dataset, held, metrics)
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
              after: Callable[[TrainState, str], None] | None = None,
              lowering: LowerTarget | None = None) -> TrainState:
        """Train `objective` on `dataset` as this config describes; every kind of run calls this after
        building both. Where `lowering` names a target, the run compiles or lowers its first step for it
        in place of training (`Prepared.lower`).

        The run lives under `name` in `trainer.checkpoint_dir`, and process zero writes
        the record there before training starts. A `trainer.wandb` opens a tracker
        under the same name, with the record, `summary` (the recipe's own view of the
        run) and the step count as its config. Only `trainer.wandb.publish` publishes the
        final checkpoint to the W&B registry, under the name with slashes and spaces
        replaced because an artifact name allows neither. Local tracking journals
        live in a separate tracking directory.

        `dataset` is compiled once with the objective's own stages (`dew.data.compile.fitted`), and its
        validation splits with it; the run's length, its ramp, its cadences and its recorded plan are
        read off that one compiled loader, which `fit` trains on. Its batch stage is the run's batch, and
        an update of tokens pools `trainer.update_tokens` over its token batch.

        A `trainer.quantization` wraps the module `objective` trains before anything
        initializes it, so the run learns through the quantized forward pass. A
        `lora` is not applied here: the recipe attaches it to the model and variables
        it builds the objective from. `rollout` is the trainer's rollout, which turns
        each prefetched batch into the one the step trains on, as an on-policy
        objective samples it. `validation` names the splits each evaluation
        scores in place of `dataset.val`, as `Trainer.fit` takes them: a held-out
        split and a test set are logged as `val/...` and `test/...`. `metrics` are
        the kind's own, which each split scores where the run names none. With no
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
        skipped = self.skipped_preview(objective)
        eval_every = trainer.eval_interval(dataset)
        # What `fit` is asked for but its resume, which a run compiled ahead compiles its evaluations from.
        fitted = FitArguments(
            # A run bounded in examples that counts no steps ahead keeps its clock (`Trainer.fit`).
            steps=steps, tokens=trainer.tokens, max_tokens=trainer.max_tokens,
            log_every=trainer.log_every,
            examples=dataset.examples if steps is None else None,
            eval_every=eval_every, checkpoint_every=trainer.checkpoint_interval(dataset),
            checkpoint_final=trainer.checkpoint_final, scored_to=trainer.scored_to,
            metrics=metrics if suites is None else (), preview=trainer.wandb is not None and skipped is None,
            best=trainer.best_policies(), stop=trainer.stop, restore_best=trainer.restore_best,
            cooldowns=trainer.cooldowns,
            # A run that names no metrics and validates no split has no suite to score.
            validation=validation if suites is None else suites or None, summaries=self.val_summaries)
        # The splits `fit` validates, resolved as it resolves them (`validation_suites`).
        validated = [] if skipped is None or lowering is not None else validation_suites(
            dataset, validation if suites is None else suites or None, metrics if suites is None else (),
            eval_every).values()
        if any(not episodic(suite.data) for suite in validated):
            # A run that validates a batch split, where a preview samples (`Evaluation.run`; a split of
            # episodes previews none), records the preview it skips, and why.
            summary = {**(summary or {}), "preview": {"skipped": skipped}}
        if isinstance(lowering, AheadTarget):
            raise self.compiled_ahead(objective, dataset, steps, rollout, name, lowering, fitted)
        lower_for = lowering.generation if isinstance(lowering, LowerFor) else None
        tracker = checkpoints = None
        try:
            wandb_tracker = None
            if trainer.wandb is not None and lower_for is None:
                # Shown under its own name where one is given, kept under the run's either way.
                wandb_tracker = WandbTracker(
                    trainer.wandb.project, name if trainer.wandb.name is None else trainer.wandb.name,
                    entity=trainer.wandb.entity, offline=trainer.wandb.offline, shared=trainer.wandb.shared)
            checkpoints = trainer.checkpoints(name)
            local = _local_tracker(checkpoints.directory, name)
            tracker = Trackers(local, *(() if wandb_tracker is None else (wandb_tracker,)))
            # The trainer (`built`) holds mesh, layout, update, fit policy, loss scaling,
            # profile, tune_kernels and any step compiled ahead; prepare_process read the process fields,
            # and the rest are fit's arguments or built the checkpoints and the tracker.
            ahead = None if trainer.ahead is None else AheadOfTime.load(trainer.ahead)
            built = self.built(objective, trainer.mesh, rollout, checkpoints=checkpoints, tracker=tracker,
                               profile=trainer.window(checkpoints.directory, name),
                               lower_for=lower_for, ahead=ahead)

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
            admitted = self._agree_resume(checkpoints, built, dataset)
            agreed("run metadata", record_run)
            with self.traced():
                state = built._fit(dataset, resume=trainer.resume, admitted=admitted, **fitted)
                if after is not None:
                    after(state, checkpoints.directory)
            def publish_checkpoint() -> None:
                """Upload the checkpoint the run ended on, where a tracker takes one."""
                if trainer.wandb is not None and trainer.wandb.publish and wandb_tracker is not None:
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

        with generation.run_precision(self.trainer.precision), dew.kernels.use(self.diagnostics.kernels):
            yield

    def compiled_ahead[LossT, EffectsT](self, objective: Objective[LossT, EffectsT], dataset: Loader,
                                         steps: int | None, rollout: Rollout | None, name: str,
                                         target: AheadTarget, fitted: FitArguments | None = None
                                         ) -> CompiledAhead:
        """The run's first step compiled for `target` and saved
        (`AheadTarget.compiled`), its trainer built as `train` builds it, its links the run's own, over
        that target's compile-only devices (`AheadTarget.compile_only`), with no tracker: what ends a run
        that compiles ahead, with its evaluations' programs where `fitted` is what `train` hands `fit`.
        Its checkpointer is the run `name`'s own, as its fit is planned with one (a `Pause` its
        checkpoints resume, a validation loss that ranks them): a directory that holds checkpoints is
        read, the step its resume takes compiled, and one that holds none is asked and not created
        (`Checkpoints.latest`)."""
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
            raise ValueError("the run names no inspection to view; name one in its trainer's settings "
                             "(`TrainerConfig.inspection`)")
        prepare_process(None, trainer.multi_host, self.diagnostics.xla_flags, trainer.compilation_cache_dir)
        return self.prepare().inspected(trainer.name or "inspected")

    def grouped(self, objective: Objective[Loss, Effects], name: str) -> OptimConfig:
        """The run's optimizer as it trains under `name`: its groups as they take what it draws fresh beside a
        loaded base (`OptimConfig.claiming`, as a `Trainer` given the config takes them): its own, or its
        named ones with the fresh group beside them, or none named, the default groups
        (`OptimConfig.fresh_groups`). A resumed run that names no groups trains in those its record names,
        as its optimizer state holds them; where its record names none, in those it takes now. One that
        names some trains in its own, their rates and a bounded group's interval as it states them. Each
        is refused where their names, patterns or which of them are bounded are not the record's, which
        shape the optimizer state it resumes (`partitioned`). Every process agrees on the groups before any
        records them."""
        def resolved() -> JSON:
            named = self.optim.param_groups
            checkpoints = self.trainer.checkpoints(name)
            saved = epath.Path(checkpoints.directory) / RUN_FILE
            recorded = recorded_param_groups(saved) if checkpoints.resumed(
                self.trainer.resume) is not None and saved.exists() else None
            # The record's own groups where the run names none, else its own; either claims what the run
            # draws fresh, or is refused where it draws none, as a `Trainer` given the config takes them
            # (`OptimConfig.claiming`), which then finds them claimed.
            chosen = recorded if recorded and not named else named
            claimed = dataclasses.replace(self.optim, param_groups=chosen).claiming(
                *objective.fresh_trained())
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
                                       built: "Trainer[LossT, EffectsT]", dataset: Loader) -> "Resumed":
        """The step the run continues, chosen and admitted once (`Resumed`, which its fit takes as it is,
        `Trainer._fit`: none of it chosen, its position read or its numerics checked again, row 13708), with
        its data's saved position as read: refused, on every rank and before
        the run's record is written over what it resumes, a checkpoint of another run under the name or a
        directory `resume` refuses (`Checkpoints.resumed`, `check_resume`) or its numerics' other
        computation (`refuse_other_computation`); then, in an agreement of its own, so every rank enters the
        position's restore only once all accept the record, a saved data position the run's data does not
        read on from (`resume_admitted`), a changed chain's refusal naming the record it resumes where that
        holds its chain (`_routed`)."""
        from dew.training._fit import Resumed

        resume = self.trainer.resume

        def check_resumed() -> tuple[int | None, Mapping[str, object] | None, bool]:
            from dew.training.distributed import under_numerics
            from dew.training.startup import refuse_other_computation, resumed_step

            # Its checkpoint chosen, and a weights-only directory's guard taken (`resumed_step`), under the
            # numerics its steps trace under, as `fit` takes them.
            with self.traced(), under_numerics(built.numerics):
                step = resumed_step(built, resume)
            if step is None:
                return None, None, False
            training = checkpoints.training(step)
            record = self._saved_record(checkpoints.directory, checkpoints)
            if record is not None:
                self.check_resume(record, numerics_stated=isinstance(training, dict)
                                  and isinstance(training.get("numerics"), dict))
            # The trainer's admission of its numerics, under the numerics its steps trace under (the run's,
            # then its trainer's own), before the record is replaced: a refusal leaves the run's record as it
            # was (row 13708).
            if training is not None:
                with self.traced():
                    refuse_other_computation(built, checkpoints, step, numerics_of=training)
            return step, record, training is not None

        # The record of the run this one resumes, as each rank read it once the pool agreed it.
        step, record, numerics = agreed("run resume", check_resumed)

        def check_data() -> tuple[bytes | None, tuple[str, ...]] | None:
            from dew.training.startup import resume_admitted

            if record is None:
                return None
            with _routed(record, self.UNSEEDED):
                return resume_admitted(built, dataset, step)

        # Its saved position read here once, which the fit's stream restores as read (`Trainer._fit`).
        position = agreed("run data", check_data)
        if position is None:
            return Resumed(step, numerics_admitted=numerics, read=False)
        return Resumed(step, *position, numerics_admitted=numerics)

    def built[LossT, EffectsT](self, objective: Objective[LossT, EffectsT],
                               mesh: MeshSpec | jax.sharding.Mesh, rollout: Rollout | None,
                               **resources) -> "Trainer[LossT, EffectsT]":
        """The run's trainer over `mesh`: how it steps, every numeric choice its `TrainerConfig`
        holds (key, layout, update, fit policy, loss scale, kernel tuning), and the
        `resources` its mode adds (checkpoints, tracker, profile, a lowering or a step compiled
        ahead), so a run that trains and one that compiles ahead build the one trainer. The trainer is
        the training package's lazy export (`dew.training`), read where a run trains or compiles
        ahead, so a config read alone reads none of it."""
        trainer = self.trainer
        checkpoints, tracker = resources.pop("checkpoints", None), resources.pop("tracker", None)
        numerics = self.trained_numerics(objective)
        scaled, drawn = (False, None) if numerics is not None else self.trained_knobs()
        return training.Trainer(
            objective, self.optim, key=trainer.key, mesh=mesh, accumulation=trainer.accumulation,
            update_tokens=trainer.update_tokens, split=trainer.split, remat=trainer.remat,
            microbatches=trainer.microbatches, checkpoints=checkpoints, tracker=tracker,
            advanced=training.TrainerAdvanced(
                layout=trainer.layout, rollout=rollout, data_resume=trainer.data_resume,
                skip_nonfinite=trainer.skip_nonfinite, numerics=numerics, dynamic_scale=scaled,
                param_dtype=drawn,
                tune_kernels=self.diagnostics.tune_kernels,
                inspection=trainer.inspection, diloco=trainer.diloco,
                **{"links": self.diagnostics.links, **resources}))

    def compute_dtype(self) -> str:
        """The dtype the run's model computes in: the one its record states, else its numerics'
        (`TrainerConfig.precision`)."""
        stated = self.model.fields.get("dtype")
        return str(stated) if stated is not None else self.trainer.precision.dtypes.compute

    def trained_numerics[LossT, EffectsT](self, objective: Objective[LossT, EffectsT]) -> Numerics | None:
        """The run's numerics as its trainer reads them over `objective` (`TrainerAdvanced.numerics`): its
        precision, with the compute dtype its modules state as their own in place of its own
        (`stated_compute`: the record's where the run built them, `numerics_fields`, a handed model's own),
        and the parameters' dtype its model's record states, which the model then draws them in itself. None
        where its modules state no one compute dtype (Flax's None, a wrapper's, several among a multipart
        objective's), which no trainer establishes: the run then hands its trainer the loss scale and drawn
        dtype its record names (`trained_knobs`), as before numerics reached the trainer, its programs traced
        under its numerics all the same (`traced`)."""
        from dew.training.transaction import stated_compute

        computes = stated_compute(objective)
        if computes is None:
            return None
        precision = self.trainer.precision
        stated = self.model.fields.get("param_dtype")
        params = precision.dtypes.params if stated is None else str(stated)
        return dataclasses.replace(precision, dtypes=dataclasses.replace(
            precision.dtypes, compute=computes, params=params))

    def trained_knobs(self) -> tuple[bool, str | None]:
        """The loss scale and drawn dtype a trainer takes for a run whose numerics it cannot hold
        (`trained_numerics`): its numerics' loss scaling at its record's compute dtype (`compute_dtype`), and
        its parameters' dtype where that is not Flax's float32 and its model's record states none."""
        precision = self.trainer.precision
        drawn = None if "param_dtype" in self.model.fields else precision.drawn_dtype()
        return precision.scales_loss(self.compute_dtype()), drawn

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
    """A run over a loaded decoder, which may run a block of its layers in passes: the loop it runs
    (`CausalTransformer.loop`) and the pass counts its steps take, each the run's choice over the
    checkpoint's own. The language-model and the decision runs take it."""

    loop: Annotated[Loop | None, tyro.conf.Suppress] = None
    """The loop the decoder runs (`dew.nn.backbones.decoder_stack.Loop`: its block, passes, recompute, exits,
    θ and messages), in place of the checkpoint's own: a pretrained decoder with no loop gains it over its
    own layers, as Qwen 3.5's middle block repeated twice (`Loop(2, layers=(8, 16), step_norm=False)`),
    and a looped one's loop is replaced whole (`dataclasses.replace(its_loop, exit_gate=True)` changes
    one of its choices). A loop that gives its layers parameters of their own (an implicit loop's θ, an exit
    gate, the messages' merges) draws them at their zeros beside the loaded weights, so the decoder
    starts as the one loaded computes (`Pretrained.with_model`). None keeps the checkpoint's own."""
    _earlier_loop: Annotated[_LoopChange | None, tyro.conf.Suppress] = dataclasses.field(
        default=None, kw_only=True)
    """A saved run's changes to its checkpoint's loop, a choice at a time, as a Dew before the run stated its
    loop whole recorded them (`dew.config.earlier._LoopChange`), which its record alone holds
    (`dew.config.earlier_run`); a run states its `loop`. Carried to 1.0 (ledger 1552)."""
    passes: Annotated[PassPolicy | None, tyro.conf.Suppress] = None
    """The passes the loop runs each step (`PassPolicy`): one count (an int); one count a phase of the
    run's data (a tuple, `dew.data.phases`, `Trainer`'s `phase_passes`), each phase's step compiled with its
    own and the run handing on the last; or a count drawn afresh for each microbatch (`SampledPasses`,
    `Trainer`'s `sampled_passes`), the model holding and handing on its cap. Over the loop's own count
    (`loop`, else the checkpoint's), whose block, recompute, exits, θ and messages it keeps. None runs
    the loop's own count."""

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.loop is not None and self._earlier_loop is not None:
            raise ValueError("a run states its loop whole (loop), or a saved run's record its changes to its "
                             "checkpoint's (_earlier_loop), not both")
        if isinstance(self.passes, tuple) and self.data is not None:
            # Refused before anything loads: a count a phase the data reads in.
            phases = self.data.phase_count
            if len(self.passes) != phases:
                raise ValueError(f"passes names {len(self.passes)} pass counts, one a phase, and the run's "
                                 f"data reads {phases} phase(s)")

    @property
    def handed_passes(self) -> int | None:
        """The pass count the run's model holds and hands on: the last phase's, where the run names one a
        phase, a sampled count's cap, else the count the run states (`passes`); None runs the loop's own."""
        if isinstance(self.passes, SampledPasses):
            return self.passes.cap
        return self.passes[-1] if isinstance(self.passes, tuple) else self.passes

    def built[LossT, EffectsT](self, objective: Objective[LossT, EffectsT],
                               mesh: MeshSpec | jax.sharding.Mesh, rollout: Rollout | None,
                               **resources) -> "Trainer[LossT, EffectsT]":
        """`RunConfig.built`, its steps run each phase's pass count where the run names one a phase, or one
        drawn per microbatch (`passes`)."""
        phased = ({"phase_passes": self.passes} if isinstance(self.passes, tuple) else
                  {"sampled_passes": self.passes} if isinstance(self.passes, SampledPasses) else {})
        return super().built(objective, mesh, rollout, **phased, **resources)

    @property
    def loops(self) -> bool:
        """Whether the run asks anything of a loop (`looped`)."""
        return self.loop is not None or self._earlier_loop is not None or self.passes is not None

    def exit_settings(self) -> dict[str, float]:
        """The exit objective's weight an objective that trains a gate the run's loop draws (`Loop.exit_gate`)
        takes: `EXIT_ENTROPY` (`dew.nn.backbones.looping`) where the run's objective states none, none
        derived where it states one; refused where it states None, which would leave the gate at its draw
        (a hazard of one half every pass), trained by nothing. None where the run states no gated loop."""
        from dew.nn.backbones.looping import EXIT_ENTROPY

        held = self.loop if self.loop is not None else self._earlier_loop
        if held is None or not held.exit_gate:
            return {}
        stated = {} if self.objective is None else self.objective.fields
        if "exit_entropy" not in stated:
            return {"exit_entropy": EXIT_ENTROPY}
        if stated["exit_entropy"] is None:
            raise ValueError("the run's loop draws an exit gate the objective's exit_entropy trains, and the "
                             "run sets it to None, which trains nothing: state a weight, or leave it out for "
                             f"{EXIT_ENTROPY}")
        return {}

    def _resumed_as(self) -> Self:
        """`RunConfig._resumed_as`: its loop's passes recomputed in its backward or held (`Loop.remat`),
        memory alone, read as held; and a saved run's change to its checkpoint's loop (`_earlier_loop`) so,
        none where it changed that alone."""
        loop = None if self.loop is None else dataclasses.replace(self.loop, remat=False)
        earlier = (None if self._earlier_loop is None
                   else dataclasses.replace(self._earlier_loop, remat=False))
        return dataclasses.replace(self, loop=loop,
                                   _earlier_loop=None if earlier == _LoopChange() else earlier)

    def looped(self, start: "Pretrained") -> "Pretrained":
        """`start` with its block run as the run asks: under the run's loop (`loop`), at the run's pass
        count (`handed_passes`), each over the checkpoint's own where the run states none; as loaded where
        it asks nothing. A run's loop with a θ row a later pass (`Loop.theta_passes`) over a checkpoint
        holding one θ reads that θ in every row, so the loop computes what the checkpoint's did."""
        if not self.loops:
            return start
        from dew.nn.protocols import Looped, Looping

        if self._earlier_loop is not None:
            # A saved run's loop changed a choice at a time, as an earlier Dew recorded it (`earlier_run`).
            return self._earlier_loop.applied(start, self.passes, self.trainer.key)
        model = start.model
        if not isinstance(model, Looping):
            raise ValueError(f"this {type(model).__name__} runs no loop: a run's loop or passes need a "
                             "decoder that loops its layers")
        if self.loop is not None:
            model = model.with_loop(self.loop)
            if self.loop.theta_passes is not None:
                start = dataclasses.replace(start, variables=_theta_rows(start.variables,
                                                                         self.loop.theta_passes))
        counted = model if self.handed_passes is None else model.with_passes(self.handed_passes)
        unlooped = isinstance(model, Looped) and model.loop is None
        if counted is None or unlooped:
            raise ValueError(f"this {type(model).__name__} has no loop: state the run's loop (`Loop`, its "
                             "block of layers and passes) beside its passes")
        return start.with_model(counted, key=self.trainer.key)


def _theta_rows(variables, passes: int):
    """`variables` with each implicit loop's one θ (`loop_theta` `[d]`) a row for each later pass of
    `passes` (`[passes - 1, d]`), every row that θ, which computes what the one θ did (`Loop.theta_passes`);
    a θ that holds rows already, as it is."""
    import jax.numpy as jnp

    def rowed(path, leaf):
        named = isinstance(path[-1], jax.tree_util.DictKey) and path[-1].key == "loop_theta"
        if not named or len(leaf.shape) != 1:
            return leaf
        shape = (passes - 1, *leaf.shape)
        # A source held as shapes alone (`weights=False`, a resume's) rows its shape, its channels placed as
        # its one θ's were and its rows whole.
        if isinstance(leaf, jax.ShapeDtypeStruct):
            placed = leaf.sharding
            if isinstance(placed, jax.sharding.NamedSharding):
                placed = jax.sharding.NamedSharding(placed.mesh,
                                                    jax.sharding.PartitionSpec(None, *placed.spec))
            return jax.ShapeDtypeStruct(shape, leaf.dtype, sharding=placed)
        return jnp.broadcast_to(leaf, shape)

    return jax.tree_util.tree_map_with_path(rowed, variables)


__all__ = ["DecoderRunConfig", "Diagnostics", "LowerTarget", "MetricConfig", "ObjectiveConfig", "Prepared",
           "Release", "RunConfig", "SourceOptions", "Suite", "TrainerConfig",
           "flag_selected"]
