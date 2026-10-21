"""A chain's record, version 1: what a run's record holds for its data, from which the same chain is
built again (`encode`, `decode`). A store and each stage are class records, `{"class": <import
path>, "fields": {...}}`, written and read by the registry against the class's own constructor; a
mixture is its corpora's records with their weights, phases theirs with the steps they end at.

A chain no record carries whole is `Unrecorded`, with the reason: a step that is a lambda or a bound
method, a store whose records are arrays, any value that would come back as something else."""

from __future__ import annotations

import dataclasses
import inspect
import json
import logging
import types
from collections.abc import Callable, Mapping
from enum import Enum
from typing import TYPE_CHECKING, TypeGuard

import numpy as np

from dew.data.pipeline import Mixture, Phases, Piped, Pipeline, Stage
from dew.data.sources.base import READS, Read, Source
from dew.records import JSON, integer, json_value, numbers, record
from dew.registry import Configured, argument_records, import_path, imported, read_arguments

if TYPE_CHECKING:
    from dew.data.dataset import Batch, Loading
    from dew.data.pipeline import Dataset

VERSION = 1

_log = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class Unrecorded:
    """A chain no record carries whole, and why: a run trains on it from Python, and its record
    says so (`{"unrecorded": reason}`) instead of naming a chain it could not rebuild."""

    reason: str


class _Unrecordable(ValueError):
    """Raised inside `encode` where a value can't be recorded, with the reason `Unrecorded` keeps."""


def encode(dataset: Dataset) -> dict[str, JSON] | Unrecorded:
    """The record of `dataset`'s chain, from which `decode` builds the same chain, or `Unrecorded`
    where a part of it can't be recorded and read back as it is."""
    try:
        return {"version": VERSION, **_chain(dataset._chain())}
    except _Unrecordable as unrecordable:
        return Unrecorded(str(unrecordable))


def decode(written: Mapping[str, object]) -> Pipeline:
    """The chain `encode` recorded."""
    version = written.get("version")
    if version != VERSION:
        raise ValueError(f"a chain's record of version {version!r}; this Dew reads version {VERSION}")
    return _read_chain(written)


def override(written: Mapping[str, object], path: str, value: JSON) -> dict[str, JSON]:
    """`written` with one setting of one call of `pipe` in it set to `value`, parsed JSON: `path` is
    "<preset>.<setting>", the preset by its name where the chain calls it once, or by its stage's
    place among the chain's stages ("2.seq_len"), its store's call as "source". Reading the record
    back makes the call again with the setting, so the stages it makes follow it."""
    target, _, setting = path.rpartition(".")
    if not target or not setting:
        raise ValueError(f"an override names a call and its setting, `<preset>.<setting>`, not {path!r}")
    changed = record(json.loads(json.dumps(written)), "chain")
    source = record(changed["source"], "chain source")
    stages = _entries(changed.get("stages", []), "stages")
    calls = [(str(index), record(stage, "stage")) for index, stage in enumerate(stages)
             if isinstance(stage, Mapping) and "pipe" in stage]
    if "pipe" in source:
        calls.append(("source", source))
    named = [call for label, call in calls if target in (label, str(call["pipe"]).rpartition(".")[2],
                                                          str(call["pipe"]).rpartition(":")[2])]
    if len(named) != 1:
        found = sorted(str(call["pipe"]) for _, call in calls)
        raise ValueError(f"an override names {target!r}, which names {len(named)} of the chain's calls "
                         f"{found}")
    (call,) = named
    function = imported(str(call["pipe"]))
    parameters = inspect.signature(function).parameters
    if setting not in parameters and not any(parameter.kind is inspect.Parameter.VAR_KEYWORD
                                             for parameter in parameters.values()):
        raise ValueError(f"{call['pipe']} takes no setting {setting!r}; it takes {sorted(parameters)[1:]}")
    settings = dict(record(call.get("settings", {}), "settings"))
    settings[setting] = _value(value, setting)
    updated = {**call, "settings": settings}
    overridden = ({**changed, "source": updated} if call is source
                  else {**changed, "stages": [updated if stage is call else stage for stage in stages]})
    written_back = json_value(overridden, "chain")
    if not isinstance(written_back, dict):
        raise ValueError("a chain's record is a mapping")
    return written_back


def _chain(pipeline: Pipeline) -> dict[str, JSON]:
    """`pipeline`'s record: each call of `pipe` that made its stages (`Piped`) written as the call
    where reading it back makes the same stages, the outermost first, and every other stage as
    itself."""
    from dew.data.dataset import Loading

    loading = pipeline._loading
    source, index = _source(pipeline.source), 0
    made = sorted(((entry, entry.input) for entry in pipeline.piped if entry.input is not None),
                  key=lambda pair: -pair[0].end)
    for entry, input in made:
        invocation = _invocation(entry, input, pipeline)
        if invocation is not None:
            source, index = {**invocation, "input": _chain(input)}, entry.end
            break
    stages: list[JSON] = []
    while index < len(pipeline.stages):
        calls = sorted((entry for entry in pipeline.piped
                        if entry.input is None and entry.start == index and entry.end > index),
                       key=lambda e: -e.end)
        prefix = Pipeline(pipeline.source, pipeline.stages[:index])
        chosen = next(((entry, invocation) for entry in calls
                       if (invocation := _invocation(entry, prefix, pipeline)) is not None), None)
        if chosen is None:
            stage = pipeline.stages[index]
            stages.append(_class_record(stage, type(stage), f"stage {index}"))
            index += 1
        else:
            stages.append(chosen[1])
            index = chosen[0].end
    return {
        "source": source,
        "name": pipeline.name,
        "read": pipeline.mode,
        "loading": None if loading is None else _class_record(loading, Loading, "loading"),
        "stages": stages,
    }


def _invocation(entry: Piped, input: Pipeline, pipeline: Pipeline) -> dict[str, JSON] | None:
    """`entry`'s call as a record, None where it can't stand for what it made: its function no
    record names, an argument no record carries, or a call that, made again from what it read,
    makes other stages or another store (a mutable argument changed since)."""
    try:
        path = import_path(entry.function)
        args = [_value(argument, f"argument {index}") for index, argument in enumerate(entry.args)]
        settings = {name: _value(setting, name) for name, setting in entry.settings}
        made = entry.function(input, *(_read_value(arg) for arg in args),
                              **{name: _read_value(setting) for name, setting in settings.items()})
    except (TypeError, ValueError) as refused:
        # `_Unrecordable` among them: the call is written as the stages it made instead.
        _log.info("the call of %s is recorded as the stages it made: %s", entry.function, refused)
        return None
    if not isinstance(made, Pipeline):
        return None
    if entry.input is None:
        same = (made.source is pipeline.source and len(made.stages) == entry.end
                and _same(made.stages[entry.start:], pipeline.stages[entry.start:entry.end], "call") is None)
    else:
        same = (len(made.stages) == entry.end
                and _same(made.source, pipeline.source, "call", sequences=True) is None
                and _same(made.stages, pipeline.stages[:entry.end], "call") is None)
    return {"pipe": path, "args": args, "settings": settings} if same else None


def _value(value: object, where: str) -> JSON:
    """A call's argument as JSON that says what it is: a scalar or a list as it is, a tuple, a
    mapping, a dataclass (a class record) or a module-level function by name."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, list):
        return [_value(entry, where) for entry in value]
    if isinstance(value, tuple):
        return {"tuple": [_value(entry, where) for entry in value]}
    if isinstance(value, Mapping) and all(isinstance(key, str) for key in value):
        return {"mapping": {str(key): _value(entry, where) for key, entry in value.items()}}
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _class_record(value, type(value), where)
    if isinstance(value, types.FunctionType):
        return _function(value, where)
    raise _Unrecordable(f"{where} is a {type(value).__name__}, which no record carries")


def _read_value(written: object) -> Configured:
    """A call's argument from what `_value` wrote."""
    if written is None or isinstance(written, (bool, int, float, str)):
        return written
    if isinstance(written, list):
        return [_read_value(entry) for entry in written]
    if not isinstance(written, Mapping):
        raise ValueError(f"a call's argument is written as a {type(written).__name__}, which no record "
                         "writes")
    if "tuple" in written:
        return tuple(_read_value(entry) for entry in _entries(written["tuple"], "tuple"))
    if "mapping" in written:
        return {key: _read_value(entry) for key, entry in record(written["mapping"], "mapping").items()}
    if "function" in written:
        function = imported(str(written["function"]))
        if not isinstance(function, types.FunctionType):
            raise ValueError(f"{written['function']} names no function")
        return function
    return _read_class(record(written, "argument"))


def _source(source: Source | Mixture | Phases) -> dict[str, JSON]:
    if isinstance(source, Mixture):
        counted = source.tokens
        where = "a mixture's token count"
        tokens = (None if counted is None else _class_record(counted, type(counted), where)
                  if dataclasses.is_dataclass(counted) else _function(counted, where))
        return {"mix": [_chain(child) for child in source.pipelines], "weights": list(source.weights),
                "tokens": tokens}
    if isinstance(source, Phases):
        return {"phases": [_chain(child) for child in source.pipelines],
                "until_steps": list(source.until_steps)}
    return _class_record(source, type(source), "store", store=True)


def _class_record(value: object, member: type, where: str, *, store: bool = False) -> dict[str, JSON]:
    """`value`, an instance of the dataclass `member`, as a class record, refused where reading it
    back builds anything but an equal value (`_same`). A store is equal where it reads the same
    records, its fingerprint the same, whichever sequence holds them."""
    if not dataclasses.is_dataclass(value) or isinstance(value, type):
        raise _Unrecordable(f"the {where} is a {type(value).__name__}, which no record carries")
    fields = {field.name: getattr(value, field.name) for field in dataclasses.fields(value) if field.init}
    try:
        path = import_path(member)
        written = argument_records(member, fields)
        rebuilt = member(**read_arguments(member, written))
    except (TypeError, ValueError, NameError) as refused:
        raise _Unrecordable(f"the {where} ({type(value).__name__}) can't be recorded: {refused}") from None
    difference = _same(value, rebuilt, where, sequences=store)
    if difference is None and store and isinstance(value, Source) and isinstance(rebuilt, Source):
        same = json_value(value.fingerprint(), where) == json_value(rebuilt.fingerprint(), where)
        difference = None if same else f"{where} reads back other records"
    if difference is not None:
        raise _Unrecordable(f"the {where} ({type(value).__name__}) reads back otherwise: {difference}")
    return {"class": path, "fields": written}


def _function(function: Callable[[Batch], int], where: str) -> dict[str, JSON]:
    try:
        path = import_path(function)
    except ValueError as refused:
        raise _Unrecordable(f"{where} can't be recorded: {refused}") from None
    if imported(path) is not function:
        raise _Unrecordable(f"{where} is not the function its import path names ({path})")
    return {"function": path}


def _same(original: object, rebuilt: object, where: str, *, sequences: bool = False) -> str | None:
    """Where `rebuilt` differs from `original`, None where it is the same value: of the same type at
    every level, a tuple a tuple and a list a list (any sequence the other with `sequences`), an
    array of the same dtype and values, a function or a class the same object (a bound method read
    back by its import path is not). A value no record carries is refused."""
    alike = sequences and isinstance(original, (list, tuple)) and isinstance(rebuilt, (list, tuple))
    if type(original) is not type(rebuilt) and not alike:
        return f"{where} is a {type(original).__name__} and reads back a {type(rebuilt).__name__}"
    if isinstance(original, np.ndarray) and isinstance(rebuilt, np.ndarray):
        same = original.dtype == rebuilt.dtype and np.array_equal(original, rebuilt)
        return None if same else f"{where} reads back other values"
    if dataclasses.is_dataclass(original) and not isinstance(original, type):
        differences = (_same(getattr(original, field.name), getattr(rebuilt, field.name),
                             f"{where}.{field.name}", sequences=sequences)
                       for field in dataclasses.fields(original))
        return next((difference for difference in differences if difference is not None), None)
    if isinstance(original, types.FunctionType) and isinstance(rebuilt, types.FunctionType):
        # A function a call makes afresh is the same where its code and what it closes over are.
        if original is rebuilt:
            return None
        if original.__code__ is not rebuilt.__code__:
            return f"{where} reads back another function"
        cells = [cell.cell_contents for cell in original.__closure__ or ()]
        again = [cell.cell_contents for cell in rebuilt.__closure__ or ()]
        return _same((cells, original.__defaults__), (again, rebuilt.__defaults__), where,
                     sequences=sequences)
    if isinstance(original, (type, types.BuiltinFunctionType, types.MethodType)):
        return None if original is rebuilt else f"{where} reads back another {type(original).__name__}"
    if isinstance(original, Mapping) and isinstance(rebuilt, Mapping):
        if list(original) != list(rebuilt):
            return f"{where} reads back other keys"
        differences = (_same(value, rebuilt[key], f"{where}[{key!r}]", sequences=sequences)
                       for key, value in original.items())
        return next((difference for difference in differences if difference is not None), None)
    if isinstance(original, (list, tuple)) and isinstance(rebuilt, (list, tuple)):
        if len(original) != len(rebuilt):
            return f"{where} reads back {len(rebuilt)} entries, not {len(original)}"
        differences = (_same(left, right, f"{where}[{index}]", sequences=sequences)
                       for index, (left, right) in enumerate(zip(original, rebuilt, strict=True)))
        return next((difference for difference in differences if difference is not None), None)
    if isinstance(original, (str, bytes, int, float, complex, bool, Enum, np.generic)) or original is None:
        return None if original == rebuilt else f"{where} reads back {rebuilt!r}, not {original!r}"
    raise _Unrecordable(f"{where} holds a {type(original).__name__}, which no record carries")


def _read_chain(written: Mapping[str, object]) -> Pipeline:
    chain = record(written, "chain")
    loading = chain.get("loading")
    source = record(chain["source"], "chain source")
    built = (_read_call(_read_chain(record(source["input"], "a call's input")), source) if "pipe" in source
             else Pipeline(_read_source(source)))
    for written_stage in _entries(chain.get("stages", []), "stages"):
        stage = record(written_stage, "stage")
        built = _read_call(built, stage) if "pipe" in stage else built._then(_read_stage(stage))
    return dataclasses.replace(
        built,
        name=None if chain.get("name") is None else str(chain["name"]),
        mode=_read_mode(chain.get("read")),
        _loading=None if loading is None else _read_loading(record(loading, "loading")),
    )


def _read_call(input: Pipeline, written: Mapping[str, object]) -> Pipeline:
    """`input` through the call of `pipe` a record wrote, made again (`Piped`)."""
    function = imported(str(written["pipe"]))
    if not _preset(function):
        raise ValueError(f"{written['pipe']} names no function a chain is piped through")
    made = input.pipe(function, *(_read_value(arg) for arg in _entries(written.get("args", []), "args")),
                      **{name: _read_value(setting)
                         for name, setting in record(written.get("settings", {}), "settings").items()})
    if not isinstance(made, Pipeline):
        raise ValueError(f"{written['pipe']} made no chain")
    return made


def _preset(value: Configured) -> TypeGuard[Callable[..., Dataset]]:
    return isinstance(value, types.FunctionType)


def _read_loading(written: Mapping[str, object]) -> Loading:
    from dew.data.dataset import Loading

    built = _read_class(written)
    if not isinstance(built, Loading):
        raise ValueError(f"a chain's loading names {written['class']}, which is no Loading")
    return built


def _read_source(written: Mapping[str, object]) -> Source | Mixture | Phases:
    if "mix" in written:
        tokens = None if written.get("tokens") is None else record(written["tokens"], "tokens")
        counted = (None if tokens is None else _read_class(tokens) if "class" in tokens
                   else imported(str(tokens["function"])))
        if counted is not None and not _counts(counted):
            raise ValueError(f"a mixture's token count names {counted!r}, which is no function")
        corpora = _entries(written["mix"], "mix")
        return Mixture(tuple(_read_chain(record(child, "corpus")) for child in corpora),
                       numbers(written["weights"], "weights"), counted)
    if "phases" in written:
        until = _entries(written["until_steps"], "until_steps")
        phases = _entries(written["phases"], "phases")
        return Phases(tuple(_read_chain(record(child, "phase")) for child in phases),
                      tuple(None if step is None else integer(step, "until_steps") for step in until))
    built = _read_class(written)
    if not isinstance(built, Source):
        raise ValueError(f"a chain's store names {written['class']}, which is no store")
    return built


def _read_stage(written: Mapping[str, object]) -> Stage:
    built = _read_class(written)
    if not _stage(built):
        raise ValueError(f"a chain's stage names {written['class']}, which is no stage")
    return built


def _stage(value: Configured) -> TypeGuard[Stage]:
    return isinstance(value, Stage.__value__)


def _counts(value: Configured) -> TypeGuard[Callable[[Batch], int]]:
    """Whether `value` is a function, as a mixture's token count is: a record's tokens."""
    return callable(value)


def _read_class(written: Mapping[str, object]) -> Configured:
    member = imported(str(written["class"]))
    return member(**read_arguments(member, record(written.get("fields", {}), "fields")))


def _entries(value: object, where: str) -> list[JSON]:
    if not isinstance(value, list):
        raise ValueError(f"a chain's {where} is a list, not {type(value).__name__}")
    return [json_value(entry, where) for entry in value]


def _read_mode(value: object) -> Read | None:
    if value is None:
        return None
    if not isinstance(value, str) or value not in READS:
        raise ValueError(f"a chain reads memory, indexed or streamed, not {value!r}")
    return READS[READS.index(value)]
