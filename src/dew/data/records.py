"""A chain's record, version 1: what a run's record holds for its data, from which the same chain is
built again (`encode`, `decode`). A store and each stage are class records, `{"class": <import
path>, "fields": {...}}`, written and read by the registry against the class's own constructor; a
mixture is its corpora's records with their weights, phases theirs with the steps they end at.

A chain no record carries whole is `Unrecorded`, with the reason: a step that is a lambda or a bound
method, a store whose records are arrays, any value that would come back as something else."""

from __future__ import annotations

import dataclasses
import types
from collections.abc import Callable, Mapping
from enum import Enum
from typing import TYPE_CHECKING, TypeGuard

import numpy as np

from dew.data.pipeline import Mixture, Phases, Pipeline, Stage
from dew.data.sources.base import READS, Read, Source
from dew.records import JSON, integer, json_value, numbers, record
from dew.registry import Configured, argument_records, import_path, imported, read_arguments

if TYPE_CHECKING:
    from dew.data.dataset import Batch, Loading
    from dew.data.pipeline import Dataset

VERSION = 1


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


def _chain(pipeline: Pipeline) -> dict[str, JSON]:
    from dew.data.dataset import Loading

    loading = pipeline._loading
    return {
        "source": _source(pipeline.source),
        "name": pipeline.name,
        "read": pipeline.mode,
        "loading": None if loading is None else _class_record(loading, Loading, "loading"),
        "stages": [_class_record(stage, type(stage), f"stage {index}")
                   for index, stage in enumerate(pipeline.stages)],
    }


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
    if isinstance(original, (type, types.FunctionType, types.BuiltinFunctionType, types.MethodType)):
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
    return Pipeline(
        _read_source(record(chain["source"], "chain source")),
        tuple(_read_stage(record(stage, "stage")) for stage in _entries(chain.get("stages", []), "stages")),
        name=None if chain.get("name") is None else str(chain["name"]),
        mode=_read_mode(chain.get("read")),
        _loading=None if loading is None else _read_loading(record(loading, "loading")),
    )


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
