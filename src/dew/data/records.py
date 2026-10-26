"""A chain's record, version 1: what a run's record holds for its data, from which the same chain is
built again (`encode`, `decode`). A store, each stage and the loading are written and read by the
registry's one typed walk (`dew.registry.to_record`, `from_record`) against what the chain declares
them as: a store, a stage, a Loading. So a record builds only those, of Dew's classes or a trusted
package's (`trust=`), and nothing else it names runs first. A call of `pipe` is written as the call,
`{"pipe": <import path>, "settings": {...}}`, its arguments bound to the function's parameters and
walked against their annotations alike; a mixture is its corpora's records with their weights and
token count, phases theirs with the steps they end at.

A chain no record carries whole is `Unrecorded`, with the reason: a step that is a lambda, a store
whose records are arrays, any value that would come back as something else."""

from __future__ import annotations

import dataclasses
import inspect
import json
import logging
import types
import typing
from collections.abc import Callable, Iterable, Mapping, Sequence
from enum import Enum
from typing import Literal, TypeGuard, Union

import numpy as np

from dew._records import JSON, integer, json_value, numbers, record
from dew.batch import Batch
from dew.data.dataset import Loading
from dew.data.pipeline import Dataset, Mixture, Phases, Piped, Pipeline, Stage
from dew.data.sources.base import READS, Read, Source
from dew.registry import (
    Annotation,
    Configured,
    argument_records,
    class_record,
    declared_type,
    from_record,
    import_path,
    import_trusted,
    imported,
    parameter_type,
    read_arguments,
    to_record,
)

VERSION = 1

_log = logging.getLogger(__name__)

# What making a call of `pipe` again can raise, the function being the caller's: a call that raises
# one is written as the stages it made instead (`_invocation`).
_REPLAY_ERRORS = (TypeError, ValueError, LookupError, OSError, ImportError)


@dataclasses.dataclass(frozen=True)
class Unrecorded:
    """A chain no record carries whole, and why: a run trains on it from Python, and its record
    says so (`{"unrecorded": reason}`) instead of naming a chain it could not rebuild."""

    reason: str


class _Unrecordable(ValueError):
    """Raised inside `encode` where a value can't be recorded, with the reason `Unrecorded` keeps."""


def encode(dataset: Dataset, *, trust: Sequence[str] = ()) -> dict[str, JSON] | Unrecorded:
    """The record of `dataset`'s chain, from which `decode` builds the same chain, or `Unrecorded`
    where a part of it can't be recorded and read back as it is. Each part is read back as it is
    written, through JSON, and compared; each call of `pipe` is made again from its recorded
    arguments, so a preset that reads files reads them again here. A class of a package outside Dew
    is read back only where `trust` names the package, as a run's record trusts it."""
    import_trusted(None, trust)
    try:
        return {"version": VERSION, **_chain(dataset._chain())}
    except _Unrecordable as unrecordable:
        return Unrecorded(str(unrecordable))


def decode(written: Mapping[str, object], *, trust: Sequence[str] = ()) -> Pipeline:
    """The chain `encode` recorded, building the classes of a package outside Dew only where
    `trust` names it."""
    version = written.get("version")
    if version != VERSION:
        raise ValueError(f"a chain's record of version {version!r}; this Dew reads version {VERSION}")
    import_trusted(json_value(dict(written), "chain"), trust)
    return _read_chain(written)


def override(written: Mapping[str, object], path: str, value: JSON) -> dict[str, JSON]:
    """`written` with one setting of one call of `pipe`, or of its store, set to `value`, parsed JSON
    and read against the setting's annotation. `path` is "<preset>.<setting>", the preset by its
    name where the chain's own stages, or the call that made its store, call it once, or by its
    entry's place among the record's stages ("2.seq_len"); or "source.<setting>" for what the chain
    reads (`_source_with`). A call inside a mix or phases is not addressed here. Reading the record
    back makes the call again, so the stages it makes follow it."""
    version = written.get("version")
    if version != VERSION:
        raise ValueError(f"a chain's record of version {version!r}; this Dew overrides version {VERSION}")
    target, _, setting = path.rpartition(".")
    if not target or not setting:
        raise ValueError(f"an override names a call and its setting, `<preset>.<setting>`, not {path!r}")
    changed = record(json.loads(json.dumps(written)), "chain")
    if target == "source":
        overridden = _source_with(changed, setting, value)
    else:
        stages = _entries(changed.get("stages", []), "stages")
        source = record(changed["source"], "chain source")
        calls = [(str(index), record(stage, "stage")) for index, stage in enumerate(stages)
                 if isinstance(stage, Mapping) and "pipe" in stage]
        if "pipe" in source:
            # The call that made the store, as token_windows makes its windows, by its name alike.
            calls.append(("source", source))
        named = [call for label, call in calls if target in (label, str(call["pipe"]).rpartition(":")[2])]
        if len(named) != 1:
            found = [f"{str(call['pipe']).rpartition(':')[2]}@{label}" for label, call in calls]
            raise ValueError(f"an override names {target!r}, which names {len(named)} of the chain's calls "
                             f"{found}; name one by its place, `<index>.{setting}`")
        (call,) = named
        updated = _call_with(call, setting, value)
        if updated is None:
            raise ValueError(f"{call['pipe']} takes no setting {setting!r}; it takes "
                             f"{list(inspect.signature(_function(call['pipe'])).parameters)[1:]}")
        overridden = ({**changed, "source": updated} if call is source
                      else {**changed, "stages": [updated if stage is call else stage for stage in stages]})
    written_back = json_value(overridden, "chain")
    if not isinstance(written_back, dict):
        raise ValueError("a chain's record is a mapping")
    return written_back


def _call_with(call: Mapping[str, object], setting: str, value: JSON) -> Mapping[str, object] | None:
    """`call`, a call of `pipe`, with its `setting` set to `value`, read against the setting's
    annotation; None where its function takes no such setting."""
    function = _function(call["pipe"])
    parameters = inspect.signature(function).parameters
    reads = next(iter(parameters))
    if setting == reads:
        raise ValueError(f"{call['pipe']}'s {reads} is the dataset it reads, which no override sets")
    if setting not in parameters and not any(parameter.kind is inspect.Parameter.VAR_KEYWORD
                                             for parameter in parameters.values()):
        return None
    # The typed walk reads a record or a list against the setting's annotation, refusing one it
    # doesn't build; `_fits` checks what a scalar or a container is.
    read_arguments(function, {setting: value})
    annotation = parameter_type(function, setting) if setting in parameters else None
    if not _fits(value, annotation):
        raise ValueError(f"{call['pipe']}'s {setting} is {annotation}, not {value!r}")
    return {**call, "settings": {**record(call.get("settings", {}), "settings"), setting: value}}


def _source_with(chain: Mapping[str, object], setting: str, value: JSON) -> Mapping[str, object]:
    """`chain`'s record with `setting` set on what it reads, one rule at every depth: the call that
    made its store where that call takes the setting, which so shadows a store's field of its name;
    else, through that call's input, the chain it read; else its store's field (`_store_with`)."""
    source = record(chain["source"], "chain source")
    if "pipe" not in source:
        return {**chain, "source": _store_with(source, setting, value)}
    updated = _call_with(source, setting, value)
    if updated is None:
        if "input" not in source:
            raise ValueError(f"{source['pipe']} made the chain's store, and takes no setting {setting!r}")
        updated = {**source, "input": _source_with(record(source["input"], "a call's input"), setting, value)}
    return {**chain, "source": updated}


def _store_with(store: Mapping[str, object], setting: str, value: JSON) -> Mapping[str, object]:
    """`store`, a store's class record, with its field `setting` set to `value`, read against the
    field's annotation; where its class has no such field, the store it selects from (its own
    `source`, as a holdout's selection holds it) with it, and so on down. This is the boundary where
    a record's JSON meets a store's fields, so it asks the value what it is."""
    named = class_record(store)
    if named is None:
        raise ValueError(f"a chain's store is a class record, {{'class': ..., 'fields': {{...}}}}, "
                         f"not {store!r}")
    name, fields = named
    kind = imported(name)
    if not isinstance(kind, type) or not dataclasses.is_dataclass(kind):
        raise ValueError(f"a chain's store names {name}, which is no store's class")
    declared = [field.name for field in dataclasses.fields(kind) if field.init]
    if setting in declared:
        annotation = declared_type(kind, setting)
        if annotation is not None and isinstance(value, (dict, list)):
            # The typed walk refuses a record or a list it doesn't build; `_fits` checks a scalar.
            from_record(_annotated(annotation), value)
        if not _fits(value, annotation):
            raise ValueError(f"{name}'s {setting} is {annotation}, not {value!r}")
        return {"class": name, "fields": {**fields, setting: value}}
    below = fields.get("source")
    if "source" not in declared or not isinstance(below, Mapping):
        raise ValueError(f"{name} has no setting {setting!r}, nor a store below it that has; it has "
                         f"{declared}")
    return {"class": name, "fields": {**fields, "source": _store_with(below, setting, value)}}


def _fits(value: JSON, annotation: Annotation) -> bool:
    """Whether a setting a record holds, JSON, is of the type its annotation declares, as far as a
    scalar shows it, and a list a sequence's or a mapping a mapping's or a class's, which the typed
    walk read against it. This is the
    boundary where a record's JSON meets a function's parameters, so it asks each value its type."""
    if annotation is None:
        return True
    if isinstance(annotation, typing.TypeAliasType):
        return _fits(value, annotation.__value__)
    origin = typing.get_origin(annotation)
    if origin in (Union, types.UnionType):
        return any(_fits(value, member) for member in typing.get_args(annotation))
    if origin is Literal:
        return value in typing.get_args(annotation)
    if isinstance(value, list):
        return (origin or annotation) in (list, tuple, Sequence, Iterable)
    if isinstance(value, dict):
        # A mapping, or a class's record, which the typed walk read against the class.
        return (origin or annotation) in (dict, Mapping) or (
            isinstance(annotation, type) and annotation not in (str, bytes, int, float, bool, type(None)))
    if annotation is type(None):
        return value is None
    if annotation is float:
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if annotation is int:
        return isinstance(value, int) and not isinstance(value, bool)
    if annotation in (str, bool):
        return isinstance(value, annotation)
    # A class or a protocol it declares is a record, which the typed walk read already.
    return value is not None


def _chain(pipeline: Pipeline) -> dict[str, JSON]:
    """`pipeline`'s record: each call of `pipe` that made its stages (`Piped`) written as the call
    where making it again from its recorded arguments makes the same stages, the outermost of
    calls over the same stages first, and every other stage as itself."""
    loading = pipeline._loading
    source, index = _source(pipeline.source), 0
    # `pipe` keeps the calls it made inside a call before the call itself: the later, the outer.
    ranked = sorted(enumerate(pipeline.piped), key=lambda pair: (-pair[1].end, -pair[0]))
    for _, entry in ranked:
        if entry.input is not None:
            invocation = _invocation(entry, entry.input, pipeline)
            if invocation is not None:
                source, index = {**invocation, "input": _chain(entry.input)}, entry.end
                break
    stages: list[JSON] = []
    while index < len(pipeline.stages):
        calls = [entry for _, entry in ranked
                 if entry.input is None and entry.start == index and entry.end > index]
        prefix = Pipeline(pipeline.source, pipeline.stages[:index])
        chosen = next(((entry, invocation) for entry in calls
                       if (invocation := _invocation(entry, prefix, pipeline)) is not None), None)
        if chosen is None:
            stages.append(_typed(pipeline.stages[index], Stage, f"stage {index}"))
            index += 1
        else:
            stages.append(chosen[1])
            index = chosen[0].end
    return {
        "source": source,
        "name": pipeline.name,
        "read": pipeline.mode,
        "loading": None if loading is None else _typed(loading, Loading, "loading"),
        "stages": stages,
    }


def _invocation(entry: Piped, input: Pipeline, pipeline: Pipeline) -> dict[str, JSON] | None:
    """`entry`'s call as a record, None where it can't stand for what it made: a function no import
    path names as it is (a bound method, a callable object, a closure), arguments the registry
    can't write and read back against the function's parameters, or a call that, made again from
    those arguments, makes other stages or another store (an argument changed since)."""
    function = entry.function
    if not isinstance(function, types.FunctionType):
        return None
    try:
        path = import_path(function)
        if imported(path) is not function:
            return None
        signature = inspect.signature(function)
        bound = signature.bind(input, *entry.args, **dict(entry.settings))
        reads = next(iter(signature.parameters))
        arguments: dict[str, Configured] = {}
        for name, argument in bound.arguments.items():
            kind = signature.parameters[name].kind
            if kind is inspect.Parameter.VAR_POSITIONAL:
                return None
            if kind is inspect.Parameter.VAR_KEYWORD:
                arguments.update(argument)
            elif name != reads:
                arguments[name] = argument
        settings = argument_records(function, arguments)
        again = read_arguments(function, json.loads(json.dumps(settings)))
        made = function(input, **again)
        same = isinstance(made, Pipeline) and _makes(entry, made, pipeline)
    except _REPLAY_ERRORS as refused:
        _log.info("the call of %s is recorded as the stages it made: %s", function.__qualname__, refused)
        return None
    return {"pipe": path, "settings": settings} if same else None


def _makes(entry: Piped, made: Pipeline, pipeline: Pipeline) -> bool:
    """Whether `made`, `entry`'s call made again, is what it made in `pipeline`: the same stages from
    the same store, or, where it made another store, an equal one that reads the same records
    (`_reads_alike`)."""
    if entry.input is None:
        return (made.source is pipeline.source and len(made.stages) == entry.end
                and _same(made.stages[entry.start:], pipeline.stages[entry.start:entry.end], "call") is None)
    again, had = made.source, pipeline.source
    same_store = _same(again, had, "call", sequences=True) is None
    if same_store and isinstance(again, Source) and isinstance(had, Source):
        same_store = _reads_alike(had, again, "call") is None
    return (same_store and len(made.stages) == entry.end
            and _same(made.stages, pipeline.stages[:entry.end], "call") is None)


def _source(source: Source | Mixture | Phases) -> dict[str, JSON]:
    if isinstance(source, Mixture):
        tokens = (None if source.tokens is None
                  else _typed(source.tokens, declared_type(Mixture, "tokens"), "a mixture's token count"))
        return {"mix": [_chain(child) for child in source.pipelines], "weights": list(source.weights),
                "tokens": tokens}
    if isinstance(source, Phases):
        return {"phases": [_chain(child) for child in source.pipelines],
                "until_steps": list(source.until_steps)}
    written = _typed(source, Source, "store")
    if not isinstance(written, dict):
        raise _Unrecordable(f"the store ({type(source).__name__}) is written as no record of its class")
    return written


def _typed(value: object, annotation: Annotation, where: str) -> JSON:
    """`value` as the registry writes it against `annotation`, refused where reading that back
    through JSON builds anything but the same value (`_same`): a store is the same where its
    settings are, whichever sequence holds them, and it reads the same records (`_reads_alike`)."""
    try:
        written = to_record(value, annotation)
        rebuilt = from_record(_annotated(annotation), json.loads(json.dumps(written)))
    except (TypeError, ValueError, NameError) as refused:
        raise _Unrecordable(f"the {where} ({type(value).__name__}) can't be recorded: {refused}") from None
    difference = _same(value, rebuilt, where, sequences=isinstance(value, Source))
    if difference is None and isinstance(value, Source) and isinstance(rebuilt, Source):
        difference = _reads_alike(value, rebuilt, where)
    if difference is not None:
        raise _Unrecordable(f"the {where} ({type(value).__name__}) reads back otherwise: {difference}")
    return written


def _reads_alike(original: Source, rebuilt: Source, where: str) -> str | None:
    """Where `rebuilt`, read back from `original`'s record, reads other records than `original` read,
    by their fingerprints: a store that resolved its files, revision or version as it was read
    reads what it resolved, which its settings alone don't say. A store not here records by its
    settings, as a run's record is written where its data may not be."""
    try:
        same = json_value(original.fingerprint(), where) == json_value(rebuilt.fingerprint(), where)
    except FileNotFoundError as missing:
        _log.info("%s (%s) is not here, so its record is its settings: %s", where, original.describe(),
                  missing)
        return None
    return None if same else f"{where} reads back other records"


def _annotated(annotation: Annotation) -> type | types.UnionType | typing.TypeAliasType:
    """`annotation` as one the registry reads a class record against: a class, a union or an alias."""
    if not isinstance(annotation, (type, types.UnionType, typing.TypeAliasType)):
        raise ValueError(f"a record is read against a class, a union or an alias, not {annotation!r}")
    return annotation


def _counts(value: Configured) -> TypeGuard[Callable[[Batch], int]]:
    """Whether a mixture's token count read back is a function, as `Mixture.tokens` is."""
    return callable(value)


def _same(original: object, rebuilt: object, where: str, *, sequences: bool = False,
          field: bool = False) -> str | None:
    """Where `rebuilt` differs from `original`, None where it is the same value: of the same type at
    every level, a tuple a tuple and a list a list, an array of the same dtype and values, a
    function of the same code over the same values, a class or a method of the same object. With
    `sequences`, a dataclass's `field`, a setting, may be any sequence where it was another, as a
    store's settings may, at any depth (a concatenation's stores' settings too), but what a
    sequence or mapping holds keeps its own types. A value no record carries is refused."""

    alike = sequences and field and isinstance(original, (list, tuple)) and isinstance(rebuilt, (list, tuple))
    if type(original) is not type(rebuilt) and not alike:
        return f"{where} is a {type(original).__name__} and reads back a {type(rebuilt).__name__}"
    if isinstance(original, np.ndarray) and isinstance(rebuilt, np.ndarray):
        same = original.dtype == rebuilt.dtype and np.array_equal(original, rebuilt)
        return None if same else f"{where} reads back other values"
    if dataclasses.is_dataclass(original) and not isinstance(original, type):
        differences = (_same(getattr(original, setting.name), getattr(rebuilt, setting.name),
                             f"{where}.{setting.name}", sequences=sequences, field=True)
                       for setting in dataclasses.fields(original))
        return next((difference for difference in differences if difference is not None), None)
    if isinstance(original, types.FunctionType) and isinstance(rebuilt, types.FunctionType):
        # A function a call makes afresh is the same where its code and what it closes over are.
        if original is rebuilt:
            return None
        if original.__code__ is not rebuilt.__code__:
            return f"{where} reads back another function"
        cells = [cell.cell_contents for cell in original.__closure__ or ()]
        again = [cell.cell_contents for cell in rebuilt.__closure__ or ()]
        return _same((cells, original.__defaults__, original.__kwdefaults__),
                     (again, rebuilt.__defaults__, rebuilt.__kwdefaults__), where, sequences=sequences)
    if isinstance(original, types.MethodType) and isinstance(rebuilt, types.MethodType):
        same = original.__func__ is rebuilt.__func__ and original.__self__ is rebuilt.__self__
        return None if same else f"{where} reads back another method"
    if isinstance(original, (type, types.BuiltinFunctionType)):
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
    if "source" not in chain:
        raise ValueError("a chain's record names its store under \"source\"")
    source = record(chain["source"], "chain source")
    if "pipe" in source:
        built = _read_call(_read_chain(record(source.get("input"), "a call's input")), source)
    else:
        built = Pipeline(_read_source(source))
    for written_stage in _entries(chain.get("stages", []), "stages"):
        stage = record(written_stage, "stage")
        built = _read_call(built, stage) if "pipe" in stage else built._then(_read_stage(stage))
    loading = chain.get("loading")
    return dataclasses.replace(
        built,
        name=None if chain.get("name") is None else str(chain["name"]),
        mode=_read_mode(chain.get("read")),
        _loading=None if loading is None else from_record(Loading, json_value(loading, "loading")),
    )


def _read_call(input: Pipeline, written: Mapping[str, object]) -> Pipeline:
    """`input` through the call of `pipe` a record wrote, made again (`Piped`)."""
    function = _function(written.get("pipe"))
    made = input.pipe(function, **read_arguments(function, record(written.get("settings", {}), "settings")))
    if not isinstance(made, Pipeline):
        raise ValueError(f"{written['pipe']} made no chain")
    return made


def _function(written: object) -> types.FunctionType:
    """The function a call's record names by its import path."""
    if not isinstance(written, str):
        raise ValueError(f"a call names its function by its import path, not {written!r}")
    function = imported(written)
    if not isinstance(function, types.FunctionType):
        raise ValueError(f"{written} names no function a chain is piped through")
    return function


def _read_source(written: Mapping[str, object]) -> Source | Mixture | Phases:
    if "mix" in written:
        tokens = written.get("tokens")
        corpora = _entries(written["mix"], "mix")
        counting = _annotated(declared_type(Mixture, "tokens"))
        counted = None if tokens is None else from_record(counting, json_value(tokens, "tokens"))
        if counted is not None and not _counts(counted):
            raise ValueError("a mixture's token count reads back as no function")
        return Mixture(tuple(_read_chain(record(child, "corpus")) for child in corpora),
                       numbers(written.get("weights"), "weights"), counted)
    if "phases" in written:
        until = _entries(written.get("until_steps"), "until_steps")
        phases = _entries(written["phases"], "phases")
        return Phases(tuple(_read_chain(record(child, "phase")) for child in phases),
                      tuple(None if step is None else integer(step, "until_steps") for step in until))
    return from_record(Source, written)


def _read_stage(written: Mapping[str, object]) -> Stage:
    built = from_record(Stage, written)
    if not _stage(built):
        raise ValueError(f"a chain's stage names {written.get('class')}, which is no stage")
    return built


def _stage(value: Configured) -> TypeGuard[Stage]:
    return isinstance(value, Stage.__value__)


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
