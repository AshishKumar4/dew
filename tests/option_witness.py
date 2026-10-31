"""The option-witness gate (row 2285): each option a saved task is built with, derived from its type
(`alternatives`, `options_of`), passed through `dew train` to the task the run saves and witnessed there
(`witnessed_on`): held by the loaded task, or read by its answers to one request (`requested`)."""

import dataclasses
import inspect
import json
import types
import typing
from collections.abc import Mapping
from pathlib import Path

import jax
import numpy as np
from model_support import moved
from run_kinds import (
    BESIDE,
    blocked,
    blockwise,
    decision,
    diffusion,
    dreamed,
    lettered,
    lm,
    masked,
    merged,
    prefixed,
    requested,
    trained_by_cli,
)

from dew import registry
from dew.cli.main import main
from dew.config import RunConfig
from dew.decision.layout import QuestionSegment, Reads, built_in
from dew.registry import objectives


def members_of(annotation: type, base=None) -> list[object]:
    """The values of Dew's classes that are `annotation`s: every member of a registry kind
    (`registry.KINDS`) and every class its package exports (`dew.sampling`'s CFG, APG, ...) that derives
    from `annotation`, or, for a protocol, holds each method it declares, each built from its defaults,
    its required fields `base`'s of those names, the value a run saved, which one without them lacks."""
    import importlib

    package = importlib.import_module(".".join(annotation.__module__.split(".")[:2]))
    found = [registry.imported(path) for kind in registry.KINDS for path in kind.paths.values()]
    found += [getattr(package, name) for name in getattr(package, "__all__", ())]
    declared = {name for name in vars(annotation) if not name.startswith("_")}
    protocol = registry.is_protocol(annotation)
    values = []
    for member in dict.fromkeys(found):
        if not isinstance(member, type):
            continue
        fits = member is not annotation and (
            all(callable(getattr(member, name, None)) for name in declared) if protocol
            else issubclass(member, annotation))
        if not fits:
            continue
        required = [field.name for field in dataclasses.fields(member) if field.init and field.default
                    is dataclasses.MISSING and field.default_factory is dataclasses.MISSING] if (
            dataclasses.is_dataclass(member)) else []
        if not all(hasattr(base, name) for name in required):
            continue
        try:
            values.append(member(**{name: getattr(base, name) for name in required}))
        except (TypeError, ValueError):
            continue
    return values


WITNESSES = {Reads: (Reads((("row", "self", "whole"),)), Reads((("row", "self", "causal"),))),
             tuple[QuestionSegment, ...]: (built_in(None, "question_first"),)}
"""Values of Dew's classes that build from no defaults, and of tuples of them, each one a run here trains
with: a row's reads of its streams (`Reads`), here its one stream reading its whole document both ways,
or up to each token (a bidirectional backbone's rows read so apart), and a question layout's own row
written as its segments (`built_in`)."""


def members_built(member: type) -> list[object]:
    """`member` built from its defaults, where it has them, else its `WITNESSES`."""
    try:
        return [member()]
    except (TypeError, ValueError):
        return list(WITNESSES.get(member, ()))


def alternatives(annotation, base=None) -> list[object]:
    """The values `annotation` declares as alternatives, in its order: a literal's values, a flag's two, an
    optional's None and its member's (an optional number's beside the saved one), Dew's classes that are
    one (`members_of`), and a dataclass's instances: `base`, the value a run saved, with one of its fields
    changed to each of that field's alternatives, or, with no base, built over its fields' alternatives. A
    plain number or text declares none, nor a record (a union holding a mapping or a list), whose values
    are its writer's."""
    if isinstance(annotation, typing.TypeAliasType):
        return alternatives(annotation.__value__, base)
    origin = typing.get_origin(annotation)
    if origin is typing.Annotated:
        return alternatives(typing.get_args(annotation)[0], base)
    if origin is typing.Literal:
        return list(typing.get_args(annotation))
    if annotation is bool:
        return [False, True]
    if origin is tuple:
        return list(WITNESSES.get(annotation, ()))
    if origin in (types.UnionType, typing.Union):
        if record(annotation):
            return []
        members = typing.get_args(annotation)
        found = []
        for member in members:
            # A count's witnesses: one, two, where one is a block no wider than causal reading (a prefix
            # of one token), and one past the saved count, which a budget that one would cut to nothing
            # takes (a layout's option tokens); a number's, a half and twice the saved one.
            counts = [1, 2, *([base + 1] if type(base) is int else [])]
            numbers = [0.5, *([base * 2] if type(base) is float else [])]
            values = ([None] if member is type(None) else counts if member is int
                      else numbers if member is float else alternatives(member, base))
            if typing.get_origin(member) is tuple:
                values = [value for value in values if value != base]
            if dataclasses.is_dataclass(member) and not isinstance(base, member):
                values = [*members_built(member), *values]
            found += [value for value in values if value not in found]
        return found
    if dataclasses.is_dataclass(annotation) and isinstance(annotation, type):
        if isinstance(base, annotation):
            # The saved value with one field changed, each field's alternatives in turn: the fields of
            # its own class, a subclass's beside the annotation's.
            saved = type(base)
            hints = typing.get_type_hints(saved)
            changed = []
            for name in (field.name for field in dataclasses.fields(saved) if field.init):
                for value in alternatives(hints[name], getattr(base, name)):
                    try:
                        changed.append(dataclasses.replace(base, **{name: value}))
                    except (TypeError, ValueError):
                        continue
            return [base, *(value for value in changed if value != base)]
        hints = typing.get_type_hints(annotation)
        fields = [field.name for field in dataclasses.fields(annotation) if field.init]
        declared = {name: alternatives(hints[name]) for name in fields}
        declared = {name: values for name, values in declared.items() if len(values) > 1}
        try:
            built = [annotation(**{name: values[index] for name, values in declared.items()})
                     for index in (0, 1)]
        except (TypeError, ValueError):
            return members_of(annotation, base)
        return list(built) if declared and built[0] != built[1] else []
    if isinstance(annotation, type) and annotation.__module__.startswith("dew."):
        return members_of(annotation, base)
    return []


def record(annotation) -> bool:
    """Whether `annotation` is a record's: a union holding a mapping or a list, whose values its writer's
    are."""
    if isinstance(annotation, typing.TypeAliasType):
        return record(annotation.__value__)
    if typing.get_origin(annotation) is typing.Annotated:
        return record(typing.get_args(annotation)[0])
    return typing.get_origin(annotation) in (types.UnionType, typing.Union) and any(
        typing.get_origin(member) in (dict, list, typing.Mapping) or member in (dict, list)
        for member in typing.get_args(annotation))


def dews(annotation) -> bool:
    """Whether `annotation` names one of Dew's types anywhere within it: alone, in a union, an alias or a
    container's arguments (a tuple of a layout's segments)."""
    if isinstance(annotation, typing.TypeAliasType):
        return dews(annotation.__value__)
    if isinstance(annotation, type) and typing.get_origin(annotation) is None:
        return annotation.__module__.startswith("dew.")
    return any(dews(argument) for argument in typing.get_args(annotation) if argument is not Ellipsis)


def bare(annotation):
    """`annotation` as a field of its type declares it: an alias's value, an annotated type's own, an
    optional's member."""
    if isinstance(annotation, typing.TypeAliasType):
        return bare(annotation.__value__)
    if typing.get_origin(annotation) is typing.Annotated:
        return bare(typing.get_args(annotation)[0])
    members = [member for member in typing.get_args(annotation) if member is not type(None)]
    if typing.get_origin(annotation) in (types.UnionType, typing.Union) and len(members) == 1:
        return bare(members[0])
    return annotation


@dataclasses.dataclass(frozen=True)
class Reach:
    """Where `dew train` sets an option (`reach`): the run's field `path`, of type `annotation`, whose value
    is the option, or, where `builds`, a builder whose call makes it (a preset its process)."""

    path: str
    annotation: object
    builds: bool = False

    def made(self, value):
        """The option a value of this field gives the task."""
        return value() if self.builds and value is not None else value

    def makes(self, value) -> bool:
        """Whether a value of this field gives the task an option: a builder may refuse its own numbers."""
        try:
            self.made(value)
        except ValueError:
            return False
        return True


def builds(annotation, option) -> bool:
    """Whether a value of `annotation` is a builder whose call returns an `option` (a preset its process)."""
    members = [member for member in typing.get_args(bare(annotation)) or (bare(annotation),)
               if member is not type(None)]
    def made(member) -> set:
        returned = typing.get_type_hints(member.__call__).get("return") if callable(member) else None
        return set(typing.get_args(returned) or (returned,))
    return any(isinstance(member, type) and bare(option) in made(member) for member in members)


def holds(field, option) -> bool:
    """Whether a run's field of type `field` takes an option of type `option`: the same type, or a tuple
    whose items' kinds hold the option's items' (a run's row segments a question layout's)."""
    if field == option:
        return True
    if typing.get_origin(field) is not tuple or typing.get_origin(option) is not tuple:
        return False
    def kinds(annotation) -> set:
        item = typing.get_args(annotation)[0]
        item = item.__value__ if isinstance(item, typing.TypeAliasType) else item
        return set(typing.get_args(item) or (item,))
    return kinds(option) <= kinds(field)


def reach(run: RunConfig, objective: str, name: str, annotation, argument: bool = True) -> Reach | None:
    """Where `dew train` sets a saved task's option `name` of `annotation`: a field of the run, or of a
    section it holds, declared of that type (a number or a text's of that name) or a builder of it
    (`builds`), else, with `argument`, its objective's argument of that name; None where it has none."""
    # A number's or a text's field is the option's by its name; any other type's, by its type.
    named = bare(annotation) in (int, float, str, bool)

    def paths(config, prefix: str):
        hints = typing.get_type_hints(type(config), include_extras=True)
        for field in dataclasses.fields(config):
            value, hint = getattr(config, field.name), hints[field.name]
            if field.name == name if named else holds(bare(hint), bare(annotation)):
                yield Reach(prefix + field.name, hint)
            elif not named and builds(hint, annotation):
                yield Reach(prefix + field.name, hint, builds=True)
            elif dataclasses.is_dataclass(value) and type(value).__module__.startswith("dew.config"):
                yield from paths(value, f"{prefix}{field.name}.")
    found = next(paths(run, ""), None)
    if found is None and argument and name in inspect.signature(objectives[objective]).parameters:
        return Reach(f"objective.{name}", annotation)
    return found


@dataclasses.dataclass(frozen=True)
class Option:
    """An option a saved task is built with: its name in the settings, dotted where it is a field of one
    of theirs (`layout.reads`), its alternatives (`witnessed`), where `dew train` sets it (`reach`), None
    where nothing does, and the field that declares it."""

    name: str
    values: list[object]
    found: Reach | None
    field: dataclasses.Field


def unwitnessed(kind: type, options) -> list[str]:
    """The fields of `kind` of Dew's types (`dews`) that declare fewer than two alternatives, so no option
    of `options` witnesses them, none declared derived from the run nor a record, its writer's."""
    hints = typing.get_type_hints(kind)
    return [field.name for field in dataclasses.fields(kind) if field.name not in options
            and dews(hints[field.name]) and "derived" not in field.metadata and not record(hints[field.name])]


def options_of(run: RunConfig, objective: str, kind: type, base, prefix: str = "") -> list[Option]:
    """The options `base`, a `kind` a run saved, holds (`witnessed`), each where `dew train` sets it. A
    field its class declares derived is the run's, set by no objective argument, which the run fills
    (`ObjectiveConfig.build`); where its saved value is a dataclass the run builds from fields of its own
    (a decision run's layout from its budgets, `_budgeted`), its options are that value's fields', each
    reached as the run sets it. A field of Dew's types that no alternative witnesses fails."""
    hints = typing.get_type_hints(kind)
    fields = {field.name: field for field in dataclasses.fields(kind)}
    declared = witnessed(kind, base)
    missing = unwitnessed(kind, declared)
    assert not missing, f"{kind.__name__}'s {', '.join(missing)} declares no alternative to witness it by"
    found = []
    for name, values in declared.items():
        derived = "derived" in fields[name].metadata
        # An objective's argument of the name is the settings' own option, not a field of a value the run
        # builds (a decision objective's segments count, a layout's segments).
        path = reach(run, objective, name, hints[name], argument=not derived and not prefix)
        value = getattr(base, name)
        # A derived value the run builds is itself an option the run fills (`--head`'s labels), and its
        # fields are options of their own.
        found.append(Option(prefix + name, values, path, fields[name]))
        if path is None and derived and dataclasses.is_dataclass(value) and not isinstance(value, type):
            found += options_of(run, objective, type(value), value, f"{prefix}{name}.")
    return found


def held_at(value, name: str):
    """The option a dotted `name` names in `value`, a field of a field where it is dotted."""
    for part in name.split("."):
        value = getattr(value, part)
    return value


def replaced(value, name: str, option):
    """`value` with the option a dotted `name` names in it replaced by `option`."""
    head, _, rest = name.partition(".")
    inner = replaced(getattr(value, head), rest, option) if rest else option
    return dataclasses.replace(value, **{head: inner})


def flag(value, annotation) -> str:
    """`value` as `--set` writes it for a field of `annotation`: a record's JSON, a tuple's values, or its
    text."""
    if value is None or isinstance(value, bool | int | str):
        return str(value)
    # A tuple of numbers or texts as its values in turn, as the command line reads one.
    if isinstance(value, tuple) and all(isinstance(item, bool | int | str) for item in value):
        return " ".join(map(str, value))
    return json.dumps(registry.to_record(value, bare(annotation)))


"""What an option's reader takes it only beside, by the run's path, set with it: a framing frames a row of
its own segments (`_budgeted`), here the question layout's own row as segments (`built_in`), in a row long
enough for its conversation's turns; a unit encodes a row of segments, the built-in row piece by piece,
here one whose state's text abuts its question's (`state_then_question`), where a tokenizer that merges
"n" and "c" (`merged`'s) reads a row encoded whole as other tokens;
a context's encoder encodes it for context layers, which read a row's context."""


TASK_RUNS = {"TextSettings": ((prefixed, "lm"),),
             "MaskedSettings": ((masked, "masked_diffusion"), (dreamed, "masked_diffusion")),
             "ImageSettings": ((diffusion, "diffusion"),), "BlockSettings": ((blockwise, "block_diffusion"),),
             "DecideSettings": ((decision, "decision"), (blocked, "decision"), (lettered, "decision"),
                                (merged, "decision"))}
"""The runs here of an objective that records each saved task's settings, and the objective's name: a
decision release's, which exports, a fresh head's over a backbone that reads blocks, a fresh letter
readout's, whose labels its layout takes, and a fresh head's over a tokenizer that merges, whose unit a row
shows; a masked run over a release, whose walk it takes."""


def witnessed(settings: type, base=None) -> dict[str, list[object]]:
    """Each field of `settings` that declares alternatives (`alternatives`), with them: about `base`'s
    value of it, the settings a run saved, where given."""
    hints = typing.get_type_hints(settings)
    declared = {field.name: alternatives(hints[field.name], getattr(base, field.name, None))
                for field in dataclasses.fields(settings)}
    return {name: values for name, values in declared.items() if len(values) > 1}


SETTINGS = sorted({cls.settings for cls in (objectives[name] for name in objectives)
                   if cls.settings is not None}, key=lambda settings: settings.__name__)
"""Every settings type an objective saves its task with, each one witnessed (`witnessed_on`): one whose
options a saved value alone declares (a block process's) is collected too, and fails where none is."""


def held_by(task, depth: int = 3) -> dict[str, tuple[object, str]]:
    """Where the loaded `task` holds each of its values, by path: the object and its attribute's name, for
    its attributes, and theirs where they are Dew's objects or dataclasses (its model's fields), `depth`
    deep."""
    found, frontier = {}, [("", task)]
    for _ in range(depth):
        following = []
        for prefix, value in frontier:
            names = ([field.name for field in dataclasses.fields(value)] if dataclasses.is_dataclass(value)
                     else list(vars(value)) if type(value).__module__.startswith("dew.") else [])
            for name in names:
                inner = getattr(value, name, None)
                found[prefix + name] = (value, name)
                if dataclasses.is_dataclass(inner) or type(inner).__module__.startswith("dew."):
                    following.append((f"{prefix}{name}.", inner))
        frontier = following
    return found


def held(task) -> dict[str, object]:
    """What the loaded `task` holds, by path (`held_by`)."""
    return {path: getattr(holder, name, None) for path, (holder, name) in held_by(task).items()}


def answers(task) -> list[object]:
    """What one request of `task` answers (`requested`), each leaf a host value, in order."""
    def leaves(value) -> list[object]:
        if isinstance(value, Mapping):
            return [leaf for name in sorted(value) for leaf in leaves(value[name])]
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            return [leaf for field in dataclasses.fields(value)
                    for leaf in leaves(getattr(value, field.name))]
        if isinstance(value, list | tuple):
            return [leaf for item in value for leaf in leaves(item)]
        # A generation's decoder, a callback each request builds anew, is how it reads, not what it answers.
        if callable(value):
            return []
        return [np.asarray(value) if isinstance(value, jax.Array | np.ndarray) else value]
    return leaves(requested(task))


def answered_alike(first: list[object], second: list[object]) -> bool:
    """Whether two requests' answers (`answers`) are one: each leaf equal, an array's in shape and in
    every entry."""
    def alike(a, b) -> bool:
        if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
            return (isinstance(a, np.ndarray) and isinstance(b, np.ndarray) and a.shape == b.shape
                    and np.array_equal(a, b, equal_nan=True))
        return bool(a == b)
    return len(first) == len(second) and all(alike(a, b) for a, b in zip(first, second, strict=True))


def saved_with(run: RunConfig, root: Path, name: str, settings: type, assigned: dict[str, str]):
    """The `settings` the task of `run` saves after one step of `dew train` with each `path=value` of
    `assigned` set (`trained_by_cli`), and the task loaded from them (`dew.pipeline`)."""
    import dew
    from dew.inference.tasks import SavedRun, saved_settings

    directory = trained_by_cli(run, root, name, assigned)
    return saved_settings(SavedRun.pinned(str(directory)).record, settings), dew.pipeline(str(directory))


def same(first, second) -> bool:
    """Whether two values are one option: of one type and one record (`registry.to_record`), as a run
    saves and reads them, where a schedule compares by its numbers, not its identity."""
    if type(first) is not type(second):
        return False
    try:
        return registry.to_record(first, type(first)) == registry.to_record(second, type(second))
    except (TypeError, ValueError):
        return bool(first == second)


def accepted(saved, name: str, value) -> bool:
    """Whether `saved` settings take `value` for the option `name` names beside the rest they hold."""
    try:
        replaced(saved, name, value)
    except (TypeError, ValueError):
        return False
    return True


def witnessed_by(run: RunConfig, root: Path, settings: type, flags, queued: dict[str, list[object]],
                 consumed) -> dict[str, list[str]]:
    """Each option of `queued` set alone through `dew train` (`flags` writes its `--set`s), so what its
    task holds differs from the run that sets none by it alone, at each alternative in turn until one is
    consumed (`consumed`: why the saved settings and the loaded task do not witness it, or None). Why
    each alternative failed, for each option none witnesses."""
    failed: dict[str, list[str]] = {}
    for name, values in queued.items():
        for index, value in enumerate(values):
            try:
                saved, task = saved_with(run, root, f"{name}{index}", settings, flags({name: value}))
            except ValueError as error:  # A reader refused it, which names what refused it: the next.
                failed.setdefault(name, []).append(f"{value!r}: {error}")
                continue
            if (why := consumed(saved, name, value, task, index)) is None:
                failed.pop(name, None)
                break
            failed.setdefault(name, []).append(f"{value!r}: {why}")
        else:
            failed.setdefault(name, ["no alternative to set"])
    return failed


def witnessed_on(builder, objective: str, settings: type, root: Path, elsewhere: frozenset[str] = frozenset()
                 ) -> tuple[dict[str, Option], dict[str, str], bool]:
    """The options of `settings` the run `builder` makes saves (`options_of`), and why each the run does not
    witness is not: one with no path its settings do not declare derived, or derived from data the run
    leaves unfilled; one set alone to each alternative in turn (`witnessed_by`) none of which trains, saves
    it and has its loaded task consume it: hold it where it held another before (`held`) and answer apart
    from the same task with what it held before in its place, the consumer's work, not its record
    (`unread`), a released task's export too (`dew export`), where the run exports, which it says. An
    option another run witnessed already (`elsewhere`) is not set again here."""
    import dew
    from dew.objectives.base import ReleasedTask

    root.mkdir(parents=True)
    # A saved task's options are its training's; no validation split scores or saves any of them.
    run = dataclasses.replace(builder(root, drawn=False), validation=None)
    saved_task = objectives[objective].saved_task
    released = saved_task is not None and issubclass(saved_task, ReleasedTask)
    base, unset = saved_with(run, root, "unset", settings, {})
    before = held(unset)
    # A released task's run exports where its model's format holds it (a model with cross-attention
    # layers its checkpoint keeps alone): its witnesses' exports then hold them too.
    exports = released and exported(root / "runs" / "unset", root / "exported" / "unset")
    options = {option.name: option for option in options_of(run, objective, settings, base)}
    failures = {}
    for name, option in options.items():
        if option.found is None and "derived" not in option.field.metadata:
            failures[name] = "it has no path"
        # A derived field the run fills, from its data or the option its declaration names (`--head`'s
        # labels): on one run at least, where it holds other than its default.
        elif (option.found is None and option.field.default is not dataclasses.MISSING
              and same(held_at(base, name), option.field.default)):
            failures[name] = "the run fills none"
    # A builder's path is witnessed by its own alternatives, each the option its call makes; an option
    # set beside what its reader takes it with (`BESIDE`) is judged by the run alone.
    queued = {name: [value for value in (alternatives(found.annotation) if found.builds else option.values)
                     if found.makes(value) and not same(found.made(value), held_at(base, name))
                     and (name in BESIDE or accepted(base, name, found.made(value)))]
              for name, option in options.items()
              if (found := option.found) is not None and name not in elsewhere}
    reachable = {name: option.found for name, option in options.items() if option.found is not None}
    hints = typing.get_type_hints(type(run), include_extras=True)

    def flags(assigned: dict[str, object]) -> dict[str, str]:
        """The `--set`s of each option `assigned`, and of what its reader takes it only beside (`BESIDE`)."""
        beside = {path: flag(value, hints[path])
                  for name in assigned for path, value in BESIDE.get(name, {}).items()}
        return {**beside, **{reachable[name].path: flag(value, reachable[name].annotation)
                             for name, value in assigned.items()}}
    def consumed(saved, name: str, value, task, index: int) -> str | None:
        """Why the task a run saved with `name` set to `value` does not witness it, None where it does: it
        saved another (`held_at`), or its loaded task, or a released task's export (`dew export`) reloaded
        as its users load one, does not consume it (`unread`). A format that holds no such option refuses
        it by name (a release's holds its budgets alone), never drops it."""
        option = reachable[name].made(value)
        if not same(held_at(saved, name), option):
            return f"saved {held_at(saved, name)!r}"
        missed = unread(task, before, option)
        destination = root / "exported" / f"{name}{index}"
        if missed is None and exports and exported(root / "runs" / f"{name}{index}", destination):
            missed = unread(dew.pipeline(str(destination)), before, option)
            return None if missed is None else f"its export: {missed}"
        return missed
    witnessing = witnessed_by(run, root, settings, flags, queued, consumed)
    failures.update({name: f"no alternative witnesses it: {why}" for name, why in witnessing.items()})
    return options, failures, exports


UNHELD = ("states a layout's window, budgets and option layout alone",
          "Laya's layout holds a ModernBERT encoder", "the source's format holds no cross-attention layers",
          "Laya's layout is a MarkerLayout over a Hugging Face tokenizer")
"""What `dew export` says refusing a run whose format holds not all it saved, rather than dropping it: a
release's config (Laya's) its layout's window, budgets and option layout alone, its weights a ModernBERT
encoder's, and its source's format no cross-attention layer a context reads through."""


def exported(run: Path, destination: Path) -> bool:
    """Whether `dew export` writes the saved `run` to `destination`, False where it refuses the run as a
    format that holds not all it saved (`UNHELD`); any other failure is the export's own, raised."""
    try:
        return main(["export", str(run), str(destination)]) == 0
    except ValueError as error:
        if not any(said in str(error) for said in UNHELD):
            raise
        return False


def unread(task, before: dict[str, object], option) -> str | None:
    """Why the loaded `task` does not consume `option`, None where it does: it holds it nowhere it held
    another before (`held`), or a request of it answers as the same task does holding there what it held
    before (`answers`): the same weights, drawn afresh so every part of the model answers its input
    (`model_support.moved`), the option alone apart, so what differs is the consumer's work with it, not its
    record. A request of one task answers alike twice, so an answer apart is the option's alone."""
    places = [path for path, inner in held(task).items()
              if same(inner, option) and path in before and not same(before[path], inner)]
    if not places:
        return "the task loaded holds none set"
    # Every weight moved, so no part of the model answers every input alike as an untrained one may (a
    # zero-initialized output layer, a run whose first step its warmup leaves unstepped).
    drawn = dataclasses.replace(task, variables=moved(task.variables, jax.random.key(2639), 0.1))
    # Swapped wherever a dataclass holds it, which rebuilds what the task derives from it; a copy an
    # object of another kind holds goes with the dataclass that made it.
    swapped = drawn
    for path in places:
        try:
            swapped = replaced(swapped, path, before[path])
        except TypeError:
            continue
    if swapped is drawn:
        return "the task loaded holds it nowhere a swap reaches"
    answered = answers(drawn)
    assert answered_alike(answered, answers(drawn)), "a request of one task answers apart twice"
    if answered_alike(answered, answers(swapped)):
        return f"a request of it answers as with {[before[path] for path in places]} in its place"
    return None


def the_attention_forms_a_run_names_reach_its_step(tmp_path):
    """A GPU flow's cell (tests/test_gpu_training.py): the attention forms a run's numerics name on the
    command line (`--set trainer.precision`, `Numerics.forms`) reach its step through the default route: a
    step in the accurate forms (IEEE fp32 products) and one in the fast (a GPU's TF32), the model's
    compute float32 in both, score its batch apart, so the route reads the run's forms, not its record
    alone (ledger 2350). A CPU multiplies fp32 one way in both, so only a GPU tells them apart."""
    from process_support import logged_losses

    from dew.nn.precision import Numerics

    run = lm(tmp_path, drawn=False)
    losses = {forms: logged_losses(trained_by_cli(run, tmp_path, forms, {
        "trainer.precision": flag(Numerics(forms=forms), Numerics)})) for forms in ("fast", "accurate")}
    assert losses["fast"] and losses["fast"] != losses["accurate"]
