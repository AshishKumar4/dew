"""What a chain's records are, read without opening a reader (`Dataset.facts`): its stores' stated
facts, folded through its stages in the order they run, each step that knows what it makes saying so
(`Declares`)."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

from dew._records import integer, record
from dew.data.pipeline import (
    Artifact,
    Declares,
    Filter,
    Interleave,
    Limit,
    Mixture,
    Phases,
    Pipeline,
    Prefetch,
    Repeat,
    Shuffle,
    Slice,
    Stage,
    TokenSplit,
    Workers,
    runs,
)
from dew.data.sources.base import stated

# Stages that order, select, drop, repeat or place records, and change none of them.
_KEEPS = (Shuffle, Repeat, Slice, Limit, Filter, Prefetch, Workers, Interleave, Artifact, TokenSplit)


def folded(pipeline: Pipeline) -> Mapping[str, object]:
    """The facts of `pipeline`'s records: its store's stated facts (`Stated`), or its corpora's
    merged (`merge_facts`), then each stage's in order. A stage that changes records and doesn't say
    how (`Declares`) leaves their fields unknown, so `fields` goes; the rest stays."""
    source = pipeline.source
    if isinstance(source, (Mixture, Phases)):
        facts = merge_facts(folded(child) for child in source.pipelines)
    else:
        facts = dict(stated(source))
    for stage in pipeline.stages:
        facts = _after(stage, facts)
    return facts


def _after(stage: Stage, facts: Mapping[str, object]) -> Mapping[str, object]:
    if isinstance(stage, _KEEPS):
        return facts
    if isinstance(stage, Declares):
        return stage.declared(facts)
    declaring = _declaring(stage)
    if declaring is not None:
        return declaring.declared(facts)
    return {name: value for name, value in facts.items() if name != "fields"}


def _declaring(stage: Stage) -> Declares | None:
    """What says the facts a stage makes: the step it runs (`runs`: a record step, a configured
    step's factory, a batch step), or a stage that runs none, a device step, itself, where it
    declares them."""
    ran = runs(stage)
    step = stage if ran is None else ran.step
    return step if isinstance(step, Declares) else None


def merge_facts(facts: Iterable[Mapping[str, object]], *,
                measured: frozenset[str] = frozenset()) -> Mapping[str, object]:
    """Facts several stores or corpora state, as one run reads them together: a fact they state
    alike, slot maxima and `measured` facts as their largest, a field's shape where every one of
    them states it alike; a fact two state differently is no fact of the run, whatever a third
    states."""
    combined: dict[str, object] = {}
    conflicted: set[str] = set()
    slots: dict[str, int] = {}
    shapes: list[Mapping[str, object]] = []
    for given in facts:
        shapes.append(record(given.get("fields", {}), "fields"))
        for key, value in given.items():
            if key == "slots":
                for name, size in record(value, "slots").items():
                    slots[name] = max(slots.get(name, 0), integer(size, f"slots.{name}"))
            elif key == "fields":
                continue
            elif key in measured:
                combined[key] = max(integer(value, key), integer(combined.get(key, 0), key))
            elif key not in combined:
                combined[key] = value
            elif combined[key] != value:
                conflicted.add(key)
    shared = {key: value for key, value in combined.items() if key not in conflicted}
    shaped = ({name: shape for name, shape in shapes[0].items()
               if all(other.get(name) == shape for other in shapes[1:])} if shapes else {})
    return {**shared, **({"slots": slots} if slots else {}), **({"fields": shaped} if shaped else {})}


def shaped(facts: Mapping[str, object], field: str, shape: list[int | None]) -> Mapping[str, object]:
    """`facts` with `field` holding arrays of `shape`."""
    return {**facts, "fields": {**record(facts.get("fields", {}), "fields"), field: shape}}


def unshaped(facts: Mapping[str, object], *names: str) -> Mapping[str, object]:
    """`facts` without a shape for `names`, fields a step replaces with values of no fixed shape."""
    fields = record(facts.get("fields", {}), "fields")
    return {**facts, "fields": {name: shape for name, shape in fields.items() if name not in names}}


def windowed(facts: Mapping[str, object], seq_len: int, fields: Sequence[str], *, packed: bool,
             along: Sequence[tuple[str, int]] = ()) -> Mapping[str, object]:
    """The facts of token rows of `seq_len` inputs and one more token, their targets, along `fields`,
    and of the fields `along` axes of their own, each its own length. A row holds those alone:
    a window's other fields are no record's, so their shapes go."""
    shapes = {name: [seq_len + 1] for name in fields}
    shapes.update({name: [length] for name, length in along})
    return {**facts, "seq_len": seq_len, "packed": packed, "fields": shapes}


def resized(facts: Mapping[str, object], field: str, height: int, width: int) -> Mapping[str, object]:
    """`facts` with the images in `field` `height` by `width`, their channels as many as they held,
    where the facts say how many, and their shape unknown where they don't."""
    shape = record(facts.get("fields", {}), "fields").get(field)
    if not (isinstance(shape, list) and len(shape) == 3 and isinstance(shape[2], int)):
        return unshaped(facts, field)
    return shaped(facts, field, [height, width, shape[2]])
