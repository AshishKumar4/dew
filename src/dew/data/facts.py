"""What a chain's records are, read without opening a reader (`Dataset.facts`): its stores' stated
facts, folded through its stages in the order they run, each step that knows what it makes saying so
(`Declares`)."""

from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping, Sequence

from dew._records import JSON, integer, record
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

IMAGE_FIELDS = ("jpg", "png", "image")
"""The fields a record holds its image in by convention (docs/design/data.md, the steps), the first a step's
default."""
TEXT_FIELDS = ("text", "txt", "caption")
"""The fields a record holds its text in by convention, the first a step's default."""


def conventional(held: Collection[str], field: str, convention: Sequence[str]) -> str:
    """The field of `held` a step reads that names `field`: `field` itself where `held` holds it; where
    `field` is the convention's first, which a step reads by default, the first of `convention` `held` holds
    (row 5757); else `field`, which the step's read refuses as missing. A field the convention names and a
    record lacks thus reads by the convention, and any other the step was given is read as it is named."""
    if field in held or field != convention[0]:
        return field
    return next((name for name in convention if name in held), field)


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


class PerMember(dict):
    """A fact each corpus states of its own member of a mixture, keyed by the member (a mixture's weight of
    each corpus, `dew.decision.data.mixture`): the corpora a run reads together state it joined, each
    member's value its own corpus's (`merge_facts`), where a whole copy in each corpus held a mixture's
    size a corpus, its square a mixture (row 4079)."""


def merge_facts(facts: Iterable[Mapping[str, object]], *,
                measured: frozenset[str] = frozenset()) -> Mapping[str, object]:
    """Facts several stores or corpora state, as one run reads them together: a fact they state
    alike, slot maxima and `measured` facts as their largest, a field's shape where every one of
    them states it alike, and a fact of each member (`PerMember`) joined; a fact two state differently
    is no fact of the run, whatever a third states."""
    combined: dict[str, object] = {}
    # A fact of each member joined in a mapping of the merge's own, each corpus's added in place.
    joined: dict[str, PerMember] = {}
    conflicted: set[str] = set()
    stated_slots: list[Mapping[str, object]] = []
    shapes: list[Mapping[str, object]] = []
    for given in facts:
        shapes.append(record(given.get("fields", {}), "fields"))
        for key, value in given.items():
            if key == "slots":
                stated_slots.append(record(value, "slots"))
            elif key == "fields":
                continue
            elif key in measured:
                combined[key] = max(integer(value, key), integer(combined.get(key, 0), key))
            elif isinstance(value, PerMember) and isinstance(prior := combined.get(key), PerMember):
                # Each member's value its own corpus's; a member two state otherwise is no fact of the run.
                if key not in joined:
                    joined[key] = PerMember(prior)
                held = joined[key]
                if any(held[member] != stated for member, stated in value.items() if member in held):
                    conflicted.add(key)
                held.update(value)
                combined[key] = held
            elif key not in combined:
                combined[key] = value
            elif combined[key] != value:
                conflicted.add(key)
    shared = {key: value for key, value in combined.items() if key not in conflicted}
    slots = largest_bounds(stated_slots)
    shaped = ({name: shape for name, shape in shapes[0].items()
               if all(other.get(name) == shape for other in shapes[1:])} if shapes else {})
    return {**shared, **({"slots": slots} if slots else {}), **({"fields": shaped} if shaped else {})}


def largest_bounds(stated: Iterable[Mapping[str, object]]) -> dict[str, JSON]:
    """The bounds of rows read together: each bound the largest any of `stated` gives it, a nested
    mapping's bounds each its own (a decision row's `slots.questions`, `slots.options`). The one
    reduction of a slot bound, which a merge of facts and a measure of rows both read."""
    bounds: dict[str, JSON] = {}
    for given in stated:
        for key, value in given.items():
            if isinstance(value, Mapping):
                bounds[key] = largest_bounds([record(bounds.get(key, {}), key), value])
            else:
                bounds[key] = max(integer(value, key), integer(bounds.get(key, 0), key))
    return bounds


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
