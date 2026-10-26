"""Over-length policies, fixed-length checks and target counts of token records."""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable, Mapping
from typing import Literal

import grain.python as grain
import numpy as np

from dew._records import boolean, integer, record, text
from dew.batch import Batch
from dew.data.pipeline import Rejected


@dataclasses.dataclass(frozen=True)
class LengthPolicy(grain.MapTransform):
    length: int
    field: str = "text"
    long: Literal["truncate", "drop"] = "truncate"
    side: Literal["left", "right"] = "right"
    aligned: tuple[str, ...] = ()

    @property
    def rejects(self) -> bool:
        """Whether a long record is dropped (`filter_map`) rather than cut (`map`)."""
        return self.long == "drop"

    def map(self, element: Batch) -> Batch | Rejected:
        count = len(element[self.field])
        if count <= self.length:
            return element
        if self.long == "drop":
            return Rejected("over_length")
        span = slice(count - self.length, None) if self.side == "left" else slice(self.length)
        return {
            key: value[span] if key in (self.field, *self.aligned) else value
            for key, value in element.items()
        } | {"_dew_truncated": True, "_dew_length": self.length}


@dataclasses.dataclass(frozen=True)
class ValidTargets:
    field: str = "text"
    role: int | None = None
    pad_id: int | None = None
    measured: bool = True

    def __call__(self, element: Batch) -> int:
        """The record's targets the loss scores, by the LM objective's own rule (`scored_targets`)."""
        if self.measured and "_dew_valid_targets" in element:
            return int(element["_dew_valid_targets"])
        from dew.batch import scored_targets

        segments = element.get(f"{self.field}_segment_ids")
        blocks = element.get(f"{self.field}_block_groups")
        roles = None if self.role is None else np.asarray(element[f"{self.field}_roles"])[None]
        return int(scored_targets(np, np.asarray(element[self.field])[None], self.pad_id, None,
                                  None if segments is None else np.asarray(segments)[None],
                                  roles=roles, role=self.role,
                                  blocks=None if blocks is None else np.asarray(blocks)[None]).sum())


@dataclasses.dataclass(frozen=True)
class CountTargets(grain.MapTransform):
    """Each record's targets the loss scores, counted once and carried on the record with the rule
    that counted them (`recounted`), so a token mixture weighs it without counting it again."""

    role: int | None = None
    pad_id: int | None = None
    rejects = True

    def declared(self, facts: Mapping[str, object]) -> Mapping[str, object]:
        """The records as they were, their targets counted."""
        return facts

    def map(self, element: Batch) -> Batch | Rejected:
        count = ValidTargets(role=self.role, pad_id=self.pad_id, measured=False)(element)
        if count == 0:
            return Rejected("zero_targets")
        rule = {name: value for name, value in zip(_RULE, (self.role, self.pad_id), strict=True)
                if value is not None}
        kept = {name: value for name, value in element.items() if name not in (*_RULE, _BOUND)}
        return {**kept, "_dew_valid_targets": count, **rule}


_RULE = ("_dew_target_role", "_dew_target_pad")
"""The fields a counted record carries its count's rule in, each where the rule sets it."""


_COUNTED = ("text", "text_segment_ids", "text_roles", "text_block_groups")
"""The fields a record's targets are counted from (`ValidTargets`), as they are first named."""

_BOUND = "_dew_target_fields"
"""Where a counted record names the fields its targets were counted from, as projections since
renamed them (`rebound`), one entry each of `_COUNTED`; an empty name is one a projection removed."""

_CARRIED = ("_dew_valid_targets", *_RULE, _BOUND)
"""What a counted record carries of its count: the count, its rule and the fields it was counted
from (`recounted`)."""


def _bound(row: Batch) -> tuple[str, ...]:
    return tuple(row.get(_BOUND, _COUNTED))


def rebound(element: Batch, kept: Batch, renamed: Mapping[str, str]) -> Batch:
    """Where a projection of a counted `element` (`Fields`) leaves the fields its count was counted
    from: renamed where it renames one, empty where it drops one, which no step can count from
    again; nothing for a record no count rides on."""
    if "_dew_valid_targets" not in element:
        return {}
    names = tuple(renamed.get(name, name) if name in kept else "" if name in element else name
                  for name in _bound(element))
    return {_BOUND: names}


def targets_changed(given: Batch, output: Batch) -> bool:
    """Whether a step that made `output` of `given` changed what its targets are counted from: a field the
    count reads that `given` held and `output` holds as a different array, or a counted field that `output`
    holds and `given` did not, as a map adding a row's blocks does. Each is read where the record names it
    before the step and where `output` names it after (`rebound`), so a projection that moves a field, or
    names another field as it was, changes nothing; a field the step removed pins the count, which nothing it
    still holds could give again."""
    pairs = list(zip(_bound(given), _bound(output), strict=True))
    held = [(before, after) for before, after in pairs if before in given]
    if "" in _bound(output) or any(after not in output for _, after in held):
        return False
    replaced = any(output[after] is not given[before] for before, after in held)
    arrived = any(before not in given and after in output for before, after in pairs)
    return replaced or arrived


def recounted(given: Batch, output: Batch) -> Batch:
    """`output` with the targets counted on `given` (`CountTargets`): carried onto a record the step
    made anew without them, with the rule and the names they were counted by; kept while the step
    left every field they were counted from as it was, as a map of a record's other fields or a
    projection does; and counted again by the rule that counted them where it changed one in place,
    as a crop does, from the fields as they are named now. Every read carries a count so, a memory
    read's among them, which keeps no count of its own beside its records."""
    if "_dew_valid_targets" not in given:
        return output
    if "_dew_valid_targets" not in output:
        output = {**output, **{name: given[name] for name in _CARRIED if name in given}}
    if not targets_changed(given, output):
        return output
    return {**output, "_dew_valid_targets": _scored(output)}


def _scored(row: Batch) -> int:
    """The targets of a counted `row` by the rule it carries, from the fields as they are named now."""
    view = {counted: row[name] for counted, name in zip(_COUNTED, _bound(row), strict=True) if name in row}
    rule = ValidTargets(role=row.get("_dew_target_role"), pad_id=row.get("_dew_target_pad"), measured=False)
    return rule(view)


@dataclasses.dataclass(frozen=True)
class CountRule:
    """The rule a record's targets were counted by (`CountTargets`) and the name its roles go by
    (`rebound`): a windows or pack stage counts the windows it makes of such records by it, as the
    loss scores them, since their documents' counts describe none of them."""

    role: int | None
    pad_id: int | None
    roles: str

    @classmethod
    def of(cls, element: Batch) -> CountRule | None:
        """The rule `element`'s targets were counted by, or None where it is every target's, as an
        uncounted row's are (`ValidTargets`): no role and no padding."""
        role, pad_id = (element.get(name) for name in _RULE)
        if "_dew_valid_targets" not in element or (role is None and pad_id is None):
            return None
        # Roles a rule does not read are no part of it.
        return cls(None if role is None else int(role), None if pad_id is None else int(pad_id),
                   "" if role is None else _bound(element)[2])

    def counted(self, window: Batch, field: str) -> Batch:
        """`window`, made of records counted by this rule from its token `field`, its own targets
        counted by the rule and carrying it, so a later step or layout counts them by it again."""
        if self.role is not None and self.roles not in window:
            raise ValueError(
                f"the documents' targets were counted in role {self.role}, and the windows made of them "
                f"drop the roles {self.roles!r}; keep them among the stage's fields")
        rule = {name: value for name, value in zip(_RULE, (self.role, self.pad_id), strict=True)
                if value is not None}
        held = {**window, **rule, _BOUND: (field, f"{field}_segment_ids", self.roles or f"{field}_roles",
                                          f"{field}_block_groups")}
        return {**held, "_dew_valid_targets": _scored(held)}


def count_rules(rows: Iterable[Batch | bytes | None]) -> list[list[int | None]]:
    """The rules other than every target's that `rows`' targets were counted by (`CountRule`), as
    a store states them (`count_rules`): each a role and a pad id."""
    rules = {(rule.role, rule.pad_id) for row in rows if isinstance(row, Mapping)
             for rule in (CountRule.of(row),) if rule is not None}
    return [list(rule) for rule in sorted(rules, key=repr)]


def stated_rules(value: object) -> set[tuple[int | None, int | None]]:
    """The rules a store's `count_rules` states, each a role and a pad id (`CountRule`), either
    None, as a generation writes them."""
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"count_rules is a list of [role, pad_id] pairs, not {value!r}")
    rules: set[tuple[int | None, int | None]] = set()
    for rule in value:
        if not (isinstance(rule, (list, tuple)) and len(rule) == 2
                and all(part is None or isinstance(part, int) for part in rule)):
            raise ValueError(f"a count rule is a [role, pad_id] pair of integers or None, not {rule!r}")
        rules.add((rule[0], rule[1]))
    return rules


def counted_window(window: Batch, rule: CountRule | None, field: str) -> Batch:
    """A window a pack or windows stage made, its targets of its token `field` counted as the loss
    scores them and carried on it, by the rule its documents were counted by (`CountRule`), or every
    target's where nothing counted them: a later step, the read that accepts it (a memory read's,
    `dew.data.lowering._Build.targeted`) and a mixture by tokens read them as a counted record's."""
    return (rule or CountRule(None, None, "")).counted(window, field)


class LayoutRule:
    """The one rule the documents a pack or windows stage lays out were counted by (`CountRule`),
    or None where nothing counted them, as each document passes. Documents counted by two rules,
    or some by none, make windows no one rule counts, and are refused."""

    def __init__(self) -> None:
        self.seen = False
        self.rule: CountRule | None = None

    @classmethod
    def of(cls, documents: Iterable[Batch]) -> CountRule | None:
        """The one rule `documents` were counted by, all of them passing."""
        seen = cls()
        for document in documents:
            seen(document)
        return seen.rule

    def __call__(self, document: Batch) -> Batch:
        rule = CountRule.of(document)
        if not self.seen:
            self.seen, self.rule = True, rule
        elif rule != self.rule:
            raise ValueError(f"the documents of one layout were counted by two rules, {self.rule!r} and "
                             f"{rule!r}, and the windows made of them hold one")
        return document

    def state(self) -> Mapping[str, object]:
        return {"seen": self.seen, "rule": None if self.rule is None else dataclasses.asdict(self.rule)}

    def restore(self, state: Mapping[str, object]) -> None:
        self.seen = boolean(state["seen"], "seen count rule")
        if state["rule"] is None:
            self.rule = None
            return
        rule = record(state["rule"], "count rule")
        role, pad_id = (None if rule[name] is None else integer(rule[name], name)
                        for name in ("role", "pad_id"))
        self.rule = CountRule(role, pad_id, text(rule["roles"], "roles"))


@dataclasses.dataclass(frozen=True)
class FixedTokens(grain.MapTransform):
    """Rows of exactly `max_length + extra` tokens of `field`, the
    `max_length` a token budget divides into rows (`TokenGeometry`)."""

    max_length: int
    extra: int = 0
    field: str = "text"

    def map(self, element: Batch) -> Batch:
        size = len(element[self.field])
        if size != self.max_length + self.extra:
            raise ValueError(f"fixed tokens need {self.max_length + self.extra} values, got {size}")
        return element
