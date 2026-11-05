"""A checkpoint step's payloads: the Orbax items a persistent step holds, each one kind of reader's
(`ROLES`), which its retention may cut it to (`dew.checkpoints.Checkpoints._project`); a step's facts as
its readers read them (`_StepFacts`); and the refusals of a read a step no longer holds the payload for."""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping, Sequence

import orbax.checkpoint as ocp
from etils import epath

from dew import _records
from dew._records import text
from dew.checkpoints._state_trees import (
    TreePath,
    _at,
    _averages_of,
    _power_profiles,
    _tree_of,
    _with,
    _with_averages,
)

MODEL, PROFILES, RESUME = "model", "profiles", "resume"


ROLES = (MODEL, PROFILES, RESUME)
"""The Orbax items of a run's persistent step, each the payload of one kind of reader: the model, its
counters, variables and EMA (`dew.state.MODEL_STATE`), which an evaluation, an export and a weights-only
snapshot read; its post-hoc EMA profiles' averages (`dew.optim.power_profiles`), which
`Checkpoints.posthoc_ema` reads; and the rest of its state, which only a resume reads: the optimizer's
state, the key, the loss scale, the window, the position, the rollout and a DiLoCo run's groups'. A step its
retention keeps for some of them alone holds those (`Checkpoints._project`, row 4311)."""


ONE_ITEM = "default"
"""The one Orbax item of a step that holds its whole named state: a step an earlier Dew wrote, and a mapping
saved in a train state's place."""


PROJECTED = "projected.json"
"""The record beside a step's items of those it keeps, written before the rest are deleted, so a step
whose deletion was cut short reads as cut (`_held_payloads`)."""


def _held_payloads(step_directory: str) -> tuple[str, ...]:
    """The items the step at `step_directory` holds for its readers (`ROLES`, or `ONE_ITEM`): those its
    projection record keeps (`PROJECTED`), or else the items it has."""
    path = epath.Path(step_directory)
    if (path / PROJECTED).exists():
        return tuple(_records.strings(json.loads((path / PROJECTED).read_text()), PROJECTED))
    return tuple(name for name in (*ROLES, ONE_ITEM) if (path / name).is_dir())


def _payload_of(tree: str) -> str:
    """The item that holds a step's variables tree, by its name (`_variable_trees`): the model's and its
    EMA in `model`, a DiLoCo group's in `resume`."""
    from dew.state import MODEL_STATE
    return MODEL if tree.split('/')[0] in MODEL_STATE else RESUME


def _frozen_held[T](frozen: Mapping[str, T], payloads: Sequence[str]) -> dict[str, T]:
    """The store entries of a step's `frozen` record (`FROZEN_STORE`) whose trees the items it holds keep:
    a step cut to its model references its groups' collections no longer."""
    return {tree: names for tree, names in frozen.items()
            if ONE_ITEM in payloads or _payload_of(tree) in payloads}


def _in_payloads(tree: Mapping, profiles: TreePath | None) -> dict[str, dict]:
    """A step's named state tree, or a restore's template or arguments over one, as its items hold it
    (`ROLES`): the model's fields in `model`; the averages of the profiles its optimizer state keeps at
    `profiles` in `profiles`, their place there left empty; every other field in `resume`; no item of no
    field."""
    from dew.state import MODEL_STATE
    rest = dict(tree)
    payloads: dict[str, dict] = {MODEL: {}, PROFILES: {}, RESUME: {}}
    held = None if profiles is None else _at(rest, profiles)
    found = None if held is None else _power_profiles(held)
    if found is not None:
        # Their place in the optimizer state holds no leaf, which every reader of a step's trees keeps.
        payloads[PROFILES] = {'averages': _averages_of(found)}
        rest = _with(rest, profiles, _with_averages(held, None))
    for name, value in rest.items():
        payloads[MODEL if name in MODEL_STATE else RESUME][name] = value
    return {payload: fields for payload, fields in payloads.items() if fields}


def _from_payloads(payloads: Mapping[str, Mapping], profiles: TreePath | None) -> dict:
    """A step's named state tree from what was read of its items (`_in_payloads`'s inverse): the profiles'
    averages back in their optimizer state where both were read."""
    tree = {name: value for payload in (MODEL, RESUME, ONE_ITEM)
            for name, value in (payloads.get(payload) or {}).items()}
    averages = payloads.get(PROFILES)
    held = None if profiles is None else _at(tree, profiles)
    if averages is not None and held is not None:
        tree = _with(tree, profiles, _with_averages(held, averages['averages']))
    return tree


def _profile_path(custom: Mapping[str, object]) -> TreePath | None:
    """Where the step whose custom metadata is `custom` kept its profiles in its state (`_profiles_at`)."""
    profiles = custom.get('profiles')
    at = profiles.get('at') if isinstance(profiles, Mapping) else None
    return None if at is None else tuple(text(at, 'profiles at').split('/'))


@dataclasses.dataclass(frozen=True)
class _StepFacts:
    """A step's metadata as its readers read it (`Checkpoints._step_metadata`): its custom metadata, its
    named state tree's metadata over the items it holds, and those items (`ROLES` or `ONE_ITEM`; none for a
    local step, its own manager's one item, or a local step this process lost, as a process holding it
    read it)."""

    custom_metadata: dict
    item_metadata: Mapping[str, object]
    payloads: tuple[str, ...] = ()
    trees: Mapping[str, Mapping[str, object]] = dataclasses.field(default_factory=dict)
    """Each item's own metadata, which `item_metadata` merges."""


def _facts(metadata: ocp.metadata.StepMetadata, payloads: tuple[str, ...] = ()) -> _StepFacts:
    """Orbax's metadata of a step, read over the `payloads` it holds (`_held_payloads`), none for a local
    step."""
    custom = dict(metadata.custom_metadata or {})
    if not payloads:
        return _StepFacts(custom, _tree_of(metadata.item_metadata))
    held = dict(metadata.item_metadata or {})
    trees = {payload: dict(_tree_of(held[payload])) for payload in payloads if held.get(payload) is not None}
    return _StepFacts(custom, _from_payloads(trees, _profile_path(custom)), payloads, trees)


_PAYLOADS = {MODEL: "its model (counters, variables and EMA)", PROFILES: "its post-hoc EMA profiles",
             RESUME: "the rest of its state a resume reads (its optimizer state, key and data position)",
             ONE_ITEM: "its whole state"}
"""How a refusal names each item a step holds or lacks (`ROLES`)."""


def _refuse_profiles_alone(step: int, facts: _StepFacts) -> None:
    """Refuse a read of a step's named state where it holds no model and no state a resume reads, only its
    profiles' averages (its retention keeps it for them alone), which `Checkpoints.restore_profiles`
    reads. A step kept for its model reads as its model; the profiles it keeps beside are read there too."""
    if facts.payloads and not {MODEL, RESUME, ONE_ITEM} & set(facts.payloads):
        raise ValueError(f"step {step} holds {_payload(facts.payloads)} alone, which its retention keeps it "
                         "for; its profiles' averages are read by Checkpoints.restore_profiles")


def _refuse_cut_resume(step: int, facts: _StepFacts, field: str) -> None:
    """Refuse a read of `field`, state only a resume reads (its data position, its rollout), at a step its
    retention cut to its model or its profiles, which held it once and holds it no longer: so a cut step
    is not read as one saved without it."""
    if facts.payloads and not {RESUME, ONE_ITEM} & set(facts.payloads):
        raise ValueError(f"step {step} holds {_payload(facts.payloads)} alone, which its retention keeps it "
                         f"for, and no {field}: read it from a step a resume reads (Checkpoints.latest)")


def _payload(payloads) -> str:
    return " and ".join(_PAYLOADS[payload] for payload in (*ROLES, ONE_ITEM) if payload in payloads)


_NEEDS = {MODEL: frozenset({MODEL}), PROFILES: frozenset({PROFILES}), RESUME: frozenset(ROLES)}
"""The items each kind of reader needs of a step (`ROLES`): a resume needs all three, its model and profiles
among its state."""
