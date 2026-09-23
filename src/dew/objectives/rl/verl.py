"""Session interchange with verl 12ebe0c's native `AgentLoopOutput` rows.

verl's agent loop emits one row per trajectory: `prompt_ids`, then
`response_ids` concatenating every assistant turn with the tool and template
ids between them, `response_mask` 1 on the assistant ids alone and
`response_logprobs` the engine's behavior likelihoods (0.0 on the rest),
beside `routed_experts` for routing replay, `reward_score` and the multimodal
payloads (`agent_loop.py` L88-L118). That is exactly one strict chain of a
`Session`, so `to_verl` writes one row per chain `sessions.chains` builds and
`from_verl` rebuilds each call: a run of mask-1 ids is one call's sampled ids,
and everything before it is that call's prompt. The oldest policy version is
verl's `extra_fields.min_global_steps` (`llm_server.py` L160-L162, L324-L348).

`extra_fields.dew` is this exporter's own, and optional on import: the
session's identity, status, verifier detail and each call's finish reason,
version and sampling support, none of which verl's fields carry. Rows without
it are read as verl wrote them. verl fields Dew keeps no record of
(multimodal data and processor payloads, metrics, verl's own extra fields)
travel beside the session in `VerlTrajectory.extras` and are written back as
they came. No torch or verl import happens in Dew.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Protocol, TypedDict

import numpy as np

from dew.records import JSON

from .records import integer, object_record, real, sequence, text
from .sessions import Call, Session, Status, _chains

MEDIA = ("multi_modal_data", "mm_processor_kwargs", "mm_processor_output")
"""The fields that put media into a row's context, which Dew's text trainer
does not read."""

EXTRAS = (*MEDIA, "metrics")
"""Native fields Dew has no record for; they ride in `VerlTrajectory.extras`."""

FIELDS = ("prompt_ids", "response_ids", "response_mask", "response_logprobs", "routed_experts",
          "multi_modal_data", "reward_score", "num_turns", "metrics", "extra_fields",
          "mm_processor_kwargs", "mm_processor_output")
"""`AgentLoopOutput`'s twelve fields at verl 12ebe0c, in its order."""


class _DewCall(TypedDict):
    """One call of a chain: where its sampled ids start in the row's ids and
    how many it sampled, which may be none (a call aborted before its first
    token)."""

    start: int
    count: int
    finish_reason: str
    version: int
    support: list[list[int]] | None


class VerlRow(TypedDict):
    """One `AgentLoopOutput` row under verl's field names, as JSON."""

    prompt_ids: list[int]
    response_ids: list[int]
    response_mask: list[int]
    response_logprobs: list[float]
    routed_experts: list[list[list[int]]] | None
    multi_modal_data: object
    reward_score: float | None
    num_turns: int
    metrics: object
    extra_fields: dict[str, object]
    mm_processor_kwargs: object
    mm_processor_output: object


@dataclass(frozen=True)
class VerlTrajectory:
    """A session and the verl fields that travel with it unread.

    `extras` holds `EXTRAS` and verl's own `extra_fields` (under
    `extra_fields`, without Dew's key), as JSON values.
    """

    session: Session
    extras: Mapping[str, object] = field(default_factory=lambda: MappingProxyType({}))


def to_verl(trajectories: Sequence[Session | VerlTrajectory]) -> list[VerlRow]:
    """Export JSON-compatible `AgentLoopOutput` rows, one per strict chain.

    A session without calls has no chain and writes no row. `routed_experts`
    is written when the chain's last recorded call covers every row.
    """
    rows: list[VerlRow] = []
    for given in trajectories:
        trajectory = given if isinstance(given, VerlTrajectory) else VerlTrajectory(given)
        session = trajectory.session
        built = _chains(session, 0, 1 << 62)
        for number, chain in enumerate(built):
            sampled = [index >= 0 for index in chain.calls]
            # A chain starts with its first call's whole prompt.
            start = chain.members[0][1]
            calls: list[_DewCall] = []
            for index, first, count in chain.members:
                call = session.calls[index]
                calls.append({"start": first, "count": count, "finish_reason": call.finish_reason,
                              "version": call.version,
                              "support": None if call.support is None else [list(kept) for kept in call.support]})
            versions = [session.calls[index].version for index, _, _ in chain.members]
            gaps = sum(1 for position in range(start + 1, len(chain.tokens))
                       if not sampled[position] and sampled[position - 1])
            given_extra = trajectory.extras.get("extra_fields")
            extra: dict[str, object] = dict(object_record(given_extra)) if given_extra else {}
            # A native row's own stamps are kept, its maximum naming the
            # newest weights; Dew's are written, and later dropped, only
            # where the row had none.
            stamped = "min_global_steps" not in extra
            if stamped:
                extra.update(min_global_steps=min(versions), max_global_steps=max(versions))
            extra["dew"] = {"session": {
                "task": session.task, "group": session.group, "sample": session.sample,
                "attempt": session.attempt, "status": session.status.value,
                "components": dict(session.components), "detail": session.detail}, "chain": number, "chains": len(built),
                            "calls": calls, "stamped": stamped}
            routing = chain.routing
            rows.append({
                "prompt_ids": chain.tokens[:start],
                "response_ids": chain.tokens[start:],
                "response_mask": [int(drawn) for drawn in sampled[start:]],
                "response_logprobs": chain.behavior[start:],
                "routed_experts": (routing.tolist() if routing is not None
                                   and len(routing) == len(chain.tokens) - 1 else None),
                "multi_modal_data": trajectory.extras.get("multi_modal_data"),
                "reward_score": session.reward,
                "num_turns": 1 + len(chain.members) + gaps,
                "metrics": trajectory.extras.get("metrics") or {},
                "extra_fields": extra,
                "mm_processor_kwargs": trajectory.extras.get("mm_processor_kwargs"),
                "mm_processor_output": trajectory.extras.get("mm_processor_output"),
            })
    return rows


def _ids(row: Mapping[str, object], name: str) -> list[int]:
    try:
        return list(sequence(row.get(name), integer))
    except ValueError:
        raise ValueError(f"verl {name} is a list of integers, got {row.get(name)!r}") from None


def _runs(mask: Sequence[int], offset: int) -> list[tuple[int, int]]:
    """Each maximal run of 1s in `mask` as (start + offset, length)."""
    runs: list[tuple[int, int]] = []
    for position, drawn in enumerate(mask):
        if drawn and (not position or not mask[position - 1]):
            runs.append((offset + position, 0))
        if drawn:
            runs[-1] = (runs[-1][0], runs[-1][1] + 1)
    return runs


def _calls(row: Mapping[str, object], version: int | None,
           described: Sequence[Mapping[str, object]] | None) -> tuple[Call, ...]:
    """Rebuild a row's model calls: each run of mask-1 ids is one call's
    sampled ids, and the row's ids before it are the call's prompt."""
    prompt, response, mask = _ids(row, "prompt_ids"), _ids(row, "response_ids"), _ids(row, "response_mask")
    if len(mask) != len(response) or set(mask) - {0, 1}:
        raise ValueError("verl response_mask holds one 0 or 1 per response id")
    given = row.get("response_logprobs")
    if 1 in mask and (given is None or len(sequence(given, real)) != len(response)):
        raise ValueError("a verl row with sampled ids needs one response_logprob per response id; "
                         "behavior likelihoods are never estimated")
    logprobs = () if given is None else sequence(given, real)
    full = prompt + response
    runs = (_runs(mask, len(prompt)) if described is None
            else [(integer(call["start"]), integer(call["count"])) for call in described])
    routed = row.get("routed_experts")
    record = None if routed is None else np.asarray(routed)
    calls = []
    for number, (begin, length) in enumerate(runs):
        meta = described[number] if described is not None else {}
        stated = meta.get("version", version)
        if stated is None:
            raise ValueError("a native verl row names its policy version in extra_fields.min_global_steps; "
                             "this one has none, so pass version=")
        forwarded = begin + length - 1
        if not begin or begin + length > len(full):
            raise ValueError(f"verl call {number} samples ids {begin}..{begin + length} of a "
                             f"{len(full)}-id row after a nonempty prompt")
        if record is not None and len(record) < forwarded:
            raise ValueError(f"verl routed_experts covers {len(record)} ids; call {number} forwarded {forwarded}")
        support = meta.get("support")
        offset = begin - len(prompt)
        calls.append(Call(
            prompt_ids=tuple(full[:begin]), sampled_ids=tuple(full[begin:begin + length]),
            behavior_log_probs=tuple(logprobs[offset:offset + length]),
            finish_reason=text(meta.get("finish_reason", "stop" if number == len(runs) - 1 else "tool_calls")),
            version=integer(stated),
            routed_experts=None if record is None else record[:forwarded],
            support=None if support is None else sequence(support, lambda kept: sequence(kept, integer))))
    return tuple(calls)


def _dew(row: Mapping[str, object]) -> Mapping[str, object]:
    return object_record(object_record(row["extra_fields"])["dew"])


def from_verl(rows: Sequence[Mapping[str, object]], *, samples: int = 1,
              version: int | None = None, media: bool = False) -> tuple[VerlTrajectory, ...]:
    """Import `AgentLoopOutput` rows as sessions.

    Rows Dew wrote carry their session's identity and every call's metadata;
    a session whose calls split into several chains spans that many
    consecutive rows. Native rows are one session each. verl repeats each
    prompt `rollout.n` times interleaved (`batch.repeat(interleave=True)`,
    `ray_trainer.py`), so with `samples` set to that n, row i is sample
    `i % samples` of group `i // samples`. A native row is scored
    (`reward_score` set) and names its version in `min_global_steps`,
    unless `version` is given.

    A row carrying media (`MEDIA`) is refused unless `media` is set. Its
    prompt holds placeholder ids the engine expanded from images or video
    that verl's actor re-reads (`mm_processor_kwargs` "must stay aligned
    across rollout and training paths", `agent_loop.py` L111-L112), and `pack`
    would score it as text. `media=True` imports it for a round trip only:
    the session comes back through `to_verl` exact, and must not be trained on.
    """
    if type(samples) is not int or samples < 1:
        raise ValueError("samples is verl's rollout.n, a positive integer")
    trajectories = []
    cursor = 0
    while cursor < len(rows):
        row = rows[cursor]
        unknown = sorted(set(row) - set(FIELDS))
        if unknown:
            raise ValueError(f"verl row {cursor} has fields {unknown} outside AgentLoopOutput")
        extra = dict(object_record(row.get("extra_fields") or {}))
        extras = {name: row[name] for name in EXTRAS if row.get(name) is not None}
        # verl's loops write empty dicts on text-only rows; only content is media.
        carried = sorted(name for name in MEDIA if row.get(name))
        if not media and carried:
            raise ValueError(
                f"verl row {cursor} carries media ({carried}), "
                "which a text trainer would score without; pass media=True to import it for a round trip")
        reward = row.get("reward_score")
        if "dew" not in extra:
            if reward is None:
                raise ValueError(f"verl row {cursor} has no reward_score; score it before training on it")
            steps = extra.get("min_global_steps")
            session = Session(str(cursor // samples), "0", cursor % samples, 0,
                              _calls(row, version if steps is None else integer(steps), None),
                              Status.COMPLETED, real(reward))
            used = 1
        else:
            dew = _dew(row)
            count = integer(dew["chains"])
            group = rows[cursor:cursor + count]
            if len(group) != count or [integer(_dew(member)["chain"]) for member in group] != list(range(count)) \
                    or any(_dew(member)["session"] != dew["session"] for member in group):
                raise ValueError(f"verl rows from {cursor} do not hold the {count} chains of one session")
            calls = tuple(call for member in group
                          for call in _calls(member, None, sequence(_dew(member)["calls"], object_record)))
            record = object_record(dew["session"])
            components = {key: real(value) for key, value in object_record(record["components"]).items()}
            session = Session(text(record["task"]), text(record["group"]), integer(record["sample"]),
                              integer(record["attempt"]), calls, Status(text(record["status"])),
                              None if reward is None else real(reward), MappingProxyType(components),
                              text(record["detail"]))
            if dew.get("stamped", False):
                for key in ("min_global_steps", "max_global_steps"):
                    extra.pop(key, None)
            used = count
        extra.pop("dew", None)
        if extra:
            extras["extra_fields"] = extra
        trajectories.append(VerlTrajectory(session, MappingProxyType(extras)))
        cursor += used
    return tuple(trajectories)


class VerlScore(Protocol):
    """verl's reward signature, `verl/utils/reward_score/__init__.py` L19-L27:
    a score, or a mapping holding `score`."""

    def __call__(self, *, data_source: str, solution_str: str, ground_truth: str,
                 extra_info: JSON = None) -> float | Mapping[str, float]: ...


def verl_reward(compute_score: VerlScore) -> Callable[[str, str, str, str], float]:
    """Adapt a verl-style reward function to Dew's `Reward`.

    Dew carries `extra_info` as JSON text (`dew.data.prompts`), which verl
    hands its scorers as a dict; the empty string is verl's missing extra.
    A mapping result gives its `score`, as verl's reward managers read it.
    """
    def reward(data_source: str, completion: str, ground_truth: str, extra_info: str) -> float:
        context = json.loads(extra_info) if extra_info else None
        # verl's reward managers pass these four by keyword
        # (`reward_loop/reward_manager/naive.py` L66-L80).
        score = compute_score(data_source=data_source, solution_str=completion,
                              ground_truth=ground_truth, extra_info=context)
        return float(score["score"] if isinstance(score, Mapping) else score)

    return reward
