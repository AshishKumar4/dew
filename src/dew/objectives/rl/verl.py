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

Rewards cross here too. verl scores a reply with `compute_score(data_source,
solution_str, ground_truth, extra_info)` (`verl/utils/reward_score/__init__.py`
`default_compute_score`); Dew's `Reward` is `(completion, PromptRecord)`.
`verl_reward` brings a verl function in and `compute_score` hands a Dew reward
to verl, each passing the record's three fields as verl names them.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TypedDict

from dew._records import JSON, finite, integer, integers, record as section
from dew.data.prompts import PromptRecord
from dew.inference.rollouts import routing_head, routing_json, routing_record, routing_rows
from dew.registry import from_record, record_fields, recorded_names

from .records import reals
from .rollout import Reward
from .sessions import Call, Session, Status, chains

MEDIA = ("multi_modal_data", "mm_processor_kwargs", "mm_processor_output")
"""The fields that put media into a row's context, which Dew's text trainer
does not read."""

EXTRAS = (*MEDIA, "metrics")
"""Native fields Dew has no record for; they ride in `VerlTrajectory.extras`."""

_CALL_IN_ROW = ("prompt", "tokens", "behavior_log_probs", "routed_experts")
"""The `Call` fields a row's own verl columns carry (its ids, mask, likelihoods and routing), which Dew's
record of each call leaves out."""

_SESSION_IN_ROWS = ("calls", "reward")
"""The `Session` fields its rows carry (its calls, its `reward_score`), which Dew's record of the session
leaves out."""


@dataclass(frozen=True)
class DewRow:
    """Dew's own record of a row (`extra_fields.dew`): its session's record beside its calls and reward,
    which chain of the session's it is and how many it has, each call's record beside its ids (where its
    sampled ids `start` in the row's ids and how many, `count`, it sampled: none for a call aborted before
    its first token), and whether Dew `stamped` the row's global steps. The session's and the calls' records
    are theirs (`registry.record_fields`), read back by `from_record`."""

    session: Mapping[str, JSON]
    chain: int
    chains: int
    calls: tuple[Mapping[str, JSON], ...]
    stamped: bool = False


class VerlRow(TypedDict):
    """One `AgentLoopOutput` row under verl's field names, as JSON."""

    prompt_ids: list[int]
    response_ids: list[int]
    response_mask: list[int]
    response_logprobs: list[float]
    routed_experts: list[list[list[int]]] | dict[str, list[list[list[int]]]] | None
    multi_modal_data: object
    reward_score: float | None
    num_turns: int
    metrics: object
    extra_fields: dict[str, object]
    mm_processor_kwargs: object
    mm_processor_output: object


FIELDS = tuple(VerlRow.__annotations__)
"""`AgentLoopOutput`'s twelve fields at verl 12ebe0c, in its order: `VerlRow`'s."""


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
    is written when the chain's last recorded call covers every row, an
    object of one such list a routed site, by its path, where the calls'
    records are keyed by site (`dew.inference.rollouts.Routing`).
    """
    rows: list[VerlRow] = []
    for given in trajectories:
        trajectory = given if isinstance(given, VerlTrajectory) else VerlTrajectory(given)
        session = trajectory.session
        built = chains(session, 0, 1 << 62)
        for number, chain in enumerate(built):
            sampled = [index >= 0 for index in chain.calls]
            # A chain starts with its first call's whole prompt.
            start = chain.members[0][1]
            calls = tuple({"start": first, "count": count,
                           **record_fields(session.calls[index], Call, without=_CALL_IN_ROW)}
                          for index, first, count in chain.members)
            versions = [session.calls[index].version for index, _, _ in chain.members]
            gaps = sum(1 for position in range(start + 1, len(chain.tokens))
                       if not sampled[position] and sampled[position - 1])
            given_extra = trajectory.extras.get("extra_fields")
            extra: dict[str, object] = dict(section(given_extra, "extra_fields")) if given_extra else {}
            # A native row's own stamps are kept, its maximum naming the
            # newest weights; Dew's are written, and later dropped, only
            # where the row had none.
            stamped = "min_global_steps" not in extra
            if stamped:
                extra.update(min_global_steps=min(versions), max_global_steps=max(versions))
            extra["dew"] = record_fields(DewRow(record_fields(session, Session, without=_SESSION_IN_ROWS),
                                                number, len(built), calls, stamped), DewRow)
            routing = chain.routing
            rows.append({
                "prompt_ids": chain.tokens[:start],
                "response_ids": chain.tokens[start:],
                "response_mask": [int(drawn) for drawn in sampled[start:]],
                "response_logprobs": chain.behavior[start:],
                "routed_experts": (routing_json(routing) if routing is not None
                                   and routing_rows(routing) == len(chain.tokens) - 1 else None),
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
        return list(integers(row.get(name), name))
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
           described: Sequence[Mapping[str, JSON]] | None) -> tuple[Call, ...]:
    """Rebuild a row's model calls: each run of mask-1 ids is one call's
    sampled ids, and the row's ids before it are the call's prompt; each
    call's own record where Dew wrote the row (`DewRow.calls`)."""
    prompt, response, mask = _ids(row, "prompt_ids"), _ids(row, "response_ids"), _ids(row, "response_mask")
    if len(mask) != len(response) or set(mask) - {0, 1}:
        raise ValueError("verl response_mask holds one 0 or 1 per response id")
    given = row.get("response_logprobs")
    if 1 in mask and (given is None or len(reals(given, "response_logprobs")) != len(response)):
        raise ValueError("a verl row with sampled ids needs one response_logprob per response id; "
                         "behavior likelihoods are never estimated")
    logprobs = () if given is None else reals(given, "response_logprobs")
    full = prompt + response
    runs = (_runs(mask, len(prompt)) if described is None
            else [(integer(call["start"], "start"), integer(call["count"], "count")) for call in described])
    routed = row.get("routed_experts")
    record = None if routed is None else routing_record(routed)
    calls = []
    for number, (begin, length) in enumerate(runs):
        forwarded = begin + length - 1
        if not begin or begin + length > len(full):
            raise ValueError(f"verl call {number} samples ids {begin}..{begin + length} of a "
                             f"{len(full)}-id row after a nonempty prompt")
        if record is not None and routing_rows(record) < forwarded:
            raise ValueError(
                f"verl routed_experts covers {routing_rows(record)} ids; call {number} forwarded {forwarded}"
            )
        offset = begin - len(prompt)
        carried = {"prompt": tuple(full[:begin]), "tokens": tuple(full[begin:begin + length]),
                   "behavior_log_probs": tuple(logprobs[offset:offset + length]),
                   "routed_experts": None if record is None else routing_head(record, forwarded)}
        if described is not None:
            own = {name: value for name, value in described[number].items() if name not in ("start", "count")}
            if "finish_reason" not in own:
                raise ValueError(f"Dew's record of verl call {number} states no finish_reason, as only "
                                 "an earlier Dew wrote it; export the sessions again with the Dew that "
                                 "collected them")
            calls.append(from_record(Call, {**own, **carried}, dtypes=False))
        elif version is None:
            raise ValueError("a native verl row names its policy version in extra_fields.min_global_steps; "
                             "this one has none, so pass version=")
        else:
            # A native row states no finish: the chain's last call stopped it and every other ended on a
            # tool call.
            finished = "stop" if number == len(runs) - 1 else "tool_calls"
            calls.append(Call(**carried, finish_reason=finished, version=version))
    return tuple(calls)


def _dew(row: Mapping[str, object]) -> DewRow:
    """Dew's record of a row it wrote (`DewRow`), its session's and each call's record whole, as Dew writes
    them (`_whole`)."""
    dew = from_record(DewRow, section(section(row["extra_fields"], "extra_fields")["dew"], "dew"),
                      dtypes=False)
    missing = sorted(set(recorded_names(Session)) - set(_SESSION_IN_ROWS) - set(dew.session))
    if missing:
        # A session's identity a default would read as another session's (its source, its group).
        raise ValueError(f"a session's record Dew wrote states each of its fields; this one lacks {missing}")
    return dew


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
        extra = dict(section(row.get("extra_fields") or {}, "extra_fields"))
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
                              _calls(row, version if steps is None else integer(steps, "min_global_steps"),
                                     None),
                              Status.COMPLETED, finite(reward, "reward_score"))
            used = 1
        else:
            dew = _dew(row)
            count = integer(dew.chains, "chains")
            if not 1 <= count <= len(rows) - cursor:
                raise ValueError(f"verl row {cursor} says its session spans {count} rows; "
                                 f"a session spans at least one, and {len(rows) - cursor} remain")
            group = [_dew(member) for member in rows[cursor:cursor + count]]
            if (
                len(group) != count
                or [member.chain for member in group] != list(range(count))
                or any(member.session != dew.session for member in group)
            ):
                raise ValueError(f"verl rows from {cursor} do not hold the {count} chains of one session")
            calls = tuple(call for member, written in zip(rows[cursor:cursor + count], group, strict=True)
                          for call in _calls(member, None, written.calls))
            scored = None if reward is None else finite(reward, "reward_score")
            session = from_record(Session, {**dew.session, "calls": calls, "reward": scored}, dtypes=False)
            if dew.stamped:
                for key in ("min_global_steps", "max_global_steps"):
                    extra.pop(key, None)
            used = count
        extra.pop("dew", None)
        if extra:
            extras["extra_fields"] = extra
        trajectories.append(VerlTrajectory(session, MappingProxyType(extras)))
        cursor += used
    return tuple(trajectories)


type ComputeScore = Callable[[str, str, JSON, JSON], float]
"""verl's reward function: `(data_source, solution_str, ground_truth, extra_info) -> score`, where
verl may pass a ground truth or extra info that is not a string."""


def verl_reward(score: ComputeScore) -> Reward:
    """The Dew `Reward` that scores with verl's `compute_score` function `score`."""

    def reward(completion: str, record: PromptRecord) -> float:
        return score(record.data_source, completion, record.ground_truth, record.extra_info)

    return reward


def compute_score(reward: Reward) -> ComputeScore:
    """verl's `compute_score` for the Dew `reward`: verl's arguments, read back as the record a
    prompt batch would hold (`PromptRecord.of`)."""

    def score(data_source: str, solution_str: str, ground_truth: JSON, extra_info: JSON) -> float:
        return reward(solution_str, PromptRecord.of(data_source, ground_truth, extra_info))

    return score
