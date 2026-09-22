"""Episode interchange with verl d040717's AgentLoopOutput wire format.

One row is one actual model call. Keeping its exact prompt supports context
compaction between turns without pretending the model saw a concatenated
transcript. response_logprobs holds behavior likelihoods. extra_fields.dew
retains raw likelihoods and episode/observation data absent from verl's
standard fields. Import refuses missing provenance instead of estimating it.
No torch or verl dependency is imported by Dew.
"""

from collections.abc import Mapping, Sequence
from dataclasses import asdict
from typing import TypedDict

from dew.records import JSON

from .episodes import Episode
from .records import episode_from_record, episode_record, integer, object_record, real, sequence


class VerlRow(TypedDict):
    """Hold one AgentLoopOutput row, under verl's own field names.

    `extra_fields.dew` is this exporter's own: verl's fields carry neither
    raw-policy likelihoods nor turn boundaries, so an import that had only
    the standard fields would have to estimate them, and `from_verl` refuses
    to. Every value is JSON, because a row is written to a file.
    """

    prompt_ids: list[int]
    response_ids: list[int]
    response_mask: list[int]
    response_logprobs: list[float] | None
    reward_score: float | None
    num_turns: int
    metrics: Mapping[str, JSON]
    extra_fields: Mapping[str, Mapping[str, object]]


def _call_row(episode: Episode, header: Mapping[str, JSON], index: int) -> VerlRow:
    """Render one of an episode's model calls as an AgentLoopOutput row.

    An episode with no turn still exports one row, so its header and its
    initial context survive the round trip.
    """
    turns = episode.transitions
    turn = turns[index] if turns else None
    action = asdict(turn.action) if turn else None
    if action is not None:
        # The wire format carries these three under verl's own names.
        for key in ("context", "tokens", "behavior_log_probs"):
            action.pop(key)
    return {
        "prompt_ids": list(turn.action.context) if turn else list(episode.initial.context if episode.initial else ()),
        "response_ids": list(turn.action.tokens) if turn else [],
        "response_mask": [1] * len(turn.action.tokens) if turn else [],
        "response_logprobs": list(turn.action.behavior_log_probs) if turn else None,
        "reward_score": episode.reward if turn else None,
        "num_turns": 1 if turn else 0, "metrics": {},
        "extra_fields": {"dew": {"episode": header, "turn": index, "turns": len(turns),
            "action": action, "observation": asdict(turn.observation) if turn else None}},
    }


def to_verl(episodes: Sequence[Episode]) -> list[VerlRow]:
    """Export JSON-compatible AgentLoopOutput rows, with lossless Dew metadata."""
    rows: list[VerlRow] = []
    for episode in episodes:
        header = episode_record(episode)
        header.pop("transitions")
        turns = episode.transitions
        for index in range(max(1, len(turns))):
            rows.append(_call_row(episode, header, index))
    return rows


def _call_transition(row: Mapping[str, object], header: Mapping[str, object],
                     count: int, index: int) -> dict | None:
    """Read one verl row as a transition of the episode `header` names.

    Returns None for the single row an empty episode exports, which
    carries no action.
    """
    extra = object_record(object_record(row["extra_fields"])["dew"])
    if extra["episode"] != header or integer(extra["turn"]) != index or extra["turns"] != count:
        raise ValueError("verl rows must contain contiguous complete episode turns")
    response = sequence(row["response_ids"], integer)
    mask = sequence(row["response_mask"], integer)
    if tuple(mask) != (1,) * len(response):
        raise ValueError("verl call rows must mask only the recorded model actions")
    if count == 0:
        if response or extra["action"] is not None:
            raise ValueError("an empty episode cannot carry a sampled action")
        return None
    action = dict(object_record(extra["action"]))
    action.update(context=row["prompt_ids"], tokens=response,
                  behavior_log_probs=sequence(row["response_logprobs"], real))
    if row["reward_score"] != header["reward"]:
        raise ValueError("verl reward_score disagrees with the episode reward")
    return {"action": action, "observation": extra["observation"]}


def _read_episode(rows: Sequence[Mapping[str, object]], cursor: int) -> tuple[Episode, int]:
    """Read the episode whose rows start at `cursor`, and how many it used.

    An episode's rows are contiguous and each one names its own turn
    index, so a truncated or interleaved export is refused here.
    """
    metadata = object_record(object_record(rows[cursor]["extra_fields"])["dew"])
    header = object_record(metadata["episode"])
    count = integer(metadata["turns"])
    if count < 0 or cursor + max(1, count) > len(rows):
        raise ValueError("incomplete verl episode turns")
    transitions = []
    for index in range(max(1, count)):
        transition = _call_transition(rows[cursor + index], header, count, index)
        if transition is not None:
            transitions.append(transition)
    return episode_from_record({**header, "transitions": transitions}), max(1, count)


def from_verl(rows: Sequence[Mapping[str, object]]) -> tuple[Episode, ...]:
    """Import call-major rows with exact raw/behavior and turn-boundary metadata.

    Native verl rows alone lack raw-policy likelihoods, turn boundaries and
    snapshot origin. Their producer must supply extra_fields.dew explicitly.
    """
    episodes = []
    cursor = 0
    while cursor < len(rows):
        try:
            episode, used = _read_episode(rows, cursor)
        except KeyError as error:
            raise ValueError(f"verl episode is missing provenance field {error}; extra_fields.dew is required") from error
        episodes.append(episode)
        cursor += used
    return tuple(episodes)
