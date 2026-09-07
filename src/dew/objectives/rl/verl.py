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

from .episodes import Episode
from .records import episode_from_record, episode_record, integer, object_record, sequence



def to_verl(episodes: Sequence[Episode]) -> list[dict[str, object]]:
    """Export JSON-compatible AgentLoopOutput rows, with lossless Dew metadata."""
    rows: list[dict[str, object]] = []
    for episode in episodes:
        header = episode_record(episode)
        header.pop("transitions")
        turns = episode.transitions
        for index in range(max(1, len(turns))):
            turn = turns[index] if turns else None
            action = asdict(turn.action) if turn else None
            if action is not None:
                for key in ("context", "tokens", "behavior_log_probs"):
                    action.pop(key)
            rows.append({
                "prompt_ids": list(turn.action.context) if turn else list(episode.initial.context if episode.initial else ()),
                "response_ids": list(turn.action.tokens) if turn else [],
                "response_mask": [1] * len(turn.action.tokens) if turn else [],
                "response_logprobs": list(turn.action.behavior_log_probs) if turn else None,
                "reward_score": episode.reward if turn else None,
                "num_turns": 1 if turn else 0, "metrics": {},
                "extra_fields": {"dew": {"episode": header, "turn": index, "turns": len(turns),
                    "action": action, "observation": asdict(turn.observation) if turn else None}},
            })
    return rows


def from_verl(rows: Sequence[Mapping[str, object]]) -> tuple[Episode, ...]:
    """Import call-major rows with exact raw/behavior and turn-boundary metadata.

    Native verl rows alone lack raw-policy likelihoods, turn boundaries and
    snapshot origin. Their producer must supply extra_fields.dew explicitly.
    """
    episodes = []
    cursor = 0
    while cursor < len(rows):
        try:
            metadata = object_record(object_record(rows[cursor]["extra_fields"])["dew"])
            header = object_record(metadata["episode"])
            count = integer(metadata["turns"])
            if count < 0 or cursor + max(1, count) > len(rows):
                raise ValueError("incomplete verl episode turns")
            transitions = []
            for index in range(max(1, count)):
                row = rows[cursor + index]
                extra = object_record(object_record(row["extra_fields"])["dew"])
                if extra["episode"] != header or integer(extra["turn"]) != index or extra["turns"] != count:
                    raise ValueError("verl rows must contain contiguous complete episode turns")
                response = sequence(row["response_ids"])
                mask = sequence(row["response_mask"])
                if tuple(mask) != (1,) * len(response):
                    raise ValueError("verl call rows must mask only the recorded model actions")
                if count == 0:
                    if response or extra["action"] is not None:
                        raise ValueError("an empty episode cannot carry a sampled action")
                    continue
                action = dict(object_record(extra["action"]))
                action.update(context=row["prompt_ids"], tokens=response,
                              behavior_log_probs=sequence(row["response_logprobs"]))
                if row["reward_score"] != header["reward"]:
                    raise ValueError("verl reward_score disagrees with the episode reward")
                transitions.append({"action": action, "observation": extra["observation"]})
            episodes.append(episode_from_record({**header, "transitions": transitions}))
            cursor += max(1, count)
        except KeyError as error:
            raise ValueError(f"verl episode is missing provenance field {error}; extra_fields.dew is required") from error
    return tuple(episodes)
