"""JSON-compatible episode records for interchange and turn-boundary recovery."""

import math
from collections.abc import Callable, Mapping
from dataclasses import asdict
from typing import TypedDict

from dew.records import JSON
from dew.sampling.text import Sampling

from .episodes import Action, Episode, EpisodeId, EpisodeStatus, Observation, Transition


def object_record(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ValueError("episode record must be a string-keyed object")
    return value


def sequence[ItemT](value: object, read: Callable[[JSON], ItemT]) -> tuple[ItemT, ...]:
    """Read every entry of a JSON array, each one through `read`."""
    if not isinstance(value, (list, tuple)):
        raise ValueError("episode field must be an array")
    return tuple(read(entry) for entry in value)


def integer(value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError("episode field must be an integer")
    return value


def real(value: object) -> float:
    if not isinstance(value, (float, int)) or isinstance(value, bool) or not math.isfinite(value):
        raise ValueError("episode field must be a finite number")
    return float(value)


def text(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("episode field must be text")
    return value


def observation_record(value: Mapping[str, object]) -> Observation:
    record = object_record(value)
    return Observation(sequence(record["context"], integer),
                       EpisodeStatus(integer(record["status"])), text(record["detail"]))


def action_record(value: Mapping[str, object]) -> Action:
    record = object_record(value)
    controls = object_record(record["sampling"])
    eos, top_k = controls["eos_id"], controls["top_k"]
    sampling = Sampling(temperature=real(controls["temperature"]),
                        top_k=None if top_k is None else integer(top_k),
                        eos_id=None if eos is None else sequence(eos, integer),
                        pad_id=integer(controls["pad_id"]), top_p=real(controls["top_p"]),
                        min_p=real(controls["min_p"]))
    terminated = record["terminated"]
    if not isinstance(terminated, bool):
        raise ValueError("episode terminated must be a boolean")
    return Action(sequence(record["context"], integer),
                  sequence(record["tokens"], integer),
                  sequence(record["raw_log_probs"], real),
                  sequence(record["behavior_log_probs"], real),
                  terminated, integer(record["policy_step"]), sampling,
                  _binding_id=text(record["_binding_id"]))


class EpisodeFields(TypedDict, total=False):
    """Name every field of `Episode` a record carries, under the field's own name.

    The keys are the dataclass's own init fields, which
    `tests/test_verl_episodes.py` pins, so a field renamed there is a failing
    test here rather than a key a reader never looks for. The nested records
    are what `asdict` made of the identity, the observations and the turns,
    and `to_verl` takes the turns back out of its copy, so this is a dict and
    not a frozen mapping. Every key is written, and an exporter that drops one
    is what `total=False` states.
    """

    identity: Mapping[str, object]
    policy_step: int
    initial: Mapping[str, object] | None
    transitions: tuple[Mapping[str, object], ...]
    status: EpisodeStatus
    detail: str
    reward: float | None
    _binding_id: str


def episode_record(episode: Episode) -> EpisodeFields:
    """Write an episode's exact turns, observations, likelihoods and collection origin."""
    return EpisodeFields(**asdict(episode))


def episode_from_record(value: Mapping[str, object]) -> Episode:
    """Read an episode without reconstructing actions from rendered text."""
    record = object_record(value)
    identity = object_record(record["identity"])
    initial, reward = record["initial"], record["reward"]
    transitions = tuple(
        Transition(action_record(object_record(turn["action"])),
                   observation_record(object_record(turn["observation"])))
        for turn in sequence(record["transitions"], object_record))
    return Episode(EpisodeId(integer(identity["task"]), integer(identity["attempt"]),
                             integer(identity["sample"]), sequence(identity["seed"], integer)),
                   integer(record["policy_step"]),
                   None if initial is None else observation_record(object_record(initial)),
                   transitions, EpisodeStatus(integer(record["status"])), text(record["detail"]),
                   None if reward is None else real(reward), _binding_id=text(record["_binding_id"]))
