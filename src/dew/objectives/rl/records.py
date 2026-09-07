"""JSON-compatible episode records for interchange and turn-boundary recovery."""

from collections.abc import Mapping, Sequence
from dataclasses import asdict
import math

from dew.sampling.text import Sampling
from .episodes import Action, Episode, EpisodeId, EpisodeStatus, Observation, Transition


def object_record(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ValueError("episode record must be a string-keyed object")
    return value


def sequence(value: object) -> Sequence[object]:
    if not isinstance(value, (list, tuple)):
        raise ValueError("episode field must be an array")
    return value


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


def observation_record(value: object) -> Observation:
    record = object_record(value)
    return Observation(tuple(integer(token) for token in sequence(record["context"])),
                       EpisodeStatus(integer(record["status"])), text(record["detail"]))


def action_record(value: object) -> Action:
    record = object_record(value)
    controls = object_record(record["sampling"])
    eos, top_k = controls["eos_id"], controls["top_k"]
    sampling = Sampling(temperature=real(controls["temperature"]),
                        top_k=None if top_k is None else integer(top_k),
                        eos_id=None if eos is None else tuple(integer(token) for token in sequence(eos)),
                        pad_id=integer(controls["pad_id"]), top_p=real(controls["top_p"]),
                        min_p=real(controls["min_p"]))
    terminated = record["terminated"]
    if not isinstance(terminated, bool):
        raise ValueError("episode terminated must be a boolean")
    return Action(tuple(integer(token) for token in sequence(record["context"])),
                  tuple(integer(token) for token in sequence(record["tokens"])),
                  tuple(real(value) for value in sequence(record["raw_log_probs"])),
                  tuple(real(value) for value in sequence(record["behavior_log_probs"])),
                  terminated, integer(record["policy_step"]), sampling,
                  _binding_id=text(record["_binding_id"]))


def episode_record(episode: Episode) -> dict[str, object]:
    """Retain exact turns, observations, likelihoods and private collection origin."""
    return asdict(episode)


def episode_from_record(value: object) -> Episode:
    """Read an episode without reconstructing actions from rendered text."""
    record = object_record(value)
    identity = object_record(record["identity"])
    initial, reward = record["initial"], record["reward"]
    transitions = []
    for entry in sequence(record["transitions"]):
        transition = object_record(entry)
        transitions.append(Transition(action_record(transition["action"]),
                                      observation_record(transition["observation"])))
    return Episode(EpisodeId(integer(identity["task"]), integer(identity["attempt"]),
                             integer(identity["sample"]), tuple(integer(token) for token in sequence(identity["seed"]))),
                   integer(record["policy_step"]), None if initial is None else observation_record(initial),
                   tuple(transitions), EpisodeStatus(integer(record["status"])), text(record["detail"]),
                   None if reward is None else real(reward), _binding_id=text(record["_binding_id"]))
