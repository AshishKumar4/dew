"""JSON-compatible episode records for interchange and turn-boundary recovery."""

import math
from collections.abc import Callable, Mapping

from dew.records import JSON, integer, integers, record as section, text
from dew.sampling.text import Sampling

from .episodes import Action, Episode, EpisodeId, EpisodeStatus, Observation, Transition


def sequence[ItemT](value: object, read: Callable[[JSON], ItemT], key: str) -> tuple[ItemT, ...]:
    """Read every entry of a JSON array, each one through `read`."""
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{key}={value!r}: this field is a list")
    return tuple(read(entry) for entry in value)


def real(value: object, key: str) -> float:
    """One score or likelihood, finite as no non-finite one is recorded."""
    if not isinstance(value, (float, int)) or isinstance(value, bool) or not math.isfinite(value):
        raise ValueError(f"{key}={value!r}: this field is a finite number")
    return float(value)


def reals(value: object, key: str) -> tuple[float, ...]:
    """One list of finite numbers: a row's likelihoods."""
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{key}={value!r}: this field is a list of finite numbers")
    return tuple(real(entry, key) for entry in value)


def observation_record(value: Mapping[str, object]) -> Observation:
    record = section(value, "observation")
    return Observation(integers(record["context"], "context"),
                       EpisodeStatus(integer(record["status"], "status")), text(record["detail"], "detail"))


def action_record(value: Mapping[str, object]) -> Action:
    from dew.registry import from_record

    record = section(value, "action")
    terminated = record["terminated"]
    if not isinstance(terminated, bool):
        raise ValueError("episode terminated must be a boolean")
    return Action(integers(record["context"], "context"), integers(record["tokens"], "tokens"),
                  reals(record["raw_log_probs"], "raw_log_probs"),
                  reals(record["behavior_log_probs"], "behavior_log_probs"),
                  terminated, integer(record["policy_step"], "policy_step"),
                  from_record(Sampling, section(record["sampling"], "sampling"), dtypes=False),
                  _binding_id=text(record["_binding_id"], "_binding_id"))


def episode_from_record(value: Mapping[str, object]) -> Episode:
    """Read an episode `asdict` wrote, without reconstructing actions from rendered text."""
    record = section(value, "episode")
    identity = section(record["identity"], "identity")
    initial, reward = record["initial"], record["reward"]
    transitions = tuple(
        Transition(action_record(section(turn["action"], "action")),
                   observation_record(section(turn["observation"], "observation")))
        for turn in sequence(record["transitions"], lambda turn: section(turn, "transition"), "transitions"))
    return Episode(EpisodeId(integer(identity["task"], "task"), integer(identity["attempt"], "attempt"),
                             integer(identity["sample"], "sample"), integers(identity["seed"], "seed")),
                   integer(record["policy_step"], "policy_step"),
                   None if initial is None else observation_record(section(initial, "initial")),
                   transitions, EpisodeStatus(integer(record["status"], "status")),
                   text(record["detail"], "detail"),
                   None if reward is None else real(reward, "reward"),
                   _binding_id=text(record["_binding_id"], "_binding_id"))
