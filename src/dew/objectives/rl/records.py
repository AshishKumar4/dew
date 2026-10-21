"""JSON-compatible episode records for interchange and turn-boundary recovery."""

import functools
from collections.abc import Mapping

from dew.records import finite, integer, integers, listed, record as section, text
from dew.sampling.text import Sampling

from .episodes import Action, Episode, EpisodeId, EpisodeStatus, Observation, Transition

real = finite
"""One score or likelihood, finite as no non-finite one is recorded (`dew.records.finite`)."""


reals = functools.partial(listed, read=real, what="a list of finite numbers")
"""One list of finite numbers: a row's likelihoods (`listed`)."""


def observation_record(value: Mapping[str, object]) -> Observation:
    record = section(value, "observation")
    reward = record["reward"]
    return Observation(integers(record["context"], "context"),
                       EpisodeStatus(integer(record["status"], "status")), text(record["detail"], "detail"),
                       None if reward is None else real(reward, "reward"),
                       messages(record["messages"], "messages"))


messages = functools.partial(listed, read=section, what="a list of chat messages")
"""A list of chat messages, each a JSON object, kept as written (`listed`)."""


def action_record(value: Mapping[str, object]) -> Action:
    from dew.registry import from_record

    record = section(value, "action")
    terminated, raw = record["terminated"], record["raw_log_probs"]
    if not isinstance(terminated, bool):
        raise ValueError("episode terminated must be a boolean")
    return Action(integers(record["context"], "context"), integers(record["tokens"], "tokens"),
                  None if raw is None else reals(raw, "raw_log_probs"),
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
        for turn in listed(record["transitions"], "transitions", read=section, what="a list of transitions"))
    return Episode(EpisodeId(integer(identity["task"], "task"), integer(identity["attempt"], "attempt"),
                             integer(identity["sample"], "sample"), integers(identity["seed"], "seed")),
                   integer(record["policy_step"], "policy_step"),
                   None if initial is None else observation_record(section(initial, "initial")),
                   transitions, EpisodeStatus(integer(record["status"], "status")),
                   text(record["detail"], "detail"),
                   None if reward is None else real(reward, "reward"),
                   _binding_id=text(record["_binding_id"], "_binding_id"))
