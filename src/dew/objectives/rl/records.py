"""JSON-compatible episode records for interchange and turn-boundary recovery."""

import functools
import json
from collections.abc import Mapping
from dataclasses import asdict

import numpy as np

from dew._records import finite, integer, integers, listed, record as section, text
from dew.inference.rollouts import Draw
from dew.sampling.text import Sampling

from .episodes import Episode, EpisodeId, EpisodeStatus, Observation, Transition

reals = functools.partial(listed, read=finite, what="a list of finite numbers")
"""One list of finite numbers: a row's likelihoods (`listed`)."""


def record_json(value: Episode | Draw) -> str:
    """`value` as JSON, its fields as `asdict` writes them and a draw's routing as nested lists of
    expert ids: the text `episode_from_record` and `draw_record` read back."""
    return json.dumps(asdict(value), allow_nan=False, default=np.ndarray.tolist)


def observation_record(value: Mapping[str, object]) -> Observation:
    record = section(value, "observation")
    reward = record["reward"]
    return Observation(integers(record["context"], "context"),
                       EpisodeStatus(integer(record["status"], "status")), text(record["detail"], "detail"),
                       None if reward is None else finite(reward, "reward"),
                       messages(record["messages"], "messages"), bool(record.get("compacted", False)))


messages = functools.partial(listed, read=section, what="a list of chat messages")
"""A list of chat messages, each a JSON object, kept as written (`listed`)."""


def draw_record(value: Mapping[str, object], sampling: Sampling) -> Draw:
    """Read a draw `asdict` wrote, checked to end as `sampling`, its episode's policy, ends one,
    with the engine records it carries (routing as nested lists of expert ids, kept ids per
    sampled id)."""
    record = section(value, "draw")
    raw, routed, support = record["raw_log_probs"], record["routed_experts"], record["support"]
    return Draw(integers(record["prompt"], "prompt"), integers(record["tokens"], "tokens"),
                reals(record["behavior_log_probs"], "behavior_log_probs"),
                text(record["finish_reason"], "finish_reason"), integer(record["version"], "version"),
                None if raw is None else reals(raw, "raw_log_probs"),
                None if routed is None else np.asarray(routed),
                None if support is None else listed(support, "support", read=integers,
                                                    what="a list of token-id lists")).checked(sampling)


def sampling_record(value: Mapping[str, object]) -> Sampling:
    """Read the sampling policy `asdict` wrote, every control included."""
    from dew.registry import from_record

    return from_record(Sampling, value, dtypes=False)


def episode_from_record(value: Mapping[str, object]) -> Episode:
    """Read an episode `asdict` wrote, without reconstructing actions from rendered text."""
    record = section(value, "episode")
    identity = section(record["identity"], "identity")
    initial, reward = record["initial"], record["reward"]
    sampling = sampling_record(section(record["sampling"], "sampling"))
    transitions = tuple(
        Transition(draw_record(section(turn["action"], "action"), sampling),
                   observation_record(section(turn["observation"], "observation")),
                   None if turn.get("request") is None else section(turn["request"], "request"))
        for turn in listed(record["transitions"], "transitions", read=section, what="a list of transitions"))
    branch = identity.get("branch")
    return Episode(EpisodeId(integer(identity["task"], "task"), integer(identity["attempt"], "attempt"),
                             integer(identity["sample"], "sample"), integers(identity["seed"], "seed"),
                             None if branch is None else text(branch, "branch")),
                   integer(record["policy_step"], "policy_step"), sampling,
                   None if initial is None else observation_record(section(initial, "initial")),
                   transitions, EpisodeStatus(integer(record["status"], "status")),
                   text(record["detail"], "detail"),
                   None if reward is None else finite(reward, "reward"),
                   _binding_id=text(record["_binding_id"], "_binding_id"))
