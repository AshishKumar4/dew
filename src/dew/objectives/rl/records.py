"""JSON-compatible episode records for interchange and turn-boundary recovery."""

import functools
import json
from collections.abc import Mapping
from dataclasses import asdict

import numpy as np

from dew._records import finite, listed, record as section
from dew.inference.rollouts import Draw
from dew.registry import from_record
from dew.sampling.text import Sampling

from .episodes import Episode
from .sessions import json_object

reals = functools.partial(listed, read=finite, what="a list of finite numbers")
"""One list of finite numbers: a row's likelihoods (`listed`), as an interchange record states them."""

messages = functools.partial(listed, read=json_object, what="a list of chat messages")
"""A list of chat messages, each a JSON object (`listed`, `json_object`)."""


def record_json(value: Episode | Draw) -> str:
    """`value` as JSON, its fields as `asdict` writes them and a draw's routing as nested lists of
    expert ids: the text `episode_from_record` and `draw_record` read back."""
    return json.dumps(asdict(value), allow_nan=False, default=_json_leaf)


def _json_leaf(value: np.ndarray | Mapping[str, np.ndarray]) -> list | dict[str, np.ndarray]:
    """A record's leaf JSON writes no other way: an array as nested lists, a routing record keyed by site
    (`dew.inference.rollouts.Routing`) as an object of them."""
    return dict(value) if isinstance(value, Mapping) else value.tolist()


def draw_record(value: Mapping[str, object], sampling: Sampling) -> Draw:
    """Read a draw `asdict` wrote, through its declaration (`from_record`), checked to end as `sampling`,
    its episode's policy, ends one."""
    return from_record(Draw, section(value, "draw"), dtypes=False).checked(sampling)


def episode_from_record(value: Mapping[str, object]) -> Episode:
    """Read an episode `asdict` wrote, through its declaration (`from_record`), without reconstructing
    actions from rendered text: each action checked to end as the episode's sampling ends one."""
    episode = from_record(Episode, section(value, "episode"), dtypes=False)
    for transition in episode.transitions:
        transition.action.checked(episode.sampling)
    return episode
