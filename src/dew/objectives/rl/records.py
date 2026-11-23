"""JSON-compatible episode records for interchange and turn-boundary recovery."""

import functools
import json
from collections.abc import Mapping
from dataclasses import asdict

from dew._records import finite, listed, record as section
from dew.inference.rollouts import Draw, routing_json
from dew.registry import from_record, to_record
from dew.sampling.text import Sampling

from .episodes import Episode
from .sessions import json_object

reals = functools.partial(listed, read=finite, what="a list of finite numbers")
"""One list of finite numbers: a row's likelihoods (`listed`), as an interchange record states them."""

messages = functools.partial(listed, read=json_object, what="a list of chat messages")
"""A list of chat messages, each a JSON object (`listed`, `json_object`)."""


def record_json(value: Episode | Draw) -> str:
    """`value` as JSON, its fields as `asdict` writes them and its arrays, a draw's routing among them, as
    `routing_json` writes a routing record (nested lists, an object of them by site), an episode's policy as
    the registry records it (`to_record`: a function by its import path, Train 1208's review F2): the text
    `episode_from_record` and `draw_record` read back."""
    fields = asdict(value)
    if isinstance(value, Episode):
        fields["sampling"] = to_record(value.sampling, Sampling)
    return json.dumps(fields, allow_nan=False, default=routing_json)


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
