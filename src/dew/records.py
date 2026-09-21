"""One reader per JSON type, for every config a published file hands over.

A decoder config, a scheduler file, a generation config and a processor file
all arrive from `json.loads`, so every field is an unnarrowed value that has
to be tested before the model is built from it. These are the tests. Each one
names the key and shows the value it refused, so a file that disagrees with
the model it claims to describe says which field disagreed.

This module sits under `dew.nn`, `dew.interop` and `dew.diffusion`, which is
why it is here and not beside the `json_value` encoder in
`dew.telemetry.records`: reading a config is not a telemetry concern, and
that encoder runs the other direction, turning a finished run record into
JSON. The `JSON` alias both directions speak is declared here, once.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

type JSON = None | bool | int | float | str | list['JSON'] | dict[str, 'JSON']


def integer(value: object, key: str) -> int:
    """One integer field. `True` is an int to Python and a flag to a config."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key}={value!r}: this field is an integer")
    return value


def number(value: object, key: str) -> float:
    """One real field: an epsilon, a rate, a frequency. An integer is one.

    transformers writes a non-finite bound as `{"__float__": "Infinity"}`
    rather than as JSON's out-of-spec `Infinity` literal, so that record is
    the one spelling of an infinity a config carries, and the only one read
    here; a bare non-finite number reached this field from somewhere else.
    """
    if isinstance(value, Mapping) and "__float__" in value:
        return float(str(value["__float__"]))
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{key}={value!r}: this field is a finite number")
    return float(value)


def boolean(value: object, key: str) -> bool:
    """One flag field, as JSON's true and false."""
    if not isinstance(value, bool):
        raise ValueError(f"{key}={value!r}: this field is a boolean")
    return value


def text(value: object, key: str) -> str:
    """One named field: a dtype, an activation, an architecture."""
    if not isinstance(value, str):
        raise ValueError(f"{key}={value!r}: this field is a string")
    return value


def record(value: object, key: str) -> Mapping[str, object]:
    """One nested section, whose own fields read through the readers above."""
    if not isinstance(value, Mapping) or any(type(name) is not str for name in value):
        raise ValueError(f"{key}={value!r}: this field is a record of named fields")
    return value


def strings(value: object, key: str) -> tuple[str, ...]:
    """One list of names: a layer pattern, an architecture list."""
    if isinstance(value, str) or not isinstance(value, (list, tuple)):
        raise ValueError(f"{key}={value!r}: this field is a list of strings")
    return tuple(text(entry, key) for entry in value)


def integers(value: object, key: str) -> tuple[int, ...]:
    """One list of integers: layer indices, token ids, per-layer widths."""
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{key}={value!r}: this field is a list of integers")
    return tuple(integer(entry, key) for entry in value)


def json_value(value: object, key: str) -> JSON:
    """One field narrowed to the JSON its file carries, section and all.

    A scalar, a list of them or a record of them is what `json.loads` can
    produce; anything else reached this field from somewhere other than the
    file, and the key names which field it was.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [json_value(entry, key) for entry in value]
    if isinstance(value, Mapping):
        return {text(name, f"{key} record key"): json_value(entry, key)
                for name, entry in value.items()}
    raise ValueError(f"{key}={value!r}: this field is not a value a JSON config carries")
