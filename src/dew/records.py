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

import datetime
import math
import re
from collections.abc import Mapping

type JSON = None | bool | int | float | str | list['JSON'] | dict[str, 'JSON']


def duration(value: str) -> datetime.timedelta:
    """A positive recorded duration, in seconds, minutes or hours."""
    match = re.fullmatch(r'(\d+(?:\.\d+)?)(s|m|h)', value)
    if match is None:
        raise ValueError("duration must be positive, such as 30m (s, m and h are supported)")
    whole, _, fraction = match[1].partition('.')
    denominator = 10 ** len(fraction)
    numerator = (
        (int(whole) * denominator + int(fraction or "0")) * {"s": 1, "m": 60, "h": 3600}[match[2]] * 1_000_000
    )
    micros, remainder = divmod(numerator, denominator)
    # timedelta's nearest-microsecond, ties-to-even rounding, without a
    # floating conversion or ambient decimal context.
    if remainder * 2 > denominator or (remainder * 2 == denominator and micros % 2):
        micros += 1
    if micros <= 0:
        raise ValueError("duration must be positive and representable in microseconds")
    return datetime.timedelta(microseconds=micros)


def recorded_duration(value: datetime.timedelta) -> str:
    """One canonical spelling, preserving timedelta's microsecond precision."""
    micros = (value.days * 86400 + value.seconds) * 1_000_000 + value.microseconds
    if micros <= 0:
        raise ValueError("duration must be positive")
    if micros % 3_600_000_000 == 0:
        return f'{micros // 3_600_000_000}h'
    if micros % 60_000_000 == 0:
        return f'{micros // 60_000_000}m'
    seconds, fraction = divmod(micros, 1_000_000)
    tail = f'.{fraction:06d}'.rstrip('0') if fraction else ''
    return f'{seconds}{tail}s'


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


def numbers(value: object, key: str) -> tuple[float, ...]:
    """One list of finite numbers: per-frequency rotary factors."""
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{key}={value!r}: this field is a list of numbers")
    return tuple(number(entry, key) for entry in value)


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
