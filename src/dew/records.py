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
import functools
import math
import re
from collections.abc import Callable, Mapping
from numbers import Real

type JSON = None | bool | int | float | str | list['JSON'] | dict[str, 'JSON']



HF_METADATA = frozenset({
    "architectures", "transformers_version", "torch_dtype", "dtype", "output_hidden_states",
    "output_attentions", "return_dict", "is_encoder_decoder", "id2label", "label2id", "problem_type",
    "chunk_size_feed_forward",
})
"""The fields every Hugging Face config serializes that describe no computation: its classes and
release, its output and return flags, its label maps and task. A translator that refuses fields it
does not read leaves these out, beside its own."""

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


def finite(value: object, what: str) -> float:
    """One finite real as a float, NumPy's scalars among them: a reward, a likelihood, a scale.
    A bool is a flag, not a number; it, every non-number, an infinity and a nan are refused with
    ValueError naming `what` and the value."""
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{what} is a finite number, got {value!r}")
    return float(value)


def number(value: object, key: str) -> float:
    """One real field: an epsilon, a rate, a frequency. An integer is one.

    transformers writes an infinite bound as `{"__float__": "Infinity"}` or
    `{"__float__": "-Infinity"}` rather than as JSON's out-of-spec literal
    (`PreTrainedConfig._encode_special_floats`, 5.16.1), so those two
    records are the one spelling of an infinity a config carries, and the
    only ones read here. Its NaN record, any other mapping and a bare
    non-finite number are refused: no field here is NaN.
    """
    if isinstance(value, Mapping):
        if dict(value) not in ({"__float__": "Infinity"}, {"__float__": "-Infinity"}):
            raise ValueError(f"{key}={value!r}: this field is a finite number or transformers' "
                             "record of an infinity")
        return float(str(value["__float__"]))
    return finite(value, key)


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


def listed[ItemT](value: object, key: str, *, read: Callable[[JSON, str], ItemT],
                  what: str) -> tuple[ItemT, ...]:
    """One list, each entry read by `read(entry, key)`; `what` names what the list holds, as a
    refusal of anything else says."""
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{key}={value!r}: this field is {what}")
    return tuple(read(entry, key) for entry in value)


integers = functools.partial(listed, read=integer, what="a list of integers")
"""One list of integers: layer indices, token ids, per-layer widths (`listed`)."""

numbers = functools.partial(listed, read=number, what="a list of numbers")
"""One list of finite numbers: per-frequency rotary factors (`listed`)."""

strings = functools.partial(listed, read=text, what="a list of strings")
"""One list of names: a layer pattern, an architecture list (`listed`)."""


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
