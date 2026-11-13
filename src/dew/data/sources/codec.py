"""The ArrayRecord byte-map layout and the artifact record codec."""

from __future__ import annotations

import dataclasses
import datetime
import hashlib
import importlib
import io
import json
import math
import pickle
import re
import struct
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from decimal import Decimal
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import jax
import numpy as np
import numpy.lib.format as npy_format

from dew._records import JSON, integer, text

if TYPE_CHECKING:
    from _typeshed import DataclassInstance

type Value = (
    JSON
    | bytes
    | complex
    | np.ndarray
    | jax.Array
    | np.generic
    | DataclassInstance
    | list[Value]
    | tuple[Value, ...]
    | Mapping[str, Value]
    | datetime.datetime
    | datetime.date
    | datetime.time
    | datetime.timedelta
    | Decimal
)


class Unencodable(TypeError):
    """A value the artifact codec has no encoding for, as a record a cache keeps meets it."""


def dtype_name(dtype: np.dtype) -> str:
    """The name `np.dtype` restores `dtype` from exactly: its array-protocol string, where that names it,
    else its name (an ML dtype's, bfloat16 or a float8, whose protocol string is a void of its width)."""
    return dtype.str if dtype.fields is not None or np.dtype(dtype.str) == dtype else dtype.name


def as_named(value: np.ndarray, name: str) -> np.ndarray:
    """`value`, an array NPY saved, as the dtype `name` (`dtype_name`) names: an ML dtype's bytes, which NPY
    reads back as a void of their width, viewed as it."""
    dtype = np.dtype(name)
    return value if value.dtype == dtype else value.view(dtype)


_SCHEMA = "schema"
_DTYPE = "dtype"
_FIELDS = "fields"
_SHAPE = "shape"
_VALUES = "values"
_FOLD = "fold"
_ZONE = "zone"
_UNCACHED = "uncached"
_TZNAME = "tzname"
_OPAQUE_TZ = "opaque_tz"
_TEMPORAL_OPTIONS = frozenset({_FOLD, _ZONE, _UNCACHED, _TZNAME})

def _dtype(value: np.ndarray) -> dict[str, str]:
    name = dtype_name(value.dtype)
    return {} if name == value.dtype.str else {_DTYPE: name}

_NPY_HEADER = re.compile(rb"\{'descr': '([<>|=][a-zA-Z][0-9]*)', 'fortran_order': (True|False), "
                         rb"'shape': \(([0-9, ]*)\), \}")
"""The header `np.save` writes for an array of one plain dtype, which `_saved` reads without
evaluating it: a record's arrays are read back one by one, and `np.load` evaluates each header as
Python (`ast.literal_eval`), a tenth of a millisecond each."""


def _saved(encoded: bytes) -> np.ndarray:
    """The array `np.save` wrote as `encoded`: one of a plain dtype read straight from its bytes, any
    other (a structured dtype's header names its fields) by `np.load`."""
    prefix = len(npy_format.MAGIC_PREFIX)
    if encoded.startswith(npy_format.MAGIC_PREFIX) and encoded[prefix] in (1, 2, 3):
        # The format's framing as NumPy states it (`numpy.lib.format`): its magic and version, then a
        # header length of 2 bytes in version 1 and 4 after. NumPy reads the header itself only by
        # evaluating it (`read_array_header_1_0`), the cost this reader keeps off every array.
        width = 2 if encoded[prefix] == 1 else 4
        size = int.from_bytes(encoded[npy_format.MAGIC_LEN:npy_format.MAGIC_LEN + width], "little")
        start = npy_format.MAGIC_LEN + width + size
        header = _NPY_HEADER.fullmatch(encoded[npy_format.MAGIC_LEN + width:start].rstrip())
        # A dtype of no width (`|V0`, `|S0`) has no bytes to read, which `np.load` allocates for.
        if header is not None and np.dtype(header.group(1).decode()).itemsize:
            descr, order, dims = header.groups()
            shape = tuple(int(dim) for dim in dims.split(b",") if dim.strip())
            dtype = np.dtype(descr.decode())
            count = math.prod(shape)
            values = np.frombuffer(encoded, dtype, count=count, offset=start)
            return values.reshape(shape, order="F" if order == b"True" else "C").copy(order="K")
    return np.load(io.BytesIO(encoded), allow_pickle=False)


def _leaf(payload: JSON, leaves: Mapping[str, bytes], what: str) -> bytes:
    key = text(payload, what)
    if key not in leaves:
        raise ValueError(f"missing {what} leaf {key!r}")
    return leaves[key]


def _array(tag: str, payload: JSON, node: Mapping[str, JSON], leaves: Mapping[str, bytes]) -> np.ndarray:
    value = _saved(_leaf(payload, leaves, f"artifact {tag}"))
    return as_named(value, text(node[_DTYPE], f"artifact {tag} dtype")) if _DTYPE in node else value

_LENGTH = struct.Struct("<I")
"""The length before each key and value of a byte map, an unsigned little-endian 32-bit count, which its
writer (`pack_dict_of_byte_arrays`) and its reader (`unpack_dict_of_byte_arrays`) both read."""


def pack_dict_of_byte_arrays(values: Mapping[str, bytes]) -> bytes:
    packed = bytearray()
    for key, value in values.items():
        encoded = key.encode()
        packed.extend(_LENGTH.pack(len(encoded)))
        packed.extend(encoded)
        packed.extend(_LENGTH.pack(len(value)))
        packed.extend(value)
    return bytes(packed)


def unpack_dict_of_byte_arrays(payload: bytes) -> dict[str, bytes]:
    values = {}
    offset = 0
    while offset < len(payload):
        if offset + _LENGTH.size > len(payload):
            raise ValueError("truncated ArrayRecord key length")
        (length,) = _LENGTH.unpack_from(payload, offset)
        offset += _LENGTH.size
        key = payload[offset : offset + length].decode()
        offset += length
        if offset + _LENGTH.size > len(payload):
            raise ValueError(f"truncated ArrayRecord value length for {key!r}")
        (length,) = _LENGTH.unpack_from(payload, offset)
        offset += _LENGTH.size
        if offset + length > len(payload) or key in values:
            raise ValueError(f"invalid ArrayRecord value for {key!r}")
        values[key] = payload[offset : offset + length]
        offset += length
    return values


def _npy_header(value: np.ndarray, *, f_order: bool) -> bytes | None:
    """NPY 1.0 header bytes written by `numpy.lib.format` for a plain dtype (`fields is None`),
    or None when `np.save` is needed."""
    if value.dtype.fields is not None or value.dtype.hasobject or value.dtype.char == "T":
        return None
    header = npy_format.header_data_from_array_1_0(value)
    header["fortran_order"] = f_order
    buffer = io.BytesIO()
    npy_format.write_array_header_1_0(buffer, header)
    return buffer.getvalue()


class _Stored(dict[str, bytes]):
    """The leaves an encoded record stores by key: a bytes value, or an array's NPY file."""

    opaque = False

    def add(self, value: bytes | np.ndarray) -> str:
        key = str(len(self))
        if isinstance(value, bytes):
            self[key] = value
            return key
        f_order = value.flags.f_contiguous and not value.flags.c_contiguous
        header = _npy_header(value, f_order=f_order)
        if header is not None:
            self[key] = header + value.tobytes("F" if f_order else "C")
        else:
            file = io.BytesIO()
            np.save(file, value, allow_pickle=False)
            self[key] = file.getvalue()
        return key


class _Writer:
    """A file that hashes what is written to it, as NumPy writes a structured array's elements."""

    def __init__(self) -> None:
        self.digest = hashlib.sha256()

    def write(self, chunk: bytes) -> None:
        self.digest.update(chunk)


class _Hashed:
    """The leaves a digest of content reads, each hashed as it is written and only its hash kept,
    so a stream's digest holds no copy of what it reads and bounds no leaf's size
    (`content_fingerprint` with `opaque`). An array is read in C order, its storage order no part of
    its content; an object the codec has no schema for is read as the array NumPy makes of it, else
    by its type alone."""

    opaque = True

    def __init__(self) -> None:
        self.hashes: list[bytes] = []

    def add(self, value: bytes | np.ndarray) -> str:
        if isinstance(value, bytes):
            self.hashes.append(hashlib.sha256(value).digest())
            return str(len(self.hashes) - 1)
        header = _npy_header(value, f_order=False)
        if header is not None:
            contiguous = (
                value.copy(order="C")
                if (value.flags.f_contiguous and not value.flags.c_contiguous)
                else np.ascontiguousarray(value)
            )
            digest = hashlib.sha256(header)
            if contiguous.size and contiguous.itemsize:
                digest.update(memoryview(contiguous.reshape(-1).view(np.uint8)))
            self.hashes.append(digest.digest())
        else:
            if value.flags.f_contiguous and not value.flags.c_contiguous:
                value = value.copy(order="C")
            writer = _Writer()
            np.save(writer, value, allow_pickle=False)
            self.hashes.append(writer.digest.digest())
        return str(len(self.hashes) - 1)


type _Leaves = _Stored | _Hashed


def _stored_array(value: np.ndarray, leaves: _Leaves) -> JSON:
    """An array's leaf, a dtype NumPy's plain form misnames viewed as its storage."""
    dtype = _dtype(value)
    return leaves.add(value.view(np.dtype(value.dtype.str)) if dtype else value)


def _pair(value: JSON, key: str) -> tuple[JSON, JSON]:
    """`value`, a node's list of two parts, refused otherwise, naming `key`."""
    match value:
        case [first, second]:
            return first, second
        case _:
            raise ValueError(f"{key} is not two parts")


def _scalar_decoded(tag: str, payload: JSON, node: Mapping[str, JSON],
                    leaves: Mapping[str, bytes]) -> Value:
    return _array(tag, payload, node, leaves)[()]

def _complex_encoded(tag: str, value: complex, leaves: _Leaves, canonical: bool) -> dict[str, JSON]:
    return {tag: [float(value.real).hex(), float(value.imag).hex()]}


def _complex_decoded(tag: str, payload: JSON, node: Mapping[str, JSON],
                     leaves: Mapping[str, bytes]) -> complex:
    real, imaginary = _pair(payload, f"artifact {tag}")
    return complex(
        float.fromhex(text(real, f"artifact {tag}")),
        float.fromhex(text(imaginary, f"artifact {tag}")),
    )


def _temporal_encoded(tag: str, value: datetime.datetime | datetime.time, leaves: _Leaves,
                      canonical: bool) -> dict[str, JSON]:
    encoded: dict[str, JSON] = {tag: value.isoformat()}
    tz = value.tzinfo
    if isinstance(tz, datetime.timezone) and type(tz) is datetime.timezone:
        reduced = tz.__reduce__()[1]
        if len(reduced) == 2:
            encoded[_TZNAME] = text(reduced[1], f"artifact {tag} tzname")
    elif isinstance(tz, ZoneInfo) and type(tz) is ZoneInfo:
        try:
            reduced_zone = tz.__reduce__()[1]
        except pickle.PicklingError as error:
            if not leaves.opaque:
                raise Unencodable("artifact codec cannot encode a file-stream ZoneInfo") from error
            encoded[_OPAQUE_TZ] = f"{type(tz).__module__}.{type(tz).__qualname__}:{tz.key}"
        else:
            if tz.key is None:
                raise Unencodable("artifact codec cannot encode an unnamed ZoneInfo")
            encoded[_ZONE] = tz.key
            if len(reduced_zone) >= 2 and not reduced_zone[1]:
                encoded[_UNCACHED] = True
    elif tz is not None:
        if not leaves.opaque:
            raise Unencodable(
                f"artifact codec has no timezone schema for {type(tz).__module__}.{type(tz).__qualname__}"
            )
        encoded[_OPAQUE_TZ] = f"{type(tz).__module__}.{type(tz).__qualname__}"
    if value.fold:
        encoded[_FOLD] = value.fold
    return encoded


def _temporal_decoded[TemporalT: (datetime.datetime, datetime.time)](
    kind: type[TemporalT], tag: str, payload: JSON, node: Mapping[str, JSON]
) -> TemporalT:
    """Restore `kind` from its ISO text and optional timezone/fold fields; earlier ISO-only records carry
    neither and read back with `fromisoformat`'s fixed offset (or naive time) and `fold=0`."""
    parsed = kind.fromisoformat(text(payload, f"artifact {tag}"))
    if _ZONE in node and _TZNAME in node:
        raise ValueError(f"artifact {tag} names both zone and tzname")
    if _UNCACHED in node and _ZONE not in node:
        raise ValueError(f"artifact {tag} marks uncached without zone")
    tz: datetime.tzinfo | None = parsed.tzinfo
    if _ZONE in node:
        zone = text(node[_ZONE], f"artifact {tag} zone")
        uncached = node.get(_UNCACHED, False)
        if not isinstance(uncached, bool):
            raise ValueError(f"artifact {tag} uncached must be a boolean")
        try:
            tz = ZoneInfo.no_cache(zone) if uncached else ZoneInfo(zone)
        except (ZoneInfoNotFoundError, ValueError) as error:
            raise ValueError(f"unknown artifact {tag} zone {zone!r}") from error
    elif _TZNAME in node:
        name = text(node[_TZNAME], f"artifact {tag} tzname")
        offset = None if parsed.tzinfo is None else parsed.tzinfo.utcoffset(None)
        if offset is None:
            raise ValueError(f"artifact {tag} tzname has no offset")
        tz = datetime.timezone(offset, name)
    fold = integer(node[_FOLD], f"artifact {tag} fold") if _FOLD in node else 0
    if fold not in (0, 1):
        raise ValueError(f"artifact {tag} fold must be 0 or 1, got {fold}")
    if tz is parsed.tzinfo and fold == parsed.fold:
        return parsed
    return parsed.replace(tzinfo=tz, fold=fold)

def _timedelta_decoded(tag: str, payload: JSON, node: Mapping[str, JSON],
                       leaves: Mapping[str, bytes]) -> datetime.timedelta:
    match payload:
        case [days, seconds, microseconds]:
            return datetime.timedelta(
                integer(days, f"artifact {tag}"),
                integer(seconds, f"artifact {tag}"),
                integer(microseconds, f"artifact {tag}"),
            )
        case _:
            raise ValueError(f"artifact {tag} is not its parts")

def _object_entries(restored: Value, dims: Sequence[int]) -> Iterator[Value]:
    if not dims:
        yield restored
        return
    if not isinstance(restored, list) or len(restored) != dims[0]:
        raise ValueError("artifact object array values do not match its shape")
    for entry in restored:
        yield from _object_entries(entry, dims[1:])


def _object_array_encoded(tag: str, value: np.ndarray, leaves: _Leaves, canonical: bool) -> dict[str, JSON]:
    if value.dtype != np.dtype(object) and not leaves.opaque:
        raise Unencodable(f"artifact codec cannot encode structured object dtype {value.dtype}")
    return {
        tag: {
            _SHAPE: list(value.shape),
            _VALUES: _tree(value.tolist(), leaves, canonical=canonical),
        }
    }


def _object_array_decoded(tag: str, payload: JSON, node: Mapping[str, JSON],
                          leaves: Mapping[str, bytes]) -> np.ndarray:
    match payload:
        case dict() if set(payload) == {_SHAPE, _VALUES}:
            raw_shape, raw_values = payload[_SHAPE], payload[_VALUES]
        case _:
            raise ValueError("artifact object array has no shape")
    match raw_shape:
        case list():
            dims = tuple(integer(size, "artifact object array shape") for size in raw_shape)
        case _:
            raise ValueError("artifact object array has no shape")
    entries = list(_object_entries(_restore(raw_values, leaves), dims))
    out = np.empty(dims, dtype=object)
    if not dims:
        out[()] = entries[0]
    else:
        flat = out.reshape(-1)
        for index, entry in enumerate(entries):
            flat[index] = entry
    return out


def _sequence_decoded(tag: str, payload: JSON, leaves: Mapping[str, bytes]) -> list[Value]:
    match payload:
        case list():
            return [_restore(entry, leaves) for entry in payload]
        case _:
            raise ValueError(f"artifact {tag} is not a list")

def _mapping_encoded(tag: str, value: Mapping, leaves: _Leaves, canonical: bool) -> dict[str, JSON]:
    if not all(isinstance(key, str) for key in value):
        raise Unencodable("artifact mappings need string keys")
    keys = sorted(value) if canonical else value
    return {tag: {key: _tree(value[key], leaves, canonical=canonical) for key in keys}}


def _mapping_decoded(tag: str, payload: JSON, node: Mapping[str, JSON],
                     leaves: Mapping[str, bytes]) -> dict[str, Value]:
    match payload:
        case dict():
            return {key: _restore(entry, leaves) for key, entry in payload.items()}
        case _:
            raise ValueError(f"artifact {tag} is not a mapping")

def _dataclass_encoded(tag: str, value: object, leaves: _Leaves, canonical: bool) -> dict[str, JSON]:
    if not dataclasses.is_dataclass(value) or isinstance(value, type):
        raise Unencodable(f"artifact codec cannot encode {type(value).__name__}")
    cls = type(value)
    if not cls.__module__.startswith("dew.") and not leaves.opaque:
        raise Unencodable(f"artifact codec has no schema for {cls.__module__}.{cls.__qualname__}")
    return {
        tag: [cls.__module__, cls.__qualname__],
        _FIELDS: {
            field.name: _tree(getattr(value, field.name), leaves, canonical=canonical)
            for field in dataclasses.fields(value)
            if field.init
        },
    }


def _dataclass_decoded(tag: str, payload: JSON, node: Mapping[str, JSON],
                       leaves: Mapping[str, bytes]) -> Value:
    module, name = _pair(payload, f"artifact {tag}")
    fields = node.get(_FIELDS)
    if not isinstance(fields, dict):
        raise ValueError(f"artifact {tag} has no fields")
    if not isinstance(module, str) or not module.startswith("dew.") or not isinstance(name, str):
        raise ValueError(f"artifact {tag} is outside the Dew schema")
    cls = getattr(importlib.import_module(module), name)
    if not isinstance(cls, type) or not dataclasses.is_dataclass(cls):
        raise ValueError(f"artifact type {module}.{name} is not a dataclass")
    return cls(**{key: _restore(entry, leaves) for key, entry in fields.items()})


@dataclasses.dataclass(frozen=True)
class _Variant[T]:
    """One wire variant of the artifact schema: its discriminator `tag`, the Python `types` and
    optional `guard` it matches, the companion keys its node may carry (`options`), and its paired
    encoder and decoder."""

    tag: str
    types: type[T]
    write: Callable[[str, T, _Leaves, bool], dict[str, JSON]]
    read: Callable[[str, JSON, Mapping[str, JSON], Mapping[str, bytes]], Value]
    guard: Callable[[T], bool] | None = None
    options: frozenset[str] = frozenset()
    fast_type: type[T] | None = None

    @property
    def exact_type(self) -> type | None:
        if self.fast_type is not None:
            return self.fast_type
        if self.types in (object, np.generic):
            return None
        return self.types

    def try_encode(self, value: Value, leaves: _Leaves, *, canonical: bool) -> dict[str, JSON] | None:
        if (
            (self.fast_type is not None and isinstance(value, self.fast_type))
            or isinstance(value, self.types)
        ) and (self.guard is None or self.guard(value)):
            return self.write(self.tag, value, leaves, canonical)
        return None

    def decode(self, node: Mapping[str, JSON], leaves: Mapping[str, bytes]) -> Value:
        if not (node.keys() <= {self.tag, *self.options}):
            raise ValueError(f"unknown fields in artifact {self.tag} record")
        return self.read(self.tag, node[self.tag], node, leaves)


# Order matters where Python types overlap: `np.generic` precedes `float`, `complex` and `bytes`;
# `datetime` precedes its base `date`; `object_array` precedes dense `array`.
_VARIANTS: tuple[_Variant, ...] = (
    _Variant(
        "scalar",
        np.generic,
        lambda tag, value, leaves, canonical: {
            tag: _stored_array(array := np.asarray(value), leaves),
            **_dtype(array),
        },
        _scalar_decoded,
        options=frozenset({_DTYPE}),
    ),
    _Variant(
        "float",
        float,
        lambda tag, value, leaves, canonical: {tag: value.hex()},
        lambda tag, payload, node, leaves: float.fromhex(text(payload, f"artifact {tag}")),
        guard=lambda value: not math.isfinite(value),
    ),
    _Variant("complex", complex, _complex_encoded, _complex_decoded),
    _Variant(
        "datetime",
        datetime.datetime,
        _temporal_encoded,
        lambda tag, payload, node, leaves: _temporal_decoded(datetime.datetime, tag, payload, node),
        options=_TEMPORAL_OPTIONS,
    ),
    _Variant(
        "date",
        datetime.date,
        lambda tag, value, leaves, canonical: {tag: value.isoformat()},
        lambda tag, payload, node, leaves: datetime.date.fromisoformat(text(payload, f"artifact {tag}")),
    ),
    _Variant(
        "time",
        datetime.time,
        _temporal_encoded,
        lambda tag, payload, node, leaves: _temporal_decoded(datetime.time, tag, payload, node),
        options=_TEMPORAL_OPTIONS,
    ),
    _Variant(
        "timedelta",
        datetime.timedelta,
        lambda tag, value, leaves, canonical: {tag: [value.days, value.seconds, value.microseconds]},
        _timedelta_decoded,
    ),
    _Variant(
        "decimal",
        Decimal,
        lambda tag, value, leaves, canonical: {tag: str(value)},
        lambda tag, payload, node, leaves: Decimal(text(payload, f"artifact {tag}")),
    ),
    _Variant(
        "bytes",
        bytes,
        lambda tag, value, leaves, canonical: {tag: leaves.add(value)},
        lambda tag, payload, node, leaves: _leaf(payload, leaves, f"artifact {tag}"),
    ),
    _Variant(
        "object_array",
        np.ndarray,
        _object_array_encoded,
        _object_array_decoded,
        guard=lambda value: value.dtype.hasobject,
    ),
    _Variant(
        "array",
        np.ndarray,
        lambda tag, value, leaves, canonical: {tag: _stored_array(value, leaves), **_dtype(value)},
        _array,
        options=frozenset({_DTYPE}),
    ),
    _Variant[tuple[Value, ...]](
        "tuple",
        tuple,
        lambda tag, value, leaves, canonical: {
            tag: [_tree(entry, leaves, canonical=canonical) for entry in value]
        },
        lambda tag, payload, node, leaves: tuple(_sequence_decoded(tag, payload, leaves)),
    ),
    _Variant[list[Value]](
        "list",
        list,
        lambda tag, value, leaves, canonical: {
            tag: [_tree(entry, leaves, canonical=canonical) for entry in value]
        },
        lambda tag, payload, node, leaves: _sequence_decoded(tag, payload, leaves),
    ),
    _Variant("mapping", Mapping, _mapping_encoded, _mapping_decoded, fast_type=dict),
    _Variant(
        "dataclass",
        object,
        _dataclass_encoded,
        _dataclass_decoded,
        guard=lambda value: dataclasses.is_dataclass(value) and not isinstance(value, type),
        options=frozenset({_FIELDS}),
    ),
)
_BY_TAG: Mapping[str, _Variant] = {variant.tag: variant for variant in _VARIANTS}
"""Every wire variant indexed by its discriminator tag, derived from `_VARIANTS`."""


def _exact_candidates(variants: Sequence[_Variant]) -> Mapping[type, tuple[_Variant, ...]]:
    grouped: dict[type, list[_Variant]] = {}
    for variant in variants:
        if (exact := variant.exact_type) is not None:
            grouped.setdefault(exact, []).append(variant)
    return {exact: tuple(entries) for exact, entries in grouped.items()}


_BY_EXACT_TYPE: Mapping[type, tuple[_Variant, ...]] = _exact_candidates(_VARIANTS)
"""Ordered candidate encoders per concrete type derived from `_VARIANTS`."""


def _non_string_mapping(value: Value) -> bool:
    if isinstance(value, dict):
        return not all(type(key) is str or isinstance(key, str) for key in value)
    return isinstance(value, Mapping) and not all(isinstance(key, str) for key in value)


def _tree(value: Value, leaves: _Leaves, *, canonical: bool = False) -> JSON:
    cls = type(value)
    if value is None or (
        cls in (bool, int, str) and isinstance(value, (bool, int, str))
    ) or (cls is float and isinstance(value, float) and math.isfinite(value)):
        return value
    if isinstance(value, jax.Array):
        # A record a caller built with jax holds its values on a device; their host copy is the record.
        value = np.asarray(value)
        cls = type(value)
    if leaves.opaque and isinstance(value, complex) and not isinstance(value, np.generic):
        value = np.complex128(value)
        cls = type(value)
    if leaves.opaque and _non_string_mapping(value):
        assert isinstance(value, Mapping)
        # Keys of any type, each read as content beside its entry, so no two keys read alike.
        entries = sorted(value.items(), key=lambda entry: repr(entry[0])) if canonical else value.items()
        return {
            "entries": [
                [_tree(key, leaves, canonical=canonical), _tree(entry, leaves, canonical=canonical)]
                for key, entry in entries
            ]
        }
    if (candidates := _BY_EXACT_TYPE.get(cls)) is not None:
        for candidate in candidates:
            if (encoded := candidate.try_encode(value, leaves, canonical=canonical)) is not None:
                return encoded
    else:
        for variant in _VARIANTS:
            if (encoded := variant.try_encode(value, leaves, canonical=canonical)) is not None:
                return encoded
    if isinstance(value, (bool, int, float, str)):
        return value
    if leaves.opaque:
        # An object no schema reads is read as the array NumPy makes of it, or where that array
        # holds the object whole, whose text may name its address, by its type alone.
        array = np.asarray(value)
        if array.ndim or not array.dtype.hasobject:
            return _tree(array, leaves, canonical=canonical)
        return {"opaque": f"{type(value).__module__}.{type(value).__qualname__}"}
    raise Unencodable(f"artifact codec cannot encode {type(value).__name__}")


def _restore(node: JSON, leaves: Mapping[str, bytes]) -> Value:
    match node:
        case None | bool() | int() | float() | str():
            return node
        case dict():
            matched = [_BY_TAG[key] for key in node if key in _BY_TAG]
            if len(matched) != 1:
                raise ValueError("unknown artifact record schema")
            return matched[0].decode(node, leaves)
        case _:
            raise ValueError("artifact node is not a scalar or tagged object")


def _schema(tree: JSON) -> bytes:
    return json.dumps(tree, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()


def encode(value: Value, *, canonical: bool = False) -> bytes:
    leaves = _Stored()
    tree = _tree(value, leaves, canonical=canonical)
    leaves[_SCHEMA] = _schema(tree)
    return pack_dict_of_byte_arrays(leaves)


def decode(value: bytes) -> Value:
    leaves = unpack_dict_of_byte_arrays(value)
    if _SCHEMA not in leaves:
        raise ValueError("artifact record has no schema")
    return _restore(json.loads(leaves.pop(_SCHEMA)), leaves)


def content_fingerprint(value: Value, *, opaque: bool = False) -> bytes:
    """SHA256 of content with every mapping ordered by key. `opaque` reads any object, a mapping's
    keys of any type and an array in any storage order, hashing each leaf as it is read (`_Hashed`),
    for a digest that guards a stream rather than an artifact anyone decodes
    (`dew.data.dataset.Replayed`)."""
    if not opaque:
        return hashlib.sha256(encode(value, canonical=True)).digest()
    leaves = _Hashed()
    schema = _schema(_tree(value, leaves, canonical=True))
    digest = hashlib.sha256(struct.pack("<Q", len(schema)) + schema)
    for leaf in leaves.hashes:
        digest.update(leaf)
    return digest.digest()


def fingerprint(values: Iterable[Value]) -> str:
    return encoded_fingerprint(encode(value) for value in values)


def encoded_fingerprint(encoded: Iterable[bytes]) -> str:
    """`fingerprint` of the values `encoded` holds, each as `encode` wrote it, for a caller that holds
    their encodings already."""
    digest = hashlib.sha256()
    for each in encoded:
        digest.update(struct.pack("<Q", len(each)))
        digest.update(each)
    return digest.hexdigest()
