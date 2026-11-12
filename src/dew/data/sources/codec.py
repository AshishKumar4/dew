"""The ArrayRecord byte-map layout and the artifact record codec."""

from __future__ import annotations

import dataclasses
import datetime
import hashlib
import importlib
import io
import json
import math
import re
import struct
from collections.abc import Iterable, Mapping
from decimal import Decimal
from typing import TYPE_CHECKING

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


def _dtype(value: np.ndarray) -> dict[str, str]:
    name = dtype_name(value.dtype)
    return {} if name == value.dtype.str else {"dtype": name}


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


def _array(node: dict, leaves: Mapping[str, bytes], kind: str) -> np.ndarray:
    value = _saved(leaves[str(node[kind])])
    return as_named(value, node["dtype"]) if "dtype" in node else value


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


class _Stored(dict[str, bytes]):
    """The leaves an encoded record stores by key: a bytes value, or an array's NPY file."""

    opaque = False

    def add(self, value: bytes | np.ndarray) -> str:
        key = str(len(self))
        if isinstance(value, bytes):
            self[key] = value
        else:
            file = io.BytesIO()
            np.save(file, value, allow_pickle=False)
            self[key] = file.getvalue()
        return key


class _Writer:
    """A file that hashes what is written to it, as NumPy writes an array's elements, in chunks."""

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
        else:
            if value.flags.f_contiguous and not value.flags.c_contiguous:
                value = value.copy(order="C")
            writer = _Writer()
            np.save(writer, value, allow_pickle=False)
            self.hashes.append(writer.digest.digest())
        return str(len(self.hashes) - 1)


def _stored_array(value: np.ndarray, leaves: _Stored | _Hashed) -> JSON:
    """An array's leaf, a dtype NumPy's plain form misnames viewed as its storage."""
    dtype = _dtype(value)
    return leaves.add(value.view(np.dtype(value.dtype.str)) if dtype else value)


_TIMES = (datetime.datetime, datetime.date, datetime.time)
"""The datetime types a record holds by their ISO text, the most specific first (a datetime is a date)."""


def _tagged(value: Value, leaves: _Stored | _Hashed, canonical: bool) -> dict[str, JSON] | None:
    """The node of a value of the artifact schema's tagged types beyond JSON's own, tried in this order: a
    non-finite float by its hex, a complex by its two parts exactly (as a record holds a Python complex
    beside its other fields), a time by its ISO text, a timedelta by its parts, a decimal by its text,
    bytes as a leaf, an object array by its shape and values, an array as a leaf beside its dtype's name, a
    tuple and a list by their entries; None for any other value (`_restore` reads each back)."""
    if isinstance(value, float) and not math.isfinite(value):
        return {"float": value.hex()}
    if isinstance(value, complex):
        return {"complex": [float(value.real).hex(), float(value.imag).hex()]}
    for kind in _TIMES:
        if isinstance(value, kind):
            return {kind.__name__: value.isoformat()}
    if isinstance(value, datetime.timedelta):
        return {"timedelta": [value.days, value.seconds, value.microseconds]}
    if isinstance(value, Decimal):
        return {"decimal": str(value)}
    if isinstance(value, bytes):
        return {"bytes": leaves.add(value)}
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            return {"object_array": {"shape": list(value.shape),
                                     "values": _tree(value.tolist(), leaves, canonical)}}
        return {"array": _stored_array(value, leaves), **_dtype(value)}
    if isinstance(value, tuple | list):
        return {"tuple" if isinstance(value, tuple) else "list": [_tree(entry, leaves, canonical)
                                                                  for entry in value]}
    return None


def _untagged(node: Mapping[str, JSON], leaves: Mapping[str, bytes]) -> tuple[Value] | None:
    """The value a tagged node holds (`_tagged`), as a one-tuple, so a value of None reads as itself; None
    where the node holds none of the tags read back (an array is read by `_restore` itself)."""
    if "float" in node:
        return (float.fromhex(text(node["float"], "artifact float")),)
    if "complex" in node:
        real, imaginary = _pair(node["complex"], "artifact complex")
        return (complex(float.fromhex(text(real, "artifact complex")),
                        float.fromhex(text(imaginary, "artifact complex"))),)
    for kind in _TIMES:
        if kind.__name__ in node:
            return (kind.fromisoformat(text(node[kind.__name__], f"artifact {kind.__name__}")),)
    if "timedelta" in node:
        parts = node["timedelta"]
        if not isinstance(parts, list):
            raise ValueError("artifact timedelta is not its parts")
        return (datetime.timedelta(*(integer(part, "artifact timedelta") for part in parts)),)
    if "decimal" in node:
        return (Decimal(text(node["decimal"], "artifact decimal")),)
    if "bytes" in node:
        return (leaves[str(node["bytes"])],)
    if "object_array" in node:
        return (_object_array_decoded(node["object_array"], leaves),)
    for name, kind in (("tuple", tuple), ("list", list)):
        if isinstance(entries := node.get(name), list):
            return (kind(_restore(value, leaves) for value in entries),)
    return None


def _pair(value: JSON, key: str) -> tuple[JSON, JSON]:
    """`value`, a node's list of two parts, refused otherwise, naming `key`."""
    match value:
        case [first, second]:
            return first, second
        case _:
            raise ValueError(f"{key} is not two parts")


def _dataclass_decoded(identity: list[JSON], fields: Mapping[str, JSON],
                       leaves: Mapping[str, bytes]) -> Value:
    module, name = _pair(identity, "artifact dataclass")
    if not isinstance(module, str) or not module.startswith("dew.") or not isinstance(name, str):
        raise ValueError("artifact dataclass is outside the Dew schema")
    cls = getattr(importlib.import_module(module), name)
    if not isinstance(cls, type) or not dataclasses.is_dataclass(cls):
        raise ValueError(f"artifact type {module}.{name} is not a dataclass")
    return cls(**{key: _restore(value, leaves) for key, value in fields.items()})


def _object_array_decoded(payload: JSON, leaves: Mapping[str, bytes]) -> Value:
    match payload:
        case {"shape": list(shape), "values": values}:
            return np.asarray(_restore(values, leaves), dtype=object).reshape(
                [integer(size, "artifact object array shape") for size in shape])
        case _:
            raise ValueError("artifact object array has no shape")


def _tree(value: Value, leaves: _Stored | _Hashed, canonical: bool = False) -> JSON:
    if isinstance(value, jax.Array):
        # A record a caller built with jax holds its values on a device; their host copy is the record.
        value = np.asarray(value)
    if leaves.opaque and isinstance(value, complex):
        value = np.complex128(value)
    if isinstance(value, np.generic):
        array = np.asarray(value)
        return {"scalar": _stored_array(array, leaves), **_dtype(array)}
    if (tagged := _tagged(value, leaves, canonical)) is not None:
        return tagged
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        if leaves.opaque and not all(isinstance(key, str) for key in value):
            # Keys of any type, each read as content beside its entry, so no two keys read alike.
            entries = sorted(value.items(), key=lambda entry: repr(entry[0])) if canonical else value.items()
            return {"entries": [[_tree(key, leaves, canonical), _tree(entry, leaves, canonical)]
                                for key, entry in entries]}
        if not all(isinstance(key, str) for key in value):
            raise Unencodable("artifact mappings need string keys")
        keys = sorted(value) if canonical else value
        return {"mapping": {key: _tree(value[key], leaves, canonical) for key in keys}}
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        cls = type(value)
        if not cls.__module__.startswith("dew.") and not leaves.opaque:
            raise Unencodable(f"artifact codec has no schema for {cls.__module__}.{cls.__qualname__}")
        return {
            "dataclass": [cls.__module__, cls.__qualname__],
            "fields": {
                field.name: _tree(getattr(value, field.name), leaves, canonical)
                for field in dataclasses.fields(value)
                if field.init
            },
        }
    if leaves.opaque:
        # An object no schema reads is read as the array NumPy makes of it, or where that array
        # holds the object whole, whose text may name its address, by its type alone.
        array = np.asarray(value)
        if array.ndim or not array.dtype.hasobject:
            return _tree(array, leaves, canonical)
        return {"opaque": f"{type(value).__module__}.{type(value).__qualname__}"}
    raise Unencodable(f"artifact codec cannot encode {type(value).__name__}")


def _restore(node: JSON, leaves: Mapping[str, bytes]) -> Value:
    match node:
        case None | bool() | int() | float() | str():
            return node
        case list():
            raise ValueError("artifact node is not a scalar or tagged object")
    if "scalar" in node:
        return _array(node, leaves, "scalar")[()]
    if "array" in node:
        return _array(node, leaves, "array")
    if (untagged := _untagged(node, leaves)) is not None:
        return untagged[0]
    values = node.get("mapping")
    if isinstance(values, dict):
        return {key: _restore(value, leaves) for key, value in values.items()}
    identity, fields = node.get("dataclass"), node.get("fields")
    if isinstance(identity, list) and len(identity) == 2 and isinstance(fields, dict):
        return _dataclass_decoded(identity, fields, leaves)
    raise ValueError("unknown artifact record schema")


def _schema(tree: JSON) -> bytes:
    return json.dumps(tree, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()


def encode(value: Value, *, canonical: bool = False) -> bytes:
    leaves = _Stored()
    tree = _tree(value, leaves, canonical)
    leaves["schema"] = _schema(tree)
    return pack_dict_of_byte_arrays(leaves)


def decode(value: bytes) -> Value:
    leaves = unpack_dict_of_byte_arrays(value)
    return _restore(json.loads(leaves.pop("schema")), leaves)


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
