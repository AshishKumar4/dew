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

from dew._records import JSON

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


def _dtype(value: np.ndarray) -> dict[str, str]:
    if value.dtype.fields is None and np.dtype(value.dtype.str) != value.dtype:
        return {"dtype": value.dtype.name}
    return {}


_NPY_HEADER = re.compile(rb"\{'descr': '([<>|=][a-zA-Z][0-9]*)', 'fortran_order': (True|False), "
                         rb"'shape': \(([0-9, ]*)\), \}")
"""The header `np.save` writes for an array of one plain dtype, which `_saved` reads without
evaluating it: a record's arrays are read back one by one, and `np.load` evaluates each header as
Python (`ast.literal_eval`), a tenth of a millisecond each."""


def _saved(encoded: bytes) -> np.ndarray:
    """The array `np.save` wrote as `encoded`: one of a plain dtype read straight from its bytes, any
    other (a structured dtype's header names its fields) by `np.load`."""
    if encoded[:6] == b"\x93NUMPY" and encoded[6] in (1, 2, 3):
        width = 2 if encoded[6] == 1 else 4
        size = int.from_bytes(encoded[8:8 + width], "little")
        start = 8 + width + size
        header = _NPY_HEADER.fullmatch(encoded[8 + width:start].rstrip())
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
    if "dtype" in node:
        value = value.view(np.dtype(node["dtype"]))
    return value


def pack_dict_of_byte_arrays(values: Mapping[str, bytes]) -> bytes:
    packed = bytearray()
    for key, value in values.items():
        encoded = key.encode()
        packed.extend(struct.pack("<I", len(encoded)))
        packed.extend(encoded)
        packed.extend(struct.pack("<I", len(value)))
        packed.extend(value)
    return bytes(packed)


def unpack_dict_of_byte_arrays(payload: bytes) -> dict[str, bytes]:
    values = {}
    offset = 0
    while offset < len(payload):
        if offset + 4 > len(payload):
            raise ValueError("truncated ArrayRecord key length")
        length = struct.unpack_from("<I", payload, offset)[0]
        offset += 4
        key = payload[offset : offset + length].decode()
        offset += length
        if offset + 4 > len(payload):
            raise ValueError(f"truncated ArrayRecord value length for {key!r}")
        length = struct.unpack_from("<I", payload, offset)[0]
        offset += 4
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


def _tree(value: Value, leaves: _Stored | _Hashed, canonical: bool = False) -> JSON:
    if isinstance(value, jax.Array):
        # A record a caller built with jax holds its values on a device; their host copy is the record.
        value = np.asarray(value)
    if leaves.opaque and isinstance(value, complex):
        value = np.complex128(value)
    if isinstance(value, np.generic):
        array = np.asarray(value)
        dtype = _dtype(array)
        return {"scalar": leaves.add(array.view(np.dtype(array.dtype.str)) if dtype else array), **dtype}
    if isinstance(value, float) and not math.isfinite(value):
        return {"float": value.hex()}
    if isinstance(value, complex):
        # Its two parts exactly, as a record holds a Python complex beside its other fields.
        return {"complex": [float(value.real).hex(), float(value.imag).hex()]}
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    for kind in (datetime.datetime, datetime.date, datetime.time):
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
        dtype = _dtype(value)
        return {"array": leaves.add(value.view(np.dtype(value.dtype.str)) if dtype else value), **dtype}
    if isinstance(value, Mapping):
        if leaves.opaque and not all(isinstance(key, str) for key in value):
            # Keys of any type, each read as content beside its entry, so no two keys read alike.
            entries = sorted(value.items(), key=lambda entry: repr(entry[0])) if canonical else value.items()
            return {"entries": [[_tree(key, leaves, canonical), _tree(entry, leaves, canonical)]
                                for key, entry in entries]}
        if not all(isinstance(key, str) for key in value):
            raise TypeError("artifact mappings need string keys")
        keys = sorted(value) if canonical else value
        return {"mapping": {key: _tree(value[key], leaves, canonical) for key in keys}}
    if isinstance(value, (list, tuple)):
        return {"tuple" if isinstance(value, tuple) else "list":
                [_tree(entry, leaves, canonical) for entry in value]}
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        cls = type(value)
        if not cls.__module__.startswith("dew.") and not leaves.opaque:
            raise TypeError(f"artifact codec has no schema for {cls.__module__}.{cls.__qualname__}")
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
    raise TypeError(f"artifact codec cannot encode {type(value).__name__}")


def _restore(node: object, leaves: Mapping[str, bytes]) -> Value:
    if node is None or isinstance(node, (bool, int, float, str)):
        return node
    if not isinstance(node, dict):
        raise ValueError("artifact node is not a scalar or tagged object")
    for kind in (datetime.datetime, datetime.date, datetime.time):
        if kind.__name__ in node:
            return kind.fromisoformat(node[kind.__name__])
    if "timedelta" in node:
        return datetime.timedelta(*node["timedelta"])
    if "decimal" in node:
        return Decimal(node["decimal"])
    if "float" in node:
        return float.fromhex(node["float"])
    if "complex" in node:
        real, imag = node["complex"]
        return complex(float.fromhex(real), float.fromhex(imag))
    if "scalar" in node:
        return _array(node, leaves, "scalar")[()]
    if "bytes" in node:
        return leaves[str(node["bytes"])]
    if "array" in node:
        return _array(node, leaves, "array")
    if "object_array" in node:
        value = node["object_array"]
        if not isinstance(value, dict) or not isinstance(value.get("shape"), list):
            raise ValueError("artifact object array has no shape")
        return np.asarray(_restore(value["values"], leaves), dtype=object).reshape(value["shape"])
    for kind in ("list", "tuple"):
        values = node.get(kind)
        if isinstance(values, list):
            restored = [_restore(value, leaves) for value in values]
            return tuple(restored) if kind == "tuple" else restored
    values = node.get("mapping")
    if isinstance(values, dict):
        return {key: _restore(value, leaves) for key, value in values.items()}
    identity, fields = node.get("dataclass"), node.get("fields")
    if isinstance(identity, list) and len(identity) == 2 and isinstance(fields, dict):
        module, name = identity
        if not isinstance(module, str) or not module.startswith("dew.") or not isinstance(name, str):
            raise ValueError("artifact dataclass is outside the Dew schema")
        cls = getattr(importlib.import_module(module), name)
        if not isinstance(cls, type) or not dataclasses.is_dataclass(cls):
            raise ValueError(f"artifact type {module}.{name} is not a dataclass")
        return cls(**{key: _restore(value, leaves) for key, value in fields.items()})
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
