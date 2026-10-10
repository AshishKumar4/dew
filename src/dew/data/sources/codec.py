"""The ArrayRecord byte-map layout and the artifact record codec."""

from __future__ import annotations

import dataclasses
import hashlib
import importlib
import io
import json
import struct
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING

import numpy as np

from dew.records import JSON

if TYPE_CHECKING:
    from _typeshed import DataclassInstance

type Value = JSON | bytes | np.ndarray | np.generic | DataclassInstance | tuple[Value, ...] | Mapping[str, Value]


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
        key = payload[offset:offset + length].decode()
        offset += length
        if offset + 4 > len(payload):
            raise ValueError(f"truncated ArrayRecord value length for {key!r}")
        length = struct.unpack_from("<I", payload, offset)[0]
        offset += 4
        if offset + length > len(payload) or key in values:
            raise ValueError(f"invalid ArrayRecord value for {key!r}")
        values[key] = payload[offset:offset + length]
        offset += length
    return values


def _tree(value: Value, leaves: dict[str, bytes]) -> JSON:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, np.generic):
        return _tree(value.item(), leaves)
    if isinstance(value, (np.ndarray, bytes)):
        key = str(len(leaves))
        if isinstance(value, bytes):
            leaves[key] = value
            return {"bytes": key}
        file = io.BytesIO()
        np.save(file, value, allow_pickle=False)
        leaves[key] = file.getvalue()
        return {"array": key}
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("artifact mappings need string keys")
        return {"mapping": {key: _tree(entry, leaves) for key, entry in value.items()}}
    if isinstance(value, (list, tuple)):
        return {"tuple" if isinstance(value, tuple) else "list": [_tree(entry, leaves) for entry in value]}
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        cls = type(value)
        if not cls.__module__.startswith("dew."):
            raise TypeError(f"artifact codec has no schema for {cls.__module__}.{cls.__qualname__}")
        return {"dataclass": [cls.__module__, cls.__qualname__],
                "fields": {field.name: _tree(getattr(value, field.name), leaves)
                           for field in dataclasses.fields(value) if field.init}}
    raise TypeError(f"artifact codec cannot encode {type(value).__name__}")


def _restore(node: object, leaves: Mapping[str, bytes]) -> Value:
    if node is None or isinstance(node, (bool, int, float, str)):
        return node
    if not isinstance(node, dict):
        raise ValueError("artifact node is not a scalar or tagged object")
    if "bytes" in node:
        return leaves[str(node["bytes"])]
    if "array" in node:
        return np.load(io.BytesIO(leaves[str(node["array"])]), allow_pickle=False)
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


def encode(value: Value) -> bytes:
    leaves: dict[str, bytes] = {}
    tree = _tree(value, leaves)
    leaves["schema"] = json.dumps(tree, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()
    return pack_dict_of_byte_arrays(leaves)


def decode(value: bytes) -> Value:
    leaves = unpack_dict_of_byte_arrays(value)
    return _restore(json.loads(leaves.pop("schema")), leaves)


def fingerprint(values: Iterable[Value]) -> str:
    digest = hashlib.sha256()
    for value in values:
        encoded = encode(value)
        digest.update(struct.pack("<Q", len(encoded)))
        digest.update(encoded)
    return digest.hexdigest()
