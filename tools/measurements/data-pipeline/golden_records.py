"""The digest and the input writers of the frozen loader capture (`data_oracle.py`, which runs the capture
script at `LEGACY_SCRIPT` inside the reference checkout, with its own copies of them), copied here for
tests/test_pipeline_golden.py: the golden fixture the capture wrote holds them to it, so an edit here that
changes a digest or a store fails the golden test rather than drifting from the capture."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Callable
from pathlib import Path

import numpy as np


def digest(batch):
    """The batch's public fields: each array's shape, dtype and content digest by its path, and each
    record's static facts (`Coordinates`' `max_segments` and `right_padded`) by theirs, not the tree's
    spelling. A field that is None holds nothing and adds no entry, where a renamed field or a changed
    static fact still changes the digest (the maintainer, 10-27: a field added to Coordinates once re-recorded
    142 tree strings whose leaves had not moved). tests/test_pipeline_golden.py reads Dew's batches by
    this digest, so a capture and the test's reading of it are one format."""
    fields = {}

    def walk(value, path):
        if value is None:
            return
        if isinstance(value, dict):
            for key in sorted(value):
                walk(value[key], f"{path}[{key!r}]")
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                walk(item, f"{path}[{index}]")
        elif dataclasses.is_dataclass(value):
            for field in dataclasses.fields(value):
                item = getattr(value, field.name)
                if field.metadata.get("pytree_node", True):
                    walk(item, f"{path}.{field.name}")
                else:
                    fields[f"{path}.{field.name}"] = {"static": item}
        else:
            array = np.asarray(value)
            encoded = (json.dumps(array.tolist(), sort_keys=True).encode()
                       if array.dtype.hasobject else array.tobytes())
            fields[path] = {"shape": list(array.shape), "dtype": array.dtype.str,
                            "sha256": hashlib.sha256(encoded).hexdigest()}

    walk(batch, "")
    return fields


def token_store(root, name, shift):
    directory = root / name
    directory.mkdir()
    train = np.asarray(
        [
            token
            for index in range(100)
            for token in [*((np.arange(5 + index % 15) + index + shift) % 200), 255]
        ],
        np.uint16,
    )
    held = train[:180]
    (directory / "train.bin").write_bytes(train.tobytes())
    (directory / "val.bin").write_bytes(held.tobytes())
    (directory / "meta.json").write_text(
        json.dumps(
            {
                "tokenizer": "byte",
                "vocab_size": 256,
                "dtype": "uint16",
                "train_tokens": len(train),
                "val_tokens": len(held),
                "eos_id": 255,
            }
        )
    )
    return str(directory)


def image_records(root: Path, pixels, pack: Callable[[dict[str, bytes]], bytes]) -> Path:
    """`pixels` written as one ArrayRecord shard of image records under `root`/images, each packed by
    `pack`: the reference checkout's packer here, and the one the checkout under test has where its
    golden test writes the same records (tests/test_pipeline_golden.py)."""
    from array_record.python.array_record_module import ArrayRecordWriter

    image_root = root / "images"
    image_root.mkdir()
    writer = ArrayRecordWriter(str(image_root / "00.array_record"), "group_size:1")
    for row in pixels:
        writer.write(
            pack(
                {
                    "image": row["image"].tobytes(),
                    "shape": np.asarray([16, 16], "<i4").tobytes(),
                    "caption": row["caption"].encode(),
                    "label": np.asarray(row["label"], np.int32).tobytes(),
                }
            )
        )
    writer.close()
    return image_root
