"""Prepared TFDS ArrayRecords as a grain source.

Preparation is a separate job: it downloads, generates and writes shards, and
it needs TensorFlow. Reading them does not, so this reads what preparation
left behind through TFDS's read-only builder and never generates anything.
A training run that could prepare its own data would download a corpus from
inside the step loop and write it into whatever directory it happened to
have, so the absence of a preparation path here is the feature.

`prepared` resolves the version directory: the directory itself when it holds
the metadata, otherwise `<data_dir>/<builder>/<config>/<version>` built from
the names a caller gave. `TFDSSource` is then the builder's own data source,
checked for the file format and the shards dew can read.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Optional

from etils import epath

METADATA = ("dataset_info.json", "features.json")
"""What a prepared version directory holds beside its shards."""

PREPARE = ("Prepare it in a separate environment with "
           "download_and_prepare(file_format='array_record'), then pass the "
           "builder.data_dir it wrote. Training never prepares its own data.")


def _version(name: str) -> tuple[int, ...]:
    """`name` as the tuple that orders TFDS versions, or () when it is not one."""
    parts = name.split(".")
    return tuple(int(part) for part in parts) if all(part.isdigit() for part in parts) else ()


def _newest(directory: epath.Path, what: str) -> epath.Path:
    """The highest-numbered version directory under `directory`."""
    versions = sorted((_version(child.name), child) for child in directory.iterdir()
                      if child.is_dir() and _version(child.name))
    if not versions:
        raise FileNotFoundError(
            f"{directory} holds no prepared version of {what}. " + PREPARE)
    return versions[-1][1]


def prepared(path: str, *, builder: Optional[str] = None, config: Optional[str] = None,
             version: Optional[str] = None) -> epath.Path:
    """The version directory holding `builder`'s prepared shards.

    `path` is either that directory, when it holds the metadata, or the
    `data_dir` a preparation run wrote under, in which case the builder, the
    config and the version name the directory inside it the way TFDS lays it
    out. An unset version takes the newest prepared one, so a caller who
    prepared once does not have to repeat its number. Without a builder name
    only the first form resolves: a data_dir holds one directory per builder
    and nothing says which of them was wanted.
    """
    root = epath.Path(os.path.expanduser(path))
    if not root.is_dir():
        raise FileNotFoundError(f"No prepared TFDS data at {path!r}. " + PREPARE)
    if all((root / name).is_file() for name in METADATA):
        if config is not None or version is not None:
            raise ValueError(
                f"{path!r} is a prepared version directory, so it names its own "
                f"config and version; drop config= and version= or pass the "
                f"data_dir above it")
        return root
    if builder is None:
        raise FileNotFoundError(
            f"No prepared TFDS metadata at {path!r}. " + PREPARE)
    directory = root / builder
    if not directory.is_dir():
        raise FileNotFoundError(
            f"{path!r} holds no prepared {builder!r} and is not a prepared "
            f"version directory itself. " + PREPARE)
    if config is not None:
        directory = directory / config
        if not directory.is_dir():
            raise FileNotFoundError(
                f"{path!r} holds no {config!r} config of {builder!r}. " + PREPARE)
    if version is not None:
        directory = directory / version
        if not directory.is_dir():
            raise FileNotFoundError(
                f"{path!r} holds no version {version!r} of {builder!r}. " + PREPARE)
    else:
        directory = _newest(directory, builder)
        if not all((directory / name).is_file() for name in METADATA):
            # A configured builder keeps its versions one level further down.
            directory = _newest(directory, builder)
    for name in METADATA:
        if not (directory / name).is_file():
            raise FileNotFoundError(
                f"{directory} has no {name}, so nothing there is a prepared "
                f"TFDS dataset. " + PREPARE)
    return directory


def read_only_builder(directory: epath.Path, *, builder: Optional[str],
                      config: Optional[str], version: Optional[str]):
    """The read-only builder over `directory`, checked against what was asked for.

    The import is here rather than at module level so `import dew.data` costs
    no TFDS, and so a missing extra names the extra. TFDS reads prepared
    ArrayRecords through this builder without importing TensorFlow, which is
    why the format is refused rather than converted: a TFRecord split would
    pull TensorFlow into the training process.

    The metadata is the authority on which dataset this is. Resolving a
    directory from names and then trusting the names would read whatever
    happens to sit at that path, so the builder, the config and the version
    the caller asked for are compared with the ones the prepared data
    reports.
    """
    try:
        import tensorflow_datasets as tfds
    except ImportError as missing:
        raise ImportError(
            "reading prepared TFDS data needs the tfds extra: "
            "pip install 'dew-ml[tfds]'") from missing
    reader = tfds.builder_from_directory(directory)
    info = reader.info
    if info.file_format != tfds.core.FileFormat.ARRAY_RECORD:
        raise ValueError(
            f"Prepared data at {directory} uses {info.file_format}, and dew reads "
            f"ArrayRecords, which is what needs no TensorFlow in the training "
            f"process. Prepare file_format='array_record' in a directory of its "
            f"own.")
    for asked, found, what in ((builder, info.name, "builder"),
                               (config, info.config_name or None, "config"),
                               (version, str(info.version), "version")):
        if asked is not None and asked != found:
            raise ValueError(
                f"{directory} holds {what} {found!r}, and load() asked for "
                f"{asked!r}. The prepared metadata is what says which dataset "
                f"this is; point path= at the {asked!r} data or ask for {found!r}.")
    return reader


def shards(builder, split: str, directory: epath.Path) -> None:
    """Refuse a split expression this directory cannot answer.

    A plain name that is not one of the prepared splits is named here, where
    the alternatives can be listed. An expression -- a union, a slice, or
    TFDS's own "all" -- is left to TFDS, and every prepared split's shards
    are checked instead, since any of them may be read.

    The shards are checked before the run rather than in a grain worker on
    the first record that needed a missing one, steps into training.
    """
    splits = builder.info.splits
    plain = not any(character in split for character in "+[")
    if plain and split != "all" and split not in splits:
        raise ValueError(
            f"{directory} holds no split {split!r}; it holds {sorted(splits)}")
    wanted = [split] if plain and split in splits else sorted(splits)
    for name in wanted:
        for instruction in splits[name].file_instructions:
            if not epath.Path(instruction.filename).is_file():
                raise FileNotFoundError(
                    f"Missing prepared ArrayRecord shard {instruction.filename!r}. "
                    f"Copy the complete prepared dataset or prepare it again.")


def prepared_source(path: str, split: str, *, builder: Optional[str] = None,
                    config: Optional[str] = None,
                    version: Optional[str] = None) -> Sequence[object]:
    """Random access over one split of a prepared TFDS dataset.

    The builder's own `as_data_source` is already grain's protocol, so what
    this adds is the resolution of the directory, the checks against its
    metadata and the two refusals a half-prepared or wrongly formatted
    directory earns. There is no wrapper object, because there would be
    nothing for one to do.

    `split` takes TFDS's own syntax, slicing included. A record is whatever
    the prepared features make it, which for a features dict is a mapping and
    for a single feature is a bare array, so the type here says `object` and
    the run's own `preprocess` is where it becomes batch fields.
    """
    directory = prepared(path, builder=builder, config=config, version=version)
    reader = read_only_builder(directory, builder=builder, config=config, version=version)
    shards(reader, split, directory)
    return reader.as_data_source(split)
