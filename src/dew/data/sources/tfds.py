"""Prepared TFDS ArrayRecords as a grain source.

Preparation downloads, generates and writes the shards, and needs
TensorFlow. Reading them needs neither: TFDS's read-only builder opens what
preparation left behind. `prepared` alone prepares, into dew's own TFDS
directory, in a separate process.

`read_only_builder` resolves the version directory, the directory itself
when it holds the metadata and otherwise through TFDS's own
`builder_from_files` from the names a caller gave, then compares the builder,
config and version asked for with what the metadata reports and refuses a
file format other than ArrayRecord.
`prepared_source` is the builder's own data source behind those checks and a
check that every shard of the split is present.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Annotated

from etils import epath
from filelock import FileLock

from dew.cache import dew_cache_dir

from ..dataset import Batch, Records, json_argument

if TYPE_CHECKING:  # tensorflow_datasets is imported on use, not at import
    from tensorflow_datasets import DecoderTree

METADATA = ("dataset_info.json", "features.json")
"""What a prepared version directory holds beside its shards."""

PREPARE = ("Prepare it in a separate environment with "
           "download_and_prepare(file_format='array_record'), then pass the "
           "builder.data_dir it wrote as path=. Where TensorFlow is installed, "
           "leaving path unset has dew prepare it into its own TFDS directory.")

# What `prepared` runs in a process of its own. TFDS's GCS copies are
# TFRecords, which dew does not read, so they are not fetched.
_PREPARING = """
import sys, tensorflow_datasets as tfds
name, data_dir, config, version = (argument or None for argument in sys.argv[1:])
tfds.builder(name, data_dir=data_dir, config=config, version=version).download_and_prepare(
    file_format="array_record", download_config=tfds.download.DownloadConfig(try_download_gcs=False))
"""


def prepared(builder: str, *, config: str | None = None, version: str | None = None) -> str:
    """Dew's own TFDS data_dir, `dew_cache_dir()/tfds`, holding `builder`.

    A missing builder is prepared there by `download_and_prepare` in a
    separate process of the same interpreter with no GPU visible, since
    TensorFlow imported into a JAX process stays loaded and claims
    accelerator memory. Processes on one host take turns under a file lock.
    Without TensorFlow, as on Python 3.14, a missing builder raises
    `FileNotFoundError`. The directory is not TFDS's `~/tensorflow_datasets`,
    where an earlier `tfds.load` may have prepared the builder as TFRecords,
    which dew refuses and TFDS would not prepare again.
    """
    directory = os.path.join(dew_cache_dir(), "tfds")

    def held() -> bool:
        try:
            read_only_builder(directory, builder=builder, config=config, version=version)
        except FileNotFoundError:
            return False
        return True

    if held():
        return directory
    if importlib.util.find_spec("tensorflow") is None:
        raise FileNotFoundError(
            f"No prepared {builder!r} in {directory}, and preparing it needs TensorFlow, "
            f"which this environment does not have. " + PREPARE)
    os.makedirs(directory, exist_ok=True)
    with FileLock(os.path.join(directory, ".prepare.lock")):
        if not held():
            done = subprocess.run(
                [sys.executable, "-c", _PREPARING, builder, directory, config or "", version or ""],
                env={**os.environ, "CUDA_VISIBLE_DEVICES": ""}, capture_output=True, text=True,
                check=False)
            if done.returncode:
                raise RuntimeError(f"preparing {builder!r} into {directory} failed:\n"
                                   + "\n".join(done.stderr.splitlines()[-20:]))
    return directory


def read_only_builder(path: str, *, builder: str | None,
                      config: str | None, version: str | None):
    """The read-only builder over the prepared data at `path`, checked against
    what was asked for.

    `path` is either a version directory, when it holds the metadata, or the
    `data_dir` a preparation run wrote under, where TFDS's own
    `builder_from_files` finds the builder's directory from the names given
    (the newest prepared version when none is named, the default config when
    none is). Without a builder name only the first form resolves, since a
    data_dir holds one directory per builder and nothing says which of them
    was wanted.

    The import is here rather than at module level so `import dew.data` costs
    no TFDS, and so a missing extra names the extra. TFDS reads prepared
    ArrayRecords through this builder without importing TensorFlow, which is
    why another format is refused rather than converted: a TFRecord split
    would pull TensorFlow into the training process.

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
            "pip install 'dewml[tfds]'") from missing
    root = epath.Path(os.path.expanduser(path))
    if not root.is_dir():
        raise FileNotFoundError(f"No prepared TFDS data at {path!r}. " + PREPARE)
    if all((root / name).is_file() for name in METADATA):
        # A resolved directory needs no names to find it. A caller who gives
        # them anyway is constraining what it must hold, which the metadata
        # answers below.
        reader = tfds.builder_from_directory(root)
    elif builder is None:
        raise FileNotFoundError(f"No prepared TFDS metadata at {path!r}. " + PREPARE)
    else:
        from tensorflow_datasets.core.read_only_builder import builder_from_files

        asked = "".join(f" {what} {value!r}" for what, value in (("config", config),
                                                                 ("version", version)) if value)
        try:
            reader = builder_from_files(builder, data_dir=os.fspath(root), config=config,
                                        version=version)
        except tfds.core.DatasetNotFoundError as missing:
            raise FileNotFoundError(
                f"{path!r} holds no prepared {builder!r}{asked}. " + PREPARE) from missing
    directory = reader.data_path
    dataset_info = reader.info
    if dataset_info.file_format != tfds.core.FileFormat.ARRAY_RECORD:
        raise ValueError(
            f"Prepared data at {directory} uses {dataset_info.file_format}, and dew reads "
            f"ArrayRecords, which is what needs no TensorFlow in the training "
            f"process. Prepare file_format='array_record' in a directory of its "
            f"own.")
    for asked, found, what in ((builder, dataset_info.name, "builder"),
                               (config, dataset_info.config_name or None, "config"),
                               (version, str(dataset_info.version), "version")):
        if asked is not None and asked != found:
            raise ValueError(
                f"{directory} holds {what} {found!r}, and load() asked for "
                f"{asked!r}. The prepared metadata is what says which dataset "
                f"this is; point path= at the {asked!r} data or ask for {found!r}.")
    return reader


def check_shards(builder, split: str, directory: epath.Path) -> None:
    """Raise unless every prepared shard the split reads is present.

    A plain name that is not a prepared split is named here, where the
    alternatives can be listed. An expression -- a union, a slice, or TFDS's
    own "all" -- is left to TFDS, and every prepared split's shards are
    checked instead, since any of them may be read.

    The check runs before the run rather than in a grain worker on the first
    record that needed a missing shard, steps into training.
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


def prepared_source(path: str, split: str, *, builder: str | None = None,
                    config: str | None = None, version: str | None = None,
                    decoders: DecoderTree | None = None) -> Records:
    """Reads one split of a prepared TFDS dataset by index.

    The builder's own `as_data_source` is already grain's protocol. What this
    adds is the resolution of the directory, the checks against its metadata,
    and the refusals a half-prepared or wrongly formatted directory earns.

    `split` takes TFDS's own syntax, slicing included, and `decoders` reaches
    the builder unchanged, so a caller can hand it `SkipDecoding()` for a
    feature it wants as the bytes on disk. A record is whatever the prepared
    features and those decoders make it: a mapping for a features dict, a
    bare array for a single feature. The run's own `preprocess` is where it
    becomes batch fields.
    """
    reader = read_only_builder(path, builder=builder, config=config, version=version)
    check_shards(reader, split, epath.Path(reader.data_path))
    return Prepared(reader.as_data_source(split, decoders=decoders),
                    str(reader.data_path), split)


class Prepared:
    """Reads one prepared split by index.

    TFDS's features and the caller's decoders decide what a record is: a
    mapping of features, or the bytes of a single feature a decoder skipped.
    A record is narrowed here, on the way out, so one that is neither is
    named where the split is known rather than inside a grain worker.

    The repr is the directory and the split, which is what a saved position
    compares against (`describe`). The builder's own source describes itself
    by its address in this process, and two addresses refuse every resume.
    """

    def __init__(self, records: Sequence[object], directory: str, split: str):
        self.directory = directory
        self.split = split
        self._records = records

    def __repr__(self) -> str:
        return f"Prepared(directory={self.directory!r}, split={self.split!r})"

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> Batch | bytes | None:
        held = self._records[index]
        if isinstance(held, bytes):
            return held
        if isinstance(held, Mapping):
            return dict(held)
        raise TypeError(
            f"record {index} of {self.directory} split {self.split!r} is "
            f"{type(held).__name__}; a prepared record is its features or the "
            f"bytes of the one feature a decoder skipped")


@dataclasses.dataclass(frozen=True)
class TFDSOptions:
    """Where a prepared TFDS dataset is, and which part of it to read.

    `path` is what a preparation run wrote, either the version directory or
    the `data_dir` above it, and is only read. None reads dew's own TFDS
    directory instead, where a missing builder is prepared first
    (`prepared`). `config` and `version` pick the directory inside a
    data_dir, and both are checked against the prepared metadata.
    `decoders` is TFDS's own decoder tree, passed to the builder unchanged,
    so a caller can ask for the bytes on disk with `SkipDecoding()`.

    The builder's name and the split expression are not options here,
    because a mixture reads several builders through one set of these
    options.
    """

    path: str | None = None
    config: str | None = None
    version: str | None = None
    decoders: DecoderTree | None = None

    def source(self, name: str, split: str) -> Records:
        """Open `split` of the prepared builder `name` for reading by index."""
        path = self.path or prepared(name, config=self.config, version=self.version)
        return prepared_source(path, split, builder=name, config=self.config,
                               version=self.version, decoders=self.decoders)


PreparedOptions = Annotated[TFDSOptions, json_argument(TFDSOptions)]
"""`TFDSOptions` as a dataset spec declares it: one JSON object on the
command line, because a decoder tree is a tree of TFDS objects with no
command-line spelling."""
