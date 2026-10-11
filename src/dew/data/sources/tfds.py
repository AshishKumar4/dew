"""Prepared TFDS data sources, with explicit preparation and remote mirrors."""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING

from etils import epath
from filelock import FileLock

from dew.cache import dew_cache_dir

from ..dataset import Records
from .base import Read, Shard, ShardManifest

if TYPE_CHECKING:  # tensorflow_datasets is imported on use, not at import
    from tensorflow_datasets import DecoderTree


def _decoder_record(value: object) -> Mapping[str, object] | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        return {name: _decoder_record(decoder) for name, decoder in value.items()}
    from dew.records import json_value

    try:
        settings = {
            name: json_value(option, f"decoder.{name}")
            for name, option in vars(value).items()
            if not name.startswith("_")
        }
    except TypeError as error:
        raise ValueError(f"TFDS decoder {type(value).__name__} has no stable public configuration") from error
    return {"class": f"{type(value).__module__}.{type(value).__qualname__}", "config": settings}


METADATA = ("dataset_info.json", "features.json")
"""What a prepared version directory holds beside its shards."""

PREPARE = (
    "Prepare it in a separate environment with "
    "download_and_prepare(file_format='array_record'), then pass the "
    "builder.data_dir it wrote as path=. Where TensorFlow is installed, "
    "leaving path unset has dew prepare it into its own TFDS directory."
)

# What `prepared` runs in a process of its own. TFDS's GCS copies are
# TFRecords, which dew does not read, so they are not fetched.
_PREPARING = """
import sys, tensorflow_datasets as tfds
name, data_dir, config, version = (argument or None for argument in sys.argv[1:])
tfds.builder(name, data_dir=data_dir, config=config, version=version).download_and_prepare(
    file_format="array_record", download_config=tfds.download.DownloadConfig(try_download_gcs=False))
"""


def prepared(name: str, *, config: str | None = None, version: str | None = None) -> str:
    """Dew's own TFDS data_dir, `dew_cache_dir()/tfds`, holding `name`.

    A missing name is prepared there by `download_and_prepare` in a
    separate process of the same interpreter with no GPU visible, since
    TensorFlow imported into a JAX process stays loaded and claims
    accelerator memory. Processes on one host take turns under a file lock.
    Without TensorFlow, as on Python 3.14, a missing name raises
    `FileNotFoundError`. The directory is not TFDS's `~/tensorflow_datasets`,
    where an earlier `tfds.load` may have prepared the name as TFRecords,
    which dew refuses and TFDS would not prepare again.
    """
    directory = os.path.join(dew_cache_dir(), "tfds")

    def held() -> bool:
        try:
            read_only_builder(directory, name=name, config=config, version=version)
        except FileNotFoundError:
            return False
        return True

    if held():
        return directory
    if importlib.util.find_spec("tensorflow") is None:
        raise FileNotFoundError(
            f"No prepared {name!r} in {directory}, and preparing it needs TensorFlow, "
            f"which this environment does not have. " + PREPARE
        )
    os.makedirs(directory, exist_ok=True)
    with FileLock(os.path.join(directory, ".prepare.lock")):
        if not held():
            done = subprocess.run(
                [sys.executable, "-c", _PREPARING, name, directory, config or "", version or ""],
                env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
                capture_output=True,
                text=True,
                check=False,
            )
            if done.returncode:
                raise RuntimeError(
                    f"preparing {name!r} into {directory} failed:\n"
                    + "\n".join(done.stderr.splitlines()[-20:])
                )
    return directory


def read_only_builder(path: str, *, name: str | None, config: str | None, version: str | None):
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
            "reading prepared TFDS data needs the tfds extra: pip install 'dewml[tfds]'"
        ) from missing
    root = epath.Path(os.path.expanduser(path))
    if not root.is_dir():
        raise FileNotFoundError(f"No prepared TFDS data at {path!r}. " + PREPARE)
    if all((root / name).is_file() for name in METADATA):
        # A resolved directory needs no names to find it. A caller who gives
        # them anyway is constraining what it must hold, which the metadata
        # answers below.
        reader = tfds.builder_from_directory(root)
    elif name is None:
        raise FileNotFoundError(f"No prepared TFDS metadata at {path!r}. " + PREPARE)
    else:
        from tensorflow_datasets.core.read_only_builder import builder_from_files

        asked = "".join(
            f" {what} {value!r}" for what, value in (("config", config), ("version", version)) if value
        )
        try:
            reader = builder_from_files(name, data_dir=os.fspath(root), config=config, version=version)
        except tfds.core.DatasetNotFoundError as missing:
            raise FileNotFoundError(f"{path!r} holds no prepared {name!r}{asked}. " + PREPARE) from missing
    directory = reader.data_path
    dataset_info = reader.info
    if dataset_info.file_format != tfds.core.FileFormat.ARRAY_RECORD:
        raise ValueError(
            f"Prepared data at {directory} uses {dataset_info.file_format}, and dew reads "
            f"ArrayRecords, which is what needs no TensorFlow in the training "
            f"process. Prepare file_format='array_record' in a directory of its "
            f"own."
        )
    for asked, found, what in (
        (name, dataset_info.name, "builder"),
        (config, dataset_info.config_name or None, "config"),
        (version, str(dataset_info.version), "version"),
    ):
        if asked is not None and asked != found:
            raise ValueError(
                f"{directory} holds {what} {found!r}, and load() asked for "
                f"{asked!r}. The prepared metadata is what says which dataset "
                f"this is; point path= at the {asked!r} data or ask for {found!r}."
            )
    return reader


@dataclasses.dataclass(frozen=True)
class TFDS:
    """A prepared TFDS split, read with TFDS's native ArrayRecord data source."""

    name: str
    split: str = "train"
    path: str | None = None
    config: str | None = None
    version: str | None = None
    decoders: DecoderTree | None = None

    @property
    def remote(self) -> bool:
        return self.path is not None and (
            "::" in self.path or "://" in self.path and not self.path.startswith(("file://", "gs://"))
        )

    @property
    def directory(self) -> str:
        if not self.remote:
            return self.path or os.path.join(dew_cache_dir(), "tfds")
        named = (self.path, self.name, self.config, self.version, self.split)
        digest = hashlib.sha256(json.dumps(named).encode()).hexdigest()
        return os.path.join(dew_cache_dir(), "tfds-mirrors", digest)

    @cached_property
    def builder(self):
        if self.remote and not Path(self.directory, "READY").is_file():
            raise FileNotFoundError(
                f"TFDS cannot read {self.path!r} natively; set provision=True to mirror the "
                "prepared split locally once, or use a gs:// prepared directory"
            )
        return read_only_builder(
            self.directory,
            name=self.name,
            config=self.config,
            version=self.version,
        )

    def describe(self) -> str:
        return f"tfds:{self.name}/{self.config}:{self.version}:{self.split}:{self.path}"

    def snapshot(self) -> Mapping[str, object]:
        from dew.data.storage import Store

        directory = str(self.builder.data_path)
        metadata = Store(directory.rstrip("/") + "/dataset_info.json").read()
        splits = self.builder.info.splits
        names = (
            tuple(splits)
            if self.split == "all"
            else tuple(part.split("[", 1)[0] for part in self.split.split("+"))
        )
        files = []
        for name in names:
            if name not in splits:
                raise ValueError(f"TFDS {self.name!r} has no split {name!r}")
            for instruction in splits[name].file_instructions:
                store = Store(instruction.filename)
                fs, path = store.resolve()
                try:
                    header = fs.info(path)
                except FileNotFoundError as error:
                    raise FileNotFoundError(
                        f"Missing prepared ArrayRecord shard {instruction.filename!r}; "
                        "copy the complete split or provision it again"
                    ) from error
                files.append({"uri": store.url, "bytes": header["size"], "version": fs.ukey(path)})
        return {
            "store": "tfds",
            "name": self.name,
            "split": self.split,
            "directory": directory,
            "metadata_sha256": hashlib.sha256(metadata).hexdigest(),
            "files": files,
            "decoders": _decoder_record(self.decoders),
            "original": self.path,
        }

    def supports(self, read: Read) -> bool:
        try:
            _ = self.builder
        except FileNotFoundError:
            return False
        return True

    def provision(self) -> TFDS:
        if self.remote:
            self._mirror()
            return self
        if self.path is not None:
            raise ValueError(
                f"TFDS {self.name!r} path={self.path!r} is read-only; prepare that directory "
                "with TFDS or omit path to use Dew's preparation directory"
            )
        prepared(self.name, config=self.config, version=self.version)
        return self

    def _mirror(self) -> None:
        from dew.data.storage import Store

        destination = Path(self.directory)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with FileLock(str(destination) + ".lock"):
            if (destination / "READY").is_file():
                return
            root = Store(str(self.path))
            fs, path = root.resolve()
            if not fs.exists(path.rstrip("/") + "/dataset_info.json"):
                raise FileNotFoundError(
                    f"remote TFDS path {self.path!r} must name one prepared version directory"
                )
            with tempfile.TemporaryDirectory(dir=destination.parent, prefix="tfds-") as temporary:
                staged = Path(temporary) / "dataset"
                staged.mkdir()
                for key in fs.find(path):
                    if "array_record" in key:
                        continue
                    relative = key[len(path) :].lstrip("/")
                    target = staged / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    fs.get_file(key, str(target))
                builder = read_only_builder(
                    str(staged), name=self.name, config=self.config, version=self.version
                )
                selected = (
                    tuple(builder.info.splits)
                    if self.split == "all"
                    else tuple(part.split("[", 1)[0] for part in self.split.split("+"))
                )
                for name in selected:
                    if name not in builder.info.splits:
                        raise ValueError(f"TFDS {self.name!r} has no split {name!r}")
                    for instruction in builder.info.splits[name].file_instructions:
                        filename = Path(instruction.filename).name
                        fs.get_file(path.rstrip("/") + "/" + filename, str(staged / filename))
                (staged / "READY").write_text(json.dumps({"source": self.path, "split": self.split}))
                shutil.move(staged, destination)

    def records(self, read: Read) -> Records:
        if not self.supports(read):
            if self.remote:
                raise FileNotFoundError(
                    f"TFDS cannot read {self.path!r} natively; set provision=True to mirror "
                    "the prepared split locally once, or use gs://"
                )
            raise FileNotFoundError(
                f"{self.describe()} has no prepared ArrayRecord data; "
                "run dew data provision or set provision=True"
            )
        self.snapshot()
        rows = self.builder.as_data_source(self.split, decoders=self.decoders)
        import grain.python as grain

        from .memory import _record

        indexed = grain.MapDataset.source(rows).map(_record)
        return tuple(indexed[index] for index in range(len(indexed))) if read == "memory" else indexed

    def manifest(self) -> ShardManifest:
        rows = self.records("indexed")
        shards = tuple(
            Shard(
                f"{self.describe()}:{start}",
                str(self.builder.data_path),
                start,
                min(start + 1024, len(rows)),
                min(1024, len(rows) - start),
            )
            for start in range(0, len(rows), 1024)
        )
        return ShardManifest(shards, self._read_shard)

    def _read_shard(self, shard: Shard):
        import grain.python as grain

        return grain.MapDataset.source(self.records("indexed"))[shard.start : shard.stop].to_iter_dataset()

    def metadata(self) -> Mapping[str, object]:
        import tensorflow_datasets as tfds

        features = self.builder.info.features
        class_names = (
            {
                name: tuple(feature.names)
                for name, feature in features.items()
                if isinstance(feature, tfds.features.ClassLabel)
            }
            if isinstance(features, tfds.features.FeaturesDict)
            else {}
        )
        return {
            "name": self.builder.info.name,
            "version": str(self.builder.info.version),
            "rows": len(self.builder.as_data_source(self.split)),
            "class_names": class_names,
        }

    def lengths(self):
        return None
