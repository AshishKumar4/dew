"""Prepared TFDS data sources, with explicit preparation and remote mirrors."""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import json
import os
import posixpath
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING

from etils import epath
from filelock import FileLock

from dew.batch import Batch
from dew.cache import dew_cache_dir
from dew.data.pipeline import Dataset
from dew.files import published

from ..dataset import Records
from .base import Read, ShardManifest, _indexed_manifest, preparing

if TYPE_CHECKING:  # tensorflow_datasets is imported on use, not at import
    from tensorflow_datasets import DecoderTree


@dataclasses.dataclass(frozen=True)
class _Texts:
    """A TFDS record with its text features decoded from the utf-8 bytes TFDS hands over."""

    fields: tuple[str, ...]

    def __call__(self, row: Batch | bytes | None) -> Batch | bytes | None:
        if not isinstance(row, Mapping):
            return row
        return {**row, **{name: row[name].decode() for name in self.fields if isinstance(row[name], bytes)}}


def _decoder_record(value: object) -> Mapping[str, object] | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        return {name: _decoder_record(decoder) for name, decoder in value.items()}
    from dew._records import json_value

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


def prepared(name: str, *, config: str | None = None, version: str | None = None) -> bool:
    """Whether `name` was prepared into Dew's own data_dir, `dew_cache_dir()/tfds`.

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
        return False
    if importlib.util.find_spec("tensorflow") is None:
        raise FileNotFoundError(
            f"No prepared {name!r} in {directory}, and preparing it needs TensorFlow, "
            f"which this environment does not have. " + PREPARE
        )
    os.makedirs(directory, exist_ok=True)
    with FileLock(os.path.join(directory, ".prepare.lock")):
        if held():
            return False
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
    return True


def _selected(builder, split: str) -> tuple[str, ...]:
    """The shard files a split expression reads, each once, as TFDS resolves the expression (a
    split, a slice, a sum of them or `all`), which raises ValueError for a split the dataset has
    not."""
    instructions = builder.info.splits[split].file_instructions
    return tuple(dict.fromkeys(instruction.filename for instruction in instructions))


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
class TFDS(Dataset):
    """A prepared TFDS split, read with TFDS's native ArrayRecord data source."""

    name: str
    split: str = "train"
    path: str | None = None
    config: str | None = None
    version: str | None = None
    decoders: DecoderTree | None = None
    shard_rows: int = 1024

    def __post_init__(self) -> None:
        if self.shard_rows < 1:
            raise ValueError("TFDS shard_rows must be positive")

    def with_split(self, name: str) -> TFDS:
        return dataclasses.replace(self, split=name)

    @property
    def remote(self) -> bool:
        return self.path is not None and (
            "::" in self.path or ("://" in self.path and not self.path.startswith(("file://", "gs://")))
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

    def label(self) -> str:
        return self.name

    def describe(self) -> str:
        return f"tfds:{self.name}/{self.config}:{self.version}:{self.split}:{self.path}"

    def fingerprint(self) -> Mapping[str, object]:
        return self._snapshot

    @cached_property
    def _snapshot(self) -> Mapping[str, object]:
        from dew.data.storage import Store, content_version, local

        directory = str(self.builder.data_path)
        metadata = Store(directory.rstrip("/") + "/dataset_info.json").read()
        fs, prefix = Store(directory).resolve()
        # A local copy is named by its content, so hosts that mounted it at different paths agree.
        mounted = local(fs)
        definitions = {}
        for path in fs.find(prefix):
            if path.endswith((".json", ".labels.txt")):
                definitions[path[len(prefix) :].lstrip("/")] = hashlib.sha256(fs.cat_file(path)).hexdigest()
        files = []
        for filename in _selected(self.builder, self.split):
            store = Store(filename)
            fs, path = store.resolve()
            try:
                header = fs.info(path)
            except FileNotFoundError as error:
                raise FileNotFoundError(
                    f"Missing prepared ArrayRecord shard {filename!r}; "
                    "copy the complete split or provision it again"
                ) from error
            files.append(
                {"uri": posixpath.relpath(path, prefix), "bytes": header["size"],
                 "version": content_version(fs, path, header["size"])}
                if mounted else {"uri": store.url, "bytes": header["size"], "version": fs.ukey(path)})
        if self.remote:
            mirrored = json.loads(Path(self.directory, "READY").read_text())
            files = mirrored["files"]
        return {
            "store": "tfds",
            "name": self.name,
            "split": self.split,
            "shard_rows": self.shard_rows,
            **({} if mounted else {"directory": self.path if self.remote else directory}),
            "metadata_sha256": hashlib.sha256(metadata).hexdigest(),
            "definitions": definitions,
            "files": files,
            "decoders": _decoder_record(self.decoders),
            **({} if mounted else {"original": self.path}),
        }

    def supports(self, read: Read) -> bool:
        try:
            _ = self.builder
        except FileNotFoundError:
            return False
        return True

    def can_provision(self, read: Read) -> bool:
        return self.path is None or self.remote

    def provision(self) -> TFDS:
        if self.remote:
            preparing(self, None, self.directory)
            changed = self._mirror()
        else:
            if self.path is not None:
                raise ValueError(
                    f"TFDS {self.name!r} path={self.path!r} is read-only; prepare that directory "
                    "with TFDS or omit path to use Dew's preparation directory"
                )
            preparing(self, None, os.path.join(dew_cache_dir(), "tfds"))
            changed = prepared(self.name, config=self.config, version=self.version)
        if changed:
            self.__dict__.pop("builder", None)
            self.__dict__.pop("_snapshot", None)
        return self

    def _mirror(self) -> bool:
        from dew.data.storage import Store

        destination = Path(self.directory)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with FileLock(str(destination) + ".lock"):
            if (destination / "READY").is_file():
                return False
            root = Store(str(self.path))
            fs, path = root.resolve()
            if not fs.exists(path.rstrip("/") + "/dataset_info.json"):
                raise FileNotFoundError(
                    f"remote TFDS path {self.path!r} must name one prepared version directory"
                )
            with tempfile.TemporaryDirectory(dir=destination.parent, prefix="tfds-") as temporary:
                staged = Path(temporary) / "dataset"
                staged.mkdir()
                copied = []

                def download(key: str, target: Path) -> None:
                    version = fs.ukey(key)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    fs.get_file(key, str(target))
                    if fs.ukey(key) != version or fs.size(key) != target.stat().st_size:
                        raise OSError(f"TFDS shard changed while provisioning {key}")
                    copied.append(
                        {
                            "uri": root.child(key[len(path) :].lstrip("/")).url,
                            "bytes": target.stat().st_size,
                            "version": version,
                        }
                    )

                for key in fs.find(path):
                    if "array_record" in key:
                        continue
                    relative = key[len(path) :].lstrip("/")
                    target = staged / relative
                    download(key, target)
                builder = read_only_builder(
                    str(staged), name=self.name, config=self.config, version=self.version
                )
                for filename in _selected(builder, self.split):
                    name = Path(filename).name
                    download(path.rstrip("/") + "/" + name, staged / name)
                (staged / "READY").write_text(
                    json.dumps({"source": self.path, "split": self.split, "files": copied})
                )
                published(staged, destination)
        return True

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
        self.fingerprint()
        rows = self.builder.as_data_source(self.split, decoders=self.decoders)
        import grain.python as grain
        import tensorflow_datasets as tfds

        from .memory import _record

        indexed = grain.MapDataset.source(rows).map(_record)
        features = self.builder.info.features
        texts = (tuple(name for name, feature in features.items() if isinstance(feature, tfds.features.Text))
                 if isinstance(features, tfds.features.FeaturesDict) else ())
        if texts:
            # TFDS hands its text features over as utf-8 bytes; a record holds them as text.
            indexed = indexed.map(_Texts(texts))
        return tuple(indexed[index] for index in range(len(indexed))) if read == "memory" else indexed

    def manifest(self) -> ShardManifest:
        return _indexed_manifest(self, str(self.builder.data_path), self.shard_rows)

    def stated(self) -> Mapping[str, object]:
        """Its builder's info, no data source opened: the dataset's name and version, its class
        names, and the split's rows the info counts. A split not prepared here, or a remote one not
        mirrored yet, states nothing: a run's record and model are built where its data may not be."""
        root = epath.Path(os.path.expanduser(self.directory))
        # A prepared directory holds the split's metadata, or a data directory the dataset's own.
        prepared = all((root / name).is_file() for name in METADATA)
        here = root.is_dir() and (prepared or (root / self.name).is_dir())
        if not here or (self.remote and not Path(self.directory, "READY").is_file()):
            return {}
        described = self.builder.info
        splits = described.splits
        counted = splits is not None and self.split in splits
        rows = {"rows": splits[self.split].num_examples} if counted else {}
        return {"name": described.name, "version": str(described.version), "class_names": self._class_names,
                **rows}

    @property
    def _class_names(self) -> Mapping[str, tuple[str, ...]]:
        import tensorflow_datasets as tfds

        features = self.builder.info.features
        return (
            {
                name: tuple(feature.names)
                for name, feature in features.items()
                if isinstance(feature, tfds.features.ClassLabel)
            }
            if isinstance(features, tfds.features.FeaturesDict)
            else {}
        )

    def metadata(self) -> Mapping[str, object]:
        class_names = self._class_names
        return {
            "name": self.builder.info.name,
            "version": str(self.builder.info.version),
            "rows": len(self.builder.as_data_source(self.split)),
            "class_names": class_names,
        }

    def lengths(self):
        return None
