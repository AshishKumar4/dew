"""Hugging Face `datasets` as a grain source.

An Arrow-backed `datasets.Dataset` answers `len()` and integer indexing, which
is the whole of grain's random-access protocol, so the wrapper here is thin. It
adds the three things grain needs: the `datasets` import happens on the first
record, not at import time; rows come back as plain dicts of arrays and
scalars; and the Arrow table stays out of the pickle that reaches the worker
processes.
"""

from __future__ import annotations

import dataclasses
import tempfile
import threading
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Annotated, Any, Protocol, runtime_checkable

if TYPE_CHECKING:  # the imports themselves happen on the first record
    from datasets import (
        Dataset as ArrowDataset,
        DownloadConfig,
        DownloadMode,
        Features,
        VerificationMode,
        Version,
    )

import numpy as np

from ..dataset import Batch, json_argument

_STREAMING_HINT = (
    "reading Hugging Face datasets needs the streaming extra: "
    "pip install 'dewml[streaming]'"
)


def _hf_datasets():
    """The HF `datasets` module, imported on use.

    At module scope it would make importing the data layer require the
    streaming extra, which only reading a dataset needs.
    """
    try:
        import datasets
    except ImportError as exc:
        raise ImportError(_STREAMING_HINT) from exc
    return datasets


@dataclasses.dataclass(frozen=True)
class HFOptions:
    """Holds the arguments of `datasets.load_dataset` as one value, each with
    its own type.

    The Arrow source and the streamed source both call `load`, so they always
    pass the same arguments. The fields have the library's names, except
    `config`, which is `load_dataset`'s `name`, because Dew already uses
    `name` for the dataset's id.
    """

    config: str | None = None
    data_dir: str | None = None
    data_files: str | Sequence[str] | Mapping[str, str | Sequence[str]] | None = None
    cache_dir: str | None = None
    features: Features | None = None
    download_config: DownloadConfig | None = None
    download_mode: DownloadMode | str | None = None
    verification_mode: VerificationMode | str | None = None
    keep_in_memory: bool | None = None
    save_infos: bool = False
    revision: str | Version | None = None
    token: str | bool | None = None
    num_proc: int | None = None
    storage_options: Mapping[str, Any] | None = None
    """Options that `datasets` passes to fsspec. Each fsspec backend defines its own options, so
    Dew passes the mapping on unread."""

    def load(self, path: str, split: str, *, streaming: bool):
        """Load the split at `path` with `datasets.load_dataset`.

        This is where the library downloads a hub dataset and writes its
        Arrow cache, as it normally does. With `streaming`, it reads the
        split as it goes. Dew adds nothing to either.
        """
        datasets = _hf_datasets()
        return datasets.load_dataset(
            path, name=self.config, split=split, streaming=streaming,
            data_dir=self.data_dir, data_files=self.data_files,
            cache_dir=self.cache_dir, features=self.features,
            download_config=self.download_config, download_mode=self.download_mode,
            verification_mode=self.verification_mode,
            keep_in_memory=self.keep_in_memory, save_infos=self.save_infos,
            revision=self.revision, token=self.token, num_proc=self.num_proc,
            storage_options=None if self.storage_options is None
            else dict(self.storage_options))


HubOptions = Annotated[HFOptions, json_argument(HFOptions)]
"""`HFOptions` as a dataset spec declares it. The value is the same; the
annotation says how the command line writes it, which for a value carrying
`Features` and a `DownloadConfig` is one JSON object rather than a flag per
field."""


type Held = Mapping[str, object]
"""What a source carries into a grain worker: its own attributes, with the
table and the lock left behind."""


@runtime_checkable
class ArrayInterface(Protocol):
    """Describes a buffer numpy reads without being told how.

    `datasets` decodes an image column into a PIL image, and a PIL image
    describes its buffer here. Strings, numbers and lists describe none and
    travel as they are.
    """

    @property
    def __array_interface__(self) -> Mapping[str, object]: ...


class HFDatasetSource:
    """Reads a Hugging Face `datasets.Dataset` by index.

    Either hand over a loaded dataset or name a hub dataset and split;
    `load_dataset` resolves the name on the first record. The rows never
    travel in the source's pickle: `datasets` pickles a table read from files
    as their paths, which a worker maps again, and a dataset handed over in
    memory is written out once and mapped from there.
    """

    def __init__(self, name: str | None = None, split: str = "train", dataset=None,
                 options: HFOptions | None = None):
        if name is None and dataset is None:
            raise ValueError(
                "HFDatasetSource needs a hub dataset name or a loaded dataset")
        self.name = name
        self.split = split
        # Whatever `load_dataset` takes beside the name and the split: the
        # config name, data_files, a revision. It travels in the pickle so a
        # worker reloads the same table, and it is in the repr so a resume
        # compares two descriptions of the same rows.
        self.options = options or HFOptions()
        self._dataset = dataset
        self._lock = threading.Lock()

    def __repr__(self) -> str:
        # The description a saved position compares against (`describe`).
        # It names the dataset rather than an address, without touching the
        # table.
        return (f"HFDatasetSource(name={self.name!r}, split={self.split!r}, "
                f"options={self.options!r})")

    def _table(self) -> ArrowDataset:
        """The one Arrow-backed split, loaded once on first access.

        Grain reads a source from several threads at a time, and two threads
        that both find no table would start two loads of it. The load happens
        under the lock and the fast path only reads the attribute.

        Reading by index is the whole promise of this source, so whatever the
        library hands back has to be one table. A directory of splits and a
        streamed split are both refused here rather than indexed into.
        """
        held = self._dataset
        if held is None:
            with self._lock:
                held = self._dataset
                if held is None:
                    held = self._dataset = self._loaded()
        return held

    def _loaded(self) -> ArrowDataset:
        """One table, from the dataset's name."""
        if self.name is None:
            raise ValueError("an HF source needs a dataset name or a loaded dataset")
        datasets = _hf_datasets()
        table = self.options.load(self.name, self.split, streaming=False)
        if not isinstance(table, datasets.Dataset):
            raise TypeError(
                f"{self.name!r} split {self.split!r} loaded as {type(table).__name__}; a "
                f"random-access source is one Arrow-backed split, so name one split, or "
                f"read it with streaming=True")
        return table

    def __len__(self) -> int:
        return len(self._table())

    @property
    def columns(self) -> list[str]:
        """The split's column names, which loads it."""
        return list(self._table().column_names)

    @property
    def features(self) -> Features:
        """The split's column types, a ClassLabel's names among them, which loads it."""
        return self._table().features

    def __getitem__(self, index: int) -> Batch:
        # `datasets` decodes an image column into a PIL image and every
        # transform in the data layer is numpy and cv2. PIL images carry the
        # array interface, so they convert here; strings, numbers and lists
        # travel as they are.
        row: Mapping[str, object] = self._table()[index]
        return {key: self.array(key, value) if isinstance(value, ArrayInterface) else value
                for key, value in row.items()}

    def array(self, name: str, value: ArrayInterface) -> np.ndarray:
        """The array of the value in column `name` that describes its own
        buffer, a decoded image's in its own mode. A reader that wants a
        column in another form overrides this."""
        return np.asarray(value)

    def __getstate__(self) -> Held:
        # grain pickles the source into every worker process. A table read
        # from files pickles as their paths and the worker maps them, with no
        # second trip through load_dataset; a table that exists only in memory
        # would travel whole, so it is written out here, once, and mapped.
        held = self._dataset
        if held is not None and not held.cache_files:
            directory = tempfile.mkdtemp(prefix="dew-hf-dataset-")
            held.save_to_disk(directory)
            self._dataset = _hf_datasets().Dataset.load_from_disk(directory)
        state = dict(self.__dict__)
        state["_lock"] = None  # a lock does not pickle; the worker gets its own
        return state

    def __setstate__(self, state: Held) -> None:
        self.__dict__.update(state)
        self._lock = threading.Lock()
