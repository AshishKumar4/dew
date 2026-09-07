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
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:  # the imports themselves happen on the first record
    from datasets import (Dataset as ArrowDataset, DownloadConfig, DownloadMode, Features,
                          VerificationMode, Version)

import numpy as np

_STREAMING_HINT = (
    "reading Hugging Face datasets needs the streaming extra: "
    "pip install 'dew-ml[streaming]'"
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
    """`datasets.load_dataset`'s arguments, as one value with their own types.

    Both hf routes call `load`, so the Arrow source and the streamed source
    cannot drift apart in what they forward. The fields are the library's,
    named as the library names them apart from `config`, which is
    `load_dataset`'s `name`: dew already uses `name` for which dataset this
    is.
    """

    config: Optional[str] = None
    data_dir: Optional[str] = None
    data_files: Optional[str | Sequence[str] | Mapping[str, str | Sequence[str]]] = None
    cache_dir: Optional[str] = None
    features: Optional["Features"] = None
    download_config: Optional["DownloadConfig"] = None
    download_mode: Optional["DownloadMode | str"] = None
    verification_mode: Optional["VerificationMode | str"] = None
    keep_in_memory: Optional[bool] = None
    save_infos: bool = False
    revision: Optional["str | Version"] = None
    token: Optional[str | bool] = None
    num_proc: Optional[int] = None
    storage_options: Optional[Mapping[str, Any]] = None
    """`datasets` passes this to fsspec, whose backends declare their own
    options; the mapping is theirs to read."""

    def load(self, path: str, split: str, *, streaming: bool):
        """The split at `path`, through `datasets.load_dataset`.

        This is where a hub dataset is downloaded and an Arrow cache written,
        by the library, on its own terms; a streamed split reads as it goes
        instead. Dew adds nothing to either.
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


def _plain_value(value: Any) -> Any:
    """A record value as an array or a Python scalar.

    `datasets` decodes an image column into a PIL image and every transform in
    the data layer is numpy and cv2. PIL images carry the array interface, so
    they convert here; strings, numbers and lists pass through.
    """
    return np.asarray(value) if hasattr(value, "__array_interface__") else value


class HFDatasetSource:
    """Random access over a Hugging Face `datasets.Dataset`.

    Either hand over a loaded dataset or name a hub dataset and split, which
    `load_dataset` resolves on the first record. The table never travels in
    the source's pickle. A named dataset reloads from its name and split
    inside the worker, and a dataset handed over in memory is written out
    once and reopened from there, the way TokenFileSource reopens its memmap.
    """

    def __init__(self, name: Optional[str] = None, split: str = "train", dataset=None,
                 options: Optional[HFOptions] = None):
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
        # Set when a dataset that arrived in memory is written out for the
        # workers; from then on it is what reloads the table.
        self._cache_path: Optional[str] = None
        self._lock = threading.Lock()

    def __repr__(self) -> str:
        # grain writes repr(source) into a DataLoader iterator's checkpoint and
        # refuses a state whose repr differs, so this names the dataset, not
        # an address, and without touching the table.
        return (f"HFDatasetSource(name={self.name!r}, split={self.split!r}, "
                f"options={self.options!r}, cache={self._cache_path!r})")

    def _table(self) -> "ArrowDataset":
        """The one Arrow-backed split, loaded once on first access.

        Grain reads a source from several threads at a time, and two threads
        that both find no table would start two loads of it. The load happens
        under the lock and the fast path only reads the attribute.

        Random access is the whole promise of this source, so whatever the
        library hands back has to be one table: a directory of splits and a
        streamed split are both refused here rather than indexed into.
        """
        held = self._dataset
        if held is None:
            with self._lock:
                held = self._dataset
                if held is None:
                    held = self._dataset = self._loaded()
        return held

    def _loaded(self) -> "ArrowDataset":
        """One table, from the cache path or from the dataset's name."""
        datasets = _hf_datasets()
        if self._cache_path is not None:
            # A directory of splits comes back as a DatasetDict; a row source
            # is one split's table.
            held = datasets.load_from_disk(self._cache_path)
            table = held[self.split] if isinstance(held, datasets.DatasetDict) else held
            what = f"the saved dataset at {self._cache_path!r}"
        elif self.name is None:
            raise ValueError("an HF source needs a dataset name or a cache path")
        else:
            table = self.options.load(self.name, self.split, streaming=False)
            what = f"{self.name!r} split {self.split!r}"
        if not isinstance(table, datasets.Dataset):
            raise TypeError(
                f"{what} loaded as {type(table).__name__}; a random-access source "
                f"is one Arrow-backed split, so name one split, or read it with "
                f"streaming=True")
        return table

    def __len__(self) -> int:
        return len(self._table())

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row: Mapping[str, Any] = self._table()[index]
        return {key: _plain_value(value) for key, value in row.items()}


    def __getstate__(self) -> Dict[str, Any]:
        # grain pickles the source into every worker process, so the table
        # must not be part of it. That would be a copy per worker of a dataset
        # that is already on disk. A named dataset reloads from the hub cache
        # on the other side; a dataset that only exists in memory has nowhere
        # to reload from yet, so it is written out here, once.
        held = self._dataset
        if held is not None and self.name is None and self._cache_path is None:
            self._cache_path = tempfile.mkdtemp(prefix="dew-hf-dataset-")
            held.save_to_disk(self._cache_path)
        state = dict(self.__dict__)
        state["_dataset"] = None
        state["_lock"] = None  # a lock does not pickle; the worker gets its own
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._lock = threading.Lock()
