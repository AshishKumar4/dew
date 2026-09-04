"""Hugging Face `datasets` as a grain source.

An Arrow-backed `datasets.Dataset` answers `len()` and integer indexing, which
is the whole of grain's random-access protocol, so the wrapper here is thin. It
adds the three things grain needs and a `Dataset` does not do on its own: the
`datasets` import happens on the first record rather than at import time, rows
come back as plain dicts of arrays and scalars, and the Arrow table stays out
of the pickle that reaches the worker processes.
"""

from __future__ import annotations

import tempfile
import threading
from collections.abc import Mapping
from typing import Any, Dict, Optional

import numpy as np

_STREAMING_HINT = (
    "reading Hugging Face datasets needs the streaming extra: "
    "pip install 'dew-ml[streaming]'"
)


def _hf_datasets():
    """The HF `datasets` module, imported on use.

    At module scope it would make importing the data layer require the
    streaming extra, which only reading a dataset actually needs.
    """
    try:
        import datasets
    except ImportError as exc:
        raise ImportError(_STREAMING_HINT) from exc
    return datasets


def _plain_value(value: Any) -> Any:
    """A record value as an array or a Python scalar.

    `datasets` decodes an image column into a PIL image and every transform in
    the data layer is numpy and cv2. PIL images carry the array interface, so
    they convert here; strings, numbers and lists already are what they say.
    """
    return np.asarray(value) if hasattr(value, "__array_interface__") else value


class HFDatasetSource:
    """Random access over a Hugging Face `datasets.Dataset`.

    Either hand over a loaded dataset or name a hub dataset and split, which
    `load_dataset` resolves on the first record. The table never travels in
    the source's pickle: a named dataset reloads by name and split inside the
    worker, and a dataset handed over in memory is written out once and
    reopened from there, the way TokenFileSource reopens its memmap.
    """

    def __init__(self, name: Optional[str] = None, split: str = "train", dataset=None):
        if name is None and dataset is None:
            raise ValueError(
                "HFDatasetSource needs a hub dataset name or a loaded dataset")
        self.name = name
        self.split = split
        self._dataset = dataset
        # Set when a dataset that arrived in memory is written out for the
        # workers; from then on it is what reloads the table.
        self._cache_path: Optional[str] = None
        self._lock = threading.Lock()

    def __repr__(self) -> str:
        # grain writes repr(source) into a DataLoader iterator's checkpoint and
        # refuses a state whose repr no longer matches, so this names the
        # dataset rather than an address, and without touching the table.
        return (f"HFDatasetSource(name={self.name!r}, split={self.split!r}, "
                f"cache={self._cache_path!r})")

    def _table(self):
        """The dataset, loaded once on first access.

        grain reads a source from several threads at a time, and two threads
        that both find no table start two loads of it. The second one raced
        the first inside `datasets`, so the load happens under the lock and
        the fast path only reads the attribute.
        """
        if self._dataset is None:
            with self._lock:
                if self._dataset is None:
                    datasets = _hf_datasets()
                    if self._cache_path is not None:
                        # A directory of splits comes back as a DatasetDict;
                        # a row source is one split's table.
                        held = datasets.load_from_disk(self._cache_path)
                        self._dataset = (held[self.split]
                                         if isinstance(held, datasets.DatasetDict) else held)
                    elif self.name is None:
                        raise ValueError(
                            "an HF source needs a dataset name or a cache path")
                    else:
                        self._dataset = datasets.load_dataset(
                            self.name, split=self.split)
        return self._dataset

    def __len__(self) -> int:
        return len(self._table())

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row: Mapping[str, Any] = self._table()[index]
        return {key: _plain_value(value) for key, value in row.items()}


    def __getstate__(self) -> Dict[str, Any]:
        # grain pickles the source into every worker process, so the table
        # must not be part of it: a copy per worker of a dataset that is
        # already on disk. A named dataset reloads from the hub cache on the
        # other side; a dataset that only exists in memory has nowhere to
        # reload from yet, so it is written out here, once.
        if self._dataset is not None and self.name is None and self._cache_path is None:
            self._cache_path = tempfile.mkdtemp(prefix="dew-hf-dataset-")
            self._dataset.save_to_disk(self._cache_path)
        state = dict(self.__dict__)
        state["_dataset"] = None
        state["_lock"] = None  # a lock does not pickle; the worker gets its own
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._lock = threading.Lock()
