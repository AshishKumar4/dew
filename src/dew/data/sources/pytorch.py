"""Map-style torch datasets, imported on use and read through the shared pipeline."""

from __future__ import annotations

from collections.abc import Mapping, Sequence, Sized
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from dew.batch import Batch
from dew.data.pipeline import Dataset

from ..dataset import Records, describe
from .base import Read, ShardManifest, _indexed_manifest

if TYPE_CHECKING:
    from torch.utils.data import Dataset as TorchDataset


class TorchRecords:
    """Reads a map-style torch dataset by index, each sample as named fields.

    A dict sample keeps its keys, and `names` names the values of a tuple
    sample, or the one value of any other. Tensors come to the host as
    numpy, and every value is a provider row's field (`fields`). The repr
    names the dataset as a saved position needs (`describe`).
    """

    def __init__(self, dataset: TorchDataset, names: Sequence[str] | None):
        from torch.utils.data import DataLoader, IterableDataset

        if isinstance(dataset, DataLoader):
            raise TypeError(
                "a DataLoader samples, batches and collates its dataset itself, and the run's "
                "stream orders, shares and batches the records; pass loader.dataset, which is "
                "then read without the loader's sampler or collate_fn"
            )
        if isinstance(dataset, IterableDataset) or not isinstance(dataset, Sized):
            raise TypeError(
                f"{type(dataset).__name__} has no length to read it by index; a run shuffles, "
                f"shares and resumes a map-style dataset, so expose unbatched records through "
                f"an explicit Pipeline Source with a checkpointable reader"
            )
        self.dataset = dataset
        self.names = None if names is None else tuple(names)

    def __repr__(self) -> str:
        return f"TorchRecords({describe(self.dataset)}, names={self.names})"

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Batch:
        import torch

        from ..steps.generic import Formatting

        sample = self.dataset[index]
        if isinstance(sample, Mapping) and self.names is None:
            named = dict(sample)
        else:
            values = tuple(sample) if isinstance(sample, (tuple, list)) else (sample,)
            if self.names is None or len(self.names) != len(values):
                raise TypeError(
                    f"sample {index} holds {len(values)} values and fields= names "
                    f"{self.names}; name each value, as fields=('image', 'label') names "
                    f"torchvision's (image, label)"
                )
            named = dict(zip(self.names, values, strict=True))
        return Formatting().map(
            {
                name: value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else value
                for name, value in named.items()
            }
        )


@dataclass(frozen=True)
class Torch(Dataset):
    """A map-style torch dataset, with explicit content identity for plans and artifacts."""

    dataset: TorchDataset
    fields: tuple[str, ...] | None = None
    identity: str | None = None
    known: Mapping[str, object] | None = None
    """Facts of its records the caller states, which its metadata and `Dataset.facts` carry."""

    def stated(self) -> Mapping[str, object]:
        return dict(self.known or {})

    def label(self) -> str:
        return self.identity or "torch"

    def describe(self) -> str:
        return f"torch:{self.identity or repr(TorchRecords(self.dataset, self.fields))}:{self.fields}"

    def fingerprint(self) -> Mapping[str, object]:
        return {
            "store": "torch",
            "identity_required": True,
            "identity": self.identity,
            "dataset": repr(TorchRecords(self.dataset, self.fields)),
            "rows": len(self.records("indexed")),
            "fields": self.fields,
        }

    def supports(self, read: Read) -> bool:
        return True

    def records(self, read: Read) -> Records:
        indexed = TorchRecords(self.dataset, self.fields)
        return tuple(indexed[index] for index in range(len(indexed))) if read == "memory" else indexed

    def manifest(self) -> ShardManifest:
        """The map-style records as a stream of consecutive index ranges (`_indexed_manifest`), as a
        memory source streams."""
        return _indexed_manifest(self, self.describe(), 1024)

    def metadata(self) -> Mapping[str, object]:
        return {**(self.known or {}), "rows": len(self.records("indexed"))}

    def lengths(self) -> np.ndarray | None:
        return None


def from_torch(dataset: TorchDataset, fields: tuple[str, ...] | None = None, *,
               identity: str | None = None) -> Torch:
    """A map-style `torch.utils.data.Dataset` as a Dew dataset, its samples
    named by `fields` where they are tuples. Plans and artifacts require a
    versioned content `identity`, including non-exact resumes (`Torch`)."""
    return Torch(dataset, fields, identity=identity)
