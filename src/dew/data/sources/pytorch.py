"""PyTorch map-style datasets as a grain source.

This module imports torch, so only `Dataset.from_torch` imports it, never
`import dew.data`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence, Sized

import torch
from torch.utils.data import DataLoader, IterableDataset

from ..dataset import Batch, describe
from ..providers import fields


class TorchRecords:
    """Reads a map-style torch dataset by index, each sample as named fields.

    A dict sample keeps its keys, and `names` names the values of a tuple
    sample, or the one value of any other. Tensors come to the host as
    numpy, and every value is a provider row's field (`fields`). The repr
    names the dataset as a saved position needs (`describe`).
    """

    def __init__(self, dataset: torch.utils.data.Dataset, names: Sequence[str] | None):
        if isinstance(dataset, DataLoader):
            raise TypeError(
                "a DataLoader samples, batches and collates its dataset itself, and the run's "
                "stream orders, shares and batches the records; pass loader.dataset, which is "
                "then read without the loader's sampler or collate_fn")
        if isinstance(dataset, IterableDataset) or not isinstance(dataset, Sized):
            raise TypeError(
                f"{type(dataset).__name__} has no length to read it by index; a run shuffles, "
                f"shares and resumes a map-style dataset, so build a grain IterDataset over "
                f"this one and pass it to Dataset.from_grain")
        self.dataset = dataset
        self.names = None if names is None else tuple(names)

    def __repr__(self) -> str:
        return f"TorchRecords({describe(self.dataset)}, names={self.names})"

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Batch:
        sample = self.dataset[index]
        if isinstance(sample, Mapping) and self.names is None:
            named = dict(sample)
        else:
            values = tuple(sample) if isinstance(sample, (tuple, list)) else (sample,)
            if self.names is None or len(self.names) != len(values):
                raise TypeError(
                    f"sample {index} holds {len(values)} values and fields= names "
                    f"{self.names}; name each value, as fields=('image', 'label') names "
                    f"torchvision's (image, label)")
            named = dict(zip(self.names, values, strict=True))
        return fields({name: value.detach().cpu().numpy() if isinstance(value, torch.Tensor)
                       else value for name, value in named.items()})
