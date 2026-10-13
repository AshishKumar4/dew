"""Field selection and record filtering shared by the typed recipes."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping

import grain.python as grain
import jax
import numpy as np

from dew.data.dataset import Batch
from dew.data.pipeline import Rejected
from dew.records import record


@dataclasses.dataclass(frozen=True)
class Fields(grain.MapTransform):
    names: tuple[str, ...]
    rename: tuple[tuple[str, str], ...] = ()

    def map(self, element: Batch) -> Batch:
        renamed = dict(self.rename)
        return {renamed.get(name, name): element[name] for name in self.names}


@dataclasses.dataclass(frozen=True)
class Placed(grain.MapTransform):
    """A training or validation batch without the fields an accelerator cannot
    hold: strings, bytes and Python objects, such as an image's catalog URL
    and content hash. The host steps before it read them; what remains is
    what the prefetcher places."""

    def map(self, element: Batch) -> Batch:
        return {name: (self.map(value) if isinstance(value, Mapping) else value)
                for name, value in element.items() if _placeable(value)}


def _placeable[Field](value: np.ndarray | str | bytes | Mapping[str, Field] | Field) -> bool:
    if isinstance(value, np.ndarray):
        return value.dtype.kind not in "OSUV"
    if isinstance(value, (list, tuple)):
        return all(_placeable(entry) for entry in value)
    return not isinstance(value, (str, bytes))


@dataclasses.dataclass(frozen=True)
class Keep(grain.MapTransform):
    predicate: Callable[[Batch], bool]
    reason: str = "filtered"
    rejects = True

    def map(self, element: Batch) -> Batch | Rejected:
        return element if self.predicate(element) else Rejected(self.reason)


@dataclasses.dataclass(frozen=True)
class Formatting(grain.MapTransform):
    def map(self, element: Batch) -> Batch:
        fields = record(element, "dataset row")
        held: Batch = {}
        for name, value in fields.items():
            if isinstance(value, Mapping):
                held[name] = self.map(value)
            elif value is None or isinstance(value, (str, bytes)):
                held[name] = value
            else:
                array = np.asarray(value)
                if array.dtype.hasobject:
                    raise ValueError(f"field {name!r} has variable shapes; add a preprocessing step")
                if array.dtype.kind in "biufc":
                    dtype = np.dtype(jax.dtypes.canonicalize_dtype(array.dtype))
                    if dtype.kind in "iu" and array.size:
                        bounds = np.iinfo(dtype)
                        if np.any((array < bounds.min) | (array > bounds.max)):
                            raise ValueError(f"field {name!r} is outside the {dtype} range")
                    array = array.astype(dtype)
                held[name] = array
        return held


@dataclasses.dataclass(frozen=True)
class Preprocessing(grain.RandomMapTransform):
    step: Callable[[Batch, np.random.Generator], Batch]

    def random_map(self, element: Batch, rng: np.random.Generator) -> Batch:
        return self.step(element, rng)
