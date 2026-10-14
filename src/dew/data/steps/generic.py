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
    names: tuple[str, ...] | None = None
    rename: tuple[tuple[str, str], ...] = ()
    remove: tuple[str, ...] = ()

    def map(self, element: Batch) -> Batch:
        renamed = dict(self.rename)
        names = tuple(element) if self.names is None else self.names
        kept = {name: element[name] for name in names if name not in self.remove}
        return {**{name: value for name, value in kept.items() if name not in renamed},
                **{new: kept[old] for old, new in self.rename}}


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
