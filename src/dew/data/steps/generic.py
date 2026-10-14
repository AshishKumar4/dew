"""Field selection and record filtering shared by the typed recipes."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping

import grain.python as grain
import jax
import numpy as np

from dew.data.dataset import Batch
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
