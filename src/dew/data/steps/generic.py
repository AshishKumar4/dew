"""Field selection, formatting and record filtering, steps any chain takes."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping

import grain.python as grain
import jax
import numpy as np

from dew._records import record
from dew.batch import Batch


@dataclasses.dataclass(frozen=True)
class Fields(grain.MapTransform):
    """A record's fields `names` (every field where None), renamed by `rename`, without `remove`.
    `bookkeeping` keeps Dew's own record of the record (`_dew_*`) whatever the projection keeps,
    its counted targets among it (`dew.data.steps.text.rebound`), as a pipeline's column steps do;
    a projection Dew reads for a packer keeps exactly the fields it names."""

    names: tuple[str, ...] | None = None
    rename: tuple[tuple[str, str], ...] = ()
    remove: tuple[str, ...] = ()
    bookkeeping: bool = False

    def declared(self, facts: Mapping[str, object]) -> Mapping[str, object]:
        """The shapes of the fields kept, under their new names."""
        shapes = record(facts.get("fields", {}), "fields")
        names = tuple(shapes) if self.names is None else self.names
        kept = {name: shapes[name] for name in names if name in shapes and name not in self.remove}
        renamed = dict(self.rename)
        return {**facts, "fields": {**{name: shape for name, shape in kept.items() if name not in renamed},
                                    **{new: kept[old] for old, new in self.rename if old in kept}}}

    def map(self, element: Batch) -> Batch:
        renamed = dict(self.rename)
        names = tuple(element) if self.names is None else self.names
        kept = {name: element[name] for name in names if name not in self.remove}
        projected = {**{name: value for name, value in kept.items() if name not in renamed},
                     **{new: kept[old] for old, new in self.rename}}
        if not self.bookkeeping:
            return projected
        from dew.data.steps.text import rebound

        held = {name: value for name, value in element.items()
                if name.startswith("_dew_") and name not in projected}
        return {**projected, **held, **rebound(element, kept, renamed)}


@dataclasses.dataclass(frozen=True)
class Coordinated(grain.MapTransform):
    """A packed training or validation batch with each window's documents as
    the model reads them: the `<ids>_segment_ids` and `<ids>_positions` a
    packer writes for its `fields` become `<ids>_coordinates`, the model's
    `dew.batch.Coordinates` built on the host, holding at most
    `max_segments` documents a row, or as many as a window holds tokens where
    None (static, so a compiled step is specialized to it). A window holding
    more than its read declared is refused, which a sequence kernel would run
    into its neighbours' sequences. `ids` is the packed ids field, the first
    of `fields`."""

    fields: tuple[str, ...]
    max_segments: int | None

    def map(self, element: Batch) -> Batch:
        from dew.batch import coordinates

        ids = self.fields[0]
        if f"{ids}_segment_ids" not in element:
            # A batch read without the training packing (a validation set of whole rows).
            return element
        segments = np.asarray(element[f"{ids}_segment_ids"])
        held = int(segments.max(initial=0))
        if self.max_segments is not None and held > self.max_segments:
            raise ValueError(f"a packed {ids} window holds {held} documents, past the {self.max_segments} "
                             "its read declares (Coordinates.max_segments)")
        packing = {f"{field}_{part}" for field in self.fields for part in ("segment_ids", "positions")}
        laid = coordinates(segments=segments, positions=element[f"{ids}_positions"],
                           max_segments=self.max_segments, right_padded=True, xp=np)
        return {**{name: value for name, value in element.items() if name not in packing},
                f"{ids}_coordinates": laid}


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
