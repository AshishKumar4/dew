"""A request's images, as the backbone's own processor lays them out in a row."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Self

import numpy as np
from PIL import Image

from dew.decision.layout import Encoded
from dew.interop.processors import Processor
from dew.nn.inputs import ModelInputs


@dataclass(frozen=True)
class Images:
    """A request's images as the backbone's processor prepares them.

    That is the tokens it expands them to, which a layout places in the row
    (`Layout.images`), each of its per-token fields over those tokens, and its
    pixel fields. The processor runs over the images' texts and a newline,
    and the row's text tokens take 0 in every per-token field.
    """

    tokens: tuple[int, ...]
    aligned: Mapping[str, np.ndarray]
    fields: Mapping[str, np.ndarray]

    @classmethod
    def of(cls, processor: Processor, images: Sequence[Image.Image], text: str) -> Self:
        """Run `processor` over `images`, each standing in the text as `text`."""
        # In the tensor type the processor computes in, read on the host as its model inputs are.
        values = processor.host_media([text * len(images) + "\n"], images=list(images))
        ids = np.asarray(values["input_ids"])
        arrays = {name: np.asarray(value) for name, value in values.items()
                  if name not in ("input_ids", "attention_mask")}
        return cls(tuple(int(token) for token in ids[0]),
                   {name: array[0] for name, array in arrays.items() if array.shape == ids.shape},
                   {name: array for name, array in arrays.items() if array.shape != ids.shape})

    def alike(self, other: Images) -> bool:
        """Whether a row of these images and a row of `other` are one pass's (`joined`): the same
        per-token and pixel fields, of one shape each."""
        return all(set(mine) == set(theirs) and all(mine[name].shape == theirs[name].shape for name in mine)
                   for mine, theirs in ((self.aligned, other.aligned), (self.fields, other.fields)))

    @staticmethod
    def joined(processor: Processor, rows: Sequence[tuple[Encoded, Images]]) -> ModelInputs:
        """The backbone's inputs for `rows`, each holding its images at its `media` span, as one batch:
        each row's per-token fields spread over its row, and the pixel fields joined along their first
        axis, the processor's own batch order. The rows are as long as each other and their images
        alike (`alike`), so no row pads and the fields join as the processor batches them."""
        widths = {len(row.tokens) for row, _ in rows}
        if len(widths) != 1 or not all(rows[0][1].alike(media) for _, media in rows):
            raise ValueError("rows of images join in one pass where they are as long as each other and "
                             "their images alike")
        if len(rows) == 1:
            return rows[0][1].inputs(processor, rows[0][0])
        aligned = {name: np.concatenate([media.spread(row)[name] for row, media in rows])
                   for name in rows[0][1].aligned}
        fields = {name: np.concatenate([media.fields[name] for _, media in rows])
                  for name in rows[0][1].fields}
        tokens = np.asarray([row.tokens for row, _ in rows])
        return processor.from_hf({"input_ids": tokens, **aligned, **fields})

    def spread(self, row: Encoded) -> dict[str, np.ndarray]:
        """Each per-token field over `row`, its images' values at its `media` span and 0 elsewhere."""
        if row.media is None or row.media[1] - row.media[0] != len(self.tokens):
            raise ValueError("the row does not hold these images")
        start, end = row.media
        aligned = {}
        for name, values in self.aligned.items():
            spread = np.zeros((1, len(row.tokens)), values.dtype)
            spread[0, start:end] = values
            aligned[name] = spread
        return aligned

    def inputs(self, processor: Processor, row: Encoded) -> ModelInputs:
        """Return the backbone's inputs for `row`, which holds these images at its `media` span."""
        return processor.from_hf({"input_ids": np.asarray([row.tokens]), **self.spread(row), **self.fields})
