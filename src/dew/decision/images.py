"""A request's images, as the backbone's own processor lays them out in a row."""

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
    (`Layout.image`), each of its per-token fields over those tokens, and its
    pixel fields. This is what Clef's `_encode_media` (joint_schema_model.py at
    Cloudflare/clef 2f3de3dd) does: it runs the processor over the images'
    texts and a newline, and the row's text tokens take 0 in every per-token
    field.
    """

    tokens: tuple[int, ...]
    aligned: Mapping[str, np.ndarray]
    fields: Mapping[str, np.ndarray]

    @classmethod
    def of(cls, processor: Processor, images: Sequence[Image.Image], text: str) -> Self:
        """Run `processor` over `images`, each standing in the text as `text`."""
        values = processor.reference(text=[text * len(images) + "\n"], images=list(images),
                                     return_tensors="np")
        ids = np.asarray(values["input_ids"])
        arrays = {name: np.asarray(value) for name, value in values.items()
                  if name not in ("input_ids", "attention_mask")}
        return cls(tuple(int(token) for token in ids[0]),
                   {name: array[0] for name, array in arrays.items() if array.shape == ids.shape},
                   {name: array for name, array in arrays.items() if array.shape != ids.shape})

    def inputs(self, processor: Processor, row: Encoded) -> ModelInputs:
        """Return the backbone's inputs for `row`, which holds these images at its `media` span."""
        if row.media is None or row.media[1] - row.media[0] != len(self.tokens):
            raise ValueError("the row does not hold these images")
        start, end = row.media
        aligned = {}
        for name, values in self.aligned.items():
            spread = np.zeros((1, len(row.tokens)), values.dtype)
            spread[0, start:end] = values
            aligned[name] = spread
        return processor.from_hf({"input_ids": np.asarray([row.tokens]), **aligned, **self.fields})
