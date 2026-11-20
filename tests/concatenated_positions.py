"""The chains whose positions main saved at 45131ca01 (`tests/fixtures/positions/main-45131ca01`), while
Grain's concatenating packer laid a concatenating pack's windows (row 9059): documents concatenated, their
cuts overlapped or not. Apart from `saved_positions_support`, whose steps' identity is their code there."""

import numpy as np
from saved_positions_support import LENGTHS, READ

from dew.data import Memory
from dew.data.compile import compile
from dew.data.pipeline import Pipeline


def concatenated(overlap: int):
    """Documents of `LENGTHS` concatenated into windows of eight ids, each cut through a document starting
    again `overlap` ids before it, read streamed, in batches of two."""
    pieces = [{"id": np.arange(length, dtype=np.int32) + 1000 * (index + 1)}
              for index, length in enumerate(LENGTHS)]
    return compile(Pipeline(Memory(pieces)).pack(8, fields=("id",), policy="concat", overlap=overlap)
                   .repeat().batch(2), read="streamed", loading=READ)


SAVED = {"concatenated": (lambda: concatenated(0), 3),
         "concatenated_overlapped": (lambda: concatenated(1), 3),
         "concatenated_overlapped_later": (lambda: concatenated(1), 5)}
"""Each saved position's loader and the steps read before it was saved."""
