"""What a fit's start reads of its data: its first batch once, kept from the read that declares the model's
input to the run's first step (training.md, the batch contract; row 6663)."""

import collections

import grain.python as grain
import jax
import numpy as np
import optax
from flax import linen as nn

from dew.objectives.supervised import Supervised
from dew.training import Trainer


def test_a_named_input_run_over_a_grain_iterable_reads_its_first_record_once():
    """A run handed a Grain iterable as it is, its objective naming its input field alone (so its start
    reads the first batch's shape before the model initializes), reads its first record once: the loader's
    probe of the batch size, the input's declaration and the run's first step take the one read. It was
    read three times, an iterator that cannot be read again (a stream from a socket) losing its first
    batches to the start."""
    reads: collections.Counter[int] = collections.Counter()
    rows = 2 * jax.device_count()

    def record(index):
        reads[index] += 1
        return {"x": np.full((8,), index, np.float32), "y": np.zeros((4,), np.float32)}

    data = (grain.MapDataset.range(8 * rows).map(record).batch(rows)
            .to_iter_dataset(grain.ReadOptions(num_threads=0, prefetch_buffer_size=0)))
    objective = Supervised(nn.Dense(4), lambda outputs, batch: (outputs - batch["y"]) ** 2, inputs="x")
    Trainer(objective, optax.sgd(0.1), key=jax.random.key(0)).fit(data, steps=2, log_every=100)
    assert reads[0] == 1, reads
