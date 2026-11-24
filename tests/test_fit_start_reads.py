"""What a fit's start reads of its data: its first batch once, kept from the read that declares the model's
input to the run's first step (training.md, the batch contract; row 6663)."""

import collections

import grain.python as grain
import jax
import numpy as np
import optax
import pytest
from flax import linen as nn

from dew.objectives.supervised import Supervised
from dew.training import Trainer, TrainerAdvanced


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


def test_a_named_input_run_resumed_exactly_over_a_changed_plan_reads_no_record_before_it_is_refused(tmp_path):
    """A named-input run resumed exactly over data whose plan changed (another shuffle) checks the saved plan
    before it reads any record, its input declared from the first batch the resume admits: refused having
    mapped none (row 9013: its start read the replacement stream's first batch before the plan refused)."""
    from dew.data import Memory
    from dew.data.compile import compile
    from dew.data.dataset import Loading
    from dew.data.pipeline import Pipeline
    from dew.training import Checkpoints

    reads: collections.Counter[int] = collections.Counter()
    rows = 2 * jax.device_count()

    def counted(row):
        reads[int(row["x"][0])] += 1
        return row

    columns = {"x": np.repeat(np.arange(8 * rows, dtype=np.float32)[:, None], 8, axis=1),
               "y": np.zeros((8 * rows, 4), np.float32)}

    def fit(seed: int, steps: int):
        chain = compile(Pipeline(Memory(columns), name="xs").map(counted, identity="counted@1")
                        .shuffle(seed=seed, window=16).repeat().batch(rows), read="streamed",
                        loading=Loading(workers=0),
                        resume="exact")
        objective = Supervised(nn.Dense(4), lambda outputs, batch: (outputs - batch["y"]) ** 2, inputs="x")
        trainer = Trainer(objective, optax.sgd(0.1), key=jax.random.key(0),
                          advanced=TrainerAdvanced(data_resume="exact"),
                          checkpoints=Checkpoints(str(tmp_path)))
        return trainer.fit(chain, steps=steps, log_every=100, checkpoint_every=1)

    fit(seed=1, steps=1)
    reads.clear()
    with pytest.raises(ValueError):
        fit(seed=2, steps=2)
    assert not reads, reads
