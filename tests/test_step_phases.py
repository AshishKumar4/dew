"""Phases on the step clock over reads cut per share (row 5480(a)): a decision run's rows are batched by their
token budget after steps that drop the rows that do not fit, so its order is each share's own, and its phases
read each its own stream (`dew.data.dataset.StepPhases`), opened at the step the phase before it ends, its
position saved and resumed as its own; every batch says the phase it was read in (`dew.batch.PHASE`)."""

import dataclasses
import itertools
from pathlib import Path

import grain.python as grain
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from affine_run import FEATURES, Affine
from test_data_mixtures import READ, assert_same, parameters

from dew.batch import PHASE, VALID_ROWS, Phase
from dew.data import Memory, phases
from dew.data.compile import compile
from dew.data.dataset import DataPartition
from dew.objectives.base import Aux, Objective
from dew.training import Checkpoints, Layout, Trainer


@dataclasses.dataclass(frozen=True)
class Tokens(grain.MapTransform):
    """Rows of `max_length` tokens, `length` of them real, each its row's id (`TokenGeometry`)."""

    max_length: int = 16

    def map(self, row):
        return {**row, "tokens": np.full(self.max_length, row["id"], np.int32),
                "valid": np.arange(self.max_length) < row["length"]}


def fits(row) -> bool:
    """Whether a row fits its window: a decision run's step drops the rows that do not."""
    return int(row["length"]) % 5 != 0


def source(tag: int, count: int = 48):
    """A corpus of `count` rows whose ids say they are the `tag`th source's, of uneven lengths, some
    dropped, batched by a budget of 64 tokens: read per share."""
    rows = [{"id": np.int32(1000 * tag + index), "length": np.int32(3 + index % 11)}
            for index in range(count)]
    return Memory(rows).map(Tokens()).filter(fits, reason="too_long").repeat().batch(tokens=64)


def chain(*ends):
    """Two sources read in two phases on the step clock, the first's to `ends[0]`, the second's on."""
    return phases((source(1), ends[0]), (source(2), None))


def read(loader, steps: int, stream=None):
    """`steps` batches of `loader`'s training read, each its real rows' ids and its phase; and the stream."""
    stream = stream if stream is not None else loader.train(DataPartition())
    batches = []
    for batch in itertools.islice(stream, steps):
        keep = batch.get(VALID_ROWS, np.ones(len(batch["id"]), bool))
        batches.append((np.asarray(batch["id"])[keep].tolist(), batch[PHASE]))
    return batches, stream


@pytest.mark.parametrize("read_as", ["indexed", "streamed"])
def test_phases_over_reads_cut_per_share_each_read_their_own_rows_and_resume_as_read(read_as):
    """Two sources in two phases on the step clock, each batched by its tokens after a step that drops
    rows: every batch of phase p holds rows of its own source alone and says its phase, and a read stopped
    at any step, inside either phase or at their boundary, resumed reads what the read never stopped
    reads."""
    loader = compile(chain(3), read=read_as, loading=READ)
    whole, stream = read(loader, 7)
    stream.close()
    assert [phase for _, phase in whole] == [Phase(0)] * 3 + [Phase(1)] * 4
    assert all(ids and {row // 1000 for row in ids} == {phase.index + 1} for ids, phase in whole)
    for stop in range(1, 7):
        head, stopped = read(loader, stop)
        saved = stopped.get_state()
        stopped.close()
        resumed = loader.train(DataPartition())
        resumed.set_state(saved)
        rest, resumed = read(loader, 7 - stop, resumed)
        resumed.close()
        assert head + rest == whole, stop


def test_a_resume_into_phases_that_moved_an_end_the_run_passed_is_refused():
    """A run saved in its second phase resumes only into phases whose first ends where the saved run's did:
    a phase the run has read cannot change under it."""
    _, stopped = read(compile(chain(3), read="indexed", loading=READ), 5)
    saved = stopped.get_state()
    stopped.close()
    moved = compile(chain(2), read="indexed", loading=READ).train(DataPartition())
    with pytest.raises(ValueError, match="cannot change under it"):
        moved.set_state(saved)
    moved.close()


def test_phases_read_each_on_its_own_stream_refuse_a_corpus_two_of_them_read():
    """Phases read each on its own stream open each corpus at its first record, so a corpus two of them read
    would be read again from its start: refused, as reading one global order is what continues it."""
    with pytest.raises(ValueError, match="both read"):
        compile(phases((source(1), 3), (source(1), None)), read="indexed", loading=READ)


class Counted(Objective):
    """Squared error of an affine map of each row's first tokens against twice its first two."""

    def __init__(self):
        self.model = Affine()

    def fresh_variables(self, key, held):
        return self.model.init(key, jnp.zeros((1, FEATURES)))

    def loss(self, variables, batch, step):
        rows = batch["tokens"][:, :FEATURES].astype(jnp.float32) / 1000.0
        return jnp.mean((self.model.apply(variables, rows) - 2 * rows[:, :2]) ** 2), Aux({})


def trainer(directory: Path) -> Trainer:
    return Trainer(Counted(), optax.sgd(0.1), key=jax.random.key(0),
                   layout=Layout(min_shard=1, tolerance=1.0), checkpoints=Checkpoints(str(directory), keep=8))


def test_a_run_over_phases_cut_per_share_stopped_in_either_phase_lands_where_the_uninterrupted_one_does(
        tmp_path):
    """A run over two phases of reads cut per share, stopped inside its first phase and again inside its
    second, each time resumed from its checkpoint, lands where the run nobody stopped lands."""
    data = compile(chain(3), read="indexed", loading=READ)
    whole = trainer(tmp_path / "whole").fit(data, steps=6, log_every=100, checkpoint_every=None)
    for stop in (2, 4, 6):
        resumed = trainer(tmp_path / "resumed").fit(data, steps=stop, log_every=100, checkpoint_every=1)
    assert int(resumed.step) == 6
    assert_same(parameters(whole), parameters(resumed), "the resumed phases did not land on the whole run")
