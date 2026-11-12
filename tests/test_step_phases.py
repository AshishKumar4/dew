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
from dew.mesh import LayoutRefused
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


def source(tag: int):
    """A corpus of 48 rows whose ids say they are the `tag`th source's, of uneven lengths, some dropped,
    batched by a budget of 128 tokens (eight rows, which a mesh of eight devices divides): read per
    share."""
    return _budget(tag, 128)


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
    would be read again from its start: refused, as reading one global order is what continues it, whatever
    each phase names it."""
    with pytest.raises(ValueError, match="both read"):
        compile(phases((source(1), 3), (source(1), None)), read="indexed", loading=READ)
    with pytest.raises(ValueError, match="both read"):
        compile(phases((source(1).named("early"), 3), (source(1).named("late"), None)), read="indexed",
                loading=READ)


@pytest.mark.parametrize(("repeats", "ends"), [(None, "never"), (1, "apart")])
def test_phases_say_where_they_end_before_any_is_read(repeats, ends):
    """A read of phases each on its own stream says where it ends before it reads a batch, as the run asks
    it as it opens: never where every phase's corpus repeats, apart where one may run out on a share."""
    def finite(tag):
        rows = [{"id": np.int32(1000 * tag + index), "length": np.int32(3 + index % 11)}
                for index in range(48)]
        return Memory(rows).map(Tokens()).filter(fits, reason="too_long").repeat(repeats).batch(tokens=128)

    stream = compile(phases((finite(1), 3), (finite(2), None)), read="indexed", loading=READ).train(
        DataPartition())
    assert stream.ends == ends
    stream.close()


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


def test_a_run_over_phases_of_other_batches_lays_each_update_out_in_its_own_phase(tmp_path):
    """Phases of other token budgets, sixteen rows then eight, under updates of two steps: the run lays each
    update's window out in its phase's batch, the second phase beginning where an update does, and stopped
    in it and resumed lands where the run nobody stopped lands. A later phase whose batch the mesh cannot
    divide is refused before any step trains."""
    data = compile(phases((_budget(1, 256), 2), (_budget(2, 128), None)), read="indexed", loading=READ)

    def accumulating(directory):
        return Trainer(Counted(), optax.sgd(0.1), key=jax.random.key(0), accumulation=2,
                       layout=Layout(min_shard=1, tolerance=1.0),
                       checkpoints=Checkpoints(str(directory), keep=8))

    whole = accumulating(tmp_path / "whole").fit(data, steps=6, log_every=100, checkpoint_every=None)
    accumulating(tmp_path / "resumed").fit(data, steps=4, log_every=100, checkpoint_every=2)
    resumed = accumulating(tmp_path / "resumed").fit(data, steps=6, log_every=100, checkpoint_every=2)
    assert int(resumed.step) == 6
    assert_same(parameters(whole), parameters(resumed), "the resumed phases did not land on the whole run")
    indivisible = compile(phases((_budget(1, 128), 2), (_budget(2, 64), None)), read="indexed", loading=READ)
    with pytest.raises(LayoutRefused):
        trainer(tmp_path / "indivisible").fit(indivisible, steps=4, log_every=100, checkpoint_every=1)
    assert not (tmp_path / "indivisible").exists() or not any((tmp_path / "indivisible").iterdir())


def _budget(tag: int, tokens: int):
    """`source(tag)`'s rows batched by a budget of `tokens` tokens."""
    rows = [{"id": np.int32(1000 * tag + index), "length": np.int32(3 + index % 11)} for index in range(48)]
    return Memory(rows).map(Tokens()).filter(fits, reason="too_long").repeat().batch(tokens=tokens)


def test_phases_name_what_each_shares_reader_reads_as_a_pool_agrees_it():
    """A read of phases each on its own stream names each phase as that phase's reader names itself, its
    plan as this share reads it, its loading workers among it: shares whose streamed phases are read in other
    workers, and so in other orders, name other reads, as a share of a read of one phase does, where the
    loader's own plan, which states no workers, named them alike."""
    def named(workers: int) -> tuple[str, str]:
        loading = dataclasses.replace(READ, workers=workers)
        phased = compile(chain(3), read="streamed", loading=loading).train(DataPartition())
        alone = compile(source(1), read="streamed", loading=loading).train(DataPartition())
        try:
            return phased.identity(), alone.identity()
        finally:
            phased.close()
            alone.close()

    (phased_none, alone_none), (phased_two, alone_two) = named(0), named(2)
    assert alone_none != alone_two, "a share's read names its workers"
    assert phased_none != phased_two
    assert alone_none in phased_none and alone_two in phased_two


def _unthird(row) -> bool:
    """Keeps a record whose id three does not divide: a filter, which makes the order a share's own."""
    return bool(int(row["id"]) % 3)


def _elastic(tag: int):
    """A shuffled, filtered corpus read elastically (`Pipeline.elastic`): one order every share reads,
    each keeping its share of each batch, so its position is every share's."""
    from dew.data.pipeline import Pipeline

    rows = tuple({"id": np.int64(1000 * tag + index)} for index in range(96))
    return (Pipeline(Memory(rows)).shuffle(5).repeat().filter(_unthird)
            .batch(8).elastic())


def _ids(batch) -> list[int]:
    keep = batch.get(VALID_ROWS, np.ones(len(batch["id"]), bool))
    return np.asarray(batch["id"])[keep].tolist()


@pytest.mark.parametrize(("saved", "resumed"), [(2, 1), (2, 4)])
def test_phases_of_elastic_reads_resume_on_another_share_count(saved, resumed):
    """Phases on the step clock over elastic reads, each its own stream: after three steps, the second
    phase's first read, every share holds one position, which a checkpoint lets another share count read
    (`dew._position.translates`, as the phase's own elastic position is), and the shares of another count
    restored from it read the next steps' rows one reader reads, between them."""
    from dew import _position

    def loader():
        return compile(phases((_elastic(1), 2), (_elastic(2), None)), loading=READ, seed=3)

    shares = [loader().train(DataPartition(index, saved)) for index in range(saved)]
    for share in shares:
        list(itertools.islice(share, 3))
    (state,) = {share.get_state() for share in shares}
    assert _position.translates(state)
    whole = [_ids(batch) for batch in itertools.islice(loader().train(DataPartition()), 5)][3:]
    again = [loader().train(DataPartition(index, resumed)) for index in range(resumed)]
    for share in again:
        share.set_state(state)
    for step in range(2):
        held = itertools.chain.from_iterable(_ids(next(share)) for share in again)
        assert sorted(held) == sorted(whole[step])
