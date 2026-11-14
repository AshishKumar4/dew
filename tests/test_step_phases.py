"""Phases on the step clock over reads cut per share (row 5480(a)): a decision run's rows are batched by their
token budget after steps that drop the rows that do not fit, so its order is each share's own, and its phases
read each its own stream (`dew.data.dataset.StepPhases`), opened at the step the phase before it ends, its
position saved and resumed as its own; every batch says the phase it was read in (`dew.batch.PHASE`)."""

import dataclasses
import itertools
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from affine_run import FEATURES, Affine
from test_chain_compile import Tokens
from test_data_mixtures import READ, assert_same, checkpointed_trainer, parameters

from dew.batch import PHASE, VALID_ROWS, Phase
from dew.data import Memory, mix, phases
from dew.data.compile import compile
from dew.data.dataset import DataPartition
from dew.mesh import LayoutRefused
from dew.objectives.base import Aux, Objective
from dew.training import Checkpoints, Layout, Trainer


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


def resumes_as_read(loader, whole, stops) -> None:
    """A read of `loader` stopped at each of `stops`, saved and resumed, reads `whole` (an unstopped read's
    steps) on from where it stopped."""
    for stop in stops:
        head, stopped = read(loader, stop)
        saved = stopped.get_state()
        stopped.close()
        resumed = loader.train(DataPartition())
        resumed.set_state(saved)
        rest, resumed = read(loader, len(whole) - stop, resumed)
        resumed.close()
        assert head + rest == whole, stop


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
    resumes_as_read(loader, whole, range(1, 7))


def test_a_resume_into_phases_that_moved_an_end_the_run_passed_is_refused():
    """A run saved in its second phase resumes only into phases whose first ends where the saved run's did:
    a phase the run has read cannot change under it."""
    _, stopped = read(compile(chain(3), read="indexed", loading=READ), 5)
    saved = stopped.get_state()
    stopped.close()
    moved = compile(chain(2), read="indexed", loading=READ).train(DataPartition())
    with pytest.raises(ValueError):
        moved.set_state(saved)
    moved.close()


def member(tag: int, endless: bool = False, rows: int = 48):
    """A corpus of `rows` rows whose ids say they are the `tag`th source's, in id order, of uneven lengths,
    some dropped as too long: a mix's member, which its mix reads again from its start where it runs out,
    or `endless` itself."""
    held = [{"id": np.int32(1000 * tag + index), "length": np.int32(3 + index % 11)} for index in range(rows)]
    chain = Memory(held).map(Tokens()).filter(fits, reason="too_long")
    return chain.repeat() if endless else chain


SHARED_ROWS = 192

SHARED = member(1, rows=SHARED_ROWS)
"""The corpus phases share: its ids in its order, those that fit (`KEPT`)."""

KEPT = [1000 + index for index in range(SHARED_ROWS) if (3 + index % 11) % 5 != 0]


def first_pass(batches, records: int):
    """`batches`' steps until they trained more than `records` of `SHARED`, short of its mix reading it a
    second time: what a read trains of it across its phases' boundaries, each record once."""
    kept, seen = [], 0
    for ids, phase in batches:
        seen += sum(row // 1000 == 1 for row in ids)
        if seen > records:
            break
        kept.append((ids, phase))
    return kept


def shared_phases(elastic: bool = False):
    """Two phases on the step clock sharing `SHARED`, each its own mix: the first with another corpus, its
    draws shuffled within windows of six after the mix and batched by 128 tokens, to step 3; the second
    with a third corpus at other weights, unshuffled and batched by 64 tokens, on; each
    read elastically where asked (`Pipeline.elastic`)."""
    first = mix([SHARED, member(2, endless=True)], weights=(0.5, 0.5)).shuffle(1, streamed=6)
    second = mix([SHARED, member(3, endless=True)], weights=(0.25, 0.75))
    if elastic:
        first, second = first.elastic(), second.elastic()
    return phases((first.batch(tokens=128), 3), (second.batch(tokens=64), None))


def shared_ids(batches) -> tuple[list[int], list[int]]:
    """The ids of `SHARED` each phase of `batches` trains, in order."""
    by_phase: tuple[list[int], list[int]] = ([], [])
    for ids, phase in batches:
        by_phase[phase.index].extend(row for row in ids if row // 1000 == 1)
    return by_phase


@pytest.mark.parametrize("read_as", ["indexed", "streamed"])
def test_phases_that_share_a_corpus_train_each_of_its_records_once_across_their_boundary(read_as):
    """Two phases on their own streams sharing a corpus, the first shuffling its draws after its mix and the
    second batching them otherwise: the second continues the corpus where the run left it, so each of its
    records is trained once, none lost, the ones the first phase's shuffle still held trained by the
    second, until its mix reads it again; and a read stopped at any step, in either phase or at their
    boundary, resumed reads what the read never stopped reads."""
    loader = compile(shared_phases(), read=read_as, loading=READ)
    whole, stream = read(loader, 24)
    stream.close()
    first, second = shared_ids(first_pass(whole, len(KEPT)))
    assert first and second and not set(first) & set(second)
    trained = sorted(first + second)
    assert len(trained) == len(set(trained)) and trained == KEPT[:len(trained)]
    assert first != sorted(first), "the first phase's shuffle reorders the draws it trains"
    resumes_as_read(loader, whole, (2, 3, 4, 6))


@pytest.mark.parametrize("read_as", ["indexed", "streamed"])
def test_phases_that_share_a_concat_continue_its_join_across_their_boundary(read_as):
    """Two phases on their own streams sharing a concatenation, a member of each one's mix: the second
    continues the join where the run left it, so each of its records is trained once across the boundary,
    in the join's order (the first dataset's, then the second's), until its mix reads it again; and a read
    stopped in either phase or at their boundary, resumed, reads what the read never stopped reads."""
    from dew.data import concat

    joined = concat([member(1, rows=96), member(4, rows=96)])
    kept = [1000 * tag + index for tag in (1, 4) for index in range(96) if (3 + index % 11) % 5 != 0]
    first = mix([joined, member(2, endless=True)], weights=(0.5, 0.5))
    second = mix([joined, member(3, endless=True)], weights=(0.25, 0.75))
    loader = compile(phases((first.batch(tokens=128), 3), (second.batch(tokens=64), None)), read=read_as,
                     loading=READ)
    whole, stream = read(loader, 24)
    stream.close()
    by_phase: tuple[list[int], list[int]] = ([], [])
    for ids, phase in whole:
        by_phase[phase.index].extend(row for row in ids if row // 1000 in (1, 4))
    trained = (by_phase[0] + by_phase[1])[:len(kept)]
    assert by_phase[0] and by_phase[1] and trained == kept[:len(trained)]
    resumes_as_read(loader, whole, (2, 3, 4, 6))


def test_three_phases_sharing_a_corpus_through_a_short_shuffled_middle_train_each_record_once():
    """Three phases sharing a corpus, the first two shuffling their draws after their mixes and the middle one
    a single step, which ends before it draws past every pull the first trained ahead: the third continues
    from those too, so each record is trained once, none lost; one of the first's members is itself a mix,
    which draws nothing the run carries; and a read stopped in the middle or the last phase, resumed, reads
    what the read never stopped reads."""
    nested = mix([member(4, endless=True), member(5, endless=True)])
    first = mix([SHARED, nested], weights=(0.5, 0.5)).shuffle(1, streamed=6).batch(tokens=128)
    middle = mix([SHARED, member(2, endless=True)], weights=(0.5, 0.5)).shuffle(2, streamed=4)
    middle = middle.batch(tokens=64)
    last = mix([SHARED, member(3, endless=True)], weights=(0.25, 0.75)).batch(tokens=64)
    loader = compile(phases((first, 3), (middle, 4), (last, None)), read="indexed", loading=READ)
    whole, stream = read(loader, 24)
    stream.close()
    passed = first_pass(whole, len(KEPT))
    trained = [row for ids, _ in passed for row in ids if row // 1000 == 1]
    assert {phase.index for _, phase in passed} == {0, 1, 2}
    assert len(trained) == len(set(trained)) and sorted(trained) == KEPT[:len(trained)]
    resumes_as_read(loader, whole, (3, 4, 5))


def test_phases_that_share_a_corpus_continue_it_on_every_share_of_their_reads():
    """Two shares of phases sharing a corpus each continue their own cut of it: on each, every record of
    the corpus the share reads is trained once across the boundary, and no share trains another's."""
    loader = compile(shared_phases(), read="indexed", loading=READ)
    held: list[list[int]] = []
    for index in range(2):
        batches, stream = read(loader, 16, loader.train(DataPartition(index, 2)))
        stream.close()
        # Short of the share's own cut of the corpus read again.
        first, second = shared_ids(first_pass(batches, len(KEPT) // 2 - 4))
        assert not set(first) & set(second)
        held.append(first + second)
    assert not set(held[0]) & set(held[1])
    assert all(len(ids) == len(set(ids)) for ids in held)


@pytest.mark.parametrize("resumed", [1, 4])
def test_elastic_phases_that_share_a_corpus_resume_inside_the_second_on_another_share_count(resumed):
    """Elastic phases sharing a corpus, saved by two shares inside the second phase, resume on another count
    of shares: between them they read the next steps' rows one reader reads, the shared corpus continued
    where the run left it, as every reader of an elastic read names its draws alike."""
    loader = compile(shared_phases(elastic=True), read="indexed", loading=READ)
    whole = [ids for ids, _ in read(loader, 7, loader.train(DataPartition()))[0]]
    shares = [loader.train(DataPartition(index, 2)) for index in range(2)]
    for share in shares:
        list(itertools.islice(share, 5))
    (state,) = {share.get_state() for share in shares}
    again = [loader.train(DataPartition(index, resumed)) for index in range(resumed)]
    for share in again:
        share.set_state(state)
    for step in (5, 6):
        held = sorted(itertools.chain.from_iterable(_ids(next(share)) for share in again))
        assert held == sorted(whole[step])


def test_phases_that_share_a_corpus_otherwise_than_as_one_member_each_are_refused():
    """A phase that reads a corpus another reads other than as a member of its own mix, whose position no key
    names, phases that read it through other steps before their mixes, where the one's draws are no draws of
    the other's, and phases that read it under other keys, which draw and shuffle it otherwise, are refused,
    as the run could not continue the corpus where it left it."""
    with pytest.raises(ValueError):
        compile(phases((source(1), 3), (source(1), None)), read="indexed", loading=READ)
    other = mix([SHARED.map(lambda row: row), member(3, endless=True)]).batch(tokens=64)
    with pytest.raises(ValueError):
        compile(phases((mix([SHARED, member(2, endless=True)]).batch(tokens=128), 3), (other, None)),
                read="indexed", loading=READ)
    # Moved to another place among the unnamed, it is drawn and shuffled under another key.
    moved = mix([member(3, endless=True), SHARED]).batch(tokens=64)
    with pytest.raises(ValueError):
        compile(phases((mix([SHARED, member(2, endless=True)]).batch(tokens=128), 3), (moved, None)),
                read="indexed", loading=READ)


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
    return checkpointed_trainer(Counted(), 0.1, directory)


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


SAVED_BY_MAIN = Path(__file__).with_name("fixtures") / "positions" / "main-0c8cf2384"
"""Positions main saved at 0c8cf2384, under plan 14, of the chains `saved_positions_support` builds
(`tools/positions_saved_by_main.py`): phases on their own streams inside the first, and a bounded mixture's
read while its layout or its held shuffle holds counts."""


@pytest.mark.parametrize("saved", ["step_phases", "step_phases_later", "bounded_mixture_strided",
                                   "bounded_mixture_held"])
def test_a_position_main_saved_under_plan_14_resumes_as_it_was_read(saved):
    """A position main saved under plan 14, which described its order's stages in place, of phases on their
    own streams, in the first or past it (its finished phase named by its plan's identity, which held 14's
    version, `saved_identity`), or of a bounded mixture read while its layout or its held shuffle holds
    counts, resumes, and reads what a read of the same chain stopped where it stopped reads after it."""
    from saved_positions_support import SAVED

    loader, steps = SAVED[saved]
    streams = [loader().train(DataPartition()) for _ in range(2)]
    list(itertools.islice(streams[0], steps))
    streams[1].set_state((SAVED_BY_MAIN / f"{saved}.position").read_bytes())
    # The rest of each read, the stopped one's and the restored one's, batch by batch, eight steps on.
    after = [[_ids(batch) for batch in itertools.islice(stream, 8)] for stream in streams]
    for stream in streams:
        stream.close()
    assert after[1] == after[0] and after[0]
