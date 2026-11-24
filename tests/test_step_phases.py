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
from test_data_mixtures import READ, Indexed, assert_same, checkpointed_trainer, ids as batch_ids, parameters

from dew.batch import PHASE, VALID_ROWS, Phase
from dew.data import Memory, Phase as DataPhase, Steps, mix, phases
from dew.data.compile import compile
from dew.data.dataset import Closeable, DataPartition
from dew.data.pipeline import Pipeline
from dew.mesh import LayoutRefused
from dew.objectives.base import Aux, Objective
from dew.training import Checkpoints, Layout, Trainer, TrainerAdvanced


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
    return phases(DataPhase(source(1), Steps(ends[0])), DataPhase(source(2)))


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
    return phases(DataPhase(first.batch(tokens=128), Steps(3)), DataPhase(second.batch(tokens=64)))


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
    loader = compile(phases(DataPhase(first.batch(tokens=128), Steps(3)),
                            DataPhase(second.batch(tokens=64))), read=read_as,
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
    loader = compile(phases(DataPhase(first, Steps(3)), DataPhase(middle, Steps(4)), DataPhase(last)),
                     read="indexed", loading=READ)
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


@pytest.mark.parametrize("read_as", ["indexed", "streamed"])
def test_phases_that_read_a_corpus_alone_continue_it_across_their_boundary(read_as):
    """Two phases on their own streams each reading a corpus alone, in batches of their own: the second
    continues it where the run left it, each record trained once and in order, as phases reading it as the
    one member of a mix of their own do (`_alone_mixed`), where they were refused (row 5400); and a read
    stopped in either phase or at their boundary, resumed, reads what the read never stopped reads."""
    alone = phases(DataPhase(SHARED.repeat().batch(tokens=128), Steps(3)),
                   DataPhase(SHARED.repeat().batch(tokens=64)))
    loader = compile(alone, read=read_as, loading=READ)
    whole, stream = read(loader, 24)
    stream.close()
    first, second = shared_ids(first_pass(whole, len(KEPT)))
    assert first and second
    assert first + second == KEPT[:len(first) + len(second)]
    resumes_as_read(loader, whole, (2, 3, 4, 6))


def test_phases_that_share_a_corpus_otherwise_than_as_one_member_each_are_refused():
    """A phase that reads a corpus another reads other than alone or as a member of its own mix (a member
    of a mix inside it), whose position no key names, phases that read it through other steps before their
    mixes, where the one's draws are no draws of the other's, and phases that read it under other keys,
    which draw and shuffle it otherwise, are refused, as the run could not continue the corpus where it
    left it."""
    nested = mix([mix([SHARED.repeat(), member(2, endless=True)]), member(3, endless=True)])
    with pytest.raises(ValueError):
        compile(phases(DataPhase(SHARED.repeat().batch(tokens=128), Steps(3)),
                       DataPhase(nested.batch(tokens=64))),
                read="indexed", loading=READ)
    other = mix([SHARED.map(lambda row: row), member(3, endless=True)]).batch(tokens=64)
    with pytest.raises(ValueError):
        compile(phases(DataPhase(mix([SHARED, member(2, endless=True)]).batch(tokens=128), Steps(3)),
                       DataPhase(other)),
                read="indexed", loading=READ)
    # Moved to another place among the unnamed, it is drawn and shuffled under another key.
    moved = mix([member(3, endless=True), SHARED]).batch(tokens=64)
    with pytest.raises(ValueError):
        compile(phases(DataPhase(mix([SHARED, member(2, endless=True)]).batch(tokens=128), Steps(3)),
                       DataPhase(moved)),
                read="indexed", loading=READ)


@pytest.mark.parametrize(("repeats", "ends"), [(None, "never"), (1, "apart")])
def test_phases_say_where_they_end_before_any_is_read(repeats, ends):
    """A read of phases each on its own stream says where it ends before it reads a batch, as the run asks
    it as it opens: never where every phase's corpus repeats, apart where one may run out on a share."""
    def finite(tag):
        rows = [{"id": np.int32(1000 * tag + index), "length": np.int32(3 + index % 11)}
                for index in range(48)]
        return Memory(rows).map(Tokens()).filter(fits, reason="too_long").repeat(repeats).batch(tokens=128)

    stream = compile(phases(DataPhase(finite(1), Steps(3)), DataPhase(finite(2))), read="indexed",
                     loading=READ).train(
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
    data = compile(phases(DataPhase(_budget(1, 256), Steps(2)), DataPhase(_budget(2, 128))), read="indexed",
                   loading=READ)

    def accumulating(directory):
        return Trainer(Counted(), optax.sgd(0.1), key=jax.random.key(0), accumulation=2,
                       advanced=TrainerAdvanced(layout=Layout(min_shard=1, tolerance=1.0)),
                       checkpoints=Checkpoints(str(directory), keep=8))

    whole = accumulating(tmp_path / "whole").fit(data, steps=6, log_every=100, checkpoint_every=None)
    accumulating(tmp_path / "resumed").fit(data, steps=4, log_every=100, checkpoint_every=2)
    resumed = accumulating(tmp_path / "resumed").fit(data, steps=6, log_every=100, checkpoint_every=2)
    assert int(resumed.step) == 6
    assert_same(parameters(whole), parameters(resumed), "the resumed phases did not land on the whole run")
    indivisible = compile(phases(DataPhase(_budget(1, 128), Steps(2)), DataPhase(_budget(2, 64))),
                          read="indexed", loading=READ)
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
        return compile(phases(DataPhase(_elastic(1), Steps(2)), DataPhase(_elastic(2))), loading=READ, seed=3)

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
    from resumed_support import resumed_as_read
    from saved_positions_support import SAVED

    loader, steps = SAVED[saved]
    # The rest of each read, the stopped one's and the restored one's, batch by batch, eight steps on.
    stopped, restored = resumed_as_read(loader, steps, (SAVED_BY_MAIN / f"{saved}.position").read_bytes(),
                                        _ids, 8)
    assert restored == stopped and stopped


def test_a_profile_window_in_a_phase_that_reads_an_earlier_phases_batches_names_that_program(tmp_path):
    """A run over phases of batches of sixteen rows, then eight, then sixteen again, profiled in its third
    phase, names the step its window ran: the one compiled for the first phase's batches, which the third
    reuses with no compile of its own (`ProfileTraced.signature`, `StepCompiled.signature`). The latest
    compile before the window, the second phase's, was the one a profile reported (row 6822)."""
    pytest.importorskip("xprof.convert.raw_to_tool_data", reason="optional profile extra")
    from recording import RecordingTracker

    from dew.telemetry.records import ProfileTraced, StepCompiled
    from dew.training import ProfileWindow

    data = compile(phases(DataPhase(_budget(1, 256), Steps(2)), DataPhase(_budget(2, 128), Steps(4)),
                          DataPhase(_budget(3, 256))),
                   read="indexed", loading=READ)
    tracker = RecordingTracker()
    Trainer(Counted(), optax.sgd(0.1), key=jax.random.key(0),
            advanced=TrainerAdvanced(layout=Layout(min_shard=1, tolerance=1.0),
                                     profile=ProfileWindow(str(tmp_path / "window"), steps=1, warmup=4)),
            tracker=tracker).fit(
                data, steps=6, log_every=100)
    compiled = [value for _, value in tracker.artifacts if isinstance(value, StepCompiled)]
    (window,) = [value for _, value in tracker.artifacts if isinstance(value, ProfileTraced)]
    assert len(compiled) == 2 and compiled[0].signature != compiled[1].signature
    assert window.signature == compiled[0].signature


def _endless_ids(chain, count: int) -> list[list[int]]:
    """The ids of the first `count` batches a run reads of `chain`, read on past its end (`fitted`)."""
    from dew.data.compile import fitted
    from dew.data.pipeline import ObjectiveData

    loader, _ = fitted(chain, None, objective=ObjectiveData())
    stream = loader.train(DataPartition())
    assert isinstance(stream, Closeable)
    try:
        return batch_ids(itertools.islice(stream, count))
    finally:
        stream.close()


def test_a_phase_batched_inside_itself_reads_on_as_a_single_stream_does():
    """A phase that batches its own records, read on past its end by a run, reads as the same chain read
    alone does, its repeat before its batch: a pass that holds a batch and a half continues its last into
    the next pass, where a repeat after the batch dropped it and read the pass again from its start."""
    first, second = Memory(Indexed(1, 12)), Memory(Indexed(2, 12))
    phased = phases(DataPhase(Pipeline(first).batch(8), Steps(2)), DataPhase(Pipeline(second).batch(8)))
    expected = (_endless_ids(Pipeline(first).batch(8), 2) + _endless_ids(Pipeline(second).batch(8), 2))
    assert _endless_ids(phased, 4) == expected
    assert expected[1] == [1008, 1009, 1010, 1011, 1000, 1001, 1002, 1003]


def test_a_phase_whose_pass_holds_less_than_a_batch_reads_whole_batches():
    """A phase that batches its own records, each pass fewer than a batch, reads whole batches of its records
    on past its end, its repeat before its batch: a repeat after it made every pass one short batch, its
    last rows padded (and a batch that drops its short last one never yielded one, the run hanging)."""
    from dew.data.compile import fitted
    from dew.data.pipeline import ObjectiveData

    phased = phases(DataPhase(Pipeline(Memory(Indexed(1, 6))).batch(8, drop_remainder=False), Steps(1)),
                    DataPhase(Pipeline(Memory(Indexed(2, 6))).batch(8, drop_remainder=False)))
    loader, _ = fitted(phased, None, objective=ObjectiveData())
    stream = loader.train(DataPartition())
    assert isinstance(stream, Closeable)
    try:
        read = list(itertools.islice(stream, 2))
    finally:
        stream.close()
    assert [int(np.sum(batch.get(VALID_ROWS, np.ones(8, bool)))) for batch in read] == [8, 8]
    assert batch_ids(read) == [[1000, 1001, 1002, 1003, 1004, 1005, 1000, 1001],
                               [2000, 2001, 2002, 2003, 2004, 2005, 2000, 2001]]


def test_phases_batched_outside_themselves_read_as_they_did():
    """Phases that batch outside themselves, one stating a repeat of its own and the chain taking from both,
    read as a run always read them: the first phase's two passes read on to its end, and the second
    continues the corpus past what the first read of it."""
    corpus = Pipeline(Memory(Indexed(1, 64)))
    shared = _endless_ids(phases(DataPhase(corpus, Steps(6)), DataPhase(corpus)).take(16).batch(4), 8)
    assert shared[6] == [1024, 1025, 1026, 1027]
    finite = Pipeline(Memory(Indexed(1, 8)))
    second = Pipeline(Memory(Indexed(2, 8)))
    repeated = _endless_ids(phases(DataPhase(finite.repeat(2), Steps(10)), DataPhase(second)).batch(8), 11)
    assert len(repeated) == 11 and repeated[2][0] // 1000 == 1 and repeated[10][0] // 1000 == 2
    # A chain that states its own repeat reads as it states it, its phases batched inside themselves too.
    once = phases(DataPhase(Pipeline(Memory(Indexed(1, 16))).batch(8), Steps(1)),
                  DataPhase(Pipeline(Memory(Indexed(2, 16))).batch(8)))
    assert _endless_ids(once.repeat(1), 4) == [list(range(1000, 1008)), list(range(2000, 2008)),
                                               list(range(2008, 2016))]
