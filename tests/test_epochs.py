"""An epoch's records and steps, and whether an order ends: a mixture read on past its end counts
until every part has come round, a part a slice or a limit bounds comes round once its bounded
records are read, an order that ends counts no more than it holds, and a training stream says it
never runs out exactly where its stages, read in turn, make it so."""

import itertools

import numpy as np
import pytest

from dew.data.compile import compile
from dew.data.dataset import DataPartition, Loading
from dew.data.pipeline import Pipeline, mix
from dew.data.sources import Memory

READ = Loading(workers=0, threads=1, read_buffer=0, worker_buffer=1)


def _tagged(tag, count):
    """`count` records whose ids say they are corpus `tag`'s: 100 * tag on."""
    return Pipeline(Memory(tuple({"id": np.int32(100 * tag + index)} for index in range(count))))


def _nested():
    """A mix at even weights of a corpus of 12 and a mix at even weights of corpora of 8 and 4."""
    return mix([_tagged(1, 12), mix([_tagged(2, 8), _tagged(3, 4)], weights=(1, 1))], weights=(1, 1))


def _ids(reader, batches=None):
    try:
        return [int(value) for batch in itertools.islice(reader, batches) for value in batch["id"]]
    finally:
        reader.close()


def test_a_nested_mixture_a_run_reads_on_counts_its_epoch_until_every_corpus_has_come_round():
    """A run reads a mixture on past its end, each part on from where it was, so its epoch is the
    fewest records in which every part comes round once, the inner mixture counted the same way:
    the inner one of 8 and 4 at even weights in 15, the outer of 12 and those 15 in 30. Every
    record of every corpus is read in it."""
    loader = compile(_nested().batch(2), loading=READ)
    assert loader.records == 30
    read = _ids(loader.train(DataPartition()), 15)
    corpora = {1: 12, 2: 8, 3: 4}
    assert {tag: len({value for value in read if value // 100 == tag}) for tag in corpora} == corpora


@pytest.mark.parametrize("epochs", [1, 2])
def test_a_nested_mixture_read_a_counted_number_of_times_counts_the_records_its_mix_reads(epochs):
    """A mixture repeated a counted number of times reads its mix to its end each time, where its
    first corpus ends, the inner mixture's at its own: the inner one of 8 and 4 ends at 8, the
    outer of 12 and those 8 at 16, and the run reads that many records each epoch and no more."""
    loader = compile(_nested().repeat(epochs).batch(2), loading=READ)
    assert loader.records == 16
    assert len(_ids(loader.train(DataPartition()))) == 16 * epochs


@pytest.mark.parametrize("taken", [8, 20])
def test_a_mixture_repeated_for_ever_then_taken_counts_no_more_than_it_reads(taken):
    """A mix of corpora of 8 and 4 at even weights, repeated for ever and then cut to its first
    records, counts the fewest in which every corpus comes round, 15, where the cut holds that
    many, and the cut where it holds fewer: a run reads that many and ends."""
    loader = compile(mix([_tagged(1, 8), _tagged(2, 4)], weights=(1, 1)).repeat().take(taken).batch(1),
                     loading=READ)
    assert loader.records == min(15, taken)
    assert len(_ids(loader.train(DataPartition()))) == taken


def test_a_mixture_of_a_part_cut_after_its_repeat_ends_where_that_part_does():
    """A mix of a part repeated for ever then cut to 8 records and a corpus repeated for ever, at even
    weights, ends where the cut part does, at 16, and counts that many: a run reads them and ends."""
    cut = mix([_tagged(1, 8), _tagged(2, 4)], weights=(1, 1)).repeat().take(8)
    loader = compile(mix([cut, _tagged(3, 8).repeat()], weights=(1, 1)).batch(1), loading=READ)
    assert loader.records == 16
    assert len(_ids(loader.train(DataPartition()))) == 16


def test_a_part_cut_after_its_repeat_comes_round_in_a_mix_read_on_once_its_cut_is_read():
    """A part repeated for ever and cut to 8 records, mixed at even weights with a corpus of 8 and read
    on past the mix's end, comes round once its 8 are read: an epoch is the 16 records in which both
    parts have, and holds the cut part's 8 and every record of the corpus."""
    cut = mix([_tagged(1, 8), _tagged(2, 4)], weights=(1, 1)).repeat().take(8)
    loader = compile(mix([cut, _tagged(3, 8)], weights=(1, 1)).repeat().batch(1), loading=READ)
    assert loader.records == 16
    read = _ids(loader.train(DataPartition()), loader.records)
    assert sorted(value for value in read if value < 300) == [*range(100, 104), *range(200, 204)]
    assert sorted(value for value in read if value >= 300) == list(range(300, 308))


def test_a_part_cut_between_two_repeats_comes_round_once_the_cut_its_last_repeat_reads_is_read():
    """A corpus of 8 repeated, cut to 3 and repeated again comes round, in a mix read on past its end,
    once the 3 its last repeat reads are: at even weights with a corpus of 3 an epoch is 6 records,
    holding the cut's 3 and the other corpus's 3."""
    part = _tagged(1, 8).repeat().take(3).repeat()
    loader = compile(mix([part, _tagged(2, 3)], weights=(1, 1)).repeat().batch(1), loading=READ)
    assert loader.records == 6
    assert sorted(_ids(loader.train(DataPartition()), loader.records)) == [100, 101, 102, 200, 201, 202]


def test_an_order_read_once_too_short_for_a_batch_it_drops_is_refused_an_epoch():
    """Three records read once in batches of four that drop a short one deliver no step, so a run
    counted in epochs of them is refused."""
    with pytest.raises(ValueError):
        compile(_tagged(1, 3).repeat(1).batch(4), loading=READ).epoch_steps(1)


@pytest.mark.parametrize(("keep", "steps"), [(False, 2), (True, 3)])
def test_an_order_read_once_counts_the_steps_its_batches_deliver(keep, steps):
    """Five records read once in batches of two take two steps where the batch drops its short last
    one, and three where it keeps it, filled: an epoch's steps are those its reader delivers."""
    loader = compile(_tagged(1, 5).repeat(1).batch(2, drop_remainder=not keep), loading=READ)
    assert loader.steps_per_epoch == steps
    reader = loader.train(DataPartition())
    try:
        assert len(list(reader)) == steps
    finally:
        reader.close()


def test_an_epoch_of_an_order_read_on_counts_the_step_that_completes_it():
    """Five records repeated for ever in batches of two take three steps an epoch, the third holding
    the last record and the next epoch's first, and two epochs five steps: the steps that read
    every record of each."""
    loader = compile(_tagged(1, 5).repeat().batch(2), loading=READ)
    assert (loader.steps_per_epoch, loader.epoch_steps(2)) == (3, 5)
    read = _ids(loader.train(DataPartition()), loader.steps_per_epoch)
    assert set(read) == set(range(100, 105))


def test_a_mixture_a_run_cuts_without_a_repeat_counts_its_cut():
    """A mix of two corpora of 8 cut to its first 3 records and written without a repeat counts those
    3 an epoch: the run's own repeat reads the cut."""
    loader = compile(mix([_tagged(1, 8), _tagged(2, 8)], weights=(1, 1)).take(3).batch(1), loading=READ)
    assert loader.records == 3


@pytest.mark.parametrize(("order", "endless"), [("take, then repeat", True), ("repeat, then take", False),
                                               ("neither", True), ("take", False)])
def test_a_training_stream_says_it_never_runs_out_by_its_stages_read_in_turn(order, endless):
    """A corpus of 8 cut to 4 and repeated for ever never runs out, and one repeated and then cut
    does, as one only cut does; one written with neither repeats for ever as training reads it. Its
    training stream says so, and reads on past 4 records exactly where it never runs out."""
    from dew.data.dataset import stream_endless

    rows = _tagged(1, 8)
    written = {"take, then repeat": rows.take(4).repeat(), "repeat, then take": rows.repeat().take(4),
               "neither": rows, "take": rows.take(4)}[order]
    reader = compile(written.batch(1), loading=READ).train(DataPartition())
    said = stream_endless(reader)
    assert len(_ids(reader, 6)) == (6 if endless else 4)
    assert said is endless


def test_a_finite_pass_repeated_after_its_batch_is_one_global_order_on_any_number_of_readers():
    """Three records cut, batched in twos and the batches repeated: the pass's one whole batch is
    replayed, the same records in the same steps on one reader or two."""
    chain = _tagged(1, 8).take(3).batch(2).repeat()
    loader = compile(chain, loading=READ)
    alone = [sorted(step) for step in _steps(loader.train(DataPartition()), 3)]
    shares = [_steps(loader.train(DataPartition(index, 2)), 3) for index in range(2)]
    together = [sorted(first + second) for first, second in zip(*shares, strict=True)]
    assert together == alone == [[100, 101]] * 3


def _steps(reader, count):
    try:
        return [[int(value) for value in batch["id"]] for batch in itertools.islice(reader, count)]
    finally:
        reader.close()


@pytest.mark.parametrize(("repeated", "endless"), [(False, False), (True, True)])
def test_a_grain_transformation_of_its_own_may_end_an_order_unless_a_repeat_follows(repeated, endless):
    """A Grain transformation of a chain's own (`grain`) may end an order a repeat made endless, as
    one that keeps the first two records does: its stream does not say it never runs out unless a
    repeat after it makes it so, and it reads what that makes."""
    from dew.data.dataset import stream_endless

    chain = _tagged(1, 8).take(4).repeat().grain(lambda rows: rows[:2])
    reader = compile((chain.repeat() if repeated else chain).batch(1), read="indexed",
                     loading=READ).train(DataPartition())
    said = stream_endless(reader)
    assert len(_ids(reader, 6)) == (6 if endless else 2)
    assert said is endless


def test_a_streamed_mix_of_a_repeated_part_and_one_written_without_says_it_never_runs_out():
    """A stream cut per share repeats a part written without a repeat at its own root, so a mix of
    a repeated part and one written without never runs out: its stream says so and reads on."""
    from dew.data.dataset import stream_endless

    reader = compile(mix([_tagged(1, 4).repeat(), _tagged(2, 2)], weights=(1, 1)).batch(1), read="streamed",
                     loading=READ).train(DataPartition())
    said = stream_endless(reader)
    assert len(_ids(reader, 12)) == 12
    assert said


def _not_201(row):
    return int(row["id"]) != 201


@pytest.mark.parametrize("shares", [1, 2])
def test_a_part_repeated_at_its_root_and_then_cut_ends_the_mix_that_reads_it(shares):
    """A read cut per share repeats a part written without a repeat at its own root, and the part's
    own limit then ends it: a mix of a repeated part and one cut to three and filtered runs out on
    every share, and its stream says so; its batch of two splits over two shares."""
    from dew.data.dataset import stream_endless

    chain = mix([_tagged(1, 8).repeat(), _tagged(2, 8).take(3).filter(_not_201)], weights=(1, 1))
    loader = compile(chain.batch(2), read="indexed", loading=READ)
    for index in range(shares):
        reader = loader.train(DataPartition(index, shares))
        said = stream_endless(reader)
        assert len(_ids(reader, 64)) < 64
        assert said is False


def test_a_nested_mix_cut_per_share_by_the_mix_around_it_never_runs_out():
    """A mix nested in one a filter cuts per share is cut per share too, so its part written without a
    repeat repeats at its root: the nested mix never ends, nor the one around it, whose stream says so
    and reads on past every part's records."""
    from dew.data.dataset import stream_endless

    inner = mix([_tagged(1, 4).repeat(), _tagged(2, 2)], weights=(1, 1))
    chain = mix([inner, _tagged(3, 4).repeat()], weights=(1, 1)).filter(lambda row: True)
    reader = compile(chain.batch(1), read="indexed", loading=READ).train(DataPartition())
    said = stream_endless(reader)
    assert len(_ids(reader, 24)) == 24
    assert said


def test_a_mixture_of_a_repeated_corpus_and_one_read_once_counts_what_a_run_reads():
    """A mix of a corpus of 8 repeated and one of 4 written without a repeat, at even weights, ends
    where the one read once does, at 8, and a run reads that many records: the order repeats
    already, so a run adds no repeat of its own."""
    loader = compile(mix([_tagged(1, 8).repeat(), _tagged(2, 4)], weights=(1, 1)).batch(1), loading=READ)
    assert loader.records == 8
    assert sorted(_ids(loader.train(DataPartition()))) == [*range(100, 104), *range(200, 204)]
