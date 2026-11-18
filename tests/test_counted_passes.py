"""A run bounded in steps whose cadence reads its passes (`epoch`, or the default `passes`) counts them
(`dew.data.compile`'s `count_passes`): over a decision table, which a memory read makes rows of with no
record count, it evaluates and checkpoints where a run of its epochs does, resumes counting where it
stopped, and a run whose cadences read no pass compiles as it did."""

import dataclasses
import json
from pathlib import Path

from run_kinds import decision, scalars, trained

from dew import _position
from dew.batch import CONSUMED_COUNT, EXAMPLES_CONSUMED
from dew.data.dataset import DataPartition
from dew.training._fit import consumption_counted
from dew.training.interval import CONSUMED_EXAMPLES, Interval

PLAIN = Path(__file__).with_name("fixtures") / "plans" / "decision-plain-main-b2d94d559.json"
"""A decision run's plan identity and batch fields, bounded in four steps with no cadence that reads its
passes, as main compiled it at b2d94d559, before runs counted their passes."""


def evaluated(root: Path, name: str) -> list[int]:
    """The steps the run `name` under `root` logged a validation at."""
    return sorted(step for step, values in scalars(root / "runs" / name).items()
                  if any(key.startswith("val/") for key in values))


def test_a_step_bounded_run_evaluates_at_each_pass_end_where_a_run_of_its_epochs_does(tmp_path):
    """A decision run bounded in eight steps whose evaluation reads its passes evaluates at the steps a run
    of two epochs over the same data does, each pass's 25 tickets' end (`Loader.pass_examples`), and at
    its own end; it was refused, its `epoch` needing a record count its read has none of."""
    run = decision(tmp_path, drawn=False)
    trained(run, tmp_path, "epochs", trainer={"epochs": 2, "steps": None, "eval_every": "epoch"})
    trained(run, tmp_path, "steps", trainer={"steps": 8, "eval_every": "epoch"})
    passes = evaluated(tmp_path, "epochs")
    assert len(passes) == 2 and evaluated(tmp_path, "steps") == [*passes, 8]


def test_a_resumed_run_counts_its_passes_on_from_where_it_stopped(tmp_path):
    """A counting run stopped inside its second pass, or at its first pass's end, and run on to its eight
    steps evaluates where the run nobody stopped does: its count rides in its checkpoint
    (`CONSUMED_EXAMPLES`) and resumes with it."""
    run = decision(tmp_path, drawn=False)
    trained(run, tmp_path, "whole", trainer={"steps": 8, "eval_every": "epoch"})
    whole = evaluated(tmp_path, "whole")
    for stopped in (whole[0], whole[0] + 1):
        name = f"stopped-{stopped}"
        for steps in (stopped, 8):
            trained(run, tmp_path, name, trainer={"steps": steps, "eval_every": "epoch",
                                                  "checkpoint_final": True})
        assert evaluated(tmp_path, name) == sorted({*whole, stopped})


def test_a_run_whose_cadences_read_no_pass_compiles_as_before_and_a_counting_one_steps_its_batches(tmp_path):
    """A decision run whose cadences read no pass compiles its data as main did before runs counted their
    passes (`PLAIN`): its plan's identity and its batch's fields. One whose evaluation reads them reads
    its examples' count beside them, which its host takes off (`consumption_counted`), so its step reads
    the fields the other's does."""
    held = json.loads(PLAIN.read_text())
    run = decision(tmp_path, drawn=False)

    def first(cadence):
        trainer = dataclasses.replace(run.trainer, steps=4, eval_every=cadence, checkpoint_every=None)
        loader = dataclasses.replace(run, trainer=trainer).prepare().dataset
        stream = loader.train(DataPartition())
        try:
            return loader, next(stream)
        finally:
            stream.close()

    plain, batch = first(None)
    assert plain.pass_examples is None and plain.plan.identity() == held["identity"]
    assert sorted(batch) == held["fields"]
    counting, batch = first("epoch")
    assert counting.pass_examples is not None and EXAMPLES_CONSUMED in batch
    stepped = consumption_counted(batch, clocked=False, counting=True)
    assert sorted(name for name in stepped if name != CONSUMED_COUNT) == held["fields"]


def test_a_checkpoint_saved_before_its_run_counted_reads_its_count_off_a_global_position():
    """A checkpoint with no count of the examples its run consumed (`CONSUMED_EXAMPLES`), saved before its
    run counted its passes, is given its global position's records, the source examples its one order
    read; a per-share position states none, and its run counts from where it resumes."""
    control: dict = {}
    Interval.counted_before(control, _position.encode(_position.Global(records=37, order="o", plan=None)))
    assert control == {CONSUMED_EXAMPLES: 37}
    kept = {CONSUMED_EXAMPLES: 5}
    Interval.counted_before(kept, _position.encode(_position.Global(records=37, order="o", plan=None)))
    assert kept == {CONSUMED_EXAMPLES: 5}
    shared: dict = {}
    Interval.counted_before(shared, json.dumps({_position.PLAN: {}, "iterator": {}}).encode())
    assert shared == {}


def _identity_map(rows):
    return rows


def _chain(read: str, *, shard_rows: int = 3):
    """Ten records, shuffled, read `read`, an objective making `id % 3 + 1` rows of each (`Questioned`),
    in batches of four."""
    import numpy as np

    from dew.data import Memory
    from dew.data.pipeline import Pipeline

    ids = np.arange(1, 11, dtype=np.int32)
    shuffled = {"seed": 3, "window": 10} if read == "streamed" else {"seed": 3}
    return Pipeline(Memory({"id": ids}, shard_rows=shard_rows)).shuffle(**shuffled).read(read).batch(4)


def _loader(chain, *, counting: bool, workers: int = 0, expands: bool = True):
    """`chain` compiled for a run of eight steps, counting its passes or not, its objective making rows of
    each record where it `expands` (`Questioned`)."""
    from data_support import Questioned

    from dew.data.compile import fitted
    from dew.data.dataset import Loading
    from dew.data.pipeline import ObjectiveData

    objective = ObjectiveData(train=(Questioned(),)) if expands else ObjectiveData()
    loader, _ = fitted(chain.loading(Loading(workers=workers)), None, objective=objective, steps=8,
                       count_passes=counting)
    return loader


def _ids(stream, batches: int) -> list[list[int]]:
    import itertools

    import numpy as np

    return [np.asarray(batch["id"]).tolist() for batch in itertools.islice(stream, batches)]


def test_a_streamed_read_its_workers_deal_counts_its_passes_and_reads_as_one_counting_none():
    """A streamed read of shards two loading workers each read their own of, which counts its passes,
    reads the batches the same read counting none reads, its examples counted beside them: its count
    bounds nothing, so no worker is dealt a part of a bound (`examples_limit`), which a counting bound
    past any count made each worker read ahead for without end (the theme's review, F1)."""
    plain = _loader(_chain("streamed"), counting=False, workers=2)
    counting = _loader(_chain("streamed"), counting=True, workers=2)
    assert counting.pass_examples == 10
    streams = [loader.train(DataPartition()) for loader in (plain, counting)]
    try:
        assert _ids(streams[1], 6) == _ids(streams[0], 6)
    finally:
        for stream in streams:
            stream.close()


def test_a_position_saved_counting_or_not_resumes_in_a_read_doing_the_other():
    """A streamed read's position saved by a run counting its passes resumes in one counting none, and the
    other way round, onto the batches the run that saved it reads next: the count adds no stage to its
    read's state, nor to its plan or order (`counting_only`), where a counting bound's limit held state of
    its own the other run's read could not restore (the theme's review, F2)."""
    for saving, resuming in ((False, True), (True, False)):
        saved, after = _loader(_chain("streamed"), counting=saving), _loader(_chain("streamed"),
                                                                              counting=resuming)
        first, second = saved.train(DataPartition()), after.train(DataPartition())
        try:
            _ids(first, 3)
            second.set_state(first.get_state())
            assert _ids(second, 4) == _ids(first, 4)
        finally:
            first.close()
            second.close()


def test_a_read_whose_count_would_change_its_rows_counts_no_pass_and_reads_as_before():
    """A memory read that packs its records and shuffles the windows after, and an indexed one a native Grain
    step reads after its pack, count no pass where their cadence would read one (`_countable`): marking the
    first would hold each example's windows back until all were read, and the second's native step keeps
    no count. Each reads the batches it reads counting none, its passes uncounted, where it reordered its
    rows or was refused (the theme's review, F3 and F4)."""
    import numpy as np

    from dew.data import Memory
    from dew.data.pipeline import Pipeline

    documents = [{"id": np.full(index % 3 + 2, index, np.int32)} for index in range(12)]
    packed = Pipeline(Memory(documents)).shuffle(seed=1).pack(8, fields=("id",)).shuffle(seed=2).read(
        "memory").batch(2)
    native = Pipeline(Memory(documents)).pack(8, fields=("id",)).grain(
        _identity_map, identity="identity@1").read("indexed").batch(2)
    for chain in (packed, native):
        plain, counting = (_loader(chain, counting=asked, expands=False) for asked in (False, True))
        assert counting.pass_examples is None
        streams = [loader.train(DataPartition()) for loader in (plain, counting)]
        try:
            assert _ids(streams[1], 4) == _ids(streams[0], 4)
        finally:
            for stream in streams:
                stream.close()
