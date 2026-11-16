"""How many records a read's order holds, counted as the lowering builds it by index (`dew.data.lowering`): a
pipeline's records through its stages (`held_records`), a global order's epoch (`epoch_records`), the
records its order holds before its batch (`order_records`) and the rows its last batch holds (`tail`),
which a compile reads to bound a run and a mix reads to end where its members end."""

from __future__ import annotations

import dataclasses

import grain.python as pygrain

from dew._records import integer
from dew.data.analysis import _until, ends
from dew.data.dataset import DataPartition, Tail
from dew.data.lowering import _Build, _closing
from dew.data.mixing import Proportions, _weights
from dew.data.pipeline import BatchStage, Mixture, Pipeline, Repeat, TokenBudget


def held_records(pipeline: Pipeline, build: _Build) -> int | None:
    """The records `pipeline` holds through its stages as `build` builds them by index (an epoch); None
    where they come back as a stream."""
    cache = build.lowering.cache
    if cache is not None and cache >= len(pipeline.stages):
        # A cache after the stages counted, as a preset's cache follows the order written before
        # it: a pass reads each input once, and a cache of one record for each input holds that
        # many, where one that drops or expands them counts none ahead of reading them.
        if not build.lowering.one_to_one:
            return None
        build = dataclasses.replace(build, lowering=dataclasses.replace(build.lowering, cache=None))
    with _closing(build.records(pipeline, DataPartition())) as rows:  # a token split's cache sealed
        return len(rows) if isinstance(rows, pygrain.MapDataset) else None


def epoch_records(train: Pipeline, build: _Build) -> int | None:
    """The records in one epoch of a global order: each corpus's indexed order
    up to its first repeat, as the lowering builds it, so a slice, a limit and
    a token split count exactly. A mixture read once counts the records its mix
    reads before it ends, as the lowering builds it. One read on past its end,
    by a repeat that never ends, a training order's own (`_Build.batched`), or a
    mix of parts that never end, reads each part on from where it was, so it
    counts until every part has come round once (`Proportions.covering`), the
    parts it mixes counted alike; a part a slice or a limit bounds comes round
    once its bounded records are read (`covered`). An order that ends counts no
    more than it reads, whatever bounds it after its repeat. A repeat written before a cache counts the
    inputs a pass reads where the cache keeps one record for each, and none
    otherwise, ahead of reading them; phases are counted by none."""
    if not build.global_order:
        return None
    # It counts the records it reads by index in the process: the same in any workers, whose automatic
    # count the read alone resolves (`_Build.tuned`).
    build = dataclasses.replace(build, loading=dataclasses.replace(build.loading, workers=0))

    def passed(pipeline: Pipeline, build: _Build) -> int | None:
        """The records `pipeline` reads up to its first repeat, as the lowering builds them."""
        once = pipeline.stages[:_until(pipeline.stages, (Repeat,))]
        return held_records(dataclasses.replace(pipeline, stages=once), build)


    def viewed(pipeline: Pipeline) -> int | None:
        """Where the first repeat after the last slice or limit of `pipeline` reads the view they
        bound, past its stages where none follows; None where nothing bounds it."""
        stages = pipeline.stages
        bounds = [at for at, stage in enumerate(stages) if ends(stage)]
        if not bounds:
            return None
        return next((at for at, stage in enumerate(stages) if at > bounds[-1] and isinstance(stage, Repeat)),
                    len(stages))

    def covering(pipeline: Pipeline, build: _Build) -> int | None:
        """The records of `pipeline` in which every part it mixes, read on past its end, comes round
        once (`covered`); a corpus's pass."""
        source = pipeline.source
        if not isinstance(source, Mixture):
            return passed(pipeline, build)
        sizes = [covered(child, build.child(index)) for index, child in enumerate(source.pipelines)]
        known = [size for size in sizes if size is not None]
        if len(known) != len(sizes):
            return None
        return Proportions.of(_weights(source), source.seed).covering(known)

    def covered(pipeline: Pipeline, build: _Build) -> int | None:
        """The records of a part a mix reads on past its end in which it comes round once: a view a
        slice or a limit bounds once its bounded records are read, those the first repeat after its
        last bound reads or the bounded order itself where none follows; else its `covering`."""
        at = viewed(pipeline)
        if at is None:
            return covering(pipeline, build)
        return held_records(dataclasses.replace(pipeline, stages=pipeline.stages[:at]), build)

    repeated, batched = _until(train.stages, (Repeat,)), _until(train.stages, (BatchStage, TokenBudget))
    order = dataclasses.replace(train, stages=train.stages[:batched])
    # Whether its records never end, and whether they are read on past an end on the way: by a repeat
    # without a count, or as a mix of parts that never end.
    unending, onward = build.lowering.records_unending, build.lowering.onward
    # A training order that repeats nowhere repeats for ever at its root (`_Build.batched`).
    rooted = build.training and build.lowering.rooted
    if onward or rooted:
        # A view the order's repeat reads is its epoch, the read's own root repeat reading the whole
        # order; a bound after its last written repeat ends the order.
        at = viewed(order)
        whole = at == len(order.stages) and not rooted
        count = (covering(train, build) if at is None or whole
                 else held_records(dataclasses.replace(order, stages=order.stages[:at]), build))
    else:
        count = passed(train, build)
    if count is None or rooted or unending or (not onward and repeated >= batched):
        return count
    # An order that ends holds no more than it reads before its batch, whatever bounds it after its repeat.
    length = _order_length(train, build)
    return count if length is None else min(count, length)


def _order_length(train: Pipeline, build: _Build) -> int | None:
    """The records `train`'s order holds before its batch, all its passes, as the lowering builds
    them; None where its read counts none ahead."""
    order = dataclasses.replace(train, stages=train.stages[:_until(train.stages, (BatchStage, TokenBudget))])
    with _closing(build.records(order, DataPartition())) as rows:
        return len(rows) if isinstance(rows, pygrain.MapDataset) else None


def order_records(train: Pipeline, build: _Build) -> int | None:
    """The records a training read of `train`'s global order hands out where it ends, the same on
    every share (`Loader.order_records`): the rows of every batch its read yields, through the
    stages after its batch, a repeat, a slice or a limit (`_Build.placed`), a kept short batch inside
    the order with its filled rows, which it marks, and the order's last batch with its real rows
    alone; None where it never ends or its read counts none ahead. The batches' sizes are what the
    stages place, so no record is read."""
    stage = next((stage for stage in train.stages if isinstance(stage, BatchStage)), None)
    if not build.global_order or build.lowering.endless or build.lowering.records_unending or stage is None:
        return None
    length = _order_length(train, build)
    if length is None:
        return None
    whole, short = divmod(length, stage.size)
    count = whole + (1 if short and not stage.drop_remainder else 0)
    if not count:
        # No batch to place: the read refuses an order of none, as it reads.
        return 0
    after = tuple(range(train.stages.index(stage) + 1, len(train.stages)))
    sizes = pygrain.MapDataset.range(count).map(lambda index: stage.size if index < whole else short)
    placed = build.placed(sizes, train, DataPartition(), after)
    if not len(placed):
        return 0
    return (len(placed) - 1) * stage.size + integer(placed[len(placed) - 1], "a batch's size")


def tail(train: Pipeline, build: _Build) -> Tail:
    """What a training read of `train` does with the records an epoch leaves short of a batch: reads
    them, where its records never end (`_Lowering.records_unending`) or the read repeats them at its root
    (`_Build.batched`), so they lead the next epoch's batch, or where its batch keeps a short one;
    drops them where its order ends and its batch drops it."""
    drops = next((stage.drop_remainder for stage in train.stages if isinstance(stage, BatchStage)), True)
    rooted = build.training and build.lowering.rooted
    return "read" if build.lowering.records_unending or rooted or not drops else "dropped"
