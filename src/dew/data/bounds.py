"""Where a training read's examples are bounded and counted: the place before which a source example is
still one example (`_one_example_until`), the bound a run in examples or epochs stands there
(`_bounded`), and the mark a run counting its passes reads its consumption by (`_bounded`'s `counting`,
`_countable`); `dew.data.compile` places them as it stages a read."""

from __future__ import annotations

import dataclasses

from dew.data.analysis import _name, _steps, _until, rejects
from dew.data.pipeline import (
    BatchStage,
    Configured,
    FlatMap,
    Limit,
    Map,
    MarkConsumed,
    Native,
    Pack,
    Phases,
    Pipeline,
    Repeat,
    Stage,
    TokenBudget,
    Windows,
    children,
    examples_bound,
)


def _native_after_bound(pipeline: Pipeline) -> bool:
    """Whether `pipeline` counts the examples its rows consume (`MarkConsumed`) and runs a native Grain step
    after the bound that marks them."""
    stages = pipeline.stages
    at = examples_bound(stages)
    return at is not None and any(isinstance(stage, Native) for stage in stages[at:])


def _bounded(pipeline: Pipeline, examples: int, *, passes: bool = False, counting: bool = False) -> Pipeline:
    """`pipeline`'s training read bounded to its first `examples` source examples: the records as they
    reach the first of the objective's steps that makes several rows of one (`FlatMap.examples`, which
    run after the chain's own record order, `_objective_branch`), or where none does, the first stage that
    packs or windows them, or its batch: never the rows an objective makes of an example or the windows
    a pack lays them in. A repeat without end goes before the bound
    where none is written, so the records may span passes, and the batch keeps the short one the
    bound ends inside, filled and marked, so each record bounded is read once (`Loader.examples`); in a
    pool, a read in one order takes the bound before it cuts its shares, and a read whose shares, or a
    share's loading workers, each read records of their own takes on each its part of the pool's first
    `examples` (`dew.data.lowering._Build.bound_at`). Refuses a bound it cannot place: a repeat with a
    count before it, or any repeat after it, the batch's included, a mixed corpus that packs on its own,
    and phases, which end by steps. A chain without a batch is left for `compile` to refuse. A bound of
    `passes` (a run's epochs) stands where a pass is counted (`_one_example_until`), before a step that
    rejects a record it cannot make. A `counting` bound marks the examples a run bounded in steps consumes
    and bounds nothing (`_UNCOUNTED`, `compile`'s `count_passes`): its batch stays as written, the read
    endless, no short batch made."""
    if examples < 1:
        raise ValueError(f"examples must be at least 1, got {examples}")
    if isinstance(pipeline.source, Phases):
        raise ValueError("phases end each phase at a step, so their run is bounded in steps, not examples")
    if any(isinstance(stage, (Pack, Windows)) for child in children(pipeline) for stage in _steps(child)):
        raise ValueError(f"{_name(pipeline)} packs inside a corpus it mixes, so its records reach the mix as "
                         "windows; pack after the mix to bound the run in examples")
    batched = _until(pipeline.stages, (BatchStage, TokenBudget))
    if batched == len(pipeline.stages):
        return pipeline
    at = _one_example_until(pipeline, passes=passes)
    repeats = [stage for stage in pipeline.stages[:at] if isinstance(stage, Repeat)]
    if any(stage.epochs is not None for stage in repeats) or any(
            isinstance(stage, Repeat) for stage in pipeline.stages[at:]):
        raise ValueError(f"{_name(pipeline)} repeats with a count, or after the records it packs or batches, "
                         f"which a bound of {examples} examples would cut or repeat; leave the repeat out")
    # Each example the bound reads is consumed by the last row a step makes of it (`MarkConsumed`).
    bound = (*(() if repeats else (Repeat(None),)), Limit(examples), Map(MarkConsumed()))
    batch = (pipeline.stages[batched] if counting
             else dataclasses.replace(pipeline.stages[batched], drop_remainder=False))
    return dataclasses.replace(pipeline, stages=(*pipeline.stages[:at], *bound, *pipeline.stages[at:batched],
                                                 batch, *pipeline.stages[batched + 1:]))


def _countable(pipeline: Pipeline) -> bool:
    """Whether a run bounded in steps can count the examples `pipeline`'s read consumes where a pass is
    counted (`_bounded`'s `counting`): a bound it can place there, so a read whose bound would be refused
    (phases, a corpus that packs inside a mix, a repeat after the place or one with a count before it)
    reads as it would counting none, its passes uncounted."""
    if isinstance(pipeline.source, Phases) or any(
            isinstance(stage, (Pack, Windows)) for child in children(pipeline) for stage in _steps(child)):
        return False
    at = _one_example_until(pipeline, passes=True)
    return not any(isinstance(stage, Repeat) and (index >= at or stage.epochs is not None)
                   for index, stage in enumerate(pipeline.stages))


def _expands(stage: Stage) -> bool:
    """Whether `stage` is an objective's step that makes, or once it binds will make, several rows of a
    source example (`FlatMap.examples`, `Configured.examples`)."""
    return isinstance(stage, (FlatMap, Configured)) and stage.examples


def _one_example_until(pipeline: Pipeline, *, passes: bool = False) -> int:
    """The place in `pipeline`'s stages before which a source example is still one example: its first
    step that packs or windows records or makes several rows of each (`_expands`), else its batch's, else
    its end. A bound in examples stands there (`_bounded`), and a memory read's epochs are bounded ahead of
    it (`_held_ahead`). With `passes`, where a pass of a run's epochs is counted (`_pass_examples`) and
    bounded: also before the first step that rejects a record it cannot make (`rejects`: an image that
    does not decode), so a pass is one walk over the source's records, each read whether or not it decoded,
    its consumption carried on by the next row kept and its rejection counted (`data/rejected/`); a run
    bounded in examples counts those it kept."""
    batched = _until(pipeline.stages, (BatchStage, TokenBudget))
    return next((index for index, stage in enumerate(pipeline.stages[:batched])
                 if isinstance(stage, (Pack, Windows)) or _expands(stage) or (passes and rejects(stage))),
                batched)


def _lays_out(pipeline: Pipeline) -> bool:
    """Whether `pipeline` packs or windows its records itself, before its batch, which its preparation
    may move into its source (`_promoted`), where a bound would come after the windows."""
    return any(isinstance(stage, (Pack, Windows))
               for stage in pipeline.stages[:_until(pipeline.stages, (BatchStage, TokenBudget))])
