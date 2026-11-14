"""Earlier data plans: positions a main saved under plan 14, which described each stage of its order in place,
read by this Dew as its own; and one of 7, which it reads none of."""

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest
from saved_positions_support import named

from dew.data import Memory
from dew.data.compile import compile
from dew.data.dataset import DataPartition, Loading
from dew.data.pipeline import Pipeline, mix


def test_a_position_a_plan_this_dew_reads_none_of_saved_is_refused():
    """Share 0's position of a two-share streamed read, saved by a version 7 main
    (tests/fixtures/version7_positions.json, written at its commit), is refused by this Dew's read of the
    same chain: it reads plans of the version before its own and its own alone."""
    recorded = json.loads((Path(__file__).with_name("fixtures") / "version7_positions.json").read_text())
    chain = Pipeline(Memory({"id": np.arange(1, 25, dtype=np.int32)}, shard_rows=4)).shuffle(seed=3, window=8)
    loader = compile(chain.batch(2), read="streamed", loading=Loading(workers=0))
    stream = loader.train(DataPartition(index=0, count=2))
    try:
        with pytest.raises(ValueError):
            stream.set_state(recorded["positions"]["plain"]["state"].encode())
    finally:
        stream.close()


SAVED_BY_MAIN = Path(__file__).with_name("fixtures") / "positions" / "main-0c8cf2384"
"""Positions main saved at 0c8cf2384 under plan 14 of the chains `saved_positions_support` builds
(`tools/positions_saved_by_main.py`)."""


def _ids(stream, batches: int) -> list[list[int]]:
    return [np.asarray(next(stream)["id"]).tolist() for _ in range(batches)]


def _read(loader, skipped: int, batches: int) -> list[list[int]]:
    """The `batches` a read of `loader` reads past its first `skipped`, uninterrupted."""
    stream = loader.train(DataPartition())
    try:
        _ids(stream, skipped)
        return _ids(stream, batches)
    finally:
        stream.close()


@pytest.mark.parametrize("read", ["streamed", "memory"])
def test_a_position_saved_under_14_resumes_where_it_stopped(read):
    """A position main saved under plan 14, which described its order's stages in place, per share or in
    a global order, resumes in this Dew's read onto the rows a read of the same chain stopped where it stopped
    reads next, a stream's through a take grown since too, which reads on past the old bound; a configured
    step whose configuration changed is refused, its rows not the saved run's."""
    from saved_positions_support import taken

    name = "taken" if read == "streamed" else "taken_in_memory"
    saved = (SAVED_BY_MAIN / f"{name}.position").read_bytes()
    after = _read(taken(read=read), 3, 4)
    for count in (20, 30) if read == "streamed" else (20,):
        resumed = taken(count, read=read).train(DataPartition())
        try:
            resumed.set_state(saved)
            assert _ids(resumed, 4) == after, count
        finally:
            resumed.close()
    changed = taken(factor=3, read=read).train(DataPartition())
    try:
        with pytest.raises(ValueError):
            changed.set_state(saved)
    finally:
        changed.close()


def test_a_position_saved_under_14_resumes_and_saves_one_of_15_that_resumes_again():
    """A seeded mix's position main saved under plan 14 resumes in this Dew's read onto the rows read next,
    and the position it saves in turn, of plan 15, holds each stage of its order once, at the step of its
    manifest that describes it, and resumes again onto the rows after."""
    from saved_positions_support import mixture

    ids = _read(mixture(), 0, 8)
    saved = (SAVED_BY_MAIN / "mixture.position").read_bytes()
    for start in (3, 5):
        resumed = mixture().train(DataPartition())
        try:
            resumed.set_state(saved)
            assert _ids(resumed, 2) == ids[start:start + 2]
            saved = resumed.get_state()
        finally:
            resumed.close()
        plan = json.loads(saved)["dew_data_plan"]
        stages = plan["order"]["stages"]
        assert plan["version"] == 15 and stages and all(isinstance(stage, int) for stage in stages)


SAVED_BEFORE_THE_MOVE = Path(__file__).with_name("fixtures") / "positions" / "main-cdc45f8f9"
"""Positions main saved at cdc45f8f9, before the condition encoders moved to dew.interop (row 5996), of
chains whose steps depend on one (`saved_positions_support.Conditioned`)."""


@pytest.mark.parametrize("name", ["conditioned_phases", "conditioned_mixture"])
def test_a_position_saved_before_an_encoder_moved_resumes_where_it_stopped(name):
    """A position saved while a step's encoder was dew.inputs's resumes onto the rows a read of the same
    chain reads next: a phased read past its first phase, whose finished phase's identity hashed its steps,
    and a mix whose member holding the step is kept by its digest. Each holds the class's earlier path, in the
    mix's manifest or hashed in a digest, which this Dew reads as its path now and hashes as the path it
    had (`dew._records.MOVED_PATHS`)."""
    import saved_positions_support

    loader, steps = saved_positions_support.SAVED[name]
    after = _read(loader(), steps, 3)
    saved = (SAVED_BEFORE_THE_MOVE / f"{name}.position").read_bytes()
    resumed = loader().train(DataPartition())
    try:
        resumed.set_state(saved)
        assert _ids(resumed, 3) == after
    finally:
        resumed.close()


def _saved(chain, **options) -> tuple[bytes, list[list[int]]]:
    """A streamed per-share read of `chain`, saved after two batches, and the two it reads next."""
    stream = compile(chain, read="streamed", loading=Loading(workers=0), **options).train(DataPartition())
    try:
        _ids(stream, 2)
        return stream.get_state(), _ids(stream, 2)
    finally:
        stream.close()


def _edited(state: bytes, edit) -> bytes:
    """`state`, a per-share position, with `edit` applied to its decoded record."""
    decoded = json.loads(state)
    edit(decoded)
    return json.dumps(decoded).encode()


def test_a_position_read_by_another_rule_is_refused_naming_it():
    """A position whose plan read by a rule this Dew reads otherwise (a mix's schedule) is refused, the
    refusal naming the rule (docs/design/data.md, the rules); the position as saved resumes."""
    chain = mix([named("a", 1), named("b", 2)], seed=4).batch(2)
    saved, after = _saved(chain)

    def other_mix(decoded):
        decoded["dew_data_plan"]["rules"]["mix"] = "another-schedule"

    for state, refused in ((_edited(saved, other_mix), True), (saved, False)):
        stream = compile(chain, read="streamed", loading=Loading(workers=0)).train(DataPartition())
        try:
            if refused:
                with pytest.raises(ValueError):
                    stream.set_state(state)
            else:
                stream.set_state(state)
                assert _ids(stream, 2) == after
        finally:
            stream.close()


def test_a_position_another_grain_wrote_resumes_where_its_iterators_read_it():
    """A consumed mixture's position whose iterator state another Grain wrote (its `grain` rule) is restored
    through the iterators that own it, and resumes onto the rows read next where they read it; one they
    cannot read (a state of another shape) is refused when it is restored."""
    chain = mix([named("a", 1), named("b", 2)], seed=4).batch(2)
    saved, after = _saved(chain)

    def other_grain(decoded):
        decoded["dew_data_plan"]["rules"]["grain"] = "0.0.1"

    def unreadable(decoded):
        other_grain(decoded)
        decoded["iterator"] = {"renamed": decoded["iterator"]}

    stream = compile(chain, read="streamed", loading=Loading(workers=0)).train(DataPartition())
    try:
        stream.set_state(_edited(saved, other_grain))
        assert _ids(stream, 2) == after
    finally:
        stream.close()
    stream = compile(chain, read="streamed", loading=Loading(workers=0)).train(DataPartition())
    try:
        with pytest.raises(ValueError):
            stream.set_state(_edited(saved, unreadable))
    finally:
        stream.close()


@dataclasses.dataclass(frozen=True)
class Recut(Memory):
    """A store of the same content and options as `Memory`'s, cut into its shards in reverse: a store a
    later Dew cuts otherwise, read in another order."""

    def manifest(self):
        cut = super().manifest()
        return dataclasses.replace(cut, shards=cut.shards[::-1])


def test_a_stream_its_store_cuts_into_other_shards_is_refused():
    """A streamed position resumes where its store cuts its content into the shards the read that saved it
    read in turn, and is refused where a store of the same content and options cuts it otherwise (its
    `shards` fact)."""
    ids = {"id": np.arange(1, 25, dtype=np.int32)}

    def chain(kind) -> Pipeline:
        return Pipeline(kind(ids, identity="ids", shard_rows=4)).repeat().batch(2)

    saved, after = _saved(chain(Memory))
    for kind, refused in ((Memory, False), (Recut, True)):
        stream = compile(chain(kind), read="streamed", loading=Loading(workers=0)).train(DataPartition())
        try:
            if refused:
                with pytest.raises(ValueError):
                    stream.set_state(saved)
            else:
                stream.set_state(saved)
                assert _ids(stream, 2) == after
        finally:
            stream.close()
