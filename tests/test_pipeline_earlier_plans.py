"""Earlier data plans: positions a main saved under plan 14, which described each stage of its order in place,
read by this Dew as its own; and one of 7, which it reads none of."""

import dataclasses
import itertools
import json
from pathlib import Path

import numpy as np
import pytest
from saved_positions_support import named

from dew.data import Memory, Phase, Steps
from dew.data.compile import compile
from dew.data.dataset import Checkpointable, Closeable, DataPartition, Loading
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


def test_a_position_saved_under_14_resumes_and_saves_one_of_this_dews_that_resumes_again():
    """A seeded mix's position main saved under plan 14 resumes in this Dew's read onto the rows read next,
    and the position it saves in turn, of this Dew's plan (17), holds each stage of its order once, at the
    step of its manifest that describes it, and resumes again onto the rows after."""
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
        assert plan["version"] == 17 and stages and all(isinstance(stage, int) for stage in stages)


SAVED_BEFORE_THE_MOVE = Path(__file__).with_name("fixtures") / "positions" / "main-cdc45f8f9"
"""Positions main saved at cdc45f8f9, before the condition encoders moved to dew.interop (row 5996), of
chains whose steps depend on one (`conditioned_step.Conditioned`)."""


SAVED_UNDER_15 = Path(__file__).with_name("fixtures") / "positions" / "main-966b2031c"
"""Positions main saved at 966b2031c, under plan 15, which described a step without the fields its repr
hides, of the chains `hidden_field_positions` builds: a read and phases through such a step
(`Remapped`), and a global order and phases of global orders batched by such a function (`Stacked`)."""

SAVED_UNDER_16 = Path(__file__).with_name("fixtures") / "positions" / "main-47a76d52d"
"""A position main saved at 47a76d52d, under plan 16, before a phases' position's version was read from its
phase's reader: phases through a step whose table its repr hides (`hidden_field_positions`)."""

SAVED_BY_GRAINS_PACKER = Path(__file__).with_name("fixtures") / "positions" / "main-45131ca01"
"""Positions main saved at 45131ca01, while Grain's concatenating packer laid a concatenating pack's windows
(row 9059): its own position, the remainder it held a slice of its document, a window emitted after it, its
cuts overlapped or not (`concatenated_positions`)."""


@pytest.mark.parametrize(("saved_at", "module", "name"), [
    (SAVED_BEFORE_THE_MOVE, "saved_positions_support", "conditioned_phases"),
    (SAVED_BEFORE_THE_MOVE, "saved_positions_support", "conditioned_mixture"),
    (SAVED_UNDER_15, "hidden_field_positions", "hidden"),
    (SAVED_UNDER_15, "hidden_field_positions", "hidden_phases"),
    (SAVED_UNDER_15, "hidden_field_positions", "hidden_global"),
    (SAVED_UNDER_15, "hidden_field_positions", "hidden_global_phases"),
    (SAVED_UNDER_16, "hidden_field_positions", "hidden_phases"),
    (SAVED_BY_GRAINS_PACKER, "concatenated_positions", "concatenated"),
    (SAVED_BY_GRAINS_PACKER, "concatenated_positions", "concatenated_overlapped"),
    (SAVED_BY_GRAINS_PACKER, "concatenated_positions", "concatenated_overlapped_later")])
def test_a_position_an_earlier_dew_saved_resumes_where_it_stopped(saved_at, module, name):
    """A position an earlier Dew saved resumes onto the rows a read of the same chain stopped where it
    stopped reads next. Saved while a step's encoder was dew.inputs's: a phased read past its first phase,
    whose finished phase's identity hashed its steps, and a mix whose member holding the step is kept by its
    digest, each holding the class's earlier path, which this Dew reads as its path now and hashes as the
    path it had (`dew._records.MOVED_PATHS`). Saved under plan 15, which described a step without the fields
    its repr hides: a read and phases through such a step, and a global order and phases of global orders
    batched by such a function, read by the plan, and compared with the order, as 15 described its steps
    (`Plan.under`, `Plan.order_under`), where the record of a run saved since plan 15 hit PlanChanged on its
    steps (train 1013's review, F3) and a global order's position on its order (train 1016's review, F1;
    phases of them, train 1018's, F2). Saved under plan 16 with no version of its own, phases through such a
    step, read as its phase's reader's plan records (train 1018's review, F3). Saved while Grain's
    concatenating packer laid a pack's windows, its position Grain's, cut through a document or not, its
    cuts overlapped or not: Dew's packer reads it (row 9059)."""
    import importlib

    from resumed_support import resumed_as_read

    loader, steps = importlib.import_module(module).SAVED[name]
    stopped, restored = resumed_as_read(loader, steps, (saved_at / f"{name}.position").read_bytes(),
                                        lambda batch: np.asarray(batch["id"]).tolist(), 3)
    assert restored == stopped and stopped


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
class Unopened(Memory):
    """A store whose records refuse to open once `refused` holds the error (a store whose manifest a later
    write left unreadable), an error of its own whatever position is restored."""

    refused: list[ValueError] = dataclasses.field(default_factory=list, compare=False)

    def records(self, read):
        if self.refused:
            raise self.refused[0]
        return super().records(read)


def test_a_position_another_grain_wrote_whose_read_fails_to_open_raises_that_failure():
    """A position another Grain wrote is refused as that Grain's only where its iterators refuse its
    state: a read that fails to open (its store's own error) raises that error, never a version's refusal
    naming Grains that read alike (row 12178: every open's failure was refused as the other Grain's)."""
    from dew.data.plan import PlanChanged

    refused: list[ValueError] = []
    chain = Pipeline(Unopened({"id": np.arange(1, 25, dtype=np.int32)}, identity="ids", refused=refused)
                     ).repeat().batch(2)
    saved, _ = _saved(chain)

    def other_grain(decoded):
        decoded["dew_data_plan"]["rules"]["grain"] = "0.0.1"

    stream = compile(chain, read="streamed", loading=Loading(workers=0)).train(DataPartition())
    assert isinstance(stream, Checkpointable) and isinstance(stream, Closeable)
    unreadable = ValueError("the store's manifest is unreadable")
    refused.append(unreadable)
    try:
        with pytest.raises(ValueError) as raised:
            stream.set_state(_edited(saved, other_grain))
        assert raised.value is unreadable and not isinstance(raised.value, PlanChanged), raised.value
    finally:
        stream.close()


@pytest.mark.parametrize("lost", ["rows", "place"])
def test_a_position_whose_own_cursor_holds_no_rows_is_refused_as_corrupt_never_restarted(lost):
    """A saved position another Grain wrote whose own sequential cursor, Dew's, holds no `rows` or no
    `place` is refused as corrupt (`MalformedPosition`), never taken for an earlier Dew's interleave nor as
    the other Grain's (`PlanChanged`), which a relaxed resume would answer by reading the data afresh
    (review of row 12178: every error the restore raised was taken as a version's, so a corrupt cursor
    restarted a run's data; review of d21: a cursor without its place was read as an interleave's)."""
    from dew import _position
    from dew.data.plan import PlanChanged

    chain = Pipeline(Memory({"id": np.arange(1, 25, dtype=np.int32)}, identity="ids")).repeat().batch(2)
    saved, _ = _saved(chain)
    cut = []

    def corrupt(node):
        if isinstance(node, dict):
            if {"place", "rows"} <= node.keys():
                del node[lost]
                cut.append(node)
            for value in list(node.values()):
                corrupt(value)
        elif isinstance(node, list):
            for value in node:
                corrupt(value)

    def other_grain_corrupt(decoded):
        decoded["dew_data_plan"]["rules"]["grain"] = "0.0.1"
        corrupt(decoded)

    edited = _edited(saved, other_grain_corrupt)
    assert cut, "the saved position holds a sequential cursor"
    stream = compile(chain, read="streamed", loading=Loading(workers=0)).train(DataPartition())
    assert isinstance(stream, Checkpointable) and isinstance(stream, Closeable)
    try:
        with pytest.raises(ValueError) as raised:
            stream.set_state(edited)
        assert not isinstance(raised.value, PlanChanged), raised.value
        assert isinstance(raised.value, _position.MalformedPosition), raised.value
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


SAVED_BEFORE_REACHED = Path(__file__).with_name("fixtures") / "positions" / "main-2880fce66"
"""A position main saved at 2880fce66 of `saved_positions_support.taken`, whose configured step (`Scaled`)
is a class of the user's: its plan holds the class's text under `held`, before Dew held the code a step's
own code reaches (`reached`)."""


def test_a_position_saved_before_reached_code_is_refused_exact_and_read_on_relaxed():
    """A position main saved holding a step's class text under `held` holds nothing that proves the code that
    step reaches unchanged, so its exact resume is refused, naming the step, and a relaxed one reads on from
    where it stopped onto the rows a read of the same chain reads next. It read as 'the code of data step
    changed' for unchanged code (train 909's F8)."""
    import saved_positions_support

    from dew._position import PlanChanged

    loader, steps = saved_positions_support.SAVED["taken"]
    saved = (SAVED_BEFORE_REACHED / "taken.position").read_bytes()
    # The same chain, compiled to resume exactly (`compile` resumes relaxed by default); the support module,
    # whose file a step's identity holds, unchanged.
    rows = Memory({"id": np.arange(1, 41, dtype=np.int32)}, shard_rows=4)
    chain = Pipeline(rows).shuffle(seed=3, window=8).map(saved_positions_support.Scaled(2)).repeat()
    exact = compile(chain.take(20).batch(2), read="streamed", loading=saved_positions_support.READ,
                    resume="exact").train(DataPartition())
    try:
        with pytest.raises(PlanChanged) as refused:
            exact.set_state(saved)
    finally:
        exact.close()
    assert [part.split(".")[0] for part in refused.value.differs] == ["code"]
    relaxed = loader().train(DataPartition())
    try:
        relaxed.set_state(saved)
        assert _ids(relaxed, 2) == _read(loader(), steps, 2)
    finally:
        relaxed.close()


def test_a_phases_position_saved_now_refuses_a_finished_phase_whose_hidden_field_changed():
    """A phases' position this Dew saves is read as the plan version its phase's reader records (16), so a
    finished phase whose step's table its repr hides changed since is refused, as every field that decides
    equality states the step (train 1015's review, F6)."""
    from hidden_field_positions import Remapped
    from saved_positions_support import READ, source

    from dew.data import phases

    def stepped(scale: int):
        table = np.arange(8, dtype=np.int32) * scale
        return compile(phases(Phase(source(1).map(Remapped(table)), Steps(3)), Phase(source(2))),
                       read="indexed",
                       loading=READ).train(DataPartition())

    stream = stepped(1)
    try:
        list(itertools.islice(stream, 5))
        saved = stream.get_state()
    finally:
        stream.close()
    same, changed = stepped(1), stepped(3)
    try:
        same.set_state(saved)
        with pytest.raises(ValueError):
            changed.set_state(saved)
    finally:
        same.close()
        changed.close()


def test_a_phased_position_whose_finished_phase_is_damaged_is_refused_not_restarted():
    """A phased position whose finished phase's saved plan is damaged (its manifest no record) is refused as
    damage, a ValueError no resume may read past, not as a plan that changed (`PlanChanged`), which a
    relaxed resume restarts the data on (train 1021's review, F2)."""
    from hidden_field_positions import SAVED

    from dew import _position

    saved = json.loads((SAVED_UNDER_15 / "hidden_global_phases.position").read_bytes())
    plans = saved[_position.ENVELOPE]["plan"]["phases"]
    plans[0] = {**plans[0], "manifest": 5}
    loader, _ = SAVED["hidden_global_phases"]
    stream = loader().train(DataPartition())
    try:
        with pytest.raises(ValueError) as refused:
            stream.set_state(json.dumps(saved).encode())
    finally:
        stream.close()
    assert not isinstance(refused.value, _position.PlanChanged)
