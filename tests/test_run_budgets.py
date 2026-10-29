"""A run's length as a count of examples, a pause partway through its budget, and a run whose data
runs out before its length."""

import collections
import dataclasses

import grain.python as grain
import jax.numpy as jnp
import numpy as np
import pytest
from recording import RecordingTracker
from reference_error import as_given, assert_as_exact

from dew.batch import EXAMPLES_CONSUMED, VALID_ROWS
from dew.config import RunConfig, TrainerConfig
from dew.data import Memory
from dew.data.compile import compile, fitted
from dew.data.dataset import Closeable, DataPartition, Loader, Loading
from dew.data.pipeline import Pipeline
from dew.objectives import Aux, Objective
from dew.optim import Cosine, Linear, OneCycle, OptimConfig
from dew.training import Best, Checkpoints, Pause, Trainer
from dew.training.interval import CONSUMED_COMMITTED, CONSUMED_EXAMPLES

FEATURES = 4
RECORDS = 10
BATCH = 8
EXAMPLES = 41
"""Four passes over the records and one more record, which fill five batches and one of a single record:
three updates of two steps."""
ACCUMULATION = 2
PEAK = 0.2

EXAMPLES_SEEDS = 64
"""Check 1's declared seeds for an examples run against the reference's Adam, sized by `seeds_for`."""
PAUSE_SEEDS = 8
"""Check 1's declared seeds for a paused run against the whole one, sized by `seeds_for`."""


class Pulled(Objective):
    """A weight pulled toward each batch's real rows: the mean over them and their coordinates of
    (w - x)^2, whose gradient over an update's window is 2 (w - the mean of its rows) / FEATURES."""

    def fresh_variables(self, key, held):
        return {"params": {"w": jnp.zeros(FEATURES)}}

    def loss(self, variables, batch, step):
        return self.row_mean(jnp.square(variables["params"]["w"] - batch["x"]), batch), Aux({})


def records(seed: int) -> tuple[np.ndarray]:
    """The corpus of one draw: RECORDS rows of FEATURES coordinates from a unit normal."""
    return (np.random.default_rng(seed).normal(size=(RECORDS, FEATURES)).astype(np.float32),)


def chain(x: np.ndarray) -> Pipeline:
    return Pipeline(Memory({"x": x})).batch(BATCH)


@as_given
def descended(x):
    """The losses of each step and the weight after them, of Adam on `Pulled` over the first EXAMPLES
    records of `x` read in order, pass after pass, in batches of BATCH (the last holding the one
    record left) and updates of ACCUMULATION batches, at a rate falling linearly from PEAK to 0 over
    the updates, in `x`'s own dtype: the reference arm, and in float64 the truth. A step reports its
    batch's loss, and the step that closes an update the loss of the whole update's rows."""
    rows = x[np.arange(EXAMPLES) % RECORDS]
    batches = [rows[start:start + BATCH] for start in range(0, EXAMPLES, BATCH)]
    updates = len(batches) // ACCUMULATION
    w = first = second = jnp.zeros(FEATURES, x.dtype)
    losses = []
    for update in range(updates):
        window = batches[update * ACCUMULATION:(update + 1) * ACCUMULATION]
        held = jnp.concatenate(window)
        losses += [jnp.mean(jnp.square(w - batch)) for batch in window[:-1]]
        losses.append(jnp.mean(jnp.square(w - held)))
        gradient = 2 * jnp.sum(w - held, axis=0) / (len(held) * FEATURES)
        first, second = 0.9 * first + 0.1 * gradient, 0.999 * second + 0.001 * jnp.square(gradient)
        moved = (first / (1 - 0.9 ** (update + 1))) / (jnp.sqrt(second / (1 - 0.999 ** (update + 1))) + 1e-8)
        w = w - PEAK * (1 - update / updates) * moved
    return {"w": w, "loss": jnp.stack(losses)}


def trainer(directory=None, tracker=None) -> Trainer:
    return Trainer(Pulled(), OptimConfig(optimizer="adam", learning_rate=Linear(peak_value=PEAK)), key=0,
                   accumulation=ACCUMULATION, tracker=tracker,
                   checkpoints=None if directory is None else Checkpoints(str(directory)))


def losses(tracker: RecordingTracker) -> list[float]:
    return [scalars["train/loss"] for _, scalars in tracker.scalars if "train/loss" in scalars]


@as_given
def trained(x, directory=None, stops: tuple[Pause | None, ...] = (None,)):
    """The losses of every step and the final weight of an examples run over `x`, fitted once under
    each of `stops` in turn, each fit resuming the one before from `directory`."""
    stepped, state = [], None
    for stop in stops:
        tracker = RecordingTracker()
        state = trainer(directory, tracker).fit(chain(x), examples=EXAMPLES, log_every=1, stop=stop)
        stepped += losses(tracker)
    assert state is not None
    return {"w": np.asarray(state.variables["params"]["w"]), "loss": np.asarray(stepped)}


def ids_read(loader: Loader, field: str) -> collections.Counter:
    """The real values of `field` in every training batch of `loader`, whose read ends at its bound."""
    stream = loader.train(DataPartition())
    read: collections.Counter = collections.Counter()
    try:
        for batch in stream:
            real = np.asarray(batch.get(VALID_ROWS, np.ones(len(batch[field]), bool)), bool)
            values = np.asarray(batch[field])[real]
            read.update(values[values > 0].tolist())
    finally:
        if isinstance(stream, Closeable):
            stream.close()
    return read


@pytest.mark.parametrize("examples", [7, 23])
def test_an_examples_budget_reads_each_record_once_and_its_read_ends_there(examples):
    """A read bounded to its first `examples` records, shuffled pass after pass, hands each out once
    for each pass it reaches, within its steps, and ends: a count under one pass reads each record of
    it once, and one over two reads every record two or three times, the third time `examples` mod
    RECORDS of them."""
    ids = np.arange(1, RECORDS + 1, dtype=np.int32)
    loader = compile(Pipeline(Memory({"id": ids})).shuffle(3).batch(BATCH), examples=examples)
    read = ids_read(loader, "id")
    passes, rest = divmod(examples, RECORDS)
    assert sum(read.values()) == examples
    assert collections.Counter(read.values()) == collections.Counter(
        {passes + 1: rest, passes: RECORDS - rest} if passes else {1: rest})


def test_an_examples_budget_counts_documents_before_they_are_packed():
    """Documents packed into windows as they arrive are counted before the pack: the read bounded to 13
    of 5 documents of 1 to 5 tokens holds each document's tokens once for each time the bound reaches
    it, in the windows of the steps one pass over it counts, and ends."""
    documents = [{"text": np.full(length, length, np.int32)} for length in range(1, 6)]
    loader = compile(Pipeline(Memory(documents)).pack(8).batch(2), examples=13)
    read = ids_read(loader, "text")
    reached = collections.Counter(1 + index % 5 for index in range(13))
    assert read == {token: count * token for token, count in reached.items()}


class Questioned:
    """An objective's expansion of each record into `id % 3 + 1` rows, one a question, as a decision
    layout of a question a row makes of an example (`dew.decision.rows.QuestionRows`)."""

    def flat_map(self, element):
        return tuple({"id": element["id"], "question": np.int32(question)}
                     for question in range(int(element["id"]) % 3 + 1))


@pytest.mark.parametrize("configured", [False, True], ids=["step", "configured"])
@pytest.mark.parametrize("read", ["memory", "streamed"])
@pytest.mark.parametrize("examples", [7, 23])
def test_an_examples_budget_counts_examples_before_the_objective_expands_them(examples, read, configured):
    """An objective that makes each record several rows (a question a row) is bounded by its examples,
    not its rows: the read bounded to `examples` records holds every row of each example the bound
    reaches, as many times as it reaches it, and ends, whatever number of rows each example makes, in a
    memory read and a streamed one alike, the expansion given as a step or as a configured one that
    binds to it from the source's facts (`Configured`)."""
    from dew.data.pipeline import Configured, ObjectiveData

    ids = np.arange(1, RECORDS + 1, dtype=np.int32)
    shuffled = {"seed": 3} if read == "memory" else {"seed": 3, "window": RECORDS}
    expansion = Configured(lambda facts: Questioned(), needs=(), expands=True) if configured else Questioned()
    loader = compile(Pipeline(Memory({"id": ids})).shuffle(**shuffled).batch(BATCH), read=read,
                     objective=ObjectiveData(train=(expansion,)), examples=examples)
    read = ids_read(loader, "id")
    passes, rest = divmod(examples, RECORDS)
    assert sum(count / (record % 3 + 1) for record, count in read.items()) == examples
    assert all(count % (record % 3 + 1) == 0 for record, count in read.items())
    assert sorted(count // (record % 3 + 1) for record, count in read.items()) == sorted(
        [passes + 1] * rest + [passes] * (RECORDS - rest) if passes else [1] * rest)


class Fanned:
    """An objective's expansion of each record `id` into `id % 4` rows, none for some, each row's two
    tokens naming it (`id * 10 + question`)."""

    def flat_map(self, element):
        record = int(element["id"])
        return tuple({"text": np.full(2, record * 10 + question, np.int32)} for question in range(record % 4))


class LastDropped(grain.MapTransform):
    """An objective's filter of the rows after an expansion: the last row of every fifth record."""

    rejects = True

    def map(self, element):
        from dew.data.pipeline import Rejected

        record, question = divmod(int(element["text"][0]), 10)
        return Rejected("dropped") if record % 5 == 0 and question == record % 4 - 1 else element


def consumed_by_row(examples: int) -> dict[int, int]:
    """The oracle of where each of the first `examples` records read in order is consumed: by the row
    (`id * 10 + question`) its expansion makes last, or where it makes none, or its last row is dropped,
    by the next row kept after it."""
    owed, pending = collections.Counter(), 0
    for record in range(1, examples + 1):
        rows = record % 4
        if not rows:
            pending += 1
        for question in range(rows):
            last = question == rows - 1
            if record % 5 == 0 and last:
                pending += 1
                continue
            owed[record * 10 + question] += pending + int(last)
            pending = 0
    assert not pending, "the oracle's last record is consumed by a row of its own"
    return owed


@pytest.mark.parametrize("packed, workers", [(False, 0), (True, 0), (False, 1)],
                         ids=["rows", "packed", "workers"])
def test_each_batch_consumes_the_examples_whose_last_rows_it_holds(packed, workers):
    """A run bounded to 23 examples, read in order as a stream, whose objective makes 0 to 3 rows of each
    and drops the last row of every fifth: each batch's consumed examples (`EXAMPLES_CONSUMED`) are those
    whose last row it holds, an example of no rows, or of a dropped last row, consumed by the next row
    kept, rows packed into windows, batched as they are, or read by loading workers in groups that hold
    their dropped rows apart; the batches together consume all 23, and the column a projection of the
    batch would leave out is kept."""
    from dew.batch import EXAMPLES_CONSUMED
    from dew.data.pipeline import ObjectiveData, Pack

    owed = consumed_by_row(23)
    chain = Pipeline(Memory({"id": np.arange(1, 24, dtype=np.int32)}))
    # The objective packs the rows it makes into windows of up to four rows, as a decision objective does.
    pack = Pack(9, fields=("text",)) if packed else None
    objective = ObjectiveData(train=(Fanned(), LastDropped()), pack=pack)
    # Loading workers run the objective's steps, so their groups hold its dropped rows' markers.
    loader = compile(chain.batch(BATCH).select_columns("text"), read="streamed", objective=objective,
                     examples=23, loading=Loading(workers=workers))
    assert loader.example_steps(23) is None
    stream = loader.train(DataPartition())
    total = 0
    try:
        for batch in stream:
            # Each row of the batch once: a short last batch is filled out with repeats of its rows.
            held = np.asarray(batch["text"])
            expected = sum(owed[row] for row in set(held[held > 0].tolist()))
            assert int(np.sum(batch[EXAMPLES_CONSUMED])) == expected
            total += expected
    finally:
        if isinstance(stream, Closeable):
            stream.close()
    assert total == 23


class Evens(grain.MapTransform):
    """An objective's filter of the odd records, which it states it drops (`rejects`)."""

    rejects = True

    def map(self, element):
        from dew.data.pipeline import Rejected

        return element if int(element["id"]) % 2 == 0 else Rejected("odd")


@pytest.mark.parametrize("read", ["memory", "streamed"])
def test_an_examples_budget_counts_the_examples_an_objectives_filter_keeps_given_or_configured(read):
    """A run bounded to `examples` reads that many examples an objective's filter keeps, the filter
    given as a step or as a configured one that binds to it alike: its kept examples, each even, the
    bound reaching past the five a pass keeps."""
    from dew.data.pipeline import Configured, ObjectiveData

    ids = np.arange(1, RECORDS + 1, dtype=np.int32)
    shuffled = {"seed": 3} if read == "memory" else {"seed": 3, "window": RECORDS}

    def kept(step) -> collections.Counter:
        loader = compile(Pipeline(Memory({"id": ids})).shuffle(**shuffled).batch(BATCH), read=read,
                         objective=ObjectiveData(train=(step,)), examples=7)
        return ids_read(loader, "id")

    given = kept(Evens())
    assert sum(given.values()) == 7 and all(record % 2 == 0 for record in given)
    assert kept(Configured(lambda facts: Evens(), needs=())) == given


def test_an_examples_budget_over_a_mixture_counts_each_corpus_examples_whatever_its_rows():
    """Two corpora mixed, whose examples the objective makes one row and three rows each: the bound
    counts examples before the objective makes rows of them, so the rows it reads are exactly as many
    examples' rows as it is bounded to, each example's rows whole."""
    from dew.data import mix
    from dew.data.pipeline import ObjectiveData

    one = Pipeline(Memory({"id": np.full(RECORDS, 3, np.int32)})).repeat()
    three = Pipeline(Memory({"id": np.full(RECORDS, 2, np.int32)})).repeat()
    loader = compile(mix([one, three], weights=[0.5, 0.5]).batch(BATCH),
                     objective=ObjectiveData(train=(Questioned(),)), examples=EXAMPLES)
    read = ids_read(loader, "id")
    assert read[2] % 3 == 0
    assert read[3] + read[2] // 3 == EXAMPLES


@pytest.mark.parametrize("shuffled", [True, False], ids=["shuffled", "in-order"])
def test_a_decision_runs_examples_budget_counts_its_source_examples_whatever_rows_each_makes(shuffled):
    """A decision objective lays an example out as a row a question: an examples budget over examples
    of two questions each reads exactly that many source examples, two rows each, its windows holding
    twice as many rows as the budget's examples, in the memory read decision rows take, whose objective
    steps start eagerly over the store: shuffled, and in order, where the bound stands before every
    objective step a memory read would run over the store first."""
    from test_decision_objective import SPECIALS, TOKENIZER, tiny_backbone

    from dew.decision import DecisionObjective, Example, Noul, StateFirstLayout, decisions

    layout = StateFirstLayout(max_len=128)
    objective = DecisionObjective(tiny_backbone().clone(max_seq_len=128), tokenizer=TOKENIZER,
                                  specials=SPECIALS, layout=layout)
    asked = {"first": Noul("Is it on?"), "second": Noul("Is it off?")}
    answers = {"first": "true", "second": "false"}
    records = [{"example": Example(f"state {index}", asked, answers)} for index in range(RECORDS)]
    chain = Pipeline(Memory(records)).pipe(decisions)
    # In order, no stage stands between the chain's records and the objective's eager steps.
    chain = chain.shuffle(3) if shuffled else chain
    loader = compile(chain.batch(BATCH), read="memory", objective=objective.data(), examples=EXAMPLES)
    rows = consumed = 0
    for batch in loader.train(DataPartition()):
        # A short last batch is filled with windows it marks as no rows of its own (`VALID_ROWS`).
        real = np.asarray(batch.get(VALID_ROWS, np.ones(len(batch["segments"]), bool)), bool)
        rows += sum(len(set(window[window > 0].tolist())) for window in np.asarray(batch["segments"])[real])
        consumed += int(np.sum(batch.get(EXAMPLES_CONSUMED, 0)))
    assert rows == 2 * EXAMPLES
    # A read no plan counts ahead keeps its progress in the examples its batches consume, each once.
    assert loader.example_steps(EXAMPLES) is not None or consumed == EXAMPLES


def test_an_examples_run_trains_each_record_once_under_the_schedule_its_plan_sizes():
    """An examples run takes the steps its data plan counts, sizes its schedule's horizon from them and
    trains on each record as many times as the bound reaches it: its losses and weight match SGD on
    the same records at a rate falling to 0 over those updates (check 1)."""
    assert_as_exact(trained, descended, records, "examples run", seeds=EXAMPLES_SEEDS)


@dataclasses.dataclass(frozen=True)
class Repeated:
    """An objective's expansion of each record into `id % 3 + 1` copies of its row, read after the bound;
    a frozen record, so a resumed run knows it for the step it saved."""

    def flat_map(self, element):
        return tuple({"x": element["x"]} for _ in range(int(element["id"]) % 3 + 1))


class Echoed(Pulled):
    """`Pulled` over the rows its data step makes of each record (`Repeated`)."""

    def data(self):
        from dew.data.pipeline import ObjectiveData

        return ObjectiveData(train=(Repeated(),))


CLOCKED_RECORDS = 18
"""A clocked run's records: their 36 rows fill five batches, two updates of two and a last window of one."""
CLOCKED_END = PEAK / 4
"""Where a clocked run's rate ends, so its last update moves the weight too."""


def clocked_descent(accumulation: int):
    """The reference arm of a clocked run updating every `accumulation` batches (`clocked_run`): the losses
    of each step and the final weight of Adam on `Echoed` over `x`'s CLOCKED_RECORDS records read in
    order, each `id % 3 + 1` rows, in batches of BATCH rows (the last short), at a rate falling linearly from
    PEAK to CLOCKED_END over the run's progress, the share of its records whose last row an earlier batch held
    (`TrainState.consumed`); a window the stream ends inside commits over its own rows at the stream's
    last batch. In float64, the truth."""
    @as_given
    def descended(x):
        ids = np.arange(1, CLOCKED_RECORDS + 1)
        owner = np.repeat(np.arange(CLOCKED_RECORDS), ids % 3 + 1)
        rows, last = x[owner], np.r_[owner[1:] != owner[:-1], True]
        batches = [(rows[start:start + BATCH], int(np.sum(last[start:start + BATCH])))
                   for start in range(0, len(rows), BATCH)]
        w = first = second = jnp.zeros(FEATURES, x.dtype)
        losses, consumed = [], 0
        for update, at in enumerate(range(0, len(batches), accumulation)):
            window = batches[at:at + accumulation]
            losses += [jnp.mean(jnp.square(w - held)) for held, _ in window[:-1]]
            # The window commits at its last step, a whole update's or the stream's last batch, the progress
            # it reads the examples its batches before that step consumed.
            consumed += sum(count for _, count in window[:-1])
            held = jnp.concatenate([rows for rows, _ in window])
            losses.append(jnp.mean(jnp.square(w - held)))
            gradient = 2 * jnp.sum(w - held, axis=0) / (len(held) * FEATURES)
            first, second = 0.9 * first + 0.1 * gradient, 0.999 * second + 0.001 * jnp.square(gradient)
            corrected = (first / (1 - 0.9 ** (update + 1)), second / (1 - 0.999 ** (update + 1)))
            moved = corrected[0] / (jnp.sqrt(corrected[1]) + 1e-8)
            w = w - (PEAK + (CLOCKED_END - PEAK) * consumed / CLOCKED_RECORDS) * moved
            consumed += window[-1][1]
        return {"w": w, "loss": jnp.stack(losses)}
    return descended


def clocked_fit(x, accumulation: int, *, directory=None, stop: Pause | None = None, log_every=1,
                learning_rate=None):
    """A run bounded to CLOCKED_RECORDS examples of `x`, read in order as a stream, whose objective makes
    `id % 3 + 1` rows of each, under `Linear` from PEAK to CLOCKED_END, updating every `accumulation`
    batches of BATCH rows, checkpointed to `directory` where given: its final state and its tracker."""
    tracker = RecordingTracker()
    ids = np.arange(1, CLOCKED_RECORDS + 1, dtype=np.int32)
    chain = Pipeline(Memory({"x": x, "id": ids})).read("streamed")
    optim = OptimConfig(optimizer="adam", learning_rate=learning_rate or Linear(peak_value=PEAK,
                                                                                end_value=CLOCKED_END))
    trainer = Trainer(Echoed(), optim, key=0, tracker=tracker, accumulation=accumulation,
                      checkpoints=None if directory is None else Checkpoints(str(directory)))
    return trainer.fit(chain.batch(BATCH), examples=CLOCKED_RECORDS, log_every=log_every, stop=stop), tracker


def clocked_run(accumulation: int, directory=None, stops: tuple[Pause | None, ...] = (None,)):
    """A clocked run (`clocked_fit`) fitted once under each of `stops` in turn, each fit resuming the one
    before from `directory`: no pass counts its steps ahead, so its schedule reads the examples its
    batches consumed. Its losses of every step and its final weight."""
    @as_given
    def trained(x):
        stepped, state = [], None
        for stop in stops:
            state, tracker = clocked_fit(x, accumulation, directory=directory, stop=stop)
            stepped += losses(tracker)
        assert state is not None and int(state.consumed) == CLOCKED_RECORDS
        return {"w": np.asarray(state.variables["params"]["w"]), "loss": np.asarray(stepped)}
    return trained


class Thinned:
    """An objective's expansion of each record into `id % 3` copies of its row, none of every third, read
    after the bound."""

    def flat_map(self, element):
        return tuple({"x": element["x"]} for _ in range(int(element["id"]) % 3))


class Thin(Pulled):
    """`Pulled` over the rows its data step makes of each record (`Thinned`)."""

    def data(self):
        from dew.data.pipeline import ObjectiveData

        return ObjectiveData(train=(Thinned(),))


def clocked_chain(x) -> Pipeline:
    """CLOCKED_RECORDS records of `x` beside their ids, read in order as a stream, in batches of BATCH
    rows."""
    ids = np.arange(1, CLOCKED_RECORDS + 1, dtype=np.int32)
    return Pipeline(Memory({"x": x, "id": ids})).read("streamed").batch(BATCH)


@pytest.mark.parametrize("accumulation", [1, 2])
def test_a_clocked_run_whose_last_examples_make_no_rows_ends_at_its_bound(accumulation):
    """The examples a bound reads after the last row it keeps are consumed at the stream's end, by the run's
    last batch: every third record makes no rows, the bound's last among them, and the run ends having
    consumed its whole bound, as its schedule's end reads it, whether its last batch ends a whole update or
    a window of its own."""
    (x,) = clocked_records(0)
    optim = OptimConfig(optimizer="adam", learning_rate=Linear(peak_value=PEAK, end_value=CLOCKED_END))
    state = Trainer(Thin(), optim, key=0, accumulation=accumulation).fit(
        clocked_chain(x), examples=CLOCKED_RECORDS, log_every=1)
    assert CLOCKED_RECORDS % 3 == 0 and int(state.consumed) == CLOCKED_RECORDS


def thin_fit(x, accumulation: int = 1, **fit):
    """A run of `Thin` over CLOCKED_RECORDS records of `x` (`clocked_chain`) under `Linear`, fitted as
    `fit` says: its final state."""
    optim = OptimConfig(optimizer="adam", learning_rate=Linear(peak_value=PEAK, end_value=CLOCKED_END))
    chain = fit.pop("chain", clocked_chain(x))
    return Trainer(Thin(), optim, key=0, accumulation=accumulation,
                   checkpoints=fit.pop("checkpoints", None)).fit(chain, log_every=1, **fit)


def test_a_clocked_run_a_take_ends_first_consumes_only_what_it_dispatched():
    """A take after the batch ends a clocked run before its bound: the run consumed the examples whose last
    rows its two batches held (ids 1 to 16, the rowless every third carried on), not the bound it never
    read to, which only a stream that ran out at its bound gives its last batch."""
    (x,) = clocked_records(0)
    state = thin_fit(x, chain=clocked_chain(x).take(2), examples=CLOCKED_RECORDS)
    assert int(state.step) == 2 and int(state.consumed) == 16


def test_a_clocked_fit_from_a_supplied_state_reads_its_bound_afresh_and_another_clock_keeps_none(tmp_path):
    """A state supplied to a fit opens its data afresh: a clocked fit from a finished clocked run's state
    consumes its whole bound again from none, ending at it; a run of steps from that state keeps no clock,
    in its state or its checkpoints, its schedules counted in updates; and a clocked fit from that run's
    state reads its progress again (`reclocked`)."""
    (x,) = clocked_records(0)
    # A fit donates the state it is given, so each one's counts are read before the next fit takes it.
    first = thin_fit(x, examples=CLOCKED_RECORDS)
    steps = int(first.step)
    assert int(first.consumed) == CLOCKED_RECORDS
    again = thin_fit(x, examples=CLOCKED_RECORDS, state=first)
    assert int(again.consumed) == CLOCKED_RECORDS and int(again.step) == 2 * steps
    checkpoints = Checkpoints(str(tmp_path))
    stepped = thin_fit(x, steps=2 * steps + 2, state=again, checkpoints=checkpoints, checkpoint_every=1)
    checkpoints.wait()
    assert stepped.consumed is None
    assert not {CONSUMED_EXAMPLES, CONSUMED_COMMITTED} & checkpoints.control(int(stepped.step)).keys()
    # Back on the examples clock, its schedule reads the run's progress from none consumed again.
    assert int(thin_fit(x, examples=CLOCKED_RECORDS, state=stepped).consumed) == CLOCKED_RECORDS


class Placed(Pulled):
    """`Pulled`, keeping the fields of every batch the trainer places for its step, as it reads each placed
    batch's row geometry (`Objective.row_tokens`)."""

    def __init__(self):
        super().__init__()
        self.fields: list[frozenset[str]] = []

    def row_tokens(self, batch):
        self.fields.append(frozenset(batch))
        return super().row_tokens(batch)


def test_only_a_run_on_the_examples_clock_carries_its_count_to_the_step(tmp_path):
    """A run bounded in examples whose plan counts its steps ahead (a global order read in memory) runs on
    steps: the count its data marks rides no batch into its step, its state keeps no clock and its
    checkpoints none, while a run whose plan counts none reads the count in its step."""
    (x,) = clocked_records(0)
    ids = np.arange(1, CLOCKED_RECORDS + 1, dtype=np.int32)
    optim = OptimConfig(optimizer="adam", learning_rate=Linear(peak_value=PEAK, end_value=CLOCKED_END))
    planned = Pipeline(Memory({"x": x, "id": ids})).read("memory").batch(BATCH)
    loader, _ = fitted(planned, None, objective=Placed().data(), examples=CLOCKED_RECORDS)
    stream = loader.train(DataPartition())
    try:
        assert EXAMPLES_CONSUMED in next(stream)
    finally:
        if isinstance(stream, Closeable):
            stream.close()
    seen, checkpoints = Placed(), Checkpoints(str(tmp_path))
    state = Trainer(seen, optim, key=0, checkpoints=checkpoints).fit(
        planned, examples=CLOCKED_RECORDS, log_every=1, checkpoint_every=1)
    checkpoints.wait()
    assert state.consumed is None and seen.fields
    assert not any(EXAMPLES_CONSUMED in held for held in seen.fields)
    assert not {CONSUMED_EXAMPLES, CONSUMED_COMMITTED} & checkpoints.control(int(state.step)).keys()
    clocked = Placed()
    chain = Pipeline(Memory({"x": x, "id": ids})).read("streamed").batch(BATCH)
    Trainer(clocked, optim, key=0).fit(chain, examples=CLOCKED_RECORDS, log_every=1)
    assert clocked.fields and all(EXAMPLES_CONSUMED in held for held in clocked.fields)


def test_a_clocked_run_whose_split_is_planned_compiles_one_step():
    """Every batch of a clocked run carries whether it closes its window, so a run whose first compile
    plans its split keeps one step program: one compile over its batches of one shape, its short last
    batch filled to it."""
    (x,) = clocked_records(0)
    tracker = RecordingTracker()
    optim = OptimConfig(optimizer="adam", learning_rate=Linear(peak_value=PEAK, end_value=CLOCKED_END))
    Trainer(Echoed(), optim, key=0, tracker=tracker).fit(clocked_chain(x), examples=CLOCKED_RECORDS,
                                                         log_every=1)
    assert len(tracker.scalars) > 1
    assert sum("train/compile_s" in scalars for _, scalars in tracker.scalars) == 1


def test_a_batched_step_that_makes_no_rows_of_a_group_carries_its_examples_on():
    """A batched step after a pack that makes no rows of a group of windows leaves the examples whose last
    tokens those windows held to the next window kept: the batches of a read bounded in examples consume
    every one once, where the step's empty group was refused."""
    docs = [np.full(index % 4 + 2, index + 1, np.int32) for index in range(12)]

    def drop_first(columns):
        # The group of windows holding the first document's tokens makes no rows; the rest keep theirs.
        if any(1 in np.asarray(window) for window in columns["text"]):
            return {name: [] for name in columns}
        return columns

    chain = (Pipeline(Memory({"text": docs})).read("streamed").pack(6, fields=("text",))
             .flat_map(drop_first, batched=True, batch_size=2).batch(2))
    stream = compile(chain, read="streamed", examples=len(docs)).train(DataPartition())
    try:
        batches = list(stream)
    finally:
        if isinstance(stream, Closeable):
            stream.close()
    assert not any(1 in np.asarray(batch["text"]) for batch in batches)
    assert sum(int(np.sum(batch[EXAMPLES_CONSUMED])) for batch in batches) == len(docs)


def clocked_records(seed: int) -> tuple[np.ndarray]:
    """A clocked run's corpus of one draw: CLOCKED_RECORDS rows of FEATURES coordinates, unit normals."""
    return (np.random.default_rng(seed).normal(size=(CLOCKED_RECORDS, FEATURES)).astype(np.float32),)


def test_a_clocked_runs_share_of_its_examples_falls_where_its_batches_pass_it():
    """A run bounded in examples that counts no steps ahead logs at a share of its examples (`25%`, every 5
    of 18) at each step whose batch brings its consumed examples past a multiple of that share
    (`Examples`), and at its last."""
    ids = np.arange(1, CLOCKED_RECORDS + 1)
    owner = np.repeat(np.arange(CLOCKED_RECORDS), ids % 3 + 1)
    last = np.r_[owner[1:] != owner[:-1], True]
    consumed = np.cumsum([int(np.sum(last[start:start + BATCH])) for start in range(0, len(owner), BATCH)])
    passing = [step for step, (now, before) in enumerate(zip(consumed, np.r_[0, consumed[:-1]], strict=True),
                                                         start=1) if now // 5 > before // 5]
    _, tracker = clocked_fit(*clocked_records(0), 1, log_every="25%")
    logged = [step for step, scalars in tracker.scalars if "train/loss" in scalars]
    assert logged == sorted({*passing, len(consumed)})


def test_a_clocked_run_paused_at_half_its_examples_and_resumed_trains_on_as_one_that_never_paused(tmp_path):
    """A run bounded in examples that counts no steps ahead, paused once half its examples are consumed and
    resumed without the pause, lands where the whole run does, step for step and in its weight
    (check 1): its checkpoint held the examples it consumed with its data position."""
    directories = iter(range(1 << 20))

    def paused(x):
        return clocked_run(2, tmp_path / str(next(directories)), (Pause(0.5), None))(x)

    assert_as_exact(as_given(paused), clocked_run(2), clocked_records, "paused clocked run",
                    seeds=PAUSE_SEEDS)


@pytest.mark.parametrize("accumulation", [1, 2])
def test_a_run_no_pass_counts_ahead_reads_its_schedule_off_the_examples_it_consumed(accumulation):
    """A run bounded in examples whose objective makes rows of each after the bound counts no steps ahead:
    its rate falls over the share of its examples its batches consumed, each example consumed by the batch
    that holds its last row, and its losses and weight are Adam's at that rate (check 1); updating every
    two batches, the window its stream ends inside commits over its own rows."""
    assert_as_exact(clocked_run(accumulation), clocked_descent(accumulation), clocked_records,
                    f"clocked run of {accumulation}", seeds=EXAMPLES_SEEDS)


def test_a_paused_run_resumed_trains_on_as_one_that_never_paused(tmp_path):
    """A run paused halfway through its budget and resumed without the pause lands where the whole
    run does, step for step and in its weight (check 1): the budget, not the pause, sized its
    schedule, and its checkpoint held its data position."""
    directories = iter(range(1 << 20))
    assert_as_exact(as_given(lambda x: trained(x, tmp_path / str(next(directories)), (Pause(0.5), None))),
                    trained,
                    records, "paused run", seeds=PAUSE_SEEDS, truth_fn=descended)


def test_a_run_paused_where_it_already_stands_trains_nothing_more(tmp_path):
    """A paused run fitted again with the same pause is already where it pauses: it returns the
    checkpointed state at the pause, two of the three updates in, and trains no further."""
    (x,) = records(0)
    first = trainer(tmp_path).fit(chain(x), examples=EXAMPLES, stop=Pause(0.5))
    again = trainer(tmp_path).fit(chain(x), examples=EXAMPLES, stop=Pause(0.5))
    assert int(first.step) == int(again.step) == 2 * ACCUMULATION
    assert Checkpoints(str(tmp_path)).latest == 2 * ACCUMULATION


def test_a_pause_on_a_step_that_ranks_a_weights_only_best_writes_the_whole_state(tmp_path):
    """A pause whose step also ranks a best kept as weights alone writes that step's whole state and
    data position, which the run resumed without the pause goes on from to the end of its budget."""
    (x,) = records(0)
    objective = Pulled()

    def fitted(stop):
        return Trainer(objective, OptimConfig(optimizer="adam", learning_rate=Linear(peak_value=PEAK)), key=0,
                       accumulation=ACCUMULATION, checkpoints=Checkpoints(str(tmp_path))).fit(
            chain(x), examples=EXAMPLES, stop=stop, eval_every=2 * ACCUMULATION,
            best=Best(objective.loss, weights_only=True))

    assert int(fitted(Pause(0.5)).step) == Checkpoints(str(tmp_path)).latest == 2 * ACCUMULATION
    assert int(fitted(None).step) == 3 * ACCUMULATION


def test_an_examples_count_whose_last_steps_make_no_whole_update_is_refused():
    """Thirty-five records fill five batches, which leave a step no update takes, so the run is refused
    before it trains rather than read records it never learns from."""
    (x,) = records(0)
    with pytest.raises(ValueError):
        trainer().fit(chain(x), examples=35)


def test_a_stream_that_runs_out_on_an_evaluated_step_scores_that_step_once():
    """A stream of three batches under a run of five steps, evaluated at every step, ends at step 3,
    whose evaluation on its cadence is the run's last: each step's split is scored once."""
    (x,) = records(0)
    tracker = RecordingTracker()
    batch = {"x": x[:BATCH]}
    finite = Loader(lambda partition: iter([batch] * 3), lambda partition: iter([batch]), None, BATCH)
    Trainer(Pulled(), OptimConfig(optimizer="adam", learning_rate=0.1), key=0, tracker=tracker).fit(
        finite, steps=5, eval_every=1, log_every=1)
    assert [step for step, scalars in tracker.scalars if "val/loss" in scalars] == [1, 2, 3]


def test_a_stream_that_runs_out_after_a_ranked_step_ends_on_a_whole_checkpoint(tmp_path):
    """A one-batch stream under a run of ten steps, its step ranking a best kept as weights alone,
    ends at step 1 on a checkpoint the run resumes from: the whole state and its data position."""
    (x,) = records(0)
    objective = Pulled()
    finite = Loader(lambda partition: iter([{"x": x[:BATCH]}]), None, None, BATCH)
    Trainer(objective, OptimConfig(optimizer="adam", learning_rate=0.1), key=0,
            checkpoints=Checkpoints(str(tmp_path))).fit(
        finite, steps=10, eval_every=1, best=Best(objective.loss, weights_only=True))
    assert Checkpoints(str(tmp_path)).latest == 1


def test_a_run_resumed_where_its_finite_stream_ran_out_ends_there_with_nothing_to_read(tmp_path):
    """A one-pass read of the corpus under a run asked for more steps than it holds ends where the read
    runs out, on a checkpoint; resumed with a higher cap it opens the read where it stopped, finds it
    has run out and ends at the same step, its state untouched, rather than refusing a stream that holds
    no batch."""
    (x,) = records(0)

    def run(steps):
        trainer = Trainer(Pulled(), OptimConfig(optimizer="adam", learning_rate=0.1), key=0,
                          checkpoints=Checkpoints(str(tmp_path)))
        return trainer.fit(compile(Pipeline(Memory({"x": x})).repeat(1).batch(BATCH), read="memory"),
                           steps=steps)

    first, again = run(10), run(12)
    assert 0 < int(first.step) == int(again.step) < 10
    np.testing.assert_array_equal(again.variables["params"]["w"], first.variables["params"]["w"])


def test_a_stream_that_runs_out_between_cadences_scores_only_the_suites_its_last_step_did_not():
    """A one-batch stream under a run of five steps ends at step 1, where the suite scored every step
    already reported and the one scored every other step did not: the run's end scores that one alone,
    each suite once at step 1."""
    from dew.training import EvalSuite

    (x,) = records(0)
    tracker = RecordingTracker()
    batch = {"x": x[:BATCH]}

    def held(partition):
        return iter([batch])

    finite = Loader(lambda partition: iter([batch]), None, None, BATCH)
    Trainer(Pulled(), OptimConfig(optimizer="adam", learning_rate=0.1), key=0, tracker=tracker).fit(
        finite, steps=5, log_every=1, validation={"often": EvalSuite(held, every=1),
                                                   "rare": EvalSuite(held, every=2)})
    reported = [(step, name) for step, scalars in tracker.scalars for name in ("often", "rare")
                if f"{name}/loss" in scalars]
    assert sorted(reported) == [(1, "often"), (1, "rare")]


def test_a_stream_that_runs_out_with_no_final_checkpoint_writes_its_weights_only_best_whole(tmp_path):
    """A one-batch stream under a run of five steps that writes no final checkpoint ends at step 1,
    whose closing evaluation ranks its first best, kept as weights alone: the stream is closed by then,
    so the run reads no further batch, and the best it writes is the whole state of its last step."""
    (x,) = records(0)
    objective = Pulled()
    finite = Loader(lambda partition: iter([{"x": x[:BATCH]}]), lambda partition: iter([{"x": x[:BATCH]}]),
                    None, BATCH)
    Trainer(objective, OptimConfig(optimizer="adam", learning_rate=0.1), key=0,
            checkpoints=Checkpoints(str(tmp_path))).fit(
        finite, steps=5, checkpoint_final=False, best=Best(objective.loss, weights_only=True))
    assert Checkpoints(str(tmp_path)).latest == 1


def test_a_run_config_counts_examples_and_pauses_as_a_fit_does(tmp_path):
    """A run config's `examples` and `stop` reach the fit, its record holding both: paused halfway it
    stands two updates in, and run again without the pause it goes on to the steps its examples fill."""
    (x,) = records(0)
    optim = OptimConfig(optimizer="adam", learning_rate=Linear(peak_value=PEAK))
    config = RunConfig(optim=optim, trainer=TrainerConfig(
        checkpoint_dir=str(tmp_path), examples=EXAMPLES, stop=Pause(0.5), accumulation=ACCUMULATION,
        log_every=1, eval_every=None, checkpoint_every=None))
    assert RunConfig.from_dict(config.to_dict()) == config
    flags = RunConfig.cli(["--trainer.examples", str(EXAMPLES), "--trainer.stop.fraction", "0.5"]).trainer
    assert (flags.examples, flags.stop) == (EXAMPLES, Pause(0.5))
    paused = config.train(Pulled(), chain(x), name="budget")
    resumed = dataclasses.replace(config, trainer=dataclasses.replace(config.trainer, stop=None)).train(
        Pulled(), chain(x), name="budget")
    assert (int(paused.step), int(resumed.step)) == (2 * ACCUMULATION, 3 * ACCUMULATION)


def test_asking_a_streamed_runs_steps_reads_none_of_its_data():
    """A run bounded in examples whose read no plan counts ahead keeps its progress in the examples it
    consumes, so asking its steps (`Loader.example_steps`) reads none of its data: its objective's
    expansion makes no row of any example."""
    from dew.data.pipeline import ObjectiveData

    seen: list[int] = []

    class Watched(Fanned):
        def flat_map(self, element):
            seen.append(int(element["id"]))
            return super().flat_map(element)

    chain = Pipeline(Memory({"id": np.arange(1, 24, dtype=np.int32)}))
    loader = compile(chain.batch(BATCH), read="streamed", objective=ObjectiveData(train=(Watched(),)),
                     examples=23)
    read = len(seen)
    assert loader.example_steps(23) is None and len(seen) == read


@pytest.mark.parametrize("schedule", [OneCycle(peak_value=PEAK, warmup_fraction=0.25),
                                      Cosine(peak_value=PEAK, warmup_steps="25%")],
                         ids=["one-cycle", "cosine"])
def test_a_clocked_run_of_a_warmup_as_a_share_trains_and_records_it_as_written(schedule):
    """A run bounded in examples that counts no steps ahead trains under a schedule whose warmup is a share
    of the run, which it records at its start as written (`OptimConfig.warmups`), there being no updates
    ahead to count it in."""
    state, tracker = clocked_fit(*clocked_records(0), 1, learning_rate=schedule)
    assert state is not None and int(state.consumed) == CLOCKED_RECORDS
    from dew.telemetry.records import FitStarted

    started = next(value for _, value in tracker.artifacts if isinstance(value, FitStarted))
    assert started.cadences["warmup_steps.learning_rate"] == "25%"


def test_a_causal_windows_last_target_is_consumed_by_the_window_that_holds_it():
    """A document whose last token is the last target of the stream's final causal window is consumed by
    that window, the first that holds it, though the next window's stride would begin at it."""
    import grain.python as grain

    from dew.data.pipeline import CONSUMED, Windows
    from dew.data.steps.windows import CausalWindows

    documents = grain.MapDataset.source([{"text": np.arange(1, 6, dtype=np.int32),
                                          CONSUMED: np.asarray([0, 0, 0, 0, 1], np.int32)}]).to_iter_dataset()
    windows = list(CausalWindows(documents, Windows(seq_len=4, stride=4, fields=("text", CONSUMED))))
    assert [window[CONSUMED] for window in windows] == [1]
