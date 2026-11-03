"""A corpus's `data/rows` counts its source examples where an objective makes several rows of each, whatever
rows its step makes of them or windows its pack lays them in, counted in the reading process as each
delivered row says (`CorpusCounts.examples`): read by workers, after every batch, and across a saved position
(row 1890)."""

import grain.python as pygrain
import numpy as np
import pytest
from data_support import Fanned, Questioned, WideDraw

from dew.batch import EXAMPLES_CONSUMED, VALID_ROWS
from dew.data import Memory
from dew.data.compile import compile
from dew.data.dataset import Closeable, DataPartition, Loading
from dew.data.pipeline import Configured, ObjectiveData, Pack, Pipeline, Rejected

BATCH = 8


def read_whole(loader, field: str) -> tuple[int, int]:
    """The real rows of every training batch `loader` reads, and the examples those batches consume."""
    stream = loader.train(DataPartition())
    rows = consumed = 0
    try:
        for batch in stream:
            real = np.asarray(batch.get(VALID_ROWS, np.ones(len(batch[field]), bool)), bool)
            rows += int(real.sum())
            consumed += int(np.sum(batch.get(EXAMPLES_CONSUMED, 0)))
    finally:
        if isinstance(stream, Closeable):
            stream.close()
    return rows, consumed


def counted(loader, prefix: str = "data/rows/") -> list[float]:
    """The values of `loader`'s metrics under `prefix`."""
    return [value for key, value in loader.statistics().items() if key.startswith(prefix)]


@pytest.mark.parametrize("read", ["memory", "streamed"])
def test_a_corpus_counts_its_source_examples_whatever_rows_an_objective_makes_of_them(read):
    """Fifteen examples, each made one to three rows of, bound in examples: the corpus counts the fifteen
    it delivered, not the rows its expansion made of them."""
    ids = np.arange(1, 11, dtype=np.int32)
    shuffled = {"seed": 3} if read == "memory" else {"seed": 3, "window": 10}
    loader = compile(Pipeline(Memory({"id": ids})).shuffle(**shuffled).batch(BATCH), read=read,
                     objective=ObjectiveData(train=(Questioned(),)), examples=15)
    rows, consumed = read_whole(loader, "id")
    assert consumed == 15 and rows > 15
    assert counted(loader) == [15]


def test_a_corpus_counts_its_source_examples_whatever_windows_a_pack_lays_their_rows_in():
    """Twenty-three examples, made 0 to 3 rows of each, packed into windows of up to four rows, bound in
    examples: the corpus counts the eighteen it made rows of, not the windows its pack made, whose counter is
    the outermost of its read, and the five it made none of as refused (`empty_expansion`), together the
    twenty-three its batches consume."""
    chain = Pipeline(Memory({"id": np.arange(1, 24, dtype=np.int32)}))
    loader = compile(chain.batch(BATCH).select_columns("text"), read="streamed",
                     objective=ObjectiveData(train=(Fanned(),), pack=Pack(9, fields=("text",))), examples=23,
                     loading=Loading(workers=0))
    windows, consumed = read_whole(loader, "text")
    assert consumed == 23 and windows < 23
    assert counted(loader) == [18]
    assert counted(loader, "data/rejected/") == [5]


def test_a_decision_corpus_counts_its_source_examples():
    """A decision run's examples, two questions each, cut by a window too short for their states: the
    corpus counts every example it delivered as `data/rows`, whatever windows its rows were packed into."""
    from test_decision_objective import SPECIALS, TOKENIZER, tiny_backbone

    from dew.decision import DecisionObjective, Example, Noul, StateFirstLayout, decisions

    layout = StateFirstLayout(max_len=128)
    objective = DecisionObjective(tiny_backbone().clone(max_seq_len=128), tokenizer=TOKENIZER,
                                  specials=SPECIALS, layout=layout)
    asked = {"first": Noul("Is it on?"), "second": Noul("Is it off?")}
    answers = {"first": "true", "second": "false"}
    records = [{"example": Example(f"state {index} " + "long " * 150, asked, answers)} for index in range(10)]
    loader = compile(Pipeline(Memory(records)).pipe(decisions).batch(BATCH), read="memory",
                     objective=objective.data(), examples=23)
    _, consumed = read_whole(loader, "segments")
    assert consumed == 23
    assert counted(loader) == [23]


def test_a_decision_corpus_counts_the_examples_it_kept_apart_from_those_its_layout_refused():
    """Ten decision examples, two of them asking a question no window holds, bound to eight examples: the
    layout refuses the two before the bound reads (a decision run fits its examples before it spreads them
    into rows), so the bound counts the eight it kept, one pass of them, which the corpus counts as
    `data/rows`, and the two refused under their reason."""
    from test_decision_objective import SPECIALS, TOKENIZER, tiny_backbone

    from dew.decision import DecisionObjective, Example, Noul, StateFirstLayout, decisions

    layout = StateFirstLayout(max_len=128)
    objective = DecisionObjective(tiny_backbone().clone(max_seq_len=128), tokenizer=TOKENIZER,
                                  specials=SPECIALS, layout=layout)
    asked = {"first": Noul("Is it on?"), "second": Noul("Is it off?")}
    unfit = {**asked, "long": Noul("Is it on? " + "Read the state whole. " * 100)}
    answers = {"first": "true", "second": "false", "long": "true"}
    records = [{"example": Example(f"state {index}", unfit if index in (3, 7) else asked,
                                   {name: answers[name] for name in (unfit if index in (3, 7) else asked)})}
               for index in range(10)]
    loader = compile(Pipeline(Memory(records)).pipe(decisions).batch(BATCH), read="memory",
                     objective=objective.data(), examples=8)
    assert read_whole(loader, "segments")[1] == 8
    assert counted(loader) == [8]
    assert counted(loader, "data/rejected/") == [2]


class Refusing:
    """An objective's expansion of each record into `id % 3 + 1` rows, refusing every record whose id is a
    multiple of five whole, as a decision layout refuses an example whose row it cannot lay out."""

    identity = "refusing@1"

    def flat_map(self, element):
        if int(element["id"]) % 5 == 0:
            return (Rejected("unfit"),)
        return tuple({"id": element["id"], "question": np.int32(question)}
                     for question in range(int(element["id"]) % 3 + 1))


@pytest.mark.parametrize(("read", "workers"), [("memory", 0), ("streamed", 0), ("memory", 2)])
def test_a_corpus_counts_the_examples_it_kept_apart_from_those_it_refused(read, workers):
    """Twenty-one examples, every fifth refused whole by the objective, read by `workers` loading workers:
    the corpus counts the seventeen it kept as `data/rows` and the four it refused under their reason,
    together the examples it read, whatever kept row carries a refused example's count on, in the reading
    process or in a worker's group. Its batches consume them all where one process reads them in order;
    workers hand their groups on in turn, so a refused example whose count no later row carries is left to
    the stream's end, which the run's last batch takes (`_fit.drained`)."""
    ids = np.arange(1, 22, dtype=np.int32)
    loader = compile(Pipeline(Memory({"id": ids}, shard_rows=3)).batch(BATCH), read=read,
                     objective=ObjectiveData(train=(Refusing(),)), examples=21,
                     loading=Loading(workers=workers))
    _, consumed = read_whole(loader, "question")
    assert consumed == 21 if workers == 0 else 21 - 4 <= consumed <= 21
    assert counted(loader) == [17]
    assert counted(loader, "data/rejected/") == [4]


@pytest.mark.parametrize("read", ["memory", "streamed"])
def test_a_corpus_counts_a_refused_last_example_its_batches_leave_to_the_streams_end(read):
    """A run bounded to twenty examples whose twentieth its objective refuses: the corpus counts the sixteen
    it kept and the four it refused, every example the bound read, while its batches consume nineteen, the
    twentieth's count owed to the stream's end, which the run's last batch takes (`_fit.drained`)."""
    ids = np.arange(1, 21, dtype=np.int32)
    loader = compile(Pipeline(Memory({"id": ids})).batch(BATCH), read=read,
                     objective=ObjectiveData(train=(Refusing(),)), examples=20)
    assert read_whole(loader, "question")[1] == 19
    assert counted(loader) == [16]
    assert counted(loader, "data/rejected/") == [4]


class DroppingLate(pygrain.RandomMapTransform):
    """A later step of the objective, a draw of its own, that drops every row of the examples 6 and 11: the
    first rows the objective kept after its refusals of 5 and 10, which carry those on."""

    identity = "dropping-late@1"
    rejects = True

    def random_map(self, element, rng):
        return Rejected("late") if int(element["id"]) in (6, 11) else element


def test_examples_refused_by_two_steps_apart_are_counted_once():
    """Twenty-one examples, every fifth refused whole by the objective's expansion and 6 and 11 dropped by a
    later step of its own after the rows carrying 5's and 10's counts were kept: the corpus counts the fifteen
    it kept, each refused example's count carried on once whatever step dropped the row carrying it, and the
    batches consume all twenty-one."""
    ids = np.arange(1, 22, dtype=np.int32)
    loader = compile(Pipeline(Memory({"id": ids})).batch(BATCH), read="memory",
                     objective=ObjectiveData(train=(Refusing(), DroppingLate())), examples=21)
    assert read_whole(loader, "question")[1] == 21
    assert counted(loader) == [15]
    assert sorted(counted(loader, "data/rejected/")) == [4, 4]


def completed(batch) -> int:
    """The examples a batch of `Refusing`'s rows completes: its real rows that are their example's last."""
    real = np.asarray(batch.get(VALID_ROWS, np.ones(len(batch["id"]), bool)), bool)
    last = np.asarray(batch["question"]) == np.asarray(batch["id"]) % 3
    return int(np.sum(real & last))


@pytest.mark.parametrize("workers", [0, 2])
def test_each_batch_states_the_examples_it_kept_by_then_and_a_resumed_read_goes_on_from_them(workers):
    """Twenty-one examples, every fifth refused whole, read eight rows a batch by `workers` loading workers:
    after each batch the corpus states the examples whose last rows it delivered by then, never a refused
    one, wherever in a worker's group a refusal fell; a read resumed from a position saved after the second
    batch states the same counts after each later batch as the read it was saved from."""
    ids = np.arange(1, 22, dtype=np.int32)

    def stream():
        loader = compile(Pipeline(Memory({"id": ids})).batch(BATCH), read="memory",
                         objective=ObjectiveData(train=(Refusing(),)), examples=21,
                         loading=Loading(workers=workers))
        return loader, loader.train(DataPartition())

    def stated(loader) -> int:
        return int(counted(loader)[0]) if counted(loader) else 0

    loader, whole = stream()
    seen, kept, saved = [], 0, None
    try:
        for index, batch in enumerate(whole):
            kept += completed(batch)
            seen.append((kept, stated(loader)))
            if index == 1:
                saved = whole.get_state()
    finally:
        if isinstance(whole, Closeable):
            whole.close()
    assert [stated_then for _, stated_then in seen] == [kept_then for kept_then, _ in seen]
    assert seen[-1][0] == 17
    resumed_loader, resumed = stream()
    resumed.set_state(saved)
    try:
        assert [stated(resumed_loader) for _ in resumed] == [stated_then for _, stated_then in seen[2:]]
    finally:
        if isinstance(resumed, Closeable):
            resumed.close()


@pytest.mark.parametrize("workers", [0, 2])
def test_each_batch_states_the_source_examples_read_by_then_and_a_resumed_read_goes_on_from_them(workers):
    """Fifteen examples in shards of three, each made one to three rows of, read eight rows a batch by
    `workers` loading workers: after each batch the corpus states the source examples whose rows it
    delivered by then, the sum of what those batches consume; a read resumed from a position saved after
    the second batch states the same counts after each later batch as the read it was saved from."""
    ids = np.arange(1, 16, dtype=np.int32)
    loading = Loading(workers=workers)

    def stream():
        loader = compile(Pipeline(Memory({"id": ids}, shard_rows=3)).batch(BATCH), read="streamed",
                         objective=ObjectiveData(train=(Questioned(),)), examples=15, loading=loading)
        return loader, loader.train(DataPartition())

    def stated(loader) -> int:
        return int(counted(loader)[0]) if counted(loader) else 0

    loader, whole = stream()
    seen, consumed, saved = [], 0, None
    try:
        for index, batch in enumerate(whole):
            consumed += int(np.sum(batch.get(EXAMPLES_CONSUMED, 0)))
            seen.append((consumed, stated(loader)))
            if index == 1:
                saved = whole.get_state()
    finally:
        if isinstance(whole, Closeable):
            whole.close()
    assert [stated_then for _, stated_then in seen] == [consumed_then for consumed_then, _ in seen]
    resumed_loader, resumed = stream()
    resumed.set_state(saved)
    try:
        assert [stated(resumed_loader) for _ in resumed] == [stated_then for _, stated_then in seen[2:]]
    finally:
        if isinstance(resumed, Closeable):
            resumed.close()



class Cut:
    """An objective's expansion of each record into `id % 3 + 1` rows, each cut short (`_dew_truncated`)
    by a cut of one row removing two tokens (`_dew_cut`), as a decision layout says of a question it fit
    by cutting its options."""

    identity = "cut@1"

    def flat_map(self, element):
        return tuple({"id": element["id"], "question": np.int32(question), "_dew_truncated": True,
                      "_dew_cut": {"options": (1, 2)}} for question in range(int(element["id"]) % 3 + 1))


class Shortened:
    """A step that says each row was cut short (`_dew_truncated`), as a layout says of one it fit by
    cutting."""

    identity = "shortened@1"

    def __call__(self, row):
        return {**row, "_dew_truncated": True}


def wide_draw(metadata):
    return WideDraw()


def test_a_memory_read_states_the_share_of_its_rows_cut_short_however_often_it_holds_them():
    """Twelve examples a memory read holds, every one cut short (`Shortened`), with a draw configured for
    the run, which a memory read binds between two passes that each hold the corpus and count it anew: the
    share of its rows cut short (data/truncated) is 1.0, of the rows the last pass holds (row 1890's
    review)."""
    ids = np.arange(1, 13, dtype=np.int32)
    drawn = Configured(wide_draw, needs=(), scope="run", random=True)
    loader = compile(Pipeline(Memory({"id": ids})).map(Shortened()).repeat(1).batch(BATCH), read="memory",
                     objective=ObjectiveData(train=(drawn,)))
    read_whole(loader, "id")
    assert counted(loader) == [12]
    assert counted(loader, "data/truncated/") == [1.0]


@pytest.mark.parametrize("bounded", [True, False])
def test_a_corpus_states_the_share_of_its_rows_cut_short_in_rows(bounded):
    """Twelve examples an objective makes 1 to 3 rows of each, every row cut short, read bounded to them
    or once: the share of the corpus's rows cut short (data/truncated) is a share of the rows it delivered,
    all of them, 1.0, and the tokens its cuts removed are its 24 rows' two each, wherever it counts its
    source examples, one per example's last row or one per example before the step (row 1890's
    review)."""
    ids = np.arange(1, 13, dtype=np.int32)
    chain = Pipeline(Memory({"id": ids}, shard_rows=4))
    # Unbounded, a training read repeats without end: it is read once.
    loader = compile((chain if bounded else chain.repeat(1)).batch(BATCH), read="streamed",
                     objective=ObjectiveData(train=(Cut(),)), examples=12 if bounded else None)
    read_whole(loader, "question")
    assert counted(loader) == [12]
    assert counted(loader, "data/truncated/") == [1.0]
    assert counted(loader, "data/cut_removed/") == [2 * 24]


def test_a_streamed_mixture_of_byte_records_counted_by_member_is_refused_not_credited_to_none():
    """A streamed run bounded in examples counts each mixture member's examples under the member its rows
    carry (`MEMBER`), which a record of raw bytes cannot: a mix of members that give byte records, decoded
    only after it, is refused (TypeError) where it would credit their examples to no member. A read in
    memory counts its members by the index it selects them by, bytes or rows alike."""
    read = "streamed"
    from test_pipeline_core import Encoded, Restored

    from dew.data.pipeline import mix

    members = [Pipeline(Memory({"id": np.arange(at * 16, at * 16 + 16, dtype=np.int32)}), name=f"member-{at}")
               .map(Encoded()) for at in range(2)]
    with pytest.raises(TypeError):
        loader = compile(mix(members).map(Restored()).batch(BATCH), read=read, examples=12)
        read_whole(loader, "id")
