"""A corpus's `data/rows` counts its source examples where an objective makes several rows of each, whatever
rows its step makes of them or windows its pack lays them in, counted in the reading process as each
delivered row says (`CorpusCounts.examples`): read by workers, after every batch, and across a saved position
(row 1890)."""

import numpy as np
import pytest
from examples_clock_worker import Questioned

from dew.batch import EXAMPLES_CONSUMED, VALID_ROWS
from dew.data import Memory
from dew.data.compile import compile
from dew.data.dataset import Closeable, DataPartition, Loading
from dew.data.pipeline import ObjectiveData, Pack, Pipeline, Rejected

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


class Fanned:
    """An objective's expansion of each record `id` into `id % 4` rows, none for some, each row's two
    tokens naming it (`id * 10 + question`)."""

    identity = "fanned@1"

    def flat_map(self, element):
        record = int(element["id"])
        return tuple({"text": np.full(2, record * 10 + question, np.int32)} for question in range(record % 4))


def test_a_corpus_counts_its_source_examples_whatever_windows_a_pack_lays_their_rows_in():
    """Twenty-three examples, made 0 to 3 rows of each, packed into windows of up to four rows, bound in
    examples: the corpus counts the twenty-three, not the windows its pack made, whose counter is the
    outermost of its read."""
    chain = Pipeline(Memory({"id": np.arange(1, 24, dtype=np.int32)}))
    loader = compile(chain.batch(BATCH).select_columns("text"), read="streamed",
                     objective=ObjectiveData(train=(Fanned(),), pack=Pack(9, fields=("text",))), examples=23,
                     loading=Loading(workers=0))
    windows, consumed = read_whole(loader, "text")
    assert consumed == 23 and windows < 23
    assert counted(loader) == [23]


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


class Refusing:
    """An objective's expansion of each record into `id % 3 + 1` rows, refusing every record whose id is a
    multiple of five whole, as a decision layout refuses an example whose row it cannot lay out."""

    identity = "refusing@1"

    def flat_map(self, element):
        if int(element["id"]) % 5 == 0:
            return (Rejected("unfit"),)
        return tuple({"id": element["id"], "question": np.int32(question)}
                     for question in range(int(element["id"]) % 3 + 1))


@pytest.mark.parametrize("read", ["memory", "streamed"])
def test_a_corpus_counts_the_examples_it_kept_apart_from_those_it_refused(read):
    """Twenty-one examples, every fifth refused whole by the objective: the corpus counts the seventeen it
    kept as `data/rows` and the four it refused under their reason, which together are the examples the
    run's batches consume, whatever the refused examples' counts ride on to the next kept row."""
    ids = np.arange(1, 22, dtype=np.int32)
    loader = compile(Pipeline(Memory({"id": ids})).batch(BATCH), read=read,
                     objective=ObjectiveData(train=(Refusing(),)), examples=21)
    _, consumed = read_whole(loader, "question")
    assert consumed == 21
    assert counted(loader) == [17]
    assert counted(loader, "data/rejected/") == [4]


@pytest.mark.parametrize("read", ["memory", "streamed"])
def test_a_run_whose_last_example_is_refused_consumes_it(read):
    """A run bounded to twenty examples whose twentieth its objective refuses: its batches consume all
    twenty, the last one's count carried by the short batch the bound ends inside, as a kept row would
    carry it."""
    ids = np.arange(1, 21, dtype=np.int32)
    loader = compile(Pipeline(Memory({"id": ids})).batch(BATCH), read=read,
                     objective=ObjectiveData(train=(Refusing(),)), examples=20)
    assert read_whole(loader, "question")[1] == 20


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

