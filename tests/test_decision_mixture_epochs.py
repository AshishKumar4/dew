"""Epochs of a slice of a packed decision mixture: every row of the slice is read once a pass,
packing neither drops nor repeats a row where a pass ends or the next begins, and a reader saved
mid-pass resumes on the rows the run that never stopped reads."""

import itertools
from collections import Counter
from dataclasses import replace

import numpy as np
from reader_support import assert_resumes_where_it_stopped
from test_decision_objective import INTENT, LAYOUT, SPECIALS, TOKENIZER, tiny_backbone
from test_decision_packing import alone, rows_of

from dew.batch import VALID_ROWS
from dew.data import Loading, Memory
from dew.data.compile import compile
from dew.data.dataset import DataPartition
from dew.data.pipeline import Pipeline, mix
from dew.decision import DecisionObjective, Example, decisions

RECORDS, HELD = 12, 4
"""Each corpus's records, and how many of its first ones the slice holds out."""
READ = Loading(workers=0, threads=1, read_buffer=0, worker_buffer=1)
LAID = replace(LAYOUT, max_len=192)
OBJECTIVE = DecisionObjective(tiny_backbone().clone(max_seq_len=LAID.max_len), tokenizer=TOKENIZER,
                              specials=SPECIALS, layout=LAID, shuffle_options=False)
TEAMS = ("billing", "technical", "sales")
EXAMPLES = {corpus: [Example(f"{corpus}{index} " + "ticket " * (index % 4), {"team": INTENT},
                             {"team": TEAMS[index % 3]}) for index in range(RECORDS)]
            for corpus in "ab"}


def row(example: Example) -> tuple[int, ...]:
    """The tokens of `example`'s one question row, as the layout lays it out."""
    (laid,) = OBJECTIVE.layout.rows(TOKENIZER, SPECIALS, example.state, {"team": INTENT})
    return tuple(laid.tokens)


SLICE = Counter(row(example) for corpus in "ab" for example in EXAMPLES[corpus][HELD:])
"""The slice's rows: each corpus's records after the first HELD."""


def chain(passes: int) -> Pipeline:
    """The two corpora's slices mixed at equal weights, `passes` passes of every row they hold."""
    sliced = []
    for corpus in "ab":
        records = Memory([{"example": example} for example in EXAMPLES[corpus]])
        train, _ = Pipeline(records).holdout(records=HELD)
        sliced.append(train.pipe(decisions))
    return mix(sliced).take(2 * (RECORDS - HELD)).repeat(passes).batch(2, drop_remainder=False)


def rows(batches) -> list[tuple[int, ...]]:
    """Each real window's rows of `batches`, in window order, each its own tokens."""
    read = []
    for batch in batches:
        batch = {name: np.asarray(value) for name, value in batch.items()}
        real = batch.get(VALID_ROWS, np.ones(len(batch["tokens"]), bool))
        read += [tuple(alone(batch, window, segment)["tokens"][0].tolist())
                 for window, segment in rows_of(batch) if real[window]]
    return read


def opened(passes: int):
    data = compile(chain(passes), objective=OBJECTIVE.data(), read="memory", loading=READ)
    return data.train(DataPartition())


def test_a_pass_reads_every_row_of_the_slice_once():
    reader = opened(1)
    try:
        read = rows(reader)
    finally:
        reader.close()
    assert Counter(read) == SLICE


def test_two_passes_read_every_row_twice_however_windows_straddle_them():
    """Packing fills windows across the boundary between passes, and drops or repeats nothing there."""
    reader = opened(2)
    try:
        read = rows(reader)
    finally:
        reader.close()
    assert Counter(read) == Counter({tokens: 2 * count for tokens, count in SLICE.items()})


def test_a_reader_saved_mid_pass_resumes_on_the_rows_the_uninterrupted_one_reads():
    expected, _ = assert_resumes_where_it_stopped(
        lambda: opened(2), lambda reader, count: rows(itertools.islice(reader, count)), 2, 3)
    assert set(expected) <= set(SLICE)
