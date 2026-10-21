"""A chain's record (`dew.data.records`): what a run's record holds for its data. A chain read back
from its record compiles to the same plan; one no record carries whole is `Unrecorded`, with the
reason."""

import json
from dataclasses import dataclass

import numpy as np
import pytest

from dew.data import Loading, Memory, from_iter, mix, presets
from dew.data.compile import compile, held
from dew.data.dataset import DataPartition
from dew.data.records import Unrecorded, decode, encode

LOADING = Loading(workers=0, threads=1, read_buffer=1)


def _documents(start: int = 0) -> Memory:
    return Memory([{"text": "word " * (2 + index)} for index in range(start, start + 8)])


def _read_back(chain):
    """`chain` recorded, written as JSON and read back."""
    written = encode(chain)
    assert not isinstance(written, Unrecorded), written
    return decode(json.loads(json.dumps(written)))


def _batches(chain, count: int = 3) -> list[list[list[int]]]:
    reader = compile(chain, read="indexed", loading=LOADING, seed=3).train(DataPartition())
    try:
        return [np.asarray(next(reader)["text"]).tolist() for _ in range(count)]
    finally:
        reader.close()


def _lm(dataset, **settings):
    return dataset.pipe(presets.lm, "byte", seq_len=8, batch=2, **settings)


CHAINS = {
    "lm": lambda: _lm(_documents().shuffle(0).repeat()),
    "windows": lambda: _lm(_documents().shuffle(0).repeat(), pack=None),
    "mix": lambda: _lm(mix([_documents().named("a").repeat(), _documents(8).named("b").repeat()],
                           [0.25, 0.75], by="tokens")),
    "steps": lambda: _lm(_documents().shuffle(0, window=4).take(6).repeat().map(Tagged(3))),
}


@pytest.mark.parametrize("name", list(CHAINS))
def test_a_chain_read_back_from_its_record_compiles_to_its_plan_and_batches(name):
    """A chain recorded and read back from its JSON: the same plan, and the same batches."""
    built = CHAINS[name]()
    again = _read_back(built)
    assert compile(again, read="indexed", loading=LOADING, seed=3).plan.identity() == compile(
        built, read="indexed", loading=LOADING, seed=3).plan.identity()
    assert _batches(again) == _batches(built)


@dataclass(frozen=True)
class Tagged:
    """A step of the caller's, a dataclass at module level, which a record names."""

    offset: int
    identity = "tagged@1"

    def __call__(self, row):
        return {**row, "text": row["text"] + " tag" * self.offset}


def test_the_record_of_a_store_and_a_preset_is_json_of_classes_and_their_fields():
    """The record of documents read through the lm preset: JSON naming the store and each stage by
    its import path."""
    written = encode(_documents().pipe(presets.lm, "byte", seq_len=8, batch=2))
    assert not isinstance(written, Unrecorded)
    assert written["version"] == 1
    assert written["source"]["class"] == "dew.data.sources.memory:Memory"
    assert [stage["class"] for stage in written["stages"]][-1] == "dew.data.pipeline:BatchStage"
    json.dumps(written)


def _rows():
    return iter([{"text": "a b"}])


class Rows:
    def opened(self):
        return iter([{"text": "a b"}])


@pytest.mark.parametrize("unrecordable", [
    pytest.param(lambda: _documents().map(lambda row: row), id="a lambda step"),
    pytest.param(lambda: Memory([{"image": np.zeros((2, 2), np.uint8)}]).batch(1), id="arrays in the store"),
    pytest.param(lambda: Memory([{"pair": (1, 2)}]).batch(1), id="a tuple read back a list"),
    pytest.param(lambda: from_iter(Rows().opened, name="rows").batch(1), id="a bound method"),
])
def test_a_chain_no_record_carries_whole_is_unrecorded_with_its_reason(unrecordable):
    """A lambda step, a store of arrays, a store whose tuples a record would read back as lists, an
    iterator opened by a bound method: each chain is Unrecorded, with a reason that names the part."""
    written = encode(unrecordable())
    assert isinstance(written, Unrecorded)
    assert written.reason


def test_a_chain_of_an_importable_iterator_is_recorded():
    """An iterator opened by a module-level function is recorded and read back."""
    again = _read_back(from_iter(_rows, name="rows").batch(1))
    reader = iter(again)
    assert next(reader)["text"] == "a b"


def test_held_reads_a_dataset_in_its_own_batch_or_the_one_given():
    """A validation dataset read once: in its own batches, in the batches given where it states
    none, and refused where it states none and none is given."""
    documents = _documents().pipe(presets.lm, "byte", seq_len=8, batch=2)
    assert {len(np.asarray(batch["text"])) for batch in held(documents)(DataPartition())} == {2}
    unbatched = _documents().map(Tagged(1))
    with pytest.raises(ValueError):
        held(unbatched)
    # Eight documents in batches of three: two full batches and the two left over.
    sizes = [len(batch["text"]) for batch in held(unbatched, batch=3)(DataPartition())]
    assert sizes == [3, 3, 2]
