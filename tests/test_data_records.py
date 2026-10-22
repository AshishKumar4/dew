"""A chain's record (`dew.data.records`): what a run's record holds for its data. A chain read back
from its record compiles to the same plan; one no record carries whole is `Unrecorded`, with the
reason."""

import dataclasses
import json
from dataclasses import dataclass

import grain.python as grain
import numpy as np
import pytest

from dew.data import Loading, Memory, from_iter, mix, phases, presets, text
from dew.data.compile import compile, held
from dew.data.dataset import DataPartition
from dew.data.pipeline import Dataset
from dew.data.records import Unrecorded, decode, encode, override

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


def _tokens(start: int, name: str):
    """Documents tokenized by bytes, ended by EOS, under `name`."""
    return _documents(start).named(name).map(text.tokenize("byte", eos=True)).repeat()


def _lm(dataset, **settings):
    return dataset.pipe(presets.lm, "byte", seq_len=8, batch=2, **settings)


CHAINS = {
    "lm": lambda: _lm(_documents().shuffle(0).repeat()),
    "windows": lambda: _lm(_documents().shuffle(0).repeat(), pack=None),
    "mix": lambda: mix([_tokens(0, "a"), _tokens(8, "b")], [0.25, 0.75], by="tokens").pipe(
        presets.lm, None, seq_len=8, batch=2, pack=None),
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
class Tagged(grain.MapTransform):
    """A step of the caller's, a Grain transform at module level, which a record names."""

    offset: int
    identity = "tagged@1"

    def map(self, row):
        return {**row, "text": row["text"] + " tag" * self.offset}


def test_the_record_of_a_store_and_its_stages_is_json_of_classes_and_their_fields():
    """The record of documents shuffled and batched: JSON naming the store and each stage by its
    import path."""
    written = encode(_documents().shuffle(0).batch(2))
    assert not isinstance(written, Unrecorded)
    assert written["version"] == 1
    assert written["source"]["class"] == "dew.data.sources.memory:Memory"
    assert [stage["class"] for stage in written["stages"]] == ["dew.data.pipeline:Shuffle",
                                                               "dew.data.pipeline:BatchStage"]
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
    pytest.param(lambda: Memory([{"x": np.float64(1.5)}]).batch(1), id="a scalar JSON makes a float"),
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
    # Eight documents in batches of three: three batches of three rows each.
    sizes = [len(batch["text"]) for batch in held(unbatched, batch=3)(DataPartition())]
    assert sizes == [3, 3, 3]


def test_a_presets_call_is_recorded_as_the_call_and_its_settings():
    """Documents through the lm preset: the record writes the call, its import path and settings,
    as a hand-written run record would, and reads back the chain it made."""
    chain = _documents().shuffle(0).repeat().pipe(presets.lm, "byte", seq_len=8, batch=2)
    written = encode(chain)
    assert not isinstance(written, Unrecorded)
    assert written["stages"][-1] == {"pipe": "dew.data.presets:lm",
                                     "settings": {"tokenizer": "byte", "seq_len": 8, "batch": 2}}
    assert _batches(decode(written)) == _batches(chain)


def test_an_override_sets_a_presets_setting_and_the_chain_follows_it():
    """The lm preset's row length overridden in the record: the chain read back lays out rows of the
    new length; a setting the preset doesn't take, or a call the chain doesn't make, is refused."""
    written = encode(_documents().shuffle(0).repeat().pipe(presets.lm, "byte", seq_len=8, batch=2))
    assert not isinstance(written, Unrecorded)
    longer = decode(override(written, "lm.seq_len", 12))
    assert longer.facts()["seq_len"] == 12
    assert {len(row) for batch in _batches(longer) for row in batch} == {13}
    with pytest.raises(ValueError):
        override(written, "lm.widths", 3)
    with pytest.raises(ValueError):
        override(written, "images.size", 3)


def test_an_override_points_a_chain_at_another_store_through_its_holdout(tmp_path):
    """A run's default chain read from another store: `source.url` sets the token store's URL, and
    through a holdout's selection `source.path` sets the TFDS split's directory below it. A field
    no store of the chain declares, or a value its annotation refuses, is refused."""
    from dew.data import TFDS
    from dew.data.sources.tokens import TokenFiles

    store = TokenFiles(str(tmp_path / "a"))
    windows = encode(store.shuffle(0).repeat().pipe(presets.token_windows, seq_len=8, batch=2))
    assert not isinstance(windows, Unrecorded)
    moved = decode(override(windows, "source.url", str(tmp_path / "b")))._chain().source
    assert isinstance(moved, TokenFiles) and moved.url == str(tmp_path / "b")
    trained, _ = TFDS("flowers", path=str(tmp_path / "tfds")).holdout(records=4)
    pictures = encode(trained.shuffle(0).repeat().pipe(presets.images, 8, batch=2))
    assert not isinstance(pictures, Unrecorded)
    split = decode(override(pictures, "source.path", str(tmp_path / "elsewhere")))._chain().source
    assert split.source.path == str(tmp_path / "elsewhere") and split.start == 4
    with pytest.raises(ValueError):
        override(pictures, "source.colour", "red")
    with pytest.raises(ValueError):
        override(windows, "source.url", 3)


def test_an_override_reads_a_number_as_its_field_and_refuses_a_list_where_no_member_is_one(tmp_path):
    """A hash holdout's bound set to a whole number reads as the fraction it is; a store's field that
    is a name or nothing refuses a list, which none of its kinds is."""
    from dew.data.sources.tokens import TokenFiles

    trained, _ = _documents().holdout(0.25)
    held_out = encode(trained.shuffle(0).repeat().pipe(presets.lm, "byte", seq_len=8, batch=2))
    assert not isinstance(held_out, Unrecorded)
    assert decode(override(held_out, "source.low", 0))._chain().source.low == 0
    store = TokenFiles(str(tmp_path))
    windows = encode(store.shuffle(0).repeat().pipe(presets.token_windows, seq_len=8, batch=2))
    assert not isinstance(windows, Unrecorded)
    named = override(windows, "source.field", "ids")
    assert named["source"]["input"]["source"]["fields"]["field"] == "ids"
    decode(named)
    with pytest.raises(ValueError):
        override(windows, "source.field", ["ids"])


def test_a_call_whose_argument_changed_since_is_recorded_as_the_stages_it_made():
    """A rename by a mapping changed after the call: the call made again would rename otherwise, so
    the record writes the stages it made, which read back as they were."""
    names = {"text": "words"}
    chain = _documents().pipe(Dataset.rename_columns, names)
    names["text"] = "other"
    written = encode(chain)
    assert not isinstance(written, Unrecorded)
    assert "pipe" not in written["stages"][-1]
    assert next(iter(decode(written)))["words"] == next(iter(chain))["words"]


def _shouted(dataset, *, times: int):
    """A preset of the caller's at module level: each record's text repeated."""
    return dataset.map(Tagged(times))


def test_a_callers_preset_is_recorded_as_its_call():
    """A preset of the caller's, a module-level function, is recorded as its call and read back."""
    chain = _documents().pipe(_shouted, times=2).pipe(presets.lm, "byte", seq_len=8, batch=2)
    written = encode(chain)
    assert not isinstance(written, Unrecorded)
    calls = [stage["pipe"] for stage in written["stages"]]
    assert calls == ["test_data_records:_shouted", "dew.data.presets:lm"]
    assert _batches(decode(written)) == _batches(chain)


@dataclass(frozen=True)
class Drawn(grain.RandomMapTransform):
    """A random step: each record's draw, from the seed the read gives it."""

    identity = "drawn@1"

    def random_map(self, row, rng):
        return {**row, "draw": np.int64(rng.integers(0, 2**31))}


def test_fitted_reads_training_and_validation_with_the_runs_data_seed():
    """A chain with a random step, fitted for a run under two data seeds: the training draws and
    the validation draws each follow the seed, and the same seed draws the same."""
    from dew.data.compile import fitted
    from dew.data.pipeline import ObjectiveData

    def draws(seed):
        chain = _documents().map(Tagged(1)).random_map(Drawn()).batch(2).workers(0)
        loader, held = fitted(chain, chain, objective=ObjectiveData(), seed=seed)
        reader = loader.train(DataPartition())
        try:
            trained = [np.asarray(next(reader)["draw"]).tolist() for _ in range(2)]
        finally:
            reader.close()
        assert held is not None
        return trained, [np.asarray(batch["draw"]).tolist() for batch in held["val"](DataPartition())]

    assert draws(1) == draws(1)
    first, second = draws(1), draws(2)
    assert first[0] != second[0] and first[1] != second[1]


def _plain(value):
    """A batch's value as plain lists, a dataclass of arrays (a row's coordinates) field by field."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {field.name: _plain(getattr(value, field.name)) for field in dataclasses.fields(value)}
    return np.asarray(value).tolist()


def _records_of(dataset, steps: int = 3, read: str = "indexed"):
    """The first `steps` batches of `dataset` compiled for `read`, as plain lists."""
    loading = Loading(workers=0, threads=1, read_buffer=1)
    reader = compile(dataset, read=read, loading=loading).train(DataPartition())
    try:
        batches = (next(reader) for _ in range(steps))
        return [{name: _plain(value) for name, value in batch.items()} for batch in batches]
    finally:
        reader.close()


def _prompts():
    rows = Memory([{"prompt": "hi", "ground_truth": "x" * index} for index in range(1, 5)])
    return rows.repeat().pipe(presets.prompts, "byte", max_prompt_len=8, batch=2)


def _cached(tmp_path):
    return _documents().map(Tagged(1)).cache(str(tmp_path / "cache")).shuffle(0).repeat().batch(2)


def _phased():
    phased = phases((_documents(0).map(Tagged(1)), 2), (_documents(8).map(Tagged(2)), None))
    return phased.shuffle(0).repeat().batch(2)


ROUND_TRIPS = {
    "phases": lambda tmp_path: _phased(),
    "a holdout": lambda tmp_path: _lm(_documents().holdout(records=2)[0].shuffle(0).repeat()),
    "a cache": _cached,
    "a prompt width measured": lambda tmp_path: _prompts(),
    "host steps": lambda tmp_path: _lm(_documents().shuffle(0).repeat(), host=(Batched(),)),
    "its loading and read": lambda tmp_path: _lm(_documents().shuffle(0).repeat()).read("indexed").workers(0),
}


@dataclass(frozen=True)
class Batched(grain.MapTransform):
    """A host step of the caller's: each batch as it is, its rows counted."""

    def map(self, batch):
        return {**batch, "rows": np.int32(len(batch["text"]))}


@pytest.mark.parametrize("name", list(ROUND_TRIPS))
def test_each_part_of_a_chain_reads_back_from_its_record(tmp_path, name):
    """Phases, a holdout's part, a cache, a measured prompt width, host steps, a loading and a read:
    each chain recorded and read back reads the same batches."""
    chain = ROUND_TRIPS[name](tmp_path)
    written = encode(chain)
    assert not isinstance(written, Unrecorded), written
    # A measured width is measured by a memory read.
    read = "memory" if name == "a prompt width measured" else "indexed"
    assert _records_of(decode(json.loads(json.dumps(written))), read=read) == _records_of(chain, read=read)


def test_a_published_generation_reads_back_from_its_record(tmp_path):
    """A generation published from documents stands in for its steps: recorded and read back, it
    reads the same batches."""
    from dew.data.cache import Cache
    from dew.data.compile import publish

    generation = publish(_documents().map(Tagged(1)), Cache(str(tmp_path / "cache")), read="indexed")
    chain = _documents().map(Tagged(1)).cache(generation).shuffle(0).repeat().batch(2)
    written = encode(chain)
    assert not isinstance(written, Unrecorded), written
    assert _records_of(decode(json.loads(json.dumps(written)))) == _records_of(chain)


def test_a_record_naming_a_class_no_store_is_refused_before_it_is_built(tmp_path):
    """A store's record naming `subprocess.Popen` with a command: refused as no store of Dew's or a
    trusted package's, before anything is built, so the command never runs."""
    marker = tmp_path / "ran"
    written = encode(_documents().batch(1))
    assert not isinstance(written, Unrecorded)
    forged = {**written, "source": {"class": "subprocess:Popen", "fields": {"args": ["touch", str(marker)]}}}
    with pytest.raises(ValueError):
        decode(forged)
    assert not marker.exists()


def test_an_override_of_a_positional_setting_replaces_it(tmp_path):
    """The images preset's size given in its call's place, overridden: the chain read back makes
    images of the new size; a setting of the wrong type, or the dataset the call reads, is refused."""
    import cv2
    import pyarrow as pa
    import pyarrow.parquet as pq

    from dew.data import Files

    images = [{"image": cv2.imencode(".png", np.full((6, 6, 3), index, np.uint8))[1].tobytes(),
               "label": index} for index in range(4)]
    pq.write_table(pa.Table.from_pylist(images), tmp_path / "images.parquet")
    written = encode(Files(str(tmp_path / "images.parquet")).repeat().pipe(presets.images, 4, batch=2))
    assert not isinstance(written, Unrecorded)
    larger = decode(override(written, "images.size", 5))
    assert np.asarray(next(iter(larger))["image"]).shape[1:] == (5, 5, 3)
    with pytest.raises(ValueError):
        override(written, "images.size", "large")
    with pytest.raises(ValueError):
        override(written, "images.dataset", 3)


def _with_width(dataset, *, options):
    """A preset of the caller's whose step takes its width from `options` as a keyword default."""
    def widen(row, *, width=options["width"]):
        return {**row, "text": row["text"] + "!" * width}

    return dataset.map(widen)


def test_a_call_whose_keyword_default_changed_since_is_recorded_as_its_stages():
    """A caller's preset whose step keeps a setting as a keyword default, the setting changed after
    the call: made again it would make another step, so the record writes the stages it made."""
    options = {"width": 1}
    chain = _documents().pipe(_with_width, options=options)
    options["width"] = 3
    assert isinstance(encode(chain), Unrecorded)


def _outer(dataset, *, times: int):
    """A preset of the caller's that pipes another."""
    return dataset.pipe(_shouted, times=times)


def test_a_preset_calling_another_is_recorded_as_the_outer_call():
    """A preset of the caller's that pipes another over the same stages is recorded as the outer
    call, which an override then addresses."""
    written = encode(_documents().pipe(_outer, times=2).batch(1))
    assert not isinstance(written, Unrecorded)
    assert written["stages"][0]["pipe"] == "test_data_records:_outer"
    changed = decode(override(written, "_outer.times", 3))
    assert next(iter(changed))["text"][0].endswith(" tag tag tag")


def test_a_chain_over_stores_not_present_here_is_recorded_and_read_back(tmp_path):
    """A run's record is written where its data may not be: a TFDS split never prepared here, a
    directory of clips not yet fetched. The record is the stores' settings and the steps, so it is
    written and read back without opening either store."""
    from dew.data import TFDS, Files
    from dew.data.sources.av_utils import FPS

    split = TFDS("never_prepared", path=str(tmp_path / "tfds"))
    clips = Files(str(tmp_path / "clips"), format="files")
    images = _read_back(split.shuffle(0).repeat().pipe(presets.images, 16, batch=4))._chain().source
    assert (images.name, images.path) == ("never_prepared", str(tmp_path / "tfds"))
    listing = clips.pipe(presets.video, 4 / FPS, frames=4, fps=FPS, size=16, batch=2)
    media = _read_back(listing)._chain().source
    assert (media.url, media.kind, media.suffixes) == (str(tmp_path / "clips"), "files",
                                                       (".mp4", ".avi", ".mov", ".webm"))


def test_a_store_that_reads_other_records_by_the_time_it_is_recorded_is_unrecorded(tmp_path):
    """A directory's files listed when its chain was read, and a file added before the record is
    written: the record would read the new listing, not the records the run read, so it is
    unrecorded rather than written as the same chain."""
    from dew.data import Files

    (tmp_path / "a.jsonl").write_text('{"text": "a"}\n')
    rows = Files(str(tmp_path), format="jsonl")
    assert len(rows.paths) == 1 and rows.fingerprint()
    (tmp_path / "b.jsonl").write_text('{"text": "b"}\n')
    assert isinstance(encode(rows.batch(1)), Unrecorded)


def test_a_concatenation_of_stores_in_memory_is_recorded_and_read_back():
    """Two stores in memory read one after the other: each reads back as the store it was, its rows
    held in whichever sequence, inside the concatenation as on its own."""
    from dew.data import concat

    chain = _lm(concat([_documents(0), _documents(8)]).shuffle(0).repeat())
    assert _batches(_read_back(chain)) == _batches(chain)
