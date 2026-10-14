"""Text data for language models: tokenizers, the token-file source, the loader.

ByteTokenizer and the token sources are pure numpy; the HF tokenizer tests only
run when a cached copy of the hub is reachable, matching the repo's policy
that no test needs the network.
"""

import itertools
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from absl import flags
from token_support import write_token_corpus

from dew.config.data import TextData, loader
from dew.data import ByteTokenizer, Loading
from dew.data.dataset import DataPartition
from dew.data.sources.tokens import TokenFiles
from dew.nn.backbones import causal_transformer as backbone
from dew.objectives.base import VALID_ROWS
from dew.objectives.lm import LMObjective
from dew.training import Step

REPO_ROOT = Path(__file__).resolve().parents[1]

# grain's worker processes read absl flags; a test that never ran absl.app
# would trip UnparsedFlagAccessError at any worker_count > 0.

if not flags.FLAGS.is_parsed():
    flags.FLAGS.mark_as_parsed()


def _token_dir(tmp_path, train_tokens, val_tokens=None, dtype=np.uint16,
               vocab_size=256, body=None, eos_id=None):
    """Write a token directory: train.bin (+ val.bin + meta.json)."""
    rng = np.random.RandomState(0)
    tokens = (rng.randint(0, vocab_size, train_tokens + (val_tokens or 0))
              if body is None else np.asarray(body, dtype=np.int64))
    if val_tokens is None:
        train, val = tokens, None
    else:
        val, train = tokens[:val_tokens], tokens[val_tokens:]
    return write_token_corpus(tmp_path, train.astype(dtype), None if val is None else val.astype(dtype),
                              vocab_size=vocab_size, eos_id=eos_id)


def _document_dir(tmp_path, documents, eos_id=0, dtype=np.uint16):
    """A token directory whose stream is `documents`, each closed by eos_id."""
    stream = np.concatenate([np.asarray([*d, eos_id], np.int64) for d in documents])
    write_token_corpus(tmp_path, stream.astype(dtype), stream.astype(dtype), eos_id=eos_id)
    return tmp_path, stream


# ---------------------------------------------------------------------------------
# ByteTokenizer
# ---------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------
# HFTokenizer, only when the hub (or its cache) cooperates
# ---------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------
# The windows of a token stream
# ---------------------------------------------------------------------------------

def _stream_dir(tmp_path, body, dtype=np.uint16):
    """A token directory whose train and val splits are both `body`."""
    stream = np.asarray(body).astype(dtype)
    write_token_corpus(tmp_path, stream, stream)
    return tmp_path


def _read(batches, count=None):
    """The rows of `batches`, the first `count` batches of an endless stream, which it closes."""
    try:
        return [{key: value[row] for key, value in batch.items() if key != VALID_ROWS}
                for batch in itertools.islice(batches, count) for row in range(len(batch["text"]))]
    finally:
        batches.close()


@pytest.mark.parametrize("seq_len,stride,tokens", [
    (4, 1, 5), (4, 1, 6), (4, 1, 13), (4, 2, 13), (4, 3, 13), (4, 4, 13), (4, 6, 13),
    (8, 3, 22), (8, None, 37), (1, None, 7),
])
def test_an_epoch_reads_every_complete_window_stride_apart(tmp_path, seq_len, stride, tokens):
    """One training epoch is each complete window of `seq_len + 1` int32 ids
    starting `stride` apart (`seq_len` by default) once, the incomplete tail
    left out; validation tiles the split `seq_len` apart, in file order."""
    body = np.arange(tokens)
    _stream_dir(tmp_path, body, dtype=np.uint32)
    data = loader(TextData(source=str(tmp_path), seq_len=seq_len, stride=stride), batch=1, read="memory",
        loading=Loading(workers=0, threads=1, read_buffer=1, worker_buffer=1))
    starts = range(0, tokens - seq_len, stride or seq_len)

    epoch = _read(data.train(DataPartition()), data.records)
    assert data.records == len(starts)
    assert all(row["text"].dtype == np.int32 for row in epoch)
    assert sorted(row["text"].tolist() for row in epoch) == [body[s:s + seq_len + 1].tolist() for s in starts]
    assert [row["text"].tolist() for row in _read(data.val(DataPartition()))] == [
        body[s:s + seq_len + 1].tolist() for s in range(0, tokens - seq_len, seq_len)]


def test_the_ids_are_read_at_the_width_meta_json_records(tmp_path):
    """uint32 when meta.json says so; without one, nanoGPT's uint16."""
    seq_len = 4
    tokens = np.arange(70000, 70000 + 3 * seq_len + 1, dtype=np.uint32)
    _stream_dir(tmp_path, tokens, dtype=np.uint32)
    data = loader(TextData(source=str(tmp_path), seq_len=seq_len), batch=1, read="memory",
        loading=Loading(workers=0, threads=1, read_buffer=1, worker_buffer=1))
    assert _read(data.val(DataPartition()))[0]["text"].tolist() == tokens[:seq_len + 1].tolist()

    bare = tmp_path / "bare"
    bare.mkdir()
    for split in ("train", "val"):
        (bare / f"{split}.bin").write_bytes(tokens.astype("<u2").tobytes())
    data = loader(TextData(source=str(bare), validation=TokenFiles(str(bare), split="val"), seq_len=seq_len),
                batch=1, read="memory", loading=Loading(workers=0, threads=1, read_buffer=1, worker_buffer=1))
    assert _read(data.val(DataPartition()))[0]["text"].tolist() == tokens[:seq_len + 1].astype("<u2").tolist()


# ---------------------------------------------------------------------------------
# Text
# ---------------------------------------------------------------------------------

def test_token_loader_stride_changes_training_windows_only(tmp_path):
    seq_len, batch = 4, 4
    _token_dir(tmp_path, train_tokens=17, val_tokens=17, body=np.arange(34))
    default = loader(TextData(source=str(tmp_path), seq_len=seq_len), batch=batch, read="memory", seed=17,
        loading=Loading(workers=0, threads=1, read_buffer=1, worker_buffer=1))
    positional = loader(TextData(str(tmp_path), seq_len=seq_len), batch=batch, read="memory", seed=17,
                      loading=Loading(workers=0, threads=1, read_buffer=1, worker_buffer=1))
    explicit = loader(TextData(source=str(tmp_path), seq_len=seq_len, stride=seq_len), batch=batch,
        read="memory", seed=17, loading=Loading(workers=0, threads=1, read_buffer=1, worker_buffer=1))
    overlap = loader(TextData(source=str(tmp_path), seq_len=seq_len, stride=1), batch=batch, read="memory",
        seed=17, loading=Loading(workers=0, threads=1, read_buffer=1, worker_buffer=1))
    assert overlap.records == 13 and overlap.steps_per_epoch == 3
    assert default.records == explicit.records == 4
    assert default.steps_per_epoch == explicit.steps_per_epoch == 1
    assert overlap.epoch_steps(3) == 9

    def first_rows(data, steps):
        stream = data.train(DataPartition())
        try:
            return np.concatenate([next(stream)["text"] for _ in range(steps)])
        finally:
            stream.close()

    assert first_rows(default, 5).tobytes() == first_rows(explicit, 5).tobytes()
    assert first_rows(default, 5).tobytes() == first_rows(positional, 5).tobytes()
    rows = first_rows(overlap, overlap.steps_per_epoch + 1)
    assert rows.dtype == np.int32
    assert sorted(rows[:overlap.records, 0]) == list(range(17, 30)), "one overlapping epoch"
    assert rows.tobytes() == first_rows(overlap, overlap.steps_per_epoch + 1).tobytes(), "same seed and order"
    for data in (default, explicit, overlap):
        validation = data.val(DataPartition())
        try:
            actual = np.concatenate([batch["text"] for batch in validation])
        finally:
            validation.close()
        expected = np.stack([np.arange(start, start + seq_len + 1, dtype=np.int32)
                             for start in range(0, 16, seq_len)])
        assert actual.tobytes() == expected.tobytes()
        np.testing.assert_array_equal(actual[:-1, -1], actual[1:, 0])


def test_token_window_stride_round_trips_and_old_records_keep_the_default(tmp_path):
    from dew.config import RunConfig

    _token_dir(tmp_path, train_tokens=17, val_tokens=17, body=np.arange(34))
    recorded = RunConfig(data=TextData(source=str(tmp_path), seq_len=4, stride=1)).to_dict()
    loaded = RunConfig.from_dict(recorded)
    assert loaded.data.stride == 1
    assert loader(loaded.data, batch=4, read="memory", loading=Loading(workers=0)).records == 13
    older = RunConfig(data=TextData(source=str(tmp_path), seq_len=4)).to_dict()
    del older["data"]["fields"]["stride"]
    restored = RunConfig.from_dict(older)
    assert restored.data.stride is None
    source = loader(restored.data, batch=4, read="memory", loading=Loading(workers=0))
    explicit = loader(TextData(source=str(tmp_path), seq_len=4, stride=4), batch=4, read="memory",
        loading=Loading(workers=0, threads=1, read_buffer=1, worker_buffer=1))
    actual, expected = source.train(DataPartition()), explicit.train(DataPartition())
    try:
        assert source.records == explicit.records == 4
        for _ in range(3):
            assert next(actual)["text"].tobytes() == next(expected)["text"].tobytes()
    finally:
        actual.close()
        expected.close()


def test_token_loader_val_is_unshuffled_and_disjoint_from_train(tmp_path):
    seq_len = 4
    # 13*4 tokens -> 12 train windows; 9*4 -> 8 val windows.
    train_tokens = np.arange(100, 100 + 13 * seq_len, dtype=np.int64)
    val_tokens = np.arange(900, 900 + 9 * seq_len, dtype=np.int64)
    _token_dir(tmp_path, train_tokens=len(train_tokens), val_tokens=len(val_tokens))
    # _token_dir draws random tokens; overwrite with a known layout.
    (tmp_path / "train.bin").write_bytes(train_tokens.astype("<u2").tobytes())
    (tmp_path / "val.bin").write_bytes(val_tokens.astype("<u2").tobytes())

    data = loader(TextData(source=str(tmp_path), seq_len=seq_len), batch=4, read="memory",
        loading=Loading(workers=0, threads=1, read_buffer=1, worker_buffer=1))

    val_batches = [b["text"] for b in data.val(DataPartition())]
    assert len(val_batches) == 2  # 8 windows, batch 4, drop_remainder
    # Unshuffled: windows come in file order, each overlapping the next by one.
    np.testing.assert_array_equal(val_batches[0][0], val_tokens[:seq_len + 1])
    np.testing.assert_array_equal(val_batches[1][0],
                                  val_tokens[4 * seq_len:5 * seq_len + 1])

    # Train and val are disjoint files: no train window equals a val window.
    train_windows = {w.tobytes() for w in np.concatenate(
        [b["text"] for b in itertools.islice(data.train(DataPartition()), data.steps_per_epoch)])}
    val_windows = {w.tobytes() for w in np.concatenate(val_batches)}
    assert not (train_windows & val_windows)


@pytest.mark.parametrize("worker_count", [0, 2])
def test_token_loader_validation_pass_reads_every_window_once(tmp_path, worker_count):
    """A validation pass reads every window of val.bin once and ends. The
    sampler carries its own epoch count, not the training stream's unbounded
    one, and grain's DataLoader batches inside each worker, so the windows a
    worker has read do not come round again.
    """
    seq_len = 4
    val_tokens = np.arange(900, 900 + 13 * seq_len, dtype=np.int64)  # 12 windows
    _token_dir(tmp_path, train_tokens=5 * seq_len, val_tokens=len(val_tokens))
    (tmp_path / "val.bin").write_bytes(val_tokens.astype("<u2").tobytes())

    data = loader(TextData(source=str(tmp_path), seq_len=seq_len), batch=4, read="memory", seed=0,
        loading=Loading(workers=worker_count))

    windows = [list(window) for batch in itertools.islice(data.val(DataPartition()), 12)
               for window in batch["text"]]
    assert windows == [list(val_tokens[start:start + seq_len + 1])
                       for start in range(0, 12 * seq_len, seq_len)]


def test_token_loader_seeds_its_train_sampler(tmp_path):
    """The seed decides the order: the same seed reads the same first batch
    twice, a different seed reads a different one. A sampler that ignored
    the seed and shuffled afresh each call would pass the second half alone."""
    seq_len, records = 8, 24
    _token_dir(tmp_path, train_tokens=(records + 1) * seq_len, val_tokens=2 * seq_len)

    def first_batch(seed):
        data = loader(TextData(source=str(tmp_path), seq_len=seq_len), batch=4, read="memory", seed=seed,
            loading=Loading(workers=0))
        return next(data.train(DataPartition()))["text"]

    assert np.array_equal(first_batch(0), first_batch(0))
    assert not np.array_equal(first_batch(0), first_batch(1))


# ---------------------------------------------------------------------------------
# The token directory a spec reads

# ---------------------------------------------------------------------------------

# Produced token stores

def export_documents(documents, destination, *, tokenizer="byte", eos=False):
    from dew.data import Memory
    from dew.data.pipeline import Pipeline
    from dew.data.sinks import export
    from dew.data.steps.text import Tokenize
    from dew.data.text import tokenizer_for

    vocabulary = tokenizer_for(tokenizer)
    pipeline = Pipeline(Memory([{"text": text} for text in documents])).filter(
        lambda row: bool(row["text"])).map(Tokenize(tokenizer, eos=eos))
    return export(pipeline, str(destination), format="array_record", length_field="text",
                  metadata={"token_store": True, "tokenizer": tokenizer, "vocab_size": vocabulary.vocab_size,
                            "dtype": "int32", "eos_id": vocabulary.eos_id if eos else None})


def test_written_tokens_round_trip_through_the_source(tmp_path):
    corpus = "\n".join(f"document {index}: the quick brown fox jumps over the lazy dog, ünïcödé {index}"
                       for index in range(40)) + "\n"
    documents = [corpus * 5, corpus * 3]
    destination = tmp_path / "tokens"
    manifest = export_documents(documents, destination)
    stored = TokenFiles(str(destination))
    assert stored.metadata()["tokenizer"] == "byte" and stored.metadata()["vocab_size"] == 256
    expected = ByteTokenizer().encode("".join(documents))
    np.testing.assert_array_equal(stored.tokens[:], expected)
    assert stored.metadata()["train_tokens"] == len(expected)
    data = loader(TextData(source=stored, seq_len=32), batch=1, read="memory", loading=Loading(workers=0))
    assert data.records == (len(expected) - 1) // 32 and manifest["rows"] == 2
    stream = data.train(DataPartition())
    try:
        rows = _read(stream, data.records)
    finally:
        stream.close()
    assert sorted(tuple(row["text"]) for row in rows) == sorted(
        tuple(expected[start:start + 33]) for start in range(0, len(expected) - 32, 32))


def test_documents_write_one_eos_each_and_empty_documents_write_nothing(tmp_path):
    documents = ["alpha beta", "gamma", "", "delta epsilon zeta"]
    export_documents(documents, tmp_path, eos=True)
    stored = TokenFiles(str(tmp_path))
    eos = ByteTokenizer().eos_id
    expected = [token for text in documents if text for token in [*text.encode(), eos]]
    np.testing.assert_array_equal(stored.tokens[:], expected)
    assert stored.metadata()["eos_id"] == eos


def _export_cli(input_path, destination):
    import subprocess

    fields = json.dumps({"source": {"class": "dew.data.sources.files:Files",
                                   "fields": {"url": str(input_path), "format": "text"}}})
    return subprocess.run([sys.executable, "-m", "dew.cli.main", "data", "export", "records",
                           str(destination), "--fields", fields, "--format", "array_record",
                           "--tokenizer", "byte", "--read", "memory"], cwd=destination.parent,
                          capture_output=True, text=True,
                          env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "src"), "JAX_PLATFORMS": "cpu"},
                          check=False)


@pytest.mark.parametrize("missing", [False, True])
def test_data_export_leaves_every_file_it_did_not_create(tmp_path, missing):
    destination = tmp_path / "out"
    destination.mkdir()
    sentinel = destination / "all.bin"
    sentinel.write_bytes(b"unrelated bytes \x00\xff" * 64)
    before = sentinel.read_bytes()
    raw = tmp_path / "corpus.txt"
    raw.write_text("one line\nanother line\n" * 20, encoding="utf-8")
    source = tmp_path / "missing.txt" if missing else raw
    result = _export_cli(source, destination)
    assert (result.returncode == 0) == (not missing), result.stderr[-2000:]
    assert sentinel.read_bytes() == before
    if not missing:
        np.testing.assert_array_equal(TokenFiles(str(destination)).tokens[:],
            ByteTokenizer().encode(raw.read_text()))


def test_an_input_inside_the_export_directory_is_read_and_left_alone(tmp_path):
    destination = tmp_path / "out"
    destination.mkdir()
    inside = destination / "all.bin"
    text = "a corpus whose file name happens to be all.bin\n" * 30
    inside.write_text(text, encoding="utf-8")
    result = _export_cli(inside, destination)
    assert result.returncode == 0, result.stderr[-2000:]
    assert inside.read_text(encoding="utf-8") == text
    np.testing.assert_array_equal(TokenFiles(str(destination)).tokens[:], ByteTokenizer().encode(text))


# ---------------------------------------------------------------------------------
# Packed documents
# ---------------------------------------------------------------------------------

@pytest.mark.parametrize("read", ["indexed", "streamed", "memory"])
@pytest.mark.parametrize("eos_id,ends,segments", [
    (0, [4, 7, 9], [1, 1, 1, 1, 2, 2, 2, 3, 3]),
    (9, [9], [1] * 9),
])
def test_a_document_runs_through_its_eos_and_the_tail_past_the_last_is_one_too(
        tmp_path, read, eos_id, ends, segments):
    stream = np.asarray([10, 11, 12, 0, 20, 21, 0, 30, 31], np.uint16)
    write_token_corpus(tmp_path, stream, stream, eos_id=eos_id)
    stored = TokenFiles(str(tmp_path))
    np.testing.assert_array_equal(stored.document_ends(), ends)
    np.testing.assert_array_equal(stored.lengths(), np.diff(ends, prepend=0))
    records = stored.records(read)
    assert [records[index]["text"].tolist() for index in range(len(records))] == [
        stream[start:stop].tolist() for start, stop in zip([0, *ends[:-1]], ends, strict=True)]
    data = loader(TextData(source=stored, pack="first-fit", seq_len=15, packing_bins=1),
                  batch=1, read=read, loading=Loading(workers=0, threads=1, read_buffer=1))
    (window,) = _read(data.val(DataPartition()))
    assert window["text"].tolist() == [*stream, *[0] * 7]
    assert window["text_segment_ids"].tolist() == [*segments, *[0] * 7]


def test_a_store_without_a_document_index_packs_its_documents_by_one_planned_order(tmp_path):
    """The old 65,536-token boundary must not cut a document or change the packed windows."""
    documents = [[10 + index] * (1100 + index % 5) for index in range(60)]
    directory, tokens = _document_dir(tmp_path / "store", documents)
    assert len(tokens) > 65536
    provisioned = shutil.copytree(directory, tmp_path / "provisioned")
    TokenFiles(str(provisioned)).provision()
    for read in ("indexed", "streamed", "memory"):
        found = TokenFiles(str(directory)).records(read)
        held = TokenFiles(str(provisioned)).records(read)
        assert len(found) == len(documents) == len(held)
        for index in range(len(documents)):
            assert found[index]["text"].tolist() == held[index]["text"].tolist() == [*documents[index], 0]
    loading = Loading(workers=0, threads=1, read_buffer=1, worker_buffer=1)
    recipe = TextData(source=str(directory), pack="first-fit", seq_len=4095, packing_bins=2)
    loaded = {read: loader(recipe, batch=2, read=read, loading=loading)
              for read in ("indexed", "memory")}
    assert all(data.plan.lowering == "global" for data in loaded.values())
    assert len({data.records for data in loaded.values()}) == 1
    rows = {read: sorted(tuple(tuple(row[key].tolist()) for key in
                              ("text", "text_segment_ids", "text_positions"))
                         for row in _read(data.train(DataPartition()), data.steps_per_epoch))
            for read, data in loaded.items()}
    assert rows["indexed"] == rows["memory"]
    streamed = loader(recipe, batch=2, read="streamed", loading=loading)
    assert rows["indexed"] == sorted(tuple(tuple(row[key].tolist()) for key in
                                          ("text", "text_segment_ids", "text_positions"))
                                    for row in _read(streamed.val(DataPartition())))
    assert not (directory / "train.documents.npy").exists()


@pytest.mark.parametrize("read", ["indexed", "streamed", "memory"])
def test_a_store_without_an_eos_id_is_one_document_windows_read_and_only_packing_refuses(tmp_path, read):
    stream = np.arange(65555, dtype=np.uint32)
    write_token_corpus(tmp_path, stream, stream)
    stored = TokenFiles(str(tmp_path))
    assert not stored.documented and stored.document_ends() is None
    records = stored.records(read)
    assert len(records) == 1
    np.testing.assert_array_equal(records[0]["text"], stream)
    loading = Loading(workers=0, threads=1, read_buffer=1, worker_buffer=1)
    data = loader(TextData(source=stored, seq_len=8), batch=1, read=read, loading=loading)
    assert data.records == (None if read == "streamed" else (len(stream) - 1) // 8)
    assert [row["text"].tolist() for row in _read(data.val(DataPartition()), 2)] == [
        stream[start:start + 9].tolist() for start in (0, 8)]
    with pytest.raises(ValueError, match="producer document boundaries or EOS"):
        loader(TextData(source=stored, pack="first-fit", seq_len=8), batch=1, read=read, loading=loading)


def test_a_memory_records_read_keeps_documents_lazy_after_one_boundary_scan(tmp_path, monkeypatch):
    stream = np.asarray([10, 11, 12, 0, 20, 21, 0, 30, 31], np.uint16)
    write_token_corpus(tmp_path, stream, eos_id=0)
    stored = TokenFiles(str(tmp_path))
    spans = []
    read = np.memmap.__getitem__

    def recording(self, span):
        spans.append((span.start, span.stop))
        return read(self, span)

    monkeypatch.setattr(np.memmap, "__getitem__", recording)
    assert stored.documented and not spans
    ends = stored.document_ends()
    for mode in ("indexed", "streamed", "memory"):
        assert len(stored.records(mode)) == 3
    assert stored.document_ends() is ends
    np.testing.assert_array_equal(stored.lengths(), [4, 3, 2])
    assert spans == [(0, len(stream))]


def test_a_document_index_is_derived_without_changing_the_local_content_identity(tmp_path):
    directory, _ = _document_dir(tmp_path / "store", [[10, 11], [20, 21, 22]])
    before = TokenFiles(str(directory)).fingerprint()
    TokenFiles(str(directory)).provision()
    provisioned = TokenFiles(str(directory)).fingerprint()
    copied = shutil.copytree(directory, tmp_path / "other-mount" / "tokens")
    assert before == provisioned == TokenFiles(str(copied)).fingerprint()
    assert not {"documents", "chunk_tokens", "url"} & before.keys()


def test_a_producer_document_index_allows_packing_without_an_eos_id(tmp_path):
    stream = np.asarray([10, 11, 20, 21, 22], np.uint16)
    write_token_corpus(tmp_path, stream, stream)
    for split in ("train", "val"):
        np.save(tmp_path / f"{split}.documents.npy", np.asarray([2, 5], np.int64), allow_pickle=False)
    stored = TokenFiles(str(tmp_path))
    assert stored.documented and stored.tokens.eos_id is None
    data = loader(TextData(source=stored, pack="first-fit", seq_len=7), batch=1, read="indexed",
                  loading=Loading(workers=0, threads=1, read_buffer=1))
    (window,) = _read(data.val(DataPartition()))
    assert window["text"].tolist() == [*stream, 0, 0, 0]
    assert window["text_segment_ids"].tolist() == [1, 1, 2, 2, 2, 0, 0, 0]


def test_packed_loader_fills_windows_with_whole_documents(tmp_path):
    seq_len = 8  # windows of 9 ids
    # 4, 6 and 5 ids once each eos is counted: first fit puts 4 + 5 in one
    # window and 6 in the next.
    _document_dir(tmp_path, [[10, 11, 12], [20, 21, 22, 23, 24], [30, 31, 32, 33]])
    data = loader(TextData(source=str(tmp_path), pack="first-fit", seq_len=seq_len, packing_bins=2), batch=2,
        read="memory", loading=Loading(workers=0, threads=1, read_buffer=1, worker_buffer=1))

    # Two windows hold the three documents, and records is that pass exactly.
    assert data.records == 2 and data.batch == 2 and data.steps_per_epoch == 1
    batch = next(data.val(DataPartition()))

    for key in ("text", "text_segment_ids", "text_positions"):
        assert batch[key].shape == (2, seq_len + 1)
        assert batch[key].dtype == np.int32
    np.testing.assert_array_equal(batch["text"], [
        [10, 11, 12, 0, 30, 31, 32, 33, 0],
        [20, 21, 22, 23, 24, 0, 0, 0, 0]])
    np.testing.assert_array_equal(batch["text_segment_ids"], [
        [1, 1, 1, 1, 2, 2, 2, 2, 2],
        [1, 1, 1, 1, 1, 1, 0, 0, 0]])
    # Positions restart at 0 inside every document; RoPE reads them.
    np.testing.assert_array_equal(batch["text_positions"], [
        [0, 1, 2, 3, 0, 1, 2, 3, 4],
        [0, 1, 2, 3, 4, 5, 0, 0, 0]])


def test_packed_loader_cuts_documents_that_outgrow_the_window(tmp_path):
    """Grain's packer refuses an over-long element, so the loader cuts first;
    each piece is its own segment and its positions start again at 0."""
    seq_len = 3  # windows of 4 ids
    _document_dir(tmp_path, [list(range(10, 19))])  # one 10-id document
    data = loader(TextData(source=str(tmp_path), pack="first-fit", seq_len=seq_len, packing_bins=1), batch=1,
        read="memory", loading=Loading(workers=0, threads=1, read_buffer=1, worker_buffer=1))

    rows = [batch["text"][0] for batch in data.val(DataPartition())]
    positions = [batch["text_positions"][0] for batch in data.val(DataPartition())]
    assert len(rows) == 3  # ceil(10 / 4) pieces, one per window
    np.testing.assert_array_equal(np.concatenate(rows)[:10],
                                  [*list(range(10, 19)), 0])
    np.testing.assert_array_equal(positions[0], [0, 1, 2, 3])


def test_a_store_packs_its_documents_whole_unless_a_chunk_step_cuts_them(tmp_path):
    """A store's planned pack places each document whole and refuses one longer than a
    window; a chunk step before the pack cuts it into window-long pieces, which the plan
    of the store's document lengths cuts alike, each piece a window of its own here."""
    from dew.data.compile import compile
    from dew.data.pipeline import Pipeline
    from dew.data.text import chunk

    _document_dir(tmp_path, [list(range(10, 19))])  # one 10-id document; windows of 4
    stored = TokenFiles(str(tmp_path))
    loading = Loading(workers=0, threads=1, read_buffer=1, worker_buffer=1)
    with pytest.raises(ValueError):
        compile(Pipeline(stored).pack(4, bins=1).repeat().batch(1), read="indexed", provision=True,
                loading=loading)
    data = compile(Pipeline(stored).flat_map(chunk(4)).pack(4, bins=1).repeat().batch(1), read="indexed",
                   provision=True, loading=loading)
    assert data.plan.lowering == "global"
    rows = [row["text"].tolist() for row in _read(data.train(DataPartition()), 3)]
    assert rows == [[10, 11, 12, 13], [14, 15, 16, 17], [18, 0, 0, 0]]


def test_packed_loader_state_restores_the_next_unseen_batch(tmp_path):
    """The trainer saves the iterator's position in the checkpoint, so a
    restored iterator has to carry on where the saved one had got to."""
    _document_dir(tmp_path, [[i, i + 1, i + 2] for i in range(10, 60, 3)])
    data = loader(TextData(source=str(tmp_path), pack="first-fit", seq_len=8, packing_bins=2), batch=2,
        read="memory", loading=Loading(workers=0, threads=1, read_buffer=1, worker_buffer=1))

    iterator = data.val(DataPartition())
    next(iterator)
    state = iterator.get_state()
    expected = [next(iterator)["text"] for _ in range(2)]

    restored = data.val(DataPartition())
    restored.set_state(state)
    for wanted, got in zip(expected, [next(restored)["text"] for _ in range(2)], strict=True):
        np.testing.assert_array_equal(wanted, got)


def test_packed_loader_windows_do_not_depend_on_worker_count(tmp_path):
    _document_dir(tmp_path, [[i, i + 1, i + 2] for i in range(10, 70, 3)])

    def windows(worker_count):
        data = loader(TextData(source=str(tmp_path), pack="first-fit", seq_len=8, packing_bins=2), batch=2,
            read="memory", loading=Loading(workers=worker_count, worker_buffer=1))
        return sorted(row.tobytes() for batch in data.val(DataPartition())
                      for row in batch["text"])

    serial = windows(0)
    assert serial, "the loader produced no windows"
    assert serial == windows(2)


@pytest.mark.parametrize("document_width", [2, 3])
def test_a_packed_validation_pass_covers_the_split_once(tmp_path, document_width):
    """The documents packed for validation are read once, without the
    training stream's repeat, so the pass finishes and scores each document
    once."""
    documents = [list(range(i, i + document_width))
                 for i in range(10, 10 + 10 * document_width, document_width)]
    _document_dir(tmp_path, documents)
    data = loader(TextData(source=str(tmp_path), pack="first-fit", seq_len=8, packing_bins=2), batch=2,
        read="memory", loading=Loading(workers=0, threads=1, read_buffer=1, worker_buffer=1))

    batches, ended = _bounded(data.val(DataPartition()), 20)
    # Ten documents of three ids (the eos counts) pack three to a nine-id
    # window, so four windows and two batches of two.
    assert len(batches) == (2 if document_width == 2 else 3) and ended, ENDLESS_VAL
    read = Counter(int(token) for row in _rows(batches)
                   for token in np.frombuffer(row, batches[0]["text"].dtype) if token)
    assert read == Counter(token for document in documents for token in document)
    again, _ = _bounded(data.val(DataPartition()), len(batches))
    assert _rows(again) == _rows(batches)


def test_a_single_file_corpus_packs_when_its_val_split_holds_no_eos(tmp_path):
    """--pack closes each input file with one eos and the val split is cut off
    the head of the token stream by fraction, so a single-file corpus leaves
    val.bin with no boundary inside it: that split is one document."""
    raw = tmp_path / "corpus.txt"
    raw.write_text("the quick brown fox jumps over the lazy dog\n" * 8,
                   encoding="utf-8")
    out = tmp_path / "tokens"

    from token_support import encoded_token_fixture

    encoded_token_fixture(out, [raw.read_text()], val_fraction=0.1, eos=True)
    val_tokens = list((out / "val.bin").read_bytes())
    assert ByteTokenizer().eos_id not in val_tokens, "the split kept a boundary"

    seq_len = 63
    data = loader(TextData(source=str(out), pack="first-fit", seq_len=seq_len), batch=1, read="memory",
        loading=Loading(workers=0, threads=1, read_buffer=1, worker_buffer=1))
    row = next(data.val(DataPartition()))

    padding = seq_len + 1 - len(val_tokens)
    np.testing.assert_array_equal(row["text"][0], val_tokens + [0] * padding)
    np.testing.assert_array_equal(row["text_segment_ids"][0],
                                  [1] * len(val_tokens) + [0] * padding)
    np.testing.assert_array_equal(row["text_positions"][0, :len(val_tokens)],
                                  np.arange(len(val_tokens)))


# ---------------------------------------------------------------------------------
# Stream termination on the token paths
# ---------------------------------------------------------------------------------

# A validation pass that never ends scores the same windows again and again;
# the tests carrying this reason are the contract a validation stream meets.
ENDLESS_VAL = "an endless validation pass"


def _bounded(loader, limit):
    """At most `limit` batches, and whether the stream ended inside them.

    Bounded on purpose: an endless stream then fails a count instead of
    hanging the suite.
    """
    iterator = iter(loader)
    taken = list(itertools.islice(iterator, limit))
    return taken, next(iterator, None) is None


def _rows(batches):
    """The real rows of `batches`, without the repeats that fill a pass's last batch out."""
    return [row.tobytes() for batch in batches
            for row in np.asarray(batch["text"])[batch.get(VALID_ROWS, np.ones(len(batch["text"]), bool))]]


@pytest.mark.parametrize("windows", [8, 10])
def test_a_token_validation_pass_reads_every_window_once(tmp_path, windows):
    """Ten windows at batch four are three batches, in file order, then the end.

    Every batch keeps the configured rows so its shape fits the device mesh:
    the two windows past the last full batch come in a third, filled out with
    repeats of them that `VALID_ROWS` marks.
    """
    seq_len = 4
    val_tokens = np.arange(900, 900 + (windows + 1) * seq_len, dtype=np.int64)
    _token_dir(tmp_path, train_tokens=13 * seq_len, val_tokens=len(val_tokens))
    (tmp_path / "val.bin").write_bytes(val_tokens.astype("<u2").tobytes())
    data = loader(TextData(source=str(tmp_path), seq_len=seq_len), batch=4, read="memory",
        loading=Loading(workers=0, threads=1, read_buffer=1, worker_buffer=1))

    batches, ended = _bounded(data.val(DataPartition()), 4)

    assert len(batches) == (windows + 3) // 4 and ended, ENDLESS_VAL
    assert len(_rows(batches)) == len(set(_rows(batches))) == windows, "a pass reads every window once"
    if windows == 10:
        np.testing.assert_array_equal(batches[-1][VALID_ROWS], [True, True, False, False])
    np.testing.assert_array_equal(batches[0]["text"][0], val_tokens[:seq_len + 1])


def test_the_token_training_stream_repeats_rather_than_ending(tmp_path):
    """The trainer keeps asking long after one pass over the windows, and the
    fixed-window loader is the one path that has no `repeat` of its own."""
    seq_len = 4
    _token_dir(tmp_path, train_tokens=9 * seq_len, val_tokens=5 * seq_len)
    data = loader(TextData(source=str(tmp_path), seq_len=seq_len), batch=4, read="memory",
        loading=Loading(workers=0, threads=1, read_buffer=1, worker_buffer=1))

    epoch = data.records // 4
    batches, ended = _bounded(data.train(DataPartition()), 3 * epoch)

    assert data.records == 8 and not ended
    assert len(batches) == 3 * epoch
    assert len(set(_rows(batches[:epoch]))) == 8, "one pass reads distinct windows"


# ---------------------------------------------------------------------------------
# Packed windows under adversity: the mask the backbone builds, the loss it counts
# ---------------------------------------------------------------------------------

# Documents are closed by this id and padding is 0, so a row's documents can be
# read back off the ids alone, without asking the segment ids the test is there
# to check.
PACK_EOS = 1


def _packed(tmp_path, documents, seq_len, batch=1, bins=4):
    """One validation pass over `documents` packed into `seq_len + 1` windows."""
    stream = np.concatenate([np.asarray([*d, PACK_EOS], np.int64) for d in documents])
    _token_dir(tmp_path, train_tokens=0, body=stream, eos_id=PACK_EOS)
    (tmp_path / "val.bin").write_bytes(stream.astype(np.uint16).tobytes())
    return list(loader(TextData(source=str(tmp_path), validation=TokenFiles(str(tmp_path), split="val"),
        pack="first-fit", seq_len=seq_len, packing_bins=bins),
        batch=batch, read="memory", loading=Loading(workers=0, threads=1, read_buffer=1,
            worker_buffer=1)).val(DataPartition()))


def _tiny_backbone(seq_len):
    return backbone.CausalTransformer(vocab_size=64, emb_features=16, num_layers=1,
                                      num_heads=2, mlp_features=32,
                                      max_seq_len=seq_len + 1)


def _documents_in(row):
    """(start, stop) of each document in a packed row, read off the ids.

    A span closes on the boundary id, or on the padding that follows the last
    one: a chunk cut out of an over-long document carries no boundary of its
    own and is still a segment.
    """
    spans, start = [], 0
    for index, token in enumerate(row):
        if token == 0:
            break
        if token == PACK_EOS:
            spans.append((start, index + 1))
            start = index + 1
    else:
        index = len(row)
    if start < index:
        spans.append((start, index))
    return spans


@pytest.mark.parametrize("name", ["bin", "records", "packed_records"])
@pytest.mark.parametrize("pack", [None, "first-fit"])
def test_memory_windows_and_packing_never_reread_the_store(stores, monkeypatch, name, pack):
    import grain.python as grain

    root, fields = stores
    stored = TokenFiles(str(root / name), field=fields[name]["field"])
    recipe = TextData(source=stored, pack=pack, seq_len=7, packing_bins=2)
    train, _, _ = recipe.pipelines()
    if pack is not None:
        from dew.data.pipeline import FlatMap, Pack

        assert train.source is stored and [type(stage) for stage in train.stages] == [FlatMap, Pack]
    data = loader(recipe,
                  batch=2, read="memory",
                  loading=Loading(workers=0, threads=1, read_buffer=1, worker_buffer=1))

    def reread(self, index):
        pytest.fail("a materialized memory read went back to its token store")

    monkeypatch.setattr(np.memmap, "__getitem__", reread)
    monkeypatch.setattr(grain.ArrayRecordDataSource, "__getitem__", reread)
    stream = data.train(DataPartition())
    try:
        next(stream)
    finally:
        stream.close()
    stream = data.train(DataPartition())
    try:
        windows = list(itertools.islice(stream, 10))
    finally:
        stream.close()
    assert all(batch["text"].shape == (2, 8) for batch in windows)


def _counted_cross_entropy(batch, seq_len):
    """The objective's ce, and the same number computed by hand.

    The hand version averages the per-token losses over the transitions that
    live inside one document, which is every target a packed row owns.
    """
    model = _tiny_backbone(seq_len)
    objective = LMObjective(model, seq_len)
    params = objective.init(jax.random.key(0))
    tokens = jnp.asarray(batch["text"], jnp.int32)
    segment_ids = jnp.asarray(batch["text_segment_ids"], jnp.int32)
    positions = jnp.asarray(batch["text_positions"], jnp.int32)

    ce, _ = objective.scalar_loss(params, batch, Step(step=jnp.zeros((), jnp.int32),
                                                key=jax.random.key(1), ema=None))

    logits = model.apply(params, tokens[:, :-1], positions=positions[:, :-1],
                         segment_ids=segment_ids[:, :-1])
    losses = np.asarray(optax.softmax_cross_entropy_with_integer_labels(
        logits.astype(jnp.float32), tokens[:, 1:]))

    rows = np.asarray(batch["text"])
    total, counted = 0.0, 0
    for row in range(rows.shape[0]):
        for start, stop in _documents_in(rows[row]):
            total += losses[row, start:stop - 1].sum()
            counted += stop - 1 - start
    return float(ce), total / counted, counted


def test_the_loss_counts_one_target_per_transition_inside_a_document(tmp_path):
    """Three documents in a row own six targets between them: the two boundary
    transitions and the padding are not the model's to predict."""
    batches = _packed(tmp_path, [[2, 3], [4, 5], [6, 7]], seq_len=8, bins=1)

    ce, by_hand, counted = _counted_cross_entropy(batches[0], seq_len=8)

    assert counted == 6
    assert ce == pytest.approx(by_hand, rel=1e-5)


# ---------------------------------------------------------------------------------
# Worker counts and restarts on the token paths
# ---------------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.parametrize("worker_count", [1, 2, 4])
def test_token_windows_are_the_same_records_at_every_worker_count(tmp_path,
                                                                  worker_count):
    """Real worker processes, one epoch: the windows a pass reads are a
    function of the seed and the file, not of how many workers read them."""
    seq_len = 8
    _token_dir(tmp_path, train_tokens=17 * seq_len, val_tokens=2 * seq_len)

    def windows(workers):
        data = loader(TextData(source=str(tmp_path), seq_len=seq_len), batch=4, read="memory", seed=7,
            loading=Loading(workers=workers))
        return sorted(row.tobytes()
                      for batch in itertools.islice(data.train(DataPartition()), data.steps_per_epoch)
                      for row in batch["text"])

    serial = windows(0)

    assert len(serial) == 16
    assert len(set(serial)) == 16
    assert windows(worker_count) == serial


@pytest.mark.slow
def test_an_interrupted_packed_epoch_resumes_through_mp_prefetch(tmp_path):
    """The packed loader reads its documents in worker processes and packs
    behind them, so a restart has to carry the packer's position too."""
    _document_dir(tmp_path, [[i, i + 1, i + 2] for i in range(10, 100, 3)])

    def opened():
        return loader(TextData(source=str(tmp_path), pack="first-fit", seq_len=8, packing_bins=2), batch=2,
            read="memory", loading=Loading(workers=2, worker_buffer=1))

    interrupted = opened().val(DataPartition())
    seen = _rows([next(interrupted)])
    state = interrupted.get_state()
    rest, ended = _bounded(interrupted, 40)
    unseen = _rows(rest)

    restored = opened().val(DataPartition())
    restored.set_state(state)
    after, ended_again = _bounded(restored, 40)
    resumed = _rows(after)

    assert unseen and ended and ended_again
    assert resumed == unseen
    assert len(set(seen + resumed)) == len(seen) + len(unseen), "a window came twice"


# ---------------------------------------------------------------------------------
# One corpus, three stores
# ---------------------------------------------------------------------------------

def _chunks(tokens, sizes):
    """`tokens` cut into consecutive pieces of `sizes`, the last taking the rest."""
    edges = np.cumsum([0, *sizes])
    pieces = [tokens[start:stop] for start, stop in itertools.pairwise(edges)]
    return [*pieces, tokens[edges[-1]:]]


def _as_records(directory, split, tokens, sizes, field=None):
    """`tokens` as arrayrecord shards, cut into records of `sizes`."""
    from array_record.python.array_record_module import ArrayRecordWriter

    from dew.data.sources.codec import pack_dict_of_byte_arrays

    paths = []
    for shard, piece in enumerate(_chunks(tokens, sizes)):
        path = str(directory / f"{split}-{shard:02d}.array_record")
        writer = ArrayRecordWriter(path, "group_size:1")
        raw = piece.astype(np.uint16).tobytes()
        writer.write(raw if field is None else pack_dict_of_byte_arrays({field: raw}))
        writer.close()
        paths.append(path)
    return paths


CORPUS = np.concatenate([np.asarray([*document, PACK_EOS], np.int64) for document in
                         ([10, 11, 12], [20, 21], [30, 31, 32, 33, 34], [40], [50, 51])])


@pytest.fixture
def stores(tmp_path):
    """The same corpus in every store, each split cut differently in each,
    as the fields of a `Text` over it.

    The pieces are deliberately unequal, so a window that crosses a record
    boundary has to be joined out of two of them; identical windows then say
    the stream view is the stream, not the chunking.
    """
    meta = {"tokenizer": "byte", "vocab_size": 256, "dtype": "uint16", "train_tokens": len(CORPUS),
            "val_tokens": len(CORPUS), "eos_id": PACK_EOS}
    held = {}
    stores = (("bin", None, None), ("records", [4, 3], None), ("packed_records", [2, 9], "ids"))
    for name, sizes, field in stores:
        directory = tmp_path / name
        directory.mkdir()
        (directory / "meta.json").write_text(json.dumps(meta))
        for split in ("train", "val"):
            if sizes is None:
                (directory / f"{split}.bin").write_bytes(CORPUS.astype(np.uint16).tobytes())
            else:
                _as_records(directory, split, CORPUS, sizes, field=field)
        held[name] = {"field": field}
    return tmp_path, held


def test_the_same_corpus_gives_the_same_windows_and_packing_in_every_store(stores):
    root, fields = stores

    def validation(name, **spec):
        recipe = TextData(source=TokenFiles(str(root / name), field=fields[name]["field"]), **spec)
        data = loader(recipe, batch=1, read="memory",
            loading=Loading(workers=0, threads=1, read_buffer=1, worker_buffer=1))
        rows = _read(data.val(DataPartition()))
        return [{key: value.tolist() for key, value in row.items()} for row in rows]

    windows = {name: validation(name, seq_len=4) for name in fields}
    packed = {name: validation(name, seq_len=5, pack="first-fit", packing_bins=2) for name in fields}
    assert [row["text"] for row in windows["bin"]] == [
        CORPUS[start:start + 5].tolist() for start in range(0, len(CORPUS) - 4, 4)]
    lengths = [count for row in packed["bin"]
               for segment, count in Counter(row["text_segment_ids"]).items() if segment]
    assert sorted(lengths) == [2, 3, 3, 4, 6], "the documents are the eos spans"
    for name in fields:
        assert windows[name] == windows["bin"], name
        assert packed[name] == packed["bin"], name


def test_a_hub_tokenizer_runs_the_revision_its_identity_names(monkeypatch, tmp_path):
    """Tokenize resolves a hub name once: its identity hashes the files of the
    resolved revision and its encoding reads those same files, here and in a
    worker it is pickled to, never the name again."""
    import pickle

    import huggingface_hub

    from dew.data.steps.text import Tokenize
    from dew.data.text import load_tokenizer

    # The Hub cache keeps a commit's files in a directory named for it.
    files = tmp_path / "snapshots" / ("a" * 40)
    shutil.copytree(Path(__file__).parent / "fixtures" / "tokenizers" / "gpt2", files)
    resolved = []

    def download(repo, revision, allow_patterns, local_files_only):
        resolved.append((repo, revision))
        return str(files)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    step = Tokenize("acme/tokenizer")
    assert step.dependencies()["tokenizer"]["revision"] == "a" * 40
    sent = pickle.loads(pickle.dumps(step))
    encoded = [int(token) for token in sent.map({"text": "hello there"})["text"]]
    assert encoded == load_tokenizer(str(files), local_files_only=True).encode("hello there")
    assert resolved == [("acme/tokenizer", None)]


def test_hosts_that_mount_one_corpus_at_different_paths_read_one_order(tmp_path):
    """Each host of a pool reads its own copy of a token corpus, mounted where it was put: the
    copies' readers agree on what they read, and a copy that holds other tokens does not."""
    stream = (np.arange(97, dtype=np.uint16) * 7) % 32
    first = write_token_corpus(tmp_path / "host0" / "tokens", stream, stream)
    second = tmp_path / "elsewhere" / "copy"
    shutil.copytree(first, second)
    other = write_token_corpus(tmp_path / "host2" / "tokens", stream[::-1].copy(), stream)

    def identity(directory):
        data = loader(TextData(source=str(directory), seq_len=4), batch=4, loading=Loading(workers=0))
        reader = data.train(DataPartition())
        try:
            return reader.identity()
        finally:
            reader.close()

    assert identity(first) == identity(second) != identity(other)


def test_a_same_size_write_between_token_samples_changes_the_readers_identity(tmp_path):
    """An exact reader identifies every token, including the middle of a large file."""
    stream = np.full(8 * 2**20, 7, dtype=np.uint16)
    directory = write_token_corpus(tmp_path / "tokens", stream)

    def reader():
        data = loader(TextData(source=str(directory), seq_len=1024), batch=1, read="indexed",
                      loading=Loading(workers=0))
        return data.train(DataPartition())

    before = reader()
    try:
        identity = before.identity()
        np.testing.assert_array_equal(next(before)["text"], np.full((1, 1025), 7, dtype=np.int32))
        position = before.get_state()
    finally:
        before.close()
    path = directory / "train.bin"
    size = path.stat().st_size
    with path.open("r+b") as file:
        file.seek(size // 2)
        file.write(np.asarray([9], dtype=np.uint16).tobytes())
    assert path.stat().st_size == size
    after = reader()
    try:
        assert after.identity() != identity
        with pytest.raises(ValueError):
            after.set_state(position)
    finally:
        after.close()
