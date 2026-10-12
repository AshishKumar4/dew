"""Text data for language models: tokenizers, the token-file source, the loader.

ByteTokenizer and the token sources are pure numpy; the HF tokenizer tests only
run when a cached copy of the hub is reachable, matching the repo's policy
that no test needs the network.
"""

import itertools
import json
import os
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

from dew.data import ByteTokenizer, DataPartition, Loading, TokenCorpus, TokenWindows
from dew.nn.backbones import causal_transformer as backbone
from dew.objectives.base import VALID_ROWS
from dew.objectives.lm import LMObjective
from dew.training import Step

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
    data = _windows(tmp_path, seq_len=seq_len, stride=stride).load(batch=1)
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
    data = _windows(tmp_path, seq_len=seq_len).load(batch=1)
    assert _read(data.val(DataPartition()))[0]["text"].tolist() == tokens[:seq_len + 1].tolist()

    bare = tmp_path / "bare"
    bare.mkdir()
    for split in ("train", "val"):
        (bare / f"{split}.bin").write_bytes(tokens.astype("<u2").tobytes())
    data = _windows(bare, seq_len=seq_len).load(batch=1)
    assert _read(data.val(DataPartition()))[0]["text"].tolist() == tokens[:seq_len + 1].astype("<u2").tolist()


# ---------------------------------------------------------------------------------
# TokenWindows
# ---------------------------------------------------------------------------------

def _windows(tmp_path, **fields):
    """The fixed-window spec over `tmp_path`, read in this process."""
    fields = {"val_batches": None,
              "loading": Loading(workers=0, threads=1, read_buffer=1, worker_buffer=1),
              **fields}
    return TokenWindows(path=str(tmp_path), **fields)


def _packed_tokens(tmp_path, **fields):
    """The packed spec over `tmp_path`, read in this process."""
    fields = {"val_batches": None, "loading": Loading(workers=0, worker_buffer=1), **fields}
    return TokenWindows(path=str(tmp_path), pack=True, **fields)

def test_token_loader_stride_changes_training_windows_only(tmp_path):
    seq_len, batch = 4, 4
    _token_dir(tmp_path, train_tokens=17, val_tokens=17, body=np.arange(34))
    default = _windows(tmp_path, seq_len=seq_len, seed=17).load(batch=batch)
    positional = TokenWindows(
        str(tmp_path),
        seq_len,
        None,
        seed=17,
        loading=Loading(workers=0, threads=1, read_buffer=1, worker_buffer=1),
    ).load(batch=batch)
    explicit = _windows(tmp_path, seq_len=seq_len, stride=seq_len, seed=17).load(batch=batch)
    overlap = _windows(tmp_path, seq_len=seq_len, stride=1, seed=17).load(batch=batch)
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
    recorded = RunConfig(data=_windows(tmp_path, seq_len=4, stride=1)).to_dict()
    loaded = RunConfig.from_dict(recorded)
    assert loaded.data.stride == 1
    assert loaded.data.load(batch=4).records == 13
    older = RunConfig(data=_windows(tmp_path, seq_len=4)).to_dict()
    del older["data"]["fields"]["stride"]
    restored = RunConfig.from_dict(older)
    assert restored.data.stride is None
    source = restored.data.load(batch=4)
    explicit = _windows(tmp_path, seq_len=4, stride=4).load(batch=4)
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

    data = _windows(tmp_path, seq_len=seq_len).load(batch=4)

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

    data = _windows(tmp_path, seq_len=seq_len, seed=0, loading=Loading(workers=worker_count)).load(batch=4)

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
        data = _windows(tmp_path, seq_len=seq_len, seed=seed, loading=Loading(workers=0)).load(batch=4)
        return next(data.train(DataPartition()))["text"]

    assert np.array_equal(first_batch(0), first_batch(0))
    assert not np.array_equal(first_batch(0), first_batch(1))


# ---------------------------------------------------------------------------------
# The token directory a spec reads

# ---------------------------------------------------------------------------------
# TokenCorpus.write and `dew tokenize`
# ---------------------------------------------------------------------------------

def test_written_tokens_round_trip_through_the_source(tmp_path):
    corpus = "\n".join(
        f"document {i}: the quick brown fox jumps over the lazy dog — ünïcödé {i}"
        for i in range(40)) + "\n"
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "a.txt").write_text(corpus * 5, encoding="utf-8")
    (raw / "nested").mkdir()
    (raw / "nested" / "b.txt").write_text(corpus * 3, encoding="utf-8")
    out = tmp_path / "tokens"

    written = TokenCorpus.write(raw, out, tokenizer="byte", val_fraction=0.1)

    meta = TokenCorpus.read(out)
    assert meta == written
    assert meta.tokenizer == "byte"
    assert meta.vocab_size == 256
    assert meta.dtype == "uint8"
    assert meta.train_tokens + meta.val_tokens == len(
        (corpus * 8).encode("utf-8"))

    seq_len = 32
    data = _windows(out, seq_len=seq_len).load(batch=1)
    assert data.records == (meta.train_tokens - 1) // seq_len

    tok = ByteTokenizer()
    whole = corpus * 8  # a.txt (5x) then nested/b.txt (3x), in path order
    whole_ids = tok.encode(whole)

    # The two files are a lossless partition of the corpus, val its head.
    val_bytes = list((out / "val.bin").read_bytes())
    train_bytes = list((out / "train.bin").read_bytes())
    assert tok.decode(val_bytes + train_bytes) == whole
    assert len(val_bytes) == meta.val_tokens
    assert len(train_bytes) == meta.train_tokens

    # Validation windows tile the split at stride seq_len, so stitching them
    # back rebuilds it up to the tokens past the last full window.
    windows = [row["text"] for row in _read(data.val(DataPartition()))]
    val_ids = np.concatenate([window[:seq_len] for window in windows] + [windows[-1][seq_len:]])
    assert len(val_ids) == ((meta.val_tokens - 1) // seq_len) * seq_len + 1
    assert list(val_ids) == whole_ids[:len(val_ids)]

    # And what the windows carry decodes back to that text (a window boundary
    # can split a multi-byte character, which decode replaces on both sides).
    assert tok.decode(val_ids.tolist()) == bytes(
        whole_ids[:len(val_ids)]).decode("utf-8", errors="replace")


def test_documents_in_memory_are_written_one_eos_terminated_document_each(tmp_path):
    """An iterable of strings is the route for text another library holds, a
    Hugging Face split's column included: each string is one document."""
    documents = ["alpha beta", "gamma", "", "delta epsilon zeta"]
    meta = TokenCorpus.write(iter(documents), tmp_path, tokenizer="byte", val_fraction=0.0,
                             pack=True)

    eos = ByteTokenizer().eos_id
    stream = list((tmp_path / "train.bin").read_bytes())
    expected = [byte for text in documents if text for byte in [*text.encode(), eos]]
    assert stream == expected, "an empty document writes nothing, not a lone eos"
    assert meta.val_tokens == 0 and meta.eos_id == eos


def _tokenize(input_path, out, *flags):
    """The real CLI of this checkout in a fresh process, run in `out`'s parent."""
    import subprocess

    source = Path(__file__).resolve().parents[1] / "src"
    return subprocess.run([sys.executable, "-m", "dew.cli.main", "tokenize", "--input", str(input_path),
                           "--out", str(out), *flags], cwd=Path(out).parent, capture_output=True, text=True,
                          env={**os.environ, "PYTHONPATH": str(source), "JAX_PLATFORMS": "cpu"}, check=False)


def _listing(directory):
    return sorted(path.name for path in Path(directory).iterdir())


@pytest.mark.parametrize("case", ["success", "missing input", "empty corpus"])
def test_tokenize_leaves_every_file_it_did_not_create(tmp_path, case):
    """Issue #24: the scratch stream is a file the run creates for itself.
    An unrelated `<out>/all.bin` (once the fixed scratch name) is
    byte-identical after a successful run, a missing input and a corpus too
    short to window, and the directory then holds that file plus, on
    success, the three outputs, no scratch left behind."""
    out = tmp_path / "out"
    out.mkdir()
    sentinel = out / "all.bin"
    sentinel.write_bytes(b"unrelated bytes \x00\xff" * 64)
    before = sentinel.read_bytes()
    raw = tmp_path / "corpus.txt"
    texts = {"success": "one line\nanother line\n" * 20, "empty corpus": "", "missing input": ""}
    raw.write_text(texts[case], encoding="utf-8")
    source = tmp_path / "missing.txt" if case == "missing input" else raw

    result = _tokenize(source, out, "--val-fraction", "0.2")

    assert (result.returncode == 0) == (case == "success"), result.stderr[-2000:]
    assert sentinel.read_bytes() == before
    expected = ["all.bin", "meta.json", "train.bin", "val.bin"] if case == "success" else ["all.bin"]
    assert _listing(out) == expected


def test_an_input_at_the_old_scratch_name_is_read_and_left_alone(tmp_path):
    """A corpus at `<out>/all.bin` tokenizes to what the same text gives from
    anywhere else, and is byte-identical afterwards."""
    text = "a corpus whose file name happens to be all.bin\n" * 30
    out = tmp_path / "out"
    out.mkdir()
    inside = out / "all.bin"
    inside.write_text(text, encoding="utf-8")
    elsewhere = tmp_path / "corpus.txt"
    elsewhere.write_text(text, encoding="utf-8")

    result = _tokenize(inside, out, "--val-fraction", "0.2")

    assert result.returncode == 0, result.stderr[-2000:]
    assert inside.read_text(encoding="utf-8") == text
    library = TokenCorpus.write(elsewhere, tmp_path / "lib", val_fraction=0.2)
    for name in ("train.bin", "val.bin"):
        assert (out / name).read_bytes() == (tmp_path / "lib" / name).read_bytes()
    assert json.loads((out / "meta.json").read_text())["train_tokens"] == library.train_tokens


# ---------------------------------------------------------------------------------
# Packed documents
# ---------------------------------------------------------------------------------

def test_packed_loader_fills_windows_with_whole_documents(tmp_path):
    seq_len = 8  # windows of 9 ids
    # 4, 6 and 5 ids once each eos is counted: first fit puts 4 + 5 in one
    # window and 6 in the next.
    _document_dir(tmp_path, [[10, 11, 12], [20, 21, 22, 23, 24], [30, 31, 32, 33]])
    data = _packed_tokens(tmp_path, seq_len=seq_len, packing_bins=2).load(batch=2)

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
    data = _packed_tokens(tmp_path, seq_len=seq_len, packing_bins=1).load(batch=1)

    rows = [batch["text"][0] for batch in data.val(DataPartition())]
    positions = [batch["text_positions"][0] for batch in data.val(DataPartition())]
    assert len(rows) == 3  # ceil(10 / 4) pieces, one per window
    np.testing.assert_array_equal(np.concatenate(rows)[:10],
                                  [*list(range(10, 19)), 0])
    np.testing.assert_array_equal(positions[0], [0, 1, 2, 3])


def test_packed_loader_state_restores_the_next_unseen_batch(tmp_path):
    """The trainer saves the iterator's position in the checkpoint, so a
    restored iterator has to carry on where the saved one had got to."""
    _document_dir(tmp_path, [[i, i + 1, i + 2] for i in range(10, 60, 3)])
    data = _packed_tokens(tmp_path, seq_len=8, packing_bins=2).load(batch=2)

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
        data = _packed_tokens(tmp_path, seq_len=8, loading=Loading(workers=worker_count, worker_buffer=1),
                              packing_bins=2).load(batch=2)
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
    documents = [list(range(i, i + document_width)) for i in range(10, 10 + 10 * document_width, document_width)]
    _document_dir(tmp_path, documents)
    data = _packed_tokens(tmp_path, seq_len=8, packing_bins=2).load(batch=2)

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

    TokenCorpus.write(raw, out, tokenizer="byte", val_fraction=0.1, pack=True)
    val_tokens = list((out / "val.bin").read_bytes())
    assert ByteTokenizer().eos_id not in val_tokens, "the split kept a boundary"

    seq_len = 63
    data = _packed_tokens(out, seq_len=seq_len).load(batch=1)
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
    data = _windows(tmp_path, seq_len=seq_len).load(batch=4)

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
    data = _windows(tmp_path, seq_len=seq_len).load(batch=4)

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
    return list(_packed_tokens(tmp_path, seq_len=seq_len, packing_bins=bins)
                .load(batch=batch).val(DataPartition()))


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
        data = _windows(tmp_path, seq_len=seq_len, seed=7, loading=Loading(workers=workers)).load(batch=4)
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

    def loader():
        return _packed_tokens(tmp_path, seq_len=8, loading=Loading(workers=2, worker_buffer=1),
                              packing_bins=2).load(batch=2)

    interrupted = loader().val(DataPartition())
    seen = _rows([next(interrupted)])
    state = interrupted.get_state()
    rest, ended = _bounded(interrupted, 40)
    unseen = _rows(rest)

    restored = loader().val(DataPartition())
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

    from dew.data.images import pack_dict_of_byte_arrays

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
    as the fields of a `TokenWindows` over it.

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
        data = _windows(root / name, field=fields[name]["field"], **spec).load(batch=1)
        rows = _read(data.val(DataPartition()))
        return [{key: value.tolist() for key, value in row.items()} for row in rows]

    windows = {name: validation(name, seq_len=4) for name in fields}
    packed = {name: validation(name, seq_len=5, pack=True, packing_bins=2) for name in fields}
    assert [row["text"] for row in windows["bin"]] == [
        CORPUS[start:start + 5].tolist() for start in range(0, len(CORPUS) - 4, 4)]
    lengths = [count for row in packed["bin"]
               for segment, count in Counter(row["text_segment_ids"]).items() if segment]
    assert sorted(lengths) == [2, 3, 3, 4, 6], "the documents are the eos spans"
    for name in fields:
        assert windows[name] == windows["bin"], name
        assert packed[name] == packed["bin"], name
