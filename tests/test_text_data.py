"""TextData data for language models: tokenizers, the token-file source, the loader.

ByteTokenizer and the token sources are pure numpy; the HF tokenizer tests only
run when a cached copy of the hub is reachable, matching the repo's policy
that no test needs the network.
"""

import itertools
import json
import os
import shutil
import sys
import threading
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
from dew.data import ByteTokenizer, Loading, Memory
from dew.data.compile import compile
from dew.data.dataset import DataPartition
from dew.data.pipeline import Pipeline
from dew.data.sources.files import TokenFiles
from dew.nn import attention
from dew.nn.backbones import causal_transformer as backbone
from dew.nn.mixers import attention as attention_kind
from dew.objectives.base import VALID_ROWS
from dew.objectives.lm import LMObjective
from dew.position import ENVELOPE
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

def test_byte_tokenizer_round_trips_unicode():
    tok = ByteTokenizer()
    for text in ("hello, world", "ünïcödé — π≈3.14159", "日本語のテキスト",
                 "emoji 🚀🔥 and \x00 control bytes\n\t"):
        ids = tok.encode(text)
        assert tok.decode(ids) == text


def test_byte_tokenizer_is_utf8_bytes_with_a_256_vocab():
    tok = ByteTokenizer()
    assert tok.vocab_size == 256
    assert tok.encode("A") == [0x41]
    assert tok.encode("é") == [0xC3, 0xA9]  # two utf-8 bytes, not one id
    assert tok.eos_id == 255


def test_byte_tokenizer_decode_tolerates_junk_bytes():
    # Generated ids will land on invalid utf-8 sequences; the model still
    # needs text out of them, not an exception.
    tok = ByteTokenizer()
    assert tok.decode([0xFF, 0xFE, 0x41]) == chr(0xFFFD) * 2 + "A"


# ---------------------------------------------------------------------------------
# HFTokenizer, only when the hub (or its cache) cooperates
# ---------------------------------------------------------------------------------

@pytest.mark.network
def test_hf_tokenizer_encodes_and_decodes():
    """Loads gpt2 from the hub or its cache. Marked network, so a suite run
    with `-m "not network"` deselects it; a broken HFTokenizer fails it."""
    from dew.data import HFTokenizer

    tok = HFTokenizer("gpt2")
    ids = tok.encode("hello world")
    assert 0 < len(ids) < 11
    assert tok.decode(ids) == "hello world"
    assert tok.vocab_size == 50257
    assert tok.eos_id == 50256


def test_hf_tokenizer_imports_lazily():
    """Constructing HFTokenizer must not import transformers."""
    import subprocess as sp
    probe = (
        "from dew.data import HFTokenizer;"
        "import sys;"
        "HFTokenizer('gpt2');"
        "assert 'transformers' not in sys.modules, 'imported at construction'"
    )
    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT / "src"))
    result = sp.run([sys.executable, "-c", probe], cwd=REPO_ROOT,
                    capture_output=True, text=True, env=env)
    assert result.returncode == 0, result.stderr


def test_every_reader_of_a_tokenizer_shares_one_load():
    """HFTokenizer and the caption tokenizers loader a name through one cache."""
    from dew.data import HFTokenizer
    from dew.data.text import load_tokenizer

    path = str(REPO_ROOT / "tests" / "fixtures" / "tokenizers" / "tiny-chat")
    first = HFTokenizer(path).tokenizer
    assert HFTokenizer(path).tokenizer is first
    assert load_tokenizer(path) is first


def test_readers_that_miss_the_cache_together_share_one_load(tmp_path):
    """The chat render map reads the tokenizer from every loader thread at
    once, and all of them get the one object the cache keeps."""
    from dew.data.text import load_tokenizer

    path = tmp_path / "tokenizer"
    shutil.copytree(REPO_ROOT / "tests" / "fixtures" / "tokenizers" / "tiny-chat", path)
    readers = 8
    start = threading.Barrier(readers)
    loaded = [None] * readers

    def read(index: int) -> None:
        start.wait()
        loaded[index] = load_tokenizer(str(path), local_files_only=True)

    threads = [threading.Thread(target=read, args=(index,)) for index in range(readers)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert all(tokenizer is loaded[0] for tokenizer in loaded)


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


def test_a_window_needs_a_positive_stride_and_length_and_a_long_enough_split(tmp_path):
    _stream_dir(tmp_path, np.zeros(4))
    for fields, refusal in (({"seq_len": 2, "stride": 0}, "stride"), ({"seq_len": 2, "stride": -1}, "stride"),
                            ({"seq_len": 0}, "seq_len"), ({"seq_len": 4}, "has no accepted rows")):
        with pytest.raises(ValueError, match=refusal):
            loader(TextData(source=str(tmp_path)), batch=1, read="memory", loading=Loading(workers=0,
                threads=1, read_buffer=1, worker_buffer=1), **fields)


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
# TextData
# ---------------------------------------------------------------------------------

def test_token_loader_yields_int32_batches_with_one_overlap_token(tmp_path):
    """Window k is tokens [k seq_len, k seq_len + seq_len], so consecutive
    windows share one token: the last of one is the first of the next, the
    target of the last position and the input of the next window's first.
    The stream is a counter, so a row says which tokens it holds."""
    seq_len, batch = 8, 4
    # (n - 1) // seq_len windows: 17*8 tokens -> 16 train, 5*8 -> 4 val.
    _token_dir(tmp_path, train_tokens=17 * seq_len, val_tokens=5 * seq_len,
               body=np.arange(22 * seq_len))
    data = loader(TextData(source=str(tmp_path), seq_len=seq_len), batch=batch, read="memory",
        loading=Loading(workers=0, threads=1, read_buffer=1, worker_buffer=1))

    assert data.records == 16 and data.batch == batch and data.steps_per_epoch == 4

    for batches, windows, first in ((itertools.islice(data.train(DataPartition()), 4), 16, 5 * seq_len),
                                    (data.val(DataPartition()), 4, 0)):
        rows = np.concatenate([b["text"] for b in batches])
        assert rows.shape == (windows, seq_len + 1) and rows.dtype == np.int32
        starts = rows[:, 0] - first
        assert sorted(starts) == [seq_len * k for k in range(windows)], "every window once"
        assert np.array_equal(rows, starts[:, None] + first + np.arange(seq_len + 1)), \
            "each row is seq_len + 1 consecutive tokens"

    val_rows = np.concatenate([b["text"] for b in data.val(DataPartition())])
    assert np.array_equal(val_rows[:-1, -1], val_rows[1:, 0]), "the pass is in order"


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


def test_token_window_resume_refuses_a_changed_stride(tmp_path):
    _token_dir(tmp_path, train_tokens=33, val_tokens=17, body=np.arange(50))
    source = loader(TextData(source=str(tmp_path), seq_len=4, stride=1), batch=4, read="memory",
        loading=Loading(workers=0, threads=1, read_buffer=1, worker_buffer=1)).train(DataPartition())
    different = loader(TextData(source=str(tmp_path), seq_len=4, stride=2), batch=4, read="memory",
        loading=Loading(workers=0, threads=1, read_buffer=1, worker_buffer=1)).train(DataPartition())
    try:
        next(source)
        with pytest.raises(ValueError, match="saved data plan order changed"):
            different.set_state(source.get_state())
    finally:
        source.close()
        different.close()


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


def test_token_loader_records_do_not_depend_on_worker_count(tmp_path):
    seq_len = 8
    records = 16  # (17 * 8 - 1) // 8 == 16 windows
    _token_dir(tmp_path, train_tokens=(records + 1) * seq_len, val_tokens=2 * seq_len)

    def by_record(worker_count):
        data = loader(TextData(source=str(tmp_path), seq_len=seq_len), batch=4, read="memory", seed=7,
            loading=Loading(workers=worker_count))
        out = {}
        for b in itertools.islice(data.train(DataPartition()), data.steps_per_epoch):
            for row in b["text"]:
                # First token ids a window; the tail keeps the record's identity.
                out[int(row[0]) * 4096 + int(row[-1])] = row.tobytes()
        return out

    serial, parallel = by_record(worker_count=0), by_record(worker_count=2)
    assert len(serial) == records
    assert serial == parallel


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

def test_a_registered_token_spec_reads_the_directory(tmp_path):
    seq_len = 64
    # (41 * 64 - 1) // 64 == 40 train windows.
    _token_dir(tmp_path, train_tokens=41 * seq_len, val_tokens=8 * seq_len)

    data = loader("text", source=str(tmp_path), seq_len=seq_len, batch=4, read="memory",
        loading=Loading(workers=0))
    batch = next(data.train(DataPartition()))
    assert batch["text"].shape == (4, seq_len + 1)
    assert batch["text"].dtype == np.int32
    assert data.records == 40 and data.batch == 4


@pytest.mark.parametrize("pack", [False, True])
def test_a_token_spec_refuses_a_directory_without_a_val_split(tmp_path, pack):
    """With val.bin missing the validation loader read train.bin, so every
    pass scored windows the model was training on."""
    _token_dir(tmp_path, train_tokens=64, eos_id=0)
    with pytest.raises(FileNotFoundError, match=r"val\.bin"):
        loader(TextData(source=str(tmp_path), validation=TokenFiles(str(tmp_path), split="val"),
                  seq_len=8, pack="first-fit" if pack else None), batch=4, read="memory",
                      loading=Loading(workers=0))


@pytest.mark.parametrize("pack", [False, True])
def test_a_token_spec_needs_a_directory_with_a_train_split(tmp_path, pack):
    (tmp_path / "not_a_dataset.txt").write_text("hello")
    with pytest.raises(FileNotFoundError, match="train"):
        loader(TextData(source=TokenFiles(str(tmp_path)), pack="first-fit" if pack else None),
             batch=4, read="memory", loading=Loading(workers=0))
    with pytest.raises(ValueError, match=r"source|format|files"):
        loader(TextData(source="", pack="first-fit" if pack else None), batch=4, read="memory",
            loading=Loading(workers=0), validation_records=4 * (4))


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


def test_the_data_export_command_writes_the_librarys_token_records(tmp_path):
    from dew.cli.main import main
    from dew.data import Files
    from dew.data.pipeline import Pipeline
    from dew.data.sinks import export
    from dew.data.steps.text import Tokenize

    raw = tmp_path / "corpus.txt"
    raw.write_text("one line\nanother line\n" * 20, encoding="utf-8")
    fields = json.dumps({"source": str(raw)})
    assert main(["data", "export", "records", str(tmp_path / "cli"), "--fields", fields,
                 "--format", "array_record", "--tokenizer", "byte", "--eos", "--read", "memory"]) == 0
    pipeline = Pipeline(Files(str(raw), format="text")).map(Tokenize("byte", eos=True))
    export(pipeline, str(tmp_path / "library"), format="array_record", length_field="text",
           metadata={"token_store": True, "tokenizer": "byte", "vocab_size": 256,
                     "dtype": "int32", "eos_id": ByteTokenizer().eos_id})
    np.testing.assert_array_equal(TokenFiles(str(tmp_path / "cli")).tokens[:],
                                  TokenFiles(str(tmp_path / "library")).tokens[:])


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


def _bos_tokenizer(directory):
    """A word-level tokenizer that starts every encode with its bos id, as
    Llama's does, saved where `HFTokenizer` loads it without the hub."""
    from tokenizers import Tokenizer, models, pre_tokenizers, processors
    from transformers import PreTrainedTokenizerFast

    words = ["<s>", "</s>", "<unk>", *(f"w{index}" for index in range(20))]
    core = Tokenizer(models.WordLevel({word: index for index, word in enumerate(words)},
                                      unk_token="<unk>"))
    core.pre_tokenizer = pre_tokenizers.Whitespace()
    core.post_processor = processors.TemplateProcessing(single="<s> $A", special_tokens=[("<s>", 0)])
    PreTrainedTokenizerFast(tokenizer_object=core, bos_token="<s>", eos_token="</s>",
                            unk_token="<unk>").save_pretrained(str(directory))
    return str(directory)


def test_a_bos_adding_tokenizer_starts_each_document_once(tmp_path):
    tokenizer = _bos_tokenizer(tmp_path / "tokenizer")
    documents = ["w1 w2 w3", "w4"]
    export_documents(documents, tmp_path / "strings", tokenizer=tokenizer, eos=True)
    stream = TokenFiles(str(tmp_path / "strings")).tokens[:]
    assert stream.tolist() == [0, 4, 5, 6, 1, 0, 7, 1]
    assert np.flatnonzero(stream == 0).tolist() == [0, 5]
    assert np.flatnonzero(stream == 1).tolist() == [4, 7]


# ---------------------------------------------------------------------------------
# Packed documents
# ---------------------------------------------------------------------------------

def test_a_document_runs_through_its_eos_and_the_tail_past_the_last_is_one_too(tmp_path):
    """The documents are the spans through each eos id, the eos closing its
    own; a split cuts the stream mid-document, so the piece past the last
    eos is a document rather than lost, and a split with no eos at all is
    one document."""
    stream = [10, 11, 12, 0, 20, 21, 0, 30, 31]
    _stream_dir(tmp_path, stream)
    meta = json.loads((tmp_path / "meta.json").read_text())
    (tmp_path / "meta.json").write_text(json.dumps({**meta, "eos_id": 0}))
    data = loader(TextData(source=str(tmp_path), pack="first-fit", seq_len=15, packing_bins=1), batch=1,
        read="memory", loading=Loading(workers=0, threads=1, read_buffer=1, worker_buffer=1))
    (window,) = _read(data.val(DataPartition()))
    assert window["text"].tolist() == [*stream, *[0] * 7]
    assert window["text_segment_ids"].tolist() == [1, 1, 1, 1, 2, 2, 2, 3, 3, *[0] * 7]

    (tmp_path / "meta.json").write_text(json.dumps({**meta, "eos_id": 9}))
    data = loader(TextData(source=str(tmp_path), pack="first-fit", seq_len=15, packing_bins=1), batch=1,
        read="memory", loading=Loading(workers=0, threads=1, read_buffer=1, worker_buffer=1))
    assert _read(data.val(DataPartition()))[0]["text_segment_ids"].tolist() == [1] * 9 + [0] * 7

    (tmp_path / "meta.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="producer document boundaries or EOS"):
        loader(TextData(source=str(tmp_path), pack="first-fit", seq_len=15), batch=1, read="memory",
            loading=Loading(workers=0, threads=1, read_buffer=1, worker_buffer=1))


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


def test_packed_loader_lengths_count_windows_not_documents(tmp_path):
    """A run turns train_len into steps_per_epoch, so a split of one document
    that fills three windows cannot report one."""
    seq_len = 3  # windows of 4 ids
    _document_dir(tmp_path, [list(range(10, 19))])  # one 10-id document
    data = loader(TextData(source=str(tmp_path), pack="first-fit", seq_len=seq_len, packing_bins=1), batch=1,
        read="memory", loading=Loading(workers=0, threads=1, read_buffer=1, worker_buffer=1))

    assert data.records == 3
    assert len(list(data.val(DataPartition()))) == 3, "the length is not the pass it counts"


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


def test_packed_train_stream_does_not_end_with_the_documents(tmp_path):
    """The trainer keeps asking for batches long after one pass over the
    documents."""
    _document_dir(tmp_path, [[i, i + 1] for i in range(10, 30, 2)])
    data = loader(TextData(source=str(tmp_path), pack="first-fit", seq_len=8, packing_bins=2), batch=2,
        read="memory", loading=Loading(workers=0, threads=1, read_buffer=1, worker_buffer=1))

    iterator = data.train(DataPartition())
    assert len([next(iterator)["text"] for _ in range(20)]) == 20


def test_a_packed_validation_pass_covers_the_split_once(tmp_path):
    """The documents packed for validation are read once, without the
    training stream's repeat, so the pass finishes and scores each document
    once."""
    documents = [[i, i + 1] for i in range(10, 30, 2)]
    _document_dir(tmp_path, documents)
    data = loader(TextData(source=str(tmp_path), pack="first-fit", seq_len=8, packing_bins=2), batch=2,
        read="memory", loading=Loading(workers=0, threads=1, read_buffer=1, worker_buffer=1))

    batches = list(itertools.islice(data.val(DataPartition()), 20))
    # Ten documents of three ids (the eos counts) pack three to a nine-id
    # window, so four windows and two batches of two.
    assert len(batches) == 2
    read = Counter(int(token) for batch in batches for row in batch["text"]
                   for token in row if token)
    assert read == Counter(token for document in documents for token in document)



def test_only_the_packed_spec_carries_segment_ids_and_positions(tmp_path):
    _document_dir(tmp_path, [[10, 11, 12], [20, 21, 22, 23], [30, 31]] * 2)

    packed = next(loader(TextData(source=str(tmp_path), pack="first-fit", seq_len=8), batch=2, read="memory",
        loading=Loading(workers=0, threads=1, read_buffer=1, worker_buffer=1)).train(DataPartition()))
    assert set(packed) == {"text", "text_segment_ids", "text_positions"}

    fixed = next(loader(TextData(source=str(tmp_path), seq_len=8), batch=2, read="memory",
        loading=Loading(workers=0, threads=1, read_buffer=1, worker_buffer=1)).train(DataPartition()))
    assert set(fixed) == {"text"}, "the fixed-window loader grew packing keys"


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


def test_a_token_validation_pass_ends_when_the_split_runs_out(tmp_path):
    """Eight windows at batch four are two batches, then the end."""
    seq_len = 4
    _token_dir(tmp_path, train_tokens=13 * seq_len, val_tokens=9 * seq_len)
    data = loader(TextData(source=str(tmp_path), seq_len=seq_len), batch=4, read="memory",
        loading=Loading(workers=0, threads=1, read_buffer=1, worker_buffer=1))

    batches, ended = _bounded(data.val(DataPartition()), 3)

    assert len(batches) == 2 and ended, ENDLESS_VAL
    assert len(set(_rows(batches))) == 8, "a pass must not repeat a window"


def test_a_token_validation_pass_reads_every_window_once(tmp_path):
    """Ten windows at batch four are three batches, in file order, then the end.

    Every batch keeps the configured rows so its shape fits the device mesh:
    the two windows past the last full batch come in a third, filled out with
    repeats of them that `VALID_ROWS` marks.
    """
    seq_len = 4
    val_tokens = np.arange(900, 900 + 11 * seq_len, dtype=np.int64)
    _token_dir(tmp_path, train_tokens=13 * seq_len, val_tokens=len(val_tokens))
    (tmp_path / "val.bin").write_bytes(val_tokens.astype("<u2").tobytes())
    data = loader(TextData(source=str(tmp_path), seq_len=seq_len), batch=4, read="memory",
        loading=Loading(workers=0, threads=1, read_buffer=1, worker_buffer=1))

    batches, ended = _bounded(data.val(DataPartition()), 4)

    assert len(batches) == 3 and ended, ENDLESS_VAL
    assert len(_rows(batches)) == len(set(_rows(batches))) == 10, "a pass reads every window once"
    np.testing.assert_array_equal(batches[-1][VALID_ROWS], [True, True, False, False])
    np.testing.assert_array_equal(batches[0]["text"][0], val_tokens[:seq_len + 1])


def test_a_packed_validation_pass_reads_each_window_once_and_stops(tmp_path):
    """No document twice, every batch full, and the same batches next time.

    Bins come out in packing order for validation, so two passes are the same
    windows; today the pass never ends and the documents come round again.
    """
    documents = [[i, i + 1, i + 2] for i in range(10, 40, 3)]
    _document_dir(tmp_path, documents)
    data = loader(TextData(source=str(tmp_path), pack="first-fit", seq_len=8, packing_bins=2), batch=2,
        read="memory", loading=Loading(workers=0, threads=1, read_buffer=1, worker_buffer=1))

    batches, ended = _bounded(data.val(DataPartition()), 2 + len(documents))
    heads = [int(t) for row in _rows(batches) for t in np.frombuffer(row, batches[0]["text"].dtype)
             if int(t) in {d[0] for d in documents}]
    again, _ = _bounded(data.val(DataPartition()), len(batches))

    assert ended, ENDLESS_VAL
    assert all(batch["text"].shape[0] == 2 for batch in batches)
    assert sorted(heads) == sorted(set(heads)), "a document was handed over twice"
    assert _rows(again) == _rows(batches), "two passes read different windows"


def test_val_batches_bounds_a_validation_pass(tmp_path):
    """A run scores `val_batches` of val.bin per pass, or all of it when the
    spec leaves that None."""
    seq_len = 4
    _token_dir(tmp_path, train_tokens=13 * seq_len, val_tokens=9 * seq_len)

    batches, ended = _bounded(loader(TextData(source=str(tmp_path), seq_len=seq_len), batch=4, read="memory",
        loading=Loading(workers=0, threads=1, read_buffer=1, worker_buffer=1)).val(DataPartition()), 3)
    assert len(batches) == 2 and ended, ENDLESS_VAL

    batches, ended = _bounded(
        loader(TextData(source=str(tmp_path), seq_len=seq_len), batch=4, read="memory",
            loading=Loading(workers=0, threads=1, read_buffer=1, worker_buffer=1),
                validation_records=(1) * (4)).val(DataPartition()), 3)
    assert len(batches) == 1 and ended
    np.testing.assert_array_equal(batches[0]["text"][0],
                                  np.fromfile(tmp_path / "val.bin", "<u2")[:seq_len + 1])


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


def _attention_mask(batch, monkeypatch):
    """The mask the backbone hands the attention kernel for this batch.

    Recorded at the kernel call, which is the only place the mask has to be
    right; rebuilding it from the batch's segment ids here would test the
    test. The call hands the kernel a causal flag, a window, a mask and the
    documents' segment ids, which the fused kernels read natively, so the
    mask is what the kernel's own helpers make of those arguments.
    """
    seen = []
    kernel = attention_kind.scaled_dot_product_attention

    def recording_kernel(query, key, value, **kwargs):
        mask, documents = kwargs.get("mask"), kwargs.get("segment_ids")
        if documents is not None:
            mask = attention.with_documents(mask, documents)
        seen.append(attention.combined_attention_mask(
            query.shape[1], key.shape[1], kwargs.get("causal", False),
            kwargs.get("sliding_window"), mask))
        return kernel(query, key, value, **kwargs)

    tokens = jnp.asarray(batch["text"][:, :-1], jnp.int32)
    model = _tiny_backbone(tokens.shape[1])
    params = model.init(jax.random.PRNGKey(0), tokens)

    monkeypatch.setattr(attention_kind, "scaled_dot_product_attention", recording_kernel)
    model.apply(params, tokens,
                positions=jnp.asarray(batch["text_positions"][:, :-1]),
                segment_ids=jnp.asarray(batch["text_segment_ids"][:, :-1]))
    monkeypatch.undo()

    assert len(seen) == 1 and seen[0] is not None, "the packed batch built no mask"
    return np.asarray(seen[0])


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


def _assert_mask_blocks_everything_it_should(batch, monkeypatch):
    mask = _attention_mask(batch, monkeypatch)
    rows = np.asarray(batch["text"])
    length = mask.shape[-1]

    for row in range(rows.shape[0]):
        inside = {}
        for start, stop in _documents_in(rows[row]):
            for position in range(start, min(stop, length)):
                inside[position] = (start, stop)
        for query in range(length):
            for key in range(length):
                allowed = bool(mask[row, 0, query, key])
                same_document = (query in inside and key in inside
                                 and inside[query] == inside[key])
                assert allowed == (same_document and key <= query), (
                    f"row {row} position {query} attending to {key}")


def test_documents_packed_online_by_grain_keep_attention_inside_each_document(monkeypatch):
    """The online route `docs/concepts/data.md` shows under Packing: grain's
    concat-then-split packer names its fields the way the decoder reads
    them, so a row it packs masks attention at document boundaries."""
    seq_len = 16
    documents = [{"text": np.full(length, value, np.int32)}
                 for value, length in enumerate((5, 9, 3, 12, 7, 4, 11, 6), start=1)]

    data = compile(Pipeline(Memory(documents)).repeat().pack(seq_len + 1, policy="concat").batch(4),
                   read="indexed", loading=Loading(workers=0))
    stream = data.train(DataPartition())
    try:
        batch = next(stream)
    finally:
        stream.close()

    assert set(batch) == {"text", "text_segment_ids", "text_positions"}
    mask = _attention_mask(batch, monkeypatch)
    tokens = np.asarray(batch["text"])[:, :-1]
    assert len(set(tokens[0])) > 1, "the first row packs more than one document"
    for row in range(tokens.shape[0]):
        for query in range(seq_len):
            for key in range(seq_len):
                same_document = tokens[row, query] == tokens[row, key]
                assert bool(mask[row, 0, query, key]) == (same_document and key <= query), (
                    f"row {row} position {query} attending to {key}")


def test_a_position_saved_under_other_packing_bins_is_refused(tmp_path):
    """Which chunks share a window depends on how many bins the plan keeps
    open. Two plans with as many windows were described alike, so a run
    resumed under another bin count read other windows at the same count."""
    _document_dir(tmp_path, [[5, 5], [6], [], []], eos_id=0)
    one, two = (loader(TextData(source=str(tmp_path), pack="first-fit", seq_len=3, packing_bins=bins),
                       batch=1,
        read="memory", loading=Loading(workers=0, threads=1, read_buffer=1,
            worker_buffer=1)) for bins in (1, 2))
    assert one.records == two.records == 2
    first = [_read(data.val(DataPartition()))[0]["text_segment_ids"].tolist() for data in (one, two)]
    assert first[0] != first[1], "the plans differ"
    source, other = one.train(DataPartition()), two.train(DataPartition())
    try:
        next(source)
        with pytest.raises(ValueError, match="saved data plan order changed"):
            other.set_state(source.get_state())
    finally:
        source.close()
        other.close()


def test_windows_read_only_their_span_and_memory_packing_never_rereads_the_store(tmp_path, monkeypatch):
    """An indexed window reads its span; materialized packing keeps its rows in memory."""
    documents = [list(range(1, 40)), [2, 3], list(range(5, 30))]
    stream = np.concatenate([np.asarray([*d, PACK_EOS], np.int64) for d in documents])
    _token_dir(tmp_path, train_tokens=0, body=stream, eos_id=PACK_EOS)
    (tmp_path / "val.bin").write_bytes(stream.astype(np.uint16).tobytes())
    packed = loader(TextData(source=str(tmp_path), pack="first-fit", seq_len=7, packing_bins=2), batch=2,
        read="memory", loading=Loading(workers=0, threads=1, read_buffer=1, worker_buffer=1))
    indexed = loader(TextData(source=str(tmp_path), seq_len=7), batch=2, read="indexed",
                   loading=Loading(workers=0, threads=1, read_buffer=1, worker_buffer=1))

    spans = []
    read = np.memmap.__getitem__

    def recording(self, span):
        spans.append(span.stop - span.start)
        return read(self, span)

    monkeypatch.setattr(np.memmap, "__getitem__", recording)
    stream = indexed.train(DataPartition())
    try:
        windows = list(itertools.islice(stream, 10))
    finally:
        stream.close()
    assert spans and max(spans) <= 8
    stream = packed.train(DataPartition())
    try:
        next(stream)
    finally:
        stream.close()
    before = len(spans)
    stream = packed.train(DataPartition())
    try:
        windows.extend(itertools.islice(stream, 10))
    finally:
        stream.close()
    assert len(spans) == before
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


def test_a_document_the_size_of_the_window_is_one_segment(tmp_path, monkeypatch):
    """Exactly the window is the case the chunker must not cut."""
    batches = _packed(tmp_path, [[2, 3, 4, 5]], seq_len=4, bins=1)

    assert len(batches) == 1
    np.testing.assert_array_equal(batches[0]["text"][0], [2, 3, 4, 5, PACK_EOS])
    np.testing.assert_array_equal(batches[0]["text_segment_ids"][0], [1] * 5)
    np.testing.assert_array_equal(batches[0]["text_positions"][0], range(5))
    _assert_mask_blocks_everything_it_should(batches[0], monkeypatch)


def test_three_documents_in_one_window_cannot_read_each_other(tmp_path, monkeypatch):
    batches = _packed(tmp_path, [[2, 3], [4, 5], [6, 7]], seq_len=8, bins=1)
    row = batches[0]

    np.testing.assert_array_equal(row["text"][0], [2, 3, 1, 4, 5, 1, 6, 7, 1])
    np.testing.assert_array_equal(row["text_segment_ids"][0],
                                  [1, 1, 1, 2, 2, 2, 3, 3, 3])
    np.testing.assert_array_equal(row["text_positions"][0], [0, 1, 2, 0, 1, 2, 0, 1, 2])
    _assert_mask_blocks_everything_it_should(row, monkeypatch)


def test_a_document_longer_than_the_window_is_cut_into_separate_segments(
        tmp_path, monkeypatch):
    """Each piece is its own segment, so no chunk attends into the one before
    it and RoPE starts again at zero."""
    batches = _packed(tmp_path, [list(range(2, 14))], seq_len=4, bins=1)

    assert len(batches) == 3
    np.testing.assert_array_equal(
        np.concatenate([b["text"][0] for b in batches])[:13],
        [*list(range(2, 14)), PACK_EOS])
    for batch in batches:
        np.testing.assert_array_equal(batch["text_positions"][0][:1], [0])
        _assert_mask_blocks_everything_it_should(batch, monkeypatch)


def test_a_document_of_one_token_is_its_own_segment(tmp_path, monkeypatch):
    """A split can end on a bare boundary, and a one-token document owns no
    transition: it must not borrow the previous document's."""
    batches = _packed(tmp_path, [[2, 3, 4], []], seq_len=8, bins=1)
    row = batches[0]

    np.testing.assert_array_equal(row["text"][0], [2, 3, 4, 1, 1, 0, 0, 0, 0])
    np.testing.assert_array_equal(row["text_segment_ids"][0],
                                  [1, 1, 1, 1, 2, 0, 0, 0, 0])
    _assert_mask_blocks_everything_it_should(row, monkeypatch)

    ce, by_hand, counted = _counted_cross_entropy(row, seq_len=8)
    assert counted == 3, "four ids of the first document, none from the second"
    assert ce == pytest.approx(by_hand, rel=1e-5)


def test_the_padded_tail_of_a_window_is_attended_by_nothing_and_counts_for_nothing(
        tmp_path, monkeypatch):
    batches = _packed(tmp_path, [[2, 3]], seq_len=8, bins=8)
    row = batches[0]

    np.testing.assert_array_equal(row["text"][0], [2, 3, 1, 0, 0, 0, 0, 0, 0])
    np.testing.assert_array_equal(row["text_segment_ids"][0], [1, 1, 1, 0, 0, 0, 0, 0, 0])
    _assert_mask_blocks_everything_it_should(row, monkeypatch)

    ce, by_hand, counted = _counted_cross_entropy(row, seq_len=8)
    assert counted == 2, "the padding owns no target"
    assert ce == pytest.approx(by_hand, rel=1e-5)


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
    assert windows(worker_count) == serial


@pytest.mark.slow
def test_an_interrupted_token_epoch_resumes_through_real_workers(tmp_path):
    """A resumed run builds its loader again, in a new process, and the saved
    position is a record count into an order the source names: the description
    has to name the file, not an address in the process that wrote it."""
    seq_len = 8
    _token_dir(tmp_path, train_tokens=33 * seq_len, val_tokens=2 * seq_len)

    def loader():
        return loader(TextData(source=str(tmp_path), seq_len=seq_len), batch=4, read="memory", seed=7,
            loading=Loading(workers=2))

    data = loader()
    epoch = data.steps_per_epoch  # 32 windows in eight batches
    interrupted = data.train(DataPartition())
    seen = [row.tobytes() for row in next(interrupted)["text"]]
    state = interrupted.get_state()
    rest = list(itertools.islice(interrupted, 2 * epoch - 1))
    unseen = [row.tobytes() for batch in rest for row in batch["text"]]

    restored = loader().train(DataPartition())
    restored.set_state(state)
    after = list(itertools.islice(restored, 2 * epoch - 1))
    resumed = [row.tobytes() for batch in after for row in batch["text"]]

    assert "object at 0x" not in json.loads(state)[ENVELOPE]["order"], (
        "a source described by its address can only be restored in the process "
        "that saved it")
    assert resumed == unseen
    assert len(set(seen + resumed[:(epoch - 1) * 4])) == 32, "the epoch reads every window once"


@pytest.mark.slow
def test_an_interrupted_packed_epoch_resumes_through_mp_prefetch(tmp_path):
    """The packed loader reads its documents in worker processes and packs
    behind them, so a restart has to carry the packer's position too."""
    _document_dir(tmp_path, [[i, i + 1, i + 2] for i in range(10, 100, 3)])

    def loader():
        return loader(TextData(source=str(tmp_path), pack="first-fit", seq_len=8, packing_bins=2), batch=2,
            read="memory", loading=Loading(workers=2, worker_buffer=1))

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
    as the fields of a `TextData` over it.

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
        data = loader(TextData(source=TokenFiles(str(root / name), field=fields[name]["field"])), batch=1,
            read="memory",
            loading=Loading(workers=0, threads=1, read_buffer=1, worker_buffer=1), **spec)
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
