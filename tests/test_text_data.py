"""Text data for language models: tokenizers, the token-file source, the loader.

ByteTokenizer and TokenFileSource are pure numpy; the HF tokenizer tests only
run when a cached copy of the hub is reachable, matching the repo's policy
that no test needs the network.
"""

import itertools
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from dew.config import DataConfig
from dew.data.dataloaders import (
    get_packed_token_dataset_grain, get_token_dataset_grain, load_data,
)
from dew.data.sources.text import TokenDocumentSource, TokenFileSource
from dew.data.text import ByteTokenizer
from dew.nn.backbones import causal_transformer as backbone
from dew.objectives.lm import LMObjective

REPO_ROOT = Path(__file__).resolve().parents[1]

# grain's worker processes read absl flags; a test that never ran absl.app
# would trip UnparsedFlagAccessError at any worker_count > 0.
from absl import flags

if not flags.FLAGS.is_parsed():
    flags.FLAGS.mark_as_parsed()


def _token_dir(tmp_path, train_tokens, val_tokens=None, seq_len=8, dtype=np.uint16,
               vocab_size=256, body=None, eos_id=None):
    """Write a token directory: train.bin (+ val.bin + meta.json)."""
    rng = np.random.RandomState(0)
    tokens = (rng.randint(0, vocab_size, train_tokens + (val_tokens or 0))
              if body is None else np.asarray(body, dtype=np.int64))
    if val_tokens is None:
        train, val = tokens, None
    else:
        val, train = tokens[:val_tokens], tokens[val_tokens:]
    (tmp_path / "train.bin").write_bytes(train.astype(dtype).tobytes())
    if val is not None:
        (tmp_path / "val.bin").write_bytes(val.astype(dtype).tobytes())
    meta = {
        "tokenizer": "byte", "vocab_size": vocab_size,
        "dtype": np.dtype(dtype).name,
        "train_tokens": len(train), "val_tokens": len(val) if val is not None else 0,
    }
    if eos_id is not None:
        meta["eos_id"] = eos_id
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    return tmp_path


def _document_dir(tmp_path, documents, eos_id=0, dtype=np.uint16):
    """A token directory whose stream is `documents`, each closed by eos_id."""
    stream = np.concatenate([np.asarray(d + [eos_id], np.int64) for d in documents])
    _token_dir(tmp_path, train_tokens=0, body=stream, dtype=dtype, eos_id=eos_id)
    (tmp_path / "val.bin").write_bytes(stream.astype(dtype).tobytes())
    return tmp_path, stream


# ---------------------------------------------------------------------------------
# ByteTokenizer
# ---------------------------------------------------------------------------------

def test_byte_tokenizer_round_trips_unicode():
    tok = ByteTokenizer()
    for text in ("hello, world", "ünïcödé — π≈3.14159", "日本語のテキスト",
                 "emoji 🚀🔥 and \x00 control bytes\n\t"):
        ids = tok.encode(text)
        assert isinstance(ids, list) and all(isinstance(i, int) for i in ids)
        assert tok.decode(ids) == text


def test_byte_tokenizer_is_utf8_bytes_with_a_256_vocab():
    tok = ByteTokenizer()
    assert tok.vocab_size == 256
    assert tok.encode("A") == [0x41]
    assert tok.encode("é") == [0xC3, 0xA9]  # two utf-8 bytes, not one id
    assert isinstance(tok.eos_id, int)


def test_byte_tokenizer_decode_tolerates_junk_bytes():
    # Generated ids will land on invalid utf-8 sequences; the model still
    # needs text out of them, not an exception.
    tok = ByteTokenizer()
    assert isinstance(tok.decode([0xFF, 0xFE, 0x41]), str)


# ---------------------------------------------------------------------------------
# HFTokenizer, only when the hub (or its cache) cooperates
# ---------------------------------------------------------------------------------

def _hf_tokenizer():
    try:
        from dew.data.text import HFTokenizer
        return HFTokenizer("gpt2")
    except Exception as exc:  # no transformers, no cache, no network
        pytest.skip(f"the HF tokenizer is unavailable here: {exc}")


def test_hf_tokenizer_encodes_and_decodes():
    tok = _hf_tokenizer()
    ids = tok.encode("hello world")
    assert 0 < len(ids) < 11 and all(isinstance(i, int) for i in ids)
    assert tok.decode(ids) == "hello world"
    assert tok.vocab_size == 50257
    assert tok.eos_id == 50256


def test_hf_tokenizer_imports_lazily():
    """Constructing HFTokenizer must not import transformers."""
    import subprocess as sp
    probe = (
        "from dew.data.text import HFTokenizer;"
        "import sys;"
        "HFTokenizer('gpt2');"
        "assert 'transformers' not in sys.modules, 'imported at construction'"
    )
    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT / "src"))
    result = sp.run([sys.executable, "-c", probe], cwd=REPO_ROOT,
                    capture_output=True, text=True, env=env)
    assert result.returncode == 0, result.stderr


# ---------------------------------------------------------------------------------
# TokenFileSource
# ---------------------------------------------------------------------------------

def test_token_file_source_length_and_window_contract(tmp_path):
    seq_len = 8
    n = 37  # tokens; (37 - 1) // 8 == 4 windows
    body = np.arange(1, n + 1, dtype=np.int64)
    _token_dir(tmp_path, train_tokens=0, body=body, dtype=np.uint32)
    source = TokenFileSource(str(tmp_path / "train.bin"), seq_len)

    assert len(source) == (n - 1) // seq_len
    for i in range(len(source)):
        window = source[i]["text"]
        assert window.dtype == np.int32
        assert window.shape == (seq_len + 1,)
        # record i covers [i*seq_len, i*seq_len + seq_len + 1)
        np.testing.assert_array_equal(
            window, body[i * seq_len:(i + 1) * seq_len + 1])
    with pytest.raises(IndexError):
        source[len(source)]


def test_token_file_source_reads_the_dtype_from_meta(tmp_path):
    seq_len = 4
    tokens = np.arange(1, 3 * seq_len + 1, dtype=np.uint32)
    (tmp_path / "train.bin").write_bytes(tokens.astype("<u4").tobytes())
    (tmp_path / "meta.json").write_text(json.dumps(
        {"dtype": "uint32", "vocab_size": 100000, "tokenizer": "gpt2"}))
    (tmp_path / "val.bin").write_bytes(tokens.astype("<u4").tobytes())

    source = TokenFileSource(str(tmp_path / "train.bin"), seq_len)
    assert source.dtype == np.dtype("uint32")
    np.testing.assert_array_equal(source[0]["text"], tokens[:seq_len + 1])
    assert source.vocab_size == 100000

    # Without meta.json the nanoGPT default applies.
    bare = tmp_path / "bare"
    bare.mkdir()
    (bare / "train.bin").write_bytes(tokens.astype("<u2").tobytes())
    default_source = TokenFileSource(str(bare / "train.bin"), seq_len)
    assert default_source.dtype == np.dtype("uint16")
    np.testing.assert_array_equal(
        default_source[0]["text"], tokens[:seq_len + 1].astype("<u2").astype(np.int32))


def test_token_file_source_rejects_files_too_short(tmp_path):
    (tmp_path / "train.bin").write_bytes(np.zeros(seq_len := 4, np.uint16).tobytes())
    with pytest.raises(ValueError, match="too few for even one window"):
        TokenFileSource(str(tmp_path / "train.bin"), seq_len)
    with pytest.raises(ValueError, match="seq_len"):
        TokenFileSource(str(tmp_path / "train.bin"), 0)


# ---------------------------------------------------------------------------------
# get_token_dataset_grain
# ---------------------------------------------------------------------------------

def test_token_loader_yields_int32_batches_with_one_overlap_token(tmp_path):
    seq_len, batch = 8, 4
    # (n - 1) // seq_len windows: 17*8 tokens -> 16 train, 5*8 -> 4 val.
    _token_dir(tmp_path, train_tokens=17 * seq_len, val_tokens=5 * seq_len,
               seq_len=seq_len)
    data = get_token_dataset_grain(
        str(tmp_path / "train.bin"), str(tmp_path / "val.bin"),
        batch_size=batch, seq_len=seq_len, seed=0, worker_count=0, num_epochs=1)

    assert data["train_len"] == 16 and data["val_len"] == 4
    assert data["local_batch_size"] == batch and data["global_batch_size"] == batch

    for split, windows in (("train", 16), ("val", 4)):
        batches = list(data[split]())
        assert len(batches) == windows // batch
        for b in batches:
            assert b["text"].shape == (batch, seq_len + 1)
            assert b["text"].dtype == np.int32


def test_token_loader_val_is_unshuffled_and_disjoint_from_train(tmp_path):
    seq_len = 4
    # 13*4 tokens -> 12 train windows; 9*4 -> 8 val windows.
    train_tokens = np.arange(100, 100 + 13 * seq_len, dtype=np.int64)
    val_tokens = np.arange(900, 900 + 9 * seq_len, dtype=np.int64)
    _token_dir(tmp_path, train_tokens=len(train_tokens),
               val_tokens=len(val_tokens), seq_len=seq_len)
    # _token_dir draws random tokens; overwrite with a known layout.
    (tmp_path / "train.bin").write_bytes(train_tokens.astype("<u2").tobytes())
    (tmp_path / "val.bin").write_bytes(val_tokens.astype("<u2").tobytes())

    data = get_token_dataset_grain(
        str(tmp_path / "train.bin"), str(tmp_path / "val.bin"),
        batch_size=4, seq_len=seq_len, seed=0, worker_count=0, num_epochs=1)

    val_batches = [b["text"] for b in data["val"]()]
    assert len(val_batches) == 2  # 8 windows, batch 4, drop_remainder
    # Unshuffled: windows come in file order, each overlapping the next by one.
    np.testing.assert_array_equal(val_batches[0][0], val_tokens[:seq_len + 1])
    np.testing.assert_array_equal(val_batches[1][0],
                                  val_tokens[4 * seq_len:5 * seq_len + 1])

    # Train and val are disjoint files: no train window equals a val window.
    train_windows = {w.tobytes() for w in np.concatenate(
        [b["text"] for b in data["train"]()])}
    val_windows = {w.tobytes() for w in np.concatenate(val_batches)}
    assert not (train_windows & val_windows)


@pytest.mark.parametrize("worker_count", [0, 2])
def test_token_loader_validation_pass_reads_every_window_once(tmp_path, worker_count):
    """num_epochs is the training stream's, and a run leaves it None.

    Validation read val.bin through a sampler carrying the same unbounded
    epoch count, and grain's DataLoader batches inside each worker, so a pass
    never ended and its batches repeated windows the worker had already read.
    """
    seq_len = 4
    val_tokens = np.arange(900, 900 + 13 * seq_len, dtype=np.int64)  # 12 windows
    _token_dir(tmp_path, train_tokens=5 * seq_len, val_tokens=len(val_tokens),
               seq_len=seq_len)
    (tmp_path / "val.bin").write_bytes(val_tokens.astype("<u2").tobytes())

    data = get_token_dataset_grain(
        str(tmp_path / "train.bin"), str(tmp_path / "val.bin"),
        batch_size=4, seq_len=seq_len, seed=0, worker_count=worker_count)

    windows = [list(window) for batch in itertools.islice(data["val"](), 12)
               for window in batch["text"]]
    assert windows == [list(val_tokens[start:start + seq_len + 1])
                       for start in range(0, 12 * seq_len, seq_len)]


def test_token_loader_records_do_not_depend_on_worker_count(tmp_path):
    seq_len = 8
    records = 16  # (17 * 8 - 1) // 8 == 16 windows
    _token_dir(tmp_path, train_tokens=(records + 1) * seq_len, seq_len=seq_len)

    def by_record(worker_count):
        data = get_token_dataset_grain(
            str(tmp_path / "train.bin"), str(tmp_path / "train.bin"),
            batch_size=4, seq_len=seq_len, seed=7, worker_count=worker_count,
            num_epochs=1)
        out = {}
        for b in data["train"]():
            for row in b["text"]:
                # First token ids a window; the tail keeps the record's identity.
                out[int(row[0]) * 4096 + int(row[-1])] = row.tobytes()
        return out

    serial, parallel = by_record(worker_count=0), by_record(worker_count=2)
    assert len(serial) == records
    assert serial == parallel


def test_token_loader_seeds_its_train_sampler(tmp_path):
    seq_len, records = 8, 24
    _token_dir(tmp_path, train_tokens=(records + 1) * seq_len, seq_len=seq_len)

    def first_batch(seed):
        data = get_token_dataset_grain(
            str(tmp_path / "train.bin"), str(tmp_path / "train.bin"),
            batch_size=4, seq_len=seq_len, seed=seed, worker_count=0)
        return next(iter(data["train"]()))["text"]

    assert not np.array_equal(first_batch(0), first_batch(1))


# ---------------------------------------------------------------------------------
# load_data dispatch

def test_load_data_dispatches_a_token_directory(tmp_path):
    seq_len = 64
    # (41 * 64 - 1) // 64 == 40 train windows.
    _token_dir(tmp_path, train_tokens=41 * seq_len, val_tokens=8 * seq_len,
               seq_len=seq_len)
    data = load_data(DataConfig(dataset=str(tmp_path), sequence_length=seq_len,
                                batch_size=4, worker_count=0))
    batch = next(iter(data["train"]()))
    assert batch["text"].shape == (4, seq_len + 1)
    assert batch["text"].dtype == np.int32
    assert data["train_len"] == 40
    assert data["local_batch_size"] == 4


def test_load_data_token_directory_requires_sequence_length(tmp_path):
    _token_dir(tmp_path, train_tokens=64, val_tokens=16)
    with pytest.raises(ValueError, match="sequence_length"):
        load_data(DataConfig(dataset=str(tmp_path), batch_size=4,
                             worker_count=0))


def test_load_data_does_not_dispatch_on_a_plain_directory(tmp_path):
    (tmp_path / "not_a_dataset.txt").write_text("hello")
    with pytest.raises(ValueError, match="not found in mediaDatasetMap"):
        load_data(DataConfig(dataset=str(tmp_path), worker_count=0))


# ---------------------------------------------------------------------------------
# tools/tokenize_text.py
# ---------------------------------------------------------------------------------

def test_tokenize_tool_round_trips_through_the_source(tmp_path):
    corpus = "\n".join(
        f"document {i}: the quick brown fox jumps over the lazy dog — ünïcödé {i}"
        for i in range(40)) + "\n"
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "a.txt").write_text(corpus * 5, encoding="utf-8")
    (raw / "nested").mkdir()
    (raw / "nested" / "b.txt").write_text(corpus * 3, encoding="utf-8")
    out = tmp_path / "tokens"

    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT / "src"))
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "tools" / "tokenize_text.py"),
         "--input", str(raw), "--out", str(out),
         "--tokenizer", "byte", "--val-fraction", "0.1"],
        capture_output=True, text=True, env=env)
    assert result.returncode == 0, result.stdout + result.stderr

    meta = json.loads((out / "meta.json").read_text())
    assert meta["tokenizer"] == "byte"
    assert meta["vocab_size"] == 256
    assert meta["dtype"] == "uint8"
    assert meta["train_tokens"] + meta["val_tokens"] == len(
        (corpus * 8).encode("utf-8"))

    seq_len = 32
    train = TokenFileSource(str(out / "train.bin"), seq_len)
    val = TokenFileSource(str(out / "val.bin"), seq_len)
    assert train.dtype == np.dtype("uint8")
    assert len(train) == (meta["train_tokens"] - 1) // seq_len
    assert len(val) == (meta["val_tokens"] - 1) // seq_len

    tok = ByteTokenizer()
    whole = corpus * 8  # a.txt (5x) then nested/b.txt (3x), in path order
    whole_ids = tok.encode(whole)

    # The two files are a lossless partition of the corpus, val its head.
    val_bytes = list((out / "val.bin").read_bytes())
    train_bytes = list((out / "train.bin").read_bytes())
    assert tok.decode(val_bytes + train_bytes) == whole
    assert len(val_bytes) == meta["val_tokens"]
    assert len(train_bytes) == meta["train_tokens"]

    # Windows tile each split at stride seq_len, so stitching them back
    # rebuilds the split up to the tokens past the last full window.
    def stitch(source):
        return np.concatenate(
            [source[i]["text"][:seq_len] for i in range(len(source))]
            + [source[len(source) - 1]["text"][seq_len:]])

    def covered(n_tokens):
        return ((n_tokens - 1) // seq_len) * seq_len + 1

    train_ids, val_ids = stitch(train), stitch(val)
    assert len(val_ids) == covered(meta["val_tokens"])
    assert len(train_ids) == covered(meta["train_tokens"])
    assert list(val_ids) == whole_ids[:len(val_ids)]
    assert list(train_ids) == whole_ids[meta["val_tokens"]:
                                        meta["val_tokens"] + len(train_ids)]

    # And what the windows carry decodes back to that text (a window boundary
    # can split a multi-byte character, which decode replaces on both sides).
    assert tok.decode(val_ids.tolist()) == bytes(
        whole_ids[:len(val_ids)]).decode("utf-8", errors="replace")


def test_tokenize_tool_writes_the_smallest_dtype_that_fits():
    from tools.tokenize_text import dtype_for
    assert dtype_for(256) == np.dtype("uint8")
    assert dtype_for(257) == np.dtype("uint16")
    assert dtype_for(50257) == np.dtype("uint16")
    assert dtype_for(70000) == np.dtype("uint32")


# ---------------------------------------------------------------------------------
# TokenDocumentSource and get_packed_token_dataset_grain
# ---------------------------------------------------------------------------------

def test_document_source_reads_one_document_per_record(tmp_path):
    documents = [[10, 11, 12], [20, 21], [30, 31, 32, 33]]
    _document_dir(tmp_path, documents, eos_id=0)
    source = TokenDocumentSource(str(tmp_path / "train.bin"))

    assert len(source) == 3
    for index, document in enumerate(documents):
        # The eos closes the document, so it belongs to the record.
        np.testing.assert_array_equal(
            source[index]["text"], np.asarray(document + [0], np.int32))
    assert list(source.lengths) == [4, 3, 5]


def test_document_source_keeps_the_tail_past_the_last_boundary(tmp_path):
    """A split cuts the stream mid-document; dropping the piece past the last
    eos would lose those tokens with nothing said about it."""
    stream = np.asarray([10, 11, 0, 20, 21], np.int64)
    _token_dir(tmp_path, train_tokens=0, body=stream, eos_id=0)
    source = TokenDocumentSource(str(tmp_path / "train.bin"))

    assert len(source) == 2
    np.testing.assert_array_equal(source[1]["text"], np.asarray([20, 21], np.int32))


def test_document_source_needs_an_eos_id(tmp_path):
    _token_dir(tmp_path, train_tokens=32)  # meta.json without eos_id
    with pytest.raises(ValueError, match="no eos_id"):
        TokenDocumentSource(str(tmp_path / "train.bin"))


def test_document_source_reads_a_split_without_a_boundary_as_one_document(tmp_path):
    _token_dir(tmp_path, train_tokens=32, body=[1, 2, 3, 4], eos_id=9)
    source = TokenDocumentSource(str(tmp_path / "train.bin"))

    assert len(source) == 1
    np.testing.assert_array_equal(source[0]["text"],
                                  np.asarray([1, 2, 3, 4], np.int32))


def test_packed_loader_fills_windows_with_whole_documents(tmp_path):
    seq_len = 8  # windows of 9 ids
    # 4, 6 and 5 ids once each eos is counted: first fit puts 4 + 5 in one
    # window and 6 in the next.
    _document_dir(tmp_path, [[10, 11, 12], [20, 21, 22, 23, 24], [30, 31, 32, 33]])
    data = get_packed_token_dataset_grain(
        str(tmp_path / "train.bin"), str(tmp_path / "val.bin"),
        batch_size=2, seq_len=seq_len, seed=0, worker_count=0, num_epochs=1,
        num_packing_bins=2)

    assert data["train_len"] == 3 and data["val_len"] == 3
    assert data["local_batch_size"] == 2 and data["global_batch_size"] == 2
    batch = next(iter(data["val"]()))

    for key in ("text", "text_segment_ids", "text_positions"):
        assert batch[key].shape == (2, seq_len + 1)
        assert batch[key].dtype == np.int32
    np.testing.assert_array_equal(batch["text"], [
        [10, 11, 12, 0, 30, 31, 32, 33, 0],
        [20, 21, 22, 23, 24, 0, 0, 0, 0]])
    np.testing.assert_array_equal(batch["text_segment_ids"], [
        [1, 1, 1, 1, 2, 2, 2, 2, 2],
        [1, 1, 1, 1, 1, 1, 0, 0, 0]])
    # Positions restart at 0 inside every document, which is what RoPE reads.
    np.testing.assert_array_equal(batch["text_positions"], [
        [0, 1, 2, 3, 0, 1, 2, 3, 4],
        [0, 1, 2, 3, 4, 5, 0, 0, 0]])


def test_packed_loader_cuts_documents_that_outgrow_the_window(tmp_path):
    """Grain's packer refuses an over-long element, so the loader cuts first;
    each piece is its own segment and its positions start again at 0."""
    seq_len = 3  # windows of 4 ids
    _document_dir(tmp_path, [list(range(10, 19))])  # one 10-id document
    data = get_packed_token_dataset_grain(
        str(tmp_path / "train.bin"), str(tmp_path / "val.bin"),
        batch_size=1, seq_len=seq_len, seed=0, worker_count=0, num_epochs=1,
        num_packing_bins=1)

    rows = [batch["text"][0] for batch in data["val"]()]
    positions = [batch["text_positions"][0] for batch in data["val"]()]
    assert len(rows) == 3  # ceil(10 / 4) pieces, one per window
    np.testing.assert_array_equal(np.concatenate(rows)[:10],
                                  list(range(10, 19)) + [0])
    np.testing.assert_array_equal(positions[0], [0, 1, 2, 3])


def test_packed_loader_lengths_count_windows_not_documents(tmp_path):
    """A run turns train_len into steps_per_epoch, so a split of one document
    that fills three windows cannot report one."""
    seq_len = 3  # windows of 4 ids
    _document_dir(tmp_path, [list(range(10, 19))])  # one 10-id document
    data = get_packed_token_dataset_grain(
        str(tmp_path / "train.bin"), str(tmp_path / "val.bin"),
        batch_size=1, seq_len=seq_len, seed=0, worker_count=0, num_epochs=1,
        num_packing_bins=1)

    assert data["train_len"] == 3 and data["val_len"] == 3
    assert len(list(data["val"]())) == 3, "the length is not the pass it counts"


def test_packed_loader_state_restores_the_next_unseen_batch(tmp_path):
    """The trainer saves the iterator's position in the checkpoint, so a
    restored iterator has to carry on where the saved one had got to."""
    _document_dir(tmp_path, [[i, i + 1, i + 2] for i in range(10, 60, 3)])
    data = get_packed_token_dataset_grain(
        str(tmp_path / "train.bin"), str(tmp_path / "val.bin"),
        batch_size=2, seq_len=8, seed=0, worker_count=0, num_epochs=1,
        num_packing_bins=2)

    iterator = iter(data["val"]())
    next(iterator)
    state = iterator.get_state()
    expected = [next(iterator)["text"] for _ in range(2)]

    restored = iter(data["val"]())
    restored.set_state(state)
    for wanted, got in zip(expected, [next(restored)["text"] for _ in range(2)]):
        np.testing.assert_array_equal(wanted, got)


def test_packed_loader_windows_do_not_depend_on_worker_count(tmp_path):
    _document_dir(tmp_path, [[i, i + 1, i + 2] for i in range(10, 70, 3)])

    def windows(worker_count):
        data = get_packed_token_dataset_grain(
            str(tmp_path / "train.bin"), str(tmp_path / "val.bin"),
            batch_size=2, seq_len=8, seed=0, worker_count=worker_count,
            num_epochs=1, num_packing_bins=2)
        return sorted(row.tobytes() for batch in data["val"]()
                      for row in batch["text"])

    serial = windows(0)
    assert serial, "the loader produced no windows"
    assert serial == windows(2)


def test_packed_train_stream_does_not_end_with_the_documents(tmp_path):
    """num_epochs None is what a run uses, and the trainer keeps asking for
    batches long after one pass over the documents."""
    _document_dir(tmp_path, [[i, i + 1] for i in range(10, 30, 2)])
    data = get_packed_token_dataset_grain(
        str(tmp_path / "train.bin"), str(tmp_path / "val.bin"),
        batch_size=2, seq_len=8, seed=0, worker_count=0, num_packing_bins=2)

    iterator = iter(data["train"]())
    assert len([next(iterator)["text"] for _ in range(20)]) == 20


def test_a_packed_validation_pass_covers_the_split_once(tmp_path):
    """The same num_epochs repeated the documents before packing for
    validation as well, so a run that leaves it None never finished a
    validation pass and scored some documents several times over."""
    documents = [[i, i + 1] for i in range(10, 30, 2)]
    _document_dir(tmp_path, documents)
    data = get_packed_token_dataset_grain(
        str(tmp_path / "train.bin"), str(tmp_path / "val.bin"),
        batch_size=2, seq_len=8, seed=0, worker_count=0, num_packing_bins=2)

    batches = list(itertools.islice(data["val"](), 20))
    # Ten documents of three ids (the eos counts) pack three to a nine-id
    # window, so four windows and two batches of two.
    assert len(batches) == 2
    read = Counter(int(token) for batch in batches for row in batch["text"]
                   for token in row if token)
    assert read == Counter(token for document in documents for token in document)



def test_load_data_selects_packing_only_when_asked(tmp_path):
    _document_dir(tmp_path, [[10, 11, 12], [20, 21, 22, 23], [30, 31]])
    shared = dict(dataset=str(tmp_path), sequence_length=8, batch_size=2,
                  worker_count=0)

    packed = next(iter(load_data(DataConfig(pack_sequences=True, **shared))["train"]()))
    assert set(packed) == {"text", "text_segment_ids", "text_positions"}

    fixed = next(iter(load_data(DataConfig(**shared))["train"]()))
    assert set(fixed) == {"text"}, "the fixed-window loader grew packing keys"


def test_a_single_file_corpus_packs_when_its_val_split_holds_no_eos(tmp_path):
    """--pack closes each input file with one eos and the val split is cut off
    the head of the token stream by fraction, so a single-file corpus leaves
    val.bin with no boundary inside it: that split is one document."""
    raw = tmp_path / "corpus.txt"
    raw.write_text("the quick brown fox jumps over the lazy dog\n" * 8,
                   encoding="utf-8")
    out = tmp_path / "tokens"

    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT / "src"))
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "tools" / "tokenize_text.py"),
         "--input", str(raw), "--out", str(out), "--tokenizer", "byte",
         "--val-fraction", "0.1", "--pack"],
        capture_output=True, text=True, env=env)
    assert result.returncode == 0, result.stdout + result.stderr
    val_tokens = list((out / "val.bin").read_bytes())
    assert ByteTokenizer().eos_id not in val_tokens, "the split kept a boundary"

    seq_len = 63
    data = load_data(DataConfig(dataset=str(out), sequence_length=seq_len,
                                batch_size=1, worker_count=0, pack_sequences=True))
    row = next(iter(data["val"]()))

    padding = seq_len + 1 - len(val_tokens)
    np.testing.assert_array_equal(row["text"][0], val_tokens + [0] * padding)
    np.testing.assert_array_equal(row["text_segment_ids"][0],
                                  [1] * len(val_tokens) + [0] * padding)
    np.testing.assert_array_equal(row["text_positions"][0, :len(val_tokens)],
                                  np.arange(len(val_tokens)))


# ---------------------------------------------------------------------------------
# Stream termination on the token paths
# ---------------------------------------------------------------------------------

# Both token factories hand their validation sampler the num_epochs a run
# passes, which is None, so today a validation pass never ends and the same
# windows come round again. wave/fix-val-split owns that; the tests carrying
# this reason are the contract its fix has to meet and they fail until it lands.
ENDLESS_VAL = "an endless validation pass; wave/fix-val-split owns the sampler"


def _bounded(loader, limit):
    """At most `limit` batches, and whether the stream ended inside them.

    Bounded on purpose: an endless stream then fails a count instead of
    hanging the suite.
    """
    iterator = iter(loader)
    taken = list(itertools.islice(iterator, limit))
    return taken, next(iterator, None) is None


def _rows(batches):
    return [row.tobytes() for batch in batches for row in batch["text"]]


def test_a_token_validation_pass_ends_when_the_split_runs_out(tmp_path):
    """Eight windows at batch four are two batches, then the end."""
    seq_len = 4
    _token_dir(tmp_path, train_tokens=13 * seq_len, val_tokens=9 * seq_len,
               seq_len=seq_len)
    data = get_token_dataset_grain(
        str(tmp_path / "train.bin"), str(tmp_path / "val.bin"),
        batch_size=4, seq_len=seq_len, seed=0, worker_count=0)

    batches, ended = _bounded(data["val"](), 3)

    assert data["val_len"] == 8
    assert len(batches) == 2 and ended, ENDLESS_VAL
    assert len(set(_rows(batches))) == 8, "a pass must not repeat a window"


def test_a_token_validation_pass_stops_at_the_last_full_batch(tmp_path):
    """Ten windows at batch four are two batches, in file order, then the end.

    Validation batches keep drop_remainder: a part-full batch cannot be
    sharded across the data axis, and it would weigh as much as a full one in
    the metric reducers. The two windows past the last full batch are simply
    not scored, which is a reason to hold out a whole number of batches.
    """
    seq_len = 4
    val_tokens = np.arange(900, 900 + 11 * seq_len, dtype=np.int64)
    _token_dir(tmp_path, train_tokens=13 * seq_len, val_tokens=len(val_tokens),
               seq_len=seq_len)
    (tmp_path / "val.bin").write_bytes(val_tokens.astype("<u2").tobytes())
    data = get_token_dataset_grain(
        str(tmp_path / "train.bin"), str(tmp_path / "val.bin"),
        batch_size=4, seq_len=seq_len, seed=0, worker_count=0)

    batches, ended = _bounded(data["val"](), 3)

    assert data["val_len"] == 10
    assert len(batches) == data["val_len"] // 4 and ended, ENDLESS_VAL
    assert len(set(_rows(batches))) == 8, "a pass must not repeat a window"
    np.testing.assert_array_equal(batches[0]["text"][0], val_tokens[:seq_len + 1])


def test_a_packed_validation_pass_reads_each_window_once_and_stops(tmp_path):
    """No document twice, every batch full, and the same batches next time.

    Bins come out in packing order for validation, so two passes are the same
    windows; today the pass never ends and the documents come round again.
    """
    documents = [[i, i + 1, i + 2] for i in range(10, 40, 3)]
    _document_dir(tmp_path, documents)
    data = get_packed_token_dataset_grain(
        str(tmp_path / "train.bin"), str(tmp_path / "val.bin"),
        batch_size=2, seq_len=8, seed=0, worker_count=0, num_packing_bins=2)

    batches, ended = _bounded(data["val"](), 2 + len(documents))
    heads = [int(t) for batch in batches for row in batch["text"] for t in row
             if int(t) in {d[0] for d in documents}]
    again, _ = _bounded(data["val"](), len(batches))

    assert ended, ENDLESS_VAL
    assert all(batch["text"].shape[0] == 2 for batch in batches)
    assert sorted(heads) == sorted(set(heads)), "a document was handed over twice"
    assert _rows(again) == _rows(batches), "two passes read different windows"


def test_a_validation_pass_through_load_data_ends(tmp_path):
    """A run reaches these loaders through load_data, which passes no
    num_epochs at all."""
    seq_len = 4
    _token_dir(tmp_path, train_tokens=13 * seq_len, val_tokens=9 * seq_len,
               seq_len=seq_len)
    data = load_data(DataConfig(dataset=str(tmp_path), sequence_length=seq_len,
                                batch_size=4, worker_count=0))

    batches, ended = _bounded(data["val"](), 3)

    assert len(batches) == 2 and ended, ENDLESS_VAL


def test_the_token_training_stream_repeats_rather_than_ending(tmp_path):
    """The trainer keeps asking long after one pass over the windows, and the
    fixed-window loader is the one path that has no `repeat` of its own."""
    seq_len = 4
    _token_dir(tmp_path, train_tokens=9 * seq_len, val_tokens=5 * seq_len,
               seq_len=seq_len)
    data = get_token_dataset_grain(
        str(tmp_path / "train.bin"), str(tmp_path / "val.bin"),
        batch_size=4, seq_len=seq_len, seed=0, worker_count=0)

    epoch = data["train_len"] // 4
    batches, ended = _bounded(data["train"](), 3 * epoch)

    assert data["train_len"] == 8 and not ended
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
    stream = np.concatenate([np.asarray(d + [PACK_EOS], np.int64) for d in documents])
    _token_dir(tmp_path, train_tokens=0, body=stream, eos_id=PACK_EOS)
    (tmp_path / "val.bin").write_bytes(stream.astype(np.uint16).tobytes())
    data = get_packed_token_dataset_grain(
        str(tmp_path / "train.bin"), str(tmp_path / "val.bin"), batch_size=batch,
        seq_len=seq_len, seed=0, worker_count=0, num_epochs=1, num_packing_bins=bins)
    return data, list(data["val"]())


def _tiny_backbone(seq_len):
    return backbone.CausalTransformer(vocab_size=64, emb_features=16, num_layers=1,
                                      num_heads=2, mlp_ratio=2,
                                      max_seq_len=seq_len + 1)


def _attention_mask(batch, monkeypatch):
    """The mask the backbone hands the attention kernel for this batch.

    Recorded at the kernel call, which is the only place the mask has to be
    right; rebuilding it from the segment ids here would test the test.
    """
    seen = []
    kernel = backbone.scaled_dot_product_attention

    def recording_kernel(query, key, value, **kwargs):
        seen.append(kwargs.get("mask"))
        return kernel(query, key, value, **kwargs)

    tokens = jnp.asarray(batch["text"][:, :-1], jnp.int32)
    model = _tiny_backbone(tokens.shape[1])
    params = model.init(jax.random.PRNGKey(0), tokens)

    monkeypatch.setattr(backbone, "scaled_dot_product_attention", recording_kernel)
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


def _counted_cross_entropy(batch, seq_len):
    """The objective's ce, and the same number computed by hand.

    The hand version averages the per-token losses over the transitions that
    live inside one document, which is every target a packed row owns.
    """
    model = _tiny_backbone(seq_len)
    objective = LMObjective(model, seq_len, vocab_size=64)
    params = objective.init_params(jax.random.PRNGKey(0))
    tokens = jnp.asarray(batch["text"], jnp.int32)
    segment_ids = jnp.asarray(batch["text_segment_ids"], jnp.int32)
    positions = jnp.asarray(batch["text_positions"], jnp.int32)

    ce, _ = objective.shifted_cross_entropy(params, tokens, segment_ids=segment_ids,
                                            positions=positions)

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
    data, batches = _packed(tmp_path, [[2, 3, 4, 5]], seq_len=4, bins=1)

    assert data["val_len"] == 1 and len(batches) == 1
    np.testing.assert_array_equal(batches[0]["text"][0], [2, 3, 4, 5, PACK_EOS])
    np.testing.assert_array_equal(batches[0]["text_segment_ids"][0], [1] * 5)
    np.testing.assert_array_equal(batches[0]["text_positions"][0], range(5))
    _assert_mask_blocks_everything_it_should(batches[0], monkeypatch)


def test_three_documents_in_one_window_cannot_read_each_other(tmp_path, monkeypatch):
    data, batches = _packed(tmp_path, [[2, 3], [4, 5], [6, 7]], seq_len=8, bins=1)
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
    data, batches = _packed(tmp_path, [list(range(2, 14))], seq_len=4, bins=1)

    assert len(batches) == 3
    np.testing.assert_array_equal(
        np.concatenate([b["text"][0] for b in batches])[:13],
        list(range(2, 14)) + [PACK_EOS])
    for batch in batches:
        np.testing.assert_array_equal(batch["text_positions"][0][:1], [0])
        _assert_mask_blocks_everything_it_should(batch, monkeypatch)


def test_a_document_of_one_token_is_its_own_segment(tmp_path, monkeypatch):
    """A split can end on a bare boundary, and a one-token document owns no
    transition: it must not borrow the previous document's."""
    data, batches = _packed(tmp_path, [[2, 3, 4], []], seq_len=8, bins=1)
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
    data, batches = _packed(tmp_path, [[2, 3]], seq_len=8, bins=8)
    row = batches[0]

    np.testing.assert_array_equal(row["text"][0], [2, 3, 1, 0, 0, 0, 0, 0, 0])
    np.testing.assert_array_equal(row["text_segment_ids"][0], [1, 1, 1, 0, 0, 0, 0, 0, 0])
    _assert_mask_blocks_everything_it_should(row, monkeypatch)

    ce, by_hand, counted = _counted_cross_entropy(row, seq_len=8)
    assert counted == 2, "the padding owns no target"
    assert ce == pytest.approx(by_hand, rel=1e-5)


def test_the_loss_counts_one_target_per_transition_inside_a_document(
        tmp_path, monkeypatch):
    """Three documents in a row own six targets between them: the two boundary
    transitions and the padding are not the model's to predict."""
    data, batches = _packed(tmp_path, [[2, 3], [4, 5], [6, 7]], seq_len=8, bins=1)

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
    _token_dir(tmp_path, train_tokens=17 * seq_len, seq_len=seq_len)

    def windows(workers):
        data = get_token_dataset_grain(
            str(tmp_path / "train.bin"), str(tmp_path / "train.bin"),
            batch_size=4, seq_len=seq_len, seed=7, worker_count=workers,
            num_epochs=1)
        return sorted(row.tobytes() for batch in data["train"]()
                      for row in batch["text"])

    serial = windows(0)

    assert len(serial) == 16
    assert windows(worker_count) == serial


@pytest.mark.slow
def test_an_interrupted_token_epoch_resumes_through_real_workers(tmp_path):
    """A resumed run builds its loader again, in a new process, and grain
    checks the saved position against `repr(source)` before it will restore:
    the description has to name the file, not an address in the process that
    wrote it."""
    seq_len = 8
    _token_dir(tmp_path, train_tokens=33 * seq_len, seq_len=seq_len)

    def loader():
        return get_token_dataset_grain(
            str(tmp_path / "train.bin"), str(tmp_path / "train.bin"),
            batch_size=4, seq_len=seq_len, seed=7, worker_count=2, num_epochs=1)

    interrupted = iter(loader()["train"]())
    seen = [row.tobytes() for row in next(interrupted)["text"]]
    state = interrupted.get_state()
    rest, ended = _bounded(interrupted, 40)
    unseen = [row.tobytes() for batch in rest for row in batch["text"]]

    restored = iter(loader()["train"]())
    restored.set_state(state)
    after, ended_again = _bounded(restored, 40)
    resumed = [row.tobytes() for batch in after for row in batch["text"]]

    assert "object at 0x" not in json.loads(state)["data_source"], (
        "a source described by its address can only be restored in the process "
        "that saved it")
    assert ended and ended_again
    assert resumed == unseen
    assert len(set(seen + resumed)) == 32, "the epoch reads every window once"


@pytest.mark.slow
def test_an_interrupted_packed_epoch_resumes_through_mp_prefetch(tmp_path):
    """The packed loader reads its documents in worker processes and packs
    behind them, so a restart has to carry the packer's position too."""
    _document_dir(tmp_path, [[i, i + 1, i + 2] for i in range(10, 100, 3)])

    def loader():
        return get_packed_token_dataset_grain(
            str(tmp_path / "train.bin"), str(tmp_path / "val.bin"), batch_size=2,
            seq_len=8, seed=0, worker_count=2, num_epochs=1, num_packing_bins=2)

    interrupted = iter(loader()["val"]())
    seen = [row.tobytes() for row in next(interrupted)["text"]]
    state = interrupted.get_state()
    rest, ended = _bounded(interrupted, 40)
    unseen = [row.tobytes() for batch in rest for row in batch["text"]]

    restored = iter(loader()["val"]())
    restored.set_state(state)
    after, ended_again = _bounded(restored, 40)
    resumed = [row.tobytes() for batch in after for row in batch["text"]]

    assert unseen and ended and ended_again
    assert resumed == unseen
    assert len(set(seen + resumed)) == len(seen) + len(unseen), "a window came twice"
