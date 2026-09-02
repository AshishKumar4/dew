"""Text data for language models: tokenizers, the token-file source, the loader.

ByteTokenizer and TokenFileSource are pure numpy; the HF tokenizer tests only
run when a cached copy of the hub is reachable, matching the repo's policy
that no test needs the network.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from dew.config import DataConfig
from dew.data.dataloaders import get_token_dataset_grain, load_data
from dew.data.sources.text import TokenFileSource
from dew.data.text import ByteTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]

# grain's worker processes read absl flags; a test that never ran absl.app
# would trip UnparsedFlagAccessError at any worker_count > 0.
from absl import flags

if not flags.FLAGS.is_parsed():
    flags.FLAGS.mark_as_parsed()


def _token_dir(tmp_path, train_tokens, val_tokens=None, seq_len=8, dtype=np.uint16,
               vocab_size=256, body=None):
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
    (tmp_path / "meta.json").write_text(json.dumps({
        "tokenizer": "byte", "vocab_size": vocab_size,
        "dtype": np.dtype(dtype).name,
        "train_tokens": len(train), "val_tokens": len(val) if val is not None else 0,
    }))
    return tmp_path


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
