"""Text data for language models: tokenizers, token stores, and the presets that lay their tokens out
in windows (`presets.token_windows`) or pack their documents (`presets.lm`).

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
from data_support import parse_grain_flags
from token_support import encoded_token_fixture, export_byte_tokens, write_token_corpus

from dew.batch import VALID_ROWS
from dew.data import ByteTokenizer, Loading, presets
from dew.data.compile import compile, held
from dew.data.dataset import DataPartition
from dew.data.pipeline import Dataset
from dew.data.sources.tokens import TokenFiles
from dew.nn.backbones import causal_transformer as backbone
from dew.objectives.lm import LMObjective
from dew.training import Step

parse_grain_flags()

REPO_ROOT = Path(__file__).resolve().parents[1]

LOADING = Loading(workers=0, threads=1, read_buffer=1, worker_buffer=1)


def _laid(rows: Dataset, seq_len: int, batch: int, *, stride: int | None = None, pack: str | None = None,
          bins: int = 8) -> Dataset:
    """A token store's tokens cut into windows of `seq_len` inputs `stride` apart, or its documents
    packed into rows of `seq_len` inputs (`pack`, `bins` rows open at once), in batches of `batch`."""
    if pack is None:
        return rows.pipe(presets.token_windows, seq_len=seq_len, batch=batch, stride=stride)
    return rows.pipe(presets.lm, None, seq_len=seq_len, batch=batch, pack=pack, bins=bins)


def _store(store: str | TokenFiles) -> TokenFiles:
    return store if isinstance(store, TokenFiles) else TokenFiles(str(store))


def _trained(store, seq_len, batch, *, stride=None, pack=None, bins=8, seed=0, read="memory",
             loading=LOADING):
    """A token store as a run trains on it: its windows shuffled by `seed` and repeated, or its
    documents packed in the store's order and the rows they make shuffled and repeated."""
    stored = _store(store)
    if pack is None:
        chain = _laid(stored.shuffle(seed).repeat(), seq_len, batch, stride=stride)
    else:
        rows = stored.pipe(presets.lm, None, seq_len=seq_len, batch=None, pack=pack, bins=bins)
        chain = rows.shuffle(seed).repeat().batch(batch)
    return compile(chain, read=read, loading=loading, seed=seed)


def _validated(store, seq_len, batch, *, pack=None, bins=8, read="memory", loading=LOADING):
    """A token store read once to validate on, in its order, laid out (`_laid`): its windows
    `seq_len` apart, or its documents packed."""
    chain = _laid(_store(store), seq_len, batch, pack=pack, bins=bins).read(read).loading(loading)
    return held(chain)(DataPartition())


def _documents_corpus(directory, documents, eos_id=0):
    """A corpus whose training and validation stores both hold `documents`, each closed by
    `eos_id`, and its token stream."""
    stream = np.concatenate([np.asarray([*document, eos_id], np.int64) for document in documents])
    return write_token_corpus(directory, stream, stream, eos_id=eos_id), stream


def _read(batches, count=None):
    """The rows of `batches`, the first `count` batches of an endless stream, which it closes."""
    try:
        return [jax.tree.map(lambda value, row=row: value[row],
                             {key: value for key, value in batch.items() if key != VALID_ROWS})
                for batch in itertools.islice(batches, count) for row in range(len(batch["text"]))]
    finally:
        batches.close()


# ---------------------------------------------------------------------------------
# The windows of a token stream
# ---------------------------------------------------------------------------------

@pytest.mark.parametrize("seq_len,stride,tokens", [
    (4, 1, 5), (4, 1, 6), (4, 1, 13), (4, 2, 13), (4, 3, 13), (4, 4, 13), (4, 6, 13),
    (8, 3, 22), (8, None, 37), (1, None, 7),
])
def test_an_epoch_reads_every_complete_window_stride_apart(tmp_path, seq_len, stride, tokens):
    """One training epoch is each complete window of `seq_len + 1` int32 ids
    starting `stride` apart (`seq_len` by default) once, the incomplete tail
    left out; validation tiles the split `seq_len` apart, in file order."""
    body = np.arange(tokens)
    corpus = write_token_corpus(tmp_path, body, body)
    data = _trained(corpus.train, seq_len, 1, stride=stride)
    starts = range(0, tokens - seq_len, stride or seq_len)

    epoch = _read(data.train(DataPartition()), data.records)
    assert data.records == len(starts)
    assert all(row["text"].dtype == np.int32 for row in epoch)
    assert sorted(row["text"].tolist() for row in epoch) == [body[s:s + seq_len + 1].tolist() for s in starts]
    assert [row["text"].tolist() for row in _read(_validated(corpus.val, seq_len, 1))] == [
        body[s:s + seq_len + 1].tolist() for s in range(0, tokens - seq_len, seq_len)]


def test_a_stride_changes_the_training_windows_only(tmp_path):
    seq_len, batch = 4, 4
    corpus = write_token_corpus(tmp_path, np.arange(17, 34), np.arange(17))
    default = _trained(corpus.train, seq_len, batch, seed=17)
    explicit = _trained(corpus.train, seq_len, batch, stride=seq_len, seed=17)
    overlap = _trained(corpus.train, seq_len, batch, stride=1, seed=17)
    # 13 windows a pass in batches of four: four steps, the last leading into the next pass.
    assert overlap.records == 13 and overlap.steps_per_epoch == 4
    assert default.records == explicit.records == 4
    assert default.steps_per_epoch == explicit.steps_per_epoch == 1
    assert overlap.epoch_steps(3) == 10

    def first_rows(data, steps):
        stream = data.train(DataPartition())
        try:
            return np.concatenate([next(stream)["text"] for _ in range(steps)])
        finally:
            stream.close()

    assert first_rows(default, 5).tobytes() == first_rows(explicit, 5).tobytes()
    rows = first_rows(overlap, overlap.steps_per_epoch + 1)
    assert rows.dtype == np.int32
    assert sorted(rows[:overlap.records, 0]) == list(range(17, 30)), "one overlapping epoch"
    assert rows.tobytes() == first_rows(overlap, overlap.steps_per_epoch + 1).tobytes(), "same seed and order"
    validation = _validated(corpus.val, seq_len, batch)
    try:
        actual = np.concatenate([batch["text"] for batch in validation])
    finally:
        validation.close()
    expected = np.stack([np.arange(start, start + seq_len + 1, dtype=np.int32)
                         for start in range(0, 16, seq_len)])
    assert actual.tobytes() == expected.tobytes()
    np.testing.assert_array_equal(actual[:-1, -1], actual[1:, 0])


def test_validation_is_unshuffled_and_disjoint_from_training(tmp_path):
    seq_len = 4
    # 13*4 tokens -> 12 train windows; 9*4 -> 8 val windows.
    train_tokens = np.arange(100, 100 + 13 * seq_len, dtype=np.int64)
    val_tokens = np.arange(900, 900 + 9 * seq_len, dtype=np.int64)
    corpus = write_token_corpus(tmp_path, train_tokens, val_tokens)
    data = _trained(corpus.train, seq_len, 4)

    val_batches = [batch["text"] for batch in _validated(corpus.val, seq_len, 4)]
    assert len(val_batches) == 2
    # Unshuffled: windows come in file order, each overlapping the next by one.
    np.testing.assert_array_equal(val_batches[0][0], val_tokens[:seq_len + 1])
    np.testing.assert_array_equal(val_batches[1][0], val_tokens[4 * seq_len:5 * seq_len + 1])

    # Train and val are disjoint stores: no train window equals a val window.
    train_windows = {window.tobytes() for window in np.concatenate(
        [batch["text"] for batch in itertools.islice(data.train(DataPartition()), data.steps_per_epoch)])}
    val_windows = {window.tobytes() for window in np.concatenate(val_batches)}
    assert not (train_windows & val_windows)


@pytest.mark.parametrize("worker_count", [0, 2])
def test_a_validation_pass_reads_every_window_once(tmp_path, worker_count):
    """A validation pass reads every window of its store once and ends, in its own workers too."""
    seq_len = 4
    val_tokens = np.arange(900, 900 + 13 * seq_len, dtype=np.int64)  # 12 windows
    corpus = write_token_corpus(tmp_path, np.arange(5 * seq_len), val_tokens)
    validation = _validated(corpus.val, seq_len, 4, loading=Loading(workers=worker_count))
    windows = [list(window) for batch in itertools.islice(validation, 12) for window in batch["text"]]
    assert windows == [list(val_tokens[start:start + seq_len + 1])
                       for start in range(0, 12 * seq_len, seq_len)]


def test_the_seed_decides_the_training_order(tmp_path):
    """The same seed reads the same first batch twice, a different seed reads a different one."""
    seq_len, records = 8, 24
    corpus = write_token_corpus(tmp_path, np.random.RandomState(0).randint(0, 256, (records + 1) * seq_len))

    def first_batch(seed):
        data = _trained(corpus.train, seq_len, 4, seed=seed, loading=Loading(workers=0))
        stream = data.train(DataPartition())
        try:
            return next(stream)["text"]
        finally:
            stream.close()

    assert np.array_equal(first_batch(0), first_batch(0))
    assert not np.array_equal(first_batch(0), first_batch(1))


# ---------------------------------------------------------------------------------
# Produced token stores

def export_documents(documents, destination, *, tokenizer="byte", eos=False):
    from dew.data import Memory
    from dew.data.pipeline import Pipeline
    from dew.data.sinks import export
    from dew.data.text import Tokenize, tokenizer_for

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
    assert stored.metadata()["tokens"] == len(expected)
    data = _trained(stored, 32, 1, loading=Loading(workers=0))
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


FIXTURES = Path(__file__).resolve().parent / "fixtures"


def eos_tokenizer(directory: Path) -> str:
    """An EOS-inserting Llama: the SentencePiece byte-fallback fixture with `</s>` appended to every
    encoding by its post-processor, saved to `directory`."""
    from tokenizers import Tokenizer, processors
    from transformers import PreTrainedTokenizerFast

    backend = Tokenizer.from_file(str(FIXTURES / "llama_cpp" / "tokenizer" / "tokenizer.json"))
    eos = backend.token_to_id("</s>")
    backend.post_processor = processors.TemplateProcessing(single="$A </s>",
                                                           special_tokens=[("</s>", eos)])
    PreTrainedTokenizerFast(tokenizer_object=backend, eos_token="</s>", bos_token="<s>",
                            unk_token="<unk>").save_pretrained(directory)
    return str(directory)


@pytest.mark.parametrize("name", ["gpt2", "t5", "llama", "eos-llama", "wordpiece"])
def test_a_tokenized_document_ends_in_one_eos_with_eos_and_none_without(tmp_path, name):
    """Whatever a tokenizer's own special tokens add (T5's and an EOS-inserting Llama's `</s>`,
    WordPiece's [CLS] and [SEP]), a document is its encoding ended by the tokenizer's EOS once with
    `eos=True` where it has one, and by none with `eos=False`: never two, and none left behind."""
    from token_support import wordpiece_tokenizer

    from dew.data.text import Tokenize, tokenizer_for

    if name == "eos-llama":
        tokenizer = eos_tokenizer(tmp_path / name)
    elif name == "wordpiece":
        wordpiece_tokenizer(tmp_path / name)
        tokenizer = str(tmp_path / name)
    else:
        tokenizer = str({"gpt2": FIXTURES / "tokenizers" / "gpt2", "t5": FIXTURES / "t5" / "tiny",
                         "llama": FIXTURES / "llama_cpp" / "tokenizer"}[name])
    vocabulary = tokenizer_for(tokenizer)
    text = "the quick brown fox"
    bare = Tokenize(tokenizer, eos=False).map({"text": text})["text"].tolist()
    assert vocabulary.eos_id is None or vocabulary.eos_id not in bare
    if vocabulary.eos_id is None:
        with pytest.raises(ValueError):
            Tokenize(tokenizer, eos=True).map({"text": text})
        return
    ended = Tokenize(tokenizer, eos=True).map({"text": text})["text"].tolist()
    assert ended == [*bare, vocabulary.eos_id]
    assert Tokenize(tokenizer, eos=None).map({"text": text})["text"].tolist() == ended


def _export_cli(input_path, destination):
    """`dew data export` of a run file whose chain reads `input_path` as text, a line a document,
    whatever its name, into `destination` as byte tokens."""
    import subprocess

    chain = destination.parent / "corpus_chain.py"
    chain.write_text(f"from dew.data import Files\n\ndata = Files({str(input_path)!r}, format='text')\n")
    return subprocess.run([sys.executable, "-m", "dew.cli.main", "data", "export", str(chain),
                           str(destination), "--format", "array_record", "--tokenizer", "byte",
                           "--read", "memory"], cwd=destination.parent,
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

def _laid_rows(rows) -> list[tuple[tuple[int, ...], ...]]:
    """Each row's tokens, and its documents and their positions as its coordinates lay them out."""
    return [(tuple(row["text"].tolist()), tuple(row["text_coordinates"].segments.tolist()),
             tuple(row["text_coordinates"].positions.tolist())) for row in rows]


@pytest.mark.parametrize("read", ["indexed", "streamed", "memory"])
@pytest.mark.parametrize("eos_id,ends,segments", [
    (0, [4, 7, 9], [1, 1, 1, 1, 2, 2, 2, 3, 3]),
    (9, [9], [1] * 9),
])
def test_a_document_runs_through_its_eos_and_the_tail_past_the_last_is_one_too(
        tmp_path, read, eos_id, ends, segments):
    stream = np.asarray([10, 11, 12, 0, 20, 21, 0, 30, 31], np.uint16)
    corpus = write_token_corpus(tmp_path, stream, stream, eos_id=eos_id)
    stored = TokenFiles(corpus.val)
    np.testing.assert_array_equal(stored.lengths(), np.diff(ends, prepend=0))
    records = stored.records(read)
    assert [records[index]["text"].tolist() for index in range(len(records))] == [
        stream[start:stop].tolist() for start, stop in zip([0, *ends[:-1]], ends, strict=True)]
    (window,) = _read(_validated(stored, 15, 1, pack="first-fit", bins=1, read=read,
                                 loading=Loading(workers=0, threads=1, read_buffer=1)))
    assert window["text"].tolist() == [*stream, *[0] * 7]
    assert window["text_coordinates"].segments.tolist() == [*segments, *[0] * 7]


@pytest.mark.parametrize("pack", ["first-fit", "best-fit", "concat"])
@pytest.mark.parametrize("read", ["indexed", "streamed", "memory"])
def test_packed_windows_keep_every_next_token_target_of_a_document_longer_than_a_window(tmp_path, read,
                                                                                       pack):
    """A document cut across packed windows keeps its boundary token on both sides of the cut, so
    every token's next token follows it inside one segment of some window: the windows' pairs are
    the documents' pairs, each once, in every read and packing. A fitting pack's documents are cut
    into pieces before it, and a concatenating pack's where its windows end, inside a document that
    starts partway into a window too."""
    stream = np.asarray([10, 11, 12, 13, 14, 15, 16, 17, 0, 20, 21, 22, 23, 24, 25, 0, 30, 31, 0], np.uint16)
    corpus = write_token_corpus(tmp_path, stream, stream, eos_id=0)
    pairs = Counter()
    validation = _validated(corpus.val, 4, 1, pack=pack, bins=2, read=read,
                            loading=Loading(workers=0, threads=1, read_buffer=1))
    for window in _read(validation):
        tokens, segments = window["text"].tolist(), window["text_coordinates"].segments.tolist()
        pairs.update((tokens[index], tokens[index + 1]) for index in range(len(tokens) - 1)
                     if segments[index] and segments[index] == segments[index + 1])
    documents = [stream[:9].tolist(), stream[9:16].tolist(), stream[16:].tolist()]
    assert pairs == Counter(pair for document in documents for pair in itertools.pairwise(document))


def test_packed_rows_fill_windows_with_whole_documents(tmp_path):
    seq_len = 8  # windows of 9 ids
    # 4, 6 and 5 ids once each eos is counted: first fit puts 4 + 5 in one
    # window and 6 in the next.
    corpus, _ = _documents_corpus(tmp_path, [[10, 11, 12], [20, 21, 22, 23, 24], [30, 31, 32, 33]])
    data = _trained(corpus.train, seq_len, 2, pack="first-fit", bins=2)

    # Two windows hold the three documents, and records is that pass exactly.
    assert data.records == 2 and data.batch == 2 and data.steps_per_epoch == 1
    batch = next(_validated(corpus.val, seq_len, 2, pack="first-fit", bins=2))

    laid = batch["text_coordinates"]
    assert set(batch) == {"text", "text_coordinates"}
    # The most documents a window holds (two), not the window's length.
    assert laid.max_segments == int(laid.segments.max()) == 2 and laid.right_padded
    assert all(isinstance(value, np.ndarray) for value in jax.tree.leaves(laid))
    for value in (batch["text"], laid.segments, laid.positions):
        assert value.shape == (2, seq_len + 1)
        assert value.dtype == np.int32
    np.testing.assert_array_equal(batch["text"], [
        [10, 11, 12, 0, 30, 31, 32, 33, 0],
        [20, 21, 22, 23, 24, 0, 0, 0, 0]])
    np.testing.assert_array_equal(laid.segments, [
        [1, 1, 1, 1, 2, 2, 2, 2, 2],
        [1, 1, 1, 1, 1, 1, 0, 0, 0]])
    # Positions restart at 0 inside every document; RoPE reads them.
    np.testing.assert_array_equal(laid.positions, [
        [0, 1, 2, 3, 0, 1, 2, 3, 4],
        [0, 1, 2, 3, 4, 5, 0, 0, 0]])
    np.testing.assert_array_equal(laid.starts, [
        [False, False, False, False, True, False, False, False, False],
        [False, False, False, False, False, False, True, False, False]])
    assert laid.starts.dtype == np.bool_


def test_packing_cuts_documents_that_outgrow_the_window(tmp_path):
    """Grain's packer refuses an over-long element, so the preset cuts first,
    into pieces that share their boundary token so each keeps its next-token
    target; each piece is its own segment and its positions start again at 0."""
    seq_len = 3  # windows of 4 ids
    corpus, _ = _documents_corpus(tmp_path, [list(range(10, 19))])  # one 10-id document
    batches = list(_validated(corpus.val, seq_len, 1, pack="first-fit", bins=1))
    # Pieces 3 ids apart, one per window, each repeating the last id of the one before it.
    np.testing.assert_array_equal([batch["text"][0] for batch in batches],
                                  [[10, 11, 12, 13], [13, 14, 15, 16], [16, 17, 18, 0]])
    for batch in batches:
        np.testing.assert_array_equal(batch["text_coordinates"].segments[0], [1, 1, 1, 1])
        np.testing.assert_array_equal(batch["text_coordinates"].positions[0], [0, 1, 2, 3])


def test_a_store_packs_its_documents_whole_unless_a_chunk_step_cuts_them(tmp_path):
    """A store's planned pack places each document whole and refuses one longer than a
    window; a chunk step before the pack cuts it into window-long pieces, which the plan
    of the store's document lengths cuts alike, each piece a window of its own here."""
    from dew.data.text import Chunks

    corpus, _ = _documents_corpus(tmp_path, [list(range(10, 19))])  # one 10-id document; windows of 4
    stored = TokenFiles(corpus.train)
    with pytest.raises(ValueError):
        compile(stored.pack(4, bins=1).repeat().batch(1), read="indexed", provision=True, loading=LOADING)
    data = compile(stored.flat_map(Chunks(4)).pack(4, bins=1).repeat().batch(1), read="indexed",
                   provision=True, loading=LOADING)
    assert data.plan.lowering == "global"
    rows = [row["text"].tolist() for row in _read(data.train(DataPartition()), 3)]
    assert rows == [[10, 11, 12, 13], [14, 15, 16, 17], [18, 0, 0, 0]]


def test_a_packed_pass_resumes_on_the_next_unseen_batch(tmp_path):
    """The trainer saves the iterator's position in the checkpoint, so a
    restored iterator has to carry on where the saved one had got to."""
    corpus, _ = _documents_corpus(tmp_path, [[i, i + 1, i + 2] for i in range(10, 60, 3)])
    iterator = _validated(corpus.val, 8, 2, pack="first-fit", bins=2)
    next(iterator)
    state = iterator.get_state()
    expected = [next(iterator)["text"] for _ in range(2)]

    restored = _validated(corpus.val, 8, 2, pack="first-fit", bins=2)
    restored.set_state(state)
    for wanted, got in zip(expected, [next(restored)["text"] for _ in range(2)], strict=True):
        np.testing.assert_array_equal(wanted, got)


def test_packed_windows_do_not_depend_on_the_worker_count(tmp_path):
    corpus, _ = _documents_corpus(tmp_path, [[i, i + 1, i + 2] for i in range(10, 70, 3)])

    def windows(worker_count):
        validation = _validated(corpus.val, 8, 2, pack="first-fit", bins=2,
                                loading=Loading(workers=worker_count, worker_buffer=1))
        return sorted(row.tobytes() for batch in validation for row in batch["text"])

    serial = windows(0)
    assert serial, "the preset produced no windows"
    assert serial == windows(2)


@pytest.mark.parametrize("document_width", [2, 3])
def test_a_packed_validation_pass_covers_the_split_once(tmp_path, document_width):
    """The documents packed for validation are read once, without the
    training stream's repeat, so the pass finishes and scores each document
    once."""
    documents = [list(range(i, i + document_width))
                 for i in range(10, 10 + 10 * document_width, document_width)]
    corpus, _ = _documents_corpus(tmp_path, documents)
    batches, ended = _bounded(_validated(corpus.val, 8, 2, pack="first-fit", bins=2), 20)
    # Ten documents of three ids (the eos counts) pack three to a nine-id
    # window, so four windows and two batches of two.
    assert len(batches) == (2 if document_width == 2 else 3) and ended, ENDLESS_VAL
    read = Counter(int(token) for row in _rows(batches)
                   for token in np.frombuffer(row, batches[0]["text"].dtype) if token)
    assert read == Counter(token for document in documents for token in document)
    again, _ = _bounded(_validated(corpus.val, 8, 2, pack="first-fit", bins=2), len(batches))
    assert _rows(again) == _rows(batches)


def test_a_single_file_corpus_packs_when_its_validation_store_holds_no_eos(tmp_path):
    """Encoding closes each input file with one eos and the validation tokens are cut off the
    head of the token stream by fraction, so a single-file corpus leaves its validation store with
    no boundary inside it: that store is one document."""
    raw = tmp_path / "corpus.txt"
    raw.write_text("the quick brown fox jumps over the lazy dog\n" * 8, encoding="utf-8")
    corpus = encoded_token_fixture(tmp_path / "tokens", [raw.read_text()], val_fraction=0.1, eos=True)
    val_tokens = TokenFiles(corpus.val).tokens[:].tolist()
    assert ByteTokenizer().eos_id not in val_tokens, "the split kept a boundary"

    seq_len = 63
    row = next(_validated(corpus.val, seq_len, 1, pack="first-fit"))

    padding = seq_len + 1 - len(val_tokens)
    np.testing.assert_array_equal(row["text"][0], val_tokens + [0] * padding)
    np.testing.assert_array_equal(row["text_coordinates"].segments[0], [1] * len(val_tokens) + [0] * padding)
    np.testing.assert_array_equal(row["text_coordinates"].positions[0, :len(val_tokens)],
                                  np.arange(len(val_tokens)))


# ---------------------------------------------------------------------------------
# Stream termination on the token paths
# ---------------------------------------------------------------------------------

# A validation pass that never ends scores the same windows again and again;
# the tests carrying this reason are the contract a validation stream meets.
ENDLESS_VAL = "an endless validation pass"


def _bounded(stream, limit):
    """At most `limit` batches, and whether the stream ended inside them.

    Bounded on purpose: an endless stream then fails a count instead of
    hanging the suite.
    """
    iterator = iter(stream)
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
    corpus = write_token_corpus(tmp_path, np.arange(13 * seq_len), val_tokens)
    batches, ended = _bounded(_validated(corpus.val, seq_len, 4), 4)

    assert len(batches) == (windows + 3) // 4 and ended, ENDLESS_VAL
    assert len(_rows(batches)) == len(set(_rows(batches))) == windows, "a pass reads every window once"
    if windows == 10:
        np.testing.assert_array_equal(batches[-1][VALID_ROWS], [True, True, False, False])
    np.testing.assert_array_equal(batches[0]["text"][0], val_tokens[:seq_len + 1])


def test_the_token_training_stream_repeats_rather_than_ending(tmp_path):
    """The trainer keeps asking long after one pass over the windows."""
    seq_len = 4
    corpus = write_token_corpus(tmp_path, np.arange(9 * seq_len), np.arange(5 * seq_len))
    data = _trained(corpus.train, seq_len, 4)

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
    stream = np.concatenate([np.asarray([*document, PACK_EOS], np.int64) for document in documents])
    corpus = write_token_corpus(tmp_path, stream, stream, eos_id=PACK_EOS)
    return list(_validated(corpus.val, seq_len, batch, pack="first-fit", bins=bins))


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


@pytest.mark.parametrize("name", ["documents", "records", "packed_records"])
@pytest.mark.parametrize("pack", [None, "first-fit"])
def test_memory_windows_and_packing_never_reread_the_store(stores, monkeypatch, name, pack):
    import grain.python as grain

    stored = stores[name]
    data = _trained(stored, 7, 2, pack=pack, bins=2)

    def reread(self, index):
        pytest.fail("a materialized memory read went back to its token store")

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
    segment_ids = jnp.asarray(batch["text_coordinates"].segments, jnp.int32)
    positions = jnp.asarray(batch["text_coordinates"].positions, jnp.int32)

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
def test_token_windows_are_the_same_records_at_every_worker_count(tmp_path, worker_count):
    """Real worker processes, one epoch: the windows a pass reads are a
    function of the seed and the store, not of how many workers read them."""
    seq_len = 8
    corpus = write_token_corpus(tmp_path, np.random.RandomState(0).randint(0, 256, 17 * seq_len))

    def windows(workers):
        data = _trained(corpus.train, seq_len, 4, seed=7, loading=Loading(workers=workers))
        return sorted(row.tobytes()
                      for batch in itertools.islice(data.train(DataPartition()), data.steps_per_epoch)
                      for row in batch["text"])

    serial = windows(0)

    assert len(serial) == 16
    assert len(set(serial)) == 16
    assert windows(worker_count) == serial


@pytest.mark.slow
def test_an_interrupted_packed_pass_resumes_through_mp_prefetch(tmp_path):
    """The packed preset reads its documents in worker processes and packs
    behind them, so a restart has to carry the packer's position too."""
    corpus, _ = _documents_corpus(tmp_path, [[i, i + 1, i + 2] for i in range(10, 100, 3)])

    def opened():
        return _validated(corpus.val, 8, 2, pack="first-fit", bins=2,
                          loading=Loading(workers=2, worker_buffer=1))

    interrupted = opened()
    seen = _rows([next(interrupted)])
    state = interrupted.get_state()
    rest, ended = _bounded(interrupted, 40)
    unseen = _rows(rest)

    restored = opened()
    restored.set_state(state)
    after, ended_again = _bounded(restored, 40)
    resumed = _rows(after)

    assert unseen and ended and ended_again
    assert resumed == unseen
    assert len(set(seen + resumed)) == len(seen) + len(unseen), "a window came twice"


# ---------------------------------------------------------------------------------
# One corpus, three stores
# ---------------------------------------------------------------------------------

CORPUS = np.concatenate([np.asarray([*document, PACK_EOS], np.int64) for document in
                         ([10, 11, 12], [20, 21], [30, 31, 32, 33, 34], [40], [50, 51])])


def _pieces(tokens, sizes):
    """`tokens` cut into consecutive pieces of `sizes`, the last taking the rest."""
    edges = np.cumsum([0, *sizes])
    return [*(tokens[start:stop] for start, stop in itertools.pairwise(edges)), tokens[edges[-1]:]]


@pytest.fixture
def stores(tmp_path):
    """The same corpus in three committed token stores: its records its documents, cut into
    pieces of other lengths, or those pieces under a field of their own.

    The pieces are deliberately unequal, so a window that crosses a record
    boundary has to be joined out of two of them; identical windows then say
    the stream view is the stream, not the chunking.
    """
    from dew.data import Memory
    from dew.data.sinks import export

    def exported(name, sizes, field):
        rows = [{field: piece.astype(np.int32)} for piece in _pieces(CORPUS, sizes)]
        stated = {"token_store": True, "tokenizer": "byte", "vocab_size": 256, "dtype": "int32",
                  "eos_id": PACK_EOS}
        export(Memory(rows), str(tmp_path / name), format="array_record", length_field=field, metadata=stated)
        return TokenFiles(str(tmp_path / name), field=field)

    documents = write_token_corpus(tmp_path / "documents", CORPUS, eos_id=PACK_EOS)
    return {"documents": TokenFiles(documents.train),
            "records": exported("records", [4, 3], "text"),
            "packed_records": exported("packed_records", [2, 9], "ids")}


def test_the_same_corpus_gives_the_same_windows_and_packing_in_every_store(stores):
    def validation(stored, **layout):
        rows = _read(_validated(stored, **layout))
        return [jax.tree.map(lambda value: value.tolist(), row) for row in rows]

    windows = {name: validation(stored, seq_len=4, batch=1) for name, stored in stores.items()}
    packed = {name: validation(stored, seq_len=5, batch=1, pack="first-fit", bins=2)
              for name, stored in stores.items()}
    assert [row["text"] for row in windows["documents"]] == [
        CORPUS[start:start + 5].tolist() for start in range(0, len(CORPUS) - 4, 4)]
    lengths = [count for row in packed["documents"]
               for segment, count in Counter(row["text_coordinates"].segments).items() if segment]
    assert sorted(lengths) == [2, 3, 3, 4, 6], "the documents are the eos spans"
    for name in stores:
        assert windows[name] == windows["documents"], name
        assert packed[name] == packed["documents"], name


def test_a_hub_tokenizer_runs_the_revision_its_identity_names(monkeypatch, tmp_path):
    """Tokenize resolves a hub name once: its identity hashes the files of the
    resolved revision and its encoding reads those same files, here and in a
    worker it is pickled to, never the name again."""
    import pickle

    import huggingface_hub

    from dew.data.text import Tokenize, load_tokenizer

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


def test_a_release_whose_remote_tokenizer_renames_qwen2s_loads_as_qwen2s_without_its_code(tmp_path):
    """Dream's release names its tokenizer DreamTokenizer, remote code that is
    transformers' Qwen2Tokenizer renamed (`RENAMED_TOKENIZERS`). A Qwen2
    tokenizer's files under that config load without the release's code, which
    a plain load refuses to run, and encode and decode as the Qwen2 tokenizer
    itself, special tokens included."""
    from transformers import AutoTokenizer

    from dew.data.text import load_tokenizer

    source = Path(__file__).parent / "fixtures" / "hf" / "qwen38-moe-tiny"
    released = tmp_path / "dream"
    released.mkdir()
    for name in ("tokenizer.json", "tokenizer_config.json"):
        shutil.copy(source / name, released / name)
    config = json.loads((released / "tokenizer_config.json").read_text())
    config.update(tokenizer_class="DreamTokenizer",
                  auto_map={"AutoTokenizer": ["tokenization_dream.DreamTokenizer", None]})
    (released / "tokenizer_config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError):
        AutoTokenizer.from_pretrained(str(released), local_files_only=True)
    loaded = load_tokenizer(str(released), local_files_only=True)
    plain = load_tokenizer(str(source), local_files_only=True)
    text = "<|im_start|>user\nThe Cascade Range: which volcano? ünïcödé\t12345<|im_end|>"
    assert loaded.encode(text) == plain.encode(text)
    assert loaded.decode(loaded.encode(text)) == plain.decode(plain.encode(text))


def test_a_tokenizer_whose_model_type_transformers_dispatches_without_remote_code_loads_without_it(tmp_path):
    """InternLM2's release names a remote tokenizer class beside a model type transformers does not
    register; transformers lists that type among those whose hub class it overrides
    (MODELS_WITH_INCORRECT_HUB_TOKENIZER_CLASS) and reads tokenizer.json without the code. The
    type, read from config.json, reaches that dispatch through `load_tokenizer` as it does
    through AutoTokenizer, and the tokenizer encodes as its tokenizer.json does."""
    from tokenizers import Tokenizer
    from transformers import AutoTokenizer

    from dew.data.text import load_tokenizer

    source = Path(__file__).parent / "fixtures" / "hf" / "dory-tiny"
    released = tmp_path / "internlm2"
    released.mkdir()
    shutil.copy(source / "tokenizer.json", released / "tokenizer.json")
    remote = {"AutoTokenizer": ["tokenization_internlm2.InternLM2Tokenizer", None]}
    (released / "tokenizer_config.json").write_text(json.dumps(
        {"tokenizer_class": "InternLM2Tokenizer", "auto_map": remote}))
    (released / "config.json").write_text(json.dumps(
        {"model_type": "internlm2", "architectures": ["InternLM2ForCausalLM"]}))
    text = "charged twice, site 500!"
    expected = Tokenizer.from_file(str(released / "tokenizer.json")).encode(text).ids
    assert AutoTokenizer.from_pretrained(str(released), local_files_only=True).encode(text) == expected
    assert load_tokenizer(str(released), local_files_only=True).encode(text) == expected


def test_hosts_that_mount_one_corpus_at_different_paths_read_one_order(tmp_path):
    """Each host of a pool reads its own copy of a token corpus, mounted where it was put: the
    copies' readers agree on what they read, and a copy that holds other tokens does not."""
    stream = (np.arange(97, dtype=np.uint16) * 7) % 32
    first = write_token_corpus(tmp_path / "host0" / "tokens", stream, stream).train
    second = str(shutil.copytree(first, tmp_path / "elsewhere" / "copy"))
    other = write_token_corpus(tmp_path / "host2" / "tokens", stream[::-1].copy(), stream).train

    def identity(store):
        reader = _trained(store, 4, 4, loading=Loading(workers=0)).train(DataPartition())
        try:
            return reader.identity()
        finally:
            reader.close()

    assert identity(first) == identity(second) != identity(other)


def test_a_one_split_token_export_refuses_a_validation_split_it_does_not_hold(tmp_path):
    """A committed token export holds one split, so a validation split it lacks is refused."""
    exported = export_byte_tokens(tmp_path, ("first", "next"), "--format", "array_record")
    with pytest.raises(ValueError):
        _, held_out = TokenFiles(str(exported)).holdout(split="val")
        _validated(held_out, 4, 1)
