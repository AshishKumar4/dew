"""Packed windows against grain's own packers: a first-fit plan reads the
windows `grain.experimental.FirstFitPackIterDataset` packs, and a memory read
packs by the stage's policy as an indexed read does."""

import itertools

import grain
import jax
import numpy as np
import pytest
from grain.experimental import ConcatThenSplitIterDataset, FirstFitPackIterDataset

from dew.batch import VALID_ROWS, Coordinates
from dew.data import Loading, Memory
from dew.data.compile import compile, validation
from dew.data.dataset import DataPartition
from dew.data.packing import ConcatPacked, FirstFitPacked
from dew.data.pipeline import Pack, Pipeline
from dew.data.plan import fingerprint
from dew.data.text import Chunks

READ = Loading(workers=0, threads=1, read_buffer=0, worker_buffer=1)


FIELDS = ("text", "text_roles")


def _drawn(count: int) -> list[tuple[int, int, int | None]]:
    """`count` packings drawn at random: a window of 2 to 48 tokens, 1 to 12 bins, and no cap or a
    cap of 1 to 6 documents a window."""
    rng = np.random.default_rng(1064)
    return [(int(rng.integers(2, 49)), int(rng.integers(1, 13)),
             None if rng.random() < 0.3 else int(rng.integers(1, 7))) for _ in range(count)]


@pytest.mark.parametrize("window, bins, segments",
                         [(8, 1, None), (8, 3, None), (16, 4, 3), (5, 2, 2), *_drawn(16)])
def test_planned_windows_are_grains_first_fit_windows(window, bins, segments):
    """Dew plans first-fit windows by documents' lengths alone (`dew.data.placement.first_fit`), a
    placement Grain's packer reaches only by packing every token, so Grain's packer is its oracle,
    over packings drawn at random as over the edge cases. Documents from one token to well past the
    window, a few exactly the window, with ids and roles, cut by a chunk step before a first-fit pack:
    grain's packer, given the chunks in file order and its bins emitted in
    order, writes window for window what a memory read and an indexed read of
    documents that state their lengths read by the plan (`plan_windows`), and
    what an indexed read of documents that state none packs by grain's packer
    as they stream: the ids, the roles and each token's document and place in
    it. Every read leaves out a window with no target; the memory read counts
    it as zero_targets, and its order facts name grain's plan and targets."""
    rng = np.random.default_rng(window * 10 + bins)
    lengths = np.concatenate([rng.integers(1, 3 * window, 60), [window, window, 1]])
    ids = [(np.arange(length) + 1000 * index).astype(np.int32) for index, length in enumerate(lengths)]
    roles = [np.full(length, index % 4, np.int32) for index, length in enumerate(lengths)]
    chunks = [{"text": tokens[start:start + window], "text_roles": role[start:start + window]}
              for tokens, role in zip(ids, roles, strict=True) for start in range(0, len(tokens), window)]
    reference = list(FirstFitPackIterDataset(
        grain.MapDataset.source(chunks).to_iter_dataset(), length_struct=dict.fromkeys(FIELDS, window),
        num_packing_bins=bins, shuffle_bins=False, max_sequences_per_bin=segments))
    targets = [int(np.count_nonzero((segment[1:] == segment[:-1]) & (segment[1:] != 0)))
               for segment in (packed["text_segment_ids"] for packed in reference)]
    targeted = [packed for packed, count in zip(reference, targets, strict=True) if count]

    def pipeline(source):
        return Pipeline(source).flat_map(Chunks(window, fields=FIELDS)).pack(
            window, fields=FIELDS, bins=bins, segments=segments)

    def windows(source, read):
        reader = validation(pipeline(source), batch=1, read=read, loading=READ)(DataPartition())
        try:
            return list(reader)
        finally:
            reader.close()

    documents = Memory([{"text": tokens, "text_roles": role}
                        for tokens, role in zip(ids, roles, strict=True)])
    stated = Memory({"text": ids, "text_roles": roles, "length": lengths.tolist()})
    for source, read in ((documents, "memory"), (stated, "indexed"), (documents, "indexed")):
        read_windows = windows(source, read)
        assert len(read_windows) == len(targeted)
        for index, (got, expected) in enumerate(zip(read_windows, targeted, strict=True)):
            laid = got["text_coordinates"]
            for value, key in ((got["text"], "text"), (got["text_roles"], "text_roles"),
                               (laid.segments, "text_segment_ids"), (laid.positions, "text_positions")):
                np.testing.assert_array_equal(value[0], expected[key], err_msg=f"{read} window {index} {key}")
    data = compile(pipeline(documents).batch(1), read="memory", loading=READ)
    (counted,) = data.counts.values()
    assert counted.accepted == len(targeted)
    assert counted.rejected["zero_targets"] == len(reference) - len(targeted)
    # The plan as grain followed it: the chunks in the order it added them, by their first token,
    # and where each window starts in that order.
    first = {int(piece["text"][0]): index for index, piece in enumerate(chunks)}
    order = [first[int(packed["text"][at])] for packed in reference
             for at in np.flatnonzero((packed["text_positions"] == 0) & (packed["text_segment_ids"] != 0))]
    starts = np.cumsum([0, *(int(packed["text_segment_ids"].max()) for packed in reference)]).tolist()
    assert data.plan.derived["0/packing"] == fingerprint([order, starts])
    assert data.plan.derived["0/valid_targets"] == fingerprint(sum(targets))


def same(got, expected):
    """Windows field for field, in order, each field's dtype and values, the fields named in the
    order grain hands them on."""
    assert len(got) == len(expected)
    for index, (window_got, window_expected) in enumerate(zip(got, expected, strict=True)):
        assert list(window_got) == list(window_expected)
        for key, value in window_expected.items():
            assert window_got[key].dtype == value.dtype, (index, key)
            np.testing.assert_array_equal(window_got[key], value, err_msg=f"window {index} {key}")


@pytest.mark.parametrize("window, bins, segments", [(8, 3, None), (16, 4, 3), (5, 2, 2), (4, 1, None),
                                                     *_drawn(8)])
def test_a_stream_packs_first_fit_as_grain_does_and_resumes_where_grains_packer_left(window, bins, segments):
    """Dew packs a stream of documents first-fit (`FirstFitPacked`) by the placement its plans of
    lengths follow (`FirstFit`), and Grain's `FirstFitPackIterDataset`, its oracle, packs the same
    windows field for field: ids and roles a token beside a field along an axis of its own, each
    with its own room and the ids and that field each with its own segments and positions, at most
    `segments` documents a window. Each saves its position as the other does, so a read of either
    stopped after any window resumes on the other."""
    rng = np.random.default_rng(window * 100 + bins)
    along = 2 * window
    documents = []
    for index, length in enumerate(rng.integers(1, window + 1, 40).tolist()):
        tokens = (np.arange(length) + 1000 * index).astype(np.int32)
        documents.append({"text": tokens, "text_roles": np.full(length, index % 4, np.int32),
                          "image": np.full(int(rng.integers(1, along + 1)), index, np.float32)})
    pack = Pack(window, fields=FIELDS, bins=bins, segments=segments, along=(("image", along),))

    def stream():
        return grain.MapDataset.source(documents).to_iter_dataset()

    rooms = dict.fromkeys(FIELDS, window) | {"image": along}
    grains = FirstFitPackIterDataset(stream(), length_struct=rooms, num_packing_bins=bins, shuffle_bins=False,
                                     max_sequences_per_bin=segments, meta_features=FIELDS[1:])
    dews = FirstFitPacked(stream(), pack)
    reference = list(grains)

    same(list(dews), reference)
    for stop in sorted({0, 1, bins, len(reference) // 2, len(reference) - 1}):
        for saving, resuming in ((grains, dews), (dews, grains)):
            read = iter(saving)
            for _ in range(stop):
                next(read)
            resumed = iter(resuming)
            resumed.set_state(read.get_state())
            same(list(resumed), reference[stop:])


def test_a_stream_lays_documents_as_grain_does_whatever_their_reader_hands_on():
    """Documents a stream hands on in one buffer it fills again for each, of empty fields, of a
    scalar along an axis of its own, and of another dtype than the first of their windows: Dew's
    packer lays each as Grain's does, copied as it is read, an empty one counted as a document, a
    scalar as one entry, and each window at the dtype of the first document of its set."""
    buffer = np.zeros(4, np.int32)

    def refilled(index):
        buffer[:] = index
        return {"text": buffer, "text_roles": buffer, "score": np.float32(index)}

    def mixed(index):
        length = (4, 0, 3, 2, 0, 1)[index]
        dtype = np.int32 if index % 2 else np.int64
        return {"text": np.full(length, index, dtype), "text_roles": np.full(length, index, dtype),
                "score": np.float32(index)}

    for laid, bins in ((refilled, 1), (mixed, 2)):
        pack = Pack(4, fields=FIELDS, bins=bins, along=(("score", 2),))
        rooms = dict.fromkeys(FIELDS, 4) | {"score": 2}

        def stream(laid=laid):
            # Read in turn, nothing ahead: each document is in the buffer only until the next is read.
            return grain.MapDataset.source(range(6)).map(laid).to_iter_dataset(
                grain.ReadOptions(num_threads=1, prefetch_buffer_size=0))

        reference = list(FirstFitPackIterDataset(stream(), length_struct=rooms, num_packing_bins=bins,
                                                 shuffle_bins=False, meta_features=FIELDS[1:]))
        same(list(FirstFitPacked(stream(), pack)), reference)


def test_a_stream_refuses_a_document_longer_than_its_windows():
    """A document longer than the windows a stream packs into is refused, as Grain's packer refuses
    it, where a cut would place its tokens in two windows no reader tells apart."""
    documents = [{"text": np.arange(3, dtype=np.int32)}, {"text": np.arange(9, dtype=np.int32)}]
    packed = FirstFitPacked(grain.MapDataset.source(documents).to_iter_dataset(), Pack(8, fields=("text",)))
    with pytest.raises(ValueError):
        list(packed)


@pytest.mark.parametrize("read", ["memory", "indexed"])
def test_a_packed_window_keeps_each_fields_trailing_shape_as_grain_packs_it(read):
    """A pack of ids beside an aligned field that holds a row of three a token (a token's features):
    every window an indexed plan or a memory read assembles holds each field with its trailing shape,
    field for field what grain's first-fit packer writes of the same documents, where every packed
    field was allocated one-dimensional and a row of three could not be assigned (row 1873)."""
    lengths = (3, 7, 2, 5, 8, 4)
    documents = [{"text": (np.arange(length) + 100 * index).astype(np.int32),
                  "text_features": np.arange(3 * length, dtype=np.float32).reshape(length, 3) + index}
                 for index, length in enumerate(lengths)]
    fields = ("text", "text_features")
    reference = list(FirstFitPackIterDataset(
        grain.MapDataset.source(documents).to_iter_dataset(), length_struct=dict.fromkeys(fields, 8),
        num_packing_bins=3, shuffle_bins=False, meta_features=fields[1:]))
    source = (Memory(documents) if read == "memory" else
              Memory({"text": [value["text"] for value in documents],
                      "text_features": [value["text_features"] for value in documents],
                      "length": list(lengths)}))
    reader = validation(Pipeline(source).pack(8, fields=fields, bins=3), batch=1, read=read,
                        loading=READ)(DataPartition())
    try:
        windows = list(reader)
    finally:
        reader.close()
    assert len(windows) == len(reference)
    for got, expected in zip(windows, reference, strict=True):
        assert got["text_features"][0].shape == (8, 3)
        for key in fields:
            np.testing.assert_array_equal(got[key][0], expected[key], err_msg=key)
        np.testing.assert_array_equal(got["text_coordinates"].segments[0], expected["text_segment_ids"])


@pytest.mark.parametrize("policy, overlap", [("first-fit", 0), ("best-fit", 0), ("concat", 0), ("concat", 1)])
def test_a_memory_read_packs_by_the_policy_an_indexed_read_packs_by(policy, overlap):
    """read='memory' packs the whole corpus before the order and an indexed
    read packs it as it streams. Both follow the stage's policy, so a pass
    over either is the same windows, segment ids and positions in the same
    order, a concatenating pack whose cuts overlap too; first-fit, which has
    its own plan, is the control."""
    lengths = (3, 7, 2, 5, 8, 4, 6, 2, 8, 3)
    documents = [{"text": (np.arange(length) + 100 * index).astype(np.int32)}
                 for index, length in enumerate(lengths)]
    pipeline = Pipeline(Memory(documents)).pack(8, bins=3, policy=policy, overlap=overlap)

    def windows(read):
        reader = validation(pipeline, batch=2, read=read, loading=READ)(DataPartition())
        try:
            # The windows' arrays: each read declares its own bound on the documents a window holds
            # (`Coordinates.max_segments`), the memory read the most it packed into one and the
            # indexed read, packing as it streams, its window's length, which its documents' firsts are
            # as wide as: their token-aligned fields compared.
            return [[np.asarray(value).tolist() for value in jax.tree.leaves(
                {name: value.tokens() if isinstance(value, Coordinates) else value
                 for name, value in batch.items()})] for batch in reader]
        finally:
            reader.close()

    memory = windows("memory")
    assert memory and memory == windows("indexed")


def read_windows(pipeline, read):
    """The windows' tokens a validation read of `pipeline` reads, one window a batch."""
    reader = validation(pipeline, batch=1, read=read, loading=READ)(DataPartition())
    try:
        return [np.asarray(batch["text"]).tolist() for batch in reader]
    finally:
        reader.close()


@pytest.mark.parametrize("read", ["indexed", "memory"])
def test_a_planned_first_fit_places_an_empty_document_as_a_stream_does(read):
    """An empty document is a document to first-fit, which counts it against a window's segments: two
    tokens, none and two at width four and two documents a window are two windows, the empty one in the
    first, whether planned from the documents' lengths (an indexed or memory read) or packed as they
    stream (data.md's one layout, row 6277)."""
    documents = [{"text": np.arange(length, dtype=np.int32) + 10 * index}
                 for index, length in enumerate((2, 0, 2))]
    pipeline = Pipeline(Memory(documents)).pack(4, bins=1, segments=2)
    streamed = read_windows(pipeline, "streamed")
    assert len(streamed) == 2 and read_windows(pipeline, read) == streamed


def pairs(batch, field="text"):
    """The next-token pairs of `field` inside one segment of each real window of a packed batch,
    the repeats that fill a short last batch (`VALID_ROWS`, which a whole batch holds none of) left
    out."""
    tokens, segments = np.asarray(batch[field]), np.asarray(batch["text_coordinates"].segments)
    real = np.asarray(batch.get(VALID_ROWS, np.ones(len(tokens), bool)))
    return [(int(row[at]), int(row[at + 1]))
            for row, segment, kept in zip(tokens, segments, real, strict=True) if kept
            for at in range(row.size - 1) if segment[at] and segment[at] == segment[at + 1]]


def leaves(stream, batches):
    """The arrays of the next `batches` batches of `stream`, the windows' coordinates among them."""
    return [[np.asarray(value).tolist() for value in jax.tree.leaves(batch)]
            for batch in itertools.islice(stream, batches)]


@pytest.mark.parametrize("segments", [None, 2])
@pytest.mark.parametrize("overlap", [0, 1])
@pytest.mark.parametrize("read", ["memory", "indexed", "streamed"])
def test_a_concatenating_pack_keeps_every_pair_where_its_cuts_overlap_and_resumes_anywhere(read, overlap,
                                                                                          segments):
    """A concatenating pack whose cuts overlap by a token (`Pack.overlap`) starts each window again
    at the last token a cut through a document left behind, in every field alike: a pass holds
    every next-token pair of every document once, each field's its own documents', and with a cap
    (`Pack.segments`, row 9059) no window holds more documents or pieces of one than it. With or
    without the overlap, a run resumed after any step reads the windows the uninterrupted one reads:
    how full the overlap left its window is saved with the position, and so is a window that closed
    where a document ended, whose next document the packer holds whole."""
    lengths = (3, 7, 2, 5, 8, 4, 6, 2, 8, 3, 17)
    documents = [{"text": (np.arange(length) + 100 * index).astype(np.int32),
                  "text_roles": (np.arange(length) + 1000 * index).astype(np.int32)}
                 for index, length in enumerate(lengths)]
    pipeline = Pipeline(Memory(documents)).pack(8, fields=FIELDS, policy="concat", overlap=overlap,
                                                segments=segments)
    reader = validation(pipeline, batch=2, read=read, loading=READ)(DataPartition())
    try:
        held = list(reader)
    finally:
        reader.close()
    most = max(int(np.asarray(batch["text_coordinates"].segments).max()) for batch in held)
    assert most == (segments or most) and (segments is not None or most > 2), most
    if overlap:
        for field in FIELDS:
            drawn = [pair for document in documents for pair in itertools.pairwise(document[field].tolist())]
            assert sorted(pair for batch in held for pair in pairs(batch, field)) == sorted(drawn)
    data = compile(pipeline.repeat().batch(2), read=read, loading=READ)
    whole = data.train(DataPartition())
    try:
        expected = leaves(whole, 8)
    finally:
        whole.close()
    for steps in range(1, 8):
        stopped = data.train(DataPartition())
        try:
            head = leaves(stopped, steps)
            saved = stopped.get_state()
        finally:
            stopped.close()
        resumed = data.train(DataPartition())
        try:
            resumed.set_state(saved)
            assert head + leaves(resumed, 8 - steps) == expected, steps
        finally:
            resumed.close()


@pytest.mark.parametrize("window", [5, 8, 13])
def test_a_stream_concatenates_as_grains_concatenating_packer_does_and_resumes_anywhere(window):
    """Dew's concatenating packer (`ConcatPacked`) lays a stream's documents out as Grain's
    `ConcatThenSplitIterDataset`, its oracle where Grain has the layout: no cap on a window's documents.
    Documents of none to past two windows of tokens, ids and roles a token, beside an image of one to three
    entries along an axis of its own, which Grain holds as a feature it never cuts (`meta_features`): field
    for field, the ids' segments and positions, and each window's images, every piece holding its
    document's whole, a window closing where the next one's no longer fit. A read stopped after any window
    resumes on the windows after it."""
    rng = np.random.default_rng(window)
    documents = [{"text": (np.arange(length) + 1000 * index).astype(np.int32),
                  "text_roles": np.full(length, index % 4, np.int32),
                  "image": np.full(int(rng.integers(1, 4)), index, np.float32)}
                 for index, length in enumerate(rng.integers(0, 2 * window + 2, 40).tolist())]

    def stream():
        return grain.MapDataset.source(documents).to_iter_dataset()

    def but(windows, prefix):
        return [dict(sorted((key, value) for key, value in packed.items() if not key.startswith(prefix)))
                for packed in windows]

    rooms = dict.fromkeys(FIELDS, window) | {"image": 4}
    reference = list(ConcatThenSplitIterDataset(stream(), length_struct=rooms, meta_features=("image",)))
    dews = ConcatPacked(stream(), Pack(window, fields=FIELDS, policy="concat", along=(("image", 4),)))
    packed = list(dews)
    # Grain pairs every field it cuts with segments and positions; Dew the ids alone, and each field along
    # an axis of its own, which Grain pairs with none.
    same(but(packed, "image_"), but(reference, "text_roles_"))
    for stop in sorted({1, len(packed) // 2, len(packed) - 1}):
        read = iter(dews)
        for _ in range(stop):
            next(read)
        resumed = iter(dews)
        resumed.set_state(read.get_state())
        same(list(resumed), packed[stop:])


@pytest.mark.parametrize("read", ["indexed", "streamed"])
def test_a_window_with_no_target_is_left_out_whatever_the_read(read):
    """A window that holds one token alone has nothing to predict; a memory read leaves it out,
    and a read that packs as it streams leaves it out too, so the steps a run takes do not change
    with how it reads."""
    documents = Memory([{"text": np.asarray([1], np.int32)}, {"text": np.asarray([2, 3], np.int32)}])
    pipeline = Pipeline(documents).pack(4, segments=1)
    memory = read_windows(pipeline, "memory")
    assert memory == [[[2, 3, 0, 0]]]
    assert read_windows(pipeline, read) == memory


@pytest.mark.parametrize("read", ["memory", "indexed", "streamed"])
@pytest.mark.parametrize("policy", ["concat", "first-fit", "windows"])
def test_a_record_whose_packed_fields_differ_in_length_is_refused(read, policy):
    """A packed or windowed record's fields hold one entry a token. One whose fields differ in length
    is refused by every read, packer and window reader: each field would be cut at other places
    (where Grain's concatenating packer loses a token of the shorter field's remainder), its windows
    holding one field's tokens beside another's from elsewhere. Records whose fields agree are read
    alike."""
    def records(others: tuple[int, int]) -> Memory:
        return Memory([{"text": np.arange(start, start + length, dtype=np.int32),
                        "other": np.arange(100 + start, 100 + start + other, dtype=np.int32)}
                       for start, length, other in zip((10, 20), (3, 4), others, strict=True)])

    def windows(source: Memory) -> list[tuple[list[int], list[int]]]:
        pipeline = Pipeline(source)
        laid = (pipeline.windows(3, fields=("text", "other")) if policy == "windows"
                else pipeline.pack(4, fields=("text", "other"), policy=policy))
        chain = laid.batch(1)
        stream = compile(chain, read=read, loading=Loading(workers=0, threads=1)).train(DataPartition())
        try:
            return [(batch["text"][0].tolist(), batch["other"][0].tolist())
                    for batch in itertools.islice(stream, 3)]
        finally:
            stream.close()

    for text, other in windows(records((3, 4))):
        assert [token + 100 if token else 0 for token in text] == other
    with pytest.raises(ValueError):
        windows(records((1, 4)))



def _unpositioned(window: dict) -> dict:
    """A packed window without its positions, as a step after the pack that dropped them hands it on."""
    return {name: value for name, value in window.items() if name != "text_positions"}


def test_a_packed_window_whose_positions_a_later_step_dropped_is_refused():
    """A packed read lays each window's documents out by their segment ids and positions (`Coordinated`); a
    window a later step left without its positions is refused, naming them, where the read dropped its
    segment ids and handed on a bound alone, the window read as one document (train 890's review, row
    6631)."""
    documents = Memory([{"text": np.arange(start, start + 3, dtype=np.int32)} for start in (10, 20, 30)])
    chain = Pipeline(documents).pack(8, bins=3).map(_unpositioned).batch(1)
    stream = compile(chain, read="memory", loading=Loading(workers=0, threads=1)).train(DataPartition())
    try:
        with pytest.raises(ValueError):
            next(stream)
    finally:
        stream.close()
