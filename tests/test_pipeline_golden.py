"""Loader batches and RNG against frozen references, with real workers.

Each case is a chain of the public presets and verbs, written as the recipe it replaced compiled:
the frozen references are the batches the old loaders read. Images pin the public steps. Packed
batches pin host Coordinates; packed text pins documents cut into pieces that share their boundary
token, where 66beb5305 lost each piece's first next-token target. The other cases pin 66beb5305.
"""

import dataclasses
import gzip
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import tensorflow_datasets as tfds
from array_record.python.array_record_module import ArrayRecordWriter
from token_support import write_token_corpus

from dew.data import Memory, images, phases, presets
from dew.data.compile import compile
from dew.data.dataset import DataPartition, Loading
from dew.data.images import EncodeCaptions
from dew.data.sources.codec import pack_dict_of_byte_arrays
from dew.data.sources.files import Files
from dew.data.sources.tfds import TFDS
from dew.data.sources.tokens import TokenFiles
from dew.data.steps.generic import Formatting

GOLD = Path(__file__).parent / "fixtures/golden/loaders-66beb5305.json.gz"
ORACLE = Path(__file__).parents[1] / "tools/measurements/data-pipeline/data_oracle_legacy.py"
CASES = ("text_windows", "text_packed", "text_phases", "chat", "prompts", "preferences", "images_memory",
         "images_arrayrecord", "images_hf", "images_tfds", "hub_records", "tfds_records")
"""The corpora the old loaders read alike. Decision rows are not among them: they pack into windows,
which no old loader did (tests/test_decision_packing.py); nor is video, whose record the media
stages changed on purpose (a waveform and its validity in place of per-frame rows and features):
tests/test_data.py holds a video listing to itself across workers and a resume."""


def _digest(batch):
    """The batch's public fields: each array's shape, dtype and content digest by its path, and each
    record's static facts (`Coordinates`' `max_segments` and `right_padded`) by theirs, not the tree's
    spelling. A field that is None holds nothing and adds no entry, where a renamed field or a changed
    static fact still changes the digest (Main, 10-27: a field added to Coordinates once re-recorded
    142 tree strings whose leaves had not moved)."""
    fields = {}

    def walk(value, path):
        if value is None:
            return
        if isinstance(value, dict):
            for key in sorted(value):
                walk(value[key], f"{path}[{key!r}]")
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                walk(item, f"{path}[{index}]")
        elif dataclasses.is_dataclass(value):
            for field in dataclasses.fields(value):
                item = getattr(value, field.name)
                if field.metadata.get("pytree_node", True):
                    walk(item, f"{path}.{field.name}")
                else:
                    fields[f"{path}.{field.name}"] = {"static": item}
        else:
            array = np.asarray(value)
            encoded = (json.dumps(array.tolist(), sort_keys=True).encode()
                       if array.dtype.hasobject else array.tobytes())
            fields[path] = {"shape": list(array.shape), "dtype": array.dtype.str,
                            "sha256": hashlib.sha256(encoded).hexdigest()}

    walk(batch, "")
    return fields


def _oracle():
    """The frozen reference's capture script, whose token stores the reference read: the stores this
    test reads are the ones it wrote (`token_store`)."""
    spec = importlib.util.spec_from_file_location("data_oracle_legacy", ORACLE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _image_sources(root, pixels):
    image_root = root / "images"
    image_root.mkdir()
    writer = ArrayRecordWriter(str(image_root / "00.array_record"), "group_size:1")
    for row in pixels:
        writer.write(pack_dict_of_byte_arrays({"image": row["image"].tobytes(),
                                               "shape": np.asarray([16, 16], "<i4").tobytes(),
                                               "caption": row["caption"].encode(),
                                               "label": np.asarray(row["label"], np.int32).tobytes()}))
    writer.close()
    image_file = root / "images.parquet"
    pq.write_table(pa.Table.from_pylist([
        {**row, "image": {"bytes": cv2.imencode(".png", row["image"][..., ::-1])[1].tobytes(), "path": None}}
        for row in pixels]), image_file)
    return str(image_root), str(image_file)


def _token_corpus(root, name, shift):
    """The token store the frozen reference read (`token_store`), its streams written as committed
    exports, each document through its EOS."""
    legacy = Path(_oracle().token_store(root, name, shift))
    train, val = (np.frombuffer((legacy / f"{split}.bin").read_bytes(), np.uint16)
                  for split in ("train", "val"))
    return write_token_corpus(root / f"{name}-export", train, val, eos_id=255)


@pytest.fixture(scope="module")
def chains(tmp_path_factory):
    """Each case's training chain and the chain it validates on, in batches of four, shuffled by
    seed 7 where the recipe ordered them: before the first random step, else before the batch."""
    root = tmp_path_factory.mktemp("pipeline-golden")
    tokens, other = _token_corpus(root, "tokens", 0), _token_corpus(root, "other", 10)
    conversations = Memory([{"messages": [{"role": "user", "content": f"question {index}"},
                                          {"role": "assistant", "content": f"answer {index % 5}"}]}
                            for index in range(40)])
    prompts = Memory([{"prompt": list(range(1, 3 + index % 6)), "data_source": "rule",
                       "ground_truth": str(index % 3), "extra_info": ""} for index in range(40)])
    pairs = Memory([{"chosen": list(range(1, 3 + index % 4)), "rejected": [2, 3, 4],
                     "chosen_mask": [0] + [1] * (1 + index % 4), "rejected_mask": [0, 1, 1]}
                    for index in range(40)])
    pixels = tuple({"image": ((np.arange(16 * 16 * 3).reshape(16, 16, 3) + index * 7) % 256).astype(np.uint8),
                    "label": index % 3, "caption": f"picture {index}"} for index in range(40))
    image_root, image_file = _image_sources(root, pixels)
    table = root / "records.parquet"
    pq.write_table(pa.table({"id": list(range(40)), "x": [[index, index + 1] for index in range(40)]}), table)
    tfds_images = "tests/fixtures/tfds/dew_images/1.0.0"

    def windows(corpus):
        return TokenFiles(corpus).shuffle(7).repeat().pipe(presets.token_windows, seq_len=16, batch=None)

    def packed(corpus, batch=None, records=None):
        return TokenFiles(corpus).pipe(presets.lm, None, seq_len=16, batch=batch, pack="first-fit", bins=8,
                                       records=records)

    def chat(batch=None, records=None):
        return conversations.pipe(presets.sft, "tests/fixtures/tokenizers/tiny-chat", seq_len=64, batch=batch,
                                  long="chunk", bins=8, records=records)

    def pictures(source, size=16, *, augment=True, held=16, templates=()):
        """The images held out first, the rest jittered and flipped on the host, every caption
        dropped before a batch reaches the run."""
        train, val = source.holdout(records=held)
        caption = images.caption(templates=templates)

        def prepared(part, **options):
            return part.pipe(presets.images, size, batch=4, rejects=False, caption=caption,
                             host=(EncodeCaptions(),), **options)

        return (prepared(train.shuffle(7).repeat(), augment=(images.Flip(), images.ColorJitter())),
                prepared(val))

    return {
        "text_windows": (windows(tokens.train).batch(4),
                         TokenFiles(tokens.val).pipe(presets.token_windows, seq_len=16, batch=4, records=16)),
        "text_packed": (packed(tokens.train).shuffle(7).repeat().batch(4), packed(tokens.val, 4, 16)),
        "text_phases": (phases((windows(tokens.train), 3), (windows(other.train), None)).batch(4),
                        TokenFiles(tokens.val).pipe(presets.token_windows, seq_len=16, batch=4, records=16)),
        "chat": (chat().shuffle(7).repeat().batch(4), chat(4, 16)),
        "prompts": (prompts.pipe(presets.prompts, "byte", max_prompt_len=16, batch=None).shuffle(7).repeat()
                    .batch(4), None),
        "preferences": (pairs.pipe(presets.preferences, seq_len=16, batch=None).shuffle(7).repeat().batch(4),
                        None),
        "images_memory": pictures(Memory(pixels)),
        "images_arrayrecord": pictures(Files(image_root, format="array_record")),
        "images_hf": pictures(Files(image_file)),
        "images_tfds": pictures(TFDS("dew_images", split="all", path=tfds_images,
                                     decoders={"image": tfds.decode.SkipDecoding()}),
                                8, held=4, templates=("a photo of a {}",)),
        "hub_records": (Files(str(table)).map(Formatting()).shuffle(7).repeat().batch(4), None),
        "tfds_records": (TFDS("dew_images", path=tfds_images).map(Formatting()).shuffle(7).repeat().batch(4),
                         None),
    }


# Each recipe reads in two opposite loadings and the next recipe in the other two, so every recipe
# reads with workers and without, sharded and whole, and every loading reads half the recipes.
LOADINGS = (((0, 1), (2, 2)), ((0, 2), (2, 1)))


@pytest.mark.parametrize("name, workers, shares", [
    pytest.param(name, workers, shares, id=f"{name}-workers={workers},shares={shares}")
    for index, name in enumerate(CASES) for workers, shares in LOADINGS[index % 2]])
def test_loader_batches_positions_and_rng_match_the_frozen_reference(chains, name, workers, shares):
    with gzip.open(GOLD, "rt") as source:
        expected = json.load(source)["cases"][name][f"workers={workers},shares={shares}"]
    train, validation = chains[name]
    loaded = compile(train, validation=validation, read="memory", seed=7,
                     loading=Loading(workers=workers, threads=1, read_buffer=4, worker_buffer=1))
    assert loaded.records == expected["records"]
    assert loaded.held_out == expected["held_out"]
    for share in range(shares):
        partition = DataPartition(share, shares)
        reader = loaded.train(partition)
        try:
            first = [_digest(row) for row in itertools.islice(reader, 4)]
            state = reader.get_state()
            rest = [_digest(row) for row in itertools.islice(reader, 4)]
        finally:
            reader.close()
        assert first == expected["shares"][share]["first"]
        assert rest == expected["shares"][share]["after_save"]
        resumed = loaded.train(partition)
        try:
            resumed.set_state(state)
            assert [_digest(row) for row in itertools.islice(resumed, 4)] == rest
        finally:
            resumed.close()
        if loaded.val is not None:
            held = loaded.val(partition)
            try:
                observed = [_digest(row) for row in itertools.islice(held, 100)]
                assert next(held, None) is None
            finally:
                held.close()
            assert observed == expected["shares"][share]["validation"]
