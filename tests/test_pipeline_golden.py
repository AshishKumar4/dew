"""Every migrated loader against bytes frozen at 66beb5305, with real workers."""

import gzip
import hashlib
import itertools
import json
import shutil
import subprocess
from pathlib import Path

import cv2
import jax
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from array_record.python.array_record_module import ArrayRecordWriter

from dew.data.dataset import DataPartition, DataPhase, Loading
from dew.data.recipes import Augment, Chat, HoldOut, Images, Preferences, Prompts, Records, Text, Video, load
from dew.data.sources.codec import pack_dict_of_byte_arrays
from dew.data.sources.files import Files
from dew.data.sources.tfds import TFDS
from dew.data.text import ByteTokenizer
from dew.decision.data import Decisions, Example
from dew.decision.layout import Specials, StateFirstLayout
from dew.decision.objective import DecisionObjective
from dew.decision.questions import Choice
from dew.nn.backbones.causal_transformer import CausalTransformer

GOLD = Path(__file__).parent / "fixtures/golden/loaders-66beb5305.json.gz"
CASES = ("text_windows", "text_packed", "text_phases", "chat", "prompts", "preferences", "images_memory",
         "images_arrayrecord", "images_hf", "images_tfds", "hub_records", "tfds_records", "videos_local",
         "decision_table", "decision_mix")


def _digest(batch):
    leaves = []
    for value in jax.tree.leaves(batch):
        array = np.asarray(value)
        encoded = (json.dumps(array.tolist(), sort_keys=True).encode()
                   if array.dtype.hasobject else array.tobytes())
        leaves.append({"shape": list(array.shape), "dtype": array.dtype.str,
                       "sha256": hashlib.sha256(encoded).hexdigest()})
    return {"tree": str(jax.tree.structure(batch)), "leaves": leaves}


def _token_files(root, name, shift):
    directory = root / name
    directory.mkdir()
    train = np.asarray([token for index in range(100)
                        for token in [*((np.arange(5 + index % 15) + index + shift) % 200), 255]], np.uint16)
    held = train[:180]
    (directory / "train.bin").write_bytes(train.tobytes())
    (directory / "val.bin").write_bytes(held.tobytes())
    (directory / "meta.json").write_text(json.dumps({"tokenizer": "byte", "vocab_size": 256,
                                                   "dtype": "uint16", "train_tokens": len(train),
                                                   "val_tokens": len(held), "eos_id": 255}))
    return str(directory)


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


def _videos(root):
    from imageio_ffmpeg import get_ffmpeg_exe
    from transformers import Wav2Vec2FeatureExtractor

    directory = root / "clips"
    directory.mkdir()
    video = directory / "00.mp4"
    subprocess.run([get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
                    "testsrc=size=16x16:rate=25", "-f", "lavfi", "-i",
                    "sine=frequency=1000:sample_rate=16000", "-t", "2", "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-c:a", "aac", str(video)], check=True)
    for index in range(1, 12):
        shutil.copyfile(video, directory / f"{index:02}.mp4")
    audio = root / "audio"
    Wav2Vec2FeatureExtractor(sampling_rate=16000).save_pretrained(audio)
    return str(directory), str(audio)


@pytest.fixture(scope="module")
def recipes(tmp_path_factory):
    root = tmp_path_factory.mktemp("pipeline-golden")
    tokens, other = _token_files(root, "tokens", 0), _token_files(root, "other", 10)
    conversations = tuple({"messages": [{"role": "user", "content": f"question {index}"},
                                        {"role": "assistant", "content": f"answer {index % 5}"}]}
                          for index in range(40))
    prompts = tuple({"prompt": list(range(1, 3 + index % 6)), "data_source": "rule",
                     "ground_truth": str(index % 3), "extra_info": ""} for index in range(40))
    pairs = tuple({"chosen": list(range(1, 3 + index % 4)), "rejected": [2, 3, 4],
                   "chosen_mask": [0] + [1] * (1 + index % 4), "rejected_mask": [0, 1, 1]}
                  for index in range(40))
    pixels = tuple({"image": ((np.arange(16 * 16 * 3).reshape(16, 16, 3) + index * 7) % 256).astype(np.uint8),
                    "label": index % 3, "caption": f"picture {index}"} for index in range(40))
    image_root, image_file = _image_sources(root, pixels)
    table = root / "records.parquet"
    pq.write_table(pa.table({"id": list(range(40)), "x": [[index, index + 1] for index in range(40)]}), table)
    clips, audio = _videos(root)
    objective = DecisionObjective(
        CausalTransformer(vocab_size=256, emb_features=8, num_layers=1, num_heads=2, mlp_features=16),
        tokenizer=ByteTokenizer(), specials=Specials(0, 1, 2, "<mark>", 3),
        layout=StateFirstLayout(max_len=128), shuffle_options=True)
    examples = [Example(f"state {index}", {"class": Choice("Pick", ["yes", "no"])}, {"class": index % 2})
                for index in range(40)]
    augment = Augment(backend="host", jitter=True)
    return {
        "text_windows": (Text(tokens, seq_len=16), {"validation_records": 16}),
        "text_packed": (Text(tokens, seq_len=16, pack="first-fit"), {"validation_records": 16}),
        "text_phases": (Text(phases=(DataPhase(tokens, 3), DataPhase(other)), seq_len=16),
                        {"validation_records": 16}),
        "chat": (Chat(conversations, "tests/fixtures/tokenizers/tiny-chat", seq_len=64, long="chunk",
                       validation=conversations), {"validation_records": 16}),
        "prompts": (Prompts(prompts, "byte", max_prompt_len=16, long="truncate"), {}),
        "preferences": (Preferences(pairs, seq_len=16), {}),
        "images_memory": (Images(pixels, image_size=16, augment=augment, validation=HoldOut(records=16)), {}),
        "images_arrayrecord": (Images(Files(image_root, format="array_record"), image_size=16,
                                      augment=augment, validation=HoldOut(records=16)), {}),
        "images_hf": (Images(Files(image_file), image_size=16, augment=augment,
                              validation=HoldOut(records=16)), {}),
        "images_tfds": (Images(TFDS("dew_images", split="all", path="tests/fixtures/tfds/dew_images/1.0.0"),
                                image_size=8, augment=augment, caption_templates=("a photo of a {}",),
                                validation=HoldOut(records=4)), {}),
        "hub_records": (Records(Files(str(table))), {}),
        "tfds_records": (Records(TFDS("dew_images", path="tests/fixtures/tfds/dew_images/1.0.0")), {}),
        "videos_local": (Video(clips, frames=4, frame_size=16, audio_padding=1, audio_model=audio,
                                caption="clip", validation=HoldOut(records=4)), {}),
        "decision_table": (Decisions(examples, validation=examples[:5]), {"objective": objective.data()}),
        "decision_mix": (Decisions({"a": examples, "b": examples[:8]}, weights={"a": 3, "b": 1},
                                    validation=examples[:5]), {"objective": objective.data()}),
    }


@pytest.mark.parametrize("name", CASES)
@pytest.mark.parametrize("workers", [0, 2])
@pytest.mark.parametrize("shares", [1, 2])
def test_loader_batches_positions_and_rng_match_the_frozen_reference(recipes, name, workers, shares):
    with gzip.open(GOLD, "rt") as source:
        expected = json.load(source)["cases"][name][f"workers={workers},shares={shares}"]
    recipe, fields = recipes[name]
    loaded = load(recipe, batch=4, read="memory", seed=7,
                  loading=Loading(workers=workers, threads=1, read_buffer=4, worker_buffer=1), **fields)
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
