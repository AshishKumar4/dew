"""Named data and run specs read the same chains, with disjoint default holdouts."""

import itertools
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from token_support import write_token_corpus

from dew.config.data import (
    ChatData,
    HoldOut,
    ImageData,
    PreferenceData,
    PromptData,
    RecordsData,
    TextData,
    TinyShakespeareData,
    VideoData,
    loader,
)
from dew.data import TFDS, Hub, Loading, Memory, load, presets
from dew.data.compile import compile
from dew.data.dataset import DataPartition
from dew.objectives.base import VALID_ROWS

FIXTURES = Path(__file__).parent / "fixtures"
READ = Loading(workers=0, threads=1, read_buffer=2, worker_buffer=1)


def batches(data, *, held=False):
    reader = data.val if held else data.train
    assert reader is not None
    stream = reader(DataPartition())
    try:
        taken = list(itertools.islice(stream, 1000 if held else data.records // data.batch))
        if held:
            assert next(stream, None) is None
        return taken
    finally:
        stream.close()


def documents(rows):
    found = set()
    for batch in rows:
        kept = batch.get(VALID_ROWS, np.ones(len(batch["text"]), bool))
        laid = np.asarray(batch["text_coordinates"].segments)
        for tokens, segments in zip(batch["text"][kept], laid[kept], strict=True):
            for segment in np.unique(segments):
                if segment:
                    found.add(bytes(tokens[segments == segment][:-1].tolist()))
    return found


@pytest.fixture
def hub(tmp_path, monkeypatch):
    import datasets
    from huggingface_hub import HfApi
    from huggingface_hub.hf_api import GitRefs

    builders = {}
    texts = {}
    for revision, offset in (("frozen-default", 0), ("frozen-release", 1000)):
        ids = list(range(offset, offset + 512))
        texts[revision] = [f"document {index:04d} " + "x" * 113 for index in ids]
        path = tmp_path / f"{revision}.parquet"
        datasets.Dataset.from_dict({"id": ids, "text": texts[revision]}).to_parquet(path)
        builders[revision] = datasets.load_dataset_builder(
            "parquet", data_files={"train": str(path)}, cache_dir=str(tmp_path / revision))

    def info(self, repo, revision=None, **options):
        assert repo == "acme/simple"
        assert revision in (None, "release", "frozen-default", "frozen-release")
        return SimpleNamespace(sha="frozen-release" if revision in ("release", "frozen-release")
                               else "frozen-default")

    def builder(repo, *, revision, **options):
        assert repo == "acme/simple"
        return builders[revision]

    monkeypatch.setattr(HfApi, "dataset_info", info)
    monkeypatch.setattr(HfApi, "list_repo_files", lambda *args, **options: ["train.tar"])
    monkeypatch.setattr(HfApi, "list_repo_refs", lambda *args, **options: GitRefs(
        branches=[], converts=[], tags=[]))
    monkeypatch.setattr(datasets, "load_dataset_builder", builder)
    return builders, texts


@pytest.mark.parametrize("revision", [None, "release"])
def test_named_hub_records_prepare_the_pinned_split_and_hold_out_its_first_256(hub, revision):
    builders, _ = hub
    pinned = "frozen-release" if revision else "frozen-default"
    assert not Path(builders[pinned].cache_dir, "dataset_info.json").exists()
    data = load("hf/acme/simple" + ("@release" if revision else ""), batch=8, seed=19, loading=READ)
    assert Path(builders[pinned].cache_dir, "dataset_info.json").exists()
    assert data.records == data.held_out == 256
    train = [int(index) for batch in batches(data) for index in batch["id"]]
    held = [int(index) for batch in batches(data, held=True) for index in batch["id"]]
    offset = 1000 if revision else 0
    assert sorted(train) == list(range(offset + 256, offset + 512))
    assert held == list(range(offset, offset + 256))
    assert set(train).isdisjoint(held)


@pytest.mark.parametrize("options", [{"seq_len": 127}, {"tokenizer": "byte"}])
@pytest.mark.parametrize("named", [True, False])
def test_text_forms_pack_each_document_on_only_one_side_of_the_default_holdout(hub, options, named):
    _, texts = hub
    source = Hub("acme/simple")
    data = load("hf/acme/simple" if named else source, batch=1, loading=READ, **options)
    train, held = source.holdout(0.01)
    reference = compile(presets.lm(train, "byte", seq_len=options.get("seq_len", 256), batch=1),
                        validation=presets.lm(held, "byte", seq_len=options.get("seq_len", 256), batch=1),
                        read="memory", loading=READ)
    trained, scored = documents(batches(data)), documents(batches(data, held=True))
    assert trained and scored and trained.isdisjoint(scored)
    assert trained | scored == {text.encode() for text in texts["frozen-default"]}
    assert trained == documents(batches(reference))
    assert scored == documents(batches(reference, held=True))
    assert all(batch["text"].shape[1] == options.get("seq_len", 256) + 1 for batch in batches(data))


def test_a_refused_read_and_disabled_provisioning_leave_the_hub_cache_unprepared(hub):
    builders, _ = hub
    for options in ({"read": "streamed"}, {"read": "memory", "provision": False}):
        with pytest.raises(ValueError):
            load("hf/acme/simple", batch=8, loading=READ, **options)
        assert not Path(builders["frozen-default"].cache_dir, "dataset_info.json").exists()
    data = load("hf/acme/simple", batch=8, read="memory", loading=READ)
    assert data.plan.order["read"] == "memory"
    assert sorted(int(value) for batch in batches(data) for value in batch["id"]) == list(range(256, 512))


@pytest.mark.parametrize("name", ["records", "acme/simple", "hf/", "hf/@release", "hf/acme/simple@",
                                  "hf://datasets/acme/simple", "tfds/", "tfds/dew_images:",
                                  "tfds://dew_images", "data/train.jsonl"])
def test_other_names_are_refused_before_reading_a_source(name):
    with pytest.raises(ValueError):
        load(name, batch=1)


@pytest.mark.parametrize("version", [None, "1.0.0"])
def test_named_tfds_reads_the_prepared_fixture_and_refuses_an_oversized_holdout(tmp_path, monkeypatch,
                                                                           version):
    pytest.importorskip("tensorflow_datasets", reason="needs the tfds extra")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    directory = tmp_path / "cache" / "dew" / "tfds" / "dew_images" / "1.0.0"
    shutil.copytree(FIXTURES / "tfds" / "dew_images" / "1.0.0", directory)
    with pytest.raises(ValueError):
        load("tfds/dew_images" + (":" + version if version else ""), batch=4, loading=READ,
             provision=False)
    source = TFDS("dew_images", version=version)
    assert source.metadata()["rows"] == 16
    data = RecordsData(source, validation=HoldOut(records=4)).load(batch=4, read="memory", loading=READ)
    train = [int(image[0, 0, 0]) for batch in batches(data) for image in batch["image"]]
    held = [int(image[0, 0, 0]) for batch in batches(data, held=True) for image in batch["image"]]
    assert sorted(train) == list(range(14, 26)) and held == list(range(10, 14))
    assert set(train).isdisjoint(held)
    with pytest.raises((FileNotFoundError, ValueError)):
        load("tfds/dew_images:2.0.0", batch=4, loading=READ, provision=False)


def test_a_dataset_keeps_its_steps_while_its_record_default_is_held_out():
    source = Memory([{"id": index} for index in range(320)]).map(
        lambda row: {"id": row["id"], "twice": 2 * row["id"]})
    data = load(source, batch=8, loading=READ, provision=False)
    train, held = batches(data), batches(data, held=True)
    trained = {int(value) for batch in train for value in batch["id"]}
    scored = {int(value) for batch in held for value in batch["id"]}
    assert trained == set(range(256, 320)) and scored == set(range(256))
    assert trained.isdisjoint(scored)
    for batch in train + held:
        np.testing.assert_array_equal(batch["twice"], 2 * batch["id"])


def draw(row, rng):
    return {**row, "draw": rng.integers(2**16)}


def test_load_keeps_an_indexed_datasets_random_draws_and_resumed_position_at_the_requested_seed():
    source = Memory([{"id": index} for index in range(320)]).random_map(draw)
    options = {"batch": 8, "read": "indexed", "seed": 37, "loading": READ}
    data = load(source, **options)
    train, held = source.holdout(records=256)
    reference = compile(train.batch(8), validation=held.batch(8), read="indexed", seed=37, loading=READ)
    actual, expected = data.train(DataPartition()), reference.train(DataPartition())
    changed = load(source, **{**options, "seed": 38}).train(DataPartition())
    try:
        first = next(actual)
        np.testing.assert_array_equal(first["draw"], next(expected)["draw"])
        assert not np.array_equal(first["draw"], next(changed)["draw"])
        saved = actual.get_state()
        following = next(actual)
        resumed = load(source, **options).train(DataPartition())
        try:
            resumed.set_state(saved)
            restored = next(resumed)
            for field in following:
                np.testing.assert_array_equal(restored[field], following[field])
        finally:
            resumed.close()
    finally:
        actual.close()
        expected.close()
        changed.close()


def test_text_data_defaults_end_documents_and_hold_out_tokens_without_replacing_named_validation():
    source = [{"text": "v" * 10 + "t" * 990}]
    data = TextData(source, seq_len=1).load(batch=1, read="memory", loading=READ)
    train, held = batches(data), batches(data, held=True)
    assert data.records == 990 and len(held) == 9
    assert {int(token) for batch in train for token in batch["text"].flat} == {ord("t"), 255}
    assert {int(token) for batch in held for token in batch["text"].flat} == {ord("v")}
    named = TextData(source, seq_len=1, validation=[{"text": "z" * 10}]).load(
        batch=1, read="memory", loading=READ)
    assert named.records == 1000
    assert {int(token) for batch in batches(named, held=True) for token in batch["text"].flat} == {
        ord("z"), 255}
    entire = TextData(source, seq_len=1, token_holdout=None).load(batch=1, read="memory", loading=READ)
    assert entire.records == 1000 and entire.val is None


def test_a_token_store_keeps_its_producer_validation_split_with_text_data_defaults(tmp_path):
    write_token_corpus(tmp_path, np.arange(100, 201, dtype=np.uint16),
                       np.arange(500, 525, dtype=np.uint16))
    data = TextData(str(tmp_path), seq_len=4).load(batch=1, read="memory", loading=READ)
    assert data.records == 25
    trained = {int(token) for batch in batches(data) for token in batch["text"].flat}
    scored = {int(token) for batch in batches(data, held=True) for token in batch["text"].flat}
    assert trained == set(range(100, 201)) and scored == set(range(500, 525))
    assert trained.isdisjoint(scored)


def test_image_data_defaults_hold_out_256_records_and_resize_both_parts():
    source = [{"image": np.full((8, 8, 3), index % 256, np.uint8), "label": index}
              for index in range(320)]
    data = ImageData(source, image_size=4, augment=None).load(batch=8, read="memory", loading=READ)
    train, held = batches(data), batches(data, held=True)
    trained = {int(label) for batch in train for label in batch["label"]}
    scored = {int(label) for batch in held for label in batch["label"]}
    assert data.records == 64 and data.held_out == 256
    assert trained == set(range(256, 320)) and scored == set(range(256))
    assert trained.isdisjoint(scored)
    for batch in train + held:
        np.testing.assert_array_equal(batch["image"], np.broadcast_to(
            (batch["label"] % 256).astype(np.uint8)[:, None, None, None], (8, 4, 4, 3)))


@pytest.mark.parametrize("kind", ["records", "text", "chat", "prompts", "preferences", "images",
                                  "video", "shakespeare"])
def test_each_run_spec_loads_the_same_batches_and_plan_as_loader(kind, tmp_path):
    if kind == "video":
        pytest.importorskip("moviepy.config", reason="needs the av extra")
        from test_data import _voxceleb_tree
        from transformers import Wav2Vec2FeatureExtractor

        _voxceleb_tree(tmp_path)
        extractor = tmp_path / "extractor"
        Wav2Vec2FeatureExtractor(sampling_rate=16000).save_pretrained(extractor)
        spec = VideoData(str(tmp_path / "train"), frames=2, frame_size=8, audio_padding=0,
                         audio_model=str(extractor), validation=HoldOut(records=1))
    else:
        text = [{"text": "v" * 10 + "t" * 990}]
        conversations = [{"messages": [{"role": "user", "content": "hello"},
                                        {"role": "assistant", "content": "world"}]}] * 8
        specs = {
            "records": RecordsData([{"id": index} for index in range(8)], validation=HoldOut(records=2)),
            "text": TextData(text, seq_len=3),
            "chat": ChatData(conversations, str(FIXTURES / "tokenizers" / "tiny-chat"), seq_len=16,
                             validation=HoldOut(records=2)),
            "prompts": PromptData([{"prompt": [1, 2, 3]}] * 8, "byte", max_prompt_len=4),
            "preferences": PreferenceData([{"chosen": [1, 2, 3], "rejected": [1, 2, 4],
                                            "chosen_mask": [0, 1, 1], "rejected_mask": [0, 1, 1]}] * 8,
                                          seq_len=4),
            "images": ImageData([{"image": np.full((4, 4, 3), index % 256, np.uint8)}
                                  for index in range(320)], image_size=4, augment=None),
            "shakespeare": TinyShakespeareData(text, seq_len=3),
        }
        spec = specs[kind]
    options = {"batch": 1, "seed": 21, "read": "memory", "loading": READ}
    direct, method = loader(spec, **options), spec.load(**options)
    assert method.records == direct.records and method.held_out == direct.held_out
    assert method.plan.record() == direct.plan.record()
    for held in (False, True) if method.val is not None else (False,):
        left, right = batches(method, held=held), batches(direct, held=held)
        assert len(left) == len(right)
        for actual, expected in zip(left, right, strict=True):
            assert actual.keys() == expected.keys()
            for field in actual:
                np.testing.assert_array_equal(actual[field], expected[field])
