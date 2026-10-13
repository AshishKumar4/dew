"""URL fetches use the ordinary pipeline worker and rejection paths."""

import itertools
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from dew.data import Files, Loading, Memory, Rejected, images, video
from dew.data.compile import compile, iterate
from dew.data.dataset import DataPartition
from dew.data.mixing import SearchBudget
from dew.data.pipeline import Pipeline
from dew.data.sources.urls import Fetch
from dew.data.steps.image import DecodeImage, Resize
from dew.objectives.base import VALID_ROWS

READ = Loading(workers=0, threads=1, read_buffer=2, worker_buffer=1)


def catalog(root: Path, count: int = 16) -> Pipeline:
    rows = []
    for index in range(count):
        path = root / f"image-{index}.png"
        Image.fromarray(np.full((8, 8, 3), index + 1, np.uint8)).save(path)
        rows.append({"url": path.as_uri(), "caption": f"sample {index}", "label": index})
    return Pipeline(Memory(rows)).filter_map(Fetch(output="image"), DecodeImage()).map(Resize(8))


def _on_white(rgba: np.ndarray) -> np.ndarray:
    """img2dataset's alpha matting: c * a / 255 + 255 - a, rounded, in float64."""
    alpha = rgba[..., 3:].astype(np.float64)
    return np.rint(rgba[..., :3] * alpha / 255 + 255 - alpha).astype(np.uint8)


@pytest.mark.parametrize("pixels", [
    pytest.param(np.random.RandomState(0).randint(0, 256, (48, 40), np.uint8), id="grayscale"),
    pytest.param(np.random.RandomState(0).randint(0, 256, (48, 40, 4), np.uint8), id="rgba"),
])
def test_a_fetched_grey_or_transparent_image_is_kept_as_rgb(tmp_path, pixels):
    """Grey is replicated and transparency composited onto white."""
    expected = np.repeat(pixels[..., None], 3, axis=-1) if pixels.ndim == 2 else _on_white(pixels)
    path = tmp_path / "pixels.png"
    Image.fromarray(pixels).save(path)
    fetched = Fetch(output="image")({"url": path.as_uri()})
    assert not isinstance(fetched, Rejected)
    decoded = DecodeImage().map(fetched)
    assert not isinstance(decoded, Rejected)
    np.testing.assert_array_equal(decoded["image"], expected)
    assert images.min_side(32)(decoded) and images.max_aspect(2.4)(decoded) and images.not_flat()(decoded)
    assert Resize(64).map(decoded)["image"].shape == (64, 64, 3)


def test_fetch_preserves_the_catalog_and_reads_exact_bytes(tmp_path):
    path = tmp_path / "payload"
    path.write_bytes(b"source bytes")
    row = Fetch()({"url": path.as_uri(), "caption": "the record"})
    assert not isinstance(row, Rejected)
    assert row["bytes"] == b"source bytes" and row["caption"] == "the record"


def test_failed_fetches_are_counted_before_batches(tmp_path):
    prefix = catalog(tmp_path)
    rows = [{"url": (tmp_path / "missing").as_uri(), "label": -1},
            *prefix.source.records("indexed")]
    pipeline = Pipeline(Memory(rows)).shuffle(7).repeat().filter_map(
        Fetch(output="image"), DecodeImage()).map(Resize(8)).batch(4)
    data = compile(pipeline, read="indexed", loading=READ)
    stream = data.train(DataPartition())
    try:
        batches = list(itertools.islice(stream, 8))
    finally:
        stream.close()
    assert all(np.all(batch["label"] >= 0) for batch in batches)
    assert all(np.array_equal(batch["image"][:, 0, 0, 0], batch["label"] + 1) for batch in batches)


def test_a_transient_failure_is_not_a_permanent_tombstone(tmp_path):
    result = Fetch()({"url": (tmp_path / "missing").as_uri()})
    assert isinstance(result, Rejected)
    assert result.reason == "transient" and not result.permanent


def test_an_oversized_object_is_rejected_before_decode(tmp_path):
    path = tmp_path / "oversized"
    path.write_bytes(b"x" * 17)
    result = Fetch(max_bytes=16)({"url": path.as_uri()})
    assert isinstance(result, Rejected) and result.reason == "too_large"


def test_an_all_rejected_catalog_exhausts_its_search_budget(tmp_path):
    pipeline = Pipeline(Memory([{"url": (tmp_path / "missing").as_uri()}])).repeat().filter_map(
        Fetch(output="image"), DecodeImage()).batch(4)
    data = compile(pipeline, read="indexed", loading=READ,
                   budget=SearchBudget(consecutive=8, seconds=2))
    stream = data.train(DataPartition())
    try:
        with pytest.raises(ValueError):
            next(stream)
    finally:
        stream.close()


@pytest.mark.parametrize("workers", [0, 2])
def test_real_workers_fetch_pixels_without_losing_validation_rows(tmp_path, workers):
    prefix = catalog(tmp_path, 13)
    data = compile(Pipeline(prefix.source).shuffle(3).repeat().filter_map(*(
                       stage.step for stage in prefix.stages)).batch(4), validation=prefix.batch(4),
                   read="indexed", loading=Loading(workers=workers, threads=1, read_buffer=2,
                                                   worker_buffer=1))
    assert data.val is not None
    stream = data.val(DataPartition())
    try:
        batches = list(stream)
    finally:
        stream.close()
    labels = np.concatenate([batch["label"][batch.get(VALID_ROWS, np.ones(4, bool))] for batch in batches])
    assert labels.tolist() == list(range(13))
    for batch in batches:
        np.testing.assert_array_equal(batch["image"][:, 0, 0, 0], batch["label"] + 1)


def test_a_finite_fetch_pass_ends_after_its_catalog(tmp_path):
    rows = list(iterate(catalog(tmp_path, 3), read="indexed", loading=READ))
    assert [row["label"] for row in rows] == [0, 1, 2]
    assert all(np.all(row["image"] == index + 1) for index, row in enumerate(rows))


def web_table(root: Path) -> list[dict]:
    """A url table as the open web serves it: images worth training on (labels 0-7), and ones
    that are too small, too elongated, one flat colour, not an image, gone, or not a url."""
    rng = np.random.default_rng(0)

    def saved(name, pixels):
        path = root / f"{name}.png"
        Image.fromarray(pixels).save(path)
        return path.as_uri()

    rows = [{"url": saved(f"good-{index}", rng.integers(0, 256, (32, 40, 3), np.uint8)), "label": index}
            for index in range(8)]
    (root / "page.html").write_text("<html>moved</html>")
    return [*rows,
            {"url": saved("small", rng.integers(0, 256, (8, 8, 3), np.uint8)), "label": -1},
            {"url": saved("strip", rng.integers(0, 256, (16, 96, 3), np.uint8)), "label": -2},
            {"url": saved("flat", np.full((32, 32, 3), 7, np.uint8)), "label": -3},
            {"url": (root / "page.html").as_uri(), "label": -4},
            {"url": (root / "gone.png").as_uri(), "label": -5},
            {"url": None, "label": -6}]


def rejected(stream) -> dict[str, float]:
    """The reader's dropped records by reason, as a run logs them."""
    return {key.rsplit("/", 1)[1]: count for key, count in stream.statistics().items()
            if key.startswith("data/rejected/") and count}


def worth_training_on(pipeline: Pipeline) -> Pipeline:
    """The old url loader's checks, as stages."""
    return (pipeline.filter_map(images.decode()).filter(images.min_side(16)).filter(images.max_aspect(2.4))
            .filter(images.not_flat()))


def test_a_web_image_flow_trains_on_the_good_images_and_counts_each_drop_by_reason(tmp_path):
    """Fetched in four threads, decoded and checked: only the good images reach a batch, and the
    reader's counts name every reason a url was dropped."""
    fetched = Pipeline(Memory(web_table(tmp_path))).shuffle(5).repeat().filter_map(images.fetch())
    data = compile(worth_training_on(fetched).map(Resize(16)).batch(4), read="indexed",
                   loading=Loading(workers=0, threads=4, read_buffer=8, worker_buffer=1))
    stream = data.train(DataPartition())
    try:
        batches = list(itertools.islice(stream, 12))
        dropped = rejected(stream)
    finally:
        stream.close()
    labels = np.concatenate([batch["label"] for batch in batches])
    assert set(labels.tolist()) == set(range(8))
    assert set(dropped) == {"too_small", "too_elongated", "flat", "undecodable", "transient", "invalid_url"}


def test_a_snapshot_of_fetched_bytes_trains_and_resumes_exactly_after_every_url_dies(tmp_path):
    """A snapshot seals what a fetch got: with every url gone, a run reading it back with Files
    trains on the original pixels, and resuming from a saved position continues it exactly."""
    table = web_table(tmp_path)
    Memory(table).filter_map(images.fetch()).snapshot(str(tmp_path / "snapshot"))
    originals = {row["label"]: np.asarray(Image.open(row["url"].removeprefix("file://")).convert("RGB"))
                 for row in table[:8]}
    for path in tmp_path.glob("*.png"):
        path.unlink()
    pipeline = worth_training_on(Files(str(tmp_path / "snapshot")).shuffle(5).repeat()).batch(4)
    data = compile(pipeline, read="indexed", loading=READ)
    stream = data.train(DataPartition())
    try:
        first = [next(stream) for _ in range(3)]
        state = stream.get_state()
        rest = [next(stream) for _ in range(3)]
    finally:
        stream.close()
    for batch in first + rest:
        for label, pixels in zip(batch["label"], batch["image"], strict=True):
            np.testing.assert_array_equal(pixels, originals[int(label)])
    resumed = data.train(DataPartition())
    try:
        resumed.set_state(state)
        again = [next(resumed) for _ in range(3)]
    finally:
        resumed.close()
    for expected, found in zip(rest, again, strict=True):
        np.testing.assert_array_equal(expected["image"], found["image"])


def test_a_web_video_flow_decodes_frames_from_fetched_bytes(tmp_path):
    """Clips fetched by url decode from their bytes to frames; bytes that are no clip are dropped."""
    from test_data_av import CLIP_FRAMES, FPS, GREY_STEP, _mux

    clip = Path(_mux(tmp_path / "clip.avi", sound=0))
    (tmp_path / "broken.mp4").write_bytes(b"not a video")
    table = [{"url": clip.as_uri(), "caption": "a fade"}, {"url": (tmp_path / "broken.mp4").as_uri()}]
    pipeline = (Pipeline(Memory(table)).shuffle(2).repeat().filter_map(video.fetch())
                .filter_map(video.pick_clip(8 / FPS)).filter_map(video.decode(8, size=16)).batch(2))
    data = compile(pipeline, read="indexed", loading=READ)
    stream = data.train(DataPartition())
    try:
        batch = next(stream)
        dropped = rejected(stream)
    finally:
        stream.close()
    assert batch["video"].shape == (2, 8, 16, 16, 3)
    for frames in batch["video"]:
        greys = [round(int(frame[8, 8, 0]) / GREY_STEP) for frame in frames]
        assert greys == list(range(greys[0], greys[0] + 8)) and greys[-1] < CLIP_FRAMES
    assert dropped["undecodable"] > 0
