"""URL fetches use the ordinary pipeline worker and rejection paths."""

import collections
import gzip
import hashlib
import itertools
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import grain.python as grain
import numpy as np
import pytest
from PIL import Image
from process_support import serving

from dew.batch import VALID_ROWS
from dew.data import Files, Loading, Memory, Rejected, images, video
from dew.data.compile import compile, iterate
from dew.data.dataset import DataPartition
from dew.data.images import DecodeImage, Resize
from dew.data.pipeline import Pipeline
from dew.data.sources.urls import Fetch

READ = Loading(workers=0, threads=1, read_buffer=2, worker_buffer=1)


def catalog(root: Path, count: int = 16) -> Pipeline:
    rows = []
    for index in range(count):
        path = root / f"image-{index}.png"
        Image.fromarray(np.full((8, 8, 3), index + 1, np.uint8)).save(path)
        rows.append({"url": path.as_uri(), "caption": f"sample {index}", "label": index})
    return Pipeline(Memory(rows)).filter_map(Fetch(output="image"), DecodeImage("image")).map(Resize(8))


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
    decoded = DecodeImage("image").map(fetched)
    assert not isinstance(decoded, Rejected)
    np.testing.assert_array_equal(decoded["image"], expected)
    assert images.MinSide(32)(decoded) and images.MaxAspect(2.4)(decoded) and images.NotFlat()(decoded)
    assert Resize(64).map(decoded)["image"].shape == (64, 64, 3)


def test_fetch_preserves_the_catalog_and_reads_exact_bytes(tmp_path):
    path = tmp_path / "payload"
    path.write_bytes(b"source bytes")
    row = Fetch()({"url": path.as_uri(), "caption": "the record"})
    assert not isinstance(row, Rejected)
    assert row["bytes"] == b"source bytes" and row["caption"] == "the record"


@pytest.mark.parametrize("read", ["indexed", "streamed", "interleaved", "memory"])
def test_a_fetch_with_threads_of_its_own_fetches_that_many_at_once_in_order_and_resumes(read):
    """A fetch given four threads of its own fetches four urls at once whatever the read's threads,
    in every read, before an interleave too; delivers the records it keeps in the order they were
    read, a gone url dropped and counted; leaves the plan as a fetch without threads makes it; a
    position saved part way restores to the same remaining records; and a closed reader leaves no
    thread of its pool running."""
    handle = Slow(gone="9", hold={"": 4})
    with serving(handle) as address:
        catalog = Memory([{"url": f"http://{address}/{index}", "id": index} for index in range(17)],
                         shard_rows=17)

        def built(threads):
            fetched = Pipeline(catalog).filter_map(images.fetch(to="bytes", threads=threads))
            chain = (fetched.interleave(1) if read == "interleaved" else fetched).batch(2)
            return compile(chain, read="streamed" if read == "interleaved" else read,
                           loading=Loading(workers=0, threads=1, read_buffer=1))

        data = built(4)
        assert data.plan.identity() == built(None).plan.identity()
        reader = data.train(DataPartition())
        try:
            first = [next(reader) for _ in range(3)]
            saved = reader.get_state()
            rest = [next(reader) for _ in range(5)]
        finally:
            reader.close()
        # The loader's counts, which every read keeps (a memory read's as it reads its source).
        dropped = rejected(data)
        assert not [thread for thread in threading.enumerate() if thread.name.startswith("dew-step")]
        resumed = data.train(DataPartition())
        try:
            resumed.set_state(saved)
            again = [next(resumed) for _ in range(5)]
        finally:
            resumed.close()
    assert handle.most == {"": 4}
    kept = [index for index in range(17) if index != 9]
    assert [int(value) for batch in [*first, *rest] for value in batch["id"]] == kept
    assert dropped == {"http_404": 1}
    assert [bytes(value) for batch in rest for value in batch["bytes"]] == [
        f"/{index}".encode() for index in kept[6:]]
    assert [[int(value) for value in batch["id"]] for batch in again] == [
        [int(value) for value in batch["id"]] for batch in rest]


@pytest.mark.parametrize("threads", [1, 8], ids=["fewer read threads", "more read threads"])
@pytest.mark.parametrize("read", ["indexed", "memory"])
def test_a_cached_fetch_with_threads_of_its_own_fetches_that_many_at_once(tmp_path, read, threads):
    """A fetch given four threads of its own before a cache, and another given two after it, each
    fetches as many urls at once as it was given whatever the read's threads, and the run reads
    what the uncached chain reads. Each run reads from a server of its own, so what it counts is
    its own fetches alone."""

    def ids(cache):
        handle = Slow(sides=True, hold={"before": 4, "after": 2})
        with serving(handle) as address:
            urls = [{"url": f"http://{address}/before/{index}", "id": index} for index in range(12)]
            catalog = Memory(urls)
            fetched = Pipeline(catalog).filter_map(images.fetch(to="bytes", threads=4))
            cached = fetched if cache is None else fetched.cache(cache)
            again = cached.map(Again(address)).filter_map(images.fetch("again", to="later", threads=2))
            loading = Loading(workers=0, threads=threads, read_buffer=threads)
            reader = compile(again.repeat(1).batch(3), read=read, loading=loading).train(DataPartition())
            try:
                # Each body is its url's path, the same from either server.
                read_ids = [[(int(row), bytes(later))
                             for row, later in zip(batch["id"], batch["later"], strict=True)]
                            for batch in reader]
            finally:
                reader.close()
        assert handle.most == {"before": 4, "after": 2}
        return read_ids

    assert ids(str(tmp_path / "cache")) == ids(None)


@dataclass(frozen=True)
class Again:
    """Gives a fetched record a second url, served after the cache."""

    address: str
    identity = "again@1"

    def __call__(self, row):
        return {**row, "again": f"http://{self.address}/after/{int(row['id'])}"}


class Slow:
    """A server's answer to each GET: its path, after a pause, or 404 for the path `gone`; it counts
    the requests in flight, by their path's first part where it reads `sides`, and the most at once.
    A request is in flight from its arrival until its answer starts, so a client that has its answer
    and asks again is never counted twice, however late its handler's thread finishes.

    Each side's first `hold[side]` requests are answered only once that many are in flight together,
    so a client that fetches that many at once is seen to, however loaded the machine that runs it:
    one that fetched fewer would leave them waiting until its own fetch timeout dropped them. The
    most at once is then exactly what the client fetches together, never more."""

    def __init__(self, *, gone: str | None = None, sides: bool = False,
                 hold: Mapping[str, int] | None = None):
        self.gone, self.sides, self.hold = gone, sides, dict(hold or {})
        self.lock = threading.Condition()
        self.active, self.most = collections.Counter(), collections.Counter()
        self.arrived = collections.Counter()

    def __call__(self, request):
        path = request.path.strip("/")
        if path == self.gone:
            request.send_response(404)
            request.end_headers()
            return
        side = path.split("/")[0] if self.sides else ""
        with self.lock:
            self.active[side] += 1
            self.most[side] = max(self.most[side], self.active[side])
            self.arrived[side] += 1
            held = self.hold.get(side, 0)
            if self.arrived[side] <= held:
                self.lock.notify_all()
                self.lock.wait_for(lambda: self.arrived[side] >= held)
        time.sleep(0.2)
        with self.lock:
            self.active[side] -= 1
        body = request.path.encode()
        request.send_response(200)
        request.send_header("Content-Length", str(len(body)))
        request.end_headers()
        request.wfile.write(body)


Parents = collections.namedtuple("Parents", ["first", "second"])


class TupleWrapped(grain.IterDataset):
    """A Grain step of the caller's, whose position keeps its parent's in a tuple, or in a named
    tuple (`named`)."""

    named = False

    def __iter__(self):
        return TupleWrappedRead(self._parent.__iter__(), self.named)


class NamedTupleWrapped(TupleWrapped):
    named = True


class TupleWrappedRead(grain.DatasetIterator):
    def __init__(self, parent, named):
        super().__init__(parent)
        self.named = named

    def __next__(self):
        return next(self._parent)

    def get_state(self):
        state = self._parent.get_state()
        return {"parents": Parents(state, None) if self.named else (state,)}

    def set_state(self, state):
        self._parent.set_state(state["parents"][0])


@pytest.mark.parametrize("repeats", ["endless", "two passes, endlessly", "behind a Grain step",
                                     "behind a Grain step's named tuple"])
def test_an_endless_read_reads_on_through_passes_whose_fetches_failed_until_one_returns(repeats):
    """A catalog's one url fails for its first three fetches, so each of those passes yields
    nothing; a failed fetch may succeed next time, so the endless read reads on, counting each
    failure through every pass, a finite repeat's within an endless one's too, and gives the record
    once the server returns it, where the first empty pass was once refused."""
    served = []

    def handle(request):
        served.append(request.path)
        if len(served) <= 3:
            request.send_response(503)
            request.end_headers()
            return
        request.send_response(200)
        request.send_header("Content-Length", "7")
        request.end_headers()
        request.wfile.write(b"payload")

    with serving(handle) as address:
        fetched = Pipeline(Memory([{"url": f"http://{address}/object"}])).filter_map(Fetch())
        if repeats.startswith("behind"):
            wrapped = NamedTupleWrapped if repeats.endswith("named tuple") else TupleWrapped
            fetched = fetched.grain(wrapped, identity=f"{wrapped.__name__}@1")
        repeated = fetched.repeat(2).repeat() if repeats == "two passes, endlessly" else fetched.repeat()
        chain = repeated.batch(1)
        data = compile(chain, read="indexed", loading=READ)
        stream = data.train(DataPartition())
        try:
            assert np.asarray(next(stream)["bytes"]).tolist() == [b"payload"]
        finally:
            stream.close()
    assert len(served) == 4
    # The passes a fetch failed in are counted, as a read that goes on through them is watched by.
    assert data.statistics()["data/rejected/memory/transient"] == 3


@pytest.mark.parametrize("header", ["Location", "URI"])
def test_a_fetch_follows_a_redirect_and_keeps_the_bytes_the_server_sent(header):
    """A fetch follows its server's redirect, named by `Location` or by `URI`, and keeps the body
    as sent, compressed bytes and all: its digest names those bytes, and its size limit counts
    them, not their inflation."""
    body = gzip.compress(b"x" * 4096)

    def handle(request):
        if request.path == "/moved":
            request.send_response(302)
            request.send_header(header, "/object")
            request.end_headers()
            return
        request.send_response(200)
        request.send_header("Content-Encoding", "gzip")
        request.send_header("Content-Length", str(len(body)))
        request.end_headers()
        request.wfile.write(body)

    with serving(handle) as address:
        row = Fetch(max_bytes=len(body))({"url": f"http://{address}/moved"})
    assert not isinstance(row, Rejected)
    assert row["bytes"] == body and row["content_sha256"] == hashlib.sha256(body).hexdigest()


@pytest.mark.parametrize("hops", [0, 10])
def test_a_redirect_to_a_local_file_is_not_followed(tmp_path, hops):
    """A server's redirect to a file on this host is never followed, first or past the hops a
    fetch follows: the fetch is dropped, and the file's bytes never enter a record."""
    secret = tmp_path / "secret"
    secret.write_bytes(b"local bytes")

    def handle(request):
        step = int(request.path.strip("/") or 0)
        request.send_response(302)
        request.send_header("Location", secret.as_uri() if step >= hops else f"/{step + 1}")
        request.end_headers()

    with serving(handle) as address:
        fetched = Fetch()({"url": f"http://{address}/0"})
    assert isinstance(fetched, Rejected) and not fetched.permanent


def test_a_fetch_reaches_a_host_its_environment_excludes_from_its_proxy(monkeypatch):
    """A host and port the environment excludes from its proxy (NO_PROXY) is fetched directly,
    though a proxy is named for its scheme: the proxy never sees the request."""
    proxied = []

    def proxy(request):
        proxied.append(request.path)
        request.send_response(502)
        request.end_headers()

    def origin(request):
        request.send_response(200)
        request.send_header("Content-Length", "6")
        request.end_headers()
        request.wfile.write(b"direct")

    with serving(proxy) as relay, serving(origin) as address:
        monkeypatch.setenv("http_proxy", f"http://{relay}")
        monkeypatch.setenv("no_proxy", address)
        row = Fetch()({"url": f"http://{address}/object"})
    assert not isinstance(row, Rejected) and row["bytes"] == b"direct" and proxied == []


def test_a_fetch_goes_through_the_proxy_its_environment_names_by_its_authority(monkeypatch):
    """A proxy the environment names by host and port alone carries the fetch, as an HTTP proxy."""
    proxied = []

    def proxy(request):
        proxied.append(request.path)
        request.send_response(200)
        request.send_header("Content-Length", "7")
        request.end_headers()
        request.wfile.write(b"relayed")

    with serving(proxy) as relay:
        monkeypatch.setenv("http_proxy", relay)
        monkeypatch.delenv("no_proxy", raising=False)
        monkeypatch.delenv("NO_PROXY", raising=False)
        row = Fetch()({"url": "http://catalog.invalid/object"})
    assert not isinstance(row, Rejected) and row["bytes"] == b"relayed"
    assert proxied == ["http://catalog.invalid/object"]


def test_a_file_that_keeps_arriving_past_the_timeout_is_dropped(tmp_path):
    """A file read by urllib whose bytes keep arriving, each well inside its timeout, until after
    it has passed (a pipe its writer trickles into) is dropped as transient, as a slow server is."""
    import os

    pipe = tmp_path / "pipe"
    os.mkfifo(pipe)

    def trickle():
        with open(pipe, "wb") as writer:
            for _ in range(15):
                writer.write(b"x")
                writer.flush()
                time.sleep(0.1)

    writer = threading.Thread(target=trickle, daemon=True)
    writer.start()
    fetched = Fetch(timeout=1.0)({"url": pipe.as_uri()})
    writer.join()
    assert isinstance(fetched, Rejected) and not fetched.permanent


def test_a_fetch_ends_at_its_timeout_however_long():
    """A fetch's timeout ends it when it passes, at five seconds and over as below: a body that
    would finish half a second later is dropped as transient by then."""
    def handle(request):
        request.send_response(200)
        request.send_header("Content-Length", "11")
        request.end_headers()
        for _ in range(11):
            request.wfile.write(b"x")
            request.wfile.flush()
            time.sleep(0.5)

    with serving(handle) as address:
        began = time.monotonic()
        fetched = Fetch(timeout=5.0)({"url": f"http://{address}/"})
        waited = time.monotonic() - began
    assert isinstance(fetched, Rejected) and not fetched.permanent
    assert waited < 5.4


def test_failed_fetches_are_counted_before_batches(tmp_path):
    prefix = catalog(tmp_path)
    rows = [{"url": (tmp_path / "missing").as_uri(), "label": -1},
            *prefix.source.records("indexed")]
    pipeline = Pipeline(Memory(rows)).shuffle(7).repeat().filter_map(
        Fetch(output="image"), DecodeImage("image")).map(Resize(8)).batch(4)
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
    """The dropped records of a reader or a loader by reason, as a run logs them."""
    return {key.rsplit("/", 1)[1]: count for key, count in stream.statistics().items()
            if key.startswith("data/rejected/") and count}


def worth_training_on(pipeline: Pipeline) -> Pipeline:
    """The old url loader's checks, as stages."""
    return (pipeline.filter_map(images.DecodeImage()).filter(images.MinSide(16)).filter(images.MaxAspect(2.4))
            .filter(images.NotFlat()))


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


@pytest.mark.parametrize(("width", "height"), [(40, 20), (23, 40), (16, 16), (8, 6)])
def test_a_fetched_web_image_is_fitted_whole_into_its_square(tmp_path, width, height):
    """A fetched image fitted into a 16-pixel square keeps all it shows, as the retired online
    loader fitted it: its longer side 16 pixels (area down, bicubic up), the shorter in proportion,
    rounded, and the rest white, split evenly with the odd pixel at the bottom or right."""
    import cv2

    rng = np.random.default_rng(width * height)
    pixels = rng.integers(0, 256, (height, width, 3), np.uint8)
    Image.fromarray(pixels).save(tmp_path / "image.png")
    chain = (Pipeline(Memory([{"url": (tmp_path / "image.png").as_uri()}])).filter_map(images.fetch(to="png"))
             .filter_map(DecodeImage(field="png", to="image")).map(images.Fit(16)).batch(1))
    reader = compile(chain, read="indexed", loading=READ).train(DataPartition())
    try:
        fitted = np.asarray(next(reader)["image"][0])
    finally:
        reader.close()
    scale = 16 / max(width, height)
    tall, wide = max(1, round(height * scale)), max(1, round(width * scale))
    top, left = (16 - tall) // 2, (16 - wide) // 2
    method = cv2.INTER_AREA if max(width, height) > 16 else cv2.INTER_CUBIC
    assert fitted.shape == (16, 16, 3)
    np.testing.assert_array_equal(fitted[top : top + tall, left : left + wide],
                                  cv2.resize(pixels, (wide, tall), interpolation=method))
    inside = np.zeros((16, 16), bool)
    inside[top : top + tall, left : left + wide] = True
    assert (fitted[~inside] == 255).all()


def test_a_fetched_clip_under_the_least_size_is_dropped_before_it_is_fitted(tmp_path):
    """Clips stored at 32 pixels a side, read for 16-pixel frames: with a least size of 48 they
    are counted and dropped as too small before they are fitted, as the retired loader dropped
    them; with 32 they are read."""
    from test_data_av import FPS, _mux

    clip = Path(_mux(tmp_path / "clip.avi", sound=0))

    def read(least):
        chain = (Pipeline(Memory([{"url": clip.as_uri()}])).repeat(1).filter_map(video.fetch())
                 .random_filter_map(video.PickClip(8 / FPS))
                 .filter_map(video.DecodeVideo(8, size=16, min_size=least)).batch(1))
        reader = compile(chain, read="indexed", loading=READ).train(DataPartition())
        try:
            return [batch["video"].shape for batch in reader], rejected(reader)
        finally:
            reader.close()

    assert read(32) == ([(1, 8, 16, 16, 3)], {})
    assert read(48) == ([], {"too_small": 1})


def test_a_web_video_flow_decodes_frames_from_fetched_bytes(tmp_path):
    """Clips fetched by url decode from their bytes to frames; bytes that are no clip are dropped."""
    from test_data_av import CLIP_FRAMES, FPS, GREY_STEP, _mux

    clip = Path(_mux(tmp_path / "clip.avi", sound=0))
    (tmp_path / "broken.mp4").write_bytes(b"not a video")
    table = [{"url": clip.as_uri(), "caption": "a fade"}, {"url": (tmp_path / "broken.mp4").as_uri()}]
    pipeline = (Pipeline(Memory(table)).shuffle(2).repeat().filter_map(video.fetch())
                .random_filter_map(video.PickClip(8 / FPS)).filter_map(video.DecodeVideo(8, size=16))
                .batch(2))
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
