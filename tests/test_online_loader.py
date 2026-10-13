"""Streaming loader tests: what reaches the model when fetching goes wrong.

Most of these replace the fetcher pool with a stub producer. One simulates a
missing HF `datasets` to check that importing the streaming stack does not
need the `streaming` extra. The tests that run the real pool cannot stub the
fetch, because a worker that was not forked holds no copy of this process's
memory, so they assert on what the workers report. Nothing here touches the
network: every url is a file:// url, of a file under tmp_path or of one that
does not exist.
"""

import contextlib
import io
import os
import sys
import threading
import time

import numpy as np
import PIL.Image
import pytest

from dew.data import online_loader
from dew.data.online_loader import Fetch, UrlStream

FETCH = Fetch(size=64, min_size=32, timeout=1, retries=0)


class _StubRows:
    """As much of an HF dataset as the stream and the fetcher pool read.

    Column oriented like HF's, so a shard is a dict of lists. After `passes`
    passes the reshuffle hands back an empty table, so the pool's otherwise
    endless loop emits nothing more and a test sees exactly the rows of those
    passes, however the workers' queue feeders interleave. The urls are files
    that do not exist, so a real fetch fails at once and without a network.
    """

    def __init__(self, size, passes=1, root="/nonexistent"):
        self.size = size
        self.passes = passes
        self.root = root

    def shard(self, num_shards, index):
        return self

    def shuffle(self, seed=0):
        return self if seed < self.passes else _StubRows(0, 0, self.root)

    def __len__(self):
        return self.size

    def __getitem__(self, window):
        indices = range(*window.indices(self.size))
        return {"url": [f"file://{self.root}/{i}.jpg" for i in indices],
                "caption": [f"caption {i}" for i in indices]}


def _png(pixels: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    PIL.Image.fromarray(pixels).save(buffer, format="png")
    return buffer.getvalue()


# ---------------------------------------------------------------------------------
# Slow fetching waits; it never invents data
# ---------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------
# A fetcher that is gone ends iteration, and says why
# ---------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------
# One url: fetch, decode, fit
# ---------------------------------------------------------------------------------

def _on_white(rgba: np.ndarray) -> np.ndarray:
    """img2dataset's alpha matting: c * a / 255 + 255 - a, rounded, in float64."""
    alpha = rgba[..., 3:].astype(np.float64)
    return np.rint(rgba[..., :3] * alpha / 255 + 255 - alpha).astype(np.uint8)


@pytest.mark.parametrize("pixels", [
    pytest.param(np.random.RandomState(0).randint(0, 256, (48, 40), np.uint8), id="grayscale"),
    pytest.param(np.random.RandomState(0).randint(0, 256, (48, 40, 4), np.uint8), id="rgba"),
])
def test_a_fetched_grey_or_transparent_image_is_kept_as_rgb(pixels):
    """Grey is replicated and transparency composited onto white."""
    expected = np.repeat(pixels[..., None], 3, axis=-1) if pixels.ndim == 2 else _on_white(pixels)

    fetched = online_loader.decode_pixels(_png(pixels))

    assert fetched is not None
    np.testing.assert_array_equal(fetched, expected)
    assert online_loader.prepare_image(fetched, size=64, min_size=32) is not None


def test_a_kept_image_is_fitted_into_the_square_and_padded_white():
    """A 48 by 24 image, the widest aspect the filter keeps, comes back with
    its longer side at the stream's size, the rest padded white on both
    sides, and its content where it was: dark on the left, bright on the
    right. The cubic upscale rings at the edge between them, so the columns
    next to it are left out."""
    pixels = np.zeros((24, 48, 3), np.uint8)
    pixels[:, 24:] = 200

    image = online_loader.prepare_image(pixels, size=64, min_size=16)

    assert image is not None and image.shape == (64, 64, 3)
    assert (image[:16] == 255).all() and (image[48:] == 255).all()
    assert (image[16:48, :28] == 0).all() and (image[16:48, 36:] == 200).all()


# ---------------------------------------------------------------------------------
# A shard of rows
# ---------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------
# The real worker pool
# ---------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------
# The streaming extra
# ---------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------
# The streaming spec: what a run gets from OnlineImages
# ---------------------------------------------------------------------------------

def test_abandoned_full_image_queue_releases_real_spawned_workers(tmp_path):
    import multiprocessing

    for index in range(12):
        pixels = np.random.default_rng(index).integers(0, 256, (64, 64, 3), np.uint8)
        (tmp_path / f"{index}.jpg").write_bytes(_png(pixels))
    before = {child.pid for child in multiprocessing.active_children()}
    stream = UrlStream(_StubRows(12, passes=10000, root=str(tmp_path)),
                       batch=1, fetch=FETCH, workers=1, threads=1, prefetch=1)
    try:
        batch = next(stream)
        assert batch["image"].shape == (1, 64, 64, 3)
        deadline = time.monotonic() + 10
        while not stream.samples.full() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert stream.samples.full(), "the sample queue never reached its bound"
        assert {child.pid for child in multiprocessing.active_children()} - before
    finally:
        stream.close()
    assert not stream.fetcher.is_alive()
    assert not ({child.pid for child in multiprocessing.active_children()} - before)
    with pytest.raises(StopIteration):
        next(stream)


def _cancel_image_packet_probe(directory):
    """Run in a disposable process: pre-fix cancellation can wedge a C pipe read."""
    import multiprocessing
    from multiprocessing.connection import Connection

    stream = UrlStream(_StubRows(12, passes=10000, root=directory),
                       batch=1, fetch=Fetch(size=256, min_size=32, timeout=1, retries=0),
                       workers=1, threads=1, prefetch=1)
    header, resume = threading.Event(), threading.Event()
    failures = []
    original_recv = Connection._recv

    def recv(connection, size, *args):
        value = original_recv(connection, size, *args)
        if threading.current_thread() is consumer and size == 4 and not header.is_set():
            header.set()
            resume.wait()
        return value

    def consume():
        try:
            with contextlib.suppress(StopIteration):
                next(stream)
        except BaseException as error:
            failures.append(error)
        finally:
            stream.close()

    consumer = threading.Thread(target=consume)
    Connection._recv = recv
    consumer.start()
    try:
        assert header.wait(10), "the image packet header never arrived"
        stream.request_stop()
        stream.fetcher.join(0.2)
        assert stream.fetcher.is_alive(), "request_stop destroyed writers during receive"
        assert multiprocessing.active_children(), "the packet writer exited before receive"
    finally:
        resume.set()
    consumer.join(10)
    assert not consumer.is_alive(), "cancelled receive remained blocked in a partial payload"
    assert not failures, failures
    assert not stream.fetcher.is_alive()
    assert not multiprocessing.active_children()
    print("partial-packet cancellation: consumer, fetcher and spawned writer released", flush=True)


def test_cancellation_during_large_packet_does_not_terminate_its_writer(tmp_path):
    import signal
    import subprocess
    from pathlib import Path

    for index in range(12):
        pixels = np.random.default_rng(index).integers(0, 256, (256, 256, 3), np.uint8)
        (tmp_path / f"{index}.jpg").write_bytes(_png(pixels))
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join((str(Path(__file__).parent),
                                                str(Path(__file__).parents[1] / "src")))
    command = ("from test_online_loader import _cancel_image_packet_probe; "
               f"_cancel_image_packet_probe({str(tmp_path)!r})")
    process = subprocess.Popen([sys.executable, "-c", command], env=environment,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, start_new_session=True)
    try:
        output, _ = process.communicate(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        output, _ = process.communicate()
        pytest.fail(f"partial-packet cancellation did not terminate:\n{output}")
    assert process.returncode == 0, output
    print(output, end="")


# ---------------------------------------------------------------------------------
# Video clips by url, served over HTTP from tmp_path
# ---------------------------------------------------------------------------------
