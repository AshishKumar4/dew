"""Streaming loader tests: what reaches the model when fetching goes wrong.

Most of these replace the fetcher pool with a stub producer. One simulates a
missing HF `datasets` to check that importing the streaming stack does not
need the `streaming` extra. The tests that run the real pool cannot stub the
fetch, because a worker that was not forked holds no copy of this process's
memory, so they assert on what the workers report. Nothing here touches the
network: every url is a file:// url, of a file under tmp_path or of one that
does not exist.
"""

import functools
import io
import os
import queue
import sys
import threading
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
import PIL.Image
import pytest

from dew.data import Loading, online_loader
from dew.data.online_loader import Fetch, ImageStream
from dew.objectives.base import Aux, Objective
from dew.training import Checkpoints, Layout, MeshSpec, Trainer

BATCH = 4
# Comfortably longer than the stream's queue timeout below, so the slow
# producer really does make it wait.
SLOW = 0.15
# The pool tests spread 12 rows over 2 workers.
POOL_ROWS = 12
FETCH = Fetch(size=64, min_size=32, timeout=1, retries=0)


@pytest.fixture
def stop():
    """Keeps a stub producer alive for the test, like the endless real one."""
    event = threading.Event()
    yield event
    event.set()


def _sample(index, size=8):
    return np.full((size, size, 3), index + 1, np.uint8), f"sample {index}"


def _stream(monkeypatch, producer, queue_timeout=0.05, **settings):
    """An ImageStream fed by `producer` instead of the fetcher pool.

    The stream binds the producer from the module global as it is built, so
    patching the name keeps the real one, and its process pool, out of reach.
    """
    monkeypatch.setattr(online_loader, "fetch_rows", producer)
    return ImageStream(_StubRows(64), batch=BATCH, size=8, min_size=4, workers=1,
                       threads=1, timeout=1, retries=0, prefetch=4,
                       queue_timeout=queue_timeout, **settings)


class _StubRows:
    """As much of an HF dataset as the stream and the fetcher pool read.

    Column oriented like HF's, so a shard is a dict of lists. `passes` bounds
    the pool's otherwise endless loop, so a test's worker pool shuts down with
    it. The urls are files that do not exist, so a real fetch fails at once
    and without a network.
    """

    def __init__(self, size, passes=1, root="/nonexistent"):
        self.size = size
        self.passes = passes
        self.root = root

    def shard(self, num_shards, index):
        return self

    def shuffle(self, seed=0):
        if seed > self.passes:
            raise RuntimeError("stub rows ran out of passes")
        return self

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


def _refuse_to_fork(*args, **kwargs):
    raise AssertionError("the fetch pool forked a process that runs JAX")


# ---------------------------------------------------------------------------------
# Slow fetching waits; it never invents data
# ---------------------------------------------------------------------------------

def test_slow_fetching_waits_instead_of_fabricating_samples(monkeypatch, stop, capsys):
    """A queue timeout waits rather than handing back zeros captioned "Timeout
    occurred while waiting for sample", which would train as data."""
    def slow(rows, sink, **kwargs):
        for index in range(BATCH):
            stop.wait(SLOW)
            sink.put(_sample(index))
        stop.wait(10)

    batch = next(_stream(monkeypatch, slow))

    assert list(batch["caption"]) == [f"sample {i}" for i in range(BATCH)]
    assert [int(image.max()) for image in batch["image"]] == [1, 2, 3, 4]
    assert "still fetching" in capsys.readouterr().out


def test_dropped_fetches_are_counted_and_never_yielded(monkeypatch, stop):
    def with_drops(rows, sink, **kwargs):
        for index in range(BATCH):
            sink.put(f"file:///nonexistent/{index}.jpg")
            sink.put(_sample(index))
        stop.wait(10)

    stream = _stream(monkeypatch, with_drops)
    batch = next(stream)

    assert list(batch["caption"]) == [f"sample {i}" for i in range(BATCH)]
    assert stream.dropped == BATCH


def test_a_malformed_sample_raises_instead_of_training_as_zeros(monkeypatch, stop):
    """The old collate caught every exception and stacked zeros captioned
    "error image" in place of the batch."""
    def producer(rows, sink, **kwargs):
        for index in range(BATCH - 1):
            sink.put(_sample(index))
        sink.put(_sample(BATCH - 1, size=4))
        stop.wait(10)

    with pytest.raises(ValueError):
        next(_stream(monkeypatch, producer))


# ---------------------------------------------------------------------------------
# A fetcher that is gone ends iteration, and says why
# ---------------------------------------------------------------------------------

def test_an_exhausted_fetcher_stops_iteration(monkeypatch):
    def exhausted(rows, sink, **kwargs):
        sink.put(_sample(0))

    stream = _stream(monkeypatch, exhausted)

    with pytest.raises(StopIteration):
        next(stream)
    # and it stays stopped rather than blocking on the empty queue
    with pytest.raises(StopIteration):
        next(stream)


def test_a_fetcher_that_dies_raises_its_own_error(monkeypatch):
    def broken(rows, sink, **kwargs):
        raise RuntimeError("the fetch pool fell over")

    stream = _stream(monkeypatch, broken)

    with pytest.raises(RuntimeError, match="url fetcher died") as failure:
        next(stream)
    assert "fell over" in str(failure.value.__cause__)


# ---------------------------------------------------------------------------------
# One url: fetch, decode, fit
# ---------------------------------------------------------------------------------

def test_a_missing_file_and_a_malformed_url_fetch_nothing():
    assert online_loader.fetch_bytes("file:///nonexistent/gone.jpg", timeout=1, retries=1) is None
    assert online_loader.fetch_bytes("not a url", timeout=1, retries=1) is None


def test_bytes_that_are_no_image_decode_to_nothing():
    pixels = np.arange(48, dtype=np.uint8).reshape(4, 4, 3)
    assert np.array_equal(online_loader.decode_pixels(_png(pixels)), pixels)
    assert online_loader.decode_pixels(b"<html>not found</html>") is None
    assert online_loader.decode_pixels(_png(pixels)[:40]) is None


@pytest.mark.parametrize("pixels", [
    pytest.param(np.full((48, 48), 7, np.uint8), id="grayscale"),
    pytest.param(np.full((48, 48, 4), 7, np.uint8), id="rgba"),
    pytest.param(np.random.RandomState(0).randint(0, 256, (31, 48, 3), np.uint8), id="too_small"),
    pytest.param(np.random.RandomState(0).randint(0, 256, (32, 78, 3), np.uint8), id="too_wide"),
    pytest.param(np.full((48, 48, 3), 200, np.uint8), id="flat"),
])
def test_an_image_not_worth_training_on_is_dropped(pixels):
    assert online_loader.prepare_image(pixels, size=64, min_size=32) is None


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


def test_a_fetched_image_reaches_the_queue_as_a_sample(tmp_path, monkeypatch):
    """What a worker does with one url, run here rather than in a worker, which
    holds no copy of a patched fetch. Of that path only the request header
    comes from HF datasets, which the streaming extra owns."""
    monkeypatch.setattr(online_loader, "_user_agent", lambda: "dew-tests")
    path = tmp_path / "3.png"
    path.write_bytes(_png(np.random.RandomState(3).randint(0, 256, (48, 48, 3), np.uint8)))
    sink = queue.Queue()

    online_loader.fetch_one(path.as_uri(), "caption 3", sink, FETCH)
    online_loader.fetch_one((tmp_path / "gone.png").as_uri(), "caption 4", sink, FETCH)

    image, caption = sink.get_nowait()
    assert caption == "caption 3" and image.shape == (64, 64, 3) and int(image.max()) > 0
    assert sink.get_nowait() == (tmp_path / "gone.png").as_uri()


# ---------------------------------------------------------------------------------
# A shard of rows
# ---------------------------------------------------------------------------------

def test_a_shard_names_its_url_and_caption_columns_by_the_known_names():
    urls, captions = online_loader.columns({"URL": ["u"], "TEXT": ["t"], "other": [1]})
    assert (urls, captions) == (["u"], ["t"])
    with pytest.raises(ValueError, match="caption"):
        online_loader.columns({"url": ["u"]})
    with pytest.raises(ValueError, match="url"):
        online_loader.columns({"caption": ["t"]})


def test_a_worker_threads_exception_reaches_the_pool(monkeypatch, tmp_path):
    """Our own code failing on one url is a bug, not a drop: the executor
    swallows an unread map, so the shard has to read its results."""
    def broken(pixels, size, min_size):
        raise TypeError("a bug in the image processor")

    path = tmp_path / "0.png"
    path.write_bytes(_png(np.random.RandomState(0).randint(0, 256, (48, 48, 3), np.uint8)))
    monkeypatch.setattr(online_loader, "_user_agent", lambda: "dew-tests")
    monkeypatch.setattr(online_loader, "prepare_image", broken)
    monkeypatch.setattr(online_loader, "_worker_sink", queue.Queue())

    with pytest.raises(TypeError, match="a bug in the image processor"):
        online_loader._fetch_shard({"url": [path.as_uri()], "caption": ["c"]}, FETCH, threads=2)


# ---------------------------------------------------------------------------------
# The real worker pool
# ---------------------------------------------------------------------------------

def _drain(sink, expected, producer_failure, deadline):
    reported = []
    while len(reported) < expected and time.monotonic() < deadline:
        try:
            reported.append(sink.get(timeout=0.5))
        except queue.Empty:
            # Nothing queued and the producer gone means no worker ever ran.
            assert not producer_failure, producer_failure[0]
    return reported


def _produce_in_a_thread(rows, sink):
    producer_failure = []

    def produce():
        try:
            online_loader.fetch_rows(rows, sink, workers=2, threads=2, fetch=FETCH)
        except BaseException as error:
            producer_failure.append(error)

    threading.Thread(target=produce, daemon=True).start()
    return producer_failure


def test_the_fetch_pool_starts_its_workers_without_forking(monkeypatch):
    """os.fork carries over only the calling thread, so a worker forked out of
    a training process inherits mutexes that JAX's other threads were holding,
    and hangs the first time it allocates. With os.fork refused the pool still
    has to start its workers and hear back from them."""
    monkeypatch.setattr(os, "fork", _refuse_to_fork)
    sink = online_loader._WORKER_CONTEXT.Queue(64)
    failure = _produce_in_a_thread(_StubRows(POOL_ROWS), sink)

    reported = _drain(sink, POOL_ROWS, failure, time.monotonic() + 60)

    # Every url is a file that does not exist, so every fetch fails, and the
    # url a worker queues for it proves the worker ran and its inherited
    # queue crossed over.
    assert sorted(reported) == sorted(f"file:///nonexistent/{i}.jpg" for i in range(POOL_ROWS))


def test_the_fetch_pool_covers_the_rows_past_an_even_split(monkeypatch):
    """Thirteen rows over two workers: an even split holds twelve, so the
    thirteenth row needs a shard of its own. A row no shard holds is a row no
    run ever trains on, however long the stream runs. The drops carry their
    urls, so two passes have to show every row twice."""
    monkeypatch.setattr(os, "fork", _refuse_to_fork)
    rows = POOL_ROWS + 1
    sink = online_loader._WORKER_CONTEXT.Queue(64)
    failure = _produce_in_a_thread(_StubRows(rows, passes=2), sink)

    reported = _drain(sink, 2 * rows, failure, time.monotonic() + 120)

    assert sorted(reported) == sorted(2 * [f"file:///nonexistent/{i}.jpg" for i in range(rows)])


# ---------------------------------------------------------------------------------
# The streaming extra
# ---------------------------------------------------------------------------------

def test_loading_rows_asks_for_the_streaming_extra(monkeypatch):
    """Importing this module must work without HF datasets; only actually
    loading a dataset needs it, and it has to say so.

    The absence is simulated so the test states the behaviour whether or not
    the extra is installed in the environment that runs it.
    """
    monkeypatch.setitem(sys.modules, "datasets", None)
    with pytest.raises(ImportError, match=r"dew-ml\[streaming\]"):
        online_loader.load_rows(["some/hf/dataset"])


# ---------------------------------------------------------------------------------
# The streaming spec: what a run gets from OnlineImages
# ---------------------------------------------------------------------------------

def _producer_of(rows, passes, stop):
    """A fetcher that walks `rows` records `passes` times, then stays alive."""

    def produce(table, sink, **kwargs):
        for _ in range(passes):
            for index in range(rows):
                sink.put(_sample(index, size=4))
        stop.wait(5)

    return produce


def _online_spec(monkeypatch, rows, passes, stop):
    """OnlineImages over a stub table, with the fetcher pool replaced."""
    from dew.data import OnlineImages

    monkeypatch.setattr(online_loader, "fetch_rows", _producer_of(rows, passes, stop))
    monkeypatch.setattr(online_loader, "load_rows", lambda sources: _StubRows(rows))
    return OnlineImages(sources=("fake_online",), image_size=4,
                        loading=Loading(workers=1))


def test_the_streaming_spec_repeats_its_records_instead_of_ending(monkeypatch, stop):
    """The streamer is documented as endless: the pool walks its shards in a
    loop and a run bounds training by steps, so `for batch in loader` must not
    stop at the end of the dataset."""
    rows, batch, passes = 12, 4, 3
    loader = _online_spec(monkeypatch, rows, passes, stop).load(batch=batch).train()

    seen = [[int(v) for v in next(loader)["image"][:, 0, 0, 0]]
            for _ in range(rows // batch * passes)]
    records = [value for row in seen for value in row]

    assert len(records) == rows * passes
    assert sorted(records[:rows]) == list(range(1, rows + 1)), "a pass repeats a record"
    assert set(records[rows:]) == set(records[:rows]), "the stream reads them again"


def test_the_streaming_spec_reports_its_records_and_holds_nothing_out(monkeypatch, stop):
    """`records` counts rows, like every grain spec's, so a recipe divides it
    by the batch for steps per epoch; a streaming run holds nothing out, so
    nothing downstream may report a validation pass over a held-out split."""
    data = _online_spec(monkeypatch, 12, 1, stop).load(batch=4)

    assert data.records == 12 and data.batch == 4 and data.steps_per_epoch == 3
    assert data.val is None
    assert len(next(data.train())["image"]) == 4


def test_the_streaming_spec_opens_nothing_before_the_stream_is_asked_for(monkeypatch, stop):
    """Loading resolves the rows; the fetcher pool and its thread start with
    `train()`, and the stream cannot record a position."""
    started = []
    producer = _producer_of(4, 1, stop)

    def produce(rows, sink, **kwargs):
        started.append(rows)
        producer(rows, sink, **kwargs)

    data = _online_spec(monkeypatch, 4, 1, stop).load(batch=4)
    monkeypatch.setattr(online_loader, "fetch_rows", produce)
    assert started == []

    stream = data.train()
    assert len(next(stream)["image"]) == 4
    assert len(started) == 1
    assert not hasattr(stream, "get_state")


def test_the_streaming_spec_stops_when_its_fetcher_is_gone(monkeypatch):
    """The one thing that does end the stream: nothing left to wait for.

    The spec leaves the stream's queue timeout at a minute, so the wait
    before it looks at the fetcher is shortened here. What is under test is
    that the wait ends in StopIteration rather than in a batch of zeros.
    """
    def exhausted(rows, sink, **kwargs):
        for index in range(4):
            sink.put(_sample(index, size=4))

    from dew.data import OnlineImages

    monkeypatch.setattr(online_loader, "fetch_rows", exhausted)
    monkeypatch.setattr(online_loader, "ImageStream",
                        functools.partial(ImageStream, queue_timeout=0.05))
    monkeypatch.setattr(online_loader, "load_rows", lambda sources: _StubRows(4))
    loader = OnlineImages(sources=("fake_online",), image_size=4,
                          loading=Loading(workers=1)).load(batch=4).train()

    assert len(next(loader)["image"]) == 4
    with pytest.raises(StopIteration):
        next(loader)


class Mean(Objective):
    """One scalar fitted to the batch mean: the smallest objective that reads a
    batch, so what is under test is the streaming data path and nothing else."""

    ema = None

    def init(self, key):
        return {"params": {"level": jnp.zeros(())}}

    def loss(self, params, batch, step):
        pixels = (jnp.asarray(batch["image"], jnp.float32) - 127.5) / 127.5
        return jnp.mean((pixels - params["params"]["level"]) ** 2), Aux({})


def _run(data, *, steps, checkpoints=None, checkpoint_every=None):
    trainer = Trainer(Mean(), optax.sgd(0.1), key=jax.random.key(0),
                      mesh=MeshSpec(), layout=Layout(), checkpoints=checkpoints)
    return trainer.fit(data, steps=steps, log_every=steps, eval_every=None,
                       checkpoint_every=checkpoint_every)


def test_the_streaming_spec_cannot_report_a_position(monkeypatch, stop):
    """The iterator is the single answer to whether a run can checkpoint: the
    fetch stream carries no get_state, and `tokenized` does not invent one."""
    data = _online_spec(monkeypatch, 12, 1, stop).load(batch=4)

    stream = data.train()
    assert not hasattr(stream, "get_state") and not hasattr(stream, "set_state")


def test_a_streaming_run_trains_when_it_never_checkpoints(monkeypatch, stop):
    """The whole point of the gate: an online stream is trainable, six steps of
    it, as long as the run does not ask for a position it cannot have."""
    data = _online_spec(monkeypatch, 24, 6, stop).load(batch=8)

    state = _run(data, steps=6)

    assert int(state.step) == 6
    assert jnp.isfinite(state.params["params"]["level"])


def test_a_streaming_run_that_asks_for_checkpoints_is_refused(monkeypatch, stop, tmp_path):
    """And the other half: the refusal is by name, before any training."""
    data = _online_spec(monkeypatch, 24, 6, stop).load(batch=8)

    with pytest.raises(ValueError, match=r"checkpoint_every=None"):
        _run(data, steps=6, checkpoints=Checkpoints(str(tmp_path / "ckpt")),
             checkpoint_every=2)
