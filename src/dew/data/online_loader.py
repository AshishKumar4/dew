"""Images and video clips fetched by url while a run trains, behind
`OnlineImages` and `OnlineVideos`.

The rows are Hugging Face `datasets` tables of urls and captions. A pool of
worker processes walks them forever, a shard each and many threads per
worker, fetching and decoding every url and putting the finished samples on
one bounded queue. `UrlStream` takes `batch` samples off that queue at a
time. A fetch that fails, or a sample not worth training on, is counted and
dropped; nothing is ever filled in for it.
"""

from __future__ import annotations

import dataclasses
import http.client
import itertools
import logging
import multiprocessing
import multiprocessing.queues
import multiprocessing.synchronize
import queue
import tempfile
import threading
import time
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache, partial
from typing import TYPE_CHECKING

import cv2
import numpy as np

from .dataset import CAPTION, Batch
from .images import decode_image
from .sources.hf import _STREAMING_HINT, _hf_datasets

if TYPE_CHECKING:
    from datasets import Dataset

# The columns the url tables name their two fields by: LAION and the
# MS-COCO url table write URL and TEXT, COYO url and text, CC12M url and
# caption, and the bucket's own tables keep whichever they were saved with.
URL_COLUMNS = ("url", "URL", "image_url")
CAPTION_COLUMNS = ("caption", "CAPTION", "text", "TEXT", "txt")

# An image whose longer side is more than this many times its shorter one
# is a banner or a strip, not a picture to train on.
MAX_ASPECT = 2.4

Sample = tuple[np.ndarray, str]
"""The pixels at the stream's size, an image or a clip of frames, and the
row's caption, queued for a url that yielded one. A dropped url is queued as
the url."""

# The fetcher's worker processes are started without forking this one. This
# loader runs inside a training process, and os.fork carries over only the
# calling thread, so a child inherits mutexes that JAX's and CUDA's other
# threads were holding and hangs the first time it allocates or logs. The
# sample queue comes from the same context, because a semaphore created for
# the fork context is unlinked as soon as it exists and a spawned worker
# cannot reopen it.
_WORKER_CONTEXT = multiprocessing.get_context("spawn")


_log = logging.getLogger(__name__)


def load_rows(sources: Sequence[str]) -> Dataset:
    """The rows of `sources`, concatenated and shuffled once.

    A `gs://` path is a dataset saved with `save_to_disk`; anything else is
    a hub dataset name, read at its train split.
    """
    hf = _hf_datasets()
    loaded: list[Dataset] = []
    for source in sources:
        # load_from_disk hands back a DatasetDict for a directory of splits,
        # and concatenate takes tables; a split is what a row source is.
        table = (hf.load_from_disk(source) if source.startswith("gs://")
                 else hf.load_dataset(source, split="train"))
        loaded.append(table["train"] if isinstance(table, hf.DatasetDict) else table)
    if len(loaded) == 1:
        return loaded[0]
    return hf.concatenate_datasets(loaded).shuffle(seed=0)


@lru_cache(maxsize=1)
def _user_agent() -> str:
    """The user agent HF `datasets` advertises, which its own url-fetching
    example sends.

    Resolved on the first fetch, so importing this module needs no
    `datasets`.
    """
    try:
        from datasets.utils.file_utils import get_datasets_user_agent
    except ImportError as exc:
        raise ImportError(_STREAMING_HINT) from exc
    return get_datasets_user_agent()


def fetch_bytes(url: str, timeout: float, retries: int) -> bytes | None:
    """The body at `url`, or None once `retries` more attempts have failed.

    A refused connection, a timeout, an HTTP error and a truncated response
    are all OSError or HTTPException, and a second attempt often mends them.
    A malformed url raises ValueError from the request itself and no retry
    mends that.
    """
    attempt = 0
    while True:
        try:
            request = urllib.request.Request(url, headers={"user-agent": _user_agent()})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except ValueError as error:
            _log.debug("skipping %r: %s", url, error)
            return None
        except (OSError, http.client.HTTPException):
            if attempt == retries:
                return None
            attempt += 1
            time.sleep(0.1 * attempt)


def decode_pixels(blob: bytes) -> np.ndarray | None:
    """`blob` as RGB uint8 by `images.decode_image`, or None when it is no image.

    The bytes are whatever the open internet returned, and `decode_image`
    raises every way they can fail, a decompression bomb included, as
    ValueError.
    """
    try:
        return decode_image(blob)
    except ValueError as error:
        _log.debug("undecodable image of %d bytes: %s", len(blob), error)
        return None


def prepare_image(pixels: np.ndarray, size: int, min_size: int) -> np.ndarray | None:
    """`pixels` fitted into a `size` square, or None when the image is not
    worth training on.

    Kept are images at least `min_size` on their shorter side, at most
    `MAX_ASPECT` times as long as wide, and not a single flat colour. The
    longer side is resized to `size`, area interpolation down and cubic up,
    and the rest is padded to the square, centred, on white.
    """
    height, width = pixels.shape[:2]
    longer, shorter = max(height, width), min(height, width)
    if shorter < min_size or longer > MAX_ASPECT * shorter:
        return None
    if pixels.min() == pixels.max():
        return None
    scale = size / longer
    resized = cv2.resize(
        pixels, (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_AREA if longer > size else cv2.INTER_CUBIC)
    pad_height, pad_width = size - resized.shape[0], size - resized.shape[1]
    if not pad_height and not pad_width:
        return resized
    top, left = pad_height // 2, pad_width // 2
    return cv2.copyMakeBorder(resized, top, pad_height - top, left, pad_width - left,
                              cv2.BORDER_CONSTANT, value=(255, 255, 255))


def prepare_clip(blob: bytes, frames: int, size: int, min_size: int) -> np.ndarray | None:
    """`frames` consecutive frames of the video in `blob` from a random start,
    each resized to a `size` square, or None when the bytes are no video that
    long or its frames are under `min_size` on their shorter side.

    The frames are read at 25 fps and resized the way `VideoDataset` reads a
    local clip. The decoder reads a file, so the bytes are written to one
    under the temporary directory for the length of the read.
    """
    from .sources.av_utils import read_random_frames
    from .video import fit_frames

    with tempfile.NamedTemporaryFile() as file:
        file.write(blob)
        file.flush()
        try:
            clip, _ = read_random_frames(file.name, num_frames=frames, padding=0,
                                         seed=int(np.random.default_rng().integers(0, 2**32 - 1)))
        except (OSError, ValueError) as error:
            _log.debug("undecodable video of %d bytes: %s", len(blob), error)
            return None
    if min(clip.shape[1:3]) < min_size:
        return None
    return fit_frames(clip, size)


@dataclasses.dataclass(frozen=True)
class Fetch:
    """Says how a worker turns one url into a sample."""

    size: int
    min_size: int
    timeout: float
    retries: int
    frames: int | None = None
    """None fetches an image; a count fetches a clip of that many frames."""


def fetch_one(url: str, caption: str, sink: queue.Queue | multiprocessing.queues.Queue,
              fetch: Fetch, stop: multiprocessing.synchronize.Event | None = None) -> None:
    """Queues the sample for `url`, or the url itself when it yields nothing."""
    if stop is not None and stop.is_set():
        return
    blob = fetch_bytes(url, fetch.timeout, fetch.retries)
    if blob is None:
        pixels = None
    elif fetch.frames is None:
        image = decode_pixels(blob)
        pixels = None if image is None else prepare_image(image, fetch.size, fetch.min_size)
    else:
        pixels = prepare_clip(blob, fetch.frames, fetch.size, fetch.min_size)
    sample = url if pixels is None else (pixels, caption)
    while stop is None or not stop.is_set():
        try:
            sink.put(sample, timeout=0.05)
            return
        except queue.Full:
            continue


def columns(shard: Mapping[str, Sequence[str]]) -> tuple[Sequence[str], Sequence[str]]:
    """The url and caption columns of `shard`, under whichever of the known
    names it uses."""
    urls = next((shard[name] for name in URL_COLUMNS if name in shard), None)
    captions = next((shard[name] for name in CAPTION_COLUMNS if name in shard), None)
    if urls is None or captions is None:
        raise ValueError(
            f"a url table needs one of {URL_COLUMNS} and one of {CAPTION_COLUMNS}, "
            f"this one has {list(shard)}")
    return urls, captions


# The sample queue each pool worker inherited through the Pool initializer. A
# multiprocessing.Queue can only cross a process boundary while the process is
# being created, so handing it to pool.map as an argument raises "Queue objects
# should only be shared between processes through inheritance".
_worker_sink: multiprocessing.queues.Queue | None = None
_worker_stop: multiprocessing.synchronize.Event | None = None


def _init_worker(sink: multiprocessing.queues.Queue,
                 stop: multiprocessing.synchronize.Event) -> None:
    global _worker_sink, _worker_stop
    _worker_sink, _worker_stop = sink, stop
    # Cancellation may abandon a full queue; never flush it at process exit.
    sink.cancel_join_thread()


def _fetch_shard(shard: Mapping[str, Sequence[str]], fetch: Fetch, threads: int) -> None:
    """Fetches every row of one shard onto this worker's queue."""
    if _worker_sink is None:
        raise RuntimeError("the fetch pool's worker was started without a queue")
    urls, captions = columns(shard)
    with ThreadPoolExecutor(max_workers=threads) as pool:
        # Reading the results raises a worker thread's exception here. An
        # unread executor.map would swallow it.
        for _ in pool.map(
            partial(fetch_one, sink=_worker_sink, fetch=fetch, stop=_worker_stop), urls, captions
        ):
            pass


def fetch_rows(rows: Dataset, sink: multiprocessing.queues.Queue, *, workers: int,
               threads: int, fetch: Fetch, stop: multiprocessing.synchronize.Event,
               shutdown: threading.Event, finish: Callable[[BaseException | None], None]) -> None:
    """Walks `rows` forever, `workers` processes fetching a shard each with
    `threads` threads, reshuffling between passes.

    Every row belongs to one shard. The bounds split len(rows) evenly, so a
    row past an even split is the last shard's tail.
    """
    bounds = [index * len(rows) // workers for index in range(workers + 1)]
    with _WORKER_CONTEXT.Pool(workers, initializer=_init_worker, initargs=(sink, stop)) as pool:
        error = None
        try:
            for iteration in itertools.count(1):
                if stop.is_set():
                    return
                pending = pool.map_async(partial(_fetch_shard, fetch=fetch, threads=threads),
                                        [rows[start:end] for start, end in itertools.pairwise(bounds)])
                while not stop.is_set():
                    try:
                        pending.get(timeout=0.05)
                        break
                    except multiprocessing.TimeoutError:
                        continue
                if stop.is_set():
                    return
                rows = rows.shuffle(seed=iteration)
        except BaseException as failure:
            error = failure
            raise
        finally:
            finish(error)
            # Queue.get's timeout covers readiness, not a partial packet.
            # Keep writers alive until the iteration owner has left next.
            # Only close, never request_stop, permits the teardown.
            shutdown.wait()
            # close set `stop`, so each worker ends its shard within a fetch's
            # timeout and exits on the pool's sentinel. Pool.terminate, which
            # leaving the `with` would run, kills workers first and then joins
            # its task handler: a worker killed while it held the result
            # queue's lock, sending its shard's result, left that handler
            # waiting on the lock for good (a 10-minute hang in
            # test_abandoned_full_image_queue_releases_real_spawned_workers).
            pool.close()
            pool.join()


class UrlStream:
    """Yields endless batches of fetched images,
    `{"image": uint8 [batch, size, size, 3], "caption": [batch] str}`, or of
    clips, `{"video": uint8 [batch, frames, size, size, 3], ...}`, when
    `fetch` names a frame count.

    A batch is `batch` samples the fetchers really produced, and a quiet
    queue is no batch. While production runs the stream keeps waiting. When
    it finishes, iteration ends or raises its exception after queued samples.
    The fetcher retains its pool until iteration-owning close, so
    cancellation cannot destroy a writer halfway through an active queue
    receive.

    `dropped` counts the urls the workers threw away, and the fetchers run at
    most `prefetch` batches ahead. The stream reports no position, so a run
    over it does not checkpoint.
    """

    def __init__(self, rows: Dataset, *, batch: int, fetch: Fetch, workers: int, threads: int,
                 prefetch: int, queue_timeout: float = 60.0):
        self.batch = batch
        self.field = "image" if fetch.frames is None else "video"
        self.queue_timeout = queue_timeout
        self.dropped = 0
        self.samples: multiprocessing.queues.Queue | None = _WORKER_CONTEXT.Queue(prefetch * batch)
        self._stop = _WORKER_CONTEXT.Event()
        self._shutdown = threading.Event()
        self._done = threading.Event()
        self._closed = False
        self._error: BaseException | None = None
        self._waiting_logged = False

        # The fetcher's exception is kept for `__next__` to re-raise.
        def produce() -> None:
            samples = self.samples
            assert samples is not None
            try:
                fetch_rows(rows, samples, workers=workers, threads=threads,
                           fetch=fetch, stop=self._stop, shutdown=self._shutdown,
                           finish=self._finish)
            except BaseException as error:
                self._finish(error)
            finally:
                self._done.set()

        self.fetcher = threading.Thread(target=produce, daemon=True)
        self.fetcher.start()

    def __iter__(self) -> UrlStream:
        return self

    def __next__(self) -> Batch:
        samples = self.samples
        if samples is None:
            raise StopIteration
        pixels: list[np.ndarray] = []
        captions: list[str] = []
        waiting_since = time.monotonic()
        while len(pixels) < self.batch:
            if self._stop.is_set():
                raise StopIteration
            try:
                sample = samples.get(timeout=min(0.05, self.queue_timeout))
            except queue.Empty:
                if (self._done.is_set()
                        or time.monotonic() - waiting_since >= self.queue_timeout):
                    self._check_fetcher()
                continue
            if isinstance(sample, str):
                self.dropped += 1
                continue
            media, caption = sample
            pixels.append(media)
            captions.append(caption)
        if self._stop.is_set():
            raise StopIteration
        return {self.field: np.stack(pixels), CAPTION: np.asarray(captions)}

    def _finish(self, error: BaseException | None) -> None:
        self._error = error
        self._done.set()

    def request_stop(self) -> None:
        """Signals cancellation; safe alongside next and final close."""
        self._stop.set()

    def close(self) -> None:
        """Finalizes on the iteration-owning thread, after next has returned."""
        self.request_stop()
        if self._closed:
            return
        self._shutdown.set()
        self.fetcher.join()
        # A worker's exit can abandon its last queue write. Do not drain/reuse
        # their pipe, or wait for abandoned payloads to flush.
        samples = self.samples
        if samples is not None:
            samples.cancel_join_thread()
            samples.close()
            # A consumer-only multiprocessing.Queue has no feeder finalizer;
            # close alone leaves its pipe descriptors held by the object.
            self.samples = None
        self._error = None
        self._closed = True

    def _check_fetcher(self) -> None:
        """Raises when there is nothing left to wait for.

        A live fetcher is only slow, so it earns one warning instead.
        """
        if not self._done.is_set():
            if not self._waiting_logged:
                self._waiting_logged = True
                # A stalled loader owes one warning to the person watching it;
                # test_slow_fetching_waits_instead_of_fabricating_samples checks
                # the notice alongside the real records, never a fabricated batch.
                _log.warning("No sample in %ss, still fetching (%s dropped so far)",
                             self.queue_timeout, self.dropped)
            return
        if self._error is not None:
            raise RuntimeError("the url fetcher died") from self._error
        raise StopIteration
