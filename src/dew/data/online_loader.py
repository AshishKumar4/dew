"""Images fetched by url while a run trains, behind `OnlineImages`.

The rows are Hugging Face `datasets` tables of urls and captions. A pool of
worker processes walks them forever, a shard each and many threads per
worker, fetching and decoding every url and putting the finished samples on
one bounded queue. `ImageStream` takes `batch` samples off that queue at a
time. A fetch that fails, or an image not worth training on, is counted and
dropped; nothing is ever filled in for it.
"""

from __future__ import annotations

import dataclasses
import http.client
import io
import itertools
import multiprocessing
import multiprocessing.queues
import queue
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache, partial
from typing import TYPE_CHECKING, Mapping, Sequence

import cv2
import numpy as np
import PIL.Image

from .dataset import CAPTION, Batch
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
"""What a worker queues for a url that yielded an image: the pixels at the
stream's size and the row's caption. A dropped url is queued as the url."""

# The fetcher's worker processes are started without forking this one. This
# loader runs inside a training process, and os.fork carries over only the
# calling thread, so a child inherits mutexes that JAX's and CUDA's other
# threads were holding and hangs the first time it allocates or logs. The
# sample queue comes from the same context, because a semaphore created for
# the fork context is unlinked as soon as it exists and a spawned worker
# cannot reopen it.
_WORKER_CONTEXT = multiprocessing.get_context("spawn")


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
    """The agent HF `datasets` advertises, which its own url-fetching example
    sends, resolved on the first fetch so importing this module needs no
    `datasets`."""
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
        except ValueError:
            return None
        except (OSError, http.client.HTTPException):
            if attempt == retries:
                return None
            attempt += 1
            time.sleep(0.1 * attempt)


def decode_pixels(data: bytes) -> np.ndarray | None:
    """`data` as the array PIL decodes it to, or None when it is no image.

    The bytes are whatever the open internet returned, and PIL reports a bad
    file as OSError, SyntaxError, ValueError, its own DecompressionBombError
    or a struct.error depending on which header is broken, so this is the
    one place a broad except is the honest statement of what can happen.
    """
    try:
        return np.asarray(PIL.Image.open(io.BytesIO(data)))
    except Exception:
        return None


def prepare_image(pixels: np.ndarray, size: int, min_size: int) -> np.ndarray | None:
    """`pixels` fitted into a `size` square, or None when the image is not
    worth training on.

    Kept are RGB images at least `min_size` on their shorter side, at most
    `MAX_ASPECT` times as long as wide, and not a single flat colour. The
    longer side is resized to `size`, area interpolation down and cubic up,
    and the rest is padded to the square, centred, on white.
    """
    if pixels.ndim != 3 or pixels.shape[2] != 3:
        return None
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


@dataclasses.dataclass(frozen=True)
class Fetch:
    """How a worker turns one url into a sample."""

    size: int
    min_size: int
    timeout: float
    retries: int


def fetch_one(url: str, caption: str, sink: queue.Queue | multiprocessing.queues.Queue,
              fetch: Fetch) -> None:
    """Queue the sample for `url`, or the url itself when it yields nothing."""
    data = fetch_bytes(url, fetch.timeout, fetch.retries)
    pixels = None if data is None else decode_pixels(data)
    image = None if pixels is None else prepare_image(pixels, fetch.size, fetch.min_size)
    sink.put(url if image is None else (image, caption))


def columns(shard: Mapping[str, Sequence[str]]) -> tuple[Sequence[str], Sequence[str]]:
    """The url and caption columns of a shard of rows, by whichever of the
    known names it uses."""
    urls = next((shard[name] for name in URL_COLUMNS if name in shard), None)
    captions = next((shard[name] for name in CAPTION_COLUMNS if name in shard), None)
    if urls is None or captions is None:
        raise ValueError(
            f"a url table needs one of {URL_COLUMNS} and one of {CAPTION_COLUMNS}, "
            f"this one has {list(shard)}")
    return urls, captions


# The sample queue each pool worker inherited through the Pool initializer. A
# multiprocessing.Queue can only cross a process boundary while the process is
# being created, so handing it to pool.map as an argument raised "Queue objects
# should only be shared between processes through inheritance" before the pool
# had fetched anything.
_worker_sink: multiprocessing.queues.Queue | None = None


def _init_worker(sink: multiprocessing.queues.Queue) -> None:
    global _worker_sink
    _worker_sink = sink


def _fetch_shard(shard: Mapping[str, Sequence[str]], fetch: Fetch, threads: int) -> None:
    """Pool entry point: fetch every row of one shard onto this worker's queue."""
    if _worker_sink is None:
        raise RuntimeError("the fetch pool's worker was started without a queue")
    urls, captions = columns(shard)
    with ThreadPoolExecutor(max_workers=threads) as pool:
        # Reading the results is what raises a worker thread's exception
        # here; an unread executor.map swallows it.
        for _ in pool.map(partial(fetch_one, sink=_worker_sink, fetch=fetch), urls, captions):
            pass


def fetch_rows(rows: Dataset, sink: multiprocessing.queues.Queue, *, workers: int,
               threads: int, fetch: Fetch) -> None:
    """Walk `rows` forever, `workers` processes fetching a shard each with
    `threads` threads, and reshuffle between passes.

    Every row belongs to one shard: the bounds split len(rows) evenly, so a
    row past an even split is the last shard's tail rather than a row no
    pass ever fetches.
    """
    bounds = [index * len(rows) // workers for index in range(workers + 1)]
    with _WORKER_CONTEXT.Pool(workers, initializer=_init_worker, initargs=(sink,)) as pool:
        for iteration in itertools.count(1):
            pool.map(partial(_fetch_shard, fetch=fetch, threads=threads),
                     [rows[start:stop] for start, stop in zip(bounds, bounds[1:])])
            rows = rows.shuffle(seed=iteration)


class ImageStream:
    """Endless batches of fetched images, `{"image": uint8 [batch, size, size, 3],
    "caption": [batch] str}`.

    A batch is `batch` samples the fetchers really produced. A quiet queue is
    not a batch: while the fetcher lives the stream keeps waiting, and once
    it is gone iteration ends, or raises the fetcher's own exception if it
    died of one. `dropped` counts the urls the workers threw away. The
    fetchers run at most `prefetch` batches ahead. There is no position to
    report, so a run over this stream cannot checkpoint.
    """

    def __init__(self, rows: Dataset, *, batch: int, size: int, min_size: int,
                 workers: int, threads: int, timeout: float, retries: int,
                 prefetch: int, queue_timeout: float = 60.0):
        self.batch = batch
        self.queue_timeout = queue_timeout
        self.dropped = 0
        self.samples: multiprocessing.queues.Queue = _WORKER_CONTEXT.Queue(prefetch * batch)
        self._error: BaseException | None = None
        self._waiting_logged = False
        fetch = Fetch(size=size, min_size=min_size, timeout=timeout, retries=retries)

        # The fetcher's exception is kept rather than printed and forgotten:
        # __next__ re-raises it instead of waiting on a queue nobody fills.
        def produce() -> None:
            try:
                fetch_rows(rows, self.samples, workers=workers, threads=threads, fetch=fetch)
            except Exception as error:
                self._error = error

        self.fetcher = threading.Thread(target=produce, daemon=True)
        self.fetcher.start()

    def __iter__(self) -> ImageStream:
        return self

    def __next__(self) -> Batch:
        images: list[np.ndarray] = []
        captions: list[str] = []
        while len(images) < self.batch:
            try:
                item = self.samples.get(timeout=self.queue_timeout)
            except queue.Empty:
                self._check_fetcher()
                continue
            if isinstance(item, str):
                self.dropped += 1
                continue
            pixels, caption = item
            images.append(pixels)
            captions.append(caption)
        return {"image": np.stack(images), CAPTION: np.asarray(captions)}

    def _check_fetcher(self) -> None:
        """Raise when there is nothing left to wait for; a live fetcher is
        only slow, and its samples are worth more than zeros."""
        if self.fetcher.is_alive():
            if not self._waiting_logged:
                self._waiting_logged = True
                print(f"No sample in {self.queue_timeout}s, still fetching "
                      f"({self.dropped} dropped so far)")
            return
        if self._error is not None:
            raise RuntimeError("the url fetcher died") from self._error
        raise StopIteration
