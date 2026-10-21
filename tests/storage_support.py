"""An S3 endpoint of a test's own: a bucket on a local moto server, and the storage options that reach
it."""

import contextlib
from collections.abc import Iterator
from uuid import uuid4

import fsspec


@contextlib.contextmanager
def moto_bucket(prefix: str, **extra: object) -> Iterator[tuple[str, dict[str, object]]]:
    """A fresh bucket named from `prefix` on a local moto S3 server, and the storage options that
    reach it with `extra` among them; the server stops when the block ends."""
    from moto.server import ThreadedMotoServer

    server = ThreadedMotoServer(ip_address="127.0.0.1", port=0, verbose=False)
    server.start()
    try:
        _, port = server.get_host_and_port()
        options: dict[str, object] = {"key": "testing", "secret": "testing",
                                      "endpoint_url": f"http://127.0.0.1:{port}", **extra}
        bucket = f"{prefix}-{uuid4().hex}"
        fsspec.filesystem("s3", **options).mkdir(bucket)
        yield bucket, options
    finally:
        server.stop()

