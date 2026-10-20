"""Conditional publication, transport bounds and URL credentials."""

import hashlib
import json
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from uuid import uuid4

import fsspec
import pytest
from fsspec.implementations.local import LocalFileSystem

from dew.data.sources import Files
from dew.data.storage import BoundedFileCache, PreconditionFailed, Store, Versioned, _profile


@pytest.fixture(params=["memory", "file", "s3"])
def store(request, tmp_path):
    if request.param == "memory":
        return Store(f"memory://storage-{uuid4().hex}/value")
    if request.param == "file":
        return Store(str(tmp_path / "value"))
    from moto.server import ThreadedMotoServer

    server = ThreadedMotoServer(ip_address="127.0.0.1", port=0, verbose=False)
    server.start()
    _, port = server.get_host_and_port()
    options = {"key": "testing", "secret": "testing", "endpoint_url": f"http://127.0.0.1:{port}"}
    fs = fsspec.filesystem("s3", **options)
    bucket = "storage-test-" + uuid4().hex
    fs.mkdir(bucket)
    request.addfinalizer(server.stop)
    return Store(f"s3://{bucket}/value", options)


def test_a_versioned_s3_read_is_the_filesystems_own_request():
    """An S3 store's versioned read sends what its filesystem's options add to every request: an
    object written under a customer-provided key (SSE-C) is read with that key, and an object that
    is absent raises FileNotFoundError."""
    import base64
    import hashlib
    import os

    from moto.server import ThreadedMotoServer

    server = ThreadedMotoServer(ip_address="127.0.0.1", port=0, verbose=False)
    server.start()
    try:
        _, port = server.get_host_and_port()
        secret = os.urandom(32)
        sealed = {"SSECustomerAlgorithm": "AES256", "SSECustomerKey": base64.b64encode(secret).decode(),
                  "SSECustomerKeyMD5": base64.b64encode(hashlib.md5(secret).digest()).decode()}
        options = {"key": "testing", "secret": "testing", "endpoint_url": f"http://127.0.0.1:{port}",
                   "s3_additional_kwargs": sealed}
        bucket = "sealed-" + uuid4().hex
        fs = fsspec.filesystem("s3", **options)
        fs.mkdir(bucket)
        fs.pipe(f"{bucket}/value", b"payload")
        # The read's own parameters, as botocore builds the request from them.
        sent = []
        fs.s3.meta.events.register("before-parameter-build.s3.GetObject",
                                   lambda params, **kwargs: sent.append(dict(params)))
        assert Store(f"s3://{bucket}/value", options).get_versioned().data == b"payload"
        assert sent and all(sent[-1].get(name) == value for name, value in sealed.items())
        with pytest.raises(FileNotFoundError):
            Store(f"s3://{bucket}/absent", options).get_versioned()
    finally:
        server.stop()


def test_compare_and_swap_rejects_stale_and_duplicate_publication(store):
    first = store.put_if_absent(b"first")
    assert store.get_versioned().data == b"first"
    assert store.get_versioned().version == first
    with pytest.raises(PreconditionFailed):
        store.put_if_absent(b"other")
    second = store.put_if_match(b"second", first)
    with pytest.raises(PreconditionFailed):
        store.put_if_match(b"stale", first)
    assert second != first
    assert store.get_versioned().data == b"second"


def test_an_immutable_put_accepts_its_own_retry_and_refuses_other_bytes(store):
    version = store.put_immutable(b"payload")
    assert store.put_immutable(b"payload") == version == store.get_versioned().version
    with pytest.raises(ValueError):
        store.put_immutable(b"different")
    assert store.read() == b"payload"


def test_a_filesystem_of_its_own_supplies_conditional_writes():
    """A third-party filesystem plugs into Store through put_conditional and
    imports nothing of Dew: a plain FileExistsError is a failed precondition,
    and its ukey names the versions put_if_match compares."""
    from fsspec.implementations.memory import MemoryFileSystem

    class Conditional(MemoryFileSystem):
        protocol = "conditional"

        @classmethod
        def _strip_protocol(cls, path):
            return super()._strip_protocol(path.replace("conditional://", "memory://", 1))

        def ukey(self, path):
            return hashlib.sha256(self.cat_file(path)).hexdigest()

        def put_conditional(self, path, payload, version):
            if (self.ukey(path) if self.exists(path) else None) != version:
                raise FileExistsError(path)
            self.pipe_file(path, payload)
            return self.ukey(path)

    fsspec.register_implementation("conditional", Conditional, clobber=True)
    store = Store(f"conditional://storage-{uuid4().hex}/value")
    first = store.put_if_absent(b"first")
    assert store.get_versioned() == Versioned(b"first", first)
    with pytest.raises(PreconditionFailed):
        store.put_if_absent(b"other")
    second = store.put_if_match(b"second", first)
    with pytest.raises(PreconditionFailed):
        store.put_if_match(b"stale", first)
    assert store.get_versioned() == Versioned(b"second", second)
    assert store.put_immutable(b"second") == second


def test_two_finalizers_publish_only_one_snapshot(store):
    start = threading.Barrier(2)

    def publish(data):
        start.wait()
        try:
            store.put_if_absent(data)
            return data
        except PreconditionFailed as error:
            return error

    with ThreadPoolExecutor(2) as pool:
        results = list(pool.map(publish, [b"a", b"b"]))
    assert sum(isinstance(result, PreconditionFailed) for result in results) == 1
    assert store.read() == next(result for result in results if isinstance(result, bytes))


def test_named_profile_resolves_secrets_lazily(monkeypatch):
    _profile("credentialtest", "memory", {"credential": "env:DEW_TEST_MISSING_SECRET"})
    monkeypatch.delenv("DEW_TEST_MISSING_SECRET", raising=False)
    with pytest.raises(ValueError):
        Store("credentialtest://bucket/key").resolve()
    monkeypatch.setenv("DEW_TEST_MISSING_SECRET", "secret-value")
    fs, path = Store("credentialtest://bucket/key").resolve()
    fs.pipe_file(path, b"present")
    assert fs.cat_file(path) == b"present"


def test_local_content_versions_are_shared_and_refreshed_after_writes(tmp_path, monkeypatch):
    """Independent readers share one complete checksum per local file version."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    path = tmp_path / "corpus.txt"
    payload = b"a" * (192 * 1024)
    path.write_bytes(payload)
    opened = []
    open_file = LocalFileSystem.open

    def count(self, filename, *args, **kwargs):
        if Path(filename).resolve() == path:
            opened.append(filename)
        return open_file(self, filename, *args, **kwargs)

    monkeypatch.setattr(LocalFileSystem, "open", count)

    def version(filename):
        return Files(str(filename), format="text").fingerprint()["files"][0]["version"]

    with ThreadPoolExecutor(4) as readers:
        versions = list(readers.map(version, [path] * 4))
    assert versions == [hashlib.sha256(payload).hexdigest()] * 4
    alias = tmp_path / "alias.txt"
    alias.symlink_to(path)
    assert version(alias) == versions[0]
    assert len(opened) == 1
    reopened = subprocess.run([sys.executable, "-c", """
import os
import sys
from fsspec.implementations.local import LocalFileSystem
from dew.data.storage import content_version

class CachedOnly(LocalFileSystem):
    def _open(self, *args, **kwargs):
        raise RuntimeError("a persisted content version reread its source")

filename = sys.argv[1]
print(content_version(CachedOnly(), filename, os.stat(filename).st_size))
""", str(path)], capture_output=True, text=True, check=True)
    assert reopened.stdout.strip() == versions[0]
    with path.open("r+b") as file:
        file.seek(len(payload) // 2)
        file.write(b"b")
    changed = version(path)
    assert changed != versions[0]
    assert len(opened) == 2
    metadata = path.stat()
    replacement = tmp_path / "replacement.txt"
    replacement.write_bytes(b"c" * len(payload))
    os.utime(replacement, ns=(metadata.st_atime_ns, metadata.st_mtime_ns))
    replacement.replace(path)
    assert version(path) != changed
    assert len(opened) == 3


def test_cache_keeps_active_file_pinned_and_evicts_lru(tmp_path):
    remote = fsspec.filesystem("memory")
    root = f"cache-{uuid4().hex}"
    for name in ("a", "b", "c"):
        remote.pipe_file(f"{root}/{name}", name.encode() * 4)
    cache = BoundedFileCache(fs=remote, cache_storage=str(tmp_path), max_bytes=8)
    active = cache.open(f"{root}/a", "rb")
    with cache.open(f"{root}/b", "rb") as file:
        assert file.read() == b"bbbb"
    with cache.open(f"{root}/c", "rb") as file:
        assert file.read() == b"cccc"
    assert active.read() == b"aaaa"
    assert cache._check_file(f"{root}/a")
    assert not cache._check_file(f"{root}/b")
    assert cache._check_file(f"{root}/c")
    active.close()
    assert (
        sum(
            path.stat().st_size
            for path in tmp_path.iterdir()
            if path.name not in ("cache", ".dew-lock") and not path.name.endswith(".pin")
        )
        <= 8
    )


def test_oversized_file_bypasses_transport_cache(tmp_path):
    remote = fsspec.filesystem("memory")
    path = f"cache-{uuid4().hex}/large"
    remote.pipe_file(path, b"abcdefgh")
    cache = BoundedFileCache(fs=remote, cache_storage=str(tmp_path), max_bytes=4)
    with cache.open(path, "rb") as file:
        assert file.read() == b"abcdefgh"
    assert not cache._check_file(path)


def test_concurrent_misses_download_once(tmp_path, monkeypatch):
    remote = fsspec.filesystem("memory")
    path = f"cache-{uuid4().hex}/one"
    remote.pipe_file(path, b"content")
    cache = BoundedFileCache(fs=remote, cache_storage=str(tmp_path), max_bytes=1024)
    download = remote.get_file
    calls = []

    def count(*args, **kwargs):
        calls.append(args[0])
        return download(*args, **kwargs)

    monkeypatch.setattr(remote, "get_file", count)

    def read(_):
        with cache.open(path, "rb") as file:
            return file.read()

    with ThreadPoolExecutor(4) as pool:
        assert list(pool.map(read, range(4))) == [b"content"] * 4
    assert len(calls) == 1


def test_distinct_transport_misses_download_concurrently_within_the_shared_budget(tmp_path, monkeypatch):
    remote = fsspec.filesystem("memory")
    prefix = f"parallel-cache-{uuid4().hex}"
    paths = [prefix + "/a", prefix + "/b"]
    for path in paths:
        remote.pipe_file(path, path.encode())
    waiting = threading.Barrier(2)
    download = remote.get_file

    def synchronize(*args, **kwargs):
        waiting.wait(timeout=5)
        return download(*args, **kwargs)

    monkeypatch.setattr(remote, "get_file", synchronize)
    cache = BoundedFileCache(fs=remote, cache_storage=str(tmp_path), max_bytes=1024)

    def read(path):
        with cache.open(path, "rb") as file:
            return file.read()

    with ThreadPoolExecutor(2) as pool:
        assert list(pool.map(read, paths)) == [path.encode() for path in paths]
    assert cache.metrics()["downloaded_bytes"] == sum(len(path) for path in paths)


def test_url_chain_reads_and_writes_memory(tmp_path):
    store = Store(
        f"dewcache::memory://chain-{uuid4().hex}/object",
        {"dewcache": {"cache_storage": str(tmp_path), "max_bytes": 1024}},
    )
    store.put_if_absent(b"through the chain")
    assert store.read() == b"through the chain"
    old = store.get_versioned()
    store.put_if_match(b"changed", old.version)
    assert store.read() == b"changed"


def test_gcs_generation_preconditions_are_sent_to_the_json_api():
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from urllib.parse import parse_qs, unquote, urlsplit

    versions = {}
    lock = threading.Lock()

    def metadata(key):
        generation, payload = versions[key][-1]
        return {
            "bucket": "bucket",
            "name": key,
            "size": str(len(payload)),
            "generation": str(generation),
            "updated": "2026-01-01T00:00:00.000Z",
        }

    class GCS(BaseHTTPRequestHandler):
        def log_message(self, *args):
            return

        def reply(self, status, payload):
            self.send_response(status)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_POST(self):
            query = parse_qs(urlsplit(self.path).query)
            key = query["name"][0]
            payload = self.rfile.read(int(self.headers["Content-Length"]))
            with lock:
                expected = str(versions[key][-1][0]) if key in versions else "0"
                if query.get("ifGenerationMatch") != [expected]:
                    self.reply(412, b'{"error":{"code":412,"message":"precondition failed"}}')
                    return
                versions.setdefault(key, []).append((int(expected) + 1, payload))
                self.reply(200, json.dumps(metadata(key)).encode())

        def do_GET(self):
            url = urlsplit(self.path)
            query = parse_qs(url.query)
            key = unquote(url.path.split("/o/", 1)[1]) if "/o/" in url.path else None
            with lock:
                if key is None:
                    prefix = query.get("prefix", [""])[0]
                    payload = json.dumps(
                        {"items": [metadata(name) for name in versions if name.startswith(prefix)]}
                    ).encode()
                elif key not in versions:
                    self.reply(404, b"missing")
                    return
                elif query.get("alt") == ["media"]:
                    wanted = int(query.get("generation", [versions[key][-1][0]])[0])
                    payload = next(payload for generation, payload in versions[key] if generation == wanted)
                else:
                    payload = json.dumps(metadata(key)).encode()
                self.reply(200, payload)

    with ThreadingHTTPServer(("127.0.0.1", 0), GCS) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        store = Store(
            "gs://bucket/value", {"token": "anon", "endpoint_url": f"http://127.0.0.1:{server.server_port}"}
        )
        try:
            first = store.put_if_absent(b"first")
            assert store.get_versioned().data == b"first"
            with pytest.raises(PreconditionFailed):
                store.put_if_absent(b"duplicate")
            second = store.put_if_match(b"second", first)
            with pytest.raises(PreconditionFailed):
                store.put_if_match(b"stale", first)
            assert store.get_versioned().version == second
            assert store.read() == b"second"
        finally:
            server.shutdown()
            thread.join()


@pytest.mark.network
def test_real_r2_conditional_publication_and_bounded_read(tmp_path):
    root = os.environ["DEW_DATA_TEST_URL"]  # a writable remote smoke prefix
    store = Store(root.rstrip("/") + "/CURRENT")
    first = ('{"generation":"' + uuid4().hex + '"}').encode()
    try:
        previous = store.get_versioned()
    except FileNotFoundError:
        store.put_if_absent(first)
    else:
        store.put_if_match(first, previous.version)
    with pytest.raises(PreconditionFailed):
        store.put_if_absent(b'{"generation":"two"}')
    cached = Store(
        "dewcache::" + store.url, {"dewcache": {"cache_storage": str(tmp_path), "max_bytes": 1024}}
    )
    assert cached.read() == first
    fs, path = store.resolve()
    assert fs.cat_file(path, start=0, end=8) == first[:8]
    fs.rm(path)
    assert not fs.exists(path)
