"""Conditional publication, transport bounds and URL credentials."""

import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4

import fsspec
import pytest

from dew.data.storage import BoundedFileCache, PreconditionFailed, Store, _profile


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
    with pytest.raises(ValueError, match=r"credentialtest.*DEW_TEST_MISSING_SECRET"):
        Store("credentialtest://bucket/key").resolve()
    monkeypatch.setenv("DEW_TEST_MISSING_SECRET", "secret-value")
    fs, path = Store("credentialtest://bucket/key").resolve()
    fs.pipe_file(path, b"present")
    assert fs.cat_file(path) == b"present"


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
    root = os.environ.get("DEW_DATA_TEST_URL")
    if not root:
        pytest.skip("DEW_DATA_TEST_URL names the writable R2 smoke prefix")
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
    fs.rm(path)
    assert not fs.exists(path)
