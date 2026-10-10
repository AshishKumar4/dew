"""Conditional publication, transport bounds and URL credentials."""

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
    assert sum(path.stat().st_size for path in tmp_path.iterdir()
               if path.name not in ("cache", ".dew-lock") and not path.name.endswith(".pin")) <= 8


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
    store = Store(f"dewcache::memory://chain-{uuid4().hex}/object", {
        "dewcache": {"cache_storage": str(tmp_path), "max_bytes": 1024}})
    store.put_if_absent(b"through the chain")
    assert store.read() == b"through the chain"
    old = store.get_versioned()
    store.put_if_match(b"changed", old.version)
    assert store.read() == b"changed"


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
    cached = Store("dewcache::" + store.url, {"dewcache": {
        "cache_storage": str(tmp_path), "max_bytes": 1024}})
    assert cached.read() == first
    fs, path = store.resolve()
    if "s3" in fs.protocol:
        fs.rm(path)
