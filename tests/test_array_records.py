"""ArrayRecord shard readers: how many each source and the process hold open, their mirrors, and
reads in flight through eviction and close."""

import os
import pickle
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from uuid import uuid4

import psutil
import pytest

from dew.data.dataset import Loading
from dew.data.storage import Store


@pytest.fixture
def arrayrecord_paths(tmp_path):
    from array_record.python.array_record_module import ArrayRecordWriter

    paths = tuple(str(tmp_path / f"{shard}.array_record") for shard in range(Loading().threads + 1))
    for shard, path in enumerate(paths):
        writer = ArrayRecordWriter(path, "group_size:1")
        for index in range(8):
            writer.write(f"{shard}:{index}".encode())
        writer.close()
    return paths


def test_arrayrecord_sources_hold_their_readers_open_within_one_bound_together(arrayrecord_paths):
    """A run of more ArrayRecord sources than Loading's read threads, one file each, as a mixture of
    hundreds reads them: each read in turn, the process holds at most as many files open as one
    source may, the least recent closed as another opens (row 2016). Before, each source kept its own
    readers, every file it had read held open."""
    from dew.data.sources.records import ArrayRecords

    real = {os.path.realpath(path) for path in arrayrecord_paths}
    sources = [ArrayRecords((path,), counts=(8,)) for path in arrayrecord_paths]

    def held() -> int:
        return sum(opened.path in real for opened in psutil.Process().open_files())
    try:
        for source in sources:
            assert source[0] is not None
        assert held() <= Loading().threads < len(sources)
    finally:
        for source in sources:
            source.close()


def test_a_reader_falling_idle_past_the_bound_is_closed_as_its_read_drains(arrayrecord_paths, monkeypatch):
    """A read in flight on the first of more sources than Loading's read threads while every other is
    read in turn: the others' readers fill the bound idle, and as the first's read drains its reader
    is the one past it, closed then, not held until another source opens (row 2016)."""
    import grain.python as grain

    from dew.data.sources.records import ArrayRecords

    real = {os.path.realpath(path) for path in arrayrecord_paths}
    sources = [ArrayRecords((path,), counts=(8,)) for path in arrayrecord_paths]
    entered, released = threading.Event(), threading.Event()
    underlying_read = grain.ArrayRecordDataSource.__getitem__
    first = os.path.realpath(arrayrecord_paths[0])

    def blocked_read(reader, index):
        if index == 1 and any(os.path.realpath(path) == first for path in reader._paths):
            entered.set()
            assert released.wait(timeout=30)
        return underlying_read(reader, index)

    monkeypatch.setattr(grain.ArrayRecordDataSource, "__getitem__", blocked_read)
    try:
        assert sources[0][0] is not None
        with ThreadPoolExecutor(1) as pool:
            pending = pool.submit(sources[0].__getitem__, 1)
            assert entered.wait(timeout=30)
            for source in sources[1:]:
                assert source[0] is not None
            released.set()
            assert pending.result(timeout=30) is not None
        assert sum(opened.path in real for opened in psutil.Process().open_files()) <= Loading().threads
    finally:
        released.set()
        for source in sources:
            source.close()


def test_arrayrecords_threaded_reads_overlap(arrayrecord_paths, monkeypatch):
    import grain.python as grain

    from dew.data.sources import records

    source = records.ArrayRecords(arrayrecord_paths[:1], counts=(8,))
    active = maximum = 0
    lock = threading.Lock()
    entered = threading.Barrier(3)
    released = threading.Event()
    underlying_read = grain.ArrayRecordDataSource.__getitem__

    def blocked_read(reader, index):
        nonlocal active, maximum
        with lock:
            active += 1
            maximum = max(maximum, active)
        try:
            if index < 2:
                entered.wait(timeout=5)
                assert released.wait(timeout=5)
            return underlying_read(reader, index)
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(grain.ArrayRecordDataSource, "__getitem__", blocked_read)
    try:
        with ThreadPoolExecutor(2) as pool:
            pending = [pool.submit(source.__getitem__, index) for index in range(8)]
            try:
                entered.wait(timeout=5)
                assert maximum == 2
            finally:
                released.set()
            assert [future.result(timeout=5) for future in pending] == [
                f"0:{index}".encode() for index in range(8)
            ]
    finally:
        source.close()


@pytest.mark.parametrize("fail", [False, True])
@pytest.mark.parametrize("retire", ["eviction", "close"])
def test_arrayrecords_keep_busy_readers_alive_during_eviction_or_close(
    arrayrecord_paths, monkeypatch, fail, retire
):
    import grain.python as grain

    from dew.data.sources import records

    source = records.ArrayRecords(arrayrecord_paths, counts=(8,) * len(arrayrecord_paths))
    entered = threading.Barrier(3)
    released = [threading.Event(), threading.Event()]
    first_reader = []
    underlying_read = grain.ArrayRecordDataSource.__getitem__
    failed_read = ValueError("failed child read")

    def descriptors(path):
        real = os.path.realpath(path)
        return {opened.fd for opened in psutil.Process().open_files() if opened.path == real}

    def blocked_read(reader, index):
        if not first_reader:
            first_reader.append(id(reader))
        held = id(reader) == first_reader[0]
        payload = underlying_read(reader, index)
        # A failed read's traceback must not keep the external reader alive after Dew closes it.
        del reader
        if held and index < 2:
            entered.wait(timeout=5)
            assert released[index].wait(timeout=5)
            if fail:
                raise failed_read
        return payload

    monkeypatch.setattr(grain.ArrayRecordDataSource, "__getitem__", blocked_read)
    try:
        assert source[2] == b"0:2"
        held = descriptors(arrayrecord_paths[0])
        assert held
        with ThreadPoolExecutor(2) as pool:
            pending = [pool.submit(source.__getitem__, index) for index in range(2)]
            try:
                entered.wait(timeout=5)
                if retire == "eviction":
                    for shard in range(1, len(arrayrecord_paths)):
                        assert source[8 * shard] == f"{shard}:0".encode()
                else:
                    source.close()
                assert descriptors(arrayrecord_paths[0]) == held
                # Close retires the busy reader; eviction keeps it and retires an idle one.
                assert source[2] == b"0:2"
                path = arrayrecord_paths[-1] if retire == "eviction" else arrayrecord_paths[0]
                replacement = descriptors(path) - held
                assert replacement
                for index, future in enumerate(pending):
                    released[index].set()
                    if fail:
                        with pytest.raises(ValueError) as raised:
                            future.result(timeout=5)
                        assert raised.value is failed_read
                    else:
                        assert future.result(timeout=5) == f"0:{index}".encode()
                    still_open = descriptors(arrayrecord_paths[0])
                    assert held & still_open == (set() if retire == "close" and index == 1 else held)
                    assert replacement <= descriptors(path)
            finally:
                for release in released:
                    release.set()
    finally:
        source.close()
    assert all(not descriptors(path) for path in arrayrecord_paths)


def test_pickled_arrayrecords_open_independent_readers(arrayrecord_paths):
    from dew.data.sources.records import ArrayRecords

    source = ArrayRecords(arrayrecord_paths, counts=(8,) * len(arrayrecord_paths))
    assert source[0] == b"0:0"
    restored = pickle.loads(pickle.dumps(source))
    try:
        source.close()
        assert restored[1] == b"0:1"
        restored.close()
        assert source[2] == b"0:2"
    finally:
        source.close()
        restored.close()


@pytest.mark.parametrize("max_bytes", [1, 1 << 30])
def test_arrayrecords_refuse_a_mirror_without_room_for_the_read_width(arrayrecord_paths, tmp_path, max_bytes):
    """A local mirror with room for the shards a read holds open serves them;
    one of a byte is refused."""
    from dew.data.sources.records import ArrayRecords

    shard = Store(f"memory://small-record-mirror-{uuid4().hex}/0.array_record")
    shard.write(Path(arrayrecord_paths[0]).read_bytes())
    source = ArrayRecords(
        (shard.url,),
        {"dewcache": {"cache_storage": str(tmp_path / "mirror"), "max_bytes": max_bytes}},
        counts=(8,),
    )
    try:
        if max_bytes > 1:
            assert source[0] == b"0:0"
            return
        with pytest.raises(ValueError):
            source[0]
    finally:
        source.close()


@pytest.mark.parametrize("busy_oldest,retire", [(False, "eviction"), (True, "eviction"), (True, "close")])
def test_arrayrecords_release_an_idle_mirror_before_opening_its_replacement(
    arrayrecord_paths, tmp_path, monkeypatch, busy_oldest, retire
):
    import grain.python as grain

    from dew.data.sources.records import ArrayRecords

    root = Store(f"memory://bounded-record-mirror-{uuid4().hex}")
    shards = [root.child(f"{index}.array_record") for index in range(len(arrayrecord_paths))]
    for shard, path in zip(shards, arrayrecord_paths, strict=True):
        shard.write(Path(path).read_bytes())
    capacity = sum(Path(path).stat().st_size for path in arrayrecord_paths[:-1])
    source = ArrayRecords(
        [shard.url for shard in shards],
        {"dewcache": {"cache_storage": str(tmp_path / "mirror"), "max_bytes": capacity}},
        counts=(8,) * len(shards),
    )
    entered, released = threading.Event(), threading.Event()
    underlying_read = grain.ArrayRecordDataSource.__getitem__

    def held(reader, index):
        if busy_oldest and not entered.is_set():
            entered.set()
            assert released.wait(timeout=5)
        return underlying_read(reader, index)

    monkeypatch.setattr(grain.ArrayRecordDataSource, "__getitem__", held)
    try:
        if busy_oldest:
            with ThreadPoolExecutor(1) as pool:
                pending = pool.submit(source.__getitem__, 0)
                try:
                    assert entered.wait(timeout=5)
                    if retire == "close":
                        source.close()
                    for shard in range(1, len(shards)):
                        assert source[8 * shard] == f"{shard}:0".encode()
                finally:
                    released.set()
                assert pending.result(timeout=5) == b"0:0"
        for _ in range(2):
            for shard in range(len(shards)):
                assert source[8 * shard] == f"{shard}:0".encode()
    finally:
        source.close()


def test_arrayrecords_wait_for_a_busy_reader_before_opening_a_missing_shard(arrayrecord_paths, monkeypatch):
    import grain.python as grain

    from dew.data.sources.records import ArrayRecords

    source = ArrayRecords(arrayrecord_paths, counts=(8,) * len(arrayrecord_paths))
    width = Loading().threads
    entered = threading.Barrier(width + 1)
    released, queued = threading.Event(), threading.Event()
    lock = threading.Lock()
    admitted = 0
    underlying_read = grain.ArrayRecordDataSource.__getitem__
    condition_wait = threading.Condition.wait

    def held(reader, index):
        nonlocal admitted
        with lock:
            ordinal = admitted
            admitted += 1
        if ordinal < width:
            entered.wait(timeout=5)
            assert released.wait(timeout=5)
        return underlying_read(reader, index)

    def waiting(condition, timeout=None):
        if condition is source._shards._lock:
            queued.set()
        return condition_wait(condition, timeout)

    monkeypatch.setattr(grain.ArrayRecordDataSource, "__getitem__", held)
    monkeypatch.setattr(threading.Condition, "wait", waiting)
    try:
        with ThreadPoolExecutor(width + 1) as pool:
            pending = [pool.submit(source.__getitem__, 8 * shard) for shard in range(width)]
            try:
                entered.wait(timeout=5)
                overflow = pool.submit(source.__getitem__, 8 * width)
                assert queued.wait(timeout=5)
                assert not overflow.done()
            finally:
                released.set()
            assert [future.result(timeout=5) for future in pending] == [
                f"{shard}:0".encode() for shard in range(width)
            ]
            assert overflow.result(timeout=5) == f"{width}:0".encode()
    finally:
        source.close()


def test_a_read_past_the_bound_never_waits_on_another_source_opening_a_shard(arrayrecord_paths, monkeypatch):
    """A source opening a shard holds its own lock while it opens (a remote shard downloads). A read
    of another source that leaves the process past its idle bound trims without waiting on it: it
    passes the opening source over and returns, and the bound holds once that source's read drains
    (row 2046). Before, the trim took every source's lock in turn and waited on the open."""
    from dew.data.sources.records import ArrayRecords

    real = {os.path.realpath(path) for path in arrayrecord_paths}
    opening, released = threading.Event(), threading.Event()
    slow = os.path.realpath(arrayrecord_paths[-1])
    underlying_open = ArrayRecords._shard

    def slow_open(self, shard):
        if os.path.realpath(self.urls[shard]) == slow:
            opening.set()
            assert released.wait(timeout=60)
        return underlying_open(self, shard)

    monkeypatch.setattr(ArrayRecords, "_shard", slow_open)
    held = [ArrayRecords((path,), counts=(8,)) for path in arrayrecord_paths[:-1]]
    late = ArrayRecords((arrayrecord_paths[-1],), counts=(8,))
    again = ArrayRecords((arrayrecord_paths[0],), counts=(8,))
    try:
        for source in held:
            assert source[0] is not None
        with ThreadPoolExecutor(2) as pool:
            pending = pool.submit(late.__getitem__, 0)
            assert opening.wait(timeout=60)
            assert pool.submit(again.__getitem__, 1).result(timeout=60) is not None
            assert not pending.done()
            released.set()
            assert pending.result(timeout=60) is not None
        assert sum(opened.path in real for opened in psutil.Process().open_files()) <= Loading().threads
    finally:
        released.set()
        for source in (*held, late, again):
            source.close()
