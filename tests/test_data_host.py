"""Automatic loading sized by its host: the CPUs this process may run on and the memory it may take,
each bounded by its cgroup's limits, read through the one cgroup reader a tool loads before Dew."""

import json
import os
import time

import grain.python as grain
import numpy as np
import pytest

from dew.data.compile import compile
from dew.data.dataset import DataPartition, Loading, grain_kill_seconds
from dew.data.pipeline import Pipeline
from dew.data.sources import Memory


def source(count=24):
    return Memory(tuple({"id": np.int32(index)} for index in range(count)))


@pytest.mark.parametrize("bound", ["cpus", "memory"])
def test_automatic_loading_starts_no_more_workers_than_the_hosts_cpus_and_memory_hold(monkeypatch, bound):
    """Automatic loading sizes its workers from the CPUs this process may run on and the memory it
    may take, together. On two CPUs of a 64-core machine it reads in at most two workers; in 512
    MiB, at most two, each a process of its own that imports the program again. Grain's pick read
    every core the machine has and counted a worker's memory as one batch, so it started 64 workers
    in both."""
    monkeypatch.setattr(os, "cpu_count", lambda: 64)
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: set(range(2 if bound == "cpus" else 16)))
    budget = {"cpus": 4096, "memory": 512}[bound]
    loading = Loading(workers="auto", ram_budget_mb=budget)
    data = compile(Pipeline(source(64)).repeat().batch(8), loading=loading)
    reader = data.train(DataPartition())
    try:
        workers = reader.stop_seconds / grain_kill_seconds() - 1
    finally:
        reader.close()
    assert workers <= 2


@pytest.mark.parametrize("limited", ["own", "parent"])
def test_a_cgroups_reclaimable_page_cache_is_room_for_automatic_loading(tmp_path, limited):
    """Under a cgroup's memory limit, the page cache its kernel reclaims first is room: a run that read
    1.5 GiB of its corpus into cache, charged to its cgroup of 2 GiB with 1.9 GiB in use, still has
    1.6 GiB, where counting the cache as taken left 0.1 GiB, room for no loading worker. The cgroup's
    CPU quota of two bounds the CPUs as its affinity does, and a limit on the container's cgroup
    binds the process's own cgroup beneath it, unlimited, as much as one on its own."""
    from dew.data.dataset import Host

    own = tmp_path / "session"
    own.mkdir()
    limits = tmp_path if limited == "parent" else own
    for level in (tmp_path, own):
        (level / "cpu.max").write_text("max 100000\n")
        (level / "memory.max").write_text("max\n")
    (limits / "cpu.max").write_text("200000 100000\n")
    (limits / "memory.max").write_text(f"{2 * 2**30}\n")
    (limits / "memory.current").write_text(f"{int(1.9 * 2**30)}\n")
    (limits / "memory.stat").write_text(f"anon {int(0.4 * 2**30)}\ninactive_file {int(1.5 * 2**30)}\n")
    host = Host.current(own, root=tmp_path)
    assert host.cpus == min(2, len(os.sched_getaffinity(0)))
    assert host.memory == 2 * 2**30 - int(1.9 * 2**30) + int(1.5 * 2**30)


def test_a_process_reads_its_cgroups_limits_through_the_reader_a_tool_loads_before_dew(tmp_path):
    """The cgroup reader Dew's loading sizes by is the one a tool loads by its path before Dew is
    importable, as CI's host keys do (row 4161): in a fresh interpreter that imports none of Dew, it
    finds the process's cgroup by the membership's v2 line and reads the binding limits up the tree."""
    import subprocess
    import sys
    from pathlib import Path

    root, own = tmp_path / "hierarchy", tmp_path / "hierarchy" / "session"
    own.mkdir(parents=True)
    (root / "cpu.max").write_text("300000 100000\n")
    (own / "memory.max").write_text(f"{3 * 2**30}\n")
    membership = tmp_path / "cgroup"
    membership.write_text("1:name=systemd:/elsewhere\n0::/session\n")
    reader = Path(__file__).resolve().parents[1] / "src" / "dew" / "data" / "cgroups.py"
    program = (
        "import importlib.util, sys\n"
        f"spec = importlib.util.spec_from_file_location('cgroups', {str(reader)!r})\n"
        "cgroups = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(cgroups)\n"
        "from pathlib import Path\n"
        f"root = Path({str(root)!r})\n"
        f"group = cgroups.own(root, Path({str(membership)!r}))\n"
        "levels = list(cgroups.levels(group, root))\n"
        "import json\n"
        "print(json.dumps([group.name, [cgroups.cpus(level) for level in levels],\n"
        "                  [cgroups.memory(level) for level in levels],\n"
        "                  any(name.split('.')[0] == 'dew' for name in sys.modules)]))\n")
    done = subprocess.run([sys.executable, "-I", "-c", program], capture_output=True, text=True, check=True)
    assert json.loads(done.stdout) == ["session", [None, 3], [3 * 2**30, None], False]


class _Filled:
    """A step that makes four batches' rows at once: every fourth row takes 40 ms, the rest none, as a
    packer fills its bins."""

    def __call__(self, index: int, row):
        if index % 4 == 0:
            time.sleep(0.04)
        return row


def test_automatic_loading_pays_a_step_that_makes_several_batches_at_once_in_their_mean():
    """Automatic loading weighs the Python a batch takes as the run pays it, over the sampled batches after
    the first: where a step makes four batches' rows at once, 40 ms every fourth batch is 10 ms a batch,
    though three batches of four take none, and it starts as many workers as cost the run least, the
    square root of the starts that Python pays for, at most the CPUs (row 4862: a decision run's packer
    filled four batches in one of 115 ms and served three in about 1 ms each, and the read stayed in the
    training process; sized by its CPUs, a 1,500-step run started 64 workers and took 168 s against 100 s
    in the process)."""
    from dew.data.dataset import Host

    rows = grain.MapDataset.source([{"x": np.zeros(4, np.float32)}] * 64).map_with_index(_Filled())
    auto = Loading(workers="auto", host=Host(4, 2**40, worker_start=1.0))

    def resolved(batches: int) -> int:
        # Read in turn, as automatic loading reads its sample (`_Build.tuned`): no thread reading ahead.
        sampled = rows.to_iter_dataset(grain.ReadOptions(num_threads=0, prefetch_buffer_size=0))
        return auto.resolved(sampled, reads=1, batches=batches).workers

    # At least 10 ms a batch over starts of a second: 2,500 batches cost least in five workers, which the
    # four CPUs cap, 500 in two (up to 12 ms a batch), and 50 pay for none.
    assert (resolved(2500), resolved(500), resolved(50)) == (4, 2, 0)


def test_automatic_loading_holds_its_reads_and_each_workers_batches_within_half_the_memory():
    """Each worker holds its `worker_buffer` batches and its `read_buffer` reads in a process of its
    own, within half the memory its host leaves the process (`Loading.host`); a read is a batch where
    the order is read by batch, a record where it streams. Eight 64 KiB rows make a 512 KiB batch:
    2 MiB holds four reads of a batch and no worker, and half a host's memory that holds two and a half
    workers holds two, reading by the batch or by the record. Workers take each CPU where there is more
    than one, at most `max_workers`. That is over a run whose batches' Python pays for the workers'
    start, a second each on these hosts (`batches`); a run too short for it, or of no stated length,
    reads in the process, its reads sized alike (row 4862)."""
    from dew.data.dataset import WORKER_PROCESS_BYTES, Host

    paid = 10**12  # A run long enough to pay for any start.
    rows = grain.MapDataset.source([{"x": np.zeros((8, 16384), np.float32)}] * 6).to_iter_dataset()
    held = WORKER_PROCESS_BYTES + 2 * 2**19 + 16 * 2**19

    def on(cpus: int, memory: int, **fields) -> Loading:
        return Loading(workers="auto", worker_buffer=2, host=Host(cpus, memory, worker_start=1.0), **fields)

    small = on(8, 2**40, ram_budget_mb=2).resolved(rows, reads=1, batches=paid)
    assert (small.workers, small.read_buffer) == (0, 4)
    roomy = on(8, 2 * (2 * held + held // 2)).resolved(rows, reads=1, batches=paid)
    assert (roomy.workers, roomy.read_buffer) == (2, 16)
    streamed = on(8, 2 * (2 * held + held // 2)).resolved(rows, reads=8, batches=paid)
    assert streamed.workers == 2 and streamed.read_buffer == 16
    assert on(3, 2**40).resolved(rows, reads=1, batches=paid).workers == 3
    assert on(1, 2**40).resolved(rows, reads=1, batches=paid).workers == 0
    assert on(8, 2**40, max_workers=1).resolved(rows, reads=1, batches=paid).workers == 1
    for batches in (1, None):
        unpaid = on(8, 2 * (2 * held + held // 2)).resolved(rows, reads=1, batches=batches)
        assert (unpaid.workers, unpaid.read_buffer) == (0, 16)
    # A host whose workers start at no cost takes as many as its CPUs hold over any stated run.
    free = Loading(workers="auto", worker_buffer=2, host=Host(3, 2**40, worker_start=0.0))
    assert free.resolved(rows, reads=1, batches=1).workers == 3
