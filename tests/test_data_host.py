"""Automatic loading sized by its host: the CPUs this process may run on and the memory it may take,
each bounded by its cgroup's limits, read through the one cgroup reader a tool loads before Dew."""

import dataclasses
import json
import os

import grain.python as grain
import numpy as np
import pytest

from dew.data.compile import compile
from dew.data.dataset import GRAIN_KILL_SECONDS, DataPartition, Loading
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
        workers = reader.stop_seconds / GRAIN_KILL_SECONDS - 1
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


def test_automatic_loading_holds_its_reads_and_each_workers_batches_within_half_the_memory():
    """Each worker holds its `worker_buffer` batches and its `read_buffer` reads in a process of its
    own, within half the memory the process may take; a read is a batch where the order is read by
    batch, a record where it streams. Eight 64 KiB rows make a 512 KiB batch: 2 MiB holds four reads
    of a batch and no worker, and half a host's memory that holds two and a half workers holds two,
    reading by the batch or by the record. Workers take each CPU where there is more than one, at
    most `max_workers`."""
    from dew.data.dataset import WORKER_PROCESS_BYTES, Host

    rows = grain.MapDataset.source([{"x": np.zeros((8, 16384), np.float32)}] * 6).to_iter_dataset()
    auto = Loading(workers="auto", worker_buffer=2)
    small = dataclasses.replace(auto, ram_budget_mb=2).resolved(rows, reads=1, host=Host(8, 2**40))
    assert (small.workers, small.read_buffer) == (0, 4)
    held = WORKER_PROCESS_BYTES + 2 * 2**19 + 16 * 2**19
    roomy = auto.resolved(rows, reads=1, host=Host(8, 2 * (2 * held + held // 2)))
    assert (roomy.workers, roomy.read_buffer) == (2, 16)
    streamed = auto.resolved(rows, reads=8, host=Host(8, 2 * (2 * held + held // 2)))
    assert streamed.workers == 2 and streamed.read_buffer == 16
    assert auto.resolved(rows, reads=1, host=Host(3, 2**40)).workers == 3
    assert auto.resolved(rows, reads=1, host=Host(1, 2**40)).workers == 0
    assert dataclasses.replace(auto, max_workers=1).resolved(rows, reads=1, host=Host(8, 2**40)).workers == 1
