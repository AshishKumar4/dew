"""The multi-process paths across real hosts: an armada gang of ARMADA_WORLD containers.

tests/test_multiprocess.py runs its pools on one host, over a loopback coordinator and one
filesystem. Here every rank of a gang (`{"gang": N}`, one container each, reached as host `rank<r>`)
runs its share of each pool's processes, which join at rank0, so a pool's collectives, its
coordinator and its checkpoints cross separate machines and separate disks. Every rank runs this
module in the same order, so the n-th pool of a run is the same pool on every rank; ranks swap what
their processes reported over the gang (`exchange`), and each asserts on the whole pool. Outside a
gang the module is skipped: CI's multihost job runs it on a gang of four.
"""

import json
import os
import re
import socket
import subprocess
import sys
import time
from itertools import count
from pathlib import Path

import pytest

RANK, WORLD = int(os.environ.get("ARMADA_RANK", "0")), int(os.environ.get("ARMADA_WORLD", "1"))
if WORLD < 2:
    pytest.skip("runs across the containers of an armada gang", allow_module_level=True)

import test_multiprocess as single  # noqa: E402
from process_support import report_of, spawn, terminate, worker_env  # noqa: E402
from test_multiprocess import (  # noqa: E402, F401
    local_flags,
    test_a_default_run_name_takes_its_timestamp_from_process_zero,
    test_four_processes_build_the_same_mesh_as_two,
    test_processes_read_disjoint_shards_that_cover_the_corpus,
    test_the_global_batch_is_the_union_of_the_process_slices,
    test_the_mesh_covers_every_process_in_the_pool,
    two_processes,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
POOLS = count()
"""The pools this run has started: the n-th is the same pool on every rank, at the same ports."""
COORDINATOR, EXCHANGE = 20000, 21000
KILL_AFTER = 60
"""Seconds into a pool's run that one of its hosts dies: past the join, in compilation or training."""


def exchange(sequence: int, mine: object) -> list:
    """What every rank gave for the `sequence`-th exchange, in rank order: each rank hands rank 0
    its part, and rank 0 hands every rank the whole."""
    if RANK == 0:
        parts = {0: mine}
        with socket.create_server(("0.0.0.0", EXCHANGE + sequence)) as server:
            server.settimeout(900)
            conns = []
            while len(parts) < WORLD:
                conn = server.accept()[0]
                rank, part = json.loads(_read(conn))
                parts[rank] = part
                conns.append(conn)
            whole = json.dumps([parts[rank] for rank in range(WORLD)]).encode()
            for conn in conns:
                conn.sendall(len(whole).to_bytes(8, "big") + whole)
                conn.close()
        return [parts[rank] for rank in range(WORLD)]
    deadline = time.monotonic() + 900
    while True:
        try:
            conn = socket.create_connection(("rank0", EXCHANGE + sequence), timeout=60)
            sent = json.dumps([RANK, mine]).encode()
            conn.sendall(len(sent).to_bytes(8, "big") + sent)
            whole = json.loads(_read(conn))
            conn.close()
            return whole
        except (ConnectionError, OSError):
            # Rank 0 has not opened this exchange yet: it may still be in the pool before it.
            if time.monotonic() > deadline:
                raise
            time.sleep(1)


def _read(conn: socket.socket) -> bytes:
    size = int.from_bytes(_exact(conn, 8), "big")
    return _exact(conn, size)


def _exact(conn: socket.socket, size: int) -> bytes:
    data = bytearray()
    while len(data) < size:
        chunk = conn.recv(min(size - len(data), 1 << 20))
        if not chunk:
            raise ConnectionError("the exchange closed early")
        data += chunk
    return bytes(data)


def placed(processes: int) -> list[int]:
    """The pool's processes this rank runs: process p runs on rank p mod WORLD."""
    return [index for index in range(processes) if index % WORLD == RANK]


def gang_pool(mode, directory: Path, processes: int, *, timeout=600, start=spawn, **flags) -> list[dict]:
    """`run_pool` across the gang: this rank's share of the pool's processes, joined at rank0, and every
    process's report, swapped over the gang. A rank whose process fails says so in the swap, so every rank
    fails rather than waiting on a report that will not come."""
    sequence = next(POOLS)
    directory.mkdir(parents=True, exist_ok=True)
    coordinator = f"rank0:{COORDINATOR + sequence}"
    outs = {index: directory / f"process{index}.json" for index in placed(processes)}
    running = {index: start(mode, out, processes=processes, process_id=index, coordinator=coordinator,
                            **flags) for index, out in outs.items()}
    mine: dict = {}
    try:
        pool = list(running.values())
        mine = {"reports": {str(index): report_of(running[index], out, timeout=timeout, pool=pool)
                            for index, out in outs.items()}}
    except BaseException as failure:
        mine = {"failed": f"rank {RANK}: {failure}"}
        raise
    finally:
        for process in running.values():
            if process.poll() is None:
                terminate(process)
        parts = exchange(sequence, mine)
    failed = [part["failed"] for part in parts if "failed" in part]
    if failed:
        pytest.fail("\n".join(failed))
    reports = {int(index): report for part in parts for index, report in part["reports"].items()}
    return [reports[index] for index in range(processes)]


# Every pool test of tests/test_multiprocess.py imported above runs its pools across the gang.
single.run_pool = gang_pool


def gang_run(command: list[str], *, devices: int, timeout: float = 1800) -> str:
    """`command` as this rank's process of a pool of one process per rank, joined at rank0 as `dew launch`
    leaves a pool (DEW_PROCESS_COUNT, DEW_PROCESS_ID, JAX_COORDINATOR_ADDRESS) on `devices` CPU devices each:
    its output, once it exited 0."""
    sequence = next(POOLS)
    env = {**worker_env(devices), "DEW_PROCESS_COUNT": str(WORLD), "DEW_PROCESS_ID": str(RANK),
           "JAX_COORDINATOR_ADDRESS": f"rank0:{COORDINATOR + sequence}"}
    ran = subprocess.run(command, cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=timeout)
    exits = exchange(sequence, ran.returncode)
    assert exits == [0] * WORLD, f"rank exits {exits}\n{ran.stdout[-6000:]}\n{ran.stderr[-6000:]}"
    return ran.stdout


@pytest.mark.distributed
def test_layouts_across_hosts_split_only_data_and_fsdp_and_match_one_device():
    """The parity tool's dense decoder and mixture of experts over one device on each of four hosts:
    the data, fsdp and replica layouts, whose sums cross the relay between containers, match one
    device, and an axis inside a model replica (tensor, stage) is refused across hosts, since each
    container is its own slice and only fsdp may cross slices."""
    if WORLD != 4:
        pytest.skip("one device on each of four hosts")
    output = gang_run([sys.executable, "tools/layout_parity.py", "--models", "dense", "moe", "--layouts",
                       "data4", "fsdp4", "replicas4", "replicas2_fsdp2", "tensor4", "stage4"], devices=1)
    if RANK == 0:
        summary = next(line for line in output.splitlines() if re.fullmatch(r"\w+ \d+(, \w+ \d+)*", line))
        counts = {status: int(count) for status, count in re.findall(r"(\w+) (\d+)", summary)}
        refusals = re.findall(r"^refused (\S+): (.*)$", output, re.M)
        assert counts == {"works": 8, "refused": 4}, output[-4000:]
        assert sorted(name for name, _ in refusals) == ["dense/stage4", "dense/tensor4", "moe/stage4",
                                                        "moe/tensor4"]
        assert all("granules" in reason for _, reason in refusals), refusals


@pytest.mark.distributed
def test_a_checkpoint_directory_on_each_hosts_own_disk_is_refused_on_every_host(tmp_path):
    """A pool whose persistent checkpoint directory is a path on each host's own disk, which no
    other host sees, is refused on every host before it trains: a step process 0 committed there
    would lack the other hosts' shards."""
    sequence = next(POOLS)
    if RANK >= 2:
        exchange(sequence, None)
        return
    process = spawn("fit", tmp_path / f"process{RANK}.json", processes=2, process_id=RANK,
                    coordinator=f"rank0:{COORDINATOR + sequence}", **local_flags(tmp_path))
    try:
        output = process.communicate(timeout=600)[0]
    except subprocess.TimeoutExpired:
        terminate(process)
        output = "still running after ten minutes"
    refusal = next((line for line in output.splitlines() if "is not shared" in line), output[-2000:])
    parts = exchange(sequence, [process.returncode, refusal])[:2]
    expected = "is not shared: process(es) [1] of 2"
    refused = [code not in (0, None) and expected in said for code, said in parts]
    assert all(refused), "\n".join(f"rank {rank} exited {code}: {said}"
                                    for rank, (code, said) in enumerate(parts))


@pytest.mark.distributed
def test_losing_one_host_ends_the_pool_on_the_others(tmp_path):
    """A host that dies while its pool trains (its process killed outright) ends the pool's process
    on the other host with an error, rather than leaving it waiting for ever on a peer that is gone."""
    sequence = next(POOLS)
    if RANK >= 2:
        exchange(sequence, None)
        return
    process = spawn("tracked", tmp_path / f"process{RANK}.json", processes=2, process_id=RANK,
                    coordinator=f"rank0:{COORDINATOR + sequence}", fsdp_size=2, steps=1_000_000)
    time.sleep(KILL_AFTER)
    alive = process.poll() is None
    if RANK == 1:
        terminate(process)
    outcome = None
    if RANK == 0:
        try:
            process.communicate(timeout=600)
            outcome = process.returncode
        except subprocess.TimeoutExpired:
            terminate(process)
    parts = exchange(sequence, [alive, outcome])[:2]
    assert [part[0] for part in parts] == [True, True], "a process ended before rank 1's host died"
    assert parts[0][1] not in (0, None), "rank 0's process kept waiting ten minutes after rank 1's host died"
