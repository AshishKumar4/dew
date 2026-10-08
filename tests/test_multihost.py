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
    LOCAL_KILL_AFTER,
    STEPS,
    dumped_params,
    local_committed,
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
def test_every_layout_across_hosts_matches_one_device():
    """Data, FSDP, tensor, expert, sequence and pipeline layouts of the parity tool's dense decoder and its
    mixture of experts, over four devices on the gang's hosts, each against one device: every split's sums
    cross the relay between containers, which one host's loopback never does."""
    if 4 % WORLD:
        pytest.skip(f"the layouts are four-device layouts, which {WORLD} hosts do not split evenly")
    output = gang_run([sys.executable, "tools/layout_parity.py", "--models", "dense", "moe"],
                      devices=4 // WORLD)
    if RANK == 0:
        assert "mismatch" not in output and "works" in output, output[-4000:]


@pytest.mark.distributed
def test_each_host_resumes_a_lost_pool_from_its_own_local_checkpoint(tmp_path):
    """Two hosts train with a local checkpoint every two steps on each host's own disk. Both are
    killed once local step eight landed, as a preemption takes a pool; the restarted pool reads step
    eight back on each host from its own disk and lands on the parameters of the run nobody killed."""
    if RANK >= 2:
        # A pool of two: the other ranks swap nothing but their place.
        exchange(next(POOLS), None)
        gang_pool("fit", tmp_path / "whole", 2, **local_flags(tmp_path / "whole-run"))
        gang_pool("fit", tmp_path / "resumed", 2, **local_flags(tmp_path))
        return
    sequence = next(POOLS)
    marker = tmp_path / "blocked"
    (tmp_path / "killed").mkdir()
    process = spawn("fit", tmp_path / "killed" / f"process{RANK}.json", processes=2, process_id=RANK,
                    coordinator=f"rank0:{COORDINATOR + sequence}",
                    **local_flags(tmp_path, block_after=LOCAL_KILL_AFTER, marker=marker))
    landed = local_committed(tmp_path, RANK, LOCAL_KILL_AFTER)
    deadline = time.monotonic() + 900
    while (not (marker.exists() and landed.exists()) and time.monotonic() < deadline
           and process.poll() is None):
        time.sleep(0.1)
    blocked = marker.exists() and landed.exists()
    terminate(process)
    output = "" if blocked else process.stdout.read()[-4000:]
    parts = exchange(sequence, [blocked, output])[:2]
    assert [part[0] for part in parts] == [True, True], "\n".join(part[1] for part in parts)

    whole = gang_pool("fit", tmp_path / "whole", 2, **local_flags(tmp_path / "whole-run"))
    resumed = gang_pool("fit", tmp_path / "resumed", 2, **local_flags(tmp_path))

    assert resumed[RANK]["restored_step"] == LOCAL_KILL_AFTER
    assert resumed[RANK]["restored_from"] == str(tmp_path / "local" / f"process{RANK}")
    assert [report["step"] for report in resumed] == [STEPS, STEPS] == [report["step"] for report in whole]
    single.assert_same_parameters(dumped_params(tmp_path / "resumed" / f"process{RANK}.json"),
                                  dumped_params(tmp_path / "whole" / f"process{RANK}.json"))


@pytest.mark.distributed
def test_losing_one_host_ends_the_pool_on_the_others(tmp_path):
    """A host that dies mid-run (its process killed outright) ends the pool's other processes with an error,
    rather than leaving them waiting for ever on a peer that is gone."""
    sequence = next(POOLS)
    if RANK >= 2:
        exchange(sequence, None)
        return
    marker = tmp_path / "blocked"
    process = spawn("fit", tmp_path / f"process{RANK}.json", processes=2, process_id=RANK,
                    coordinator=f"rank0:{COORDINATOR + sequence}",
                    **local_flags(tmp_path, block_after=3, marker=marker))
    deadline = time.monotonic() + 900
    while not marker.exists() and process.poll() is None and time.monotonic() < deadline:
        time.sleep(0.1)
    if RANK == 1:
        terminate(process)
        exchange(sequence, "killed")
        return
    exchange(sequence, "waiting")
    try:
        output = process.communicate(timeout=600)[0]
    except subprocess.TimeoutExpired:
        terminate(process)
        pytest.fail("rank 0's process kept waiting ten minutes after rank 1's host died")
    assert process.returncode != 0, output[-4000:]
