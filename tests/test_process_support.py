"""Worker reports and cleanup survive failed starts, failed exits and timeouts."""

import os
import select
import subprocess
import sys
from pathlib import Path

import pytest
from process_support import ProcessGroup, run_pool, start_process


def test_pool_reports_follow_rank_order_and_workers_own_sessions(tmp_path):
    def start(mode, out, processes, process_id, coordinator):
        code = (
            "import json, os, sys\n"
            "from pathlib import Path\n"
            "rank = int(sys.argv[1])\n"
            "Path(sys.argv[2]).write_text(json.dumps(\n"
            "    dict(rank=rank, session=os.getsid(0), pid=os.getpid())))\n"
        )
        return start_process([sys.executable, "-c", code, str(process_id), str(out)], devices=1)

    reports = run_pool("reports", tmp_path, 3, start=start)
    assert [report["rank"] for report in reports] == [0, 1, 2]
    assert len({report["session"] for report in reports}) == 3
    assert all(report["session"] == report["pid"] for report in reports)


def test_finalizer_kills_descendants_after_the_session_leader_exits():
    code = (
        "import os, time\n"
        "pid = os.fork()\n"
        "if pid == 0:\n"
        "    time.sleep(600)\n"
        "else:\n"
        "    print(pid, flush=True)\n"
    )
    with ProcessGroup() as pool:
        process = start_process([sys.executable, "-c", code], devices=1)
        pool.running.append(process)
        assert process.stdout is not None
        descendant = int(process.stdout.readline())
        process.wait(timeout=5)
        os.kill(descendant, 0)
    try:
        status = Path(f"/proc/{descendant}/stat").read_text().split()[2]
    except (FileNotFoundError, ProcessLookupError):
        # Reaped: its stat went before it could be read, or while it was.
        status = "gone"
    assert status in ("Z", "gone")


def test_failed_worker_keeps_every_ranks_output_and_stops_its_peers(tmp_path, capsys):
    running: list[subprocess.Popen[str]] = []

    def start(mode, out, processes, process_id, coordinator):
        code = (
            "import json, sys, time\n"
            "from pathlib import Path\n"
            "rank, out, count = int(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])\n"
            "print(f'worker {rank}', flush=True)\n"
            "out.with_suffix('.ready').touch()\n"
            "while len(list(out.parent.glob('*.ready'))) != count:\n"
            "    time.sleep(.01)\n"
            "if rank == 1:\n"
            "    sys.exit(3)\n"
            "if rank == 2:\n"
            "    time.sleep(600)\n"
            "out.write_text(json.dumps(dict(rank=rank)))\n"
        )
        process = start_process(
            [sys.executable, "-c", code, str(process_id), str(out), str(processes)], devices=1)
        running.append(process)
        return process

    with pytest.raises(AssertionError):
        run_pool("failure", tmp_path, 3, start=start, timeout=5)
    logs = capsys.readouterr().out
    # Each rank's own output reaches the failing pool's log.
    for rank in range(3):
        assert f"worker {rank}" in logs
    assert all(process.poll() is not None for process in running)


def test_failed_start_finalizes_workers_that_already_started(tmp_path):
    running: list[subprocess.Popen[str]] = []

    def start(mode, out, processes, process_id, coordinator):
        if process_id:
            raise OSError("cannot start the next rank")
        process = start_process([sys.executable, "-c", "import time; time.sleep(600)"], devices=1)
        running.append(process)
        return process

    with pytest.raises(OSError):
        run_pool("startup", tmp_path, 2, start=start)
    assert len(running) == 1 and running[0].poll() is not None


def test_timed_out_pool_reports_every_rank_and_reaps_them(tmp_path):
    running: list[subprocess.Popen[str]] = []

    def start(mode, out, processes, process_id, coordinator):
        process = start_process([sys.executable, "-c", "import time; time.sleep(600)"], devices=1)
        running.append(process)
        return process

    with pytest.raises(pytest.fail.Exception) as caught:
        run_pool("timeout", tmp_path, 2, start=start, timeout=.05)
    assert all(f"--- process {rank}, exit " in str(caught.value) for rank in range(2))
    assert all(process.poll() is not None for process in running)


def test_a_worker_ends_with_its_owner_however_the_owner_dies(tmp_path):
    """A worker in a session of its own, and what it started, end when the harness that owns them dies, by
    SIGKILL from outside as by any signal that unwinds nothing: their ownership outlives no owner (Main,
    2026-10-28: a worker outlived its killed owner 43 hours at 195% CPU on the VM; armada job
    20261028061537-7e66486a)."""
    owner = (
        "import sys, time\n"
        f"sys.path.insert(0, {str(Path(__file__).resolve().parent)!r})\n"
        "from process_support import start_process\n"
        "code = ('import os, time\\npid = os.fork()\\n'\n"
        "        'if pid:\\n    print(os.getpid(), pid, flush=True)\\ntime.sleep(600)\\n')\n"
        "worker = start_process([sys.executable, '-c', code], devices=1)\n"
        "print(worker.stdout.readline().strip(), flush=True)\n"
        "time.sleep(600)\n"
    )
    for signal_number in (9, 15):
        harness = subprocess.Popen([sys.executable, "-c", owner], stdout=subprocess.PIPE, text=True,
                                   start_new_session=True)
        assert harness.stdout is not None
        workers = [int(pid) for pid in harness.stdout.readline().split()]
        assert len(workers) == 2
        os.kill(harness.pid, signal_number)
        harness.wait(timeout=30)
        for worker in workers:
            assert gone_within(worker, 30), (signal_number, worker)


def test_a_worker_that_ends_by_itself_leaves_its_owner_nothing_running(tmp_path):
    """A worker that ends on its own, its owner reading its output and never stopping it (a rank that
    finishes), leaves no process of its ownership running beside the owner: its lifeline ends with it and
    kills nothing, as the group's id is then free for another process."""
    before = children()
    process = start_process([sys.executable, "-c", "print('done')"], devices=1)
    assert process.communicate(timeout=60)[0].strip() == "done"
    assert all(gone_within(pid, 30) for pid in children() - before), children() - before


def children() -> set[int]:
    """This process's live children, every thread's (/proc's children lists)."""
    return {int(pid) for task in Path("/proc/self/task").iterdir()
            for pid in (task / "children").read_text().split()}


def gone_within(pid: int, seconds: float) -> bool:
    """Whether process `pid` has ended (gone, or a zombie no one reaps yet) within `seconds`: its pidfd is
    readable once it ends."""
    try:
        ended = os.pidfd_open(pid)
    except ProcessLookupError:
        return True
    try:
        return bool(select.select([ended], [], [], seconds)[0])
    finally:
        os.close(ended)
