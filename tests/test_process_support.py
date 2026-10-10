"""Worker reports and cleanup survive failed starts, failed exits and timeouts."""

import os
import subprocess
import sys
from pathlib import Path

import pytest
from process_support import ProcessGroup, prepare_worker, run_pool, start_process


def test_worker_bootstrap_keeps_launched_detection_distinct_from_single_host(monkeypatch):
    from types import SimpleNamespace

    from dew.training import runtime

    joins = []
    monkeypatch.setattr(runtime, "prepare_process", lambda **kwargs: joins.append(kwargs["multi_host"]))
    args = SimpleNamespace(coordinator=None, processes=2, process_id=1)
    prepare_worker(args)
    prepare_worker(args, multi_host=None)
    expected = {"OMPI_MCA_orte_hnp_uri": "0.0;tcp://127.0.0.1:1234", "OMPI_COMM_WORLD_SIZE": "2",
                "OMPI_COMM_WORLD_RANK": "1", "OMPI_COMM_WORLD_LOCAL_RANK": "1",
                "JAX_COORDINATOR_ADDRESS": "127.0.0.1:1234"}
    for name in expected:
        monkeypatch.setenv(name, "previous")
    args.coordinator = "127.0.0.1:1234"
    prepare_worker(args)
    assert joins == [False, None, True]
    assert {name: os.environ[name] for name in expected} == expected


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

    with pytest.raises(AssertionError, match=r"process1\.json exited 3"):
        run_pool("failure", tmp_path, 3, start=start, timeout=5)
    logs = capsys.readouterr().out
    for rank in range(3):
        assert f"--- process {rank}, exit " in logs
        assert f"worker {rank}" in logs
    assert all(process.poll() is not None for process in running)


def test_failed_start_finalizes_workers_that_already_started(tmp_path, capsys):
    running: list[subprocess.Popen[str]] = []

    def start(mode, out, processes, process_id, coordinator):
        if process_id:
            raise OSError("cannot start the next rank")
        process = start_process([sys.executable, "-c", "import time; time.sleep(600)"], devices=1)
        running.append(process)
        return process

    with pytest.raises(OSError, match="cannot start the next rank"):
        run_pool("startup", tmp_path, 2, start=start)
    assert len(running) == 1 and running[0].poll() is not None
    assert "--- process 0, exit " in capsys.readouterr().out


def test_timed_out_pool_reports_every_rank_and_reaps_them(tmp_path):
    running: list[subprocess.Popen[str]] = []

    def start(mode, out, processes, process_id, coordinator):
        process = start_process([sys.executable, "-c", "import time; time.sleep(600)"], devices=1)
        running.append(process)
        return process

    with pytest.raises(pytest.fail.Exception, match="the pool did not") as caught:
        run_pool("timeout", tmp_path, 2, start=start, timeout=.05)
    assert all(f"--- process {rank}, exit " in str(caught.value) for rank in range(2))
    assert all(process.poll() is not None for process in running)
