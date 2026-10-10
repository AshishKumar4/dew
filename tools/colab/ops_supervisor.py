"""Independent lifecycle watchdog, receipts and exact-owned-endpoint cleanup."""

import argparse
import json
import os
import signal
import subprocess
import time
from pathlib import Path
from urllib.parse import urljoin

import requests

from tools.colab import colab_batch, colab_io, metadata_capture
from tools.colab.colab_new import Colab
from tools.colab.common import REPO, STATE, driver_sha, lock, name, read, save

RELEASED = {"released", "already absent", "no assignment acquired"}
RELEASE_BACKOFF = (60, 180, 600)


def accounting(run: dict, allocation: dict, driver: dict, released_at: float | None, now: float) -> dict:
    end = released_at if released_at is not None else now
    assigned = allocation.get("assigned_at", driver.get("assigned_at", end))
    requested = allocation.get("requested_at", assigned)
    seconds = max(0, end - assigned)
    result = {
        "session_seconds": seconds,
        "cu_estimate": seconds * run["rate"] / 3600,
        "cu_upper_bound": max(0, end - min(requested, assigned)) * run["rate"] / 3600,
        "cu_provisional": released_at is None,
        "billing_finished_at": released_at,
        "accounting": "observed allocation through release; upper bound includes provisioning",
    }
    jobs = driver.get("results", [])
    if jobs and driver.get("progress", {}).get("phase") == "done":
        overhead = max(0, seconds - sum(job["stage_seconds"] for job in jobs))
        result["shared_overhead_seconds"] = overhead
        result["per_job_cu"] = [
            (job["stage_seconds"] + overhead / len(jobs)) * run["rate"] / 3600 for job in jobs
        ]
    return result


def remaining_budget(run: dict, allocation: dict, now: float) -> float:
    if run.get("budget_cu") is None or "requested_at" not in allocation:
        return float("inf")
    return max(0, allocation["requested_at"] + run["budget_cu"] * 3600 / run["rate"] - now)


def finalize(run_dir: Path, client: Colab, reason: str | None = None, code: int | None = None) -> dict:
    previous = read(run_dir / "result.json")
    if previous.get("release") in RELEASED or previous.get("needs_attention"):
        return previous
    run = read(run_dir / "run.json")
    driver = read(run_dir / "driver.json")
    allocation = read(run_dir / "allocation.json")
    result = previous or {
        "session": run["session"],
        "driver_sha": run["driver_sha"],
        "driver_exit": code,
        "reason": reason,
        "job_results": driver.get("results", []),
        "profile_preamble": driver.get("profile_preamble"),
        "artifact": str(run_dir),
        "started_at": run["started_at"],
        "finished_at": time.time(),
    }
    save(run_dir / "result.json", result)
    if time.time() < result.get("release_retry_at", 0):
        result.update(accounting(run, allocation, driver, None, time.time()))
        save(run_dir / "result.json", result)
        return result
    released_at = None
    try:
        endpoint = driver.get("endpoint") or client.recover_endpoint(run_dir / "allocation.json")
        if not endpoint and allocation:
            # An absent reply can still provision later. Retain the UUID and
            # leave the account estimate open for the reconciler.
            result["release"] = (
                "allocation unresolved"
                if time.time() < allocation["requested_at"] + 1800
                else "already absent"
            )
            if result["release"] == "already absent":
                released_at = time.time()
        else:
            if endpoint and driver.get("progress", {}).get("phase") != "done":
                emergency(run_dir, run, allocation, client, endpoint, result)
            result["release"] = client.release(run["session"], endpoint)
            released_at = time.time()
    except Exception as error:
        result["release"] = "failed: " + type(error).__name__
    history = result.setdefault("cleanup_history", [])
    history.append({"at": time.time(), "release": result["release"], "trigger": reason})
    if result["release"] not in RELEASED:
        attempt = len(history) - 1
        if attempt >= len(RELEASE_BACKOFF):
            result.update(
                release="release failed: needs attention",
                needs_attention=True,
                notify="Main",
                release_retry_at=None,
            )
            save(
                run_dir / "attention.json",
                {
                    "notify": "Main",
                    "endpoint": driver.get("endpoint"),
                    "reason": "bounded release attempts exhausted",
                },
            )
            print(f"NOTIFY Main: release needs attention; {run_dir / 'result.json'}", flush=True)
        else:
            result["release_retry_at"] = time.time() + RELEASE_BACKOFF[attempt]
    else:
        result["release_retry_at"] = None
    result.update(accounting(run, allocation, driver, released_at, time.time()))
    save(run_dir / "result.json", result)
    plan = read(run_dir / "staging/plan.json")
    if plan:
        directories = [run_dir / "out/_preamble/profile-check"]
        for job in plan["jobs"]:
            directories.extend(
                [
                    STATE / "colab-resume" / job["resume_key"],
                    colab_batch.artifact_path(plan, run_dir.name, job),
                ]
            )
        colab_io.reconcile_transfers(tuple(directories))
        colab_batch.hold_unreported(
            plan, {job["index"] for job in driver.get("results", [])}, reason or "unobserved completion"
        )
    return result


def emergency(run_dir: Path, run: dict, allocation: dict, client: Colab, endpoint: str, result: dict) -> None:
    try:
        plan = read(run_dir / "staging/plan.json")
        if plan:
            result["emergency_metadata"] = metadata_capture.capture(
                run_dir,
                plan,
                client.session(endpoint),
                run["remote_root"],
                min(60, remaining_budget(run, allocation, time.time())),
            )
    except Exception as error:
        result["emergency_metadata"] = {"error": type(error).__name__}


def reconcile(root: Path, client: Colab) -> None:
    for journal in root.glob("*/run.json"):
        run = read(journal)
        try:
            alive = colab_io.process_start(run["pid"]) == run["process_start"]
        except (OSError, KeyError, IndexError):
            alive = False
        if not alive:
            with lock(journal.parent / "finalize.lock"):
                finalize(journal.parent, client, "orphaned controller; no inferred job exit")
    pending = [read(path.parent / "result.json") for path in root.glob("*/run.json")]
    if not any(item.get("release") not in RELEASED and not item.get("needs_attention") for item in pending):
        subprocess.run(["systemctl", "--user", "stop", "dew-colab-ops-reconcile.timer"], check=True)


def watch(child: subprocess.Popen, run_dir: Path, run: dict, client: Colab) -> str | None:
    last_seen = time.time()
    missing = 0
    while child.poll() is None:
        time.sleep(15)
        driver = read(run_dir / "driver.json")
        allocation = read(run_dir / "allocation.json")
        now = time.time()
        if remaining_budget(run, allocation, now) <= 0:
            return "packed-session CU budget reached"
        if (
            run.get("hard_cap_minutes")
            and allocation
            and now - allocation["requested_at"] >= run["hard_cap_minutes"] * 60
        ):
            return "session time cap reached"
        if now > (driver.get("action_deadline") or driver.get("heartbeat", run["started_at"]) + 240):
            return "driver heartbeat/action deadline exceeded"
        if not driver.get("endpoint"):
            continue
        try:
            present = any(item.endpoint == driver["endpoint"] for item in client.assignments())
            missing = 0 if present else missing + 1
            last_seen = now
            if missing >= 2:
                return "owned assignment absent in two checks"
            if present and driver.get("work_kernel") and not driver.get("work_execution_ended_at"):
                session = client.session(driver["endpoint"])
                response = requests.get(
                    urljoin(session.url, "api/kernels/" + driver["work_kernel"]),
                    headers=session.headers,
                    timeout=20,
                )
                response.raise_for_status()
                if response.json().get("execution_state") in ("idle", "dead"):
                    progress = json.loads(
                        colab_io.content_bytes(
                            colab_io.contents(session, run["remote_root"] + "/progress.json", content=True)
                        )
                    )
                    if progress.get("phase") not in ("done", "failed"):
                        return "work kernel inactive before completion"
        except Exception as error:
            save(run_dir / "watchdog.json", {"error": type(error).__name__, "at": now})
            if now - last_seen > 300:
                save(
                    run_dir / "watchdog-attention.json",
                    {
                        "notify": "Main",
                        "at": now,
                        "reason": "assignment checks unavailable; workload not confirmed dead",
                    },
                )
    return None


def supervise(args, client: Colab) -> dict:
    directory = args.run_dir
    with lock(STATE / "colab-ops" / (name(args.session) + ".lock"), blocking=False):
        if (directory / "run.json").exists():
            raise ValueError("attempt already exists; no blind relaunch")
        command = args.command[1:] if args.command[:1] == ["--"] else args.command
        root = name(command[command.index("--root") + 1])
        cap = (
            float(command[command.index("--hard-cap-minutes") + 1])
            if "--hard-cap-minutes" in command
            else None
        )
        run = {
            "session": args.session,
            "rate": args.rate,
            "budget_cu": args.budget_cu,
            "hard_cap_minutes": cap,
            "driver_sha": driver_sha(),
            "started_at": time.time(),
            "pid": os.getpid(),
            "process_start": colab_io.process_start(os.getpid()),
            "remote_root": "/content/" + root,
        }
        save(directory / "run.json", run)
        code = None
        reason = None
        with (directory / "driver.log").open("a") as log:
            child = subprocess.Popen(
                command,
                cwd=REPO,
                env={**os.environ, "DEW_OPS_RUN": str(directory)},
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        try:

            def terminated(signum, frame):
                raise KeyboardInterrupt(f"supervisor signal {signum}")

            signal.signal(signal.SIGTERM, terminated)
            signal.signal(signal.SIGINT, terminated)
            reason = watch(child, directory, run, client)
        except BaseException as error:
            reason = type(error).__name__
        finally:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
            code = child.returncode
            with lock(directory / "finalize.lock"):
                result = finalize(directory, client, reason, code)
        return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--session")
    parser.add_argument("--rate", type=float)
    parser.add_argument("--budget-cu", type=float)
    parser.add_argument("--finalize-only", action="store_true")
    parser.add_argument("--reconcile", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    client = Colab()
    if args.reconcile:
        reconcile(STATE / "colab-ops", client)
    elif args.finalize_only:
        if (args.run_dir / "run.json").exists():
            with lock(args.run_dir / "finalize.lock"):
                finalize(args.run_dir, client, "systemd stop-post")
    else:
        if not args.rate or not 0 < args.rate < float("inf"):
            parser.error("rate must be finite and positive")
        result = supervise(args, client)
        print(result)
        if result["driver_exit"] or result["reason"] or not result["job_results"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
