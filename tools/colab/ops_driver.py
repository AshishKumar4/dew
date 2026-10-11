"""Execute one approved packed queue under the independent supervisor."""

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path

import requests

from tools.colab import colab_batch, colab_io, colab_sync
from tools.colab.colab_new import Colab, Session
from tools.colab.common import STATE, command, name, read, redact, save


class TransportError(RuntimeError):
    """A control channel failed without evidence that the workload stopped."""


class Driver:
    def __init__(self, run_dir: Path, client: Colab, cli: str):
        self.run_dir = run_dir
        self.client = client
        self.cli = cli
        self.state = read(run_dir / "driver.json") or {"results": [], "started_at": time.time()}
        self.guard = threading.RLock()
        self.runtime = None

    def record(self, **fields) -> None:
        with self.guard:
            self.state.update(fields, heartbeat=time.time())
            save(self.run_dir / "driver.json", self.state)

    def control(self, session: Session, script: str, timeout: int = 300, *, live_read: bool = False) -> str:
        runtime = self.client.kernel(session)
        self.record(action="control", action_deadline=time.time() + timeout + 30)
        try:
            try:
                outputs = runtime.execute_code(script, timeout=timeout)
            except Exception as error:
                raise TransportError(type(error).__name__) from error
            errors = [item for item in outputs if item.get("output_type") == "error"]
            if errors:
                if live_read:
                    names = [item.get("ename", "unknown") for item in errors]
                    benign = all(name in ("FileNotFoundError", "NotADirectoryError") for name in names)
                    self.record(collection_error_names=names, collection_needs_attention=not benign)
                    raise TransportError(
                        "live output collection failed; re-list on next cycle, not a workload exit"
                    )
                raise RuntimeError(redact(str(errors)))
            return "".join(item.get("text", "") for item in outputs)
        finally:
            try:
                runtime.stop(shutdown_kernel=True)
            except Exception as error:
                self.record(control_cleanup_error=type(error).__name__)
            self.record(action="poll", action_deadline=None)

    def upload(self, session_name: str, source: Path, remote: str, *, private: bool = False) -> None:
        self.record(action="upload", action_deadline=time.time() + 1230)
        try:
            command([self.cli, "upload", "-s", session_name, str(source), remote], timeout=1200)
        except RuntimeError:
            if private:
                raise RuntimeError("private credential upload failed; output withheld") from None
            raise
        self.record(action="poll", action_deadline=None)

    def deploy(self, session: Session, plan: dict, work: Path, remote: str) -> None:
        # An interrupted handshake can be retried before any START. The marker
        # makes directory creation idempotent but refuses another live batch.
        script = (
            f"from pathlib import Path\np=Path({remote!r})\n"
            "assert not (p/'STARTED').exists(), 'batch already started'\np.mkdir(exist_ok=True)\n"
        )
        for attempt in range(3):
            try:
                self.control(session, script)
                break
            except RuntimeError:
                if attempt == 2:
                    raise
        for file in sorted(work.iterdir()):
            if file.suffix == ".part" or file.name in ("runner.tar.gz", "plan.json"):
                self.upload(
                    plan["device"],
                    file,
                    remote + "/" + file.name,
                    private=file.name in plan.get("private_parts", []),
                )

    def start(self, session: Session, plan: dict, remote: str) -> None:
        ready = threading.Event()
        ended = threading.Event()
        self.runtime = self.client.kernel(
            session, on_kernel_started=lambda kernel: self.record(work_kernel=kernel)
        )
        runtime = self.runtime
        script = (
            "import hashlib,subprocess,tarfile\nfrom pathlib import Path\n"
            f"root=Path({remote!r})\narchive=root/'runner.tar.gz'\n"
            f"assert hashlib.sha256(archive.read_bytes()).hexdigest()=={plan['runner_sha256']!r}\n"
            "with tarfile.open(archive) as packed: packed.extractall(root, filter='data')\n"
            "(root/'STARTED').touch(exist_ok=False)\n"
            "process=subprocess.Popen(['python3','-m','tools.colab.colab_batch_vm',str(root)],cwd=root,"
            "stdout=open(root/'setup.log','a'),stderr=subprocess.STDOUT)\n"
            "print('OPS_STARTED',process.pid,flush=True)\n"
            "print('OPS_EXIT',process.wait(),flush=True)\n"
        )

        def output(item):
            if "OPS_STARTED" in str(item.get("text", "")):
                self.record(work_acknowledged_at=time.time())
                ready.set()

        def execute():
            try:
                outputs = runtime.execute_code(script, timeout=24 * 3600, output_hook=output)
                self.record(
                    work_execution_ended_at=time.time(),
                    kernel_errors=[item for item in outputs if item.get("output_type") == "error"],
                )
            except Exception as error:
                self.record(work_channel_error=type(error).__name__)
            finally:
                ended.set()

        self.record(action="START", action_deadline=time.time() + 180)
        threading.Thread(target=execute, daemon=True).start()
        deadline = time.monotonic() + 120
        while not ready.wait(0.2):
            if ended.is_set() or time.monotonic() >= deadline:
                raise RuntimeError("START acknowledgement lost; no ambiguous retry")
        self.record(action="poll", action_deadline=None)

    def sync(self, session: Session, remote: str, job: dict) -> Path:
        cache = STATE / "colab-resume" / job["resume_key"]
        known = read(cache / "manifest.json")
        home = remote + f"/job-{job['index']}"
        script = (
            f"import sys,json\nsys.path.insert(0,{remote!r})\n"
            "from tools.colab.colab_sync import snapshot\n"
            f"manifest,count=snapshot({home + '/dew'!r},{job['outputs']!r},{known!r},"
            f"{home + '/sync.tar.gz'!r},policy={job['artifact_policy']!r},"
            f"excludes={job.get('artifact_excludes', [])!r})\n"
            "print('SYNC',json.dumps([manifest,count]))\n"
        )
        output = self.control(session, script, timeout=600, live_read=True)
        row = next(line[5:] for line in output.splitlines() if line.startswith("SYNC "))
        current, changed = json.loads(row)
        if changed:
            archive = cache / "delta.tar.gz"
            self.record(action="collect", action_deadline=time.time() + 3630)
            colab_io.download(session, home + "/sync.tar.gz", archive)
            colab_sync.apply_snapshot(archive, cache / "outputs", current, known, cache / "manifest.json")
            archive.unlink()
        else:
            (cache / "outputs").mkdir(parents=True, exist_ok=True)
            save(cache / "manifest.json", current)
        self.record(action="poll", action_deadline=None)
        return cache

    def collect(self, session: Session, remote: str, plan: dict, job: dict, status: dict) -> None:
        cache = self.sync(session, remote, job)
        target = colab_batch.artifact_path(plan, self.run_dir.name, job)
        target.mkdir(parents=True, exist_ok=True)
        for filename in ("status.json", "run.log"):
            colab_io.download(session, remote + f"/job-{job['index']}/" + filename, target / filename)
        colab_sync.publish_retained(cache, target / "outputs")
        result = {
            "index": job["index"],
            "lane": job["lane"],
            "status": f"exit {status['exit']}",
            "observed_exit": status["exit"],
            "results": str(target),
            "commit": job["commit"],
            "driver_sha": plan["driver_sha"],
            "seconds": status["seconds"],
            "stage_seconds": status["stage_seconds"],
            "finished_at": time.time(),
        }
        colab_batch.finish(job["entry"], result)
        self.record(results=[*self.state["results"], result])

    def poll(self, endpoint: str, remote: str, plan: dict) -> None:
        collected = set()
        synced = {}
        failures = 0
        while True:
            try:
                finished = self.poll_once(endpoint, remote, plan, collected, synced)
                failures = 0
                if finished:
                    return
                time.sleep(30)
            except (requests.RequestException, TransportError, ConnectionError, TimeoutError) as error:
                failures += 1
                delay = min(30, 2 ** min(failures, 5))
                self.record(
                    transport_error=type(error).__name__,
                    consecutive_transport_failures=failures,
                    retry_seconds=delay,
                    action="poll",
                    action_deadline=None,
                )
                time.sleep(delay)

    def collect_action(self, job: dict, operation, *args) -> None:
        failures = self.state.get("collection_failures", {})
        key = str(job["index"])
        if failures.get(key, {}).get("needs_attention"):
            return
        try:
            operation(*args)
            failures[key] = {"count": 0, "needs_attention": False}
            self.record(collection_failures=failures)
        except (OSError, ValueError, RuntimeError) as error:
            count = failures.get(key, {}).get("count", 0) + 1
            failures[key] = {"count": count, "error": type(error).__name__, "needs_attention": count >= 3}
            self.record(collection_failures=failures)
            self.record(collection_error=type(error).__name__, collection_needs_attention=True)
            if count >= 3:
                return
            raise TransportError("artifact collection failed; workload liveness is independent") from error

    def failed_collections(self, plan: dict, progress: dict) -> None:
        failures = self.state.get("collection_failures", {})
        for job in plan["jobs"]:
            if not failures.get(str(job["index"]), {}).get("needs_attention"):
                continue
            status = progress.get("jobs", {}).get(str(job["index"]), {})
            result = {"index": job["index"], "lane": job["lane"], "commit": job["commit"],
                      "driver_sha": plan["driver_sha"], "status": "collection failed: needs attention",
                      "observed_exit": status.get("exit"), "collection_failed": True,
                      "results": str(self.run_dir), "seconds": status.get("seconds", 0),
                      "stage_seconds": status.get("stage_seconds", 0), "finished_at": time.time()}
            colab_batch.finish(job["entry"], result)
            self.record(results=[*self.state["results"], result])

    def poll_once(self, endpoint: str, remote: str, plan: dict, collected: set, synced: dict) -> bool:
        progress: dict
        try:
            session = self.client.session(endpoint)
        except LookupError:
            # Independent supervisor checks decide whether absence is real.
            raise TransportError("assignment lookup absent; awaiting supervisor confirmation") from None
        try:
            progress = json.loads(
                colab_io.content_bytes(colab_io.contents(session, remote + "/progress.json", content=True))
            )
        except FileNotFoundError:
            progress = {"phase": "starting"}
        self.record(progress=progress)
        for job in plan["jobs"]:
            index = job["index"]
            status = progress.get("jobs", {}).get(str(index), {})
            if index in collected:
                continue
            if status.get("artifact_manifest"):
                self.collect_action(job, self.collect, session, remote, plan, job, status)
                collected.add(index)
            elif status and time.monotonic() - synced.get(index, 0) >= 600:
                self.collect_action(job, self.sync, session, remote, job)
                synced[index] = time.monotonic()
        if progress.get("phase") in ("done", "failed"):
            self.failed_collections(plan, progress)
            self.preamble(session, remote, plan, progress)
            if progress["phase"] == "failed":
                raise RuntimeError("VM batch failed: " + progress.get("error", "unknown cause"))
            return True
        return False

    def preamble(self, session: Session, remote: str, plan: dict, progress: dict) -> None:
        receipt = progress.get("profile_preamble", {"attempted": False})
        target = self.run_dir / "out/_preamble/profile-check"
        save(target / "receipt.json", {**receipt, "driver_sha": plan["driver_sha"]})
        if receipt.get("attempted"):
            for filename in ("check.json", "run.log"):
                try:
                    colab_io.download(
                        session, remote + "/out/_preamble/profile-check/" + filename, target / filename
                    )
                except FileNotFoundError:
                    receipt["missing_" + filename] = True
        self.record(profile_preamble=receipt)
        for job in self.state["results"]:
            save(
                Path(job["results"]) / "session.json",
                {
                    "driver_sha": plan["driver_sha"],
                    "profile_preamble": receipt,
                    "profile_artifacts": str(target),
                },
            )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("device")
    result.add_argument("--commit", required=True)
    result.add_argument("--only", required=True)
    result.add_argument("--root", required=True, type=name)
    result.add_argument("--hardware", choices=("gpu:T4", "gpu:L4", "gpu:A100", "gpu:G4"), required=True)
    result.add_argument("--high-mem", action="store_true")
    result.add_argument("--order", default="")
    result.add_argument("--reuse-staging", action="store_true")
    result.add_argument("--dry-run", action="store_true")
    result.add_argument("--hard-cap-minutes", type=float)
    return result


def main() -> None:
    args = parser().parse_args()
    chosen = colab_batch.select(args.device, args.only, args.hardware, args.order.split(","))
    if not chosen:
        raise SystemExit("no approved, unheld entries selected")
    plan = colab_batch.plan_jobs(chosen, args.device, args.hardware, args.commit)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "identity": colab_batch.identity(plan),
                    "expected_cu": sum(float(job["economics"]["expected_cu"]) for job in plan["jobs"]),
                }
            )
        )
        return
    run_dir = Path(os.environ["DEW_OPS_RUN"])
    work = run_dir / "staging"
    client = Colab()
    driver = Driver(
        run_dir, client, os.environ.get("DEW_COLAB_CLI", str(Path(sys.executable).with_name("colab")))
    )
    driver.record(driver_sha=plan["driver_sha"], action="staging", action_deadline=time.time() + 3600)
    if args.reuse_staging:
        plan = colab_batch.staged(plan, work)
    else:
        colab_batch.stage(plan, work)
    driver.record(driver_sha=plan["driver_sha"], action="allocate", action_deadline=time.time() + 1830)
    endpoint = client.allocate(args.device, args.hardware, args.high_mem, run_dir / "allocation.json")
    driver.record(endpoint=endpoint, assigned_at=time.time(), action="poll", action_deadline=None)
    session = client.session(endpoint)
    remote = "/content/" + args.root
    driver.deploy(session, plan, work, remote)
    driver.start(session, plan, remote)
    driver.poll(endpoint, remote, plan)


if __name__ == "__main__":
    main()
