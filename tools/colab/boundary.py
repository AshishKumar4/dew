"""Cooperative experiment boundaries, owned by one unchanged VM job budget."""

import hashlib
import os
import time
from pathlib import Path

from tools.colab.common import read, save

REPLACED = 79


def checkpoint(name: str) -> None:
    """Call only after a child experiment and its output writers are quiescent.

    With no controller request this is a no-op. A requested boundary pauses
    before the next experiment; only the supervising runner may continue it.
    """
    folder = os.environ.get("DEW_COLAB_BOUNDARY_DIR")
    if not folder:
        return
    directory = Path(folder)
    generation = int(os.environ["DEW_COLAB_GENERATION"])
    request = read(directory / "request.json")
    if request.get("after") != name or request.get("generation") != generation:
        return
    if request.get("expires_at", 0) <= time.time():
        save(directory / "ready.json", {"name": name, "generation": generation,
             "request_id": request["request_id"], "state": "expired before boundary; continued", "at": time.time()})
        return
    ready = {
        "name": name,
        "generation": generation,
        "request_id": request["request_id"],
        "pid": os.getpid(),
        "ready_at": time.time(),
        "state": "quiescent",
        "decision_deadline": min(request["expires_at"], time.time() + min(180, request.get("decision_seconds", 180))),
    }
    save(directory / "ready.json", ready)
    while True:
        if time.time() >= ready["decision_deadline"]:
            ready.update(state="boundary decision timed out; continued", at=time.time())
            save(directory / "ready.json", ready)
            return
        decision = read(directory / "decision.json")
        if decision.get("request_id") == ready["request_id"] and decision.get("generation") == generation:
            if decision.get("action") == "continue":
                return
            if decision.get("action") in ("replace", "stop"):
                if not decision.get("snapshot_acknowledged"):
                    raise ValueError("boundary cannot exit before its outputs are acknowledged")
                raise SystemExit(REPLACED)
            raise ValueError("unknown boundary decision")
        time.sleep(0.1)


def validate_replacement(job: dict, decision: dict, directory: Path, generation: int) -> None:
    if decision.get("generation") != generation or decision.get("commit") != job["commit"]:
        raise ValueError("replacement changed the job generation/commit")
    if not decision.get("preflight_verified"):
        raise ValueError("replacement must be preflighted by its controller")
    targets = set(job.get("replaceable_scripts", []))
    for file in decision.get("files", []):
        staged = Path(file["staged"])
        target = file["target"]
        if (
            ".." in staged.parts
            or not staged.resolve().is_relative_to((directory / "inputs").resolve())
            or staged.is_symlink()
        ):
            raise ValueError("replacement file escapes owned boundary inputs")
        if target not in targets or Path(target).suffix not in (".py", ".sh"):
            raise ValueError("replacement may change only declared script inputs")
        with staged.open("rb") as source:
            if hashlib.file_digest(source, "sha256").hexdigest() != file["sha256"]:
                raise ValueError("replacement script checksum mismatch")


def run(job: dict, home: Path, tree: Path, env: dict, timeout: float, run_logged, update) -> int | str:
    """Replace only a cooperating driver; never heuristically kill active work."""
    directory = home / "boundary"
    directory.mkdir(exist_ok=True)
    began = time.monotonic()
    generation = 0
    command = job["command"]
    lineage = []
    while True:
        remaining = timeout - (time.monotonic() - began)
        if remaining <= 0:
            return "job timeout, including all boundary generations"
        child_env = {**env, "DEW_COLAB_BOUNDARY_DIR": str(directory), "DEW_COLAB_GENERATION": str(generation)}
        code = run_logged(
            ["bash", "-c", command], home / "run.log", cwd=tree, env=child_env, timeout=remaining
        )
        observed = read(directory / "ready.json")
        if "continued" in observed.get("state", ""):
            update(boundary_timeout=observed)
        if code != REPLACED:
            return code
        ready, decision = read(directory / "ready.json"), read(directory / "decision.json")
        if ready.get("request_id") != decision.get("request_id") or ready.get("generation") != generation:
            raise ValueError("boundary completion is not owned by this generation")
        if not decision.get("snapshot_acknowledged"):
            raise ValueError("boundary replacement requires an acknowledged output snapshot")
        if decision.get("action") == "stop":
            update(boundary_stop=ready, full_job_exit_observed=False)
            return "stopped at acknowledged experiment boundary"
        if decision.get("action") != "replace":
            raise ValueError("quiescent generation received an invalid replacement action")
        validate_replacement(job, decision, directory, generation)
        old = hashlib.sha256(command.encode()).hexdigest()
        for file in decision.get("files", []):
            os.replace(file["staged"], file["target"])
        command = decision["command"]
        lineage.append(
            {
                "at": time.time(),
                "after": ready["name"],
                "generation": generation,
                "old_command_sha256": old,
                "new_command_sha256": hashlib.sha256(command.encode()).hexdigest(),
                "files": [
                    {"target": file["target"], "sha256": file["sha256"]} for file in decision.get("files", [])
                ],
            }
        )
        generation += 1
        update(boundary_lineage=lineage, generation=generation)
        (directory / "ready.json").unlink()
        (directory / "decision.json").unlink()


def request(directory: Path, name: str, generation: int, request_id: str, *,
            expires_at: float | None = None, decision_seconds: float = 180) -> None:
    current = read(directory / "request.json")
    if current and current.get("generation") == generation and current.get("expires_at", 0) > time.time():
        raise ValueError("this generation already has a boundary request")
    save(directory / "request.json", {"after": name, "generation": generation, "request_id": request_id,
                                      "expires_at": expires_at if expires_at is not None else time.time() + 1200,
                                      "decision_seconds": min(180, max(0.1, decision_seconds))})


def decide(directory: Path, decision: dict) -> None:
    ready, requested = read(directory / "ready.json"), read(directory / "request.json")
    if (ready.get("request_id") != decision.get("request_id")
            or ready.get("generation") != decision.get("generation") or ready.get("state") != "quiescent"):
        raise ValueError("decision is not owned by the current quiescent boundary")
    if time.time() >= min(ready["decision_deadline"], requested["expires_at"]):
        raise ValueError("boundary request expired; refusing a late decision")
    save(directory / "decision.json", {**decision, "decided_at": time.time()})
