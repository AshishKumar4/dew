"""Explicit, preflighted same-VM replacement at a cooperating child boundary."""

import argparse
import hashlib
import json
import subprocess
import sys
import time
import uuid
from pathlib import Path

from tools.colab import colab_batch, colab_io, preflight
from tools.colab.colab_new import Colab, Session
from tools.colab.common import REPO, read, redact, save


def rpc(client: Colab, session: Session, root: str, script: str) -> None:
    kernel = client.kernel(session)
    try:
        outputs = kernel.execute_code(script, timeout=60)
        errors = [item for item in outputs if item.get("output_type") == "error"]
        if errors:
            raise RuntimeError(redact(str(errors)))
    finally:
        kernel.stop(shutdown_kernel=True)


def replacement(plan: dict, job: dict, spec: dict) -> tuple[list, dict]:
    if colab_batch.exact_ref(REPO, spec["ref"]) != job["commit"]:
        raise ValueError("boundary replacement cannot change the pinned Dew commit")
    inputs = colab_batch.input_paths(spec)
    allowed = set(job["replaceable_scripts"])
    hashes = []
    for source, target in inputs:
        if target not in allowed or Path(target).suffix not in (".py", ".sh"):
            raise ValueError("replacement inputs must be originally declared scripts, never data/credentials")
        with source.open("rb") as stream:
            hashes.append((target, hashlib.file_digest(stream, "sha256").hexdigest()))
    request = {
        "commit": job["commit"],
        "hardware": plan["hardware"],
        "command": spec["command"],
        "runtime": spec.get("runtime_pins", "").split(),
    }
    receipt = preflight.verify(Path(spec["preflight"]).expanduser(), request, hashes)
    return inputs, {
        "command": spec["command"],
        "commit": job["commit"],
        "preflight_verified": True,
        "preflight_receipt": receipt,
        "script_hashes": dict(hashes),
    }


def control(
    run_dir: Path,
    index: int,
    after: str,
    action: str,
    spec: dict | None,
    client: Colab,
    cli: str,
    timeout: float = 1200,
) -> dict:
    run = read(run_dir / "run.json")
    driver = read(run_dir / "driver.json")
    plan = read(run_dir / "staging/plan.json")
    job = next(job for job in plan["jobs"] if job["index"] == index)
    if not job.get("boundary_control"):
        raise ValueError("this immutable job did not opt in to cooperative boundaries")
    if (run_dir / "result.json").exists():
        raise ValueError("controller attempt already ended")
    inputs, decision = replacement(plan, job, spec) if action == "replace" and spec else ([], {})
    if action == "replace" and not spec:
        raise ValueError("replace requires a preflighted specification")
    session = client.session(driver["endpoint"])
    root = run["remote_root"]
    status = driver.get("progress", {}).get("jobs", {}).get(str(index), {})
    generation = status.get("generation", 0)
    request_id = uuid.uuid4().hex
    directory = root + f"/job-{index}/boundary"
    files = []
    for offset, (source, target) in enumerate(inputs):
        staged = directory + f"/inputs/{request_id}-{offset}{source.suffix}"
        script = (
            f"from pathlib import Path\nPath({directory + '/inputs'!r}).mkdir(parents=True,exist_ok=True)\n"
        )
        rpc(client, session, root, script)
        subprocess.run([cli, "upload", "-s", run["session"], str(source), staged], check=True, timeout=120)
        files.append({"staged": staged, "target": target, "sha256": decision["script_hashes"][target]})
    script = (
        f"import sys\nsys.path.insert(0,{root!r})\nfrom tools.colab.boundary import request\n"
        f"from pathlib import Path\nrequest(Path({directory!r}),{after!r},{generation!r},{request_id!r})\n"
    )
    rpc(client, session, root, script)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if (run_dir / "result.json").exists():
            raise ValueError("session ended while awaiting boundary; no relaunch")
        session = client.session(driver["endpoint"])
        try:
            ack = json.loads(
                colab_io.content_bytes(
                    colab_io.contents(session, directory + "/snapshot-ack.json", content=True)
                )
            )
            if ack.get("request_id") == request_id and ack.get("generation") == generation:
                break
        except FileNotFoundError:
            save(
                run_dir / "boundary-requests" / (request_id + ".json"),
                {"request_id": request_id, "generation": generation,
                 "status": "waiting for acknowledged boundary", "at": time.time()},
            )
        time.sleep(1)
    else:
        raise TimeoutError("boundary was not acknowledged; live work remains owned by its supervisor")
    decision.update(
        request_id=request_id, generation=generation, action=action, snapshot_acknowledged=True, files=files
    )
    script = (
        f"import sys\nsys.path.insert(0,{root!r})\nfrom tools.colab.common import save\n"
        f"from pathlib import Path\nsave(Path({directory + '/decision.json'!r}),{decision!r})\n"
    )
    rpc(client, session, root, script)
    return {
        "request_id": request_id,
        "generation": generation,
        "action": action,
        "after": after,
        "snapshot_acknowledged": True,
        "session": run["session"],
        "budget_unchanged": run["budget_cu"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--job", type=int, required=True)
    parser.add_argument("--after", required=True)
    parser.add_argument("--action", choices=("replace", "continue", "stop"), required=True)
    parser.add_argument("--spec", type=Path)
    parser.add_argument("--timeout", type=float, default=1200)
    args = parser.parse_args()
    spec = read(args.spec) if args.spec else None
    result = control(
        args.run_dir,
        args.job,
        args.after,
        args.action,
        spec,
        Colab(),
        str(Path(sys.executable).with_name("colab")),
        args.timeout,
    )
    print(json.dumps(result))


if __name__ == "__main__":
    main()
