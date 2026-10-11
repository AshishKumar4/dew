"""One sequential batch on a VM, from the immutable host plan."""

import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import tomllib
from pathlib import Path

from tools.colab import colab_sync
from tools.colab.common import command, read, save


def run_logged(args: list[str], log: Path, *, timeout: float, cwd: Path, env: dict) -> int | str:
    """Bound the whole child process group, including profiling grandchildren."""
    with log.open("a") as output:
        process = subprocess.Popen(
            args, cwd=cwd, env=env, stdout=output, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            return process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            return f"timeout after {timeout} s"


def public_environment() -> dict[str, str]:
    private = {"MPLBACKEND", "WANDB_API_KEY", "WANDB_ENTITY", "WANDB_PROJECT"}
    return {key: value for key, value in os.environ.items() if key not in private}


def environment(venv: Path) -> dict[str, str]:
    return {
        **public_environment(),
        "PATH": f"{venv / 'bin'}:{os.environ['PATH']}",
        "VIRTUAL_ENV": str(venv),
        "PYTHONUNBUFFERED": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "JAX_PLATFORMS": "cuda,cpu",
    }


def join_parts(root: Path, parts: list[str], target: Path, expected: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".part")
    digest = hashlib.sha256()
    try:
        with temporary.open("wb") as destination:
            os.fchmod(destination.fileno(), 0o600)
            for name in parts:
                if Path(name).name != name:
                    raise ValueError("input part escapes staging")
                with (root / name).open("rb") as source:
                    while block := source.read(1024 * 1024):
                        digest.update(block)
                        destination.write(block)
        if digest.hexdigest() != expected:
            raise ValueError("staged input hash mismatch")
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def install(venv: Path, tree: Path, log: Path) -> None:
    declared = tomllib.loads((tree / "pyproject.toml").read_text())["project"]["optional-dependencies"]
    gpu = command(
        ["nvidia-smi", "--query-gpu=driver_version,compute_cap", "--format=csv,noheader"], timeout=30
    ).stdout
    driver, compute = gpu.strip().splitlines()[0].split(",")
    major = int(driver.split(".")[0])
    sm = tuple(map(int, compute.strip().split(".")))
    # Bootstrap precedes importing Dew. These are the floors documented by
    # dew.telemetry.devices.CUDA_BUILDS and JAX's installation guide.
    if major >= 580 and sm >= (7, 5):
        cuda = "cuda13"
    elif major >= 525 and sm >= (5, 2):
        cuda = "cuda12"
    else:
        raise ValueError(f"unsupported CUDA driver/SM: {gpu.strip()}")
    extras = [
        item
        for item in ("test", "av", "tfds", "metrics", "plots", "inference-clients", "vision", "profile")
        if item in declared
    ]
    extras.append(cuda)
    python = str(venv / "bin/python")
    commands = [
        ["uv", "venv", "-q", "--seed", "--python", "3.12", str(venv)],
        [
            "uv",
            "pip",
            "install",
            "-q",
            "--python",
            python,
            "torch",
            "torchvision",
            "--index-url",
            "https://download.pytorch.org/whl/cpu",
        ],
        ["uv", "pip", "install", "-q", "--python", python, "-e", f"{tree}[{','.join(extras)}]"],
    ]
    if (tree / "constraints.txt").is_file():
        commands[-1].extend(["-c", str(tree / "constraints.txt")])
    for args in commands:
        code = run_logged(args, log, timeout=1200, cwd=tree, env=public_environment())
        if code != 0:
            raise RuntimeError(f"environment setup failed: {code}")
    check = [
        python,
        "-c",
        "import jax; devices=jax.devices(); print(devices); "
        "assert any(device.platform=='gpu' for device in devices), 'JAX has no GPU backend'",
    ]
    if run_logged(check, log, timeout=120, cwd=tree, env=environment(venv)) != 0:
        raise RuntimeError("installed environment has no working JAX GPU backend")


def profile_preamble(tree: Path, venv: Path, output: Path) -> dict:
    receipt = output / "receipt.json"
    if receipt.exists():
        return read(receipt)
    if not (tree / "tools/gpu_profile_check.py").is_file():
        return {"attempted": False, "status": "skipped: tool absent"}
    output.mkdir(parents=True, exist_ok=True)
    status = {"attempted": True, "started_at": time.time()}
    try:
        code = run_logged(
            [str(venv / "bin/python"), "tools/gpu_profile_check.py", "--out", str(output)],
            output / "run.log",
            timeout=180,
            cwd=tree,
            env=environment(venv),
        )
        status.update(exit=code, broken=code != 0 or not (output / "check.json").is_file())
    except OSError as error:
        status.update(exit=type(error).__name__, broken=True)
    status["seconds"] = time.time() - status["started_at"]
    save(receipt, status)
    return status


def compilation_cache(job: dict, tree: Path, venv: Path) -> dict[str, str]:
    if job["compile_cache"] != "on":
        return {}
    roots = [
        tree / path for path in job["outputs"] if path.endswith("/") and not any(c in path for c in "*?[")
    ]
    if not roots:
        raise ValueError("compile_cache:on requires a declared output directory")
    versions = command(
        [
            str(venv / "bin/python"),
            "-c",
            "import importlib.metadata as m,json,sys; "
            "print(json.dumps([sys.version,m.version('jax'),m.version('jaxlib')]))",
        ],
        timeout=30,
    )
    gpu = command(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"], timeout=30
    ).stdout.strip()
    identity = {
        "commit": job["commit"],
        "gpu": gpu,
        "versions": json.loads(versions.stdout),
        "flags": os.environ.get("XLA_FLAGS", ""),
    }
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    directory = roots[0] / ".ops/compile-cache" / fingerprint
    save(directory / "runtime.json", identity)
    return {
        "JAX_COMPILATION_CACHE_DIR": str(directory / "cache"),
        "JAX_COMPILATION_CACHE_MAX_SIZE": str(512 * 1024 * 1024),
        "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS": "1",
    }


class Batch:
    def __init__(self, root: Path):
        self.root = root
        self.plan = read(root / "plan.json")
        self.repo = root / "repo"
        self.progress = {
            "phase": "setup",
            "jobs": {},
            "driver_sha": self.plan["driver_sha"],
            "started_at": time.time(),
        }

    def save(self, **fields) -> None:
        self.progress.update(fields, elapsed=time.time() - self.progress["started_at"])
        save(self.root / "progress.json", self.progress)

    def prepare(self) -> None:
        command(["git", "init", "-q", str(self.repo)], timeout=30)
        for sha in self.plan["fetch_commits"]:
            command(
                [
                    "git",
                    "-C",
                    str(self.repo),
                    "fetch",
                    "--depth=1",
                    "-q",
                    self.plan["origin"],
                    f"{sha}:refs/colab/{sha}",
                ],
                timeout=300,
            )
        if self.plan.get("bundle_parts"):
            bundle = self.root / "dew.bundle"
            join_parts(self.root, self.plan["bundle_parts"], bundle, self.plan["bundle_sha256"])
            command(
                [
                    "git",
                    "-C",
                    str(self.repo),
                    "fetch",
                    "-q",
                    str(bundle),
                    f"{self.plan['refs']}*:refs/colab/*",
                ],
                timeout=300,
            )
        for key, sha in self.plan["source_refs"].items():
            command(["git", "-C", str(self.repo), "update-ref", "refs/colab/" + key, sha], timeout=30)
        for target, source in self.plan["uploads"].items():
            join_parts(self.root, source["parts"], Path(target).expanduser(), source["sha256"])
        if self.plan.get("data_sources"):
            from tools.colab.data_cache import DataCache

            data_began = time.monotonic()
            verified_bytes = 0
            for source in self.plan["data_sources"]:
                cache = DataCache(source["endpoint"], Path("/content/.dew-data-read.token"))
                restored = cache.restore(source["prefix"], source["manifest_sha256"], Path(source["target"]),
                                         manifest_key=source.get("manifest_key"))
                verified_bytes += sum(record["bytes"] for record in restored["files"])
            self.save(data_restore={"seconds": time.monotonic() - data_began, "verified_physical_bytes": verified_bytes,
                                    "includes": "download or verified reuse, hashing, and declared assemblies"})
        if shutil.which("uv") is None:
            command([sys.executable, "-m", "pip", "install", "uv"], timeout=300, env=public_environment())

    def run_job(self, job: dict) -> dict:
        started = time.time()
        home = self.root / f"job-{job['index']}"
        home.mkdir()
        tree = home / "dew"
        command(
            ["git", "-C", str(self.repo), "worktree", "add", "-q", "--detach", str(tree), job["commit"]],
            timeout=60,
        )
        commit = command(["git", "-C", str(tree), "rev-parse", "HEAD"], timeout=30).stdout.strip()
        if commit != job["commit"]:
            raise RuntimeError("VM commit differs from the saved plan")
        status = {"commit": commit, "driver_sha": self.plan["driver_sha"], "stage_started_at": started}
        self.save_job(job, status)
        venv = self.root / "venvs" / commit
        if not (venv / ".ready").is_file():
            install(venv, tree, self.root / "setup.log")
            (venv / ".ready").touch()
        if not self.progress.get("profile_preamble", {}).get("attempted"):
            preamble = profile_preamble(tree, venv, self.root / "out/_preamble/profile-check")
            self.save(profile_preamble=preamble)
        if job.get("resume_parts"):
            packed = home / "resume.tar.gz"
            join_parts(self.root, job["resume_parts"], packed, job["resume_sha256"])
            colab_sync.restore(packed, tree, job["outputs"])
            packed.unlink()
        from tools.colab.credentials import environment as private_environment

        env = {**environment(venv), **compilation_cache(job, tree, venv), **private_environment(job)}
        # One venv per commit, but every job imports its own checked-out tree.
        env["PYTHONPATH"] = str(tree / "src")
        began = time.time()
        status.update(command_started_at=began)
        self.save_job(job, status)
        code = run_logged(
            ["bash", "-c", job["command"]],
            home / "run.log",
            cwd=tree,
            env=env,
            timeout=max(1800, 3 * 60 * job["minutes"]),
        )
        status.update(exit=code, seconds=round(time.time() - began), command_finished_at=time.time())
        self.save_job(job, status)
        colab_sync.pack_outputs(
            tree,
            job["outputs"],
            home / "outputs.tar.gz",
            job["artifact_policy"],
            job.get("artifact_excludes", []),
        )
        status.update(
            stage_seconds=round(time.time() - started), artifact_manifest="outputs.tar.gz.manifest.json"
        )
        self.save_job(job, status)
        return status

    def save_job(self, job: dict, status: dict) -> None:
        save(self.root / f"job-{job['index']}" / "status.json", status)
        self.progress["jobs"][str(job["index"])] = status
        self.save(phase="jobs")

    def run(self) -> None:
        try:
            self.prepare()
            for job in self.plan["jobs"]:
                self.run_job(job)
            self.save(phase="done")
        except Exception as error:
            self.save(phase="failed", error=str(error))
            raise
        finally:
            (self.root / "DONE").touch()


if __name__ == "__main__":
    Batch(Path(sys.argv[1])).run()
