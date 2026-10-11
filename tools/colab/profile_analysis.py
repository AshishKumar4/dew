"""Shared VM-side profile finalization; return small breakdowns, not raw dumps."""

import contextlib
import os
import subprocess
import time
import uuid
from pathlib import Path

from tools.colab import colab_sync
from tools.colab.common import command, read, save

TOOL_REF = "b420100f20856822ae19d7f314a9a22ba24e26df"


def memory_bound() -> int:
    total = next(int(line.split()[1]) * 1024 for line in Path("/proc/meminfo").read_text().splitlines()
                 if line.startswith("MemTotal:"))
    for file in (Path("/sys/fs/cgroup/memory.max"), Path("/sys/fs/cgroup/memory/memory.limit_in_bytes")):
        if file.is_file():
            value = file.read_text().strip()
            if value.isdigit() and int(value) > 0:
                total = min(total, int(value))
    return max(1, total // 4)


@contextlib.contextmanager
def memory_scope(limit: int):
    group = Path("/sys/fs/cgroup") / ("dew-profile-" + uuid.uuid4().hex)
    group.mkdir()
    try:
        (group / "memory.max").write_text(str(limit))
        (group / "memory.swap.max").write_text("0")
        (group / "memory.oom.group").write_text("1")
        yield group
    finally:
        group.rmdir()


def profile_settings(value):
    if isinstance(value, dict):
        profile = value.get("profile")
        if isinstance(profile, dict) and type(profile.get("steps")) is int:
            yield profile
        for child in value.values():
            yield from profile_settings(child)
    elif isinstance(value, list):
        for child in value:
            yield from profile_settings(child)


def discover(job: dict, tree: Path) -> list[dict]:
    files = colab_sync.output_files(
        tree, job["outputs"], "measurements", job.get("artifact_excludes", []), include_raw=True
    )
    captures: set[Path] = {
        file.parent.parent
        for file in files.values()
        if file.name == "hlo_stats.json" and file.parent.name == "reports"
    }
    raw = [file for file in files.values() if file.name.endswith(".xplane.pb")]
    for file in raw:
        candidate = next((parent for parent in file.parents if (parent / "manifest.json").is_file()), None)
        if candidate is not None:
            captures.add(candidate)
    settings = []
    for file in files.values():
        try:
            if file.name == "run.json" and file.stat().st_size < 2 * 1024 * 1024:
                settings.extend(profile_settings(read(file)))
        except (FileNotFoundError, ValueError):
            continue
    found = []
    for capture in sorted(captures):
        entries = [
            file
            for file in files.values()
            if file.name == "entry.json" and capture.is_relative_to(file.parent)
        ]
        entry = max(entries, key=lambda file: len(file.parts), default=None)
        archives = [
            file for file in files.values() if file.name == "hlo.tgz" and capture.is_relative_to(file.parent)
        ]
        hlo = max(archives, key=lambda file: len(file.parts), default=None)
        if hlo is None:
            candidates = {file.parent for file in files.values() if file.name.endswith("after_optimizations.txt")}
            matching = [path for path in candidates if capture.is_relative_to(path.parent.parent)]
            hlo = matching[0] if len(matching) == 1 else None
        matched = [
            profile
            for profile in settings
            if profile.get("directory") and capture.is_relative_to(Path(profile["directory"]))
        ]
        steps = matched[0]["steps"] if matched else job.get("profile_steps")
        found.append({"capture": capture, "entry": entry, "hlo": hlo, "steps": steps})
    return found


def tool_tree(tree: Path, repo: Path, root: Path, origin: str, ref: str) -> Path:
    if (tree / "tools/attribute_step.py").is_file():
        return tree
    target = root / "profile-tools" / ref
    if not target.exists():
        command(
            ["git", "-C", str(repo), "fetch", "--depth=1", "-q", origin, f"{ref}:refs/colab-analysis/{ref}"],
            timeout=120,
        )
        command(["git", "-C", str(repo), "worktree", "add", "-q", "--detach", str(target), ref], timeout=60)
    if not (target / "tools/attribute_step.py").is_file():
        raise ValueError("pinned analysis tree lacks tools/attribute_step.py")
    return target


def analyze(
    job: dict, tree: Path, venv: Path, repo: Path, root: Path, origin: str, env: dict, run_logged
) -> dict:
    try:
        captures = discover(job, tree)
    except (OSError, ValueError) as error:
        return {"attempted": True, "complete": False, "error": type(error).__name__, "captures": []}
    if not captures:
        return {"attempted": False, "reason": "no native device profile"}
    result = {
        "attempted": True,
        "started_at": time.time(),
        "captures": [],
        "analysis_scope": "on original VM; no raw capture transferred",
    }
    limit = memory_bound()
    for index, capture in enumerate(captures):
        output = capture["capture"].parent / "analysis" / f"capture-{index}"
        output.mkdir(parents=True, exist_ok=True)
        receipt = {"capture": str(capture["capture"]), "output": str(output), "steps": capture["steps"]}
        receipt["hlo"] = str(capture["hlo"]) if capture["hlo"] else None
        receipt.update(memory_limit_bytes=limit, memory_policy="cgroup resident memory, 25% of VM bound", module="jit_step")
        result["captures"].append(receipt)
        try:
            tools = tool_tree(tree, repo, root, origin, job.get("profile_analysis_ref", TOOL_REF))
            args = [
                str(venv / "bin/python"),
                "tools/attribute_step.py",
                "--trace",
                str(capture["capture"]),
                "--out",
                str(output / "attribution.json"),
            ]
            if capture["hlo"]:
                args.extend(["--hlo", str(capture["hlo"])])
            if capture["steps"]:
                args.extend(["--steps", str(capture["steps"])])
                if capture["entry"]:
                    args.extend(["--entry", str(capture["entry"])])
            else:
                receipt["normalization"] = "capture total; steps unknown, not a per-step/MFU claim"
            analysis_env = {
                **env,
                "JAX_PLATFORMS": "cpu",
                "CUDA_VISIBLE_DEVICES": "",
                "PYTHONPATH": str(tools / "src"),
            }
            args.extend(["--module", receipt["module"]])
            with memory_scope(limit) as group:
                receipt["exit"] = run_logged(args, output / "analysis.log", timeout=180, cwd=tools,
                                             env=analysis_env, memory_group=group)
                events = dict(line.split() for line in (group / "memory.events").read_text().splitlines())
                if int(events.get("oom_kill", "0")) or int(events.get("oom_group_kill", "0")):
                    receipt["state"] = "analysis exceeded its memory bound"
            receipt["tool_commit"] = command(
                ["git", "-C", str(tools), "rev-parse", "HEAD"], timeout=30
            ).stdout.strip()
            receipt["complete"] = receipt["exit"] == 0 and (output / "attribution.json").is_file()
            if receipt["complete"]:
                receipt["module_number"] = read(output / "attribution.json").get("memory", {}).get("module_number")
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
            receipt.update(exit=type(error).__name__, complete=False, state="bounded analysis unavailable or failed")
        save(output / "receipt.json", receipt)
    result.update(
        seconds=time.time() - result["started_at"],
        complete=all(capture["complete"] for capture in result["captures"]),
    )
    return result
