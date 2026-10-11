"""Queue parsing, exact experiment identities and immutable batch staging."""

import hashlib
import os
import re
import subprocess
import tarfile
from pathlib import Path
from urllib.parse import urlparse

from tools.colab import colab_sync, preflight
from tools.colab.common import HERE, PART, REPO, STATE, driver_sha, git, lock, name, read, save

QUEUE = STATE / "colab-queue.md"
FIELD = re.compile(r"^(?:- |  )([a-z_]+):\s?(.*)$")


def queue_lock():
    return lock(QUEUE.with_suffix(".lock"))


def sections() -> tuple[str, str, str]:
    head, _, rest = QUEUE.read_text().partition("## Pending\n")
    pending, _, done = rest.partition("## Done\n")
    return head, pending, done


def write_queue(text: str) -> None:
    temporary = QUEUE.with_name(QUEUE.name + ".tmp")
    temporary.write_text(text)
    os.replace(temporary, QUEUE)


def entries(section: str) -> list[dict]:
    found = []
    current = None
    key = None
    for line in section.splitlines(keepends=True):
        if line.startswith("- "):
            current, key = {"text": "", "fields": {}}, None
            found.append(current)
        elif current is None:
            continue
        elif line.strip() and not line.startswith("  "):
            current = None
            continue
        current["text"] += line
        match = FIELD.match(line.rstrip("\n"))
        if match and not line.startswith("    "):
            key, value = match.groups()
            current["fields"][key] = value.strip()
        elif key and (line.strip() or key == "command"):
            value = line[4:].rstrip("\r\n") if key == "command" and line.startswith("    ") else line.strip()
            current["fields"][key] += "\n" + value
    for entry in found:
        fields = entry["fields"]
        for field in ("device", "ref", "lane", "minutes"):
            fields[field] = (fields.get(field, "").split(" #")[0].split() or [""])[0]
        shell = fields.get("command", "")
        fields["command"] = shell[1:].strip() if shell.startswith(("|", ">")) else shell
        fields["outputs"] = re.findall(
            r"[\w./-]*\*[\w./*-]*|[\w./-]+\.\w+|[\w./-]+/(?=\s|,|$)",
            first_sentence(fields.get("outputs", "").split(" #")[0]),
        )
    return found


def first_sentence(text: str) -> str:
    return re.split(r"\.\s+(?=[A-Z])", re.sub(r"\([^)]*\)", " ", text))[0]


def input_paths(fields: dict) -> list[tuple[Path, str]]:
    sources, _, destinations = first_sentence(fields.get("inputs", "")).partition(" to ")
    pattern = r"(?<![\w.])~?/[\w./+-]+"
    remote = re.findall(pattern, destinations)
    pairs = []
    for value in re.findall(pattern, sources):
        source = Path(value).expanduser()
        if not source.exists():
            raise ValueError(f"missing declared input: {source}")
        if not remote:
            raise ValueError("declared inputs need a VM destination")
        for file in sorted(source.rglob("*")) if source.is_dir() else [source]:
            if file.is_file():
                suffix = file.relative_to(source) if source.is_dir() else file.name
                pairs.append((file, remote[0].rstrip("/.") + "/" + str(suffix)))
    for value in re.findall(r"~/[\w./+-]+", fields["command"]):
        path = Path(value).expanduser()
        if path.is_file():
            pairs.append((path, value))
    return pairs


def entry_key(entry: dict) -> tuple:
    fields = entry["fields"]
    return fields.get("device"), fields.get("ref"), entry["text"].splitlines()[0], fields.get("command")


def finish(entry: dict, result: dict) -> None:
    with queue_lock():
        head, pending, done = sections()
        current = next((item for item in entries(pending) if entry_key(item) == entry_key(entry)), None)
        if current is None:
            return
        pending = pending.replace(current["text"], "", 1)
        receipt = "".join(f"  {key}: {value}\n" for key, value in result.items())
        write_queue(head + "## Pending\n" + pending + "## Done\n" + current["text"] + receipt + "\n" + done)


def exact_ref(repo: Path, ref: str) -> str:
    return git(repo, "rev-parse", f"{ref}^{{commit}}")


def artifact_path(plan: dict, attempt: str, job: dict) -> Path:
    return (
        STATE
        / "integration"
        / plan["base"]
        / name(job["lane"])
        / f"{name(plan['device'])}-{name(attempt)}-job-{job['index']}"
    )


def hold_unreported(plan: dict, completed: set[int], reason: str) -> None:
    with queue_lock():
        head, pending, done = sections()
        for job in plan.get("jobs", []):
            if job["index"] in completed:
                continue
            current = next(
                (item for item in entries(pending) if entry_key(item) == entry_key(job["entry"])), None
            )
            if current and not current["fields"].get("hold"):
                pending = pending.replace(
                    current["text"],
                    current["text"].rstrip() + "\n  hold: " + reason + "; owner approval required\n\n",
                    1,
                )
        write_queue(head + "## Pending\n" + pending + "## Done\n" + done)


def select(device: str, selector: str, hardware: str, order: list[str]) -> list[dict]:
    chosen = [
        entry
        for entry in entries(sections()[1])
        if entry["fields"].get("device") == device
        and re.search(selector, entry["text"].splitlines()[0], re.I)
        and not entry["fields"].get("hold")
    ]
    for entry in chosen:
        fields = entry["fields"]
        if fields.get("hardware", hardware) != hardware:
            raise ValueError("selected job requests different hardware")
        required = ("ref", "gpu_reason", "expected_cu", "decision", "memory_estimate", "outputs", "preflight")
        for field in required:
            if not fields.get(field):
                raise ValueError(f"GPU request lacks {field}: {entry['text'].splitlines()[0]}")
        estimate = float(fields["expected_cu"])
        if not 0 < estimate < float("inf"):
            raise ValueError("expected_cu must be finite and positive")
        if fields.get("artifact_policy", "all") not in ("all", "measurements"):
            raise ValueError("invalid artifact_policy")
        if any(Path(pattern).is_absolute() for pattern in fields.get("artifact_excludes", "").split()):
            raise ValueError("artifact exclusions must be archive-relative")
        if fields.get("compile_cache", "off") not in ("on", "off"):
            raise ValueError("compile_cache must be on or off")
        if fields.get("wandb", "off") not in ("on", "off"):
            raise ValueError("wandb must be on or off")
        if fields.get("raw_profiles", "off") not in ("on", "off"):
            raise ValueError("raw_profiles must be on or off")
        if fields.get("profile_steps") and int(fields["profile_steps"]) <= 0:
            raise ValueError("profile_steps must be positive")
    return sorted(
        chosen,
        key=lambda entry: next(
            (i for i, part in enumerate(order) if part.lower() in entry["text"].splitlines()[0].lower()),
            len(order),
        ),
    )


def identity(plan: dict) -> dict:
    return {
        "driver_sha": plan["driver_sha"],
        "base": plan["base"],
        "hardware": plan["hardware"],
        "jobs": [
            {
                "artifact_excludes": job.get("artifact_excludes", []),
                "data_sources": job.get("data_sources", []),
                "wandb": job.get("wandb", False),
                "raw_profiles": job.get("raw_profiles", False),
                "profile_steps": job.get("profile_steps"),
                "profile_analysis_ref": job.get(
                    "profile_analysis_ref", "b420100f20856822ae19d7f314a9a22ba24e26df"
                ),
                **{
                    key: job[key]
                    for key in ("commit", "command", "outputs", "artifact_policy", "ref", "compile_cache")
                },
            }
            for job in plan["jobs"]
        ],
    }


def validate_reuse(saved: dict, requested: dict) -> None:
    if identity(saved) != identity(requested):
        raise ValueError("staged driver/base/job identity changed; refusing reuse")


def plan_jobs(chosen: list[dict], device: str, hardware: str, base: str, repo: Path = REPO) -> dict:
    jobs = []
    for index, entry in enumerate(chosen, 1):
        fields = entry["fields"]
        job = {key: fields.get(key, "") for key in ("lane", "command", "outputs", "ref")}
        job.update(
            index=index,
            commit=exact_ref(repo, fields["ref"]),
            minutes=int(fields.get("minutes") or 30),
            artifact_policy=fields.get("artifact_policy", "all"),
            artifact_excludes=fields.get("artifact_excludes", "").split(),
            compile_cache=fields.get("compile_cache", "off"),
            wandb=fields.get("wandb", "off") == "on",
            raw_profiles=fields.get("raw_profiles", "off") == "on",
            profile_steps=int(fields["profile_steps"]) if fields.get("profile_steps") else None,
            profile_analysis_ref=fields.get(
                "profile_analysis_ref", "b420100f20856822ae19d7f314a9a22ba24e26df"
            ),
            entry=entry,
            economics={
                key: fields[key] for key in ("gpu_reason", "expected_cu", "decision", "memory_estimate")
            },
        )
        inputs = []
        for source, target in input_paths(fields):
            with source.open("rb") as stream:
                inputs.append((target, hashlib.file_digest(stream, "sha256").hexdigest()))
        if fields.get("preflight"):
            receipt = Path(fields["preflight"]).expanduser()
            request = {"commit": job["commit"], "hardware": hardware, "command": job["command"]}
            request["runtime"] = fields.get("runtime_pins", "").split()
            job["preflight"] = preflight.verify(receipt, request, inputs)
        job.update(data_inputs(fields))
        job["resume_key"] = colab_sync.retained_key(
            job["commit"],
            {
                "device": device,
                "lane": job["lane"],
                "command": job["command"],
                "outputs": job["outputs"],
                "artifact_policy": job["artifact_policy"],
                "artifact_excludes": job["artifact_excludes"],
                "inputs": sorted(inputs),
                "data_sources": job["data_sources"],
                "wandb": job["wandb"],
            },
        )
        jobs.append(job)
    return {
        "schema": 1,
        "driver_sha": driver_sha(),
        "base": exact_ref(repo, base),
        "hardware": hardware,
        "device": name(device),
        "jobs": jobs,
        "uploads": {},
    }


def data_inputs(fields: dict) -> dict:
    """An explicit private descriptor names cached datasets and their credential."""
    if not fields.get("data_inputs"):
        return {"data_sources": []}
    descriptor = read(Path(fields["data_inputs"]).expanduser())
    if not descriptor.get("read_token_file"):
        raise ValueError("restore-only data inputs require a read_token_file descriptor")
    result = {"data_sources": [], "data_token_file": descriptor["read_token_file"]}
    for source in descriptor["sources"]:
        if not re.fullmatch(r"[0-9a-f]{64}", source["manifest_sha256"]):
            raise ValueError("cached input descriptor requires an immutable manifest hash")
        destination = Path(source["target"])
        if not destination.is_absolute() or ".." in destination.parts:
            raise ValueError("cached dataset requires an absolute VM target")
        result["data_sources"].append(
            {
                "endpoint": descriptor["url"],
                "target": str(destination),
                "prefix": source["prefix"],
                "manifest_sha256": source["manifest_sha256"],
                "manifest_key": source.get("manifest_key", source["prefix"].rstrip("/") + "/manifest.json"),
            }
        )
    return result


def split_file(source: Path, work: Path, prefix: str) -> tuple[list[str], str]:
    names = []
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        while block := stream.read(PART):
            file = work / f"{prefix}-{len(names):04d}.part"
            file.write_bytes(block)
            digest.update(block)
            names.append(file.name)
    return names, digest.hexdigest()


def stage(
    plan: dict,
    work: Path,
    repo: Path = REPO,
    *,
    netrc_file: Path | None = None,
    wandb_config: Path | None = None,
) -> None:
    work.mkdir(mode=0o700, parents=True, exist_ok=False)
    stage_sources(plan, work, repo)
    stage_inputs(plan, work)
    from tools.colab.credentials import stage_wandb

    stage_wandb(
        plan,
        work,
        netrc_file if netrc_file is not None else Path.home() / ".netrc",
        wandb_config if wandb_config is not None else Path.home() / ".config/dew/wandb.toml",
    )
    with tarfile.open(work / "runner.tar.gz", "w:gz") as archive:
        for file in HERE.glob("*.py"):
            archive.add(file, arcname="tools/colab/" + file.name)
    plan["runner_sha256"] = hashlib.sha256((work / "runner.tar.gz").read_bytes()).hexdigest()
    plan["staged_files"] = {}
    for file in work.iterdir():
        if file.is_file():
            with file.open("rb") as stream:
                plan["staged_files"][file.name] = hashlib.file_digest(stream, "sha256").hexdigest()
    save(work / "plan.json", plan)


def stage_sources(plan: dict, work: Path, repo: Path) -> None:
    refs = {"base": plan["base"], **{f"job-{job['index']}": job["commit"] for job in plan["jobs"]}}
    for job in plan["jobs"]:
        for ref in job["entry"]["fields"].get("refs", "").split():
            refs[f"extra-{len(refs)}"] = exact_ref(repo, ref)
    origin = git(repo, "remote", "get-url", "origin")
    if origin.startswith("git@github.com:"):
        origin = "https://github.com/" + origin.split(":", 1)[1]
    if urlparse(origin).username or urlparse(origin).password:
        raise ValueError("origin URL must not embed credentials")
    remote = [line.split() for line in git(repo, "ls-remote", "--refs", "origin").splitlines()]
    remote.sort(key=lambda pair: pair[1] != "refs/heads/main")
    known = [
        sha
        for sha, _ in remote
        if subprocess.run(
            ["git", "-C", str(repo), "cat-file", "-e", sha], capture_output=True, check=False
        ).returncode
        == 0
    ]
    if not known:
        raise ValueError("no known published base; fetch origin before staging")

    def published(sha):
        return any(
            sha == head
            or subprocess.run(
                ["git", "-C", str(repo), "merge-base", "--is-ancestor", sha, head],
                capture_output=True,
                check=False,
            ).returncode
            == 0
            for head in known
        )

    pending = {key: sha for key, sha in refs.items() if not published(sha)}
    prerequisites = sorted({git(repo, "merge-base", sha, known[0]) for sha in pending.values()})
    plan.update(
        origin=origin,
        source_refs=refs,
        fetch_commits=sorted(set(refs.values()) - set(pending.values()) | set(prerequisites)),
    )
    if not pending:
        return
    private = f"refs/colab/{plan['device']}/{name(work.name)}/"
    try:
        for key, sha in pending.items():
            git(repo, "update-ref", private + key, sha)
        git(
            repo,
            "bundle",
            "create",
            str(work / "dew.bundle"),
            *[private + key for key in pending],
            *["^" + sha for sha in prerequisites],
        )
    finally:
        for key in pending:
            git(repo, "update-ref", "-d", private + key)
    plan["refs"] = private
    plan["bundle_parts"], plan["bundle_sha256"] = split_file(work / "dew.bundle", work, "bundle")
    (work / "dew.bundle").unlink()


def stage_inputs(plan: dict, work: Path) -> None:
    plan["data_sources"] = []
    for job in plan["jobs"]:
        for source in job.get("data_sources", []):
            if source in plan["data_sources"]:
                continue
            target = Path(source["target"])
            if any(
                target.is_relative_to(Path(known["target"])) or Path(known["target"]).is_relative_to(target)
                for known in plan["data_sources"]
            ):
                raise ValueError("different R2 datasets have overlapping VM targets")
            plan["data_sources"].append(source)
        if job.get("data_sources"):
            source = Path(job["data_token_file"]).expanduser()
            parts, digest = split_file(source, work, "data-read-token")
            target = "/content/.dew-data-read.token"
            previous = plan["uploads"].get(target)
            if previous and previous["sha256"] != digest:
                raise ValueError("packed jobs request different data-cache credentials")
            plan["uploads"][target] = {"parts": parts, "sha256": digest, "private": True}
            plan.setdefault("private_parts", []).extend(parts)
        for source, target in input_paths(job["entry"]["fields"]):
            if any(Path(target).is_relative_to(Path(data["target"])) for data in job.get("data_sources", [])):
                raise ValueError("R2 dataset inputs must not also be staged through the PC")
            parts, digest = split_file(source, work, "input-" + str(len(plan["uploads"])))
            previous = plan["uploads"].get(target)
            if previous and previous["sha256"] != digest:
                raise ValueError(f"two inputs target {target} with different bytes")
            plan["uploads"][target] = {"parts": parts, "sha256": digest}
            if source.stat().st_mode & 0o077 == 0:
                plan["uploads"][target]["private"] = True
                plan.setdefault("private_parts", []).extend(parts)
        cache = STATE / "colab-resume" / job["resume_key"]
        if job["artifact_policy"] != "measurements" and (cache / "manifest.json").is_file():
            archive = work / f"resume-{job['index']}.tar.gz"
            colab_sync.pack_retained(cache, archive)
            job["resume_parts"], job["resume_sha256"] = split_file(archive, work, f"resume-{job['index']}")
            archive.unlink()


def staged(plan: dict, work: Path) -> dict:
    saved = read(work / "plan.json")
    if any(upload.get("private") for upload in saved.get("uploads", {}).values()):
        raise ValueError("private credentials require fresh staging at launch; do not reuse their snapshot")
    validate_reuse(saved, plan)
    for filename, expected in saved["staged_files"].items():
        file = work / filename
        if Path(filename).name != filename or file.is_symlink():
            raise ValueError("staged parts must be immutable regular files")
        with file.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
                raise ValueError("staged file checksum changed")
    return saved
