"""Job-opted private credentials; account settings remain local configuration."""

import hashlib
import netrc
import os
import tomllib
from pathlib import Path


def stage_wandb(plan: dict, work: Path, netrc_file: Path, config_file: Path) -> None:
    requested = [job for job in plan["jobs"] if job.get("wandb")]
    if not requested:
        return
    try:
        entry = netrc.netrc(str(netrc_file)).authenticators("api.wandb.ai")
    except (OSError, netrc.NetrcParseError):
        raise ValueError("wandb:on requires a readable netrc api.wandb.ai entry") from None
    if entry is None or not entry[2]:
        raise ValueError("wandb:on requires a netrc api.wandb.ai password")
    try:
        config = tomllib.loads(config_file.read_text())
    except (OSError, ValueError):
        raise ValueError("wandb:on requires a readable local wandb.toml configuration") from None
    if not all(isinstance(config.get(key), str) and config[key] for key in ("entity", "project")):
        raise ValueError("wandb:on requires local entity and project settings")
    payload = entry[2].encode()
    filename = "private-wandb.part"
    descriptor = os.open(work / filename, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, "wb") as output:
        output.write(payload)
    target = "/content/.dew-wandb.key"
    plan["uploads"][target] = {
        "parts": [filename],
        "sha256": hashlib.sha256(payload).hexdigest(),
        "private": True,
    }
    plan.setdefault("private_parts", []).append(filename)
    for job in requested:
        job["secret_env"] = {"WANDB_API_KEY": target}
        job["job_env"] = {"WANDB_ENTITY": config["entity"], "WANDB_PROJECT": config["project"]}


def environment(job: dict) -> dict[str, str]:
    values = dict(job.get("job_env", {}))
    for key, filename in job.get("secret_env", {}).items():
        path = Path(filename)
        if path.is_symlink() or not path.is_file():
            raise ValueError("invalid private environment file")
        values[key] = path.read_text().strip()
    return values
