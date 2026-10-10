"""Private launch credentials; plans name files and environment variables only."""
import hashlib
import netrc
import os
from pathlib import Path


def stage_wandb(plan: dict, work: Path, netrc_file: Path) -> None:
    try:
        entry = netrc.netrc(str(netrc_file)).authenticators("api.wandb.ai")
    except (OSError, netrc.NetrcParseError):
        raise ValueError("W&B launch credentials could not be read") from None
    if entry is None or not entry[2]:
        raise ValueError("W&B launch requires an api.wandb.ai netrc entry")
    payload = entry[2].encode()
    filename = "private-wandb.part"
    descriptor = os.open(work / filename, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, "wb") as output:
        output.write(payload)
    target = "/content/.dew-wandb.key"
    plan["uploads"][target] = {"parts": [filename], "sha256": hashlib.sha256(payload).hexdigest(), "private": True}
    plan.setdefault("private_parts", []).append(filename)
    plan["secret_env"] = {"WANDB_API_KEY": target}
    plan["job_env"] = {"WANDB_ENTITY": "ashishkumar4", "WANDB_PROJECT": "dew"}


def environment(plan: dict) -> dict[str, str]:
    values = dict(plan.get("job_env", {}))
    for key, filename in plan.get("secret_env", {}).items():
        path = Path(filename)
        if path.is_symlink() or not path.is_file():
            raise ValueError("invalid private environment file")
        values[key] = path.read_text().strip()
    return values
