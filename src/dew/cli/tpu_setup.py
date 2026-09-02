"""The scripts dew-tpu runs on the workers.

The setup script is rendered, not checked in: the Python version, the extras
and the source mode are its parameters. Running it twice changes nothing.
"""

from __future__ import annotations

import shlex

#: Paths the remote side owns. The commands agree on these and nothing else.
VENV = "$HOME/dew-venv"
ENV_FILE = "$HOME/.dew-env"
RUNS_DIR = "$HOME/dew-runs"
MOUNT_PATH = "$HOME/gcs_mount"
DISK_MOUNT = "/mnt/persist"

JAX_SPEC = "jax[tpu]"
PACKAGE = "dew-ml"

_ENV_LINES = (
    'export PATH="$HOME/dew-venv/bin:$HOME/.local/bin:$PATH"',
    "export TOKENIZERS_PARALLELISM=false",
    "export WANDB_CACHE_DIR=/tmp/wandb-cache",
    "ulimit -n 1048576 2>/dev/null || true",
)

_GCSFUSE_YML = """\
file-cache:
  max-size-mb: 40960
  cache-file-for-range-read: true
metadata-cache:
  stat-cache-max-size-mb: 4096
  ttl-secs: 60
  type-cache-max-size-mb: 4096
file-system:
  kernel-list-cache-ttl-secs: 60
  ignore-interrupts: true
"""

_BODY = """
step() { echo "== $*"; }

step "apt packages"
if ! command -v gcsfuse >/dev/null || ! command -v rsync >/dev/null; then
  repo="gcsfuse-$(lsb_release -c -s)"
  echo "deb [signed-by=/usr/share/keyrings/cloud.google.asc] https://packages.cloud.google.com/apt $repo main" \\
    | sudo -n tee /etc/apt/sources.list.d/gcsfuse.list >/dev/null
  curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg \\
    | sudo -n tee /usr/share/keyrings/cloud.google.asc >/dev/null
  sudo -n apt-get update -qq
  sudo -n apt-get install -y -qq gcsfuse libgl1 rsync lsof
fi

step "open file limits"
grep -q "dew nofile" /etc/security/limits.conf || printf '%s\\n' \\
  "# dew nofile" "* soft nofile 1048576" "* hard nofile 1048576" \\
  | sudo -n tee -a /etc/security/limits.conf >/dev/null
override=/etc/systemd/system.conf.d/99-dew-nofile.conf
wanted="[Manager]
DefaultLimitNOFILE=1048576"
if [ "$(cat $override 2>/dev/null)" != "$wanted" ]; then
  sudo -n mkdir -p /etc/systemd/system.conf.d
  echo "$wanted" | sudo -n tee $override >/dev/null
  sudo -n systemctl daemon-reload
fi

step "uv"
command -v uv >/dev/null || [ -x "$HOME/.local/bin/uv" ] || curl -fsSL https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

step "python $PYTHON_VERSION"
have=$("$VENV/bin/python" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || true)
[ "$have" = "$PYTHON_VERSION" ] || uv venv --python "$PYTHON_VERSION" "$VENV"
PY="$VENV/bin/python"

step "$JAX_SPEC"
uv pip install --quiet --python "$PY" "$JAX_SPEC"

step "$PACKAGE_SPEC"
if [ "$EDITABLE" = 1 ]; then
  uv pip install --quiet --python "$PY" -e "$HOME/$PACKAGE_SPEC"
else
  uv pip install --quiet --python "$PY" "$PACKAGE_SPEC"
fi

step "environment"
cat > "$ENV_FILE" <<'DEW_ENV'
@ENV_LINES@
DEW_ENV
grep -q '.dew-env' "$HOME/.bashrc" || echo '. $HOME/.dew-env' >> "$HOME/.bashrc"
mkdir -p "$RUNS_DIR"

if [ -n "$GCS_BUCKET" ]; then
  step "gcsfuse $GCS_BUCKET"
  cat > "$HOME/gcsfuse.yml" <<'DEW_GCSFUSE'
@GCSFUSE_YML@
DEW_GCSFUSE
  mkdir -p "$MOUNT_PATH"
  mountpoint -q "$MOUNT_PATH" \\
    || gcsfuse --config-file "$HOME/gcsfuse.yml" --implicit-dirs "$GCS_BUCKET" "$MOUNT_PATH"
fi

step "ready: $("$PY" -V)"
"""

RESET = """\
set -u
devices=$(ls /dev/accel* 2>/dev/null || true)
if [ -z "$devices" ]; then echo "no accelerator devices"; exit 0; fi
pids=$(sudo -n lsof -t $devices 2>/dev/null | sort -u | tr '\\n' ' ')
if [ -z "$pids" ]; then echo "nothing holds $devices"; exit 0; fi
echo "killing $pids"
sudo -n kill -9 $pids
"""

#: One line of `uptime|dew processes|processes holding a device|devices`.
STATUS = (
    "printf '%s|%s|%s|%s\\n' "
    '"$(uptime -p 2>/dev/null | sed s/^up.//)" '
    '"$(pgrep -fc \'[d]ew-venv/bin/python\' || true)" '
    '"$(sudo -n lsof -t /dev/accel* 2>/dev/null | sort -u | wc -l)" '
    '"$(ls /dev/accel* 2>/dev/null | wc -l)"'
)

DEVICE_COUNT = (
    f'"{VENV}/bin/python" -c '
    "'import jax; print(jax.device_count(), jax.local_device_count())'"
)


def render(
    *,
    python_version: str,
    package_spec: str,
    editable: bool = False,
    jax_spec: str = JAX_SPEC,
    gcs_bucket: str = "",
) -> str:
    """The setup script for one worker."""
    header = [
        "#!/bin/bash",
        "# Rendered by dew-tpu setup. A second run changes nothing.",
        "set -euo pipefail",
        f"PYTHON_VERSION={shlex.quote(python_version)}",
        f"JAX_SPEC={shlex.quote(jax_spec)}",
        f"PACKAGE_SPEC={shlex.quote(package_spec)}",
        f"EDITABLE={int(editable)}",
        f"GCS_BUCKET={shlex.quote(gcs_bucket)}",
        f'VENV="{VENV}"',
        f'ENV_FILE="{ENV_FILE}"',
        f'RUNS_DIR="{RUNS_DIR}"',
        f'MOUNT_PATH="{MOUNT_PATH}"',
    ]
    body = _BODY.replace("@ENV_LINES@", "\n".join(_ENV_LINES))
    body = body.replace("@GCSFUSE_YML@", _GCSFUSE_YML.rstrip("\n"))
    return "\n".join(header) + body


def package_spec(source_dir: str, extras: str, version: str) -> tuple[str, bool]:
    """What to install: a directory under the worker's home, or a release."""
    suffix = f"[{extras}]" if extras else ""
    if source_dir:
        return f"{source_dir}{suffix}", True
    pin = f"=={version}" if version else ""
    return f"{PACKAGE}{suffix}{pin}", False


def wrap(command: str) -> str:
    """A remote command that sees the dew venv, over a non-login shell."""
    return f". {ENV_FILE} 2>/dev/null; {command}"


def log_path(job: str, worker: int) -> str:
    return f"{RUNS_DIR}/{job}/worker-{worker}.log"


def detached(command: str, job: str, worker: int, cwd: str = "") -> str:
    """Start a command under nohup and return once it is running."""
    inner = wrap(f"cd {cwd} && {command}" if cwd else command)
    log = log_path(job, worker)
    return (
        f"mkdir -p {RUNS_DIR}/{job} && "
        f"nohup bash -c {shlex.quote(inner)} > {log} 2>&1 < /dev/null & "
        f'echo "job {job} pid $!"'
    )


def tail(job: str, worker: int, follow: bool, lines: int = 200) -> str:
    flag = "-f " if follow else ""
    return f"tail {flag}-n {lines} {log_path(job, worker)}"


def disk_startup_script() -> str:
    """Mount an attached data disk on boot, and keep it mounted."""
    mount = DISK_MOUNT
    return (
        "#!/bin/bash\n"
        f"mkdir -p {mount}\n"
        f"mountpoint -q {mount} || mount /dev/sdb {mount} || true\n"
        f"chmod 777 {mount}\n"
        f"grep -q ' {mount} ' /etc/fstab || echo '/dev/sdb {mount} ext4 defaults,nofail 0 2' >> /etc/fstab\n"
    )
