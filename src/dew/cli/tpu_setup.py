"""The scripts dew tpu runs on the workers.

The setup script is rendered from the Python version, the extras and the
source mode. Every step is guarded, so a second run re-creates nothing, but
the `uv pip install` lines resolve against PyPI and Dew's package index, which
serves Dew's JAX build, each time. A checkout is
installed alone with its `tpu` extra, the libtpu for the jax the checkout
requires. A release installs after jax[tpu], so a jax[tpu] or dewml release
since the last run is installed; `setup --version` pins the dewml side.
"""

from __future__ import annotations

import shlex

from dew.builds import PACKAGE_INDEX

#: Paths the remote side owns, and the only thing the commands agree on.
VENV_NAME = "dew-venv"
VENV = f"$HOME/{VENV_NAME}"
ENV_NAME = ".dew-env"
ENV_FILE = f"$HOME/{ENV_NAME}"
RUNS = "dew-runs"
RUNS_DIR = f"$HOME/{RUNS}"
MOUNT_PATH = "$HOME/gcs_mount"
DISK_MOUNT = "/mnt/persist"
OPEN_FILES = 1048576
"""The open-file limit a worker's shell, its PAM sessions and its systemd manager all take."""

PACKAGE = "dewml"

_ENV_LINES = (
    f'export PATH="{VENV}/bin:$HOME/.local/bin:$PATH"',
    "export TOKENIZERS_PARALLELISM=false",
    "export WANDB_CACHE_DIR=/tmp/wandb-cache",
    f"ulimit -n {OPEN_FILES} 2>/dev/null || true",
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
  sudo -n tee /etc/apt/sources.list.d/gcsfuse.list >/dev/null \\
    <<<"deb [signed-by=/usr/share/keyrings/cloud.google.asc] https://packages.cloud.google.com/apt $repo main"
  sudo -n curl -fsSL -o /usr/share/keyrings/cloud.google.asc https://packages.cloud.google.com/apt/doc/apt-key.gpg
  sudo -n apt-get update -qq
  sudo -n apt-get install -y -qq gcsfuse libgl1 rsync lsof
fi

step "open file limits"
# Each privileged write reads a here-document, written whole before the command runs, so no writer
# can outlive a reader that exits without reading, which `set -o pipefail` reports as a failure.
limits=/etc/security/limits.conf
grep -q "dew nofile" "$limits" || sudo -n tee -a "$limits" >/dev/null <<'LIMITS'
# dew nofile
* soft nofile @OPEN_FILES@
* hard nofile @OPEN_FILES@
LIMITS
override=/etc/systemd/system.conf.d/99-dew-nofile.conf
wanted="[Manager]
DefaultLimitNOFILE=@OPEN_FILES@"
if [ "$(cat $override 2>/dev/null)" != "$wanted" ]; then
  sudo -n mkdir -p /etc/systemd/system.conf.d
  sudo -n tee "$override" >/dev/null <<<"$wanted"
  sudo -n systemctl daemon-reload
fi

step "uv"
# BEGIN uv bootstrap (site/scripts/gen_install.py)
uv_least=0.13.4
# uv at uv_least or newer (pyproject.toml's [tool.uv] required-version). bootstrap_uv [NAME=VALUE...]
# installs that release from astral's installer, NAME=VALUE in the installer's environment, where uv is
# missing or older; uv_version is the release of the uv on PATH (none where there is none), and
# uv_older VERSION says whether VERSION is older: a pre-release of uv_least is, a build of it is not.
uv_version() { (uv --version 2>/dev/null || true) | awk '{print $2}'; }
uv_older() {
	awk -v have="${1%%+*}" -v least="$uv_least" 'BEGIN {
		split(have, h, "."); split(least, l, ".")
		for (i = 1; i <= 3; i++) if (h[i] + 0 != l[i] + 0) exit !(h[i] + 0 < l[i] + 0)
		exit have !~ /[^0-9.]/
	}'
}
bootstrap_uv() {
	have=$(uv_version)
	if [ -z "$have" ] || uv_older "$have"; then
		curl -LsSf "https://astral.sh/uv/$uv_least/install.sh" | env "$@" sh
	fi
}
# END uv bootstrap
export PATH="$HOME/.local/bin:$PATH"
bootstrap_uv

step "python $PYTHON_VERSION"
have=$("$VENV/bin/python" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || true)
[ "$have" = "$PYTHON_VERSION" ] || uv venv --python "$PYTHON_VERSION" "$VENV"
PY="$VENV/bin/python"

step "$PACKAGE_SPEC"
if [ "$EDITABLE" = 1 ]; then
  uv pip install --quiet --python "$PY" --extra-index-url "$PACKAGE_INDEX" -e "$HOME/$PACKAGE_SPEC"
else
  uv pip install --quiet --python "$PY" --extra-index-url "$PACKAGE_INDEX" "$PACKAGE_SPEC"
fi

step "environment"
cat > "$ENV_FILE" <<'DEW_ENV'
@ENV_LINES@
DEW_ENV
grep -q '@ENV_NAME@' "$HOME/.bashrc" || echo '. @ENV_FILE@' >> "$HOME/.bashrc"
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

if [ "$GIT_KEY" = 1 ]; then
  step "git key"
  chmod 600 "$HOME/.ssh/id_ed25519"
  grep -qs '^github.com ' "$HOME/.ssh/known_hosts" \\
    || ssh-keyscan -t ed25519 github.com >> "$HOME/.ssh/known_hosts" 2>/dev/null
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
    f'"$(pgrep -fc \'[{VENV_NAME[0]}]{VENV_NAME[1:]}/bin/python\' || true)" '
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
    gcs_bucket: str = "",
    git_key: bool = False,
) -> str:
    """The setup script for one worker."""
    header = [
        "#!/bin/bash",
        "# Rendered by dew tpu setup.",
        "set -euo pipefail",
        f"PYTHON_VERSION={shlex.quote(python_version)}",
        f"PACKAGE_SPEC={shlex.quote(package_spec)}",
        f"PACKAGE_INDEX={shlex.quote(PACKAGE_INDEX)}",
        f"EDITABLE={int(editable)}",
        f"GCS_BUCKET={shlex.quote(gcs_bucket)}",
        f"GIT_KEY={int(git_key)}",
        f'VENV="{VENV}"',
        f'ENV_FILE="{ENV_FILE}"',
        f'RUNS_DIR="{RUNS_DIR}"',
        f'MOUNT_PATH="{MOUNT_PATH}"',
    ]
    body = (_BODY.replace("@ENV_LINES@", "\n".join(_ENV_LINES)).replace("@OPEN_FILES@", str(OPEN_FILES))
            .replace("@ENV_FILE@", ENV_FILE).replace("@ENV_NAME@", ENV_NAME))
    body = body.replace("@GCSFUSE_YML@", _GCSFUSE_YML.rstrip("\n"))
    return "\n".join(header) + body


def package_spec(source_dir: str, extras: str, version: str) -> tuple[str, bool]:
    """What to install, in one resolution: a directory under the worker's home, or a release, each
    with its tpu extra, which pins the JAX range and the libtpu release Dew is built against, and
    `extras`."""
    names = dict.fromkeys(["tpu", *filter(None, (name.strip() for name in extras.split(",")))])
    extra = f"[{','.join(names)}]"
    if source_dir:
        return f"{source_dir}{extra}", True
    pin = f"=={version}" if version else ""
    return f"{PACKAGE}{extra}{pin}", False


def wrap(command: str) -> str:
    """A remote command that sees the dew venv, over a non-login shell."""
    return f". {ENV_FILE} 2>/dev/null; {command}"


def home_path(*parts: str) -> str:
    """A path under the worker's home, each part quoted for the remote shell.

    $HOME stays outside the quotes so the shell still expands it. A part that
    needs no quoting is left as it is, which is every job name the CLI accepts.
    """
    return "$HOME/" + "/".join(shlex.quote(part) for part in parts)


def log_path(job: str, worker: int) -> str:
    return home_path(RUNS, job, f"worker-{worker}.log")


def detached(command: str, job: str, worker: int, home_dir: str = "") -> str:
    """Start a command under nohup and return once it is running.

    home_dir is a directory under the worker's home to run in; `sync` puts
    the working tree there.
    """
    inner = wrap(f"cd {home_path(home_dir)} && {command}" if home_dir else command)
    log = log_path(job, worker)
    return (
        f"mkdir -p {home_path(RUNS, job)} && "
        f"nohup bash -c {shlex.quote(inner)} > {log} 2>&1 < /dev/null & "
        f'echo "job {job} pid $!"'
    )


def tail(job: str, worker: int, follow: bool, lines: int = 200) -> str:
    following = "-f " if follow else ""
    return f"tail {following}-n {lines} {log_path(job, worker)}"


def disk_startup_script(read_only: bool = False) -> str:
    """Mount an attached data disk on boot, and keep it mounted: opened to every user, or, attached
    `read_only`, mounted read-only, as a disk several workers share is. A disk that does not mount
    fails the script, before the boot disk's directory is opened up or named in fstab as the disk's:
    writes under the mount would otherwise land on the boot disk, which the VM does not keep."""
    mount = DISK_MOUNT
    options = "ro,nofail" if read_only else "defaults,nofail"
    return (
        "#!/bin/bash\n"
        "set -euo pipefail\n"
        f"mkdir -p {mount}\n"
        f"mountpoint -q {mount} || mount {'-o ro ' if read_only else ''}/dev/sdb {mount}\n"
        + ("" if read_only else f"chmod 777 {mount}\n")
        + f"grep -q ' {mount} ' /etc/fstab || echo '/dev/sdb {mount} ext4 {options} 0 2' >> /etc/fstab\n"
    )
