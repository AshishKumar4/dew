#!/bin/sh
# Trusted preparation precedes every guest context and offline runtime restore.
set -eu
commit=$1
source=$2
for value in "$commit" "$source"; do
    case "$value" in ''|*[!0-9a-f]*) exit 2;; esac
    test ${#value} = 40
done
apt-get update
apt-get install -y --no-install-recommends python3 python3-venv ca-certificates curl libcap2-bin libseccomp2 bubblewrap
python3 -m venv /opt/venv
# streaming for the corpora the cells read, profile for the profile cell's XProf.
/opt/venv/bin/pip install --no-cache-dir \
    "dewml[streaming,profile] @ https://github.com/AshishKumar4/dew/archive/$commit.tar.gz" \
    -c "https://raw.githubusercontent.com/AshishKumar4/dew/$commit/constraints.txt" \
    ipykernel==7.3.0 jupyter-kernel-gateway==3.0.1 jupyter-client==8.10.0 pillow==12.3.0 websockets==17.1
id -u model >/dev/null 2>&1 || useradd --uid 5000 --create-home --shell /usr/sbin/nologin model
mkdir -p /opt/live /opt/models /opt/hf /opt/xla /opt/train /run/dew /sessions
chown model:model /opt/models /opt/hf /opt/xla /opt/train
chmod 0755 /opt/models /opt/hf /opt/xla
# The page's cells, which prepare-cells.py fetches the inputs of and the context smoke runs as a
# visitor would (benchmark_gateway.py).
mkdir -p /opt/live/cells
for file in snippets/framework.py snippets/cells.py snippets/cells.json src/data/hero.py; do
    curl -fsSL "https://raw.githubusercontent.com/AshishKumar4/dew/$commit/site/$file" -o "/opt/live/cells/${file##*/}"
done
for file in text-to-image text-models; do
    curl -fsSL "https://raw.githubusercontent.com/AshishKumar4/dew/$commit/site/live/container/$file" -o "/opt/live/$file"
done
for file in guest_limits.py guest_entry.py gateway_manager.py start-gateway.sh benchmark_gateway.py warm-managed.py progress.py model_client.py model_service.py live_training.py \
            shared_bridge.py kernel_outputs.py start-shared.sh smoke-shared.py prepare-cells.py; do
    curl -fsSL "https://raw.githubusercontent.com/AshishKumar4/dew/$source/site/live/container/$file" -o "/opt/live/$file"
done
printf '%s\n' "$commit" > /opt/live/dew-commit
install -m 0644 -o model -g model /dev/null /opt/live/prepared.json
# The pool cells' inputs download while the model process's models warm: the build, snapshot
# included, must end within its 15-minute alarm (site/live/src/preparer.ts).
runuser -u model -- /opt/venv/bin/python /opt/live/prepare-cells.py > /root/prepare-cells.log 2>&1 &
cells=$!
runuser -u model -- env HF_HOME=/opt/hf JAX_PLATFORMS=cpu \
    JAX_COMPILATION_CACHE_DIR=/opt/xla JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0 \
    JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=-1 XLA_FLAGS=--xla_cpu_max_isa=AVX2 \
    /opt/venv/bin/python /opt/live/warm-managed.py
if ! wait "$cells"; then tail -c 4000 /root/prepare-cells.log >&2; exit 1; fi
# A file both the model process and a training context read, as SmolLM2's weights, is stored once.
/opt/venv/bin/python - <<'PY'
import hashlib
import os
from pathlib import Path


def files(root):
    return [path for path in Path(root).rglob("*") if path.is_file() and not path.is_symlink()
            and path.stat().st_size > 1 << 20]


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


models = {}
for path in files("/opt/models"):
    models.setdefault(path.stat().st_size, []).append(path)
for path in files("/opt/train"):
    twin = next((model for model in models.get(path.stat().st_size, []) if digest(model) == digest(path)), None)
    if twin:
        path.unlink()
        os.link(twin, path)
PY
chmod 0750 /opt/models /opt/hf /opt/xla
# Training contexts see /opt/train through an overlay of their own (gateway_manager.py), and every
# write lands in its upper layer; nothing may write the prepared files themselves. The preparation's
# locks go: a context opening one would have to copy a root-owned file up, which its user namespace
# cannot, so it creates its own.
find /opt/train -name '*.lock' -delete
chown -R root:root /opt/train
chmod -R a+rX,go-w /opt/train
rm -rf /var/lib/apt/lists/* /root/.cache /root/prepare-cells.log
