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
/opt/venv/bin/pip install --no-cache-dir \
    "dewml @ https://github.com/AshishKumar4/dew/archive/$commit.tar.gz" \
    -c "https://raw.githubusercontent.com/AshishKumar4/dew/$commit/constraints.txt" \
    ipykernel==7.3.0 jupyter-kernel-gateway==3.0.1 jupyter-client==8.10.0 pillow
id -u model >/dev/null 2>&1 || useradd --uid 5000 --create-home --shell /usr/sbin/nologin model
mkdir -p /opt/live /opt/models /opt/hf /opt/xla /run/dew /sessions
chown model:model /opt/models /opt/hf /opt/xla
chmod 0755 /opt/models /opt/hf /opt/xla
for file in text-to-image text-models; do
    curl -fsSL "https://raw.githubusercontent.com/AshishKumar4/dew/$commit/site/live/container/$file" -o "/opt/live/$file"
done
for file in guest_limits.py guest_entry.py gateway_manager.py start-gateway.sh benchmark_gateway.py warm-managed.py; do
    curl -fsSL "https://raw.githubusercontent.com/AshishKumar4/dew/$source/site/live/container/$file" -o "/opt/live/$file"
done
printf '%s\n' "$commit" > /opt/live/dew-commit
install -m 0644 -o model -g model /dev/null /opt/live/prepared.json
runuser -u model -- env HF_HOME=/opt/hf JAX_PLATFORMS=cpu \
    JAX_COMPILATION_CACHE_DIR=/opt/xla JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0 \
    JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=-1 XLA_FLAGS=--xla_cpu_max_isa=AVX2 \
    /opt/venv/bin/python /opt/live/warm-managed.py
chmod 0750 /opt/models /opt/hf /opt/xla
rm -rf /var/lib/apt/lists/* /root/.cache
