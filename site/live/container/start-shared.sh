#!/bin/sh
# Only trusted serving processes can read model weights; Python contexts use IPC.
set -eu
umask 077
# The gateway, the model process and the bridge inherit this, so the OOM killer never picks them;
# the gateway launches each guest at 1000, floor included (gateway_manager.py).
echo -1000 > /proc/self/oom_score_adj
mkdir -p /run/dew/model
chmod 0711 /run/dew
chown model:model /run/dew/model
chmod 0711 /run/dew/model
sh /opt/live/start-gateway.sh
runuser -u model -- env HF_HOME=/opt/hf HF_HUB_OFFLINE=1 JAX_PLATFORMS=cpu \
    JAX_COMPILATION_CACHE_DIR=/opt/xla XLA_FLAGS=--xla_cpu_max_isa=AVX2 \
    /opt/venv/bin/python /opt/live/model_service.py > /run/dew/model.log 2>&1 &
exec /opt/venv/bin/python /opt/live/shared_bridge.py
