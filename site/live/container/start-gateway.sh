#!/bin/sh
# Private administrative evaluation. Public routing waits for boundary and
# memory/concurrency checks; no production configuration invokes this script.
set -eu
mkdir -p /run/dew/gateway /sessions/connections /work
chmod 0711 /sessions /sessions/connections
: > /kernel.json
for index in $(seq 0 99); do
  id -u "ctx$index" >/dev/null 2>&1 || useradd --uid "$((6100+index))" --no-create-home --home-dir /work --shell /usr/sbin/nologin "ctx$index"
done
/opt/venv/bin/python - <<'PY'
import os,secrets
path='/run/dew/gateway-token'
fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_TRUNC,0o600)
with os.fdopen(fd,'w') as f:f.write(secrets.token_urlsafe(32))
PY
KG_AUTH_TOKEN=$(cat /run/dew/gateway-token)
export KG_AUTH_TOKEN
nohup env JUPYTER_RUNTIME_DIR=/run/dew/gateway PYTHONPATH=/opt/live /opt/venv/bin/jupyter-kernelgateway \
  --KernelGatewayApp.kernel_manager_class=gateway_manager.LimitedMappingKernelManager \
  --KernelGatewayApp.ip=127.0.0.1 --KernelGatewayApp.port=8890 \
  --KernelGatewayApp.max_kernels=8 \
  > /run/dew/gateway.log 2>&1 </dev/null &
