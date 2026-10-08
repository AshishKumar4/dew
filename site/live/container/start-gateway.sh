#!/bin/sh
# Private administrative evaluation. Public routing waits for boundary and
# memory/concurrency checks; no production configuration invokes this script.
set -eu
umask 077
mkdir -p /run/dew/gateway /sessions/connections /sessions/ipc /work
chmod 0711 /sessions /sessions/connections /sessions/ipc
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
/opt/venv/bin/python - <<'PYTHON'
import json
from pathlib import Path
path = Path('/opt/venv/share/jupyter/kernels/python3/kernel.json')
spec = json.loads(path.read_text())
spec['interrupt_mode'] = 'message'
path.write_text(json.dumps(spec))
PYTHON
KG_AUTH_TOKEN=$(cat /run/dew/gateway-token)
export KG_AUTH_TOKEN
nohup env JUPYTER_RUNTIME_DIR=/run/dew/gateway PYTHONPATH=/opt/live /opt/venv/bin/jupyter-kernelgateway \
  --KernelGatewayApp.kernel_manager_class=gateway_manager.LimitedMappingKernelManager \
  --KernelGatewayApp.ip=127.0.0.1 --KernelGatewayApp.port=8890 \
  --KernelGatewayApp.max_kernels=24 \
  > /run/dew/gateway.log 2>&1 </dev/null &
/opt/venv/bin/python - <<'PY'
import time
import urllib.error
import urllib.request
from pathlib import Path

token = Path('/run/dew/gateway-token').read_text()
request = urllib.request.Request('http://127.0.0.1:8890/api/swagger.json',
                                 headers={'Authorization': 'token ' + token})
deadline = time.monotonic() + 30
while True:
    try:
        with urllib.request.urlopen(request, timeout=1) as response:
            if response.status != 200:
                raise RuntimeError('Kernel Gateway did not serve its API specification')
        break
    except urllib.error.HTTPError as error:
        detail = error.read().decode(errors='replace')[:1000]
        raise RuntimeError(f'Gateway HTTP {error.code}: {detail}') from error
    except (urllib.error.URLError, TimeoutError):
        if time.monotonic() >= deadline:
            raise TimeoutError('Kernel Gateway did not listen within 30 seconds') from None
        time.sleep(0.1)
PY
