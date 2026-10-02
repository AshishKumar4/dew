#!/bin/sh
# Trusted preparation only: no visitors enter this container until setup, warm-up
# and hardening checks finish. Runtime restores have outbound Internet disabled.
set -eu
commit=$1
case "$commit" in ''|*[!0-9a-f]*) exit 2;; esac
[ ${#commit} = 40 ]
apt-get update
apt-get install -y --no-install-recommends python3 python3-venv ca-certificates curl util-linux libcap2-bin bubblewrap
python3 -m venv /opt/venv
/opt/venv/bin/pip install --no-cache-dir \
  "dewml @ https://github.com/AshishKumar4/dew/archive/$commit.tar.gz" \
  -c "https://raw.githubusercontent.com/AshishKumar4/dew/$commit/constraints.txt" \
  ipykernel jupyter-kernel-gateway pillow
id -u model >/dev/null 2>&1 || useradd --uid 5000 --create-home --shell /usr/sbin/nologin model
mkdir -p /opt/live /opt/models /opt/hf /opt/xla /run/dew /sessions
chown model:model /opt/models /opt/hf /opt/xla /run/dew
chmod 0755 /opt/models /opt/hf /opt/xla
chmod 0700 /sessions
curl -fsSL "https://raw.githubusercontent.com/AshishKumar4/dew/$commit/site/live/container/text-to-image" -o /opt/live/text-to-image
curl -fsSL "https://raw.githubusercontent.com/AshishKumar4/dew/$commit/site/live/container/text-models" -o /opt/live/text-models
curl -fsSL "https://raw.githubusercontent.com/AshishKumar4/dew/$commit/site/live/container/demo.py" -o /opt/live/demo.py
printf '%s\n' "$commit" > /opt/live/dew-commit
cat > /opt/live/warm-managed.py <<'PY'
import json, os, pathlib, time
from huggingface_hub import snapshot_download
start=time.perf_counter()
repo, revision=pathlib.Path('/opt/live/text-to-image').read_text().strip().split('@')
snapshot=pathlib.Path(snapshot_download(repo_id=repo, revision=revision))
refs=snapshot.parent.parent/'refs'; refs.mkdir(exist_ok=True); (refs/'main').write_text(revision)
for line in pathlib.Path('/opt/live/text-models').read_text().split():
    name, revision=line.split('@')
    snapshot_download(repo_id=name, revision=revision, local_dir=f'/opt/models/{name}',
                      allow_patterns=['*.json','*.safetensors','*.txt'])
print('download_seconds',time.perf_counter()-start,flush=True)
# Install configs used by from_pretrained while outbound access is still allowed.
from dew.sampling import TextToImage, CFG, DPMSolverMultistep
import jax.numpy as jnp
from dew.interop import PretrainedDecoder
from dew.inference import Server
from dew.sampling import Sampling
import jax
from jax._src.lib import xla_client
pipe=TextToImage.from_pretrained(repo)
for steps in (15,30):
    t=time.perf_counter()
    pipe(['a turquoise alpine lake'],key=0,steps=steps,solver=DPMSolverMultistep(),guidance=CFG(5.0)).host()
    print('image_warm',steps,time.perf_counter()-t,flush=True)
bundle=PretrainedDecoder.load('/opt/models/HuggingFaceTB/SmolLM2-135M-Instruct',dtype=jnp.float32,max_seq_len=256)
server=Server.from_task(bundle.text_generation(sampling=Sampling(temperature=0)),slots=4,capacity=128)
ids=bundle.processor('The capital of France is').tokens[0]
for _ in range(4): server.submit(ids,24,key=0)
server.run()
report={'dew':pathlib.Path('/opt/live/dew-commit').read_text().strip(),'jax':jax.__version__,
        'cpu_topology':xla_client.get_topology_for_devices(jax.devices()).fingerprint(),
        'xla_flags':os.environ.get('XLA_FLAGS'),'total_seconds':time.perf_counter()-start,
        'cache_bytes':sum(p.stat().st_size for p in pathlib.Path('/opt/xla').rglob('*') if p.is_file())}
pathlib.Path('/opt/live/prepared.json').write_text(json.dumps(report))
print(json.dumps(report),flush=True)
PY
chown model:model /opt/live
runuser -u model -- env HF_HOME=/opt/hf JAX_PLATFORMS=cpu \
  JAX_COMPILATION_CACHE_DIR=/opt/xla JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0 \
  JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=-1 XLA_FLAGS=--xla_cpu_max_isa=AVX2 \
  /opt/venv/bin/python /opt/live/warm-managed.py
# Report namespace feasibility rather than assuming uid alone is isolation.
bwrap --unshare-user --unshare-pid --unshare-net --ro-bind / / --proc /proc \
  --tmpfs /tmp --cap-drop ALL /bin/sh -c 'id; grep Cap /proc/self/status' \
  > /opt/live/namespace-probe.txt 2>&1
cat /opt/live/namespace-probe.txt
rm -rf /var/lib/apt/lists/* /root/.cache
