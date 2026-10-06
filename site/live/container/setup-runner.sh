#!/bin/sh
# A warm editable CI environment; repository access is public and read-only.
set -eu
commit=$1
python=$2
case "$commit" in ''|*[!0-9a-f]*) exit 2;; esac
test ${#commit} = 40
case "$python" in 3.12|3.14) ;; *) exit 2;; esac
apt-get update
apt-get install -y --no-install-recommends python3 python3-venv git ca-certificates curl ffmpeg libgl1 libglib2.0-0
python3 -m venv /opt/bootstrap
/opt/bootstrap/bin/pip install --no-cache-dir uv==0.9.5
/opt/bootstrap/bin/uv python install "$python"
git clone --filter=blob:none --no-checkout https://github.com/AshishKumar4/dew.git /workspace
git -C /workspace fetch --depth=1 origin "$commit"
git -C /workspace checkout --detach "$commit"
cd /workspace
/opt/bootstrap/bin/uv venv --python "$python" .venv
/opt/bootstrap/bin/uv pip install --python .venv/bin/python torch torchvision --index-url https://download.pytorch.org/whl/cpu
/opt/bootstrap/bin/uv pip install --python .venv/bin/python -e '.[test,av,tfds,metrics,plots,inference-clients,vision,quantization,profile,torchax,gguf]' tokamax -c constraints.txt
.venv/bin/python -c 'import dew, jax, pytest; print("CI environment ready", jax.__version__)'
rm -rf /var/lib/apt/lists/* /root/.cache/uv /root/.cache/pip
