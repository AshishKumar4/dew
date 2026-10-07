#!/bin/sh
# As the user in the checkout, once per armada environment (.armada.json): an
# environment for each Python CI proves, installed as the lint job installs
# its own (.github/workflows/ci.yml): CPU torch, then dewml with every extra a
# test file imports (the workflow's EXTRAS) under constraints.txt, which pins
# the patched jax 0.11.2 the multi-process cache tests need. tokamax is not a
# dependency (docs/installation.md); the MoE tests run its grouped matmul.
set -eu
EXTRAS=test,av,tfds,metrics,plots,inference-clients,serve,vision,quantization,profile,torchax,gguf
for python in 3.12 3.14; do
  uv venv -q --python "$python" ".venv-$python"
  VIRTUAL_ENV=".venv-$python" uv pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cpu
  VIRTUAL_ENV=".venv-$python" uv pip install -q -e ".[$EXTRAS]" tokamax -c constraints.txt
done
