#!/bin/sh
# As the user in the checkout, once per armada environment (.armada.json): the
# test suite's environment (pyproject.toml's `ci` extra) for each Python CI
# proves, beside CPU torch, under constraints.txt, which pins the patched jax
# 0.11.2 the multi-process cache tests need.
set -eu
for python in 3.12 3.14; do
  uv venv -q --python "$python" ".venv-$python"
  VIRTUAL_ENV=".venv-$python" sh tools/install_host_torch.sh -q
  VIRTUAL_ENV=".venv-$python" uv pip install -q -e ".[ci]" -c constraints.txt
done
