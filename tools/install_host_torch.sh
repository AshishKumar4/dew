#!/bin/sh
# PyPI's Linux torchvision is CUDA-enabled; it cannot load beside CPU torch.
# Both host libraries stay on CPU; JAX owns accelerator work.
set -eu
exec uv pip install "$@" --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cpu
