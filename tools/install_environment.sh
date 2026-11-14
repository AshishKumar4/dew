#!/bin/sh
# Installs the test suite's locked environment (locks/<name>.txt, from tools/lock_environment.sh) and
# Dew from this checkout. NAME is cpu, cuda12 (a driver below 580) or cuda13. Further arguments go to
# both uv calls, such as --system or --python. Every file the lock names is installed only where its bytes
# hash to the lock's sha256 (--require-hashes): one replaced under its name and version fails by name. Dew's
# own checkout, installed editable after it with --no-deps, is the one install that takes no hash.
# Usage: sh tools/install_environment.sh NAME [uv pip options]
set -eu
name=$1
shift
uv pip install "$@" --torch-backend cpu --require-hashes -r "locks/$name.txt"
uv pip install "$@" --no-deps -e .
