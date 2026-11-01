#!/bin/sh
# Installs the test suite's locked environment (locks/<name>.txt, from tools/lock_environment.sh) and
# Dew from this checkout. NAME is cpu, cuda12 (a driver below 580) or cuda13, an aarch64 host's GPU lock with
# -aarch64 (tools/environments.py's machine_lock). Further arguments go to
# both uv calls, such as --system or --python.
# Usage: sh tools/install_environment.sh NAME [uv pip options]
set -eu
name=$1
shift
uv pip install "$@" --torch-backend cpu -r "locks/$name.txt"
uv pip install "$@" --no-deps -e .
