#!/bin/sh
# As the user in the checkout, once per armada environment (.armada.json): the test suite's locked
# environment (locks/cpu.txt) for each Python CI proves.
set -eu
for python in $(sed -n 's/.*"Programming Language :: Python :: \([0-9]*\.[0-9]*\)".*/\1/p' pyproject.toml); do
  uv venv -q --python "$python" ".venv-$python"
  VIRTUAL_ENV=".venv-$python" sh tools/install_environment.sh cpu -q
done
