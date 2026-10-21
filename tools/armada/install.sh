#!/bin/sh
# As the user in the checkout, once per armada environment (.armada.json): the test suite's locked
# environment (locks/cpu.txt) for each Python CI proves, and the recorded fixtures the tests read from
# the fetched store (tools/fetched_fixtures.py), which the environment's key pins by their pins.
set -eu
for python in $(python3 tools/armada/ci.py pythons); do
  uv venv -q --python "$python" ".venv-$python"
  VIRTUAL_ENV=".venv-$python" sh tools/install_environment.sh cpu -q
done
".venv-$(tr -d '\n' < .python-version)/bin/python" tools/fetched_fixtures.py fetch
