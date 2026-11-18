#!/bin/sh
# As the user in the checkout, once per armada environment (.armada.json): the test suite's locked
# environment (locks/cpu.txt) for each Python CI proves, its modules compiled to bytecode as installed, the
# interpreter's own standard library and Dew's src/ (tools/armada/compiled.sh) too, so no task compiles them from
# source as it imports them (row 3215), and the recorded fixtures the tests read from the fetched store
# (tools/fetched_fixtures.py) as this commit pins them. The pins are no part of the environment's key: a task
# fetches those its own commit adds or moves (tools/armada/session.py's `fetched_absent`), where every
# re-pinned fixture rebuilt the whole environment, 6.5-16 minutes, for each run of a commit that moved one
# (2026-11-18).
set -eu
for python in $(python3 tools/armada/ci.py pythons); do
  uv venv -q --python "$python" ".venv-$python"
  VIRTUAL_ENV=".venv-$python" sh tools/install_environment.sh cpu -q --compile-bytecode
  # uv's interpreters ship most of their standard library without pycs (183 of 936 modules on 3.12).
  ".venv-$python/bin/python" -m compileall -q "$(".venv-$python/bin/python" -c 'import sysconfig; print(sysconfig.get_path("stdlib"))')"
  sh tools/armada/compiled.sh ".venv-$python/bin/python"
done
# The repository's own Python's environment as .venv too, the one pyproject.toml's [tool.pyright] names, so pyright
# resolves it here as in a developer's checkout.
ln -sfn ".venv-$(tr -d '\n' < .python-version)" .venv
".venv-$(tr -d '\n' < .python-version)/bin/python" tools/fetched_fixtures.py fetch
