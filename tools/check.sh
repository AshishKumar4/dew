#!/bin/sh
# Every cheap gate CI runs on a checkout, all of them, reporting every failure before it exits: the
# locks, flake8, ruff, slop lint, the fetched fixtures' pins (tools/fetched_fixtures.py), the
# complexity record (complexity.json), the generated API pages, installer and site colors, the tested Python
# versions, duplicate code (tools/duplicates.py) and pyright. PYTHON names
# the interpreter of the test suite's environment (tools/install_environment.sh); default: python. A
# lane runs it before every LAND, and CI's lint job runs it as is.
# Usage, from the checkout: sh tools/check.sh
set -u
. tools/lint_inputs.sh
python=${PYTHON:-python}
failed=""
gate() {
  name=$1
  shift
  if out=$("$@" 2>&1); then
    echo "=== $name ok"
  else
    failed="$failed $name"
    printf '=== %s FAILED\n%s\n' "$name" "$out"
  fi
}
# The duplicate-code scan takes as long as several gates together and reads nothing they write, so it
# runs beside them and reports last. Main still holds copies (ledger 896-1287, 1309), so a checkout is held to
# adding none to origin/main's (tools/duplicates.py); the scan needs that ref and its history.
duplicates=$(mktemp)
"$python" tools/duplicates.py --base=origin/main src/dew tools tests examples site > "$duplicates" 2>&1 &
scan=$!
gate locks sh -c 'sh tools/lock_environment.sh && git diff --exit-code --stat locks/'
gate flake8 "$python" -m flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics \
  --extend-exclude=.venv*
gate ruff "$python" -m ruff check $ruff_inputs
gate slop env JAX_PLATFORMS=cpu "$python" tools/lint_slop.py
gate fixtures "$python" tools/fetched_fixtures.py check
gate complexity sh -c "\"$python\" tools/complexity.py src/dew tools/lint_slop.py tools/lint_scopes.py examples recipes > complexity.json"
gate api-pages uv run --no-project --with-requirements site/scripts/requirements.txt python site/scripts/gen_api.py --check
gate install-script "$python" site/scripts/gen_install.py --check
gate theme "$python" site/scripts/gen_theme.py --check
gate pythons "$python" tools/armada/ci.py pythons --check
gate api-tests uv run --no-project --with-requirements site/scripts/requirements.txt --with pytest \
  python -m pytest -q --noconftest tests/test_gen_api.py
# pyright runs on Node, whose default heap limit follows the machine's memory: 2,240 MiB on an armada
# medium container (12 GiB), where checking src/dew ran out of heap. With 8 GiB it passed, the task
# peaking at 4.3 GiB (2026-10-20); 8 GiB also fits GitHub's runner (16 GiB), so both check alike.
gate pyright env NODE_OPTIONS=--max-old-space-size=8192 uvx pyright@1.1.406 --pythonpath "$(command -v "$python")" src/dew
wait "$scan"
gate duplicates sh -c 'cat "$1"; rm -f "$1"; exit "$2"' _ "$duplicates" "$?"
if [ -z "$failed" ]; then echo "check: all gates green"; else echo "check: FAILED$failed"; exit 1; fi
