#!/bin/sh
# Every cheap gate CI runs on a checkout, all of them, reporting every failure before it exits: the
# locks, flake8, ruff, slop lint, the complexity record (complexity.json), the generated API pages,
# duplicate code (tools/duplicates.py) and pyright. PYTHON names the interpreter of the test suite's
# environment (tools/install_environment.sh); default: python. A lane runs it before every LAND, and CI's lint job runs it as is.
# Usage, from the checkout: sh tools/check.sh
set -u
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
gate locks sh -c 'sh tools/lock_environment.sh && git diff --exit-code --stat locks/'
gate flake8 "$python" -m flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics \
  --extend-exclude=site/src/data/sampler.py,site/src/data/text.py,.venv*
gate ruff "$python" -m ruff check src/dew tests tools/lint_slop.py tools/lint_scopes.py tools/complexity.py tools/mutation.py examples recipes
gate slop env JAX_PLATFORMS=cpu "$python" tools/lint_slop.py
gate complexity sh -c "\"$python\" tools/complexity.py src/dew tools/lint_slop.py tools/lint_scopes.py examples recipes > complexity.json"
gate api-pages uv run --no-project --with griffe==2.3.0 python site/scripts/gen_api.py --check
gate api-tests uv run --no-project --with griffe==2.3.0 --with pytest python -m pytest -q --noconftest tests/test_gen_api.py
gate duplicates uv run --no-project --with pylint==4.1.3 python tools/duplicates.py src/dew tools tests
gate duplicates-test uv run --no-project --with pylint==4.1.3 --with pytest python -m pytest -q --noconftest \
  tests/test_duplicates.py
gate pyright uvx pyright@1.1.406 --pythonpath "$(command -v "$python")" src/dew
if [ -z "$failed" ]; then echo "check: all gates green"; else echo "check: FAILED$failed"; exit 1; fi
