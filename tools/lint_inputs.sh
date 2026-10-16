# The files ruff reads: tools/check.sh lints all of them, and tools/pre-commit the staged Python
# files among them, so the gate and the hook read one list. A directory is every Python file under
# it. tools/lint_slop.py owns its own roots (`--print-roots`).
ruff_inputs="src/dew tests tools/lint_slop.py tools/lint_scopes.py tools/complexity.py tools/mutation.py examples recipes"
