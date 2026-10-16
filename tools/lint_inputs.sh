# The files each linter reads: tools/check.sh lints all of them, and tools/pre-commit the staged
# Python files among them, so the gate and the hook read one list. A directory is every Python file
# under it.
ruff_inputs="src/dew tests tools/lint_slop.py tools/lint_scopes.py tools/complexity.py tools/mutation.py examples recipes"
slop_inputs="src/dew tests tools recipes examples"
