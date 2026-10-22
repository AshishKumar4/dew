# The files ruff reads: tools/check.sh lints all of them, and tools/pre-commit the staged Python
# files among them, so the gate and the hook read one list. A directory is every Python file under
# it. tools/lint_slop.py owns its own roots (`--print-roots`).
ruff_inputs="src/dew tests tools/lint_slop.py tools/lint_scopes.py tools/complexity.py tools/mutation.py examples recipes"

# The Python files under the given paths (a directory is every Python file under it) that changed,
# NUL-separated: with --cached, those staged (tools/pre-commit); with a commit, those that differ from
# it in the working tree, committed or not, and those untracked (tools/check.sh --changed).
python_changed() {
    since=$1
    shift
    for path in "$@"; do
        case $path in *.py) set -- "$@" "$path" ;; *) set -- "$@" "$path/*.py" ;; esac
        shift
    done
    git diff --name-only --diff-filter=ACMR -z $since -- "$@"
    [ "$since" = --cached ] || git ls-files --others --exclude-standard -z -- "$@"
}
