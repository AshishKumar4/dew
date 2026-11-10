# The files ruff reads: tools/check.sh lints all of them, and tools/pre-commit the staged Python
# files among them, so the gate and the hook read one list. A directory is every Python file under
# it. tools/lint_slop.py owns its own roots (`--print-roots`).
ruff_inputs="src/dew src/_dew_torch tests tools/lint_slop.py tools/lint_scopes.py tools/lint_assertions.py tools/lint_layers.py tools/changelog_released.py tools/complexity.py examples recipes"

# The roots pyright reads (tools/check_gates); check.sh --changed binds it to the changed Python files under them.
pyright_inputs="src/dew src/_dew_torch"

# The roots the duplicate-code gate reads (tools/duplicates.py), as text and as structure (tools/check_gates).
duplicate_roots="src/dew src/_dew_torch tools tests examples site"

# The Python files under the given paths (a directory is every Python file under it) that changed,
# NUL-separated: with --cached, those staged (tools/pre-commit); with a commit, those that differ from
# it in the working tree, committed or not, and those untracked (tools/check.sh --changed). Its status is
# git's: a listing git could not make (a commit the checkout lacks) fails, never reads as nothing changed.
python_changed() {
    since=$1
    shift
    for path in "$@"; do
        case $path in *.py) set -- "$@" "$path" ;; *) set -- "$@" "$path/*.py" ;; esac
        shift
    done
    git diff --name-only --diff-filter=ACMR -z $since -- "$@" || return
    [ "$since" = --cached ] || git ls-files --others --exclude-standard -z -- "$@"
}
