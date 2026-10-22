#!/bin/sh
# Makes origin/main and its history present in this checkout, as the gates that hold a checkout to main need
# them (tools/duplicates.py, the slop ratchet). A shallow checkout, as armada's are, or one without origin/main
# fetches main from its origin remote, or, with none, from the project's repository (pyproject.toml's
# [project.urls] Repository). A complete checkout that has origin/main is left as it is.
set -e
shallow=$(git rev-parse --is-shallow-repository)
if [ "$shallow" = false ] && git rev-parse -q --verify origin/main > /dev/null; then exit 0; fi
url=$(git remote get-url origin 2> /dev/null) ||
  url=$("${PYTHON:-python3}" -c 'import tomllib; print(tomllib.load(open("pyproject.toml", "rb"))["project"]["urls"]["Repository"])')
if [ "$shallow" = true ]; then
  git fetch -q --unshallow "$url" main:refs/remotes/origin/main
else
  git fetch -q "$url" main:refs/remotes/origin/main
fi
