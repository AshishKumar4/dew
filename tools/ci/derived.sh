#!/bin/sh
# One CUDA generation's template derivations, as a release's wheel ships them (tools/derived_catalogue.py,
# docs/design/algebra.md §6.5), written to OUT: on armada, in the commit's environment (.venv-<python>: its
# locks' egglog and JAX builds, which every entry's key states). Lowered with no device attached, as a build host
# has none: a step Dew routes to cuDNN's attention where cuDNN loads cannot lower so (cuDNN's lowering reads the
# devices), so the environment holds no cuDNN, and the tool refuses those steps for the generation, their
# derivations made at a user's first run instead. CI's release run maps it over each generation and its package
# job ships their union (tools/ci/package.sh's DEW_DERIVED).
#
# Usage: sh tools/ci/derived.sh GENERATION OUT   from the checkout's top; OUT absent or holding entries alone
set -eu
"$PWD/.venv-$(tr -d '\n' < .python-version)/bin/python" tools/derived_catalogue.py --generations "$1" --out "$2"
