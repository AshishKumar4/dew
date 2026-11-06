#!/bin/sh
# CI's lint gate on one machine: every gate of tools/check_gates, one after another (tools/check.sh), each one's
# failure reported and the gate red where any is. The land queue runs each gate on an armada container of its own
# (dew-ops queue/lint-gate.sh); GitHub's CI runs this for a pull request from a fork, on a hosted runner, which holds
# no armada credentials.
#
# Usage: sh tools/ci/lint.sh <base>   from the checkout's top, PYTHON the suite environment's interpreter; <base> is
#                                     the commit main's history ratchet reads (the merge-base with the target).
set -u
[ $# -eq 1 ] || { echo "usage: tools/ci/lint.sh <base>" >&2; exit 2; }
red=
for gate in $(grep -v '^#' tools/check_gates | cut -f1); do
  echo "=== $gate"
  BASE=$1 sh tools/check.sh --gate "$gate" || red="$red $gate"
done
[ -z "$red" ] || { echo "lint: red$red"; exit 1; }
echo "lint: green"
