#!/bin/sh
# Dew's own modules (src/, in the checkout) compiled for PYTHON with each source's hash in its pyc, which
# Python compares at every import: a container's boot checkout (git checkout -f) gives every tracked file a new
# mtime on CI's hosts, which a pyc recording its source's mtime would read as stale for every module, so the
# first interpreters of every container compiled src/ again (ledger 3215: 1.4 s of a first collect); a module a
# task's commit leaves as it was loads from its pyc, an edited one compiles anew.
# Usage, from the checkout: sh tools/armada/compiled.sh PYTHON
set -eu
"$1" -m compileall -q --invalidation-mode checked-hash src
