#!/bin/sh
# Makes origin/main, the base the gates that hold a checkout to main read (tools/duplicates.py, the slop
# ratchet, the public-text gate), present in this checkout from the history it holds, fetching nothing: a
# gate's container on armada carries its commit's history (.armada.json's history full) and is given BASE,
# its commit's merge-base with main (the lanes' lint gate passes it), which origin/main then names. A
# checkout that has origin/main is left as it is. One whose history lacks BASE fails, naming it: the
# gates read no network, where a fetch of main failed lanes' lint (Main, 2026-10-28: curl 92 over HTTPS).
set -e
if git rev-parse -q --verify origin/main > /dev/null; then exit 0; fi
base=${BASE:-origin/main}
if ! git merge-base --is-ancestor "$base" HEAD 2> /dev/null; then
  echo "base_history: this checkout's history holds no $base: give the gate BASE, its commit's merge-base with" \
    "main, in a checkout carrying its history (git fetch origin main, outside a gate's container)" >&2
  exit 1
fi
git update-ref refs/remotes/origin/main "$(git rev-parse "$base^{commit}")"
