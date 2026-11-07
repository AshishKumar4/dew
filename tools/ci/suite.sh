#!/usr/bin/env bash
# CI's test gate: the suite on armada and, beside it, tests/test_multihost.py's multi-host gangs, and the whole
# suite's release checks after it, from the machine that sends them (a GitHub runner). It sends the commit, waits,
# prints each red row's output, and fails on a failed test, a test skipped on a missing import, a file left
# unreported or a gang that did not pass. A commit armada already graded green is not run again, nor one PROVEN.
#
# The suite runs both Pythons as tasks of about five minutes each on Dew's armada deployment (.armada.json,
# tools/armada/ci.py), on at most its pool, the gangs' containers beside it. A straggler at the end runs again
# (task.speculative), and tasks share XLA's compilations through the dew-xla-cache Worker (tests/remote_cache.py).
# Every suite task runs on AMD EPYC 25/17 or 26/17 (task.cpu): a row is recorded, and reused, on the class it ran on.
# The gangs run tests/test_multihost.py's pools across four containers of an armada gang, one host each, which reach
# each other through armada's relay; `ci.py gangs` reads them from the test's MODELS, so every case runs in one.
#
# Usage: tools/ci/suite.sh <sha>   from a checkout of <sha>, with the environment:
#   ARMADA           the armada CLI's command (default `armada`)
#   ARMADA_URL, ARMADA_TOKEN   for a red gang's logs, which the map's summary does not hold
#   LABEL            the jobs' label
#   PRIORITY         the jobs' armada priority (none: armada's default)
#   DEW_TEST_CACHE   the test cache's policy (tools/armada/ci.py policy; `off` runs every file)
#   DEW_WHOLE_SUITE  1 for the whole suite: the whole_suite cells, every file on the newest Python, and the release
#                    checks after it (tools/armada/session.py `selected`, ci.py `plan`, `release-plan`)
#   GANGLESS         1 to run no gangs (a base/kernels push and a base/<producer> promote: main's trains run them)
#   POOL             the suite's pool (none: .armada.json's), and BACKGROUND, 1 for a run that takes a bounded share
#                    of the fleet with a longer deadline
#   PROVEN           non-empty where the commit was proven green already: nothing runs
#   OUT              where the clients' logs and the jobs' ids (jobs.txt) are kept (default: a new directory)
set -u
sha=$1
armada=${ARMADA:-armada}
out=${OUT:-$(mktemp -d)}
mkdir -p "$out"
priority=${PRIORITY:+--priority=$PRIORITY}
echo "the test cache's policy: ${DEW_TEST_CACHE:-}"
# Each client runs in a session of its own, so the interrupt a cancelled run sends this shell reaches it and it cancels
# its armada job; the shell waits on them, where a trap runs.
trap 'for group in ${gangs:-} ${suite:-}; do kill -INT -- "-$group" 2>/dev/null || true; done' INT TERM
jobs() {  # every armada job the clients named, kept for whoever cancels this run
  cat "$out"/gangs.log "$out"/suite.log "$out"/suite.err "$out"/release.log 2>/dev/null |
    grep -oE '\bjob [0-9]{14}-[0-9a-f]{8}' | cut -d' ' -f2 | sort -u > "$out/jobs.txt"
}
# The gangs' and release checks' own deadline: a background run's (BACKGROUND, main's daily whole run below every
# lane's), which waits for capacity behind them, a longer one (train 663's review).
side=1800
[ -z "${BACKGROUND:-}" ] || side=6000
if [ -z "${PROVEN:-}" ] && [ -z "${GANGLESS:-}" ]; then
  # A gang member lost before the start fails it within 300 s (armada#22).
  items=$(python3 tools/armada/ci.py gangs)
  echo "$items" | setsid $armada map --commit="$sha" --items=- --size=medium \
    --pool="$(echo "$items" | python3 -c 'import json, sys; print(sum(gang["gang"] for gang in json.load(sys.stdin)))')" \
    --deadline=$side --gang-start-timeout=300 $priority --label="${LABEL:-$sha}" -- python3 tools/armada/ci.py pytest \
    --python="$(tr -d '\n' < .python-version)" --report=/tmp/dew-gang.xml \
    -- -k "{select}" --basetemp=/tmp/dew-gang tests/test_multihost.py > "$out/gangs.log" 2>&1 &
  gangs=$!
fi
# Its own deadline sits above what every file takes on its containers when no row is cached: 2700 s; a cold suite
# (the cache's policy off: a lock, JAX's build or the recorder changed) compiles everything anew, 5400 s (train
# 254's cold suite left 780 tasks unstarted at 2,698 s); a background run takes a bounded share, 6000 s.
if [ -n "${BACKGROUND:-}" ]; then bound="--deadline=6000"
elif [ "${DEW_TEST_CACHE:-}" = off ]; then bound="--deadline=5400"
else bound="--deadline=2700"; fi
[ -z "${POOL:-}" ] || bound="$bound --pool=$POOL"
# A commit PROVEN runs no suite again, whatever its stored verdict or the cache's policy; the whole suite reads no
# proof, so it proves the commit anew.
if [ -z "${PROVEN:-}" ] && { [ "${DEW_TEST_CACHE:-}" = off ] || ! $armada verdict "$sha"; }; then
  setsid $armada run "$sha" $bound $priority --label="${LABEL:-$sha}" \
    > >(tee "$out/suite.log") 2> >(tee "$out/suite.err" >&2) &
  suite=$!
fi
failed=0
if [ -n "${suite:-}" ]; then wait "$suite" || failed=1; fi
jobs
# The whole suite's release checks, which its own tasks set aside, a task each requirement (ci.py release-plan), each
# pytest under uv's overlay of the requirement on the suite's environment, on the suite's CPU (2173). A verdict that
# cannot be read or a plan that fails fails the gate: no release check is skipped unseen.
if [ -n "${DEW_WHOLE_SUITE:-}" ]; then
  read=0
  $armada verdict "$sha" --json > "$out/verdict.json" || read=$?
  if [ "$read" -gt 1 ]; then
    echo "the suite's verdict could not be read (armada verdict exit $read): its release checks are unplanned"
    failed=1
  elif ! checks=$(python3 tools/armada/ci.py release-plan "$out/verdict.json"); then
    echo "the suite's release checks could not be planned (ci.py release-plan)"
    failed=1
  elif [ "$checks" != "[]" ]; then
    echo "$checks" | $armada map --commit="$sha" --items=- --size=medium \
      --pool="$(echo "$checks" | jq length)" --deadline=$side $priority --label="${LABEL:-$sha}" \
      --cpu="$(jq -r '[.task.cpu] | flatten | map("\(.vendor)/\(.family)/\(.model)") | join(",")' .armada.json)" \
      -- python3 tools/armada/ci.py release --python={python} --tests={tests} --requirement={release} \
      --out=/tmp/dew-release.json > "$out/release.log" 2>&1 || failed=1
    cat "$out/release.log"
  fi
fi
if [ -n "${gangs:-}" ]; then
  wait "$gangs" || failed=1
  echo "--- the multi-host gangs"
  cat "$out/gangs.log"
  # A red gang's logs, every rank's and attempt's with each one's relay (armada's GET /jobs/<id>/tasks/<i>/runs, and
  # each run's log): the map's summary names no cause, and a gang's cause can sit on a rank other than the one that
  # failed first.
  job=$(grep -oE '\bjob [0-9]{14}-[0-9a-f]{8}' "$out/gangs.log" | head -1 | cut -d' ' -f2)
  if [ -n "$job" ] && [ -n "${ARMADA_URL:-}" ]; then
    # A task row: its index, then "exit <code>" or what else ended it (TIMEOUT, LOST, CANCELLED).
    for task in $(awk '$1 ~ /^[0-9]+$/ && (($2 == "exit" && $3 != 0) || $2 ~ /^(TIMEOUT|LOST|CANCELLED)$/) { print $1 }' \
                    "$out/gangs.log"); do
      runs=$(curl -sS --compressed --fail-with-body "${ARMADA_URL%/}/jobs/$job/tasks/$task/runs" \
        -H "authorization: Bearer $ARMADA_TOKEN" -H "armada-protocol: 10") || runs='{"runs": []}'
      for run in $(echo "$runs" | jq -r '.runs[] | "\(.vessel).\(.attempt):\(.rank)"'); do
        echo "--- gang $task of job $job, rank ${run#*:} (${run%:*})"
        curl -sS --compressed --fail-with-body "${ARMADA_URL%/}/jobs/$job/tasks/$task/runs/${run%:*}/log" \
          -H "authorization: Bearer $ARMADA_TOKEN" -H "armada-protocol: 10" || echo "(armada stored no log for it)"
      done
    done
  fi
fi
jobs
exit $failed
