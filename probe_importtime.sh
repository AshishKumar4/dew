set -u
out=$1
py=.venv-3.12/bin/python
$py -X importtime -c "import dew" 2> "$out/dew.importtime" > /dev/null
for i in 1 2 3; do /usr/bin/time -f "%e" $py -c "import dew" 2>> "$out/dew.wall"; done
for i in 1 2 3; do /usr/bin/time -f "%e" $py -c "import jax" 2>> "$out/jax.wall"; done
$py -X importtime -c "import sys; sys.path.insert(0, 'tests'); import conftest" 2> "$out/conftest.importtime" > /dev/null
$py - > "$out/threads.txt" 2>&1 <<'PY'
import os, sys, threading, time
sys.path.insert(0, "tests")
def count(stage):
    names = [open(f"/proc/self/task/{t}/comm").read().strip() for t in os.listdir("/proc/self/task")]
    from collections import Counter
    print(stage, len(names), dict(Counter(names)), flush=True)
count("start")
import dew
count("import dew")
import jax
jax.config.read("jax_platforms")
count("jax config")
import conftest
count("conftest")
PY
$py -m pytest -q -p no:cacheprovider tests/test_barren_reads.py > "$out/barren.txt" 2>&1; echo "exit=$?" >> "$out/barren.txt"
