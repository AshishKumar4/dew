set -u
out=$1
py=.venv-3.12/bin/python
for i in 1 2 3 4 5; do
  $py - >> "$out/startup.jsonl" 2>/dev/null <<'PY'
import json, subprocess, sys, time
def t(argv):
    s = time.perf_counter(); r = subprocess.run(argv, capture_output=True, text=True); return round(time.perf_counter() - s, 3), r.returncode
py = sys.executable
print(json.dumps({"import_dew": t([py, "-c", "import dew"]),
                  "collect_data": t([py, "-m", "pytest", "-q", "--collect-only", "-p", "no:cacheprovider", "tests/test_config_paths.py"]),
                  "collect_cache": t([py, "-m", "pytest", "-q", "--collect-only", "-p", "no:cacheprovider", "tests/test_compilation_cache.py"]),
                  "run_cache": t([py, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests/test_compilation_cache.py", "-k", "users_cache_directory"])}))
PY
done
