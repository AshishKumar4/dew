import json, sys, time, tempfile
from pathlib import Path
sys.path.insert(0, "tests")
import test_distribution as t
processes = 4
def run(arguments, stop):
    began = time.perf_counter()
    pool = t.start("--processes-per-host", str(processes), "--", sys.executable, str(t.WORKER),
                   "--mesh", json.dumps({"fsdp": processes}), *arguments, devices=1)
    if stop:
        out = t.stopped_by_sigterm(pool, "] step 2/")
    else:
        done = t.finished(pool, timeout=300); out = (done.returncode, done.stdout + done.stderr)
    text = out[1]
    first = [l for l in text.splitlines() if "first step after" in l][:1]
    print(f"{'stopped' if stop else 'run'} {time.perf_counter() - began:.1f}s {first}", flush=True)
    return out
began = time.perf_counter()
t.resumed_where_it_stopped(Path(tempfile.mkdtemp()), run)
print(f"total {time.perf_counter() - began:.1f}s")
