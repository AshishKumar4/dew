"""The pre-imported parent's memory: RSS and PSS after importing jax, Dew's top modules and tests/conftest, and a
forked child's private memory after it collects one test file (pytest in-process), against the same file collected
from a fresh interpreter."""
import json, os, subprocess, sys, time

def mem(pid):
    rss = pss = private = 0
    with open(f"/proc/{pid}/smaps_rollup") as f:
        for line in f:
            k, v = line.split(":", 1)
            kb = int(v.split()[0])
            if k == "Rss": rss = kb
            elif k == "Pss": pss = kb
            elif k in ("Private_Clean", "Private_Dirty"): private += kb
    return {"rss_mb": rss // 1024, "pss_mb": pss // 1024, "private_mb": private // 1024}

sys.path.insert(0, "tests")
start = time.perf_counter()
import jax, jax.numpy, flax, optax, numpy, dew, dew.training, dew.nn, dew.data  # noqa
import conftest  # noqa
parent = {"import_s": round(time.perf_counter() - start, 2), **mem(os.getpid())}
files = sys.argv[1:]
rows = []
for name in files:
    r, w = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(r)
        import pytest
        t = time.perf_counter()
        pytest.main(["-q", "--collect-only", "-p", "no:cacheprovider", name])
        took = time.perf_counter() - t
        os.write(w, json.dumps({"collect_s": round(took, 2), **mem(os.getpid())}).encode())
        os._exit(0)
    os.close(w)
    data = b""
    while chunk := os.read(r, 65536):
        data += chunk
    os.waitpid(pid, 0)
    child = json.loads(data)
    t = time.perf_counter()
    fresh = subprocess.run([sys.executable, "-c", f"import pytest, os; pytest.main(['-q', '--collect-only', '-p', 'no:cacheprovider', {name!r}]); "
                            "print('MEM', open(f'/proc/{os.getpid()}/smaps_rollup').read().split('Rss:')[1].split()[0])"],
                           capture_output=True, text=True)
    fresh_rss = int(fresh.stdout.split("MEM")[-1].split()[0]) // 1024 if "MEM" in fresh.stdout else None
    rows.append({"file": name, "child": child, "fresh_s": round(time.perf_counter() - t, 2), "fresh_rss_mb": fresh_rss})
print(json.dumps({"parent": parent, "rows": rows}, indent=1))
