"""The pre-imported parent: its threads and memory after importing jax, Dew's top modules and tests/conftest; then a
forked child, bounded by an alarm, doing nothing, then collecting one test file in-process: whether it ends, its time and
private memory."""
import faulthandler, json, os, sys, threading, time

def mem(pid):
    out = {}
    with open(f"/proc/{pid}/smaps_rollup") as f:
        for line in f:
            k, v = line.split(":", 1)
            if k in ("Rss", "Pss", "Private_Clean", "Private_Dirty"):
                out[k] = int(v.split()[0]) // 1024
    return out

def say(**kw):
    print(json.dumps(kw), flush=True)

sys.path.insert(0, "tests")
start = time.perf_counter()
import jax, jax.numpy, flax, optax, numpy, dew, dew.training, dew.nn, dew.data  # noqa
say(stage="imports", s=round(time.perf_counter() - start, 2), threads=threading.active_count(),
    os_threads=len(os.listdir(f"/proc/{os.getpid()}/task")), mem=mem(os.getpid()))
import conftest  # noqa
say(stage="conftest", s=round(time.perf_counter() - start, 2), threads=threading.active_count(),
    os_threads=len(os.listdir(f"/proc/{os.getpid()}/task")), mem=mem(os.getpid()))

def child(work, bound=120):
    r, w = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(r)
        faulthandler.dump_traceback_later(bound - 5, exit=True)
        t = time.perf_counter()
        work()
        os.write(w, json.dumps({"s": round(time.perf_counter() - t, 2), "mem": mem(os.getpid())}).encode())
        os._exit(0)
    os.close(w)
    deadline = time.time() + bound
    while time.time() < deadline:
        done, status = os.waitpid(pid, os.WNOHANG)
        if done:
            data = os.read(r, 65536)
            return {"status": status, **(json.loads(data) if data else {})}
        time.sleep(0.2)
    os.kill(pid, 9); os.waitpid(pid, 0)
    return {"status": "hung"}

say(stage="fork-nothing", **child(lambda: None))
say(stage="fork-jit", **child(lambda: jax.jit(lambda x: x + 1)(numpy.ones(3)).block_until_ready()))
import pytest  # noqa
for name in sys.argv[1:]:
    say(stage="fork-collect", file=name, **child(lambda: pytest.main(["-q", "--collect-only", "-p", "no:cacheprovider", name])))
