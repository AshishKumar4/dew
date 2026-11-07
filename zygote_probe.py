"""Per-task startup on an armada container, three ways, each in a fresh interpreter, each file of the sample: an empty
Python; Python importing JAX (backend not started) and Dew's test-wide imports (conftest's); and pytest collecting the
file (interpreter, imports, plugin load, collection). Prints one JSON line a file."""
import json, subprocess, sys, time

files = json.load(open(sys.argv[1]))["files"]
python = sys.executable

def timed(argv):
    start = time.perf_counter()
    done = subprocess.run(argv, capture_output=True, text=True)
    return round(time.perf_counter() - start, 3), done.returncode

bare = timed([python, "-c", "pass"])
imports = timed([python, "-c", "import jax, jax.numpy, flax, optax, numpy, dew, dew.training, dew.nn, dew.data; "
                 "import sys; sys.path.insert(0, 'tests'); import conftest"])
print(json.dumps({"bare": bare, "imports": imports}), flush=True)
for name in files:
    collect = timed([python, "-m", "pytest", "-q", "--collect-only", "-p", "no:cacheprovider", name])
    mod = name.replace("/", ".").removesuffix(".py").removeprefix("tests.")
    module = timed([python, "-c", f"import sys; sys.path.insert(0, 'tests'); import conftest, {mod}"])
    print(json.dumps({"file": name, "collect": collect, "module_import": module}), flush=True)
