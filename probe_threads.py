"""Which line of tests/conftest.py starts the OS threads: conftest executed statement by statement (its top-level
statements, in order), the process's OS threads counted after each that changes the count."""
import ast, os, sys, time
sys.path.insert(0, "tests")
def threads():
    return len(os.listdir("/proc/self/task"))
source = open("tests/conftest.py").read()
tree = ast.parse(source)
namespace = {"__name__": "conftest", "__file__": os.path.abspath("tests/conftest.py")}
last = threads()
print("start", last, flush=True)
for node in tree.body:
    code = compile(ast.Module([node], []), "tests/conftest.py", "exec")
    t = time.perf_counter()
    exec(code, namespace)
    now = threads()
    took = time.perf_counter() - t
    if now != last or took > 0.2:
        print(f"line {node.lineno}: {ast.get_source_segment(source, node).splitlines()[0][:100]!r} threads {last}->{now} {took:.2f}s", flush=True)
    last = now
import jax
print("backends initialized:", jax._src.xla_bridge.backends_are_initialized() if hasattr(jax._src.xla_bridge, "backends_are_initialized") else "?", flush=True)
