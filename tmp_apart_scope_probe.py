import sys, os, tempfile, traceback, contextlib
sys.path[:0] = ["src", "tests", "."]
os.environ["DEW_RUNG_RECORDS"] = tempfile.mkdtemp()
import jax, optax
import test_narrow as t
from placement_parity import x64
from arm_arithmetic import summed_as_run
from dew.algebra import passes
free = 2 ** 39
jax.Device.memory_stats = lambda device: {"bytes_limit": 2 ** 40, "bytes_in_use": 2 ** 40 - (2 ** 39 if x64() else free)}
def trained(*, copies, rows=8, longer=None, steps=0):
    return t._trained(optax.sgd(0.5), copies=copies, steps=steps, accumulation=2, split=1, remat="full",
                      rows=rows, longer=longer)
def used(rows, *, copies):
    return free - trained(copies=copies, rows=rows)[2].plan["probes"][0]["headroom"]
baseline = [used(8, copies=True), used(64, copies=True), used(64, copies=False)]
free = max(baseline[0], baseline[2])
print("K calibration", baseline, "free", free, "precision", jax.config.jax_default_matmul_precision, flush=True)
for label, scope in [("bare", contextlib.nullcontext), ("summed", summed_as_run),
                     ("unrewritten", lambda: passes(on=False))]:
    for copies in [True, False]:
        try:
            with scope():
                print("K scope", label, copies, "precision", jax.config.jax_default_matmul_precision, flush=True)
                _, _, trainer = trained(copies=copies, longer=64, steps=3)
            print("K PASS", label, copies, trainer.plan["probes"], flush=True)
        except Exception as error:
            print("K FAIL", label, copies, type(error).__name__, str(error), flush=True)
