import sys, os, tempfile, contextlib
sys.path[:0] = ["src", "tests", "."]
os.environ["DEW_RUNG_RECORDS"] = tempfile.mkdtemp()
import lane_environment
os.environ.update(lane_environment.configured(os.environ))
import jax, optax, pytest
import test_narrow as t
from arm_arithmetic import summed_as_run
from dew.algebra import passes
free = 2 ** 39
jax.Device.memory_stats = lambda device: {"bytes_limit": 2 ** 40, "bytes_in_use": 2 ** 40 - free}
def trained(copies, rows):
    return t._trained(optax.sgd(0.5), copies=copies, steps=0, accumulation=2, split=1, remat="full",
                      rows=rows)[2]
for label, scope in [("bare", contextlib.nullcontext), ("summed", summed_as_run)]:
    with scope():
        for copies, rows in [(True, 8), (True, 64), (False, 64)]:
            trainer = trained(copies, rows)
            p = trainer.plan["probes"][0]
            print("K footprint", label, copies, rows, "used", free - p["headroom"],
                  "temp", p["temp"], "argument", p["argument"], flush=True)
for label, scope in [("original", contextlib.nullcontext), ("same-arithmetic-calibration", summed_as_run)]:
    try:
        with pytest.MonkeyPatch.context() as patch, scope():
            t.test_a_run_whose_shapes_plan_their_copies_apart_steps_each_as_its_plan_reads(patch)
        print("K TEST PASS", label, flush=True)
    except Exception as error:
        print("K TEST FAIL", label, type(error).__name__, str(error), flush=True)
