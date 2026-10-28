"""Temporary compile-plan timings for a unet cell; removed with the restructure."""

import time

import jax
import pytest
from tool_support import load


@pytest.mark.mesh(devices=8)
def test_unet_compile_plans():
    tool = load("layout_parity")
    times = {}

    def timed(name, fn):
        started = time.perf_counter()
        found = fn()
        times[name] = round(time.perf_counter() - started, 2)
        return found

    with jax.enable_x64(new_val=True):
        case, reference = timed("model_case", lambda: tool.model_case("unet", "float32", {}, {}))
        judge = tool.Judge(reference, tool.References(), "unet")
        timed("audit", judge.audit)
        timed("reference_arm", lambda: judge.arm)
        ref_plan = repr(judge.arm.trainer.plan)
        timed("reference_run", lambda: judge.reference(0))
        arm = timed("layout_arm", lambda: tool.Arm(case, tool.LAYOUTS["tensor4"]))
        layout_plan = repr(arm.trainer.plan)
        timed("layout_run", lambda: arm(0))
    raise AssertionError("phases:\n" + "\n".join(
        [*(f"{name}={seconds}" for name, seconds in times.items()),
         f"reference_plan={ref_plan}", f"layout_plan={layout_plan}"]))
