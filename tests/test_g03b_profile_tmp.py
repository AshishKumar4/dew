"""Temporary phase timings for one run-once matrix cell; removed with the restructure."""

import time

import jax
import pytest
from tool_support import load


@pytest.mark.mesh(devices=8)
def test_unet_run_once_phases(tmp_path):
    tool = load("layout_parity")
    times = {}

    def timed(name, fn):
        started = time.perf_counter()
        found = fn()
        times[name] = round(time.perf_counter() - started, 2)
        return found

    with jax.enable_x64(new_val=True):
        case, reference = timed("model_case", lambda: tool.model_case("unet", "float32", {}, {}))
        timed("source_digest", tool.source_digest)
        timed("drawn_batch", lambda: tool.drawn_batch(case, 0))
        held = tool.References(tmp_path, prepare=True, reserve=4 << 30)
        judge = tool.Judge(reference, held, "unet")
        timed("audit", judge.audit)
        timed("reference_arm_construct", lambda: judge.arm)
        timed("reference_first", lambda: judge.reference(0))
        timed("reference_second", lambda: judge.reference(0))
        arm = timed("layout_arm_construct", lambda: tool.Arm(case, tool.LAYOUTS["tensor4"]))
        timed("layout_arm_run", lambda: arm(0))
        timed("ran_whole", lambda: tool.ran("unet", "tensor4", arm, judge))
        timed("flops_per_device", lambda: judge.flops_per_device)
        trainer = timed("trainer_build", lambda: tool._trainer(case, tool.LAYOUTS["fsdp2_tensor2"],
                                                                record=True))
        state = timed("trainer_place", lambda: trainer.place()[0])
        batch = timed("placed_batch", lambda: tool.placed(tool.drawn_batch(case, 0), trainer.device_mesh))
        timed("trainer_compile_other_layout", lambda: trainer.compile(state, batch))
    assert not times, times
