"""Temporary phase timings for one run-once matrix cell; removed with the restructure."""

import time

import jax
import pytest
from tool_support import load


@pytest.mark.mesh(devices=8)
def test_unet_run_once_phases(tmp_path_factory):
    times = {}
    counts = {}

    def timed(name, fn):
        def wrapped(*args, **kwargs):
            started = time.perf_counter()
            found = fn(*args, **kwargs)
            times[name] = round(times.get(name, 0.0) + time.perf_counter() - started, 2)
            counts[name] = counts.get(name, 0) + 1
            return found
        return wrapped

    started = time.perf_counter()
    fixture_tool = load("layout_parity")
    times["load_fixture_module"] = round(time.perf_counter() - started, 2)
    draws = fixture_tool.References(tmp_path_factory.mktemp("layout-draws"), prepare=True,
                                    reserve=4 << 30)
    started = time.perf_counter()
    tool = load("layout_parity")
    times["load_test_module"] = round(time.perf_counter() - started, 2)
    for module, prefix in ((fixture_tool, "fixture_"), (tool, "")):
        module.source_digest = timed(f"{prefix}source_digest", module.source_digest.__wrapped__
                                     if hasattr(module.source_digest, "__wrapped__")
                                     else module.source_digest)
    fixture_tool.References.held = timed("held", fixture_tool.References.held)
    fixture_tool.References.noted = timed("noted", fixture_tool.References.noted)
    tool.model_case = timed("model_case", tool.model_case)
    tool.Judge.audit = timed("audit", tool.Judge.audit)
    tool.Judge.reference = timed("judge_reference", tool.Judge.reference)
    tool.Judge.batch_seed = timed("batch_seed", tool.Judge.batch_seed)
    tool.audited = timed("audited", tool.audited)
    tool.recording_loss = timed("recording_loss", tool.recording_loss)
    tool._trainer = timed("trainer_build", tool._trainer)
    tool.Arm.__init__ = timed("arm_construct", tool.Arm.__init__)
    tool.Arm.__call__ = timed("arm_call", tool.Arm.__call__)
    tool.ran = timed("ran", tool.ran)
    tool.drawn_batch = timed("drawn_batch", tool.drawn_batch)
    with jax.enable_x64(new_val=True):
        rows = timed("run_whole", tool.run)(
            ["unet"], ("tensor4",), dtype="float32", seeds=0, windows={"dense_grpo": 4}, mixture={},
            objective={}, references=draws, speak=lambda line: None, keep=lambda rows: None)
    lines = [f"{name}={seconds} (x{counts.get(name, 1)})" for name, seconds in times.items()]
    raise AssertionError("phases:\n" + "\n".join([*lines, f"row={rows}"]))
