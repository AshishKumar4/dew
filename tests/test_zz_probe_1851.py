"""Probe (not for landing): where the slow tests of row 1851 spend their time."""

import time

import jax
import jax.numpy as jnp
import numpy as np
import pytest


def timed(label, fn, *args):
    start = time.perf_counter()
    out = fn(*args)
    jax.block_until_ready(out)
    print("PROBE", label, f"{time.perf_counter() - start:.2f}")
    return out


UNETS = {
    "current": ({"stages": [{"features": 32, "heads": 4, "depth": 2}, {"features": 64, "heads": 4, "depth": 2}],
                 "blocks_per_level": 1, "in_channels": 4, "out_channels": 4}, 16),
    "one_level": ({"stages": [{"features": 32, "heads": 4, "depth": 2}], "blocks_per_level": 1,
                   "in_channels": 4, "out_channels": 4}, 8),
    "one_level_no_middle": ({"stages": [{"features": 32, "heads": 4, "depth": 2}], "blocks_per_level": 1,
                             "in_channels": 4, "out_channels": 4, "middle_attention": False}, 8),
}


@pytest.mark.mesh(devices=8)
@pytest.mark.parametrize("variant", list(UNETS))
@pytest.mark.parametrize("layout", ["fsdp2_stage2", "stage2_sequence2"])
def test_unet(variant, layout, monkeypatch):
    import test_stage_blocks as blocks
    from reference_error import _arms, _truth_program

    from dew.registry import models
    from dew.training import MeshSpec

    fields, res = UNETS[variant]
    monkeypatch.setattr(blocks, "RES", res)
    model = models.build("unet_2d_condition", **fields)
    draw, call = blocks.drawn("latents"), blocks.call_of("latents")
    inputs, _ = draw(0)
    variables = timed(f"{variant} {layout} init", blocks.initialized, model, "latents", inputs)
    spec, micro = blocks.LAYOUTS[layout]
    assert blocks.runs_a_pipeline(spec, model, variables, call, inputs, micro)
    mine, theirs = _arms(blocks.on_mesh(spec, 8, model, variables, call, micro),
                         blocks.on_mesh(MeshSpec(), 1, model, variables, call))
    truth = _truth_program(blocks.float64_pulled(model, variables, call))
    for seed in range(3):
        drawn = draw(seed)
        timed(f"{variant} {layout} mesh seed{seed}", mine, *drawn)
        timed(f"{variant} {layout} one seed{seed}", theirs, *drawn)
        with jax.enable_x64(True):
            from reference_error import widened
            timed(f"{variant} {layout} truth seed{seed}", truth, *widened(drawn))


@pytest.mark.parametrize("arm", ["marker"])
def test_packing(arm, monkeypatch):
    import test_decision_packing as packing
    from reference_error import _truth_program

    calls = {}

    def recorded(dew_fn, reference_fn, draw, label, *, seeds, truth_fn=None, orders=None):
        calls.update(dew=dew_fn, reference=reference_fn, draw=draw, truth=truth_fn)

    monkeypatch.setattr(packing, "assert_as_exact", recorded)
    objective = packing.objective_for(arm)
    start = time.perf_counter()
    packing.assert_rows_decided_alone(objective, (objective.model.backbone.clone(dtype=jnp.float64),
                                                  objective.model.head.clone(dtype=jnp.float64)), 8)
    print("PROBE packing setup", f"{time.perf_counter() - start:.2f}")
    truth = _truth_program(calls["truth"])
    from reference_error import widened
    for seed in range(3):
        drawn = timed(f"packing draw seed{seed}", calls["draw"], seed)
        timed(f"packing packed seed{seed}", calls["dew"], *drawn)
        timed(f"packing alone seed{seed}", calls["reference"], *drawn)
        with jax.enable_x64(True):
            timed(f"packing truth seed{seed}", truth, *widened(drawn))


def test_option_slots(monkeypatch):
    import test_decision_option_slots as slots
    from reference_error import _arms, widened

    calls = {}

    def recorded(dew_fn, reference_fn, draw, label, *, seeds, truth_fn=None, orders=None):
        calls.update(dew=dew_fn, reference=reference_fn, draw=draw, truth=truth_fn)
        raise StopIteration

    monkeypatch.setattr(slots, "assert_as_exact", recorded)
    start = time.perf_counter()
    with pytest.raises(StopIteration):
        slots.test_a_ragged_batch_trains_on_compacted_options_as_on_the_grid()
    print("PROBE slots setup", f"{time.perf_counter() - start:.2f}")
    mine, theirs = _arms(calls["dew"], calls["reference"])
    for seed in range(4):
        drawn = timed(f"slots draw seed{seed}", calls["draw"], seed)
        timed(f"slots compact seed{seed}", mine, *drawn)
        timed(f"slots grid seed{seed}", theirs, *drawn)
        with jax.enable_x64(True):
            timed(f"slots truth seed{seed}", calls["truth"], *widened(drawn))
    print("PROBE slots arrays", [np.shape(leaf) for leaf in jax.tree.leaves(calls["draw"](0))][:6])
