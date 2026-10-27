"""Probe (not for landing): where the slow tests of row 1851 spend their time."""

import time

import pytest


def clock(label, fn):
    start = time.perf_counter()
    fn()
    print("PROBE", label, f"{time.perf_counter() - start:.1f}")


UNETS = {
    "current": ({"stages": [{"features": 32, "heads": 4, "depth": 2}, {"features": 64, "heads": 4, "depth": 2}],
                 "blocks_per_level": 1, "in_channels": 4, "out_channels": 4}, 16),
    "one_level": ({"stages": [{"features": 32, "heads": 4, "depth": 2}], "blocks_per_level": 1,
                   "in_channels": 4, "out_channels": 4}, 8),
    "one_level_no_middle": ({"stages": [{"features": 32, "heads": 4, "depth": 2}], "blocks_per_level": 1,
                             "in_channels": 4, "out_channels": 4, "middle_attention": False}, 8),
}


@pytest.mark.mesh(devices=8)
@pytest.mark.parametrize("seeds", [8, 16])
@pytest.mark.parametrize("variant", list(UNETS))
@pytest.mark.parametrize("layout", ["fsdp2_stage2", "stage2_sequence2"])
def test_unet(variant, layout, seeds, monkeypatch):
    import test_stage_blocks as blocks

    fields, res = UNETS[variant]
    monkeypatch.setattr(blocks, "RES", res)
    monkeypatch.setitem(blocks.CASES, "unet", ("unet_2d_condition", fields, "latents"))
    monkeypatch.setitem(blocks.SEEDS, "unet", seeds)
    clock(f"unet {variant} {layout} {seeds}",
          lambda: blocks.test_a_stack_of_like_blocks_over_stages_computes_what_one_device_computes("unet", layout))


@pytest.mark.parametrize("seeds", [16, 32])
def test_packing(seeds, monkeypatch):
    import test_decision_packing as packing

    monkeypatch.setitem(packing.PACKING_SEEDS, "marker", seeds)
    clock(f"packing marker {seeds}", lambda: packing.test_a_packed_window_decides_each_row_as_the_row_alone("marker"))


@pytest.mark.parametrize("seeds", [32, 64])
def test_option_slots(seeds, monkeypatch):
    import test_decision_option_slots as slots

    monkeypatch.setattr(slots, "OPTION_SEEDS", seeds)
    clock(f"slots {seeds}", slots.test_a_ragged_batch_trains_on_compacted_options_as_on_the_grid)
