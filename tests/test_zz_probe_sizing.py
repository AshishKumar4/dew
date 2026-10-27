"""Probe (not for landing): size the stage test's UNet cells, paired, by `seeds_for`."""

import time

import pytest
from reference_error import seeds_for
from test_stage_blocks import CASES, LAYOUTS, call_of, drawn, float64_pulled, initialized, on_mesh

from dew.registry import models
from dew.training import MeshSpec


@pytest.mark.mesh(devices=8)
@pytest.mark.parametrize("layout", ["fsdp2_stage2", "stage2_sequence2"])
def test_probe(layout):
    spec, microbatches = LAYOUTS[layout]
    name, fields, kind = CASES["unet"]
    model = models.build(name, **fields)
    draw, call = drawn(kind), call_of(kind)
    variables = initialized(model, kind, draw(0)[0])
    start = time.perf_counter()
    seeds = seeds_for(on_mesh(MeshSpec(), 1, model, variables, call), draw, f"unet {layout}",
                      truth_fn=float64_pulled(model, variables, call),
                      dew_fn=on_mesh(spec, 8, model, variables, call, microbatches))
    print("PROBE unet", layout, seeds, f"{time.perf_counter() - start:.0f}")
