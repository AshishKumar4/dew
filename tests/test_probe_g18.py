"""Scratch probe (row 3197): which branch of `spread` the UNet's biases take, one device and stage2_sequence2."""
import collections

import jax
import pytest

import test_stage_blocks as sb
from dew.nn import precision

SEEN = collections.Counter()


@pytest.mark.mesh(devices=8)
@pytest.mark.parametrize("layout", ["one", "stage2_sequence2"])
def test_probe(layout, monkeypatch):
    real_wide, real_doubled = precision._sums_in_float64, precision._doubled

    def wide(*operands):
        found = real_wide(*operands)
        SEEN[(layout, "wide" if found else "narrow", jax.config.jax_default_matmul_precision,
              precision._intercepted(), precision.wide_cpu_arithmetic())] += 1
        return found

    monkeypatch.setattr(precision, "_sums_in_float64", wide)
    name, fields, kind = sb.CASES["unet"]
    model = sb.models.build(name, **fields)
    draw, call = sb.drawn(kind), sb.call_of(kind)
    inputs, cotangent = draw(0)
    variables = sb.initialized(model, kind, inputs)
    if layout == "one":
        run = sb.on_mesh(sb.MeshSpec(), 1, model, variables, call)
    else:
        spec, microbatches = sb.LAYOUTS[layout]
        run = sb.on_mesh(spec, 8, model, variables, call, microbatches)
    run(inputs, cotangent)
    print("\nPROBE", layout, dict(SEEN))
