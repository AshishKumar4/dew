"""The derived forward's kernel against the hand-written one's, compiled for an H100 on the CPU: a derived
step schedules its two consumers as the hand step does, and a launch that lowers its exponentials exactly
takes no range reduction of its own (docs/design/algebra.md §6.9); and the derived fp16 forward over raw
products on par with the reference path (check 1), its time against the hand step's an operator row's."""

import functools
import re
import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest
import test_attention_mosaic as mosaic
from fla_gdn_support import KernelCase
from generation_support import COMPILES_ON_CPU, INTERPRETS
from kernel_compile import _compiled

from dew.algebra import forms
from dew.nn.attention import scaled_dot_product_attention


@functools.cache
def _forward_ptx(shape: tuple[int, int, int, int, int], dtype: str, causal: bool,
                 softcap: float | None = None) -> dict[bool, str]:
    """The forward kernel's PTX for an H100, hand-written (False) and derived (True), at `shape` (batch,
    tokens, heads, key heads, width): compiled once a process, as JAX prints none for a lowering it cached."""
    batch, tokens, heads, kv_heads, width = shape
    args = (jax.ShapeDtypeStruct((batch, tokens, heads, width), dtype),
            *[jax.ShapeDtypeStruct((batch, tokens, kv_heads, width), dtype)] * 2)

    def attend(q, k, v):
        return scaled_dot_product_attention(q, k, v, implementation="template", causal=causal,
                                            softcap=softcap)

    found = {}
    for derived in (False, True):
        record = {}
        with tempfile.TemporaryDirectory() as directory, forms.derived(derived):
            _compiled(KernelCase(str(derived), lambda: (attend, args)), "sm90", record, directory)
            assert "error" not in record, record
            [ptx] = Path(directory).glob("attention_forward__*.ptx")
            found[derived] = ptx.read_text()
    return found


def _exponentials_before_values(ptx: str) -> int:
    """The exponentials a step takes between its logits' products and its value tile's acquisition (the
    first wait past the products): the softmax a consumer computes while its partner holds the tensor
    cores."""
    after_logits = ptx.split("wgmma.wait_group.sync.aligned 0;", 1)[1]
    before_values = after_logits.split("mbarrier.try_wait.parity.shared.b64", 1)[0]
    return len(re.findall(r"\bex2\.approx(?:\.ftz)?\.f32\b", before_values))


@COMPILES_ON_CPU
@pytest.mark.parametrize("width", [64, 128])
def test_a_derived_step_computes_its_softmax_before_it_acquires_its_values(width):
    """A derived fp16 step acquires its value tile where its form first reads it, past its weights, as the
    hand step does: acquired first, its wait for its partner's turn took the softmax off the tensor cores'
    overlap (1.3x the hand forward's time on an H100)."""
    found = {derived: _exponentials_before_values(ptx)
             for derived, ptx in _forward_ptx((2, 4096, 32, 8, width), "float16", causal=False).items()}
    assert found[False] > 0
    assert found[True] >= found[False]


@COMPILES_ON_CPU
def test_an_exactly_lowered_launch_takes_no_range_reduction_of_its_own():
    """A softcap's launch lowers its exponentials exactly (its tanh does): the hand step takes `jnp.exp`
    there as the derived one does, no conversions of each exponent to an integer and back of its own (the
    hand softcap forward's were 0.6 ms against the derived 0.26 on an H100)."""
    found = {derived: len(re.findall(r"\bcvt\.(?:rzi\.s32\.f32|rn\.f32\.s32)\b", ptx))
             for derived, ptx in _forward_ptx((1, 4096, 8, 4, 256), "float16", causal=True,
                                              softcap=50.0).items()}
    assert found[False] <= found[True]


@INTERPRETS
@pytest.mark.parametrize("call", ["full", "causal"])
def test_a_derived_fp16_forward_over_raw_products_is_on_par_interpreted(call):
    """A derived fp16 forward over the raw products at a positive factor (the declaration's extracted step,
    `logits_of`: the hand step scales its maximum once) through the Mosaic mainloop, interpreted, on par with
    the reference path (check 1). Its kernel time against the hand step's is an operator row's measurement,
    never a count of its instructions."""
    with forms.derived():
        mosaic._check_forward(call, jnp.float16, mosaic.SHAPES["interpreted"], natural=False, interpret=True)
