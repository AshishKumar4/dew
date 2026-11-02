"""The derived forward's kernel against the hand-written one's, compiled for an H100 on the CPU: a derived
step schedules its two consumers as the hand step does, and a launch that lowers its exponentials exactly
takes no range reduction of its own (docs/design/algebra.md §6.9)."""

import functools
import re
import tempfile
from pathlib import Path

import jax
import pytest
from fla_gdn_support import KernelCase
from generation_support import COMPILES_ON_CPU
from kernel_compile import _compiled

from dew.algebra import forms
from dew.nn.attention import scaled_dot_product_attention

pytestmark = COMPILES_ON_CPU


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


@pytest.mark.parametrize("width", [64, 128])
def test_a_derived_step_computes_its_softmax_before_it_acquires_its_values(width):
    """A derived fp16 step acquires its value tile where its form first reads it, past its weights, as the
    hand step does: acquired first, its wait for its partner's turn took the softmax off the tensor cores'
    overlap (1.3x the hand forward's time on an H100)."""
    found = {derived: _exponentials_before_values(ptx)
             for derived, ptx in _forward_ptx((2, 4096, 32, 8, width), "float16", causal=False).items()}
    assert found[False] > 0
    assert found[True] >= found[False]


def test_an_exactly_lowered_launch_takes_no_range_reduction_of_its_own():
    """A softcap's launch lowers its exponentials exactly (its tanh does): the hand step takes `jnp.exp`
    there as the derived one does, no conversions of each exponent to an integer and back of its own (the
    hand softcap forward's were 0.6 ms against the derived 0.26 on an H100)."""
    found = {derived: len(re.findall(r"\bcvt\.(?:rzi\.s32\.f32|rn\.f32\.s32)\b", ptx))
             for derived, ptx in _forward_ptx((1, 4096, 8, 4, 256), "float16", causal=True,
                                              softcap=50.0).items()}
    assert found[False] <= found[True]


def test_a_derived_step_scales_its_maximum_not_its_logits():
    """A derived fp16 step over the raw products (a positive scale, `logits_of`) takes their maximum and
    scales it once, as the hand step does (a positive scale out of a maximum, `rules._linear`): its kernel
    multiplies no more than the hand one's, where scaling each logit before the maximum took 62 more
    multiplies a step at Qwen3's tile."""
    found = {derived: len(re.findall(r"\bmul(?:\.rn)?(?:\.ftz)?\.f32\b", ptx))
             for derived, ptx in _forward_ptx((2, 4096, 32, 8, 128), "float16", causal=False).items()}
    assert found[True] <= found[False]
