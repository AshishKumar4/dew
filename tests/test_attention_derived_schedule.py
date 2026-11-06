"""The derived fp16 forward over raw products on par with the reference path through the Mosaic mainloop
(check 1). Its schedule's speed against the hand step's, a
consumer's softmax before its value tile's acquisition and a launch that lowers its exponentials exactly
taking no range reduction of its own (docs/design/algebra.md §6.9), is an H100 operator row's measurement
(ledger 3720), never a count of the kernel's instructions."""

import jax.numpy as jnp
import pytest
import test_attention_mosaic as mosaic
from generation_support import INTERPRETS


@INTERPRETS
@pytest.mark.parametrize("call", ["full", "causal"])
def test_a_derived_fp16_forward_over_raw_products_is_on_par_interpreted(call):
    """A derived fp16 forward over the raw products at a positive factor (the declaration's extracted step,
    `logits_of`: the hand step scales its maximum once) through the Mosaic mainloop, interpreted, on par with
    the reference path (check 1)."""
    mosaic._check_forward(call, jnp.float16, mosaic.SHAPES["interpreted"], natural=False, interpret=True)
