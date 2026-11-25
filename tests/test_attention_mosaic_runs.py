"""The attention template's forward over a call of several key runs a query as its register model prices it,
compiled for a compile-only GPU (row 6766)."""

import jax
import jax.numpy as jnp
import pytest
from generation_support import COMPILES_ON_CPU

from dew.nn.attention import scaled_dot_product_attention
from dew.nn.kernels import generation

pytestmark = [pytest.mark.kernel, COMPILES_ON_CPU]


@pytest.mark.parametrize("count", [2, 3])
def test_a_forward_over_several_runs_compiles_every_config_it_certifies_spill_free(count,
                                                                                  without_deterministic_ops):
    """A causal forward over `count` key runs a query (eight heads 64 wide over 2 x 2,048 tokens, runs traced
    as a call's operand) compiles on a compile-only H100 with no kernel its derivation certifies spilled or
    refused: its runs' union counted, one select an entry (`attention_algebra.seen`), each further run's
    bounds priced as the rows its walk holds (`attention_mosaic_registers._runs_bounds`), and two runs'
    walk charged a logits tile more (a conservative charge for a measured spill). On main three runs' walk
    at 64 x 128 three consumers spilled 120 bytes at its raised 160 and stepped down (FS-6766-MAIN1); two
    runs' (MoBA's top_k 2) spilled 220 where three spilled none (FS-6766-ATT1, row 6766)."""
    from fla_gdn_support import KernelCase
    from kernel_compile import compile_mosaic_kernels, mosaic_failures

    q = jax.ShapeDtypeStruct((2, 2048, 8, 64), jnp.bfloat16)
    runs = jax.ShapeDtypeStruct((2, 2048, count, 2), jnp.int32)

    def forward(q, k, v, runs):
        return scaled_dot_product_attention(q, k, v, causal=True, key_runs=runs, implementation="template")

    with generation.assumed("sm90"):
        found = compile_mosaic_kernels([KernelCase(f"{count} runs", lambda: (forward, (q, q, q, runs)))],
                                       "sm90")
    assert not (failed := mosaic_failures(found)), "\n".join(failed)
    forwards = [certified for record in found for certified in record.get("certifications", [])
                if certified["op"] == "MosaicForward"]
    # Each kernel's registers, spilled bytes and refusal (`kernel_compile._certified_since`); main's walk at
    # 64 x 128 refused at 244 bytes, eleven configs certified on its way down.
    spilled = {name: kernel for certified in forwards for name, kernel in certified["kernels"].items()
               if kernel[1] or kernel[2]}
    assert forwards and not spilled, forwards
