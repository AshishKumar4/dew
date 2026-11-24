"""The attention template's forward over a call of several key runs a query as its register model prices it,
compiled for a compile-only GPU (row 6766)."""

import jax
import jax.numpy as jnp
import pytest
from generation_support import COMPILES_ON_CPU

from dew.nn.attention import scaled_dot_product_attention
from dew.nn.kernels import generation

pytestmark = [pytest.mark.kernel, COMPILES_ON_CPU]


def _runs(count: int, rows: int, tokens: int) -> jax.Array:
    """`count` key runs a query `[rows, tokens, count, 2]`: the first key of each 64-key block before the
    query's own, then the 64 keys up to its own."""
    query = jnp.arange(tokens, dtype=jnp.int32)
    runs = [jnp.stack([jnp.minimum(query, run * 64), jnp.minimum(query, run * 64) + 1], -1)
            for run in range(count - 1)]
    runs.append(jnp.stack([jnp.maximum(query - 63, 0), query + 1], -1))
    return jnp.broadcast_to(jnp.stack(runs, 1)[None], (rows, tokens, count, 2))


def test_a_forward_over_several_runs_compiles_every_config_it_certifies_spill_free(without_deterministic_ops):
    """A causal forward over three key runs a query (eight heads 64 wide over 2 x 2,048 tokens) compiles on
    a compile-only H100 with no kernel its derivation certifies spilled or refused: its runs' union
    counted, one select an entry (`attention_algebra.seen`), each further run's bounds priced as the rows its
    walk holds (`attention_mosaic_registers._runs_bounds`); on main its walk at 64 x 128 three consumers
    spilled 120 bytes at its raised 160 and stepped down (FS-6766-MAIN1, row 6766; four runs' scaled rewalk
    still steps down once, that row's)."""
    from fla_gdn_support import KernelCase
    from kernel_compile import compile_mosaic_kernels, mosaic_failures

    q = jax.ShapeDtypeStruct((2, 2048, 8, 64), jnp.bfloat16)
    runs = _runs(3, 2, 2048)

    def forward(q, k, v):
        return scaled_dot_product_attention(q, k, v, causal=True, key_runs=runs, implementation="template")

    with generation.assumed("sm90"):
        found = compile_mosaic_kernels([KernelCase("three runs", lambda: (forward, (q, q, q)))], "sm90")
    assert not (failed := mosaic_failures(found)), "\n".join(failed)
    forwards = [certified for record in found for certified in record.get("certifications", [])
                if certified["op"] == "MosaicForward"]
    # Each kernel's registers, spilled bytes and refusal (`kernel_compile._certified_since`); main's walk at
    # 64 x 128 refused at 244 bytes, eleven configs certified on its way down.
    spilled = {name: kernel for certified in forwards for name, kernel in certified["kernels"].items()
               if kernel[1] or kernel[2]}
    assert forwards and not spilled, forwards
