"""The attention template's forward over a call of several key runs a query (MoBA's, a union of blocks) as
its register model prices it, compiled for a compile-only GPU (row 6766)."""

import jax
import jax.numpy as jnp
import pytest
from generation_support import COMPILES_ON_CPU

from dew.nn.block_sparse import moba_attention
from dew.nn.kernels import generation

pytestmark = [pytest.mark.kernel, COMPILES_ON_CPU]


def test_a_calls_union_of_runs_certifies_its_derived_forward_without_stepping_down(without_deterministic_ops):
    """A MoBA training step's forward (three key runs a query, the step memory test's block-sparse layer,
    eight heads 64 wide over 2,048 tokens) compiles on a compile-only H100 at the config its derivation
    gives, certified once and spill-free: its register model prices the runs' union mask and bounds
    (`attention_mosaic_registers._runs_union`), where priced as one run its natural scaled launch took 255
    registers and 24 to 32 bytes spilled at 240 and its walk 128 and 220 bytes spilled at 138, stepping down
    and recompiling four times (row 6766)."""
    from fla_gdn_support import KernelCase
    from kernel_compile import compile_mosaic_kernels, mosaic_failures

    q = jax.ShapeDtypeStruct((2, 2048, 8, 64), jnp.bfloat16)

    def step(q, k, v):
        return jax.grad(lambda q, k, v: jnp.sum(moba_attention(q, k, v, block=2, top_k=2,
                                                               implementation="template")
                                                .astype(jnp.float32)), argnums=(0, 1, 2))(q, k, v)

    with generation.assumed("sm90"):
        found = compile_mosaic_kernels([KernelCase("moba step", lambda: (step, (q, q, q)))], "sm90")
    assert not (failed := mosaic_failures(found)), "\n".join(failed)
    forwards = [certified for record in found for certified in record.get("certifications", [])
                if certified["op"] == "MosaicForward"]
    assert len(forwards) == 1, forwards
