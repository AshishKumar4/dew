"""Compile-only diagnostic of mma.sync's output layout and row-copy geometries."""
import json
import jax
import jax.numpy as jnp
import pytest
from dew.nn.kernels.templates import mosaic_sm80
from dew.nn.kernels.templates.gemm import mosaic_gpu
from kernel_compile import compile_mosaic_kernels
import fla_gdn_support as support


@pytest.mark.parametrize("generation_name", ["sm80", "sm86"])
@pytest.mark.parametrize("block_m", [64, 128])
@pytest.mark.parametrize("block_n", [64, 128, 192, 256])
@pytest.mark.parametrize("m_warps", [1, 2, 4])
def test_gmm_store_geometry(generation_name, block_m, block_n, m_warps, monkeypatch):
    family=mosaic_sm80.Mma(jnp.dtype(jnp.bfloat16),m_warps)
    monkeypatch.setattr(mosaic_gpu,"family_of",lambda tiles,dtype:family)
    tile=mosaic_gpu.Tiles(block_m,block_n,32,2,1,family="mma.sync",programs=1)
    def call():
        def gmm(lhs,rhs,sizes):
            return mosaic_gpu.gmm(lhs,rhs,sizes,tiles=tile,transpose_rhs=False,
                                  out_dtype=jnp.bfloat16,multiprocessors=1,dense=True)
        return gmm,(jax.ShapeDtypeStruct((256,128),jnp.bfloat16),
                    jax.ShapeDtypeStruct((1,128,block_n*2),jnp.bfloat16),
                    jax.ShapeDtypeStruct((1,),jnp.int32))
    records=compile_mosaic_kernels([support.KernelCase("store",call)],generation_name)
    print("BC_STORE",json.dumps(dict(generation=generation_name,m=block_m,n=block_n,
                                    m_warps=m_warps,records=records)),flush=True)
    assert len(records)==1 and not records[0].get("error"), records
