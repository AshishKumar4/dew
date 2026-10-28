"""Compile-only isolation of row-boundary copies from the MMA-layout staging store."""
import jax
import jax.numpy as jnp
import pytest
from jax.experimental import pallas as pl
from dew.nn.kernels.templates import mosaic, mosaic_sm80
from dew.nn.kernels.templates.gemm import mosaic_gpu
from kernel_compile import compile_mosaic_kernels
import fla_gdn_support as support

@pytest.mark.parametrize("generation_name",["sm80","sm86"])
@pytest.mark.parametrize("dense",[False,True])
@pytest.mark.parametrize("out_dtype",[jnp.bfloat16,jnp.float32])
@pytest.mark.parametrize("copy_mode",["ladder","whole"])
def test_ragged_copy(generation_name,dense,out_dtype,copy_mode,monkeypatch):
    family=mosaic_sm80.Mma(jnp.dtype(jnp.bfloat16),2)
    monkeypatch.setattr(mosaic_gpu,"family_of",lambda tiles,dtype:family)
    if copy_mode=="whole":
        def whole(smem,gmem,row_start,first,stored,column,most,**kw):
            mosaic.copied(smem,gmem.at[pl.ds(row_start,smem.shape[0]),
                                       pl.ds(column,smem.shape[1])])
        monkeypatch.setattr(mosaic,"store_rows",whole)
    tile=mosaic_gpu.Tiles(64,64,64,2,1,family="mma.sync",programs=1)
    def call():
        def gmm(lhs,rhs,sizes):
            return mosaic_gpu.gmm(lhs,rhs,sizes,tiles=tile,transpose_rhs=False,
                                  out_dtype=out_dtype,multiprocessors=1,dense=dense)
        return gmm,(jax.ShapeDtypeStruct((256,128),jnp.bfloat16),
                    jax.ShapeDtypeStruct((1,128,128),jnp.bfloat16),
                    jax.ShapeDtypeStruct((1,),jnp.int32))
    records=compile_mosaic_kernels([support.KernelCase("ragged-store",call)],generation_name)
    assert len(records)==1 and not records[0].get("error") and records[0].get("kernels"),records
