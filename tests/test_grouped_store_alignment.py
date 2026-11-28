"""A grouped GEMM's output stores from each group's first row, known only as it runs: which tiles its
derivation may take where a tensor copy's shared source must start on a 128-byte unit (row 14182)."""

import jax
import jax.numpy as jnp

from dew.nn.kernels import generation, grouped_matmul
from dew.nn.kernels.templates.gemm import tiling


def test_a_grouped_tile_whose_staged_rows_start_off_a_tensor_copys_unit_is_never_a_candidate():
    """A grouped forward's tile stores each group's rows by TMA from its group's first row, known only as it
    runs: on an RTX PRO 6000 a tile whose bf16 output stages in 32-column chunks (64-byte rows: block_n 160
    and 224) faults at an odd group start (CUDA_ERROR_MISALIGNED_ADDRESS, M-FS-115), so it is no candidate of
    the grouped call and its derivation takes none, where 128 and 192 (64-column chunks, 128-byte rows) are.
    The same 160 columns stay a candidate where nothing so stores them: a depth split's tile (its fp32
    partial's 32 columns, 128 bytes a row), a scattered output's (its rows landing a lane at a time), an
    A100's (through registers), and a dense call's, from row 0 (row 14182)."""
    gmm = grouped_matmul.MosaicGmm()
    x = jax.ShapeDtypeStruct((4096, 4096), jnp.bfloat16)
    w = jax.ShapeDtypeStruct((8, 4096, 13440), jnp.bfloat16)
    sizes = jax.ShapeDtypeStruct((8,), jnp.int32)
    forward = {"interpret": False, "out_dtype": "bfloat16", "trans_rhs": False}

    def taken(ba, block_n: int, family: str = "mma.sync.fed", **tile) -> bool:
        return gmm.candidate(ba, tiling.Tiles(128, block_n, 32, 3, 1, **{"warpgroups": 2, **tile},
                                              family=family), generation.gpu_target())

    with generation.assumed("sm120"):
        grouped = gmm.bind(x, w, sizes, **forward)
        assert {block_n: taken(grouped, block_n) for block_n in (128, 160, 192, 224)} == {
            128: True, 160: False, 192: True, 224: False}
        assert gmm.derived(grouped, generation.gpu_target()).block_n not in (160, 224)
        assert taken(grouped, 160, splits=2)
        assert taken(gmm.bind(x, w, sizes, scatter=jax.ShapeDtypeStruct((4096,), jnp.int32), **forward), 160)
        dense = gmm.bind(x, jax.ShapeDtypeStruct((1, 4096, 13440), jnp.bfloat16),
                         jax.ShapeDtypeStruct((1,), jnp.int32), dense=True, **forward)
        assert taken(dense, 160) and taken(dense, 224)
    with generation.assumed("sm80"):
        assert taken(gmm.bind(x, w, sizes, **forward), 160, "mma.sync", warpgroups=1)
