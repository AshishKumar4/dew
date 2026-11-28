"""A shared-memory ref's staging as a kernel body reads it while it traces (`swizzling.staging_of`): the
transforms its allocation staged it at, whether the ref states them itself (JAX 0.12.0's `get_ref_aval`) or
states none (JAX 0.12.1's, which hands a body its logical ref and keeps the staging a `run_scoped`
equation's parameter), through either allocation Dew makes (`swizzling.run_scoped`, a kernel's scratch
through `launches.kernel`'s `_scratch_noted`)."""
import jax
import jax.numpy as jnp
import pytest
from jax._src import core as jax_core
from jax._src.pallas.mosaic_gpu import core as plgpu_core
from jax._src.state import types as state_types
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu

from dew.nn.kernels import launches
from dew.nn.kernels.templates import mosaic_sm80, rings, swizzling

_ROW = jax.ShapeDtypeStruct((128,), jnp.float32)


def _logical(self):
    """JAX 0.12.1's `GPUMemoryRef.get_ref_aval`: the logical ref alone, none of its staging its own."""
    aval = jax_core.ShapedArray(self.shape, self.dtype)
    return state_types.AbstractRef(aval, memory_space=self.memory_space)


def _states_staging() -> bool:
    """Whether the installed JAX hands a body a staged ref stating its staging itself (0.12.0's)."""
    tile = swizzling.staged((8, 32), jnp.bfloat16, 64)
    return isinstance(tile.get_ref_aval(), state_types.TransformedRef)


@pytest.mark.parametrize("through", ["run_scoped", "scratch"])
@pytest.mark.parametrize("logical", [False, True], ids=["staging stated", "logical ref"])
def test_a_staged_refs_readers_read_its_staging_however_its_ref_states_it(monkeypatch, through, logical):
    """A 64 x 64 bf16 block staged at a 64-byte swizzle (8 x 32 tiles), allocated in a body's scope or as its
    kernel's scratch, read as it traces: its swizzle (`swizzle_of`, which the rings' piece copies, a GEMM's
    row copies and SYRK's staging read) and a view's of it the block's, and the sm80 product's depth slice of
    it a whole tile's 32 columns (`mosaic_sm80._depth_granule`, an MMA's 16 where it reads no staging),
    alike whether the ref states its staging or JAX hands the body its logical ref (`_logical`: JAX
    0.12.1's, on which the slice of 16 failed to lower, "slice size (16) ... divisible by the tile size
    (32)"); its staging noted for the body's trace alone, none past it."""
    if logical:
        monkeypatch.setattr(plgpu_core.GPUMemoryRef, "get_ref_aval", _logical)
    tile = swizzling.staged((64, 64), jnp.bfloat16, 64)
    found, held = {}, []

    def read(block):
        held.append(block)
        found.update(swizzle=swizzling.swizzle_of(block), view=swizzling.swizzle_of(block.at[pl.ds(0, 32)]),
                     granule=mosaic_sm80._depth_granule(block, 1, 16))

    def scoping(out):
        swizzling.run_scoped(read, tile)
        out[...] = jnp.zeros(out.shape, out.dtype)

    def scratching(out, block):
        read(block)
        out[...] = jnp.zeros(out.shape, out.dtype)

    scratch = launches._scratch_noted(scratching, (tile,))
    launch = (plgpu.kernel(scoping, out_type=_ROW) if through == "run_scoped" else
              plgpu.kernel(scratch, out_type=_ROW, scratch_types=(tile,)))
    jax.make_jaxpr(launch)()
    assert found == {"swizzle": 64, "view": 64, "granule": 32}
    # Past the body's trace, a ref reads the staging it states itself alone (none where JAX states none).
    own = held[0].transforms if isinstance(held[0], state_types.TransformedRef) else ()
    assert swizzling.staging_of(held[0]) == own


@pytest.mark.parametrize("logical", [False, True], ids=["staging stated", "logical ref"])
def test_a_ref_unions_members_read_each_its_own_staging(monkeypatch, logical):
    """A ref union of two blocks staged apart (64 x 64 bf16 at 64 bytes, 32 x 128 at 128), allocated in a
    body's scope: each member as the union hands it out (`ExtractAliasedRef`) reads its own staging
    (`swizzle_of`), whether the members state their staging or JAX hands the body their logical refs."""
    if logical:
        monkeypatch.setattr(plgpu_core.GPUMemoryRef, "get_ref_aval", _logical)
    union = plgpu.RefUnion(swizzling.staged((64, 64), jnp.bfloat16, 64),
                           swizzling.staged((32, 128), jnp.bfloat16, 128))
    found = []

    def read(held):
        first, second = held
        found.extend([swizzling.swizzle_of(first), swizzling.swizzle_of(second)])

    def body(out):
        swizzling.run_scoped(read, union)
        out[...] = jnp.zeros(out.shape, out.dtype)

    jax.make_jaxpr(plgpu.kernel(body, out_type=_ROW))()
    assert found == [64, 128]


@pytest.mark.parametrize("logical", [False, True], ids=["installed JAX", "logical ref"])
def test_a_rings_split_piece_copies_as_its_barrier_counts_or_refuses(monkeypatch, logical):
    """A piece of three rows copied into a ring slot of 64 x 64 bf16 staged at 64 bytes (`rings.Ring.parts`),
    whose rows are no whole tile: where the ref states its staging (the installed JAX's, where it does), a
    copy for each row's tile, as many as its load barrier counts (`piece_copies`); where it states none
    (JAX 0.12.1's), which gives no physical view to address, refused, where it was copied whole, one
    arrival of the six its barrier counts."""
    if logical:
        monkeypatch.setattr(plgpu_core.GPUMemoryRef, "get_ref_aval", _logical)
    slots = swizzling.staged((2, 64, 64), jnp.bfloat16, 64)
    copies = []

    def body(piece, out):
        def scoped(smem):
            # Its barriers no part of its copies: the slots stand in for them.
            copies.extend(rings.Ring(smem, smem, smem).parts(0, 0, piece))

        swizzling.run_scoped(scoped, slots)
        out[...] = jnp.zeros(out.shape, out.dtype)

    launch = plgpu.kernel(body, out_type=_ROW)
    piece = jax.ShapeDtypeStruct((3, 64), jnp.bfloat16)
    if logical or not _states_staging():
        with pytest.raises(NotImplementedError):
            jax.make_jaxpr(launch)(piece)
        return
    jax.make_jaxpr(launch)(piece)
    assert len(copies) == rings.piece_copies((64, 64), jnp.bfloat16, 64, 3) == 6
