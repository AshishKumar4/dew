"""Groups of products summed together (`Issuer.summed_together`): every group issued before any is read,
a warpgroup MMA's each read as its own groups complete, the later ones running on, what each is read into
(`then`) made there (row 10734)."""
import re

import jax
import jax.experimental.mosaic.gpu as mgpu
import jax.numpy as jnp
import numpy as np
from generation_support import COMPILES_ON_CPU, INTERPRETS
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu
from jax.sharding import Mesh

from dew.nn.kernels import generation
from dew.nn.kernels.lowered import kernel_calls
from dew.nn.kernels.templates import mosaic, mosaic_sm90, resources, stores, swizzling

TILE = 64

INTERPRETED = plgpu.InterpretGPUParams()


def _together(a, b, c, d, *, interpret):
    """`a @ b` doubled and `c @ d`, both products a warpgroup MMA's (`mosaic_sm90.FED`, one warpgroup),
    summed together, the first doubled where it is read (`then`)."""
    family = mosaic_sm90.FED
    widest = resources.copied_rows(TILE, jnp.float32)

    def body(a_gmem, b_gmem, c_gmem, d_gmem, out_gmem):
        def scoped(a_smem, b_smem, c_smem, d_smem, *outs):
            for source, staged in ((a_gmem, a_smem), (b_gmem, b_smem), (c_gmem, c_smem), (d_gmem, d_smem)):
                swizzling.copied(source, staged)
            mosaic.synced(interpret)
            readers = family.summed_together(
                ((TILE, TILE), (TILE, TILE)), (lambda acc: family.products(acc, a_smem, b_smem),
                                               lambda acc: family.products(acc, c_smem, d_smem)),
                then=(lambda value: 2.0 * value, None))
            for out, read in zip(outs, readers, strict=True):
                plgpu.store(out, read(), optimized=False)
            mosaic.synced(interpret)
            for index, out in enumerate(outs):
                stores.store_rows(out, out_gmem.at[index], 0, 0, jnp.int32(TILE), 0, TILE,
                                  copy=swizzling.coalesced, widest=widest)

        swizzle = resources.widest_swizzle(jnp.bfloat16, TILE)
        pl.run_scoped(scoped, *(swizzling.staged((TILE, TILE), jnp.bfloat16, swizzle) for _ in range(4)),
                      *(swizzling.plain((TILE, TILE), jnp.float32) for _ in range(2)), collective_axes="wg")

    out = jax.ShapeDtypeStruct((2, TILE, TILE), jnp.float32)
    return family.kernel(body, out_type=out, grid=(1,), grid_names=("programs",), consumers=1,
                         interpret=interpret, name="together")(a, b, c, d)


def _operands():
    keys = jax.random.split(jax.random.key(10734), 4)
    return [jax.random.randint(key, (TILE, TILE), -3, 4).astype(jnp.bfloat16) for key in keys]


@INTERPRETS
def test_groups_summed_together_are_each_read_into_what_their_caller_makes_of_them():
    """Interpreted: two groups of a warpgroup MMA's products summed together, the first read doubled (its
    `then`): exactly `2 a @ b` and `c @ d`."""
    a, b, c, d = _operands()
    with generation.assumed("sm90"):
        found = np.asarray(_together(a, b, c, d, interpret=INTERPRETED))
    f32 = [np.asarray(x, np.float32) for x in (a, b, c, d)]
    np.testing.assert_array_equal(found, np.stack([2 * f32[0] @ f32[1], f32[2] @ f32[3]]))


@COMPILES_ON_CPU
def test_a_warpgroup_mmas_groups_summed_together_are_each_read_as_their_own_complete():
    """Lowered for an H100: both groups issued before either is read, the first read once its own group is
    done with the second still in flight (a warpgroup MMA wait leaving one group pending), then the second;
    where each group was summed in turn, every wait left none in flight (row 10734: the attention backward's
    `product_pair` its own such path)."""
    shapes = [jax.ShapeDtypeStruct(x.shape, x.dtype) for x in _operands()]
    mesh = generation.gpu_mesh(Mesh(np.array(jax.devices()[:1]), ("data",)), "sm90")
    with generation.assumed("sm90"), jax.sharding.use_abstract_mesh(mesh):
        lowered = jax.jit(lambda *operands: _together(*operands, interpret=False)).trace(*shapes).lower(
            lowering_platforms=("cuda",))
    (call,) = [call for call in kernel_calls(lowered) if call.kind == "mosaic_gpu"]
    typed = mgpu.deserialized(call.ir)
    with typed.context:
        waits = [int(left) for left in re.findall(r"wgmma\.wait_group\.sync\.aligned (\d+);", str(typed))]
    assert waits == [1, 0], waits

