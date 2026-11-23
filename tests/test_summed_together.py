"""Groups of products summed together (`Issuer.summed_together`): every group issued before any is read,
a warpgroup MMA's each read as its own groups complete, the later ones running on, what each is read into
(`then`) made there (row 10734)."""
import jax
import jax.experimental.mosaic.gpu as mgpu
import jax.numpy as jnp
import numpy as np
import pytest
from generation_support import COMPILES_ON_CPU, INTERPRETS
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu
from jax.extend.mlir import ir
from jax.sharding import Mesh

from dew.nn.kernels import generation
from dew.nn.kernels.lowered import kernel_calls
from dew.nn.kernels.templates import mosaic, mosaic_sm90, resources, stores, swizzling

TILE = 64

INTERPRETED = plgpu.InterpretGPUParams()


def _together(a, b, c, d, issued, *, interpret, guard: str = "predicated"):
    """`a @ b` doubled and `c @ d`, both products a warpgroup MMA's (`mosaic_sm90.FED`, one warpgroup),
    summed together, the first doubled where it is read (`then`); the second issued as `guard` says:
    "plain", always; "predicated", where `issued[0]` (an int32 a run reads) is nonzero, its accumulator zero
    elsewhere; "chosen", as the one case of a choice (`Issuer.issued_one_of`) whose products are so
    predicated within it."""
    family = mosaic_sm90.FED
    widest = resources.copied_rows(TILE, jnp.float32)

    def body(a_gmem, b_gmem, c_gmem, d_gmem, issued_gmem, out_gmem):
        def scoped(a_smem, b_smem, c_smem, d_smem, *outs):
            for source, staged in ((a_gmem, a_smem), (b_gmem, b_smem), (c_gmem, c_smem), (d_gmem, d_smem)):
                swizzling.copied(source, staged)
            mosaic.synced(interpret)
            def second(acc):
                def predicated(acc):
                    return family.issued_when(issued_gmem[0] != 0, acc,
                                              lambda: family.products(acc, c_smem, d_smem))
                if guard == "chosen":
                    return family.issued_one_of(jnp.int32(0), (acc,), lambda case, acc: predicated(acc))[0]
                return predicated(acc) if guard == "predicated" else family.products(acc, c_smem, d_smem)

            readers = family.summed_together(
                ((TILE, TILE), (TILE, TILE)), (lambda acc: family.products(acc, a_smem, b_smem), second),
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
                         interpret=interpret, name="together")(a, b, c, d, issued)


def _operands():
    keys = jax.random.split(jax.random.key(10734), 4)
    return [jax.random.randint(key, (TILE, TILE), -3, 4).astype(jnp.bfloat16) for key in keys]


@INTERPRETS
@pytest.mark.parametrize("guard", ["predicated", "chosen"])
@pytest.mark.parametrize("issued", [1, 0])
def test_groups_summed_together_are_each_read_into_what_their_caller_makes_of_them(issued, guard):
    """Interpreted: two groups of a warpgroup MMA's products summed together, the first read doubled (its
    `then`), the second issued or not as a run decides, alone or as a choice's case: exactly `2 a @ b`, and
    `c @ d` or zero."""
    a, b, c, d = _operands()
    with generation.assumed("sm90"):
        found = np.asarray(_together(a, b, c, d, jnp.array([issued], jnp.int32), interpret=INTERPRETED,
                                     guard=guard))
    f32 = [np.asarray(x, np.float32) for x in (a, b, c, d)]
    np.testing.assert_array_equal(found, np.stack([2 * f32[0] @ f32[1], issued * (f32[2] @ f32[3])]))


def _groups_and_waits(module: bytes) -> list[str]:
    """A Mosaic GPU kernel's warpgroup MMA commits and waits in program order (its typed module's inline
    assembly, `wgmma.commit_group` and `wgmma.wait_group`), each wait with the groups it leaves in flight."""
    found: list[str] = []

    def walk(operation: ir.Operation) -> None:
        for region in operation.regions:
            for block in region.blocks:
                for inner in block.operations:
                    op = inner.operation
                    if op.name == "llvm.inline_asm":
                        text = ir.StringAttr(op.attributes["asm_string"]).value
                        if text.startswith("wgmma.commit_group"):
                            found.append("commit")
                        elif text.startswith("wgmma.wait_group.sync.aligned "):
                            found.append(f"wait {text.split()[-1].rstrip(';')}")
                    walk(op)

    typed = mgpu.deserialized(module)
    with typed.context:
        walk(typed.operation)
    return found


@COMPILES_ON_CPU
@pytest.mark.parametrize("guard", ["plain", "predicated", "chosen"])
def test_a_warpgroup_mmas_groups_summed_together_are_each_read_as_their_own_complete(guard):
    """Lowered for an H100: both groups committed before either is read, the first read once its own group
    is done with the second still in flight (a wait leaving one group in flight), then the second; where
    the second is issued under a predicate a run decides, alone or within the one case of a choice, the
    first's wait leaves none in flight, as a run that skips the second would read the first unfinished (row
    10734; theme K01-e's reviews' P0s: a choice's case read as committing its inner predicate's groups)."""
    shapes = [jax.ShapeDtypeStruct(x.shape, x.dtype) for x in _operands()] + [
        jax.ShapeDtypeStruct((1,), jnp.int32)]
    mesh = generation.gpu_mesh(Mesh(np.array(jax.devices()[:1]), ("data",)), "sm90")
    with generation.assumed("sm90"), jax.sharding.use_abstract_mesh(mesh):
        run = jax.jit(lambda *operands: _together(*operands, interpret=False, guard=guard))
        lowered = run.trace(*shapes).lower(lowering_platforms=("cuda",))
    (call,) = [call for call in kernel_calls(lowered) if call.kind == "mosaic_gpu"]
    first_read = "wait 1" if guard == "plain" else "wait 0"
    assert _groups_and_waits(call.ir) == ["commit", "commit", first_read, "wait 0"]
