"""CUTLASS's grouped launches (`grouped_cutlass`) as a CPU traces them, against a stand-in wheel (no CPU runs
their kernels): a stack's layer, rows a prologue forms and a checked shard_map's types (the numbers on a GPU,
tests/test_grouped_gemm.py's `cutlass_runs_the_projection_as_exactly_as_xla_where_chosen` in the GPU flow)."""

import functools
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from reference_error import equations

from dew.nn.kernels import gemm, generation, grouped_cutlass
from dew.nn.kernels.templates.gemm import sides as hook_sides
from dew.nn.kernels.templates.gemm.hooks import Hooks
from dew.nn.kernels.templates.hooks import Hook, Side


@pytest.mark.mesh(devices=2)
def test_a_cutlass_launch_under_a_checked_shard_map_varies_as_its_operands_do(monkeypatch):
    """A CUTLASS launch inside a checked `shard_map` (`check_vma=True`) whose layer each device reads its own
    of, its rows and stack read whole: the layer's matrices it slices vary over the mesh's axis, so its result
    is typed as varying there, its operands made to vary alike before the FFI call takes them (#3976's
    review P0: the result was typed the same on every device). Traced on the CPU, the library's targets
    named as a stand-in (no CPU runs its kernels)."""
    _stand_in_cutlass(monkeypatch)
    gmm = grouped_cutlass.CutlassGmm()
    mesh = jax.sharding.Mesh(np.asarray(jax.devices()[:2]), ("data",))
    data, whole = jax.sharding.PartitionSpec("data"), jax.sharding.PartitionSpec()

    def each_device(tokens, stack, sizes, layer):
        out, _ = gmm._fwd(tokens, stack, sizes, first=layer[0] * sizes.shape[0], out_dtype="bfloat16",
                          trans_rhs=False, interpret=False, config=None)
        assert jax.typeof(out).mat.varying == frozenset({"data"}), jax.typeof(out)
        return out

    jax.make_jaxpr(jax.shard_map(each_device, mesh=mesh, in_specs=(whole, whole, whole, data),
                                 out_specs=data, check_vma=True))(
        jax.ShapeDtypeStruct((64, 32), jnp.bfloat16), jax.ShapeDtypeStruct((4, 32, 16), jnp.bfloat16),
        jax.ShapeDtypeStruct((2,), jnp.int32), jax.ShapeDtypeStruct((2,), jnp.int32))


def _stand_in_cutlass(monkeypatch) -> None:
    """A dew_cutlass wheel for a CPU's trace (no CPU runs its kernels): its targets, its sm90 build (every
    Hopper part runs it, `architecture`) and that build's one kernel, what a route to CUTLASS and its launch
    read of the wheel (`generation.cutlass_built`, `grouped_cutlass.cutlass_tile`)."""
    import importlib.machinery
    import types

    kernel = types.SimpleNamespace(tile=(128, 256, 64), stages=4, consumers=256, cluster=2,
                                   instructions="wgmma")
    wheel = types.ModuleType("dew_cutlass")
    vars(wheel).update(
        __spec__=importlib.machinery.ModuleSpec("dew_cutlass", None, origin="stand-in://dew_cutlass/sm90"),
        TARGETS=("dew_cutlass_gmm", "dew_cutlass_tgmm"),
        KERNELS={("sm90", "bfloat16"): kernel, ("sm90", "float32"): kernel}, registered=lambda: None,
        architecture=lambda major, minor: "sm90" if major == 9 else None)
    monkeypatch.setitem(sys.modules, "dew_cutlass", wheel)


def test_a_call_naming_cutlass_takes_its_rows_through_a_prologue(monkeypatch):
    """A grouped product whose rows a prologue forms (an input-scaled MoE's rows times their routing
    weights), its route named CUTLASS on an sm90: its forward and both gradients run CUTLASS's launches, the
    rows formed by XLA before them (`grouped_cutlass._formed`), where the named route refused the prologue
    and raised (models-28's review)."""
    _stand_in_cutlass(monkeypatch)
    bf16 = functools.partial(jax.ShapeDtypeStruct, dtype=jnp.bfloat16)
    sizes = jnp.asarray([64, 64], jnp.int32)

    def loss(x, w, weights):
        out = gemm.gemm(x, w, sizes, prologue=Hook(lambda rows, scale: rows * scale[:, None],
                                                   (Side(weights, ("rows",)),)), out_dtype=jnp.bfloat16)
        return jnp.sum(out.astype(jnp.float32))

    with generation.assumed("sm90"), generation.chosen(grouped_matmul="cutlass"):
        graph = jax.make_jaxpr(jax.grad(loss, argnums=(0, 1, 2)))(
            bf16((128, 64)), bf16((2, 64, 128)), bf16((128,)))
    targets = [eqn.params["target_name"] for eqn in equations(graph, platform="cuda")
               if eqn.primitive.name == "ffi_call"]
    assert sorted(set(targets)) == ["dew_cutlass_gmm", "dew_cutlass_tgmm"], targets


def test_a_cutlass_launch_forms_a_singleton_sides_rows_as_xla_forms_them():
    """The rows a CUTLASS launch multiplies, formed by XLA before it (`grouped_cutlass._formed`), from a side
    of one value a row given `[rows, 1]` (axes `('rows', None)`) and a prologue reducing along that singleton:
    the launch's squeezed side given back as `[rows, 1]`, so the formed rows are the prologue of the call's
    own arrays, as the portable program forms them, where a kernel tile's broadcast of it across the depth
    read `depth` times the scale (#3985's review)."""
    rng = np.random.default_rng(3985)
    rows = jnp.asarray(rng.normal(size=(64, 32)), jnp.bfloat16)
    scale = jnp.asarray(rng.normal(size=(64, 1)), jnp.float32)

    def prologue(x, s):
        return x * s.sum(axis=1, keepdims=True)

    squeezed, axes, positions = hook_sides.launchable((scale,), (("rows", None),))
    hooks = Hooks(prologue, axes, prologue_broadcast=positions)
    formed = grouped_cutlass._formed(rows, squeezed, hooks)
    want = prologue(rows, scale).astype(jnp.bfloat16)
    np.testing.assert_array_equal(np.asarray(formed).view(np.uint16), np.asarray(want).view(np.uint16))


def test_a_steps_reach_holds_its_live_cutlass_launch_and_not_its_dead_one(monkeypatch):
    """A step lowered for an sm90 running CUTLASS's grouped product at two shapes, one in its own flow and
    one in a loop no run enters: its FFI calls carry their launches' digests (`instrumentation.LAUNCH`), so a
    tuning run's reach holds the live launch alone (`tuning.reach`, what `tune` settles and
    tools/kernel_picks.py counts), though both traced, and names no foreign kernel. Traced on the CPU, the
    library's targets named as a stand-in (no CPU runs its kernels)."""
    from kernel_compile import live_and_dead, lowered_for

    from dew.nn.kernels import tuning

    _stand_in_cutlass(monkeypatch)
    sizes = jnp.asarray([64, 64], jnp.int32)

    def product(x, w):
        return gemm.gemm(x, w, sizes, out_dtype=jnp.bfloat16)

    bf16 = functools.partial(jax.ShapeDtypeStruct, dtype=jnp.bfloat16)
    shapes = [bf16((128, 64)), bf16((2, 64, 128)), bf16((128, 128)), bf16((2, 128, 128))]
    with generation.assumed("sm90"), generation.chosen(grouped_matmul="cutlass"):
        with tuning.seen() as found:
            lowered = lowered_for(live_and_dead(product), shapes, "sm90")
        # The live launch's own step, its key the one a reach holds.
        with tuning.seen() as alone:
            lowered_for(product, shapes[:2], "sm90")
    reached = tuning.reach(found, lowered)
    assert len(tuning._distinct(list(found))) == 2 and reached.foreign == ()
    assert [ba.autotuning_cache_key for _, ba in reached.launches] == [
        ba.autotuning_cache_key for _, ba in tuning._distinct(list(alone))]
