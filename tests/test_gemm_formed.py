"""The GEMM template's rules for a product's operand formed by its producer (`gemm`'s `applied`), an
entrywise prologue's VJP in the transposed product's epilogue, and an output no reverse pass reaches, which
its product's reverse mode forms nothing for: each interpreted as a GPU on a CPU, and a gated MLP's three
products composed on them compiled for compile-only GPUs within their resources."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from generation_support import generations
from reference_error import assert_as_exact, with_gradients

from dew.nn.gated_mlp import fused_products
from dew.nn.kernels import gemm, generation

ROWS, FEATURES, HIDDEN = 136, 128, 256
"""Rows no tile divides, the features a gated MLP reads and writes, its hidden width."""
SEEDS = 8


def _draw(seed, dtype=jnp.bfloat16):
    rng = np.random.default_rng(seed)
    return (jnp.asarray(rng.normal(size=(ROWS, FEATURES)), dtype),
            jnp.asarray(rng.normal(size=(FEATURES, HIDDEN)) / np.sqrt(FEATURES), dtype),
            jnp.asarray(rng.normal(size=(FEATURES, HIDDEN)) / np.sqrt(FEATURES), dtype),
            jnp.asarray(rng.normal(size=(HIDDEN, FEATURES)) / np.sqrt(HIDDEN), dtype),
            jnp.asarray(rng.normal(size=(ROWS, FEATURES)), dtype))


def _glu(gate, up):
    """A SwiGLU of a gate and an up projection's tiles, each rounding a cast pair a kernel lowers: the
    activation in fp32 rounded to the gate's dtype, the product rounded."""
    activated = jax.nn.silu(gate.astype(jnp.float32)).astype(gate.dtype).astype(jnp.float32)
    return (activated * up.astype(jnp.float32)).astype(gate.dtype)


def _composed(x, gate, up, down, *, names=None, interpret: bool = False):
    """A gated MLP's three products on the template's hooks (`dew.nn.gated_mlp.fused_products`, SwiGLU's,
    whose rounded arithmetic is `_glu`'s): the paired gate|up product whose epilogue writes the projections
    and their SwiGLU, then the down product of that formed operand (`applied`) with the SwiGLU its prologue
    over the gate and the up projection."""
    return fused_products(x, (gate, up, down), "swiglu", names=names, interpret_on_cpu=interpret)


def _apart(x, gate, up, down, *, truth: bool = False):
    """The three products apart in XLA: each projection one dot rounded once, the SwiGLU, the down
    projection; as `truth`, in float64 with nothing rounded."""
    if truth:
        x, gate, up, down = (each.astype(jnp.float64) for each in (x, gate, up, down))
        return (jax.nn.silu(x @ gate) * (x @ up)) @ down
    return _glu(x @ gate, x @ up) @ down


@pytest.mark.parametrize("generation_name", generations("sm90", "sm100"))
def test_a_composed_mlp_holds_check_1_with_its_gradients(generation_name):
    """A gated MLP's three products composed on the template's hooks (`_composed`), interpreted as
    `generation_name`: its output, and the gradients of its rows and its three kernels through the template's
    reverse modes (the paired product's, and the down product's with the SwiGLU's VJP, JAX's of the traced
    hook, in its input gradient's epilogue where the launch takes it), each as exact as the products apart
    from float64 truth (check 1), every product on the template."""
    def composed(x, gate, up, down):
        return _composed(x, gate, up, down, interpret=True)

    truth = with_gradients(functools.partial(_apart, truth=True))
    with (generation.assumed(generation_name), generation.chosen(grouped_matmul="mosaic"),
          generation.routes() as routes):
        assert_as_exact(with_gradients(composed), jax.jit(with_gradients(_apart)), _draw,
                        f"a composed MLP on {generation_name}", seeds=SEEDS, truth_fn=truth)
    assert {route.kernel for route in routes if route.choice == "grouped_matmul"} == {"mosaic"}


@pytest.mark.parametrize("generation_name", ["sm90", "sm100"])
def test_a_composed_mlps_training_step_compiles_within_its_resources(generation_name):
    """A gated MLP's three products composed on the template's hooks (`_composed`) at the 4B's MLP (2560 to
    9216 to 2560) over 2048 rows, its training step compiled for a compile-only `generation_name`, every
    product on the template: every kernel within its shared and tensor memory, none spilling. On sm100 the
    input gradient's VJP epilogue derives no tile and runs in XLA, and the down product's kernel gradient,
    whose prologue tcgen05's kernel gradient takes none of, multiplies the operand XLA forms."""
    import fla_gdn_support as support
    from kernel_compile import compile_mosaic_kernels, mosaic_failures

    shaped = functools.partial(jax.ShapeDtypeStruct, dtype=jnp.bfloat16)

    def step(x, gate, up, down, cotangent):
        out, pull = jax.vjp(_composed, x, gate, up, down)
        return out, pull(cotangent)

    operands = (shaped((2048, 2560)), shaped((2560, 9216)), shaped((2560, 9216)), shaped((9216, 2560)),
                shaped((2048, 2560)))
    with generation.chosen(grouped_matmul="mosaic"):
        found = compile_mosaic_kernels([support.KernelCase("composed MLP step", lambda: (step, operands))],
                                       generation_name)
    failed = mosaic_failures(found)
    assert not failed, "\n".join(failed)


def test_a_higher_order_through_an_output_no_reverse_pass_reaches_holds_check_1():
    """A paired product's two outputs, a loss reading the first alone: its gradient and that gradient's own
    derivative (forward over reverse, `jax.jvp` of `jax.grad`), whose pullback takes the unreached output's
    cotangent as zeros of its output's type (`linear_tangent`'s `zero_cotangents`), each as exact as the
    portable program's (the two products and the epilogue in XLA) from float64 truth (check 1)."""
    def draw(seed):
        rng = np.random.default_rng(seed)
        return tuple(jnp.asarray(rng.normal(size=shape), jnp.float32)
                     for shape in ((16, 32), (32, 24), (32, 24), (16, 32)))

    def fused(x, gate, up):
        return gemm.gemm(x, gate, paired=up, epilogue=lambda g, u: (g * u, u), out_dtype=x.dtype,
                         out_dtypes=(x.dtype, x.dtype))

    def portable(x, gate, up):
        g, u = x @ gate, x @ up
        return g * u, u

    def orders(product, *, wide: bool = False):
        def run(x, gate, up, tangent):
            if wide:
                x, gate, up, tangent = (each.astype(jnp.float64) for each in (x, gate, up, tangent))

            def loss(x):
                return jnp.sum(product(x, gate, up)[0] ** 2)

            return jax.jvp(jax.grad(loss), (x,), (tangent,))
        return run

    assert_as_exact(orders(fused), jax.jit(orders(portable)), draw, "an unreached output's higher order",
                    seeds=SEEDS, truth_fn=orders(portable, wide=True))


def _formed(x, side):
    """An entrywise prologue: the rows times a side operand of their shape, then a SiLU, in the rows'
    dtype."""
    return jax.nn.silu(x.astype(jnp.float32) * side.astype(jnp.float32)).astype(x.dtype)


@pytest.mark.parametrize("generation_name", generations("sm90"))
def test_a_formed_operand_is_its_prologues_value_and_derivative(generation_name):
    """A product of a formed operand (`gemm`'s `applied`: the prologue's value, written by its producer)
    multiplies it, its value the plain product's of that operand bit for bit, and pulls back through the
    prologue: every gradient (the rows', the side operand's, the matrix's) that of the product through the
    prologue bit for bit (the prologue's VJP the input gradient's epilogue, which writes the operand the
    kernel gradient reads), the formed operand taking none and its value read by no gradient (one formed
    as zeros pulls back alike), so a remat policy keeping the rows and the side recomputes no product."""
    rng = np.random.default_rng(3)
    x, side, out = (jnp.asarray(rng.normal(size=(ROWS, FEATURES)), jnp.bfloat16) for _ in range(3))
    matrix = jnp.asarray(rng.normal(size=(FEATURES, FEATURES)) / 8, jnp.bfloat16)

    def through(x, side, matrix):
        return gemm.gemm(x, matrix, prologue=_formed, prologue_operands=((side, ("rows", "depth")),),
                         out_dtype=jnp.bfloat16, interpret_on_cpu=True)

    def applied(x, side, matrix, formed):
        return gemm.gemm(x, matrix, prologue=_formed, prologue_operands=((side, ("rows", "depth")),),
                         applied=formed, out_dtype=jnp.bfloat16, interpret_on_cpu=True)

    def plain(formed, matrix):
        return gemm.gemm(formed, matrix, out_dtype=jnp.bfloat16, interpret_on_cpu=True)

    formed = _formed(x, side)
    with generation.assumed(generation_name), generation.chosen(grouped_matmul="mosaic"):
        gradients = jax.vjp(through, x, side, matrix)[1](out)
        value, pull = jax.vjp(applied, x, side, matrix, formed)
        formed_gradients = pull(out)
        zeroed_gradients = jax.vjp(applied, x, side, matrix, jnp.zeros_like(formed))[1](out)
        plain_value = plain(formed, matrix)
    names = ("rows", "side", "matrix")
    alike = [("value", value, plain_value),
             *((name, formed_gradients[at], gradients[at]) for at, name in enumerate(names)),
             *((f"{name} of a zeroed formed operand", zeroed_gradients[at], gradients[at])
               for at, name in enumerate(names))]
    for name, mine, theirs in alike:
        np.testing.assert_array_equal(np.asarray(mine, np.float32), np.asarray(theirs, np.float32),
                                      err_msg=name)
    assert not np.any(np.asarray(formed_gradients[3], np.float32))


@pytest.mark.parametrize("generation_name", generations("sm90"))
def test_an_output_no_reverse_pass_reaches_costs_its_reverse_mode_nothing(generation_name):
    """A paired product's epilogue writing the two projections and their gated product, the last read only
    as a formed operand (a fused MLP's, its derivative held still): its reverse mode forms the two
    projections' cotangents alone, launching what the same product's does whose epilogue writes the
    projections alone, and no product to recompute the accumulators the gated product's VJP would read (as
    the same product's does whose gated product a loss reads), one launch selected a route
    (`generation.routes`)."""
    hook = _glu

    def projected(gate_acc, up_acc, *, formed: bool):
        gate, up = gate_acc.astype(jnp.bfloat16), up_acc.astype(jnp.bfloat16)
        return (gate, up, hook(gate, up)) if formed else (gate, up)

    def loss(outputs, x, gate, up):
        formed = outputs != "projections"
        found = gemm.gemm(x, gate, paired=up, epilogue=functools.partial(projected, formed=formed),
                          out_dtype=jnp.bfloat16, out_dtypes=(jnp.bfloat16,) * (3 if formed else 2),
                          interpret_on_cpu=True)
        total = jnp.sum(found[0].astype(jnp.float32) * found[1].astype(jnp.float32))
        if outputs == "read":
            total += jnp.sum(found[2].astype(jnp.float32))
        elif formed:
            total += jnp.sum(jax.lax.stop_gradient(found[2]).astype(jnp.float32))
        return total

    x, gate, up = _draw(0)[:3]

    def launches(outputs):
        with generation.assumed(generation_name), generation.routes() as routes:
            jax.make_jaxpr(jax.grad(functools.partial(loss, outputs), argnums=(0, 1, 2)))(x, gate, up)
        return sum(route.choice == "grouped_matmul" for route in routes)

    assert launches("held") == launches("projections") < launches("read")


def test_a_remat_keeping_the_projections_forms_the_hidden_activation_by_no_product():
    """A gated MLP's three products composed on the template's hooks (`_composed`), its training step under a
    remat policy saving its named projections alone (`save_only_these_names`), lowers for a compile-only H100
    as many kernel launches as the step with nothing rematerialized (`kernel_launches`, each the times it
    runs): the formed hidden activation (`gemm`'s `applied`) is no residual of the reverse mode
    (`linear_tangent`'s `unread`), so no paired product runs again to form it."""
    from kernel_compile import lowered_for

    from dew.nn.kernels.lowered import kernel_launches

    names = ("gate", "up", "down")

    def loss(x, gate, up, down):
        return jnp.sum(_composed(x, gate, up, down, names=names).astype(jnp.float32))

    policy = jax.checkpoint_policies.save_only_these_names(*names)
    steps = {"saved": jax.grad(jax.checkpoint(loss, policy=policy), argnums=(0, 1, 2, 3)),
             "whole": jax.grad(loss, argnums=(0, 1, 2, 3))}
    shapes = tuple(jax.ShapeDtypeStruct(each.shape, each.dtype) for each in _draw(0)[:4])
    with generation.chosen(grouped_matmul="mosaic"):
        launched = {name: sum(times for _, times in kernel_launches(lowered_for(step, shapes, "sm90")))
                    for name, step in steps.items()}
    assert launched["saved"] == launched["whole"], launched
