"""The fusion search (docs/design/algebra.md §2 "Fusion", `dew.algebra.fusion`): a program's products and
their elementwise producers and consumers fused into the GEMM template's prologue and epilogue by the
compile-time pass, where the template takes the product and its price says fused wins. Every program
computes what it does as written, its gradients too, whatever the pass fuses."""

import contextlib
import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from generation_support import lowers

from dew import _facts, algebra
from dew.nn.kernels import generation
from dew.nn.kernels.templates import report
from dew.nn.precision import Numerics


def draw(seed: int = 7):
    """Integer operands whose sums pass bf16's 256, so a dot's own rounding to bf16 matters, every product
    and fp32 sum exact: host arrays, NumPy's, which a GPU tier part's prepare places nowhere (bible §8: a
    draw of JAX's ran on its compile-only device, rows 10096 on)."""
    rng = np.random.default_rng(seed)
    return (rng.integers(-7, 8, (256, 512)).astype(jnp.bfloat16),
            rng.integers(-7, 8, (512, 256)).astype(jnp.bfloat16),
            rng.integers(-3, 4, (256,)).astype(jnp.bfloat16),
            rng.integers(-3, 4, (256, 256)).astype(jnp.bfloat16))


def ran(fn, label: str, *operands):
    """`fn` jitted over host arrays `operands`, as a GPU tier part's prepare compiles it ahead and its run
    runs it (`reference_error.ahead_arm`: a bitwise check's program, computing nothing in the prepare);
    outside a part, run as it is."""
    from reference_error import ahead_arm

    return ahead_arm(jax.jit(fn), label)(*operands)


@pytest.mark.parametrize("boundary", ["none", "escaped_gate", "extra_hidden_read", "separate_rows"])
def test_decode_region_keeps_shared_projections_and_their_actual_outputs(boundary):
    from dew.algebra.fusion import decode_regions
    from dew.nn.activations import gated_product

    def layer(x, other, w_gate, w_up, w_down):
        gate, up = x @ w_gate, (other if boundary == "separate_rows" else x) @ w_up
        hidden = gated_product("swiglu", implementation="xla")(gate, up)
        down = hidden @ w_down
        if boundary == "escaped_gate":
            return down, gate
        if boundary == "extra_hidden_read":
            return down + hidden.sum()
        return down

    shapes = ((1, 8), (1, 8), (8, 16), (8, 16), (16, 8))
    traced = jax.make_jaxpr(layer)(*(jax.ShapeDtypeStruct(shape, jnp.float32) for shape in shapes)).jaxpr
    found = decode_regions(traced, rows=1)
    if boundary != "none":
        assert not found
        return
    region, = found
    assert region.inputs == (traced.invars[0], *traced.invars[2:])
    assert region.outputs == tuple(traced.outvars)
    assert not decode_regions(traced, rows=8)


def _decode_layer(x, gate, up, down):
    from dew.nn.activations import gated_product

    def product(left, right):
        return jax.lax.dot_general(left, right, (((1,), (0,)), ((), ())),
                                   precision=jax.lax.Precision.HIGHEST)

    hidden = gated_product("swiglu", implementation="xla")(product(x, gate), product(x, up))
    return product(hidden, down)


def _wide_decode(x, gate, up, down):
    x, gate, up, down = (a.astype(jnp.float64) for a in (x, gate, up, down))
    g = x @ gate
    return ((g / (1 + jnp.exp(-g))) * (x @ up)) @ down


def _decode_draw(rows, seed):
    rng = np.random.default_rng(seed)
    return tuple(rng.normal(size=shape).astype(jnp.bfloat16)
                 for shape in ((rows, 8), (8, 16), (8, 16), (16, 8), (rows, 8)))


def _selected_decode(shapes):
    from dew.algebra.fusion import lower_decode

    rows = shapes[0].shape[0]
    # Ahead preparation places its executable on a one-device mesh. Select the unplaced
    # region before that placement, as a caller opting into decode lowering does.
    with jax.sharding.use_abstract_mesh(jax.sharding.AbstractMesh((), ())):
        traced = jax.make_jaxpr(_decode_layer)(*shapes)
        original = traced.jaxpr
        assert lower_decode(original, rows=rows) is original
        with jax.disable_jit():
            assert lower_decode(original, rows=rows, enabled=True) is original
        selected = lower_decode(original, rows=rows, enabled=True)
    (call,) = selected.eqns
    body = call.params["jaxpr"].jaxpr
    lhs = [(eq.primitive, eq.params) for eq in body.eqns if eq.primitive is jax.lax.dot_general_p]
    rhs = [(eq.primitive, eq.params) for eq in original.eqns if eq.primitive is jax.lax.dot_general_p]
    assert len(lhs) == 3 and lhs == rhs

    mesh = jax.sharding.AbstractMesh((1,), ("data",), axis_types=(jax.sharding.AxisType.Explicit,))
    with jax.sharding.use_abstract_mesh(mesh):
        placed = jax.make_jaxpr(_decode_layer)(*shapes).jaxpr
    assert lower_decode(placed, rows=rows, enabled=True) is placed
    return traced.replace(jaxpr=selected)


def _decoded_layer(*operands):
    from jax.extend import core

    # Construction stays in the arm trace, so Check 1 can name and compile it from the draw.
    shapes = tuple(jax.ShapeDtypeStruct(x.shape, x.dtype) for x in operands)
    (out,) = core.jaxpr_as_fun(_selected_decode(shapes))(*operands)
    return out


@pytest.mark.parametrize("rows", [1, 8])
def test_the_opt_in_decode_region_lowers_as_one_call_with_the_written_math(rows):
    from reference_error import assert_as_exact, with_gradients

    assert_as_exact(
        with_gradients(_decoded_layer), with_gradients(_decode_layer), functools.partial(_decode_draw, rows),
        "one-call decode SwiGLU and all four gradients", seeds=8, truth_fn=with_gradients(_wide_decode))


def test_decode_lowering_does_not_move_across_a_named_residual():
    from jax.ad_checkpoint import checkpoint_name

    from dew.algebra.fusion import decode_regions, lower_decode
    from dew.nn.activations import gated_product

    def layer(x, gate, up, down):
        a = x @ gate
        held = checkpoint_name(x, "residual")
        hidden = gated_product("swiglu", implementation="xla")(a, x @ up)
        return hidden @ down, held

    shapes = ((1, 8), (8, 16), (8, 16), (16, 8))
    traced = jax.make_jaxpr(layer)(*(jax.ShapeDtypeStruct(shape, jnp.float32) for shape in shapes)).jaxpr
    assert decode_regions(traced, rows=1)
    assert lower_decode(traced, rows=1, enabled=True) is traced


def _passed(fn, *operands, kernels: bool = True):
    """`fn` under the algebra pass (`algebra.rewritten`, its kernels too where `kernels`), made where it is
    traced: a program of it is stated by `fn` (`truths_ahead.arm_role`), where the pass's own closure holds
    JAX's token, which no statement holds."""
    return algebra.rewritten(fn, kernels=kernels)(*operands)


def biased(x, w, bias, other):
    """A projection, its bias and a scale after it."""
    del other
    return (x @ w + bias) * jnp.bfloat16(2)


def escaping(x, w, bias, other):
    """A projection whose product is read after its consumers too."""
    del other
    product = x @ w
    return product + bias, product


def widened(x, w, bias, other):
    """A projection rounded to bf16, then widened and added to in float32."""
    del other
    return (x @ w).astype(jnp.float32) + bias.astype(jnp.float32)


def chained(x, w, bias, other):
    """A projection whose consumer feeds a second projection, its own consumer after it."""
    hidden = jnp.maximum(x @ w + bias, jnp.bfloat16(0))
    return hidden @ other - bias


def bare(x, w, bias, other):
    """A projection alone, nothing after it."""
    del bias, other
    return x @ w


def kernel_gradient(x, w, bias, other):
    """A kernel gradient's product, xᵀ @ dy: the rows contracted."""
    del w, bias
    return x.T @ other


SIZES = (96, 160)
"""Two groups' rows of `draw`'s 256."""


def grouped(x, w, bias, other):
    """Each group's rows times its matrix (`lax.ragged_dot`, an MoE's experts), a bias after it."""
    del other
    return jax.lax.ragged_dot(x, jnp.stack([w, -w]), jnp.asarray(SIZES, jnp.int32)) + bias


def grouped_kernel_gradient(x, w, bias, other):
    """Each group's rows of x contracted with its rows of dy, `[groups, depth, columns]`: the grouped
    kernel gradient's form (`lax.ragged_dot_general`, the rows ragged within the contraction)."""
    del w, bias
    numbers = jax.lax.RaggedDotDimensionNumbers(dot_dimension_numbers=(((0,), (0,)), ((), ())),
                                               lhs_ragged_dimensions=[0], rhs_group_dimensions=[])
    return jax.lax.ragged_dot_general(x, other, jnp.asarray(SIZES, jnp.int32), numbers)


def scaled_by_its_first_bias(x, w, bias, other):
    """A projection times a scalar the program reads from its bias: an epilogue capturing an array that
    follows no launch axis, which the template refuses, so its product is weighed alone (row 8996)."""
    del other
    return (x @ w) * bias[0]


def formed_and_added_back(x, w, bias, other):
    """Rows formed before a square projection (its prologue), its product scaled by a scalar after it (an
    epilogue the template refuses) and the rows added back: the region without its epilogue keeps the rows
    its restored consumer reads (row 8996's head review: it removed them)."""
    del x, w
    rows = other * jnp.bfloat16(2)
    return (rows @ other) * bias[0] + rows


def formed_and_returned(x, w, bias, other):
    """Rows formed before a projection and returned beside it, its product scaled by a scalar after it: a
    prologue the region replays though the program keeps it, whose operand the region without its epilogue
    reads (row 8996's head review: its inputs missed it)."""
    del other
    rows = x * jnp.bfloat16(2)
    return (rows @ w) * bias[0], rows


LAYERS = [biased, escaping, widened, chained, bare, kernel_gradient, grouped, grouped_kernel_gradient,
          scaled_by_its_first_bias, formed_and_added_back, formed_and_returned]


def as_written(layer, *operands):
    """`layer` as written over `operands`, prepared ahead where a part prepares (`ran`)."""
    return ran(layer, f"{layer.__name__} as written", *operands)


def under_the_pass(layer, *operands):
    """`layer` under the pass with its kernels over `operands`, prepared ahead where a part prepares
    (`ran`). Named apart from `algebra.rewritten`, which `_passed` calls: a statement reads each name a
    function's code holds, an attribute's too (`dew._sources.names_read`), so a global of that name is
    read, and this one reads `ran`'s store (rows 10064 on: an arm no statement held)."""
    return ran(functools.partial(_passed, layer), f"{layer.__name__} under the pass", *operands)


def assert_equal(got, expected):
    for value, reference in zip(jax.tree.leaves(got), jax.tree.leaves(expected), strict=True):
        np.testing.assert_array_equal(np.asarray(value, np.float32), np.asarray(reference, np.float32))


ARMS = [pytest.param(name, named, marks=lowers(name), id=label)
        for name, named, label in (("sm90", {"grouped_matmul": "mosaic"}, "sm90-named"),
                                   ("sm80", {"grouped_matmul": "mosaic"}, "sm80-named"),
                                   ("sm90", {}, "sm90"), ("sm80", {}, "sm80"))]
"""Each GPU and the kernels a call names: an H100's template named for its projections (an A/B's arm, where
the pass fuses their consumers into its epilogue), an A100's (whose template takes no side operand, so the
pass keeps the program as traced), and an H100's and an A100's selectors, which give a dense projection
XLA's dot. Each runs where this process lowers its generation's Mosaic kernels as it does (`lowers`): a
CPU interprets every one; a GPU runs its own (an A100 refuses an H100's tile, past its shared memory, and an
H100 an A100's copies)."""


def test_a_program_writing_a_ref_inside_a_nested_body_computes_as_written():
    """A program one of whose equations makes no value but writes a ref in its nested body, which binds the
    anchors the pass reads regions around (an exponential, an inverse square root, a sigmoid: a launch that
    rewalks into its outputs in place, a fused backward's dQ), under the pass with its kernels: the
    recognizer reads no region around it, and the program computes as written, bitwise."""
    def written(x):
        out = jax.new_ref(jnp.zeros_like(x))

        def anchored():
            out[...] = jnp.exp(x) + jax.lax.rsqrt(jnp.abs(x) + 1) + jax.nn.sigmoid(x)

        jax.lax.cond(jnp.all(jnp.isfinite(x)), anchored, lambda: None)
        return jax.freeze(out) * 2

    x = np.random.default_rng(4783).standard_normal((4, 8)).astype(np.float32)
    with generation.assumed("sm90"):
        assert_equal(under_the_pass(written, x), as_written(written, x))


@pytest.mark.parametrize("layer", LAYERS, ids=lambda layer: layer.__name__)
@pytest.mark.parametrize(("generation_name", "named"), ARMS)
def test_a_layer_under_the_pass_computes_what_it_does_as_written(layer, generation_name, named):
    """Each layer under the pass computes what it does as written, bitwise, in each arm (`ARMS`): an
    epilogue reads its product as the dot rounds it, a product read after its consumers is still its own,
    a widened product widens the dot's rounding."""
    operands = draw()
    with generation.assumed(generation_name), generation.chosen(**named):
        assert_equal(under_the_pass(layer, *operands), as_written(layer, *operands))


@pytest.mark.parametrize("layer", [biased, chained], ids=lambda layer: layer.__name__)
def test_a_layers_gradient_under_the_pass_is_its_gradient_as_written(layer):
    """The gradient of each layer's sum against a fixed cotangent, under the pass on a GPU whose template a
    call names (its backward's products candidates of their own), is the gradient as written, bitwise."""
    operands = draw()
    cotangent = np.random.default_rng(11).integers(-2, 3, (256, 256)).astype(jnp.bfloat16)

    def loss(f):
        return lambda *xs: jnp.sum((f(*xs) * cotangent).astype(jnp.float32))

    grad = jax.grad(loss(layer), argnums=(0, 1, 2))
    with generation.assumed("sm90"), generation.chosen(grouped_matmul="mosaic"):
        assert_equal(ran(functools.partial(_passed, grad, kernels=False), "the gradient under the pass",
                         *operands),
                     ran(grad, "the gradient as written", *operands))


@pytest.mark.parametrize("forms", ["fast", "accurate"])
@pytest.mark.parametrize("layer", [bare, kernel_gradient, grouped, grouped_kernel_gradient],
                         ids=lambda layer: layer.__name__)
def test_a_plain_programs_product_runs_the_template_its_call_names(layer, forms):
    """A plain JAX program's product of each form the template takes (`fusion.Form`: alone, a kernel
    gradient's, each group's rows times its matrix, a grouped kernel gradient's), under the pass on an H100
    whose template a call names, runs the template's launch (its route reported as the selector's), where
    the program as written runs XLA's dot; under the accurate forms too, which limit the pass's
    reassociation alone (row 4730)."""
    operands = draw()
    with (generation.assumed("sm90"), generation.chosen(grouped_matmul="mosaic"),
          generation.run_precision(Numerics(forms=forms))):
        with generation.routes() as written:
            jax.jit(layer).trace(*operands)
        with generation.routes() as passed:
            jax.jit(algebra.rewritten(layer, kernels=True)).trace(*operands)
    assert not any(route.choice == "grouped_matmul" for route in written)
    assert any(route.choice == "grouped_matmul" and route.kernel == "mosaic" for route in passed)


def _normalized(x, scale):
    """An RMSNorm's rows of `x`, widened, by their inverse root mean square and `scale`, cast back."""
    wide = x.astype(jnp.promote_types(x.dtype, jnp.float32))
    inverse = jax.lax.rsqrt(jnp.mean(jnp.square(wide), axis=-1, keepdims=True) + 1e-6)
    return (wide * inverse * scale.astype(wide.dtype)).astype(x.dtype)


def normed(x, w):
    """An RMSNorm's rows projected: the normalization the projection's prologue."""
    return _normalized(x, w[:, 0]) @ w


def normed_twice(x, w):
    """One RMSNorm's rows read by two projections (a block's q and k), each its own prologue."""
    rows = _normalized(x, w[:, 0])
    return rows @ w, rows @ w[:, ::-1]


@pytest.mark.parametrize("layer", [normed, normed_twice], ids=lambda layer: layer.__name__)
@lowers("sm90")
def test_a_norms_normalization_runs_in_its_projections_prologue(layer):
    """An RMSNorm's rows projected, under the pass on an H100 whose template a call names: each product's
    prologue is the normalization (the row statistic outside it), which the template runs on each loaded
    tile (its 'gemm_prologue' route the fused kernel's), each projection of one normalized row its own,
    where the pass took the normalized row as a given operand; and the program computes as written,
    gradients too (check 1)."""
    from contract_support import normal_drawn
    from reference_error import assert_as_exact, with_gradients

    shapes = (jax.ShapeDtypeStruct((256, 512), jnp.bfloat16), jax.ShapeDtypeStruct((512, 256), jnp.bfloat16))
    products = len(jax.tree.leaves(jax.eval_shape(layer, *shapes)))
    with generation.assumed("sm90"), generation.chosen(grouped_matmul="mosaic"):
        with generation.routes() as found:
            jax.jit(algebra.rewritten(layer, kernels=True)).trace(*shapes)
        fused = [route for route in found if route.choice == "gemm_prologue" and route.kernel == "fused"]
        assert len(fused) == products, [(route.choice, route.kernel) for route in found]

        outputs = jax.eval_shape(layer, *shapes)
        operands = normal_drawn(5096, (256, 512), (512, 256), dtype=jnp.bfloat16)
        cotangents = normal_drawn(5097, *(leaf.shape for leaf in jax.tree.leaves(outputs)),
                                  dtype=jnp.bfloat16)
        # Each projection's own output and gradient, a cotangent of its own: two outputs swapped or one
        # returned twice fail where their sum would not.
        assert_as_exact(jax.jit(with_gradients(functools.partial(_passed, layer))),
                        jax.jit(with_gradients(layer)),
                        lambda seed: (*operands(seed), jax.tree.unflatten(jax.tree.structure(outputs),
                                                                          list(cotangents(seed)))),
                        f"{layer.__name__} under the pass", seeds=8,
                        truth_fn=with_gradients(lambda *x: layer(*(a.astype(jnp.float64) for a in x))))


def normed_from_float32_twice(x, w):
    """A float32 row normalized and cast to bfloat16, read by two projections (a block's q and k): a
    prologue the template refuses (`fused_gemm`'s 'dtype_changes', it would multiply the bfloat16 base in
    float32)."""
    wide = x.astype(jnp.float32)
    inverse = jax.lax.rsqrt(jnp.mean(jnp.square(wide), axis=-1, keepdims=True) + 1e-6)
    rows = (wide * inverse * w[:, 0].astype(jnp.float32)).astype(jnp.bfloat16)
    return rows @ w, rows @ w[:, ::-1]


def test_a_projection_whose_forced_prologue_refuses_is_fused_without_it():
    """A float32 row normalized to bfloat16 and read by two projections, on an H100 whose call names the
    template and its prologue: the prologue's stage refuses before the analysis reaches the launch, so the
    launch is undecided and each region is weighed again without its prologue, its product the template's
    (the region dropped when only a launch reached and found XLA's rules it out, train 965's F2)."""
    shapes = (jax.ShapeDtypeStruct((256, 512), jnp.float32), jax.ShapeDtypeStruct((512, 256), jnp.bfloat16))
    with generation.assumed("sm90"), generation.chosen(grouped_matmul="mosaic", gemm_prologue="fused"), \
            generation.routes() as found:
        jax.jit(algebra.rewritten(normed_from_float32_twice, kernels=True)).trace(*shapes)
    assert {route.kernel for route in found if route.choice == "grouped_matmul"} == {"mosaic"}, [
        (route.choice, route.kernel) for route in found]


def test_a_prologue_the_template_refuses_leaves_the_rows_computed_once():
    """A float32 row normalized to bfloat16 and read by two projections, under the pass on an H100 whose
    template a call names: the template refuses the prologue, so each region is weighed without it, each
    projection's call reading the normalized bfloat16 rows the program computes once, never the float32 row
    it would normalize again inside (train 961's F4); no prologue route is taken."""
    shapes = (jax.ShapeDtypeStruct((256, 512), jnp.float32), jax.ShapeDtypeStruct((512, 256), jnp.bfloat16))
    with generation.assumed("sm90"), generation.chosen(grouped_matmul="mosaic"), generation.routes() as found:
        traced = jax.jit(algebra.rewritten(normed_from_float32_twice, kernels=True)).trace(*shapes)
    assert not [route for route in found if route.choice == "gemm_prologue"], [
        (route.choice, route.kernel) for route in found]
    program = traced.jaxpr.jaxpr
    projections = [equation for equation in program.eqns if set(equation.outvars) & set(program.outvars)]
    rows = [var.aval.dtype for equation in projections for var in equation.invars
            if isinstance(var, jax.extend.core.Var) and var.aval.shape == (256, 512)]
    assert len(projections) == 2 and rows == [jnp.bfloat16, jnp.bfloat16], rows


def scaled_by_an_array(x, w, scale):
    """A projection times a scalar array: an epilogue capturing an array that follows no launch axis tile by
    tile, which the template refuses (`fused_gemm`'s 'value_off_axes')."""
    return (x @ w) * scale


def test_a_product_whose_epilogue_the_template_refuses_is_fused_without_it():
    """A projection times a scalar array, on an H100 whose call names the template: the template refuses the
    epilogue, so the region is weighed again without it, its product the template's launch and the multiply
    after it as the program wrote it, outside the written region (row 8996: the region refused, its product
    stayed a plain dot)."""
    shapes = (jax.ShapeDtypeStruct((256, 512), jnp.bfloat16), jax.ShapeDtypeStruct((512, 256), jnp.bfloat16),
              jax.ShapeDtypeStruct((), jnp.bfloat16))
    with generation.assumed("sm90"), generation.chosen(grouped_matmul="mosaic"), generation.routes() as found:
        traced = jax.jit(algebra.rewritten(scaled_by_an_array, kernels=True)).trace(*shapes)
    assert {route.kernel for route in found if route.choice == "grouped_matmul"} == {"mosaic"}, [
        (route.choice, route.kernel) for route in found]
    program = traced.jaxpr.jaxpr
    (last,) = (equation for equation in program.eqns if set(equation.outvars) & set(program.outvars))
    assert last.primitive is jax.lax.mul_p


def four_float32_terms(x, w, a, b, c, d):
    """A projection whose consumers read four float32 arrays of its product's shape: on an A100 its epilogue's
    side tiles derive the launch at one program a multiprocessor, where the product alone takes two
    (FS-HOOKED-RES)."""
    return ((x @ w).astype(jnp.float32) * a + b * c + d).astype(jnp.bfloat16)


def one_float32_term(x, w, a):
    """A projection plus one float32 array of its product's shape, which keeps the launch's programs."""
    return ((x @ w).astype(jnp.float32) + a).astype(jnp.bfloat16)


def test_a_neighbour_costing_the_launch_a_program_stays_outside_it():
    """A neighbour joins the kernel where the kernel with it keeps its residency (KERNEL-TEMPLATES-13, row
    9032). On an A100 whose template a call names: a projection whose consumers read four float32 arrays of
    its product's shape is not fused, though its price favours it, its product the template's launch at the
    programs a multiprocessor the product alone takes and its consumers XLA's after it; one float32 array,
    which keeps the launch's programs, is its epilogue."""
    from dew.nn.kernels import tuning

    x, w = jax.ShapeDtypeStruct((2048, 2560), jnp.bfloat16), jax.ShapeDtypeStruct((2560, 9216), jnp.bfloat16)
    term = jax.ShapeDtypeStruct((2048, 9216), jnp.float32)
    with generation.assumed("sm80"), generation.chosen(grouped_matmul="mosaic"):
        with tuning.seen() as alone:
            jax.jit(algebra.rewritten(lambda x, w: x @ w, kernels=True)).trace(x, w)
        with generation.routes() as found, tuning.seen() as four:
            terms = (term,) * 4
            traced = jax.jit(algebra.rewritten(four_float32_terms, kernels=True)).trace(x, w, *terms)
        fused = [route for route in found if route.choice == "gemm_epilogue" and route.kernel == "fused"]
        assert not fused, [(route.choice, route.kernel) for route in found]
        programs = [config.programs for config, _ in four.taken]
        assert programs and programs == [config.programs for config, _ in alone.taken], (four.taken,
                                                                                         alone.taken)
        program = traced.jaxpr.jaxpr
        (last,) = (equation for equation in program.eqns if set(equation.outvars) & set(program.outvars))
        assert last.primitive is jax.lax.convert_element_type_p, last.primitive
        with generation.routes() as found:
            jax.jit(algebra.rewritten(one_float32_term, kernels=True)).trace(x, w, term)
        assert any(route.choice == "gemm_epilogue" and route.kernel == "fused" for route in found), [
            (route.choice, route.kernel) for route in found]


@pytest.mark.parametrize(("function", "target", "fuses"),
                         [(jnp.exp2, "sm90", True), (jnp.sin, "sm90", False), (jnp.exp, "sm100", True),
                          (jnp.exp, "sm120", True)],
                         ids=["exp2-sm90", "sin-sm90", "exp-sm100", "exp-sm120"])
def test_a_consumer_its_launch_prices_no_work_of_is_never_fused_for_free(function, target, fuses):
    """A projection's consumer is fused where its price says fused wins and its launch carries no more work
    no rule prices than XLA's fusion of it. On a GPU whose template a call names: an exp2 after the product,
    one MUFU either way, is its epilogue on an H100; a sine, which the launch compiles in an accurate form no
    count prices (`emission.ACCURATE_FORMS`) where XLA's fusion is counted one MUFU an element, stays XLA's
    after it, where the fused arm took it for no time (train 1165's F2); an exp on a B200 or an RTX 50, whose
    accurate form's integer operations run at a rate their facts leave unread, is its epilogue, where those
    counted as unpriced kept every exp, SiLU and GELU off the launch (row 11806). Shapes only."""
    x, w = jax.ShapeDtypeStruct((8192, 1024), jnp.bfloat16), jax.ShapeDtypeStruct((1024, 4096), jnp.bfloat16)

    def projected(x, w):
        return function((x @ w).astype(jnp.float32)).astype(jnp.bfloat16)

    with generation.assumed(target), generation.chosen(grouped_matmul="mosaic"), generation.routes() as found:
        jax.jit(algebra.rewritten(projected, kernels=True)).trace(x, w)
    fused = any(route.choice == "gemm_epilogue" and route.kernel == "fused" for route in found)
    assert fused == fuses, [(route.choice, route.kernel) for route in found]


def normed_and_terms(x, scale, w, *terms):
    """An RMS norm feeding a bf16 projection (the norm the product's prologue, as a measured cell times it)
    whose consumers add float32 arrays of its product's shape (`terms`)."""
    wide = x.astype(jnp.float32)
    inverse = jax.lax.rsqrt(jnp.mean(wide * wide, axis=-1, keepdims=True) + 1e-6)
    out = jnp.dot((wide * inverse * scale).astype(x.dtype), w, preferred_element_type=jnp.float32)
    return sum(terms, out).astype(jnp.bfloat16)


def test_a_neighbour_costing_a_selected_launch_a_program_stays_outside_it():
    """KERNEL-TEMPLATES-13 (row 9032) where the selector, no call, takes the template: on an A100, the normed
    q/k/v projection of Qwen3.5-0.8B (its prologue's cell measured the template faster, the product unnormed
    XLA's dot) whose consumers read four float32 arrays of its product's shape is not fused, its launch at one
    program a multiprocessor where the template's launch of the product alone takes two; with one array,
    which keeps its programs, it is."""
    x, scale = jax.ShapeDtypeStruct((12288, 1024), jnp.bfloat16), jax.ShapeDtypeStruct((1024,), jnp.float32)
    w = jax.ShapeDtypeStruct((1024, 5120), jnp.bfloat16)
    term = jax.ShapeDtypeStruct((12288, 5120), jnp.float32)
    with generation.assumed("sm80"):
        for terms, fuses in (((term,) * 4, False), ((term,), True)):
            with generation.routes() as found:
                jax.jit(algebra.rewritten(normed_and_terms, kernels=True)).trace(x, scale, w, *terms)
            fused = any(route.choice == "gemm_epilogue" and route.kernel == "fused" for route in found)
            assert fused == fuses, (len(terms), [(route.choice, route.kernel) for route in found])


def named_gate(x, w, up):
    """A gate projection named for remat (`checkpoint_name`, as Dew's models name gate_proj) and its SwiGLU
    with `up` after it, the activation written entry by entry (`jax.nn.silu`'s own derivative rule is a call
    no epilogue enters)."""
    from jax.ad_checkpoint import checkpoint_name

    gate = checkpoint_name(x @ w, "gate_proj")
    return gate * jax.lax.logistic(gate) * up


@lowers("sm90")
def test_a_named_products_consumers_are_its_epilogue_its_name_the_launchs_residual():
    """A product named for remat, its SwiGLU after the name, under the pass on an H100 whose template a call
    names: the SwiGLU is the product's epilogue (its 'gemm_epilogue' route the fused kernel's), where the
    name ended the region; the name is the launch's own, on a value the fused call hands out (its residual,
    `fused_gemm._named`), no product of XLA's left to name; the program computes as written, gradients too
    (check 1)."""
    from contract_support import normal_drawn
    from reference_error import assert_as_exact, with_gradients

    shapes = (jax.ShapeDtypeStruct((256, 512), jnp.bfloat16), jax.ShapeDtypeStruct((512, 256), jnp.bfloat16),
              jax.ShapeDtypeStruct((256, 256), jnp.bfloat16))
    with generation.assumed("sm90"), generation.chosen(grouped_matmul="mosaic"):
        with generation.routes() as found:
            traced = jax.jit(algebra.rewritten(named_gate, kernels=True)).trace(*shapes)
        assert any(route.choice == "gemm_epilogue" and route.kernel == "fused" for route in found), [
            (route.choice, route.kernel) for route in found]
        assert not [equation for equation in traced.jaxpr.jaxpr.eqns
                    if equation.primitive is jax.lax.dot_general_p]
        operands = normal_drawn(5098, (256, 512), (512, 256), (256, 256), dtype=jnp.bfloat16)
        cotangent = normal_drawn(5099, (256, 256), dtype=jnp.bfloat16)
        assert_as_exact(jax.jit(with_gradients(functools.partial(_passed, named_gate))),
                        jax.jit(with_gradients(named_gate)),
                        lambda seed: (*operands(seed), *cotangent(seed)), "a named gate under the pass",
                        seeds=8,
                        truth_fn=with_gradients(lambda *x: named_gate(*(a.astype(jnp.float64) for a in x))))


def gated_mlp_product(x, w, up):
    """A gated MLP's gate projection, named for remat (`gate_proj`), and its SwiGLU with `up` as a model takes
    it (`activations.gated_product`: one marked region, `derivatives.Activation`, whose body is SwiGLU's
    Pallas kernel where an H100's selector chooses it)."""
    from jax.ad_checkpoint import checkpoint_name

    from dew.nn import activations

    return activations.gated_product("swiglu")(checkpoint_name(x @ w, "gate_proj"), up)


@lowers("sm90")
def test_a_models_gated_product_is_its_gate_projections_epilogue():
    """A model's SwiGLU (`gated_product`, its region marked `Activation`, its body SwiGLU's Pallas kernel on
    an H100) after its named gate projection, under the pass on an H100 whose template a call names: the
    region is the product's epilogue (its 'gemm_epilogue' route the fused kernel's), replayed as the
    activation's arithmetic on the tile (`activations.gated_hook`), no region of it or dot left in the
    program; the program computes as written, gradients too (check 1 against float64)."""
    from contract_support import normal_drawn
    from reference_error import assert_as_exact, with_gradients

    from dew.nn import derivatives

    shapes = (jax.ShapeDtypeStruct((256, 512), jnp.bfloat16), jax.ShapeDtypeStruct((512, 256), jnp.bfloat16),
              jax.ShapeDtypeStruct((256, 256), jnp.bfloat16))
    with generation.assumed("sm90"), generation.chosen(grouped_matmul="mosaic"):
        with generation.routes() as found:
            traced = jax.jit(algebra.rewritten(gated_mlp_product, kernels=True)).trace(*shapes)
        assert any(route.choice == "gemm_epilogue" and route.kernel == "fused" for route in found), [
            (route.choice, route.kernel) for route in found]
        left = [equation for equation in traced.jaxpr.jaxpr.eqns
                if equation.primitive is jax.lax.dot_general_p
                or isinstance(equation.params.get("kind"), derivatives.Activation)]
        assert not left, left
    operands = normal_drawn(5101, (256, 512), (512, 256), (256, 256), dtype=jnp.bfloat16)
    cotangent = normal_drawn(5102, (256, 256), dtype=jnp.bfloat16)

    def wide(x, w, up):
        return jax.nn.silu(x.astype(jnp.float64) @ w.astype(jnp.float64)) * up.astype(jnp.float64)

    # Run as the H100's routes on this CPU: their kernels interpreted where it places them.
    with generation.assumed("sm90", interpreted=True), generation.chosen(grouped_matmul="mosaic"):
        assert_as_exact(jax.jit(with_gradients(functools.partial(_passed, gated_mlp_product))),
                        jax.jit(with_gradients(gated_mlp_product)),
                        lambda seed: (*operands(seed), *cotangent(seed)),
                        "a model's gated product under the pass", seeds=8, truth_fn=with_gradients(wide))


def gated_projection(gate, up, down):
    from dew.nn import activations

    return activations.gated_product("swiglu")(gate, up) @ down


@lowers("sm90")
def test_a_gated_activation_can_feed_the_next_products_prologue():
    from contract_support import normal_drawn
    from reference_error import assert_as_exact, with_gradients

    dims = ((256, 512), (256, 512), (512, 256))
    operands = normal_drawn(3422, *dims, (256, 256), dtype=jnp.bfloat16)
    shapes = tuple(jax.ShapeDtypeStruct(shape, jnp.bfloat16) for shape in dims)
    with (generation.assumed("sm90"), generation.chosen(grouped_matmul="mosaic", gemm_prologue="fused"),
          generation.routes() as routes):
        jax.jit(algebra.rewritten(gated_projection, kernels=True)).trace(*shapes)
    assert any(route.choice == "gemm_prologue" and route.kernel == "fused" for route in routes)

    def truth(gate, up, down):
        gate, up, down = (part.astype(jnp.float64) for part in (gate, up, down))
        return (jax.nn.silu(gate) * up) @ down

    def directional(call):
        def run(*values):
            return jax.jvp(call, values[:3], values[3:])
        return jax.jit(run)

    with (generation.assumed("sm90", interpreted=True),
          generation.chosen(grouped_matmul="mosaic", gemm_prologue="fused")):
        fused = functools.partial(_passed, gated_projection)
        assert_as_exact(jax.jit(with_gradients(fused)), jax.jit(with_gradients(gated_projection)), operands,
                        "gated product before a projection", seeds=8, truth_fn=with_gradients(truth))
        assert_as_exact(directional(fused), directional(gated_projection),
                        normal_drawn(3423, *dims, *dims, dtype=jnp.bfloat16), "gated projection tile JVP",
                        seeds=8, truth_fn=directional(truth))
        gate, up, down = (jnp.ones(shape, jnp.bfloat16) for shape in dims)

        def loss(value):
            return fused(gate.at[0, 0].set(value), up, down)[0, 0].astype(jnp.float32)

        assert jnp.isfinite(jax.jit(jax.grad(jax.grad(loss)))(jnp.bfloat16(-100)))


def test_a_gated_product_of_no_product_stays_its_region():
    """A model's SwiGLU (`gated_product`) over projections the program takes as inputs, no product before it:
    the pass leaves its region whole (`derivatives.Activation`, its body the route its dispatcher chose on an
    H100), the activation fused into no launch, and the program computes it bit for bit as written."""
    from dew.nn import activations, derivatives

    product = activations.gated_product("swiglu")
    gate, up = (np.random.default_rng(seed).standard_normal((256, 256)).astype(jnp.bfloat16)
                for seed in (1, 2))
    with generation.assumed("sm90"):
        traced = jax.jit(algebra.rewritten(product, kernels=True)).trace(gate, up)
        assert [equation for equation in traced.jaxpr.jaxpr.eqns
                if isinstance(equation.params.get("kind"), derivatives.Activation)]
    assert_equal(ran(functools.partial(_passed, product), "a SwiGLU under the pass", gate, up),
                 ran(product, "a SwiGLU as written", gate, up))


def named_rows(x, w):
    """An exp2 of rows named for remat before it, then their product: a name on a prologue's value."""
    from jax.ad_checkpoint import checkpoint_name

    return jnp.exp2(checkpoint_name(x, "rows")) @ w


def test_a_parts_prepare_holds_a_prologues_custom_derivative_by_its_name(tmp_path, monkeypatch):
    """In a GPU tier part's prepare (its node an H100), the norm-in-prologue cell's arms are held with the
    launches they took, each launch's prologue the normalization's custom derivative: held by its module and
    qualified name (`truths_ahead.HeldName`, which its run never calls), where pickling it refused the store
    and left the file unprepared (row 12657). Every site's held record reads back."""
    import pickle

    import tier_wire
    from generation_support import TOPOLOGY, node_target, node_topology
    from truths_ahead import HELD, prepared

    monkeypatch.setenv(TOPOLOGY, str(node_topology(tmp_path / "topology.bin", gpu="sm90")))
    monkeypatch.setenv(tier_wire.PREPARE, "1")
    with prepared(tmp_path / "store", stamp="a", target=node_target()):
        test_a_norms_normalization_runs_in_its_projections_prologue(normed)
    records = [path.read_bytes() for path in (tmp_path / "store").rglob(HELD)]
    assert records and any(b"HeldName" in record for record in records)
    for record in records:
        pickle.loads(record)


@pytest.mark.compiles_at_run(reason="a template's certification compiles its configs as the call traces, "
                                   "no check-1 site")
def test_a_name_on_a_products_rows_stays_outside_its_prologue():
    """A remat name on the value a product's prologue reads (`named_rows`) bounds the prologue, which still
    fuses the exp2 after it on an H100 whose template a call names: a producer's stage runs no name (a name
    pulled into it was refused there, and the exp2's operand formed whole). Its launch is certified as an
    H100's start certifies it, against a compile-only one, and a config is taken. (An accurate sine, whose
    form no count prices, is no free fused work, `test_a_consumer_its_launch_prices_no_work_of_is_never_
    fused_for_free`; a frame its range reduction keeps refuses no config, test_kernel_spills, row 9649.)"""
    shapes = (jax.ShapeDtypeStruct((256, 512), jnp.bfloat16), jax.ShapeDtypeStruct((512, 256), jnp.bfloat16))
    [device] = generation.compile_only_gpus("sm90")
    with generation.assumed("sm90"), generation.chosen(grouped_matmul="mosaic"), \
            report.certifying(device), generation.routes() as found:
        traced = jax.jit(algebra.rewritten(named_rows, kernels=True)).trace(*shapes)
    assert any(route.choice == "gemm_prologue" and route.kernel == "fused" for route in found), [
        (route.choice, route.kernel) for route in found]
    assert not [equation for equation in traced.jaxpr.jaxpr.eqns if equation.primitive is jax.lax.exp2_p]


@pytest.mark.mesh(devices=2)
def test_a_product_mapped_per_shard_runs_the_template_and_computes_as_written():
    """A projection, its bias and a scale (`biased`) mapped per shard (`jax.shard_map` over rows split two
    ways, each device's program its own rows' product, as `dew.nn.sharding.on_shards` maps a layer) under
    the pass on an H100 whose template a call names: the map's body is rewritten in the map's own context,
    so its product runs the template's launch per shard (row 6578: an opaque body ran XLA's dot), and the
    program computes as written, as every program under the pass does (`assert_equal`)."""
    from jax.sharding import Mesh, PartitionSpec

    mesh = Mesh(np.asarray(jax.devices()[:2]), ("data",))
    rows, whole = PartitionSpec("data"), PartitionSpec()
    mapped = jax.shard_map(biased, mesh=mesh, in_specs=(rows, whole, whole, rows), out_specs=rows)
    operands = draw()
    with generation.assumed("sm90"), generation.chosen(grouped_matmul="mosaic"):
        with generation.routes() as found:
            jax.jit(algebra.rewritten(mapped, kernels=True)).trace(*operands)
        assert any(route.choice == "grouped_matmul" and route.kernel == "mosaic" for route in found)
        assert_equal(under_the_pass(mapped, *operands), as_written(mapped, *operands))


def test_a_starts_forward_nothing_reads_is_never_written_as_a_kernel():
    """A start-like program (row 6783): a region that makes a parameter and computes a grouped product from
    it, as a flax init's block makes its parameters inside its forward (here a checkpointed one), the
    product's result read by nothing. Through `dew.jit` on an H100 whose template a call names, no route is
    taken and no launch traced for it, so nothing of it is certified or built; the same program reading the
    product runs the template's launch."""
    import dew
    from dew.nn.kernels import tuning

    def start(read: bool):
        def program(x, w, bias, other):
            def block(x, w):
                made = w * 2
                return made, grouped(x, made, bias, other)

            made, out = jax.checkpoint(block)(x, w)
            return (made, out) if read else made
        return program

    operands = draw()
    taken = []
    with generation.assumed("sm90"), generation.chosen(grouped_matmul="mosaic"):
        for read in (True, False):
            with generation.routes() as found, tuning.seen() as launches:
                dew.jit(start(read)).trace(*operands)
            took = any(route.choice == "grouped_matmul" and route.kernel == "mosaic" for route in found)
            taken.append((took, len(launches)))
    (read, launched), (unread, unlaunched) = taken
    assert read and launched
    assert not unread and not unlaunched


def test_a_steps_relowering_writes_the_program_its_first_lowering_wrote():
    """A step lowered again in one process (a planner's candidate, a relowering after a launch spilled: each
    lowering's records fresh) writes its regions from the traces its first lowering made (`fusion._traced`):
    the program the first wrote, reporting the routes it did and the launches it did, each by its launch's
    type beside the config it took (`tuning.seen`, whose launches and configs pair one to one: a tuner's
    and a test's reads, row 6572); and one lowered under other state a trace reads (a context variable of
    Dew's set, attention's forward mode), whose regions are traced again, takes those routes and launches
    too. The facts its replayed traces read are the next test's (`generation.recording`)."""
    import dew
    from dew.algebra import program
    from dew.nn.attention import forward_mode_attention
    from dew.nn.kernels import tuning

    operands = draw()
    step = dew.jit(chained)
    lowerings = []
    with generation.assumed("sm90"), generation.chosen(grouped_matmul="mosaic"):
        for state in (contextlib.nullcontext(), contextlib.nullcontext(), forward_mode_attention()):
            with state, generation.routes() as found, tuning.seen() as launches:
                made = step.trace(*operands)
            # Each launch by its op's type and the config it took, paired one to one (`tuning.Seen`).
            assert [op for op, _ in launches] == [bound.op for _, bound in launches.taken]
            took = [(type(bound.op).__name__, config) for config, bound in launches.taken]
            lowerings.append((program.structure(made.jaxpr.jaxpr),
                              [(route.choice, route.kernel) for route in found], took))
    cold, warm, other = lowerings
    assert ("grouped_matmul", "mosaic") in cold[1] and cold[2], cold
    assert warm == cold
    # Its regions traced again under the other state, which their program records (a forward-mode trace's
    # own marks): the routes and the launches it took are the first's.
    assert other[1:] == cold[1:]


def test_a_written_region_records_the_facts_its_trace_read_wherever_it_is_written():
    """A step lowered twice in one process (`dew.jit`), each within a recording (`_facts.recording`): the
    second writes its regions from the traces the first made (`fusion._traced`), and its recording holds the
    facts their traces read (the fused products' launches' legality, `report.refused`), as the first's did,
    where a held region dropped them and a trace kept under the second recording could not tell when they
    no longer held (row 6708)."""
    import dew

    operands = draw()
    step = dew.jit(chained)
    recorded = []
    with generation.assumed("sm90"), generation.chosen(grouped_matmul="mosaic"):
        for _ in range(2):
            with _facts.recording() as reads:
                step.trace(*operands)
            recorded.append(reads)
    cold, warm = recorded
    assert cold and set(warm) >= set(cold), (cold, warm)


def attended(query, key, value):
    """Attention written by hand, its scores in fp32: a region recognition weighs (`recognize`)."""
    scores = jnp.einsum("bqhd,bkhd->bhqk", query, key) * 0.125
    weights = jax.nn.softmax(scores.astype(jnp.float32), -1).astype(query.dtype)
    return jnp.einsum("bhqk,bkhd->bqhd", weights, value)


@pytest.mark.parametrize("case", ["declined products", "attention by hand"])
def test_a_declined_region_records_the_facts_its_probes_read_wherever_it_is_weighed(case):
    """A step lowered twice in one process on an H100 (`dew.jit`), each within a recording, whose regions
    the pass weighs and declines (its products XLA's where no call names the template; attention written by
    hand, which recognition weighs and leaves): the second reads its decisions from those the first held
    (`_cache.holding`), and its recording holds the facts their probes read (a launch's legality,
    `report.refusals_of`; FlashAttention's lowering and the device a call is placed on) as the first's did,
    and so does one around it (row 7167: a declined region's held decision dropped them, so a warm
    lowering's recording could not tell when they no longer held)."""
    import dew

    rng = np.random.default_rng(0)
    function, operands = ((chained, draw()) if case == "declined products" else
                          (attended, tuple(rng.normal(size=(1, 256, 2, 64)).astype(jnp.bfloat16)
                                           for _ in range(3))))
    step = dew.jit(function)
    with generation.assumed("sm90"):
        with _facts.recording() as cold:
            step.trace(*operands)
        with _facts.recording() as outer, _facts.recording() as warm:
            step.trace(*operands)
    assert cold and set(warm) >= set(cold) and set(outer) >= set(cold), (cold, warm)


def _template_routes(trace) -> list:
    """The grouped matmul's routes `trace()` reports that run the template."""
    with generation.routes() as found:
        trace()
    return [route for route in found if route.choice == "grouped_matmul" and route.kernel == "mosaic"]


@pytest.mark.parametrize("order", ["float16 by bfloat16", "bfloat16 by float16"])
def test_a_product_of_two_16_bit_formats_is_jaxs_own_through_dew_jit(order):
    """A product of a float16 and a bfloat16 matrix, no preferred type stated, in either order: through
    `dew.jit` (the algebra's pass, which prices it) its value is `512 2^-8 2^16 = 2^17` exactly, every product
    and partial sum a power of two; under the pass on an H100 whose template a call names, it is no launch of
    the template, which casts its right operand to its left's dtype (a bfloat16 `2^16` an infinite float16:
    base review 7754e5fde F1), but JAX's own product, which multiplies each as it is. Pricing it, NumPy's
    promotion of the pair raised (base review 0b8b7ed98 F1)."""
    import dew

    small, large = np.full((256, 512), 2.0 ** -8, np.float16), np.full((512, 256), 2.0 ** 16, jnp.bfloat16)
    left, right = (small, large) if order == "float16 by bfloat16" else (large.T, small.T)

    def product(left, right):
        return jax.lax.dot_general(left, right, (((1,), (0,)), ((), ())))

    made = jax.eval_shape(product, left, right)
    value = np.asarray(ran(dew.jit(product), f"a mixed product through dew.jit, {order}", left, right),
                       np.float64)
    assert (value.shape, value.min(), value.max()) == (made.shape, 2.0 ** 17, 2.0 ** 17)
    with (generation.assumed("sm90"), generation.chosen(grouped_matmul="mosaic"),
          generation.routes() as passed):
        jax.jit(algebra.rewritten(product, kernels=True)).trace(left, right)
    assert not any(route.choice == "grouped_matmul" for route in passed)


@pytest.mark.gpu(reason="a mixed product through dew.jit, Mosaic's template named, on an H100")
@pytest.mark.parametrize("order", ["float16 by bfloat16", "bfloat16 by float16"])
def test_a_product_of_two_16_bit_formats_through_dew_jit_naming_mosaic_holds_check_1(order):
    """A product of a float16 and a bfloat16 matrix in either order, through `dew.jit` on an H100 whose
    grouped products' template a call names (Mosaic's): no launch of the template (its route unreported),
    and its value on par with JAX's own product from float64 (check 1), drawn over the formats' range (base
    review 3232beb7c: the CPU witness traced it under the pass and never ran it through `dew.jit` there)."""
    from reference_error import as_given, assert_as_exact

    import dew
    from dew.parity import FEWEST_SEEDS

    def product(left, right):
        return jax.lax.dot_general(left, right, (((1,), (0,)), ((), ())))

    def draw(seed):
        rng = np.random.default_rng([0x8818, seed])
        small = jnp.asarray(rng.standard_normal((256, 512)) * 2.0 ** -8, jnp.float16)
        large = jnp.asarray(rng.standard_normal((512, 256)) * 2.0 ** 8, jnp.bfloat16)
        return (small, large) if order == "float16 by bfloat16" else (large.T, small.T)

    compiled = dew.jit(product)
    routed: list = []
    dtypes: set = set()

    def named(left, right):
        with generation.chosen(grouped_matmul="mosaic"), generation.routes() as taken:
            made = compiled(left, right)
        routed.extend(taken)
        # Its dtype beside the one JAX's own product makes, which check 1's float64 comparison never reads.
        dtypes.add((made.dtype, jax.eval_shape(product, left, right).dtype))
        return made

    assert_as_exact(as_given(named), product, draw, f"a mixed product, {order}, Mosaic named",
                    seeds=FEWEST_SEEDS)
    assert not any(route.choice == "grouped_matmul" for route in routed)
    assert all(made == own for made, own in dtypes)


@lowers("sm90")
def test_a_plain_programs_gradient_runs_the_template_in_either_order_through_dew_jit():
    """`dew.jit` over a plain JAX loss on an H100 whose template a call names: its gradient's products run
    the template whether `jax.grad` is taken of the compiled program (inside a plain jit, the pass reads the
    program as it traces) or the program compiles the gradient (the pass reads the joint program), each
    the gradient as written, bitwise. The loss contracts its projection whole with a fixed cotangent, so
    the projection is a product alone: a fused epilogue's derivative recomputes its sums in IEEE fp32,
    which the template does not take, and a call naming it refuses."""
    import dew

    operands = draw()
    cotangent = np.random.default_rng(11).integers(-2, 3, (256, 256)).astype(jnp.bfloat16)

    def loss(x, w, bias, other):
        del bias, other
        return jnp.tensordot(x @ w, cotangent, axes=2).astype(jnp.float32)

    plain = jax.grad(loss, argnums=(0, 1))
    outside = jax.jit(jax.grad(dew.jit(loss), argnums=(0, 1)))
    inside = dew.jit(jax.grad(loss, argnums=(0, 1)))
    with generation.assumed("sm90"), generation.chosen(grouped_matmul="mosaic"):
        assert not _template_routes(lambda: jax.jit(plain).trace(*operands))
        assert _template_routes(lambda: outside.trace(*operands))
        assert _template_routes(lambda: inside.trace(*operands))
        expected = ran(plain, "the gradient as written", *operands)
        assert_equal(ran(outside, "the gradient of the compiled loss", *operands), expected)
        assert_equal(ran(inside, "the compiled gradient", *operands), expected)


@pytest.mark.mesh(devices=2)
def test_a_mapped_products_written_casts_are_replayed_as_written():
    """A manual mesh's product of row-split operands by a scale and a weight the program cast to vary itself
    (`pcast`, JAX writing no cast of its own: `jax_auto_pcast` off) under the pass on an H100 whose template
    a call names: the fusion analysis's replay of the region binds each operand as the program typed it, the
    casts' results among them (row 6578's review, F2: a cast replayed as the value it cast, the multiply's
    operands varied apart and JAX refused it), its product the template's launch per shard, and the program
    computes as written."""
    from jax._src import config as jax_config
    from jax.sharding import Mesh, PartitionSpec

    def scaled_product(x, w, scale):
        scale, w = (jax.lax.pcast(value, ("data",), to="varying") for value in (scale, w))
        return (x * scale) @ w

    mesh = Mesh(np.asarray(jax.devices()[:2]), ("data",))
    rows, whole = PartitionSpec("data"), PartitionSpec()
    mapped = jax.shard_map(scaled_product, mesh=mesh, in_specs=(rows, whole, whole), out_specs=rows)
    x, w, *_ = draw()
    operands = (x, w, jnp.bfloat16(2))
    with (jax_config.auto_pcast(new_val=False), generation.assumed("sm90"),
          generation.chosen(grouped_matmul="mosaic")):
        with generation.routes() as found:
            jax.jit(algebra.rewritten(mapped, kernels=True)).trace(*operands)
        assert any(route.choice == "grouped_matmul" and route.kernel == "mosaic" for route in found)
        assert_equal(under_the_pass(mapped, *operands), as_written(mapped, *operands))
