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

from dew import algebra
from dew.nn.kernels import generation
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


def host_normal(seed: int, *shapes, dtype) -> tuple:
    """Standard normal arrays of `shapes` in `dtype`, from `seed`, as the host holds them (NumPy's), which a
    GPU tier part's prepare places nowhere (`draw`)."""
    rng = np.random.default_rng(seed)
    return tuple(rng.standard_normal(shape).astype(dtype) for shape in shapes)


def ran(fn, label: str, *operands):
    """`fn` jitted over host arrays `operands`, as a GPU tier part's prepare compiles it ahead and its run
    runs it (`reference_error.ahead_arm`: a bitwise check's program, computing nothing in the prepare);
    outside a part, run as it is."""
    from reference_error import ahead_arm

    return ahead_arm(jax.jit(fn), label)(*operands)


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
    """A sine of rows named for remat before it, then their product: a name on a prologue's value."""
    from jax.ad_checkpoint import checkpoint_name

    return jnp.sin(checkpoint_name(x, "rows")) @ w


def test_a_name_on_a_products_rows_stays_outside_its_prologue():
    """A remat name on the value a product's prologue reads (`named_rows`) bounds the prologue, which still
    fuses the sine after it on an H100 whose template a call names: a producer's stage runs no name (a name
    pulled into it was refused there, and the sine's operand formed whole)."""
    shapes = (jax.ShapeDtypeStruct((256, 512), jnp.bfloat16), jax.ShapeDtypeStruct((512, 256), jnp.bfloat16))
    with generation.assumed("sm90"), generation.chosen(grouped_matmul="mosaic"), \
            generation.routes() as found:
        traced = jax.jit(algebra.rewritten(named_rows, kernels=True)).trace(*shapes)
    assert any(route.choice == "gemm_prologue" and route.kernel == "fused" for route in found), [
        (route.choice, route.kernel) for route in found]
    assert not [equation for equation in traced.jaxpr.jaxpr.eqns if equation.primitive is jax.lax.sin_p]


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


def test_a_steps_relowering_traces_none_of_its_written_regions_again():
    """A step lowered again in one process (a planner's candidate, a relowering after a launch spilled: each
    lowering's records fresh, so JAX traces the step again) writes its regions from the traces its first
    lowering made (`fusion._traced`): the second traces nothing but the step itself (JAX's trace events, by
    the function each traced), where the first traced its fused products' calls too, and it writes the
    program the first did, reporting the routes it did and the launches it did, each beside the config it
    took (`tuning.seen`, whose launches and configs pair one to one: a tuner's and a test's reads, row
    6572). One lowered under other state a trace reads (a context variable of Dew's set, attention's
    forward mode) traces its regions again."""
    from jax._src import dispatch

    import dew
    from dew.algebra import program
    from dew.nn.attention import forward_mode_attention
    from dew.nn.kernels import tuning

    traced: list[str] = []

    def heard(event: str, duration: float, **values) -> None:
        del duration
        if event == dispatch.JAXPR_TRACE_EVENT:
            traced.append(values.get("fun_name"))

    jax.monitoring.register_event_duration_secs_listener(heard)
    operands = draw()
    step = dew.jit(chained)
    lowerings = []
    try:
        with generation.assumed("sm90"), generation.chosen(grouped_matmul="mosaic"):
            for state in (contextlib.nullcontext(), contextlib.nullcontext(), forward_mode_attention()):
                traced.clear()
                with state, generation.routes() as found, tuning.seen() as launches:
                    made = step.trace(*operands)
                assert [op for op, _ in launches] == [ba.op for _, ba in launches.taken]
                took = [(ba, config) for (_, ba), (config, _) in zip(launches, launches.taken, strict=True)]
                lowerings.append((set(traced), program.structure(made.jaxpr.jaxpr),
                                  [(route.choice, route.kernel) for route in found], took))
    finally:
        jax.monitoring.unregister_event_duration_listener(heard)
    (first, written, routes, took), (again, rewritten, rerouted, retook), (other, _, _, _) = lowerings
    assert first > {chained.__name__} and again == {chained.__name__}, (first, again)
    assert ("grouped_matmul", "mosaic") in routes and took
    assert rewritten == written and rerouted == routes and retook == took
    assert other > {chained.__name__}, other


def test_a_written_region_records_the_facts_its_trace_read_wherever_it_is_written():
    """A step lowered twice in one process (`dew.jit`), each within a recording (`generation.recording`): the
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
            with generation.recording() as reads:
                step.trace(*operands)
            recorded.append(reads)
    cold, warm = recorded
    assert cold and set(warm) >= set(cold), (cold, warm)


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


def jax_attention(*, causal: bool):
    """jax.nn's attention as a plain program writes it (its XLA form)."""
    def attend(q, k, v):
        return jax.nn.dot_product_attention(q, k, v, is_causal=causal, implementation="xla")
    return attend


def flax_attention(*, causal: bool, wide: bool = True):
    """Flax's attention as a Flax model writes it, its causal mask built from the rows, its softmax in
    float32 where `wide` (`force_fp32_for_softmax`, the kernels' arithmetic) and Flax's default, the
    operands' dtype, else."""
    from flax import linen as nn

    def attend(q, k, v):
        mask = nn.make_causal_mask(jnp.ones(q.shape[:2])) if causal else None
        return nn.dot_product_attention(q, k, v, mask=mask, force_fp32_for_softmax=wide)
    return attend


SPELLINGS = {"jax.nn causal, grouped": (jax_attention(causal=True), 2, True),
             "jax.nn, grouped": (jax_attention(causal=False), 2, False)}
"""Each plain program's attention, its key heads over 4 query heads, and whether it is causal."""

FLAX_SPELLINGS = {"softmax in bfloat16": False, "probabilities in float32": True}
"""Flax's attention, its softmax in its operands' bfloat16 (Flax's default) or in float32
(`force_fp32_for_softmax`), each different arithmetic from Dew's call: the one a softmax no kernel of Dew's
takes, the other the float32 probabilities carried into the value product unrounded (row 4860)."""


@pytest.mark.parametrize("spelling", list(FLAX_SPELLINGS))
def test_flaxs_attention_computes_as_written(spelling):
    """Flax's attention in either form (`FLAX_SPELLINGS`), compiled through `dew.jit` on an H100, runs as
    written, causal or not: no region is written as a kernel of Dew's (row 4783)."""
    import dew

    shapes = (jax.ShapeDtypeStruct((2, 128, 4, 64), jnp.bfloat16),) * 3
    with generation.assumed("sm90"):
        for causal in (False, True):
            attend = flax_attention(causal=causal, wide=FLAX_SPELLINGS[spelling])
            assert not _attention_routes(lambda attend=attend: dew.jit(attend).trace(*shapes))


def _attention_routes(trace) -> list[str]:
    """The kernels the attention routes `trace()` reports name."""
    with generation.routes() as found:
        trace()
    return [route.kernel for route in found if route.choice == "attention"]


@pytest.mark.parametrize("spelling", list(SPELLINGS))
def test_a_plain_programs_attention_runs_dews_kernel_through_dew_jit(spelling):
    """A plain JAX or Flax program's softmax attention (`dew.algebra.recognize`) compiled through
    `dew.jit` on an H100 runs the kernels Dew's own attention call routes to (`scaled_dot_product_attention`,
    or the template a call names), where the program as written runs XLA's; on this host, its output and
    gradients under the template (its portable mainloop) on par with the program as written (check 1)."""
    from attention_oracle import attention_on_par, kept
    from contract_support import normal_drawn

    import dew
    from dew.algebra.recognize import KEPT
    from dew.nn.attention import scaled_dot_product_attention

    attend, key_heads, causal = SPELLINGS[spelling]
    shapes = (jax.ShapeDtypeStruct((2, 128, 4, 64), jnp.bfloat16),
              *(jax.ShapeDtypeStruct((2, 128, key_heads, 64), jnp.bfloat16),) * 2)

    def own(q, k, v):
        return scaled_dot_product_attention(q, k, v, scale_logits=True, causal=causal)

    with generation.assumed("sm90"):
        assert not _attention_routes(lambda: jax.jit(attend).trace(*shapes))
        expected = {kernel for kernel in _attention_routes(lambda: jax.jit(own).trace(*shapes))
                    if kernel not in KEPT}
        assert expected and set(_attention_routes(lambda: dew.jit(attend).trace(*shapes))) == expected
        with generation.chosen(attention="template"):
            assert "template" in _attention_routes(lambda: dew.jit(attend).trace(*shapes))
    draw = normal_drawn(37, (2, 64, 4, 32), (2, 64, key_heads, 32), (2, 64, key_heads, 32), (2, 64, 4, 32),
                        dtype=jnp.bfloat16)
    with generation.chosen(attention="template"):
        attention_on_par(dew.jit(attend), attend, draw, f"dew.jit {spelling}",
                         keep=kept(64, 64, causal=causal))


def input_mask(q, k):
    """A mask drawn from the queries, `[B, H, Q, K]`: each query row keeps every key where its first entry is
    positive, none else."""
    rows = (q[..., :1] > 0).transpose(0, 2, 1, 3)
    return jnp.broadcast_to(rows, (q.shape[0], q.shape[2], q.shape[1], k.shape[1]))


def input_bias(q, k):
    """A bias drawn from the queries, `[B, H, Q, K]`: a query row's first entry over every key."""
    rows = (q[..., 0] * 0.25).transpose(0, 2, 1)[..., None]
    return jnp.broadcast_to(rows, (q.shape[0], q.shape[2], q.shape[1], k.shape[1])).astype(q.dtype)


def _wide(x):
    """The dtype a plain program computes `x`'s logits or statistics in: float32 over bfloat16 or float32
    operands, as its author writes it, and the operands' own where they are wider, so the program run on
    float64 operands is its float64 truth and no second fp32 run."""
    return jnp.promote_types(x.dtype, jnp.float32)


SOFTCAP = 50.0


def softcapped_attention(q, k, v):
    """Gemma 2's attention as a plain program writes it: scaled logits softcapped at 50, causal."""
    logits = jnp.einsum("bqhd,bkhd->bhqk", q, k, preferred_element_type=_wide(q)) * q.shape[-1] ** -0.5
    logits = SOFTCAP * jnp.tanh(logits / SOFTCAP)
    causal = jnp.tril(jnp.ones((q.shape[1], k.shape[1]), bool))
    weights = jax.nn.softmax(jnp.where(causal, logits, -jnp.inf), axis=-1).astype(v.dtype)
    return jnp.einsum("bhqk,bkhd->bqhd", weights, v)


JAX_FILL = -0.7 * float(jnp.finfo(jnp.float32).max)
"""jax.nn's fill for masked logits (its `_get_large_negative` at float32)."""


def keyed_mask(q, k):
    """`input_mask` with every query row keeping its first key: no row wholly masked."""
    return input_mask(q, k).at[..., 0].set(True)


def masked_attention(mask):
    """Attention written out over `mask(q, k)`, its masked logits -inf: a row the mask leaves no key is
    NaN."""
    def attend(q, k, v):
        logits = jnp.einsum("bqhd,bkhd->bhqk", q, k, preferred_element_type=_wide(q)) * q.shape[-1] ** -0.5
        weights = jax.nn.softmax(jnp.where(mask(q, k), logits, -jnp.inf), axis=-1).astype(v.dtype)
        return jnp.einsum("bhqk,bkhd->bqhd", weights, v)
    return attend


def jax_nn_written(*, mask=None, bias=None, scale=1, causal=False, window=None):
    """`jax.nn.dot_product_attention` under those arguments, written out at its operands' width: its logits
    scaled (`scale` times the default), biased, then filled with jax.nn's large negative (`JAX_FILL`)
    wherever the mask of the inputs, causality or a local window of `window` keys back drops a key. jax.nn's
    own runs its softmax in float32 whatever its operands' (`_dot_product_attention_core`), so on float64
    operands it is a second fp32 run and no truth of itself; this is the function it computes."""
    def attend(q, k, v):
        logits = jnp.einsum("bqhd,bkhd->bhqk", q, k, preferred_element_type=_wide(q))
        logits = logits * q.shape[-1] ** -0.5 * scale
        if bias is not None:
            logits = logits + bias(q, k)
        kept = jnp.ones(logits.shape, bool) if mask is None else mask(q, k)
        rows, keys = jnp.arange(q.shape[1])[:, None], jnp.arange(k.shape[1])[None, :]
        if causal:
            kept &= keys <= rows
        if window is not None:
            kept &= rows <= keys + window
        weights = jax.nn.softmax(jnp.where(kept, logits, JAX_FILL), axis=-1).astype(v.dtype)
        return jnp.einsum("bhqk,bkhd->bqhd", weights, v)
    return attend


WINDOW = 16
"""The causal window's keys a query sees, itself among them (jax.nn's lookback of WINDOW - 1): one width its
program, Dew's call and the oracle's keep set read."""

FORMS = {
    "a mask of its inputs": (
        lambda q, k, v: jax.nn.dot_product_attention(q, k, v, mask=input_mask(q, k), implementation="xla"),
        lambda q, k, v: {"mask": input_mask(q, k), "mask_fill": JAX_FILL}, jax_nn_written(mask=input_mask)),
    "a mask of its inputs filled with -inf": (
        masked_attention(keyed_mask), lambda q, k, v: {"mask": keyed_mask(q, k), "mask_fill": -jnp.inf},
        masked_attention(keyed_mask)),
    "a bias of its inputs": (
        lambda q, k, v: jax.nn.dot_product_attention(q, k, v, bias=input_bias(q, k), implementation="xla"),
        lambda q, k, v: {"bias": input_bias(q, k)}, jax_nn_written(bias=input_bias)),
    "twice the default scale": (
        lambda q, k, v: jax.nn.dot_product_attention(q, k, v, scale=2 * q.shape[-1] ** -0.5,
                                                     implementation="xla"),
        None, jax_nn_written(scale=2)),
    "a causal window": (
        lambda q, k, v: jax.nn.dot_product_attention(q, k, v, is_causal=True,
                                                     local_window_size=(WINDOW - 1, 0), implementation="xla"),
        lambda q, k, v: {"causal": True, "sliding_window": WINDOW, "mask_fill": JAX_FILL},
        jax_nn_written(causal=True, window=WINDOW - 1)),
    "a softcap": (softcapped_attention,
                  lambda q, k, v: {"causal": True, "softcap": SOFTCAP, "mask_fill": -jnp.inf},
                  softcapped_attention),
}
"""Each plain program's attention under a form beside the default scale and a causal mask, the arguments
Dew's own attention takes for it, its fill the program's (None where no call of Dew's computes it: a scale
other than the default is another constant, which the compile point keeps as written), and the program
whose run on float64 operands is its truth: itself where it is written at its operands' width, jax.nn's
written out (`jax_nn_written`) where it is jax.nn's. The masks leave rows partly masked (a window,
causality, -inf's mask) and wholly masked (jax.nn's mask, uniform rows under its finite fill); a row -inf
masks wholly is NaN, as written, which check 1 does not compare."""


@pytest.mark.parametrize("form", list(FORMS))
def test_a_plain_programs_attention_under_each_form_runs_the_kernel_dews_own_call_takes(form):
    """A plain program's attention under a mask (filled as jax.nn fills it, or with -inf) or a bias it draws
    from its inputs, a causal window or a softcap, compiled through `dew.jit` on an H100, runs the kernel
    Dew's own attention call under those arguments routes to (none where that is XLA's), and under a scale
    no call of Dew's takes runs as written, XLA's; on this host, under the template, its output and
    gradients on par with the program as written (check 1)."""
    from attention_oracle import attention_on_par, kept
    from contract_support import normal_drawn

    import dew
    from dew.algebra.recognize import KEPT
    from dew.nn.attention import scaled_dot_product_attention

    attend, arguments, exact = FORMS[form]
    shapes = (jax.ShapeDtypeStruct((2, 128, 4, 64), jnp.bfloat16),) * 3

    def own(q, k, v):
        return scaled_dot_product_attention(q, k, v, scale_logits=True, **arguments(q, k, v))

    with generation.assumed("sm90"):
        assert not _attention_routes(lambda: jax.jit(attend).trace(*shapes))
        expected = [] if arguments is None else [
            kernel for kernel in _attention_routes(lambda: jax.jit(own).trace(*shapes)) if kernel not in KEPT]
        assert set(_attention_routes(lambda: dew.jit(attend).trace(*shapes))) == set(expected)
    drawn = (2, 64, 4, 32)
    draw = normal_drawn(41, *(drawn,) * 4, dtype=jnp.bfloat16)
    # The draw's rows by its declared shape, no draw computed: a GPU tier part's prepare computes nothing on
    # its compile-only device (bible §8).
    queries = keys = drawn[1]
    # Size the structured masks' singleton rows by the terms the fast backward subtracts (dP and D),
    # not by the written softmax's already-cancelled gradient. The causal window's finite fill contributes
    # no weight on these draws.
    if form == "a softcap":
        oracle = {"keep": kept(queries, keys, causal=True), "softcap": SOFTCAP}
    elif form == "a causal window":
        oracle = {"keep": kept(queries, keys, causal=True, sliding_window=WINDOW)}
    else:
        oracle = {"truth": lambda *x: exact(*(a.astype(jnp.float64) for a in x))}
    with generation.chosen(attention="template"):
        attention_on_par(dew.jit(attend), attend, draw, f"dew.jit {form}", **oracle)

def written_out(*, fill=-jnp.inf, logits_cast=None, initial=-jnp.inf, other_rows=False, sum_cast=None,
                root=None):
    """Causal softmax attention as a program writes it out (its products as einsums, a causal where, the
    safe softmax), each keyword a way it computes something other than attention: a finite fill keeping the
    future keys' weight, logits truncated or narrowed through `logits_cast`, a maximum's initial value above
    the logits, each row divided by another row's sum (heads and rows swapped in its broadcast), the sum
    narrowed through `sum_cast` (to float16, where 65536 keys' sum overflows), the queries divided by a
    `root` near sqrt(D) but not it (Flax's division, its divisor another scale). Under a `root` it is
    Flax's attention as Flax writes it, its masked logits its dtype's least value before they widen."""
    def attend(q, k, v):
        if root is not None:
            # Flax's (`dot_product_attention_weights`): the queries divided before the product, no scale after
            # it, the logits in the operands' dtype masked there with its least value, then widened.
            s = jnp.einsum("bthd,bshd->bhts", q / jnp.float32(root).astype(q.dtype), k)
            s = jnp.where(jnp.tril(jnp.ones(s.shape[-2:], bool)), s, jnp.finfo(s.dtype).min)
            s = s.astype(_wide(s))
        else:
            s = jnp.einsum("bthd,bshd->bhts", q, k, preferred_element_type=_wide(q)) * q.shape[-1] ** -0.5
            if logits_cast is not None:
                s = s.astype(logits_cast).astype(_wide(q))
            s = jnp.where(jnp.tril(jnp.ones(s.shape[-2:], bool)), s, fill)
        e = jnp.exp(s - jax.lax.stop_gradient(jnp.max(s, -1, keepdims=True, initial=initial)))
        total = jnp.sum(e, -1)
        if sum_cast is not None:
            total = total.astype(sum_cast).astype(total.dtype)
        if other_rows:
            total = jax.lax.broadcast_in_dim(total, (*e.shape[:-1], 1), (0, 2, 1))
        else:
            total = total[..., None]
        return jnp.einsum("bhts,bshd->bthd", (e / total).astype(v.dtype), v)
    return attend


def test_a_programs_own_choice_by_the_platform_stays_as_written():
    """A plain program's attention whose weights pass through a choice by the platform it lowers for, which
    doubles them on CUDA alone, is no call of Dew's on an H100: its choice computes another function there,
    so the compile point keeps the program as written, XLA's, where the same attention without it runs Dew's
    kernel."""
    import dew

    def attend(doubled):
        def run(q, k, v):
            s = jnp.einsum("bthd,bshd->bhts", q, k, preferred_element_type=_wide(q)) * q.shape[-1] ** -0.5
            s = jnp.where(jnp.tril(jnp.ones(s.shape[-2:], bool)), s, -jnp.inf)
            weights = jax.nn.softmax(s, axis=-1)
            if doubled:
                weights = jax.lax.platform_dependent(weights, cuda=lambda w: 2 * w, default=lambda w: w)
            return jnp.einsum("bhts,bshd->bthd", weights.astype(v.dtype), v)
        return run

    shapes = (jax.ShapeDtypeStruct((1, 16, 16, 32), jnp.bfloat16),) * 3
    with generation.assumed("sm90"):
        assert _attention_routes(lambda: dew.jit(attend(doubled=False)).trace(*shapes))
        assert not _attention_routes(lambda: dew.jit(attend(doubled=True)).trace(*shapes))


@pytest.mark.mesh(devices=2)
def test_a_plain_programs_masked_attention_compiles_under_a_sequence_axis():
    """A plain program's causal attention, its masked logits filled, compiled through `dew.jit` on a mesh
    that splits the sequence, where Dew's call takes no fill (its exchanges weigh a masked key nothing):
    no candidate the call refuses is tried, and the program computes as written."""
    import dew
    from dew.training import MeshSpec

    attend = written_out()
    mesh = MeshSpec(sequence=2).build(jax.devices()[:2])
    q, k, v = (jax.random.normal(key, (1, 16, 4, 32), jnp.float32)
               for key in jax.random.split(jax.random.key(7), 3))
    with jax.set_mesh(mesh):
        ours, written = dew.jit(attend)(q, k, v), jax.jit(attend)(q, k, v)
    np.testing.assert_array_equal(np.asarray(ours), np.asarray(written))


@pytest.mark.parametrize(("variant", "options"), [("attention", {}), ("finite fill", {"fill": 0.0}),
                                                  ("a fill of NaN", {"fill": jnp.nan}),
                                                  ("truncated logits", {"logits_cast": jnp.int32}),
                                                  ("initial above", {"initial": 1e30}),
                                                  ("another row's sum", {"other_rows": True}),
                                                  ("logits narrowed", {"logits_cast": jnp.float16}),
                                                  ("sum narrowed", {"sum_cast": jnp.float16}),
                                                  ("attention by a root", {"root": 32 ** 0.5}),
                                                  ("attention by a root", {"root": 32 ** 0.5,
                                                                           "dtype": jnp.float32}),
                                                  ("a root near sqrt(D)", {"root": 32 ** 0.5 * 1.0005,
                                                                           "dtype": jnp.float32})])
def test_only_attention_written_out_is_rewritten_as_dews(variant, options):
    """Causal softmax attention written out runs Dew's kernel through `dew.jit` on an H100, scaled after its
    product or its queries divided by sqrt(D) before it (Flax's), its masked logits filled with -inf or a
    finite constant (Dew's call's `mask_fill`, the future keys weighed as that fill weighs them); written out
    with any step computing otherwise (a NaN fill, which no call of Dew's takes, truncated or narrowed logits,
    an initial maximum above the logits, a row normalized by another's sum, a sum narrowed past its range, a
    divisor near sqrt(D) but not it, in float32, where bfloat16 holds no value that near), it stays as
    written, XLA's."""
    import dew

    options = dict(options)
    shapes = (jax.ShapeDtypeStruct((1, 16, 16, 32), options.pop("dtype", jnp.bfloat16)),) * 3
    with generation.assumed("sm90"):
        routes = _attention_routes(lambda: dew.jit(written_out(**options)).trace(*shapes))
    assert bool(routes) == (variant in ("attention", "finite fill", "attention by a root")), routes


@pytest.mark.parametrize("order", [("xla", "template"), ("template", "xla")],
                         ids=["xla first", "template first"])
def test_a_bodys_attention_takes_each_scopes_kernel_choice(order):
    """Written-out attention inside a scanned body of a program held as a jaxpr, which every trace of it
    binds again as the same body, traced through `dew.jit` under one kernel choice and then another, in
    either order: each trace takes its own scope's choice, the template's kernel under
    `attention="template"` and none under `attention="xla"`, not the body the earlier scope rewrote."""
    from jax.extend.core import jaxpr_as_fun

    import dew

    attend = written_out()

    def layers(q, k, v):
        def one(carried, _):
            return attend(carried, k, v), None
        return jax.lax.scan(one, q, None, length=2)[0]

    shapes = (jax.ShapeDtypeStruct((1, 16, 16, 32), jnp.bfloat16),) * 3
    held = jaxpr_as_fun(jax.make_jaxpr(layers)(*shapes))
    with generation.assumed("sm90"):
        for choice in order:
            with dew.kernels.use(attention=choice):
                routes = _attention_routes(lambda: dew.jit(held).trace(*shapes))
            assert bool(routes) == (choice == "template"), (choice, routes)


def flax_rms(x, w):
    """Flax's RMSNorm as a Flax model applies it."""
    from flax import linen as nn

    return nn.RMSNorm(epsilon=1e-6, dtype=x.dtype).apply({"params": {"scale": w}}, x)


def widened_rms(x, w):
    """An RMSNorm as a plain program writes it: float32 statistics, the weight before the cast."""
    widened = x.astype(_wide(x))
    inverse = jax.lax.rsqrt(jnp.mean(widened * widened, axis=-1, keepdims=True) + 1e-6)
    return (widened * inverse * w).astype(x.dtype)


def offset_rms(x, w):
    """Gemma's RMSNorm: the normalized row cast first, then times one more than its weight."""
    widened = x.astype(_wide(x))
    normed = widened * jax.lax.rsqrt(jnp.mean(jnp.square(widened), axis=-1, keepdims=True) + 1e-6)
    return normed.astype(x.dtype) * (1 + w.astype(x.dtype))


def gated_product_of(a, b):
    """A gated MLP's product as a plain program writes Dew's (torch's) math: the activation in float32,
    rounded once to the gate's dtype, then times the up projection."""
    return jax.nn.silu(a.astype(_wide(a))).astype(a.dtype) * b


def dews_gated_product(a, b):
    """A gated MLP's product written with Dew's own silu (`activations.silu`: its sigmoid Dew's one logistic,
    a declared derivative over a span stating it), as Dew's layers write theirs (TSSM's, Mamba-3's gate)."""
    from dew.nn.activations import silu

    return silu(a) * b


def _norms_rewritten(fn, *operands):
    """`fn` with its norms written as the compile point writes them on an H100 (`_norms_written`), its first
    output, made where it is traced at `operands`' shapes: a program of it is stated by `fn`
    (`truths_ahead.arm_role`), where the written jaxpr's function holds JAX's token, which no statement
    holds."""
    shapes = [jax.ShapeDtypeStruct(jnp.shape(x), jnp.result_type(x)) for x in operands]
    written, _ = _norms_written(fn, *shapes)
    return written(*operands)[0]


def _norms_written(fn, *shapes):
    """`fn` with each RMSNorm and SwiGLU the compile point takes on an H100 written as Dew's norm
    (`recognize.recognized`), decided under the H100 and written on this host, whose norms run their portable
    arithmetic: what the compile point's rewrite computes, run here."""
    from jax.extend.core import jaxpr_as_fun

    from dew.algebra import fusion, recognize

    closed = jax.make_jaxpr(fn)(*shapes)
    with generation.assumed("sm90"):
        written = recognize.recognized(closed, fusion.Index(closed), recognize.NORMS)
    return jaxpr_as_fun(closed.replace(jaxpr=fusion._spliced(closed.jaxpr, written))), len(written)


def residual_rms(x, branch, w):
    """A pre-norm block's seam as a plain program writes it: the stream `x + branch`, its RMSNorm, and the
    stream read on after it."""
    stream = x + branch
    return widened_rms(stream, w) + stream


def gated_rms(x, gate, w):
    """Qwen3-Next's gated RMSNorm as a plain program writes it: fp32 statistics, the weight, then the gate's
    SiLU, cast once."""
    widened = x.astype(_wide(x))
    inverse = jax.lax.rsqrt(jnp.mean(widened * widened, axis=-1, keepdims=True) + 1e-6)
    return (widened * inverse * w.astype(_wide(w)) * jax.nn.silu(gate.astype(_wide(gate)))).astype(x.dtype)


def packed_swiglu(gate_up):
    """A packed gate|up projection's SwiGLU as a plain program writes Dew's math (`gated_product_of`): its
    two halves sliced out."""
    gate = gate_up[..., 0, :]
    return jax.nn.silu(gate.astype(_wide(gate))).astype(gate.dtype) * gate_up[..., 1, :]


def _own_norm(*, offset: bool, after_cast: bool):
    from dew.nn.norms import rms_normalized

    return lambda x, w: rms_normalized(x, w, 1e-6, x.dtype, scale_offset=offset, scale_after_cast=after_cast,
                                       fp32_statistics=True)


def _own_residual(x, branch, w):
    from dew.nn.norms import stream_rms_normalized

    stream, normed = stream_rms_normalized(x, branch, w, 1e-6, x.dtype, scale_offset=False,
                                           scale_after_cast=False, fp32_statistics=True)
    return normed + stream


def _own_gated(x, gate, w):
    from dew.nn.norms import gated_rms_normalized

    return gated_rms_normalized(x, gate, w, 1e-6, x.dtype, "silu")


def _own_swiglu(gate, up):
    from dew.nn.activations import gated_product

    return gated_product("swiglu")(gate, up)


def _own_packed(gate_up):
    from dew.nn.activations import packed_gated_product

    return packed_gated_product("swiglu")(gate_up)


ROWS, WEIGHT, PACKED = (256, 1024), (1024,), (256, 2, 1024)
NORM_FORMS = {
    "flax RMSNorm": (flax_rms, _own_norm(offset=False, after_cast=False), (ROWS, WEIGHT)),
    "plain RMSNorm": (widened_rms, _own_norm(offset=False, after_cast=False), (ROWS, WEIGHT)),
    "Gemma RMSNorm": (offset_rms, _own_norm(offset=True, after_cast=True), (ROWS, WEIGHT)),
    "residual RMSNorm": (residual_rms, _own_residual, (ROWS, ROWS, WEIGHT)),
    "gated RMSNorm": (gated_rms, _own_gated, (ROWS, ROWS, WEIGHT)),
    "SwiGLU": (gated_product_of, _own_swiglu, (ROWS, ROWS)),
    "SwiGLU of Dew's silu": (dews_gated_product, _own_swiglu, (ROWS, ROWS)),
    "packed SwiGLU": (packed_swiglu, _own_packed, (PACKED,)),
}
"""Each plain program's fused norm, Dew's own call of it, and its operands' shapes."""


@pytest.mark.parametrize("form", list(NORM_FORMS))
def test_a_plain_programs_norms_run_the_kernel_dews_own_call_takes(form):
    """A plain program's RMSNorm (Flax's, a plain one, Gemma's offset weight after the cast, a residual
    stream's read on after it, Qwen3-Next's gated one) or SwiGLU (two projections, its sigmoid Dew's one
    logistic too, which train 1026's review found no anchor read, or one packed one's halves), compiled
    through `dew.jit` on an H100, runs the kernel Dew's own call routes it to (none where
    that is XLA's arithmetic), where the program as written runs XLA's; on this host, its output and
    gradients on par with the program as written (check 1)."""
    from contract_support import normal_drawn
    from reference_error import assert_as_exact, with_gradients

    import dew

    program, own, sizes = NORM_FORMS[form]
    shapes = [jax.ShapeDtypeStruct(size, jnp.bfloat16) for size in sizes]

    def fused(trace) -> list[str]:
        with generation.routes() as found:
            trace()
        return [route.kernel for route in found if route.choice in ("rms_norm", "swiglu")
                and route.kernel != "xla"]

    with generation.assumed("sm90"):
        assert not fused(lambda: jax.jit(program).trace(*shapes))
        expected = set(fused(lambda: jax.jit(own).trace(*shapes)))
        assert set(fused(lambda: dew.jit(program).trace(*shapes))) == expected
    output = jax.eval_shape(program, *shapes).shape
    draw = normal_drawn(43, *sizes, output, dtype=jnp.bfloat16)
    _, found = _norms_written(program, *shapes)
    assert found == 1
    assert_as_exact(jax.jit(with_gradients(functools.partial(_norms_rewritten, program))),
                    jax.jit(with_gradients(program)),
                    draw, f"dew.jit {form}", seeds=8, truth_fn=with_gradients(
                        lambda *x: program(*(a.astype(jnp.float64) for a in x))))


def checkpointed_swiglu(a, b):
    """A SwiGLU whose sigmoid of the fp32-widened gate alone is checkpointed (`jax.checkpoint`): what the
    derivative saves is the user's choice at that boundary."""
    widened = a.astype(_wide(a))
    return (widened * jax.checkpoint(jax.nn.sigmoid)(widened)).astype(a.dtype) * b


def _rows_inverse(x):
    """Each row's inverse root mean square, fp32 `[rows]`, the feature axis reduced."""
    widened = x.astype(_wide(x))
    return jax.lax.rsqrt(jnp.sum(widened * widened, axis=-1) / x.shape[-1] + 1e-6)


def _named_silu(x):
    return jnp.tanh(x)


_named_silu.__name__ = "silu"


NEAR_NORMS = {
    "statistics in bfloat16, the widening dropped": lambda x, w: (
        x * jax.lax.rsqrt(jnp.mean(x * x, axis=-1, keepdims=True) + 1e-6) * w.astype(x.dtype)),
    "the weight applied before the statistic": lambda x, w: (
        _rows_inverse(x * w.astype(x.dtype))[:, None] * x.astype(_wide(x))).astype(x.dtype),
    "a SwiGLU whose sigmoid rounds to bfloat16": lambda x, w: jax.nn.silu(x) * (x * w.astype(x.dtype)),
    "columns scaled by the rows' statistics":
        lambda x, w: (x.astype(_wide(x)) * _rows_inverse(x)[None, :]).astype(x.dtype),
    "a weight along the rows":
        lambda x, w: (x.astype(_wide(x)) * _rows_inverse(x)[:, None] * w[:, None]).astype(x.dtype),
    "a jit named silu that is tanh": lambda x, w: jax.jit(_named_silu)(x) * w.astype(x.dtype),
    "a zero mean multiplier": lambda x, w: (x.astype(_wide(x)) * jax.lax.rsqrt(
        jnp.sum(jnp.square(x.astype(_wide(x))), axis=-1, keepdims=True) * 0.0 + 1e-6)).astype(x.dtype),
    "its own sum as the weight": lambda x, w: (
        x.astype(_wide(x)) * _rows_inverse(x)[:, None]
        * jnp.sum(jnp.square(x.astype(_wide(x))), axis=-1)[None, :]).astype(x.dtype),
    "a weight rounded to bf16 before it weighs": lambda x, w: (
        x.astype(_wide(x)) * _rows_inverse(x)[:, None]
        * (w.astype(_wide(w)) * 1.001).astype(jnp.bfloat16).astype(_wide(w))).astype(x.dtype),
}
"""Programs a norm's recogniser reads almost as an RMSNorm or SwiGLU and computes otherwise (a square row,
so every axis has the feature axis's extent)."""


@pytest.mark.parametrize("width", [256, 256.001])
def test_a_norm_is_its_mean_over_the_rows_width_alone(width):
    """A row's sum of squares over its width is its mean, which the compile point writes as Dew's norm; over
    a divisor near the width, closer than any output of the row's dtype shows, it is no norm, and the
    compile point writes none."""
    def program(x, w):
        widened = x.astype(_wide(x))
        inverse = jax.lax.rsqrt(jnp.sum(jnp.square(widened), axis=-1, keepdims=True) / width + 1e-6)
        return (widened * inverse * w.astype(_wide(w))).astype(x.dtype)

    _, found = _norms_written(program, jax.ShapeDtypeStruct((256, 256), jnp.bfloat16),
                              jax.ShapeDtypeStruct((256,), jnp.bfloat16))
    assert found == (width == 256)


def _two_results(x):
    """Two rows of one shape from one nested program: one equation making both."""
    return jax.jit(lambda x: (x * 2, x * 3))(x)


def _norm_of(rows, w):
    widened = rows.astype(_wide(rows))
    return (widened * jax.lax.rsqrt(jnp.mean(widened * widened, axis=-1, keepdims=True) + 1e-6)
            * w.astype(_wide(w))).astype(rows.dtype)


def _first_normed(x, w):
    a, b = _two_results(x)
    return _norm_of(a, w), a, b


def _second_normed(x, w):
    a, b = _two_results(x)
    return _norm_of(b, w), a, b


def _divided_by(divisor):
    """A norm-shaped program whose statistic divides by `divisor`, a constant array it closes over: the host's
    (NumPy's), which a statement of the program states by its values alone, where JAX's is stated by its
    device too, the prepare's CPU and the run's GPU two programs (rows 10008, 10009: the run found no
    reference its prepare compiled)."""
    held = np.asarray([divisor], np.float32)

    def program(x, w):
        widened = x.astype(_wide(x))
        mean = jnp.sum(widened * widened, axis=-1, keepdims=True) / held
        return (widened * jax.lax.rsqrt(mean + 1e-6) * w.astype(_wide(w))).astype(x.dtype)
    return program


def _permuted_statistic(x, w):
    """A row normalized by the statistic of its elements read in another order."""
    swapped = jax.lax.reshape(x, x.shape, dimensions=(1, 0))
    widened, other = x.astype(_wide(x)), swapped.astype(_wide(swapped))
    inverse = jax.lax.rsqrt(jnp.mean(other * other, axis=-1, keepdims=True) + 1e-6)
    return (widened * inverse * w.astype(_wide(w))).astype(x.dtype)


def _stream_read_early(x, branch, w):
    """A residual stream another value reads before its norm does."""
    stream = x + branch
    early = jnp.sin(stream)
    return early, _norm_of(stream, w)


WRITTEN_AS_PROGRAMS = {
    "the first of two results normed": (_first_normed, 1),
    "the second of two results normed": (_second_normed, 1),
    "a statistic over the width": (_divided_by(256.0), 1),
    "a statistic over twice the width": (_divided_by(512.0), 0),
    "a statistic of the elements in another order": (_permuted_statistic, 0),
}
"""Programs the compile point writes, each with the norms it writes in it: the same structure apart only
in which result is normed or in a constant it closes over, each weighed for its own (row 4783's review)."""


@pytest.mark.parametrize("name", list(WRITTEN_AS_PROGRAMS))
def test_the_compile_point_writes_a_program_by_its_own_values(name):
    """Each program (`WRITTEN_AS_PROGRAMS`), written after the others in either order, writes the norms it
    holds and computes what it does as written, on par from float64 (check 1: the norm's call, a GPU's kernel
    summing in its own order, and the program as written are two programs XLA promises no common bits): a
    region is the call's only where the call computes the region's own arithmetic, a statistic over twice the
    width or of permuted elements written as the norm off by its whole factor."""
    from contract_support import normal_draw
    from reference_error import as_given, assert_as_exact

    from dew.parity import FEWEST_SEEDS

    shapes = (jax.ShapeDtypeStruct((256, 256), jnp.bfloat16), jax.ShapeDtypeStruct((256,), jnp.bfloat16))
    names = list(WRITTEN_AS_PROGRAMS)
    for other in (*names[names.index(name) + 1:], *names[:names.index(name)]):
        _norms_written(WRITTEN_AS_PROGRAMS[other][0], *shapes)
    program, norms = WRITTEN_AS_PROGRAMS[name]
    written, found = _norms_written(program, *shapes)
    assert found == norms
    assert_as_exact(as_given(lambda x, w: tuple(written(x, w))),
                    lambda x, w: tuple(jax.tree.leaves(program(x, w))),
                    lambda seed: normal_draw(seed, (256, 256), (256,), dtype=jnp.bfloat16),
                    f"the compile point's writing of {name}", seeds=FEWEST_SEEDS)


def test_a_stream_read_before_its_norm_computes_as_written():
    """A residual stream another value reads before its norm, compiled through the rewrite, computes as
    written, on par from float64 (check 1, two programs: the norm's call and the stream's sum as written):
    the stream's sum stays where its first reader needs it."""
    from contract_support import normal_draw
    from reference_error import as_given, assert_as_exact

    from dew.parity import FEWEST_SEEDS

    rows = jax.ShapeDtypeStruct((256, 256), jnp.bfloat16)
    shapes = (rows, rows, jax.ShapeDtypeStruct((256,), jnp.bfloat16))
    written, _ = _norms_written(_stream_read_early, *shapes)
    assert_as_exact(as_given(lambda x, branch, w: tuple(written(x, branch, w))), _stream_read_early,
                    lambda seed: normal_draw(seed, (256, 256), (256, 256), (256,), dtype=jnp.bfloat16),
                    "a stream read before its norm, written", seeds=FEWEST_SEEDS)


def test_a_norm_of_a_stopped_row_passes_no_gradient_to_it():
    """A norm of a row whose gradient the program stops, rewritten, passes the row no gradient, as written."""
    def program(x, w):
        return _norm_of(jax.lax.stop_gradient(x), w).astype(jnp.float32).sum()

    x, w = host_normal(4785, (256, 256), (256,), dtype=np.float32)
    grad = jax.grad(functools.partial(_norms_rewritten, program))
    assert not np.any(np.asarray(ran(grad, "a stopped row's gradient, its norm written", x, w)))


def test_a_gate_less_a_negative_zero_keeps_its_sign_as_written():
    """A SwiGLU whose gate is the program's `g - (-0.0)`, at a gate of -0.0, computes as written bit for
    bit: subtracting a negative zero makes a negative zero positive, which no rewrite drops."""
    def program(g, u):
        gate = g - jnp.asarray(-0.0, g.dtype)
        return gated_product_of(gate, u)

    g = np.full((256, 256), -0.0, jnp.bfloat16)
    u = np.ones((256, 256), jnp.bfloat16)
    np.testing.assert_array_equal(
        np.signbit(np.asarray(ran(functools.partial(_norms_rewritten, program), "the gate, its norm written",
                                  g, u))),
        np.signbit(np.asarray(ran(program, "the gate as written", g, u))))


@jax.custom_jvp
def _doubled_identity(x):
    return x


@_doubled_identity.defjvp
def _doubled_identity_tangent(primals, tangents):
    return primals[0], 2 * tangents[0]


def test_a_norm_of_a_row_under_its_own_derivative_rule_keeps_the_rule():
    """A norm of a row a nested program's own derivative rule makes (a jitted custom JVP doubling its
    tangent), rewritten, keeps that rule: the row's gradient twice the norm's own, as written."""
    def program(x, w):
        return _norm_of(jax.jit(_doubled_identity)(x), w).astype(jnp.float32)[0, 0]

    x, w = host_normal(4786, (256, 256), (256,), dtype=jnp.bfloat16)
    np.testing.assert_array_equal(
        np.asarray(ran(jax.grad(functools.partial(_norms_rewritten, program)),
                       "the row's gradient, its norm written", x, w)),
        np.asarray(ran(jax.grad(program), "the row's gradient as written", x, w)))


def test_a_norm_of_a_row_an_effect_reads_keeps_the_effect():
    """A norm of a row a nested program passes through a host callback, rewritten, keeps the callback: the
    program's effects are the program's as written, the callback's, which XLA runs as written. Nothing
    computes: a program holding a callback of this process's compiles for no compile-only GPU (a GPU tier
    part's prepare) and runs from no executable another process compiled."""
    from jax.experimental import io_callback

    def recorded(x):
        io_callback(lambda _: None, None, x)
        return x

    def program(x, w):
        return _norm_of(jax.jit(recorded)(x), w)

    x, w = np.ones((256, 256), jnp.bfloat16), np.ones((256,), jnp.bfloat16)
    written = jax.make_jaxpr(program)(x, w).effects
    assert written
    assert jax.make_jaxpr(functools.partial(_norms_rewritten, program))(x, w).effects == written


@pytest.mark.parametrize("program", list(NEAR_NORMS))
def test_a_plain_program_near_a_norm_computes_as_written(program):
    """A program whose arithmetic resembles an RMSNorm or SwiGLU without being one (the statistics or the
    weight laid along another axis, a nested program only named silu, a mean scaled by zero, the region's own
    sum read as its weight, a weight rounded by a cast the norm would drop, its statistics left in bfloat16,
    the weight applied to the row before its statistic, a SwiGLU rounding its sigmoid and each product to
    bfloat16 where Dew's rounds once after a float32 activation), compiled through `dew.jit` on an H100,
    computes what it does as written: its output and gradients on par with the program's (check 1),
    whatever of it the compile point writes as Dew's norm (an unweighted RMSNorm inside it, its product with
    another axis's weight left as written)."""
    from contract_support import normal_drawn
    from reference_error import assert_as_exact, with_gradients


    fn = NEAR_NORMS[program]
    draw = normal_drawn(47, (256, 256), (256,), (256, 256), dtype=jnp.bfloat16)
    assert_as_exact(jax.jit(with_gradients(functools.partial(_norms_rewritten, fn))),
                    jax.jit(with_gradients(fn)), draw,
                    f"dew.jit {program}", seeds=8,
                    truth_fn=with_gradients(lambda *x: fn(*(a.astype(jnp.float64) for a in x))))


@pytest.mark.parametrize("enclosing", ["jit", "grad"])
def test_a_norm_whose_epsilon_an_enclosing_trace_closes_over_compiles_through_dew_jit(enclosing):
    """A plain RMSNorm in `dew.jit` whose epsilon an enclosing `jax.jit` or `jax.grad` traces (a value with no
    number until it runs, which no fused call takes as its own) compiles on an H100, and on this host
    computes what the program as written computes, bitwise."""
    import dew

    def norm(x, w, epsilon):
        widened = x.astype(jnp.float32)
        inverse = jax.lax.rsqrt(jnp.mean(widened * widened, axis=-1, keepdims=True) + epsilon)
        return (widened * inverse * w).astype(x.dtype)

    x = np.random.default_rng(5).standard_normal(ROWS).astype(jnp.bfloat16)
    w = np.random.default_rng(6).standard_normal(WEIGHT).astype(jnp.bfloat16)

    def enclosed(compiled):
        """The enclosing program, its norm compiled by `compiled` over the epsilon it traces."""
        def normed(epsilon):
            return compiled(lambda x, w: norm(x, w, epsilon))(x, w)

        if enclosing == "jit":
            return jax.jit(normed)
        return jax.jit(jax.grad(lambda epsilon: normed(epsilon).astype(jnp.float32).sum()))

    through, written = enclosed(dew.jit), enclosed(jax.jit)
    epsilon = np.float32(1e-6)
    with generation.assumed("sm90"):
        through.trace(epsilon)
    np.testing.assert_array_equal(np.asarray(ran(through, "the norm through dew.jit", epsilon)),
                                  np.asarray(ran(written, "the norm as written", epsilon)))


def _held_logistic(x):
    """Dew's one logistic's value (`precision.logistic_value`), rounded once to `x`'s dtype."""
    from dew.nn.precision import logistic_value

    given = jnp.asarray(x)
    return logistic_value(given).astype(given.dtype)


def _no_tangent(primals, tangents):
    """A tangent of the logistic's own: none."""
    value = _held_logistic(primals[0])
    return value, jnp.zeros_like(value)


def held_logistics_swiglu(a, b):
    """A SwiGLU whose sigmoid states Dew's one logistic's operation (`Protected.computes`) but is declared
    with a rule of its own, no tangent: one a fused call written in its place would not keep (train 1026's
    F6 review: the primal alone read as Dew's logistic would have fused it)."""
    from dew.nn.derivatives import Protected, declared_tangent
    from dew.nn.precision import logistic_value

    held = declared_tangent(Protected(computes=logistic_value), _held_logistic, _no_tangent)
    work = a.astype(jnp.float32)
    return (work * held(work)).astype(a.dtype) * b


def batched_by_its_rule(a, b):
    """A SwiGLU whose SiLU carries a custom batching rule of its own (`jax.custom_batching.custom_vmap`),
    one a fused call written in its place would not keep."""
    from jax.custom_batching import custom_vmap

    silu = custom_vmap(jax.nn.silu)
    silu.def_vmap(lambda size, batched, x: (jax.nn.silu(x) + 1, batched[0]))
    return silu(a.astype(jnp.float32)).astype(a.dtype) * b


def swiglu_traced(program):
    """`program` over two bf16 operands traced through `dew.jit` on an assumed H100: its trace, and whether
    a SwiGLU in it runs Dew's fused kernel (`generation.routes`)."""
    import dew

    shapes = [jax.ShapeDtypeStruct(ROWS, jnp.bfloat16)] * 2
    with generation.assumed("sm90"), generation.routes() as found:
        made = dew.jit(program).trace(*shapes)
    return made, any(route.choice == "swiglu" and route.kernel != "xla" for route in found)


@pytest.mark.parametrize("barred", [checkpointed_swiglu, batched_by_its_rule, held_logistics_swiglu],
                         ids=["a checkpointed sigmoid", "an activation's own batching rule",
                              "a logistic declared with a rule of its own"])
def test_a_gated_product_across_a_barrier_is_never_fused(barred):
    """A SwiGLU across a barrier runs as written through `dew.jit` on an H100, where the same product
    without it runs Dew's fused kernel: a checkpoint (`checkpointed_swiglu`), whose boundary decides what
    its derivative saves (§6.2), its program keeping the checkpoint; an activation's own batching rule
    (`batched_by_its_rule`), and a logistic stating Dew's operation declared with a derivative of its own
    (`held_logistics_swiglu`), which a fused call written in its place would not keep."""
    from jax.extend.core import primitives

    from dew.nn.programs import equations

    _, fused = swiglu_traced(gated_product_of)
    made, crossed = swiglu_traced(barred)
    assert fused and not crossed
    if barred is checkpointed_swiglu:
        assert any(equation.primitive is primitives.remat_p for equation in equations(made.jaxpr.jaxpr))


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
