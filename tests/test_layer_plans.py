"""A whole decoder layer's fusion plan (`dew.nn.kernels.templates.layers`, row 9038): each body as the pass
weighs it, the layer flattened with each boundary a region crosses stated, and the checked-in plans as
planning makes them."""
import os
import subprocess
import sys

import jax
import jax.numpy as jnp
import pytest
from jax._src import core as jcore

from dew.nn.kernels import generation
from dew.nn.kernels.templates import layers


def _gated(x, w):
    """A product and an activation behind its own custom derivative (`jax.nn.relu`'s), as a layer's MLP
    writes them."""
    return jax.nn.relu(x @ w)


def _shapes():
    return (jax.ShapeDtypeStruct((256, 512), jnp.bfloat16), jax.ShapeDtypeStruct((512, 1024), jnp.bfloat16))


def test_a_flattened_layer_runs_its_calls_in_place_and_states_what_each_equation_was_within():
    """A program's calls run in their caller (`layers.flattened`): no custom derivative's call is left, and
    each equation its body held states it was within that boundary, outermost first (`Flat.within`: relu's
    derivative, then its own jit); a region of the product and the activation crosses both (`Flat.crossed`),
    the product alone none."""
    closed = jax.make_jaxpr(_gated)(*_shapes())
    flat = layers.flattened(closed.jaxpr)
    names = [equation.primitive.name for equation in flat.body.eqns]
    assert "custom_jvp_call" not in names and "dot_general" in names
    product = names.index("dot_general")
    inner = [place for place, frames in enumerate(flat.within) if frames]
    assert inner and all(flat.crossings[flat.within[place][0]] == layers.DERIVATIVE for place in inner)
    assert flat.crossed([product]) == ()
    assert flat.crossed([product, inner[-1]]) == (layers.DERIVATIVE, layers.CALL)



def test_a_helper_called_twice_is_flattened_to_two_of_its_own():
    """A helper traced once and called twice (one jaxpr, two call sites: a layer's q and k norms, a mask's
    conversion at each use) flattened to two copies, every value bound once (`layers.flattened`, through
    `programs.inlined`), so each copy's product has its own readers (row 9038's review: the second
    call's equations rebound the first's variables, and each product read both)."""
    @jax.jit
    def scaled(x, w):
        return jnp.tanh(x @ w)

    def twice(x, w):
        return scaled(x, w) + scaled(x * 2, w)

    flat = layers.flattened(jax.make_jaxpr(twice)(*_shapes()).jaxpr)
    bound = [var for equation in flat.body.eqns for var in equation.outvars if isinstance(var, jcore.Var)]
    assert len(bound) == len(set(bound))
    assert [equation.primitive.name for equation in flat.body.eqns].count("dot_general") == 2

@pytest.mark.parametrize("generation_name", ["sm90", "sm80"])
def test_a_plan_states_what_the_pass_takes_and_never_pays_a_region_across_a_boundary(generation_name):
    """A product and an activation behind a custom derivative (`plan`): weighed as the pass weighs each body
    (`today`), no variant crosses anything; flattened (`search`), each variant holding the activation states
    the derivative it crosses and never pays (`Segment.pays`), whatever its price, so the pass's choice
    across the layer is never one it could not take; every variant's product routed as its selector routes
    it (`Segment.kernel`), and a fused epilogue's price citing M-AL-7."""
    with generation.assumed(generation_name):
        planned = layers.plan(jax.make_jaxpr(_gated)(*_shapes()).jaxpr)
    assert planned.today and planned.search
    assert all(not variant.crossings for found in planned.today for variant in found.variants)
    crossing = [variant for found in planned.search for variant in found.variants if variant.crossings]
    assert crossing and all(not variant.pays for variant in crossing)
    assert all(variant.crossings[0] == layers.DERIVATIVE for variant in crossing)
    every = [variant for found in (*planned.today, *planned.search) for variant in found.variants]
    assert all(variant.kernel is not None for variant in every)
    assert all(("M-AL-7" in variant.cites) == ("epilogue" in variant.stages) for variant in every)



def _scaled_by_an_array(x, w, scale):
    """A projection times a scalar array: an epilogue the template refuses (`fused_gemm`'s
    'value_off_axes'), so the pass fuses the product alone."""
    return (x @ w) * scale


def test_todays_choice_is_the_region_the_pass_writes():
    """A projection times a scalar array on an H100 whose call names the template: the plan's choice of its
    body (`Weighed.passed`, the pass's own selection, `fusion.taken`) is the product alone, every variant
    holding the epilogue not taken (`fusion.decided`), and the pass writes that region: the multiply after
    the template's launch, outside it, as the program wrote it (`fusion.fused_program` through
    `algebra.rewritten`, its route the template's, `generation.routes`)."""
    from dew import algebra

    shapes = (jax.ShapeDtypeStruct((256, 512), jnp.bfloat16), jax.ShapeDtypeStruct((512, 256), jnp.bfloat16),
              jax.ShapeDtypeStruct((), jnp.bfloat16))
    with generation.assumed("sm90"), generation.chosen(grouped_matmul="mosaic"):
        (found,) = layers.plan(jax.make_jaxpr(_scaled_by_an_array)(*shapes).jaxpr).today
        with generation.routes() as routes:
            written = jax.jit(algebra.rewritten(_scaled_by_an_array, kernels=True)).trace(*shapes).jaxpr.jaxpr
    assert found.passed is not None and found.passed.stages == ("product",), found
    assert not any(variant.takes for variant in found.variants if "epilogue" in variant.stages)
    assert {route.kernel for route in routes if route.choice == "grouped_matmul"} == {"mosaic"}
    (last,) = (equation for equation in written.eqns if set(equation.outvars) & set(written.outvars))
    assert last.primitive is jax.lax.mul_p


def test_a_forced_epilogue_the_template_refuses_is_reported_with_its_stage_and_refusal():
    """A projection times a scalar array with its epilogue forced to the template's (`gemm_epilogue`
    'fused'), which refuses it before any route is chosen (`fused_gemm.Analysis.refused`): its variant
    holding the multiply reports the epilogue stage and the refusal, not a bare product with none."""
    shapes = (jax.ShapeDtypeStruct((256, 512), jnp.bfloat16), jax.ShapeDtypeStruct((512, 256), jnp.bfloat16),
              jax.ShapeDtypeStruct((), jnp.bfloat16))
    with generation.assumed("sm90"), generation.chosen(grouped_matmul="mosaic", gemm_epilogue="fused"):
        (found,) = layers.plan(jax.make_jaxpr(_scaled_by_an_array)(*shapes).jaxpr).today
    whole = found.variants[0]
    assert whole.stages == ("product", "epilogue") and whole.refused and not whole.takes, whole


def _ported(x, w):
    """A product and its activation in a platform choice's portable branch, the GPU's another launch."""
    def portable(x, w):
        return jnp.tanh(x @ w)

    return jax.lax.platform_dependent(x, w, cuda=lambda x, w: (x @ w) * 0, default=portable)


def test_a_candidate_inside_a_platform_choices_portable_branch_is_never_ranked():
    """A product and its activation wholly inside a platform choice's portable branch, flattened: each
    variant states the platform choice (`Flat.crossed`: a GPU's launch may replace the branch whole, however
    the region lies in it), so none pays and none is the plan's best (`Weighed.best`), its portable math no
    price of the launch it replaces."""
    with generation.assumed("sm90"):
        planned = layers.plan(jax.make_jaxpr(_ported)(*_shapes()).jaxpr)
    assert planned.search
    assert all(layers.PLATFORM in variant.crossings and not variant.pays
               for found in planned.search for variant in found.variants)
    assert all(found.best is None and found.passed is None for found in planned.search)

def test_a_product_whose_variants_save_alike_takes_its_most_staged_as_its_best():
    """`Weighed.best` among variants saving the same seconds is the first, the one holding the most stages
    (`Weighed.variants`' order): a tie never reads a later, smaller region as the better plan."""
    def variant(stages: tuple[str, ...], fused: float) -> layers.Segment:
        return layers.Segment(places=(0,), stages=stages, kernel="gemm", refusal=None, refused=None,
                              takes=True, cheaper=True, fused=fused, unfused=fused + 1.0)

    most, fewer = variant(("prologue", "epilogue"), 2.0), variant(("epilogue",), 3.0)
    assert layers.Weighed(body="step", product=0, variants=(most, fewer)).best == most


def test_the_checked_in_plans_are_the_plans_planning_makes():
    """`_layer_plans.json` holds what planning makes of each case, as its command checks it in a process of
    its own (`python -m dew.nn.kernels.templates.layers --check`, which regenerates them and fails naming
    the lines that differ): the plans a developer's run writes, no test suite's settings among them."""
    from process_support import REPO_ROOT

    # A developer's environment, none of the suite's (its matmul precision and XLA flags move the step's
    # equations, FS-9038-ENV3).
    developer = {name: os.environ[name] for name in ("PATH", "HOME") if name in os.environ}
    checked = subprocess.run([sys.executable, "-m", "dew.nn.kernels.templates.layers", "--check"],
                             capture_output=True, text=True, check=False,
                             env={**developer, "PYTHONPATH": str(REPO_ROOT / "src")})
    assert checked.returncode == 0, checked.stderr[-6000:]
