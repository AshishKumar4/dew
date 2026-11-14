"""A launch's work counted from its kernel's emission (`templates.emission`, row 5895) against the cost it
states: the tensor work and exponentials each attention kernel issues, a step of its walk counted from its
body and its steps the one statement of them (`attention_mosaic_program.walked_steps`,
`cost.Stated.steps`)."""
import math

import jax
import jax.numpy as jnp
import pytest
from jax import lax
from jax._src import core as jax_core
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu

from dew.nn import derivatives
from dew.nn.kernels import generation
from dew.nn.kernels.templates import attention as template, attention_mosaic, emission
from dew.nn.kernels.templates.cost import Stated, Work


def _tensor(work: Work) -> float:
    return work.flops + work.tf32_flops + work.fp8_flops


def _exponentials(work: Work) -> float:
    return work.exponentials


def _kernels(fn, args) -> list[tuple]:
    """Each Mosaic GPU kernel `fn` launches (its `mpmd_map`), with its mesh, body and stated cost."""
    found = []

    def find(jaxpr):
        for equation in jaxpr.eqns:
            if equation.primitive.name == "mpmd_map":
                (mesh,), (body,) = equation.params["meshes"], equation.params["jaxprs"]
                found.append((mesh, body, equation.params.get("cost_estimate")))
            for param in equation.params.values():
                for inner in (param if isinstance(param, (tuple, list)) else (param,)):
                    if isinstance(inner, jax_core.Jaxpr):
                        find(inner)

    find(jax.make_jaxpr(fn)(*args).jaxpr)
    return found


def _counted(mesh, body, unit, steps: int) -> float:
    """A kernel's work in `unit` by emission: a consumer's walk step times the launch's stated steps, and
    every warpgroup's work outside its walks times the programs."""
    counted = emission.emitted(body, mesh.num_threads, mesh.thread_name)
    stepping = [role.step(unit) for role in counted.roles if role.walks]
    return max(stepping) * steps + unit(counted.fixed) * math.prod(mesh.grid)


GATED = {"sm90": ("attention_peaks", "attention_training", "attention_dkv_fused"),
         "sm100": ("attention_dq", "attention_dkv")}
"""The kernels whose count meets their stated tensor work, by generation; sm100's chunks forwards wait for
their walk's first and last step peeled from its loop (row 5895)."""

EXPONENTIALS_GATED = ("attention_dkv_fused", "attention_dq", "attention_dkv")
"""The kernels whose count meets their stated exponentials; the forwards' formula states the peaks pass's
none and the training pass's per-row rescales none (row 5895's drift)."""


@pytest.mark.parametrize("causal", [True, False], ids=["causal", "full"])
@pytest.mark.parametrize("generation_name", sorted(GATED))
def test_attentions_issued_work_is_its_stated_cost(generation_name, causal):
    """Traced for a compile-only GPU: each gated attention kernel's tensor work counted from its emission
    (a consumer's step times its stated walk steps) is exactly the FLOPs it states to XLA, and its
    exponentials too where gated."""
    tokens, heads, kv_heads, width = 1024, 4, 2, 128
    spans = template.spans(tokens, tokens, 1, causal=causal)
    query = jax.ShapeDtypeStruct((1, tokens, heads, width), jnp.bfloat16)
    key = jax.ShapeDtypeStruct((1, tokens, kv_heads, width), jnp.bfloat16)

    def both(query, key, value):
        out, *residuals = attention_mosaic.forward(query, key, value, spans, None,
                                                   residuals=template.Kept.SPLIT)
        return attention_mosaic.backward(query, key, value, spans, None, out, residuals=tuple(residuals))

    with generation.assumed(generation_name, portable=True):
        kernels = _kernels(both, (query, key, key))
    gated = [(mesh, body, cost) for mesh, body, cost in kernels
             if mesh.kernel_name.split("__")[0] in GATED[generation_name]]
    assert gated, [mesh.kernel_name for mesh, _, _ in kernels]
    for mesh, body, cost in gated:
        assert isinstance(cost, Stated), (mesh.kernel_name, cost)
        assert _counted(mesh, body, _tensor, cost.steps) == cost.flops, mesh.kernel_name
        if mesh.kernel_name.split("__")[0] in EXPONENTIALS_GATED:
            assert _counted(mesh, body, _exponentials, cost.steps) == cost.transcendentals, mesh.kernel_name


_STEP = derivatives.marked(derivatives.WalkStep(), lambda index: index)
"""A walk step's mark, as `mosaic.stepped` makes it."""

_ROW = jax.ShapeDtypeStruct((128,), jnp.float32)


def _operations(work: Work) -> float:
    return work.operations


def _emitted(fn, *args) -> emission.Emitted:
    """`fn` counted as each of two warpgroups along `wg` runs it."""
    return emission.emitted(jax.make_jaxpr(fn, axis_env=[("wg", 2)])(*args), 2, "wg")


def test_a_step_counts_each_trip_of_a_loop_inside_it():
    """A walk step holding a loop of four trips counts its body four times."""
    def walk(row):
        def step(row, index):
            _STEP(index)
            return lax.scan(lambda row, _: (row + 1, None), row, length=4)[0], None

        return lax.scan(step, row, jnp.arange(8))[0]

    assert [role.step(_operations) for role in _emitted(walk, _ROW).roles] == [4 * 128] * 2


def _one(row, index):
    _STEP(index)
    return row + 1


def _two(row, index):
    return _one(_one(row, index), index)


def test_a_step_divides_by_the_steps_its_warpgroups_branch_takes():
    """A trip whose warpgroup's branch takes one step for one warpgroup and two for the other counts each
    one step's work: a step's work over the marks the warpgroup runs, not the most any branch holds."""
    def walk(row):
        def step(row, index):
            return lax.cond(lax.axis_index("wg") == 0, _one, _two, row, index), None

        return lax.scan(step, row, jnp.arange(8))[0]

    assert [role.step(_operations) for role in _emitted(walk, _ROW).roles] == [128, 128]


def test_a_scoped_bodys_results_resolve_the_roles_after_it():
    """A warpgroup's index a scoped body returns (`pl.run_scoped`'s) resolves the role's branch after it."""
    def walk(row):
        def step(row, index):
            role = pl.run_scoped(lambda scratch: lax.axis_index("wg"), plgpu.SMEM((8,), jnp.float32))
            return lax.cond(role == 0, _one, _two, row, index), None

        return lax.scan(step, row, jnp.arange(8))[0]

    assert [role.step(_operations) for role in _emitted(walk, _ROW).roles] == [128, 128]


def test_an_unknown_branch_counts_the_most_of_each_unit_any_branch_issues():
    """A conditional its data decides counts its conversion-only branch's conversions, though its other
    branch issues nothing."""
    def converted(row, index):
        return lax.switch(index, [lambda row: row, lambda row: row.astype(jnp.bfloat16).astype(jnp.float32)],
                          row)

    index = jax.ShapeDtypeStruct((), jnp.int32)
    assert _emitted(converted, _ROW, index).roles[0].fixed.conversions == 2 * 128


def test_an_unknown_branch_between_walks_steps_the_costliest():
    """A conditional its data decides between a walk of two segments of one operation a row a step and a
    walk of three a step counts a step of three."""
    def walked(work):
        def step(row, index):
            _STEP(index)
            return work(row), None

        return lambda row: lax.scan(step, row, jnp.arange(4))[0]

    once, thrice = walked(lambda row: row + 1), walked(lambda row: row + 1 + 1 + 1)

    def either(row, index):
        return lax.switch(index, [lambda row: once(once(row)), thrice], row)

    index = jax.ShapeDtypeStruct((), jnp.int32)
    assert _emitted(either, _ROW, index).roles[0].step(_operations) == 3 * 128
