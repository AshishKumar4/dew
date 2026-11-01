"""Instrument the identities-only path and audit fusion input-memory accounting."""

import dataclasses
import functools
import json
import math

import jax
import jax.numpy as jnp
import pytest
from egglog import DagCostModel, EGraph, get_callable_args

from dew import algebra
from dew.algebra import cost, program, rules, terms
from dew.nn.kernels.generation import run_precision
from dew.nn.kernels.templates.cost import Work
from dew.nn.precision import ACCURATE


def separate(x, y):
    return jnp.sum(jnp.tanh(x)) + jnp.sum(jnp.tanh(y))


def joined(x, y):
    return jnp.sum(jnp.tanh(x) + jnp.tanh(y))


@pytest.mark.parametrize("skip", [False, True], ids=["original-red", "skip-identities-control-green"])
def test_accurate_only_identities_does_not_build_saturate_extract(monkeypatch, skip):
    """Count actual phases, not just selected rules or equality of the final jaxprs."""
    seen, reports = {}, []

    def instrument(owner, name):
        original = getattr(owner, name)

        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            seen[name] = seen.get(name, 0) + 1
            value = original(*args, **kwargs)
            if name == "_region":
                reports.append(dataclasses.asdict(value.report))
            return value
        monkeypatch.setattr(owner, name, wrapped)

    for owner, name in ((terms, "region"), (program, "graphed"), (program, "saturate"),
                        (program, "extracted"), (program, "_region"), (EGraph, "extract")):
        instrument(owner, name)
    if skip:
        original = program._windowed

        def only_novel(jaxpr, accurate, saturation):
            if all(admitted.name == "identities" for admitted in rules.admitted(accurate)):
                return jaxpr
            return original(jaxpr, accurate, saturation)
        monkeypatch.setattr(program, "_windowed", only_novel)
    shape = jax.ShapeDtypeStruct((8192,), jnp.float32)
    with run_precision(ACCURATE):
        admitted = [rule.name for rule in rules.admitted(True)]
        before = jax.make_jaxpr(separate)(shape, shape)
        after = jax.make_jaxpr(algebra.rewritten(separate))(shape, shape)
    print(json.dumps({"skip": skip, "admitted": admitted, "calls": seen, "reports": reports,
                      "unchanged": str(before) == str(after)}), flush=True)
    assert admitted == ["identities"]
    assert str(before) == str(after)
    if seen:
        pytest.fail(f"Identities-only tracing still did the runtime algebra phases: {seen}", pytrace=False)


def leaf_read_control(region, schedule):
    """Test-only cause removal for these two pure elementwise/reduction DAGs; not a general cost model."""
    def marginal(graph, node):
        work = cost.work(graph, node, region)
        if get_callable_args(node, terms.Term.reduce) is not None:
            work = dataclasses.replace(work, device_bytes=0)
        elif get_callable_args(node, terms.Term.leaf) is not None:
            work = work + Work(device_bytes=math.prod(region.shape(node)) * 4)
        return cost.Seconds(work, schedule)
    return DagCostModel(marginal_cost=marginal, identity=cost.Seconds(Work(), schedule))


@pytest.mark.parametrize("leaf_reads", [False, True], ids=["original-red", "leaf-reads-control-green"])
def test_two_inputs_are_read_in_both_reduction_forms(leaf_reads):
    """Print actual cost.work accumulation, its Fused price and CPU XLA's own optimized analysis."""
    shape = jax.ShapeDtypeStruct((8192,), jnp.float32)
    schedule = cost.fused(cost.target())
    rows = []
    for name, function in (("separate", separate), ("joined", joined)):
        region = terms.region(jax.make_jaxpr(function)(shape, shape))
        graph, bundle, _ = program.graphed(region)
        model = leaf_read_control(region, schedule) if leaf_reads else cost.model(region, schedule)
        _, price = graph.extract(bundle, include_cost=True, cost_model=model, extractor="greedy-dag")
        compiled = jax.jit(function).lower(shape, shape).compile()
        row = {"form": name, "work": dataclasses.asdict(price.work), "Fused_seconds": price.value,
               "XLA_CPU_cost_analysis": compiled.cost_analysis()}
        rows.append(row)
        print(json.dumps({"leaf_reads": leaf_reads, "target": cost.target().device_kind,
                          "Fused_rates": dataclasses.asdict(schedule), **row}), flush=True)
    expected = 2 * 8192 * 4
    observed = [row["work"]["device_bytes"] for row in rows]
    if observed != [expected, expected]:
        pytest.fail(f"Both forms read two inputs of {expected} bytes in all, but cost.work gives {observed}",
                    pytrace=False)


def test_default_pass_selects_joined_but_accurate_does_not():
    """Connect the cost observation to the consumer: the program actually accepts the cheaper joined form."""
    shape = jax.ShapeDtypeStruct((8192,), jnp.float32)
    rewritten = jax.make_jaxpr(algebra.rewritten(separate))(shape, shape)
    with run_precision(ACCURATE):
        accurate = jax.make_jaxpr(algebra.rewritten(separate))(shape, shape)
    counts = [sum(eqn.primitive.name == "reduce_sum" for eqn in closed.jaxpr.eqns)
              for closed in (rewritten, accurate)]
    print(json.dumps({"default_and_accurate_reductions": counts}), flush=True)
    assert counts == [1, 2]
