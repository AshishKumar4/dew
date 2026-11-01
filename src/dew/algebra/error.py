"""Error-aware extraction (docs/design/algebra.md §1.5), at dev time: a function's equal forms the e-graph
holds under every rule set, the stable ones among them (`rules.admitted(dev=True)`), proposed as candidates
(the form as written, the cheapest by time, the one the stable proposal prices cheapest), each candidate's
error measured by sampling against the function at float64 on its cases' draws, and the cheapest whose error
meets the precision policy's declared accuracy chosen. Error is a candidate's, never an e-class fact: the
forms of one class are equal in the reals and differ in error, which depends on every operand form a
candidate reads."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
from egglog import Vec

from dew.algebra import cost, program, rules, terms

PARITY = 1.5
"""The ratio check 1 holds a form's error to its reference's at (the accuracy gate's `PARITY`,
tests/check1_statistics.py): a candidate within it of its policy's reference is admitted."""


@dataclasses.dataclass(frozen=True)
class Cases:
    """Where a function is chosen for: its operands (`operands`, shapes and dtypes), a draw of them by seed
    (`draw`, the gate's drawn and adversarial families), and the seeds its error is pooled over."""

    operands: tuple[jax.ShapeDtypeStruct, ...]
    draw: Callable[[int], tuple]
    seeds: int


@dataclasses.dataclass(frozen=True)
class Candidate:
    """A form of the function: how it was proposed, its price in seconds (`cost.model`), its error relative
    to the form as written's (pooled RMS distance to the float64 truth, check 1's statistic), and its
    function over the operands."""

    name: str
    seconds: float
    error: float
    function: Callable


@dataclasses.dataclass(frozen=True)
class Choice:
    """The candidate chosen and every candidate measured."""

    chosen: Candidate
    candidates: tuple[Candidate, ...]


def choose(function: Callable, cases: Cases, *, accurate: bool, saturation: program.Saturation) -> Choice:
    """`function`'s form chosen for `cases`: under default numerics the cheapest whose error is within
    `PARITY` of the form as written's; under `--precision accurate` the cheapest within `PARITY` of the most
    accurate candidate's, the form as written among them, so a choice is never less accurate than the policy
    asks."""
    closed = jax.make_jaxpr(function)(*cases.operands)
    region = terms.region(closed.jaxpr)
    graph, bundle, roots = program.graphed(region)
    device = cost.Device.of()
    timed = cost.model(device)
    _, written_seconds = graph.extract(bundle, include_cost=True, cost_model=timed, extractor="greedy-dag")
    program.saturate(graph, region, rules.admitted(accurate=False, dev=True), saturation)
    proposals = {"as written": (list(roots), written_seconds.value)}
    for name, model in (("cheapest", timed), ("stable", cost.stable(device))):
        forms, _ = program.extracted(graph, bundle, model)
        proposals[name] = (forms, _priced(region, forms, timed))
    reference = _errors(function, function, cases)
    candidates = tuple(
        Candidate(name, seconds, _errors(_valued(region, closed.consts, forms), function, cases) / reference,
                  _valued(region, closed.consts, forms))
        for name, (forms, seconds) in proposals.items())
    best = min(candidate.error for candidate in candidates)
    allowed = PARITY * (best if accurate else 1.0)
    admitted = [candidate for candidate in candidates if candidate.error <= allowed]
    chosen = min(admitted, key=lambda candidate: candidate.seconds)
    return Choice(chosen, candidates)


def _priced(region: terms.Region, forms: list[terms.Term], model) -> float:
    """The price of `forms` together under `model`, each shared value once: extracted from an unsaturated
    graph of `region`'s facts holding them, where no rule has made another form of theirs."""
    graph, _, _ = program.graphed(region)
    alone = terms.Term.bundle(Vec[terms.Term](*forms))
    graph.register(alone)
    graph.run(rules.FACTS.saturate())
    _, price = graph.extract(alone, include_cost=True, cost_model=model, extractor="greedy-dag")
    return price.value


def _valued(region: terms.Region, consts, forms: list[terms.Term]) -> Callable:
    """A function evaluating `forms` over `region`'s leaves: its constants, then the operands."""
    def valued(*operands):
        values = [*consts, *operands]
        return tuple(terms.form_value(region, form, values) for form in forms)
    return valued


def _errors(candidate: Callable, function: Callable, cases: Cases) -> float:
    """`candidate`'s pooled RMS distance from `function` at float64 over the cases' seeds, every output's."""
    squared = 0.0
    for seed in range(cases.seeds):
        operands = cases.draw(seed)
        with jax.enable_x64(new_val=True):
            truths = jax.tree.leaves(function(*(jnp.asarray(operand, jnp.float64) for operand in operands)))
            truths = [np.asarray(truth, np.float64) for truth in truths]
        computed = jax.tree.leaves(jax.jit(candidate)(*operands))
        outputs = [np.asarray(output, np.float64) for output in computed]
        squared += sum(float(np.sum((output - truth) ** 2))
                       for output, truth in zip(outputs, truths, strict=True))
    return float(np.sqrt(squared)) or float(np.finfo(np.float64).tiny)
