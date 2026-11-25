"""The merged state (`means.merged_state`, docs/design/algebra.md §6.12), split out of
`tests/test_algebra_scaled.py` so each file stays well under the CI wall bound (row 12434): value, single-
operand forward derivatives, batched parity, landed anchor, axis invariance, and broadcast anchors."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from reference_error import ahead_arm, as_given, assert_as_exact, dew_path
from test_algebra_scaled import NATURAL, WAYS, _batch, _state, _ways

from dew.algebra import means
from dew.parity import FEWEST_SEEDS


@functools.cache
def _apart(axis: int):
    """The merged output and the merged total and its landed anchor of states along `axis`, apart, one
    function an axis (`_state`)."""
    return lambda outputs, totals, anchors: (means.merged_output(outputs, totals, anchors, NATURAL, axis),
                                             *means.merged_total(totals, anchors, NATURAL, axis))


def _zero_massed(operands):
    """`operands` with each merge's second state of no total."""
    outputs, totals, anchors = operands
    return outputs, totals * np.asarray([1, 0, 1, 1], np.float32), anchors


def _on_par(state, apart, way: str, draw, label: str) -> None:
    """`state` held on par with `apart` from float64 (check 1, the gate's `assert_as_exact`, `apart` one of
    Dew's own paths), each run `way` (`_ways`; op by op both as given): two programs, which XLA promises no
    common bits, held so and not bit for bit (docs/design/testing.md; the merge P0's base review 11, F10).
    Each draw's arms read alike in every draw, so a pair of one value resolves at the fewest seeds."""
    if way == "op by op":
        state, apart = (as_given(_ways(fn, f"{label}, {role}")["op by op"])
                        for fn, role in ((state, "state"), (apart, "apart")))
    assert_as_exact(state, dew_path(apart), draw, f"{label}, {way}", seeds=FEWEST_SEEDS)


def _with_directions(draw):
    """`draw`'s operands and a direction along each, drawn from its seed (the draw holds what the claim
    varies)."""
    def drawn(seed):
        operands = draw(seed)
        rng = np.random.default_rng(seed + 1)
        return (*operands, *(np.asarray(rng.standard_normal(x.shape), np.float32) for x in operands))
    return drawn


@functools.cache
def _value(form):
    """`form`'s merged output and total at a draw's operands, its directions read by none; one function a
    form, as `_state`'s are, so each cell's arms reuse another's programs."""
    return lambda outputs, totals, anchors, *_: form(outputs, totals, anchors)[:2]


@functools.cache
def _moved(form, at: tuple[int, ...]):
    """`form`'s merged output and total moved along its operands `at` by a draw's directions (a forward
    derivative), one function a form and a choice of operands (`_value`)."""
    def ran(*xs):
        operands, along = xs[:3], xs[3:]

        def partial(*ys):
            placed = list(operands)
            for index, y in zip(at, ys, strict=True):
                placed[index] = y
            return form(*placed)[:2]
        return jax.jvp(partial, *(tuple(each[index] for index in at) for each in (operands, along)))[1]
    return ran


@functools.cache
def _second(form):
    """The forward derivative along the totals of the gradient of `form`'s output sum as to its totals, one
    function a form (`_value`)."""
    def ran(outputs, totals, anchors, _, along_totals, __):
        def gradient(totals):
            return jax.grad(lambda totals: jnp.sum(form(outputs, totals, anchors)[0]))(totals)
        return jax.jvp(gradient, (totals,), (along_totals,))
    return ran


@functools.cache
def _batched_form(made):
    """`made(0)` mapped over a batch (`jax.vmap`), one function a form (`_value`)."""
    return jax.vmap(made(0))


MERGED_DRAWS = {"drawn": _batch, "a zero mass": lambda seed: _zero_massed(_batch(seed))}
"""The merged state's draws: three merges of four states over fp32's range (`_batch`), and the same with a
state of no total (the attention merge's zero-mass draw: an operand held is never moved by a zero, whose term
would be a zero mass's factor past the format, row 7124)."""

MERGED_SITES = {
    "its value": lambda: (_value(_state(1)), _value(_apart(1))),
    "along its outputs": lambda: (_moved(_state(1), (0,)), _moved(_apart(1), (0,))),
    "along its totals": lambda: (_moved(_state(1), (1,)), _moved(_apart(1), (1,))),
    "along its anchors": lambda: (_moved(_state(1), (2,)), _moved(_apart(1), (2,))),
    "batched": lambda: (_value(_batched_form(_state)), _value(_batched_form(_apart))),
}
"""The merged state's lighter sites, each its form and its parts' (`_state`, `_apart`): a cell a site, its
draws in it, so each compiles its own programs once; heavier derivative sites live in
`tests/test_algebra_merged_state_derivatives.py`."""


def _assert_site_on_par(sites, way: str, site: str) -> None:
    """Hold `sites[site]`'s merged-state form on par with its separate parts over `MERGED_DRAWS`."""
    state, apart = sites[site]()
    for name, draw in MERGED_DRAWS.items():
        _on_par(state, apart, way, _with_directions(draw),
                f"a merged state against its output and its total, {site}, {name}")


@pytest.mark.parametrize("site", MERGED_SITES)
@pytest.mark.parametrize("way", WAYS)
def test_a_merged_state_is_on_par_with_its_output_and_its_total_and_their_derivatives(way, site):
    """The merged state (`means.merged_state`, one operation a template may run as one kernel) is on par with
    the merged output and the merged total from float64 (`_on_par`), and its single-operand derivatives and
    batched form with theirs, each a site (`MERGED_SITES`), at each draw (`MERGED_DRAWS`)."""
    _assert_site_on_par(MERGED_SITES, way, site)


@pytest.mark.parametrize("way", WAYS)
def test_a_merged_states_landed_anchor_is_its_parts_bit_for_bit(way):
    """The merged state's landed anchor (`means.merged_state`), an exact contract, is the merged total's bit
    for bit (`means.merged_total`), at each draw (`MERGED_DRAWS`)."""
    state, apart = (_ways(form, label)[way] for form, label in (
        (_state(1), "a merged state's anchor"), (_apart(1), "its parts' anchor")))
    # Every arm run before any is judged (`reference_error.ran_here`).
    found = [(state(*draw(0))[2], apart(*draw(0))[2]) for draw in MERGED_DRAWS.values()]
    for mine, theirs in found:
        np.testing.assert_array_equal(np.asarray(mine), np.asarray(theirs))


@pytest.mark.parametrize("way", WAYS)
def test_a_merged_state_is_the_same_on_whichever_axis_its_states_lie(way):
    """16 states of 256 rows, their anchors drawn 3 apart, merged along their first axis and along their
    second (`means.merged_state`): its output and total on par from float64 (`_on_par`: two layouts, two
    programs, held by check 1, not bit for bit, the merge P0's base review 13, F4), and its landed anchor,
    an exact contract, the same bit for bit (XLA's of a column and a merge kernel's of a row, a Triton tree,
    gave 29 of 64 totals apart on a GPU; XLA's column and row on a CPU 118 of these 256)."""
    def drawn(seed):
        rng = np.random.default_rng([0, seed])
        return (rng.standard_normal((16, 256, 8)).astype(np.float32),
                np.abs(rng.standard_normal((16, 256))).astype(np.float32),
                (3 * rng.standard_normal((16, 256))).astype(np.float32))

    def first(o, w, m):
        return means.merged_state(o, w, m, NATURAL, 0)

    def second(o, w, m):
        return means.merged_state(jnp.swapaxes(o, 0, 1), w.T, m.T, NATURAL, 1)

    _on_par(lambda *xs: first(*xs)[:2], lambda *xs: second(*xs)[:2], way, drawn,
            "a merged state along either axis")
    operands = drawn(0)
    anchors = [np.asarray(_ways(fn, f"a merged state's anchor, {index}")[way](*operands)[2]).view(np.uint32)
               for index, fn in enumerate((first, second))]
    np.testing.assert_array_equal(*anchors)


@pytest.mark.parametrize("anchors", [np.float32(-0.5), np.linspace(-3.0, 0.0, 5, dtype=np.float32)],
                         ids=["a common scalar", "shared across the batch"])
def test_a_merged_state_of_broadcast_anchors_is_its_output_and_its_total(anchors):
    """A merged state whose anchors are common to its states (a scalar) or shared across its batch (one a
    state) is on par with its merged output and merged total from float64 (`_on_par`; two programs, the
    merge P0's base review 12, F5), which broadcast them as the law weighs them, its landed anchor theirs
    bit for bit (the review of the streamed law: its chunk loader indexed each operand's states axis and
    refused them)."""
    def drawn(seed):
        rng = np.random.default_rng([5, seed])
        return (rng.standard_normal((3, 5, 2)).astype(np.float32),
                rng.uniform(0.5, 2.0, (3, 5)).astype(np.float32), anchors)

    def value(form):
        return lambda *operands: form(*operands)[:2]

    _on_par(value(_state(1)), value(_apart(1)), "jitted", drawn, "a merged state of broadcast anchors")
    operands = drawn(0)
    anchors_of = [np.asarray(ahead_arm(jax.jit(form), f"broadcast anchors' {label}")(*operands)[2]).tolist()
                  for form, label in ((_state(1), "state"), (_apart(1), "parts"))]
    assert anchors_of[0] == anchors_of[1]
