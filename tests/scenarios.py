"""The gate's boundary, derivative and payload scenarios (row 4514): one set of helpers a feature's tests plug
their own function into, each scenario judged by the gate's existing checks, never a new bound.

- Sizes at an axis's boundaries (`boundary_sizes`): empty, one, one short of a block, a block, one past it.
- A function's derivative orders and its batching (`transformed`): the primal, its gradient, its forward
  derivative, the forward derivative of its gradient (contract_support's orders, the one owner of them) and
  the function under `jax.vmap`, each held to the same transform of the reference by check 1
  (`assert_transforms_as_exact`).
- Integer and boolean inputs differentiated (`assert_differentiates_past_integers`): their tangents are
  float0 and their cotangents zero, the float inputs' derivatives unchanged.
- Masked rows filled with what no unmasked output may read (`assert_masked_rows_unread`): NaN, the infinities
  and float32's largest, every unmasked output bit for bit what a zero fill gives.
- Examples reordered or split into parts (`assert_examples_independent`): a function of independent examples
  gives each example's output whatever its neighbours: reordered, bit for bit (one program on the same
  values); split into parts, each another program at its part's shape, on par by check 1.
- A run restarted midstream (`assert_restartable`): steps carried in two runs from the state the first leaves
  are the steps of one run, bit for bit.

Bit-for-bit claims are held where the program is one program on the same values (a deterministic CPU run);
values two programs compute are held by check 1 (`assert_transforms_as_exact`)."""

import itertools
from collections.abc import Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from contract_support import orders
from reference_error import as_given, assert_as_exact

FILLS = (np.nan, np.inf, -np.inf, float(np.finfo(np.float32).max))
"""What a masked row holds in `assert_masked_rows_unread`: values any arithmetic that reads them carries into
its output (NaN and the infinities), and one that overflows a product."""
TRANSFORMS = ("primal", "vjp", "jvp", "jvp of grad", "vmap")
"""The transforms `transformed` gives, by name: contract_support's orders' first four, and batching."""


def boundary_sizes(block: int) -> tuple[int, ...]:
    """The sizes at an axis's boundaries for a block of `block`: 0, 1, block - 1, block and block + 1, each
    once, none below zero."""
    return tuple(sorted({0, 1, max(block - 1, 0), block, block + 1}))


def transformed(fn: Callable, name: str) -> Callable:
    """`fn` under transform `name` (`TRANSFORMS`), as a function of the draw `(primals, direction,
    cotangent)`: an order of contract_support's (`orders`), or `fn` over a leading axis of the draw's
    examples under `jax.vmap`."""
    if name == "vmap":
        return lambda primals, direction, cotangent: jax.vmap(fn)(*primals)
    return lambda primals, direction, cotangent: orders(fn, primals, direction, cotangent)[name]()


def assert_transforms_as_exact(dew_fn: Callable, reference_fn: Callable, draw: Callable, label: str, *,
                               seeds: int, truth_fn: Callable | None = None,
                               names: Sequence[str] = TRANSFORMS) -> None:
    """Each transform of `names` (`transformed`) of `dew_fn` on par with the same transform of
    `reference_fn` over `seeds` draws, check 1 under `label` and the transform's name, the truth the same
    transform of `truth_fn` (of the reference widened, by default). `draw(seed)` gives `(primals,
    direction, cotangent)`, `direction` a tangent of each primal and `cotangent` one of the output; the
    vmapped transform reads the primals' leading axis as its examples."""
    for name in names:
        assert_as_exact(transformed(dew_fn, name), transformed(reference_fn, name), draw, f"{label}, {name}",
                        seeds=seeds,
                        truth_fn=None if truth_fn is None else transformed(truth_fn, name))


def _float0(value) -> bool:
    return np.asarray(value).dtype == jax.dtypes.float0


def assert_differentiates_past_integers(fn: Callable, primals: tuple, label: str) -> None:
    """`fn` differentiated with every input, integer and boolean ones among them: forward, with each integer
    or boolean input's tangent float0, the derivative is what it is with those inputs held as constants;
    reverse, each of their cotangents is float0, so JAX carries no derivative through one."""
    integral = [not jnp.issubdtype(jnp.asarray(value).dtype, jnp.inexact) for value in primals]
    tangents = tuple(np.zeros(np.shape(value), jax.dtypes.float0) if whole else jnp.ones_like(value)
                     for value, whole in zip(primals, integral, strict=True))
    _, tangent = jax.jvp(fn, primals, tangents)

    def floats_only(*floats):
        values = iter(floats)
        return fn(*(value if whole else next(values) for value, whole in zip(primals, integral, strict=True)))

    floats = tuple(value for value, whole in zip(primals, integral, strict=True) if not whole)
    _, alone = jax.jvp(floats_only, floats, tuple(jnp.ones_like(value) for value in floats))
    for found, expected in zip(jax.tree.leaves(tangent), jax.tree.leaves(alone), strict=True):
        np.testing.assert_array_equal(np.asarray(found), np.asarray(expected), err_msg=label)
    _, pull = jax.vjp(fn, *primals)
    cotangents = pull(jax.tree.map(jnp.ones_like, fn(*primals)))
    held = zip(cotangents, integral, strict=True)
    assert all(_float0(cotangent) for cotangent, whole in held if whole), label


def assert_masked_rows_unread(fn: Callable, inputs: tuple, masked: Callable, label: str) -> None:
    """`fn` on `inputs` with the rows `masked(inputs)` names (a tuple of each input's boolean row mask, None
    for an input it does not mask, each mask broadcasting over its input's trailing axes) filled with each of
    FILLS gives, bit for bit, what a zero fill gives on every output it does not mask, and no output it
    keeps is non-finite: a masked row no unmasked output reads. `fn` returns `(outputs, kept)`, `kept` the
    boolean mask of each output's entries the masked rows may not reach."""
    masks = masked(inputs)

    def filled(fill):
        # Each mask names leading rows of its input, read over the input's trailing axes.
        return tuple(value if mask is None else jnp.where(
            jnp.reshape(mask, np.shape(mask) + (1,) * (np.ndim(value) - np.ndim(mask))),
            jnp.asarray(fill, jnp.asarray(value).dtype), value)
            for value, mask in zip(inputs, masks, strict=True))

    base, kept = fn(*filled(0.0))
    for fill in FILLS:
        outputs, _ = fn(*filled(fill))
        for found, expected, keep in zip(jax.tree.leaves(outputs), jax.tree.leaves(base),
                                         jax.tree.leaves(kept), strict=True):
            found, expected = np.asarray(found)[np.asarray(keep)], np.asarray(expected)[np.asarray(keep)]
            assert np.all(np.isfinite(found)), (label, fill)
            np.testing.assert_array_equal(found, expected, err_msg=f"{label}, a masked row of {fill}")


def assert_examples_independent(fn: Callable, draw: Callable, label: str, *, seeds: int, parts: int = 2,
                                truth_fn: Callable | None = None) -> None:
    """`fn` over a batch of independent examples (each input's leading axis, `draw(seed)`'s) gives each
    example's outputs whatever the batch around it: the batch reversed gives the outputs reversed bit for
    bit (one program on the same values), and the batch in `parts` consecutive parts, each another
    program at its own batch's shape, gives outputs on par with the whole batch's (check 1 over `seeds`,
    the truth `truth_fn` or the whole batch's widened)."""
    examples = draw(0)
    # One program at the batch's shape for the whole batch and its reversal, each part's at its own, as
    # the gate compiles an arm (`jax.jit`).
    part = jax.jit(fn)
    whole = part(*examples)
    reverse = part(*(value[::-1] for value in examples))
    for kept, reordered in zip(jax.tree.leaves(whole), jax.tree.leaves(reverse), strict=True):
        np.testing.assert_array_equal(np.asarray(reordered)[::-1], np.asarray(kept),
                                      err_msg=f"{label}, reversed")
    bounds = np.linspace(0, len(examples[0]), parts + 1).astype(int)

    def joined(*examples):
        pieces = [part(*(value[start:stop] for value in examples))
                  for start, stop in itertools.pairwise(bounds) if stop > start]
        return jax.tree.map(lambda *leaves: np.concatenate([np.asarray(leaf) for leaf in leaves]), *pieces)

    assert_as_exact(as_given(joined), fn, draw, f"{label}, in {parts} parts", seeds=seeds, truth_fn=truth_fn)


def assert_restartable(step: Callable, state, inputs: Sequence, label: str, *, at: int) -> None:
    """`step(state, input) -> state` over `inputs` from `state`, restarted after `at` steps from the state
    the first `at` left, ends where one run over them all ends, bit for bit: a run's state is all it
    carries."""
    def run(state, steps):
        for value in steps:
            state = step(state, value)
        return state

    whole = run(state, inputs)
    resumed = run(jax.tree.map(np.asarray, run(state, inputs[:at])), inputs[at:])
    for found, expected in zip(jax.tree.leaves(resumed), jax.tree.leaves(whole), strict=True):
        np.testing.assert_array_equal(np.asarray(found), np.asarray(expected),
                                      err_msg=f"{label}, at {at}")
