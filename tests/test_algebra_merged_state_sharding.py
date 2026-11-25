"""`vmap`, `shard_map`, sharded `jit`, and explicit-mesh tests for `means.merged_output` and
`means.merged_state` (docs/design/algebra.md §6.12), split out of `tests/test_algebra_scaled.py` so every
file stays well under the train CI wall bound (row 12434)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from reference_error import ahead_arm, as_given, assert_as_exact, dew_path
from test_algebra_scaled import NATURAL, _batch, _merge, _state

from dew.algebra import means
from dew.parity import FEWEST_SEEDS


def _gradient(merge):
    """The gradient of `merge`'s outputs' sum as to its outputs, totals and anchors."""
    return jax.grad(lambda *state: jnp.sum(merge(*state)), argnums=(0, 1, 2))


def _batched(outputs, totals, anchors):
    """The law over a leading batch of merges, its states along axis 1."""
    return means.merged_output(outputs, totals, anchors, NATURAL, 1)


def test_a_vmapped_merge_and_its_gradient_are_each_merges_own():
    """`jax.vmap` over a batch of merges, the law's value and its gradient as to every operand, each the
    merge's own run alone (`merged_p`'s, `gapped_p`'s, `applied_p`'s and `split_p`'s batching), with the
    totals unbatched as well as batched."""
    outputs, totals, anchors = _batch(0)
    gradients = _gradient(_merge)
    merge_alone, gradient_alone = (ahead_arm(jax.jit(fn), label) for fn, label in (
        (_merge, "a merge alone"), (gradients, "a merge's gradient alone")))
    # Every arm run before any is judged (`ran_here`).
    found = []
    for in_axes in ((0, 0, 0), (0, None, 0)):
        shared = totals if in_axes[1] == 0 else totals[0]
        mapped, derived = (ahead_arm(jax.jit(jax.vmap(fn, in_axes)), f"{label} mapped over {in_axes}")(
            outputs, shared, anchors) for fn, label in ((_merge, "merges"), (gradients, "gradients")))
        alone = [(outputs[k], shared if in_axes[1] is None else shared[k], anchors[k])
                 for k in range(len(outputs))]
        found.append((mapped, derived, [merge_alone(*each) for each in alone],
                      [gradient_alone(*each) for each in alone]))
    for mapped, derived, merged, own_gradients in found:
        for k, (merge, own) in enumerate(zip(merged, own_gradients, strict=True)):
            assert np.asarray(mapped[k]).tolist() == np.asarray(merge).tolist()
            for got, expected in zip(derived, own, strict=True):
                assert np.asarray(got[k]).tolist() == np.asarray(expected).tolist()


def _each_alone(placed, alone, draw, label: str) -> None:
    """`placed`, a batch of merges run split over devices, held on par from float64 with each merge's own run
    of `alone` (check 1, `assert_as_exact`; each arm run as given, its placement its own): two programs, held
    so and not bit for bit (docs/design/testing.md; the merge P0's base review 13, F4)."""
    def each(*operands):
        runs = [alone(*(x[k:k + 1] for x in operands)) for k in range(len(operands[0]))]
        return jax.tree.map(lambda *parts: jnp.concatenate(parts), *runs)

    assert_as_exact(as_given(placed), dew_path(as_given(each)), draw, label, seeds=FEWEST_SEEDS)


@pytest.mark.mesh(devices=2)
@pytest.mark.parametrize("mode", ["shard_map", "sharded_jit"])
def test_a_merge_under_shard_map_and_under_a_sharded_jit_is_each_shards_own(mode):
    """The law and its gradient over a batch of merges split over two devices, in `jax.shard_map` (each
    shard's merges its own, its types' varying axes kept, `merged_p`'s result type) and in a jit of
    operands placed by `NamedSharding`: each merge's value and gradient on par from float64 with the merge's
    own run alone (`_each_alone`)."""
    from jax.sharding import NamedSharding, PartitionSpec

    mesh = jax.make_mesh((2,), ("merges",), devices=jax.devices()[:2])

    def value_and_gradient(o, w, m):
        gradient = jax.grad(lambda *state: jnp.sum(_batched(*state)), argnums=(0, 1, 2))(o, w, m)
        return _batched(o, w, m), gradient

    split = PartitionSpec("merges")
    alone = jax.jit(value_and_gradient)
    if mode == "shard_map":
        ran = jax.jit(jax.shard_map(value_and_gradient, mesh=mesh, in_specs=(split,) * 3,
                                    out_specs=(split, (split,) * 3)))
        label = "in a shard map"
    else:
        ran = alone
        label = "in a sharded jit"
    # Each merge's value and gradient on par from float64 with its own run alone (two programs, held by
    # check 1, the merge P0's base review 13, F4).
    _each_alone(lambda *xs: ran(*(jax.device_put(x, NamedSharding(mesh, split)) for x in xs)),
                alone, lambda seed: _batch(seed + 1, count=4), f"merges {label}")


@pytest.mark.mesh(devices=2)
def test_a_merged_state_under_an_explicit_mesh_drops_its_states_axis_from_its_shardings():
    """A batch of merges and their gradient under an explicit mesh (`AxisType.Explicit`, JAX's manual
    sharding): the merged state's reduced outputs drop the states' axis from their shardings (a ratio's
    result type kept it, `_without_axis`, and `broadcast_to` over a shape without it was refused), each
    constant and mask the law builds of its operands' sharding (one of a shape alone was refused beside them
    in a select), and each merge's state and gradient on par from float64 with the merge's own run alone
    (`_each_alone`), its landed anchor the merge's own bit for bit."""
    from jax.sharding import AxisType, NamedSharding, PartitionSpec

    mesh = jax.make_mesh((2,), ("merges",), axis_types=(AxisType.Explicit,), devices=jax.devices()[:2])
    split = PartitionSpec("merges")
    gradient = _gradient(_batched)

    def merged(o, w, m):
        state = _state(1)(o, w, m)
        return (*state[:2], *gradient(o, w, m)), state[2]

    # Hoist `jax.jit(merged)` once so every seed and per-row check reuses the compiled executable (row 12434).
    merged_jit = jax.jit(merged)

    def placed(*operands):
        with jax.set_mesh(mesh):
            return merged_jit(*(jax.device_put(x, NamedSharding(mesh, split)) for x in operands))

    # Its state's output and total and the gradient on par with each merge's alone; its landed anchor, an
    # exact contract, each merge's own bit for bit.
    _each_alone(lambda *xs: placed(*xs)[0], lambda *xs: merged_jit(*xs)[0],
                lambda seed: _batch(seed + 1, count=4), "merged states under an explicit mesh")
    operands = _batch(1, count=4)
    anchors = np.asarray(placed(*operands)[1])
    for k in range(len(operands[0])):
        alone = merged_jit(*(x[k:k + 1] for x in operands))[1]
        assert anchors[k].tolist() == np.asarray(alone)[0].tolist()


@pytest.mark.mesh(devices=2)
def test_a_long_row_of_states_under_an_explicit_mesh_merges_as_each_merge_alone():
    """Merges of seventeen states each (a row the law reads in several chunks, its loops' carries then a
    mesh's types: `means._streamed`) split over two devices under an explicit mesh: each merge's state on par
    from float64 with its own run alone (`_each_alone`), its landed anchor its own bit for bit (the review of
    the streamed law: carries of a shape alone, replicated, were refused beside the steps' sharded
    reductions)."""
    from jax.sharding import AxisType, NamedSharding, PartitionSpec

    def drawn(seed):
        rng = np.random.default_rng([17, seed])
        return (rng.standard_normal((2, 17, 3)).astype(np.float32),
                rng.uniform(0.5, 2.0, (2, 17)).astype(np.float32),
                rng.uniform(-30, 0, (2, 17)).astype(np.float32))

    mesh = jax.make_mesh((2,), ("merges",), axis_types=(AxisType.Explicit,), devices=jax.devices()[:2])
    state = jax.jit(_state(1))

    def placed(*operands):
        with jax.set_mesh(mesh):
            return state(*(jax.device_put(x, NamedSharding(mesh, PartitionSpec("merges"))) for x in operands))

    # Its output and total on par with each merge's alone; its landed anchor each merge's own bit for bit.
    _each_alone(lambda *xs: placed(*xs)[:2], lambda *xs: state(*xs)[:2], drawn,
                "long rows under an explicit mesh")
    operands = drawn(0)
    anchors = np.asarray(placed(*operands)[2])
    for k in range(2):
        assert anchors[k].tolist() == np.asarray(state(*(x[k:k + 1] for x in operands))[2])[0].tolist()
