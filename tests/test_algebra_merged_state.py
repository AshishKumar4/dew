"""The merged state (`means.merged_state`, docs/design/algebra.md §6.12), split out of
`tests/test_algebra_scaled.py` so each file prepares well under the GPU tier's train bound (row 12434):
parity against separate `merged_output` + `merged_total` and their derivatives, axis invariance, `vmap`,
`shard_map`, sharded `jit`, explicit meshes, and broadcast anchors."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from reference_error import ahead_arm, as_given, assert_as_exact, dew_path

from dew.algebra import means, windows
from dew.parity import FEWEST_SEEDS

NATURAL = windows.Units()
"""Natural anchors, compensated: the accurate form's units (`windows.Units`)."""


def _merge(outputs, totals, anchors):
    """The law over natural anchors along the states' first axis."""
    return means.merged_output(outputs, totals, anchors, NATURAL, 0)


def _ways(fn, label: str):
    """`fn` each way a cell runs the law (`WAYS`): op by op, eagerly, and jitted as an arm a GPU tier part's
    prepare compiles ahead at its host inputs and its run runs from that program (`reference_error.ahead_arm`,
    `fn` jitted outside a part), under `label`, one program a label."""
    def eager(*args):
        with jax.disable_jit():
            return fn(*args)
    jitted = ahead_arm(jax.jit(fn), label)
    return {"jitted": jitted, "op by op": eager, "on a gpu": jitted, "on a tpu": jitted}


CPU_LANE = pytest.mark.cpu(reason="eager dispatch and the CPU lane's jit: a GPU's tier runs the law on a gpu")
WAYS = [pytest.param("jitted", marks=CPU_LANE), pytest.param("op by op", marks=CPU_LANE),
        pytest.param("on a gpu", marks=pytest.mark.gpu(reason="the primitives on a GPU")),
        pytest.param("on a tpu", marks=pytest.mark.tpu(reason="the primitives on a TPU"))]


def _batch(seed: int, count: int = 3) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`count` merges of four states of eight channels each: outputs, totals and anchors over fp32's range."""
    rng = np.random.default_rng(seed)
    outputs = (rng.standard_normal((count, 4, 8)) * 2.0 ** rng.integers(-60, 60, (count, 4, 1)))
    totals = 2.0 ** rng.integers(-100, 100, (count, 4)) * rng.uniform(1, 2, (count, 4))
    anchors = rng.uniform(-30, 0, (count, 4))
    return tuple(np.asarray(x, np.float32) for x in (outputs, totals, anchors))


def _gradient(merge):
    """The gradient of `merge`'s outputs' sum as to its outputs, totals and anchors."""
    return jax.grad(lambda *state: jnp.sum(merge(*state)), argnums=(0, 1, 2))


def _batched(outputs, totals, anchors):
    """The law over a leading batch of merges, its states along axis 1."""
    return means.merged_output(outputs, totals, anchors, NATURAL, 1)


@functools.cache
def _state(axis: int):
    """The merged state of states along `axis` (`means.merged_state`), one function an axis, so a program of
    it traced and compiled for one cell serves every other at its shapes (rows 9806 and on: the merged
    state's cells compiled each draw's arms again)."""
    return lambda outputs, totals, anchors: means.merged_state(outputs, totals, anchors, NATURAL, axis)


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
    "along every operand": lambda: (_moved(_state(1), (0, 1, 2)), _moved(_apart(1), (0, 1, 2))),
    "along its outputs": lambda: (_moved(_state(1), (0,)), _moved(_apart(1), (0,))),
    "along its totals": lambda: (_moved(_state(1), (1,)), _moved(_apart(1), (1,))),
    "along its anchors": lambda: (_moved(_state(1), (2,)), _moved(_apart(1), (2,))),
    "its gradient's derivative": lambda: (_second(_state(1)), _second(_apart(1))),
    "batched": lambda: (_value(_batched_form(_state)), _value(_batched_form(_apart))),
}
"""The merged state's sites, each its form and its parts' (`_state`, `_apart`): a cell a site, its draws in
it, so each compiles its own programs once (row 10154: a cell a draw compiled every site's, 250 to 280 s a
cell past the group target, its two draws' programs alike)."""


@pytest.mark.parametrize("site", MERGED_SITES)
@pytest.mark.parametrize("way", WAYS)
def test_a_merged_state_is_on_par_with_its_output_and_its_total_and_their_derivatives(way, site):
    """The merged state (`means.merged_state`, one operation a template may run as one kernel) is on par with
    the merged output and the merged total from float64 (`_on_par`), and its derivatives with theirs: along
    every operand and along each alone, the gradient's derivative along the totals, and batched, each a
    site (`MERGED_SITES`), at each draw (`MERGED_DRAWS`)."""
    state, apart = MERGED_SITES[site]()
    for name, draw in MERGED_DRAWS.items():
        _on_par(state, apart, way, _with_directions(draw),
                f"a merged state against its output and its total, {site}, {name}")


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


def _gradients_derivatives(form):
    """The derivatives of a two-state merge's gradient as to its totals (its states' anchors 0, one channel)
    under a cotangent of its output and its total: along the outputs, along the totals forward over
    reverse and reverse over reverse (the attention merge's second-order site's)."""
    def ran(outputs, totals, along_outputs, along_totals):
        def gradient(o, w):
            def objective(each):
                output, total, _ = form(o[:, None], each, jnp.zeros_like(each))
                return 0.75 * output[0] + 0.5 * total
            return jax.grad(objective)(w)

        mixed = jax.jvp(lambda o: gradient(o, totals), (outputs,), (along_outputs,))[1]
        forward = jax.jvp(lambda w: gradient(outputs, w), (totals,), (along_totals,))[1]
        reverse = jax.grad(lambda w: jnp.vdot(gradient(outputs, w), along_totals))(totals)
        return mixed, forward, reverse
    return ran


@pytest.mark.parametrize("totals", [[1.25, 0.0], [1.25, 0.5]], ids=["a zero mass", "two masses"])
@pytest.mark.parametrize("way", WAYS)
def test_a_merged_states_gradients_derivatives_are_on_par_with_its_parts(way, totals):
    """The derivatives of a merged state's gradient (`means.merged_state`), forward and reverse over its
    reverse, on par with the separate merged output's and total's from float64 (`_on_par`), at a state of no
    total beside one of a drawn mass too (the attention merge's zero-mass draw, row 7124, where a rule that
    moved every operand read the anchors' zero tangent as a term and missed by its whole value)."""
    def drawn(seed):
        rng = np.random.default_rng(seed)
        directions = (np.asarray(rng.standard_normal(2), np.float32) for _ in range(2))
        return (np.asarray(rng.uniform(-1, 1, 2), np.float32), np.asarray(totals, np.float32), *directions)

    _on_par(_gradients_derivatives(_state(0)), _gradients_derivatives(_apart(0)), way, drawn,
            f"a merged state's gradient's derivatives against its parts', {totals}")


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
    """The merged state and the law's gradient over merges split over two devices under an explicit mesh
    (each array's sharding a part of its type), its states along their second axis: each result's spec
    without that axis (base review F3: the spec kept whole beside a shape without it was refused), each
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
