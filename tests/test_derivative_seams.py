"""The derivative contract's own validation (tests/contract_support.py's `assert_contract`). Every custom
derivative of Dew's declares the contract that holds it at its seam (`dew.nn.derivatives.contracted`, which
tools/lint_slop.py's SLOP028 checks), each one of tests/test_contracts_*.py."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from contract_support import EDGES, LessFinite, assert_contract, assert_drawn_contract, normal_draw


def test_the_contract_refuses_a_formula_less_finite_than_the_plain_one():
    """The rule's validation: a formula overflowing where the plain one does not (|x| as the root of its
    square, past 2**64) is refused, at its primal and its derivatives; the plain formula against itself
    holds, at the same edge, where float32's own arithmetic holds every value."""
    x = (jnp.asarray([2.0**70, -3.0, 2.0**-70], jnp.float32),)
    direction = (jnp.ones((3,), jnp.float32),)
    cotangent = jnp.ones((3,), jnp.float32)
    assert_contract(jnp.abs, x, direction, cotangent, "the plain formula", plain_fn=jnp.abs)
    with pytest.raises(LessFinite):
        assert_contract(lambda x: jnp.sqrt(x * x), x, direction, cotangent, "a squared root",
                        plain_fn=jnp.abs)


def test_the_contract_holds_the_orders_a_case_takes_alone():
    """A case naming the orders it holds (`taking`, row 9077) holds those alone, off training's mesh and on
    it: a custom VJP, whose forward-mode orders do not run, holds its primal and its reverse derivative; a
    name of no order, taken or left out, is refused, and so is a case left no order to take."""
    @jax.custom_vjp
    def doubled(x):
        return 2 * x

    doubled.defvjp(lambda x: (2 * x, None), lambda _, g: (2 * g,))
    x = (jnp.asarray([1.0, -3.0], jnp.float32),)
    direction = (jnp.ones((2,), jnp.float32),)
    cotangent = jnp.ones((2,), jnp.float32)
    assert_contract(doubled, x, direction, cotangent, "a custom VJP's reverse derivative",
                    taking=("primal", "vjp"))
    for named in ({"taking": ("jvp of grads",)}, {"without": ("vjp twice",)}):
        with pytest.raises(ValueError):
            assert_contract(doubled, x, direction, cotangent, "a misnamed order", **named)
    # A case left no order, every one it takes left out, holds nothing: refused, never passed (row 11778: the
    # remat order sets under an interpreted kernel's `without` passed 10 cells taking no order).
    with pytest.raises(ValueError):
        assert_contract(doubled, x, direction, cotangent, "a case of no order", taking=("vjp",),
                        without=("vjp",))


def split_dot(seed, scale):
    """A split sum's draw (`_summed_over_the_mesh`): two vectors of 1024 at `scale`, their directions, and a
    cotangent, each stream salted apart (row 8464: seeds seed, seed + 1 and seed + 2 made one seed's
    direction the next seed's primals, and a shared scalar factor left its jvp of grad unresolved)."""
    primals = normal_draw((0, seed), (1024,), (1024,), scale=scale)
    return primals, normal_draw((1, seed), (1024,), (1024,), scale=scale), normal_draw((2, seed), (1,))[0]


def _split_over_the_mesh(specs: int):
    """A product of two operands split over the context mesh's data axis by a shard map given `specs` of their
    specs, as a template's launch reads the mesh it runs under (row 7973's: one whose specs left two of its
    operands out); off a mesh, the product itself."""
    from jax.sharding import PartitionSpec as P

    def product(x, y):
        if jax.sharding.get_abstract_mesh().empty:
            return x * y
        return jax.shard_map(lambda a, b: a * b, in_specs=(P("data"),) * specs, out_specs=P("data"))(x, y)

    return product


def _summed_over_the_mesh(psums: int):
    """A dot product split over the context mesh's data axis by a shard map, both operands' halves on each
    device, each device's part summed across the devices `psums` times (a launch counting its sum twice at
    two); off a mesh, the dot itself."""
    from jax.sharding import PartitionSpec as P

    def total(x, w):
        if jax.sharding.get_abstract_mesh().empty:
            return jnp.sum(x * w, keepdims=True)

        def local(a, b):
            summed = jax.lax.psum(jnp.sum(a * b, keepdims=True), "data")
            for _ in range(psums - 1):
                summed = jax.lax.psum(jax.lax.pvary(summed, ("data",)), "data")
            return summed

        return jax.shard_map(local, in_specs=(P("data"), P("data")), out_specs=P())(x, w)

    return total


@pytest.mark.mesh(devices=2)
def test_the_contract_runs_every_order_on_trainings_mesh():
    """Every order is a check-1 site on training's mesh of two devices (`contract_support._on_a_mesh`), as a
    training step runs it, its reference the order off a mesh: a launch given its operands' specs holds,
    and one a spec short, the same product off a mesh, raises there (row 7973: Mamba-3's forked launch, two
    specs short, raised under any mesh, and the contract cells, run off one, passed it); a sum split over the
    devices holds, and one counted twice fails check 1 by its own size."""
    x = tuple(jnp.asarray(value, jnp.float32) for value in ([1.5, -2.0, 3.0, 0.25], [2.0, 0.5, -1.0, 4.0]))
    cotangent = jnp.ones((4,), jnp.float32)
    assert_contract(_split_over_the_mesh(2), x, x, cotangent, "a product split over the mesh")
    with pytest.raises(AssertionError):
        assert_contract(_split_over_the_mesh(1), x, x, cotangent, "a product a spec short")

    # A dot of 1024, its rounding errors check 1's to resolve (four of them resolve no ratio).
    assert_drawn_contract(_summed_over_the_mesh(1), split_dot, EDGES["normal"], "a split sum")
    with pytest.raises(AssertionError):
        assert_drawn_contract(_summed_over_the_mesh(2), split_dot, EDGES["normal"],
                              "a split sum counted twice")


@pytest.mark.mesh(devices=2)
def test_a_contracts_mesh_sites_draw_their_declared_seeds_and_size_none(monkeypatch):
    """Given `seeds` (a cell sharded across CI's map tasks, its mark's count), every order's check-1 site on
    training's mesh draws that count's seeds, its second stage's past them where its first does not decide,
    and in an ordinary run sizes none (`seeds_for`'s draws, from SIZING_FROM on; a qualification run sizes a
    declared count, `_qualified`): a cell's sizing on every run took most of a 300 s contract cell (row
    8429's), which a shard mark's count in its place cuts."""
    from check1_wire import QUALIFY
    from reference_error import staged_units

    monkeypatch.delenv(QUALIFY, raising=False)
    drawn: set[int] = set()

    def inputs(seed, scale):
        drawn.add(int(seed))
        return split_dot(seed, scale)

    assert_drawn_contract(_summed_over_the_mesh(1), inputs, EDGES["normal"], "a split sum of declared seeds",
                          seeds=16)
    assert set(range(16)) <= drawn <= set(range(staged_units(16)))


def test_the_contract_refuses_a_bfloat16_formula_less_finite_than_the_plain_one():
    """The same refusal where the formula's values are bfloat16 (row 4499): the squared root overflows
    bfloat16's range where |x| does not, and its leaves are held as a float32 formula's are."""
    x = (jnp.asarray([2.0**70, -3.0, 2.0**-70], jnp.bfloat16),)
    direction = (jnp.ones((3,), jnp.bfloat16),)
    cotangent = jnp.ones((3,), jnp.bfloat16)
    with pytest.raises(LessFinite):
        assert_contract(lambda x: jnp.sqrt(x * x), x, direction, cotangent, "a bf16 squared root",
                        plain_fn=jnp.abs)


def test_the_contract_holds_a_sum_only_where_no_order_of_it_can_overflow():
    """A sum is held finite where the magnitudes of its terms sum inside float32's range, whatever their
    order: a form that sums before the scale that brings it back overflows where the plain formula, scaling
    each term first, does not, and is refused (row 1768's class). Where float32 overflows the terms in one
    order and not another (terms near the largest that cancel), no order is held: a form that sums the
    alternate terms first, which overflows where the plain formula's pairs cancel, holds the contract (row
    5426, two summation orders)."""
    direction, cotangent = (jnp.ones((8,), jnp.float32),), jnp.ones((), jnp.float32)

    def scaled_first(x):
        return jnp.sum(x * jnp.float32(2.0**-10))

    alike = (jnp.full((8,), 2.0**126, jnp.float32),)
    assert_contract(scaled_first, alike, direction, cotangent, "terms scaled first", plain_fn=scaled_first)
    with pytest.raises(LessFinite):
        assert_contract(lambda x: jnp.sum(x) * jnp.float32(2.0**-10), alike, direction, cotangent,
                        "a sum scaled after", plain_fn=scaled_first)
    cancelling = (jnp.asarray([2.0**127, -(2.0**127)] * 4, jnp.float32),)
    assert_contract(lambda x: jnp.sum(x[::2]) + jnp.sum(x[1::2]), cancelling, direction, cotangent,
                    "alternate terms first", plain_fn=lambda x: jnp.sum(jnp.sum(x.reshape(4, 2), axis=1)))


def test_the_contract_holds_terms_that_cancel_inside_the_range_in_every_order():
    """Of 2^126 twice and its negative twice every partial sum, in every order, stays within 2^127, inside
    float32, though their magnitudes sum to 2^128. The position is held: the plain sum is finite there, and
    a form that scales its terms up before it sums them and back down after overflows and is refused.
    Judged by the magnitudes' sum the position was dropped, and that form passed (row 5426)."""
    x = (jnp.asarray([2.0**126, 2.0**126, -(2.0**126), -(2.0**126)], jnp.float32),)
    direction, cotangent = (jnp.ones((4,), jnp.float32),), jnp.ones((), jnp.float32)
    assert_contract(jnp.sum, x, direction, cotangent, "the plain sum", plain_fn=jnp.sum)
    with pytest.raises(LessFinite):
        assert_contract(lambda x: jnp.sum(x * 4) / 4, x, direction, cotangent, "terms scaled up first",
                        plain_fn=jnp.sum)


def test_the_contract_reads_a_contraction_behind_a_mask_by_its_own_terms():
    """A plain formula that selects each row's product by a mask sums its groups last, the contraction
    behind that sum: the position is held by the contraction's own terms. Where they cancel past float32's
    range, a form that sums them in another order than the formula's and overflows is held to neither (a
    grouped GEMM's second derivative near float32's largest, row 5426)."""
    groups = np.asarray([0, 1])
    rows = (jnp.asarray([[2.0**64, -(2.0**64)] * 4] * 2, jnp.float32),)
    weights = jnp.full((2, 8, 1), 2.0**63, jnp.float32)
    direction, cotangent = (jnp.zeros((2, 8), jnp.float32),), jnp.ones((2, 1), jnp.float32)

    def masked(rows):
        pairs = jnp.einsum("rpk,gpkn->rgpn", rows.reshape(2, 4, 2), weights.reshape(2, 4, 2, 1))
        return jnp.einsum("rg,rgn->rn", jax.nn.one_hot(groups, 2, dtype=rows.dtype), pairs.sum(axis=2))

    def alternate_first(rows):
        chosen = weights[groups]
        return (jnp.einsum("rk,rkn->rn", rows[:, ::2], chosen[:, ::2])
                + jnp.einsum("rk,rkn->rn", rows[:, 1::2], chosen[:, 1::2]))

    assert_contract(alternate_first, rows, direction, cotangent, "alternate terms of a selected product",
                    truth_fn=masked, plain_fn=masked)
