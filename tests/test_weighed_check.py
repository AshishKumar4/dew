"""Check 1's conditioning (tests/reference_error.py's `assert_as_exact(..., conditioning=)`, ledger 1714):
where the unweighted check stays undecided with its budget spent, each arm's error weighed by the reference
program's own rounding there (tests/rounding_bound.py's `rounding_scale`), so a fault on the elements a
program computes well shows beside the one it amplifies.

The operation holds both in one row: `amplified`, a value pushed through 2^k and back, k drawn per seed from
0 to 24, whose rounding is up to 2^24 times its own size's and differs by orders from draw to draw, and the
products of the rest, each rounded once. Unweighed, a row's error is its amplified element's, heavy-tailed
across draws, and the check stays undecided over its seeds; weighed, every element's error is in units of
its own rounding."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from reference_error import GateFailure, Inconclusive, assert_as_exact, null_rate_bound, null_runs
from rounding_bound import rounding_scale

ROWS, HALF = 1, 16

WEIGHED_SEEDS = 8
"""Check 1's declared seeds here, the unweighted check's and the weighed one's alike: over them and their
REDRAW stage the unweighted check stays undecided on every arm below that its rows do not decide."""

NULL_RUNS = 64
"""The null control's runs: the reference against its reformulation on the same draws, judged as a weighed
site is."""

WEIGHED_DECIDED = 35
"""The null runs the weighed check decides at least: 35 of 64 on armada at 256 conditioned seeds, each run's
draws past both budgets of the run before it (`null_runs`); the unweighted check decides the rest, its
worst-row rule failing some (row 3684; run gate-r12 at fa6c49b25a)."""

CONDITIONED_SEEDS = 256
"""The weighed check's declared seeds here: its heavy class holds one value a draw, some draws exact in both
arms (no paired ratio), so its pooled interval resolves within 1.5 at 256 over the null's runs (64 left 10
of 64 runs undecided; job 20261102233337, paired)."""


def draw(seed: int):
    """A seed's two `[ROWS, HALF]` standard normal operands, float32, and its amplification 2^k."""
    rng = np.random.default_rng([0x1714, 2, seed])
    a, b = (rng.standard_normal((ROWS, HALF)).astype(np.float32) for _ in range(2))
    return a, b, np.float32(2.0 ** int(rng.integers(0, 25)))


def amplified(a, big):
    """`a`'s first element through `big` and back: exact in float64, its float32 rounding big's."""
    return jax.lax.optimization_barrier(a[:, :1] + big) - big


def cancelled(a, big):
    """`amplified` the other way, `big` less `a` taken from `big`: exact in float64, rounded alike."""
    return -(jax.lax.optimization_barrier(big - a[:, :1]) - big)


def reference(a, b, big):
    """Each row's amplified element beside the products of the rest."""
    return jnp.concatenate([amplified(a, big), a[:, 1:] * b[:, 1:]], axis=-1)


def reformulated(a, b, big):
    """The reference's own math reformulated: its amplified element cancelled the other way, its products'
    operands swapped."""
    return jnp.concatenate([cancelled(a, big), b[:, 1:] * a[:, 1:]], axis=-1)


def biased(a, b, big):
    """The reformulated arm (`reformulated`'s cancellation) with every product 8 units of its rounding off."""
    return jnp.concatenate([cancelled(a, big), (a[:, 1:] * b[:, 1:]) * jnp.float32(1 + 2.0 ** -21)],
                           axis=-1)


def failed(dew, label: str, against=reference) -> GateFailure:
    """The failure check 1 decides `dew` against `against` with conditioning, never an Inconclusive."""
    with pytest.raises(GateFailure) as raised:
        assert_as_exact(jax.jit(dew), jax.jit(against), draw, label, seeds=WEIGHED_SEEDS,
                        conditioning=CONDITIONED_SEEDS)
    assert not isinstance(raised.value, Inconclusive) and raised.value.findings
    return raised.value


def test_a_fault_on_the_elements_a_program_computes_well_is_decided_weighed():
    """A product eight units of its rounding off beside the amplified element, cancelled the other way,
    leaves the unweighted check undecided over its seeds, and the weighed check decides it a failure of its
    ratio and rows, under the weighed check's own label."""
    with pytest.raises(Inconclusive):
        assert_as_exact(jax.jit(biased), jax.jit(reference), draw, "a biased product, unweighted",
                        seeds=WEIGHED_SEEDS)
    failure = failed(biased, "a biased product beside an amplified value")
    light = [finding for finding in failure.findings if "['light']" in finding.leaf]
    assert light and {finding.rule for finding in light} <= {"pooled", "row"}
    assert all(finding.leaf.startswith("a biased product beside an amplified value, conditioned")
               for finding in failure.findings)


def test_an_arm_as_exact_as_the_reference_passes_conditioned():
    """The reference's own math, the amplified element cancelled the other way and the products' operands
    swapped, passes as a weighed site is judged."""
    assert_as_exact(jax.jit(reformulated), jax.jit(reference), draw,
                    "the products swapped, cancelled the other way", seeds=WEIGHED_SEEDS,
                    conditioning=CONDITIONED_SEEDS)


def test_an_error_moved_onto_the_amplified_element_fails_decided():
    """An arm that doubles the amplified element's rounding (through 2^(k+1)), its products the reference's,
    the reference's errors (M, 1) against its (2M, 1), fails decided: an error on the amplified element never
    hides behind the rest's."""
    def moved(a, b, big):
        return jnp.concatenate([amplified(a, 2 * big), a[:, 1:] * b[:, 1:]], axis=-1)

    failure = failed(moved, "an error moved onto the amplified element")
    heavy = [finding for finding in failure.findings if "['heavy']" in finding.leaf]
    # A row of one heavy value is held by the gate's factor for one (NARROW_FACTORS): the pooled check
    # decides.
    assert heavy and {finding.rule for finding in heavy} == {"pooled"}


def test_an_element_no_rounding_reaches_must_be_exact_in_every_arm():
    """An element no rounding of the reference's program reaches (`a` copied) must be exact in every arm: an
    arm one unit of its rounding off there fails decided."""
    def copied(a, b, big):
        return jnp.concatenate([a[:, :1], a[:, 1:] * b[:, 1:]], axis=-1)

    def nudged(a, b, big):
        return jnp.concatenate([a[:, :1] * jnp.float32(1 + 2.0 ** -23), a[:, 1:] * b[:, 1:]], axis=-1)

    assert_as_exact(jax.jit(copied), jax.jit(copied), draw, "a copy", seeds=WEIGHED_SEEDS,
                    conditioning=CONDITIONED_SEEDS)
    failed(nudged, "a copy nudged", against=copied)


def test_the_weighed_check_passes_the_reference_against_its_reformulation():
    """The null control, judged as a weighed site is (paired, as the corner sites run): the reference
    against its own math reformulated, the amplified element cancelled the other way and the products'
    operands swapped, on the same draws. Over `NULL_RUNS` runs no verdict the weighed check decides is red,
    and it decides at least `WEIGHED_DECIDED` of them, so the control is never vacuous. The unweighted
    check's own verdicts are not asserted here: its worst-row rule decides some of these heavy-tailed runs
    falsely, the central gate's row 3684, whose own null judges it."""
    verdicts = null_runs(jax.jit(reference), draw, "the reference against its reformulation, weighed",
                         seeds=WEIGHED_SEEDS, runs=NULL_RUNS, dew_fn=jax.jit(reformulated), paired=True,
                         conditioning=CONDITIONED_SEEDS)
    weighed = [verdict for verdict, _, _ in verdicts if verdict.endswith(", conditioned")]
    reds = sum(verdict != "pass, conditioned" for verdict in weighed)
    assert reds == 0, (reds, null_rate_bound(reds, len(weighed)), verdicts)
    assert len(weighed) >= WEIGHED_DECIDED, (len(weighed), verdicts)


def test_a_rounding_reaches_an_element_its_first_derivative_does_not():
    """((a + 2^20) - 2^20 - a)^2 is zero in float64 and 2^-20 or so in float32: its rounding's first
    derivative is zero there, and the scale, read by moving each rounding a rounding's size, is not."""
    def cancelled(a):
        big = jnp.asarray(2.0 ** 20, a.dtype)
        return jnp.square(jax.lax.optimization_barrier(a + big) - big - a)

    value, scale = rounding_scale(cancelled, np.full((4,), 1 + 2.0 ** -10, np.float32))
    assert np.all(value == 0) and np.all(scale > 0)


def test_a_rounding_behind_a_stop_gradient_reaches_its_output():
    """stop_gradient(a * b) holds its product's rounding, which the scale reads, a stop gradient moving
    nothing of a value's own."""
    a, b, _ = draw(0)
    _, scale = rounding_scale(lambda a, b: jax.lax.stop_gradient(a * b), a, b)
    assert np.all(scale > 0)


def test_an_integer_rounds_where_its_float_is_too_narrow_to_hold_it():
    """An int32 past 2^24 converts to float32 with rounding the scale reads; one below converts exactly,
    at no scale."""
    _, scale = rounding_scale(lambda n: n.astype(jnp.float32), np.asarray([2 ** 24 + 1, 3], np.int32))
    assert scale[0] > 0 and scale[1] == 0


def test_a_programs_integer_output_has_no_scale_and_its_float_one_does():
    """A program of a float and an integer output (a product and an argmax) scales its float alone."""
    a, b, _ = draw(0)
    _, (product, chosen) = rounding_scale(lambda a, b: (a * b, jnp.argmax(a, axis=-1)), a, b)
    assert chosen is None and np.all(product > 0)


def test_a_bfloat16_program_is_weighed_at_its_own_rounding():
    """A bfloat16 product's scale is bfloat16's rounding: of the same values, each element's 2^16 times
    float32's (their unit roundoffs' ratio), the taps drawn alike, but for the float64 rounding of each
    displaced value, at most 2^-53 of the product a draw, 2^-36 of it once scaled by 2^16 and doubled for
    both programs."""
    a, b, _ = draw(0)
    narrow_a, narrow_b = a.astype(jnp.bfloat16), b.astype(jnp.bfloat16)
    _, narrow = rounding_scale(lambda a, b: a * b, narrow_a, narrow_b)
    _, wide = rounding_scale(lambda a, b: a * b, narrow_a.astype(jnp.float32), narrow_b.astype(jnp.float32))
    product = np.abs(np.asarray(narrow_a, np.float64) * np.asarray(narrow_b, np.float64))
    assert np.all(np.abs(np.asarray(narrow, np.float64) - np.asarray(wide, np.float64) * 2.0 ** 16)
                  <= 2.0 ** -36 * product)


def test_a_reduce_precision_rounds_at_its_stated_significand():
    """A float32 value rounded to bfloat16's significand in place (`reduce_precision`) is weighed at that
    format's rounding, as a cast to bfloat16 and back is, not at its float32 storage's."""
    a, _, _ = draw(0)
    _, reduced = rounding_scale(lambda a: jax.lax.reduce_precision(a, exponent_bits=8, mantissa_bits=7), a)
    _, cast = rounding_scale(lambda a: a.astype(jnp.bfloat16).astype(jnp.float32), a)
    np.testing.assert_array_equal(reduced, cast)


def test_a_float16_cast_to_bfloat16_rounds():
    """A float16 value cast to bfloat16, a format of the same width but fewer significand bits, rounds: its
    scale is nonzero wherever the value is."""
    a, _, _ = draw(0)
    _, scale = rounding_scale(lambda a: a.astype(jnp.bfloat16), a.astype(jnp.float16))
    assert np.all((np.asarray(scale) > 0) == (a.astype(jnp.float16) != 0))


@pytest.mark.mesh(devices=2)
def test_a_sum_across_devices_is_weighed_as_a_sum_and_a_gather_as_a_move():
    """A program sharded over two devices sums each device's partial product across them (`psum`) and
    gathers an operand (`all_gather`): the device sum's scale is a sum's, over its partial sums' rounding
    and its own, and the gathered operand, a move of an input, rounds nothing."""
    from jax.sharding import PartitionSpec as P

    mesh = jax.make_mesh((2,), ("x",), devices=jax.devices()[:2])
    sharded = jax.shard_map(lambda a, b: (jax.lax.psum(a @ b, "x"), jax.lax.all_gather(a, "x", tiled=True,
                                                                                         axis=1)),
                            mesh=mesh, in_specs=(P(None, "x"), P("x", None)), out_specs=(P(), P()),
                            check_vma=False)
    a, b, _ = draw(0)
    placed = (jax.device_put(a, jax.NamedSharding(mesh, P(None, "x"))),
              jax.device_put(b.T, jax.NamedSharding(mesh, P("x", None))))
    _, (summed, gathered) = rounding_scale(sharded, *placed)
    assert np.all(summed > 0) and np.all(gathered == 0)


def test_a_sharded_weighed_site_judges_its_weighed_check_from_its_own_map_tasks(monkeypatch, tmp_path):
    """A weighed site run as CI's map and judge: map tasks over the plan's units (tools/armada/ci.py
    `draws_of`), its unweighted units first, then its weighed check's past them, each task drawing and
    recording only its own check's units, so each check's tasks cost its own draws; its judge, drawing
    nothing, defers to the weighed check where the unweighted stays undecided and decides it from the
    weighed tasks' records, as a run that drew every unit itself does: the biased product fails on its light
    class, the reformulation passes."""
    from map_judge import DRAWS, POOL, RECORDS

    plain, weighed = 5 * WEIGHED_SEEDS, 5 * CONDITIONED_SEEDS

    for name, dew in (("biased", biased), ("reformulated", reformulated)):
        label = f"a sharded weighed site, {name}"
        for start in [*range(0, plain, 16), *range(plain, plain + weighed, 320)]:
            stop = min(start + (16 if start < plain else 320), plain if start < plain else plain + weighed)
            monkeypatch.setenv(DRAWS, f"{start}:{stop}")
            task = tmp_path / name / f"task {start}"
            monkeypatch.setenv(RECORDS, str(task))
            assert_as_exact(jax.jit(dew), jax.jit(reference), draw, label, seeds=WEIGHED_SEEDS,
                            conditioning=CONDITIONED_SEEDS)
            # A task draws its own check's units alone: one record, the other check's none.
            assert len(list(task.glob("*.pkl"))) == 1
        monkeypatch.delenv(DRAWS)
        monkeypatch.delenv(RECORDS)
        monkeypatch.setenv(POOL, str(tmp_path / name))
        judged = jax.jit(dew), jax.jit(reference), draw, label
        if name == "biased":
            with pytest.raises(GateFailure) as raised:
                assert_as_exact(*judged, seeds=WEIGHED_SEEDS, conditioning=CONDITIONED_SEEDS)
            assert not isinstance(raised.value, Inconclusive)
            assert any("['light']" in finding.leaf and finding.leaf.startswith(f"{label}, conditioned")
                       for finding in raised.value.findings)
        else:
            assert_as_exact(*judged, seeds=WEIGHED_SEEDS, conditioning=CONDITIONED_SEEDS)
        monkeypatch.delenv(POOL)
