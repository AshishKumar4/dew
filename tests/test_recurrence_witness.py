"""The recurrence step witness (tests/recurrence_witness.py) on the forms and geometries its native cells
take (tests/test_native_certificates.py: two value heads on a key head, on mma.sync's bundle and warpgroup
MMA's), interpreted: each record the launches keep inside its step's enclosure (rows 5412, 5421)."""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import recurrence_witness as rw
from interval_certificates import point, staged, within
from op_certificates import FLOAT32, exp2_rounded

from dew.nn.kernels import generation
from dew.nn.kernels.templates import product

TARGETS = ("sm80", "sm90")
"""The native cells' bundles: mma.sync's (an A100's) and warpgroup MMA's (an H100's)."""


def test_a_decay_encloses_every_fraction_the_algebra_raises():
    """A chunk's decays as `algebra.raised` forms them, an integer power of two times the exp2 of an fp32
    fraction correctly rounded, rounded to fp32 as the scaling's last product rounds it (subnormals kept,
    `power2`), lie inside the witness's enclosure (`rw.raised_exponent`) at prefixes down to -192: taken as
    one fp32 operand of exp2, the exponent's grid there excluded 778 of the 1024 legal decays."""
    decays = -np.random.default_rng(2084).uniform(0, 3, (64, 16)).astype(np.float32)
    exponents = rw.Exponents.of(decays)
    whole = np.floor(exponents.exact)
    fraction = (exponents.exact - whole).astype(np.float32)
    legal = np.ldexp(exp2_rounded(fraction, FLOAT32), whole.astype(np.int32)).astype(np.float32)
    enclosed = exponents.raised(rw.interpreted_lane())
    assert np.all(within(legal, enclosed)), int(np.sum(~within(legal, enclosed)))


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("delta", [True, False], ids=["delta", "keys gated"])
def test_a_grouped_backwards_records_lie_in_their_key_heads_enclosures(target, delta):
    """Two value heads on a key head: the fast backward's queries' and keys' gradients, and the scores'
    without the delta rule, are the key head's (each value head's summed in fp32 where it leaves,
    `mainloop_backward._query_key`); each record inside the sum of its value heads' enclosures, every other
    a value head's own (`rw.back_witnessed`)."""
    with generation.assumed(target, interpreted=True):
        back = rw.recorded_backward(rw.draw(0, "mild", heads=2), delta=delta)
        held = rw.back_witnessed(back, rw.interpreted_lane())
    for name, inside in held.items():
        assert np.all(inside), (name, float(np.mean(inside)))


def test_an_fp32_forward_the_template_does_not_train_is_witnessed_as_its_inference_forward():
    """An fp32 call on mma.sync's bundle, whose fast backward holds no A100 thread (264 to 328 registers):
    a training step takes another kernel, so the witness holds the template's inference forward
    (`template.Mosaic.trains`), every record inside its step's enclosure."""
    with generation.assumed("sm80", interpreted=True):
        run = rw.recorded(*rw.draw(0, "mild", heads=2, dtype=jnp.float32))
        held = rw.witness(run, rw.interpreted_lane())
    for name, inside in held.items():
        assert np.all(inside), (name, float(np.mean(inside)))


@pytest.mark.parametrize("target", (*TARGETS, "sm100"))
def test_an_inference_forward_is_witnessed_at_its_own_tiles(target):
    """A bf16 call no gradient follows, with the delta rule and without it (its keys taking the write gate),
    its forward at its own tiles (`Tiles.forward`) as its states then output launches, no walk forming the
    output in one (rows 5668, 7877): every record inside its step's enclosure (`rw.witness`). Its records
    are the keeping build's at those tiles, its output and final memory the inference forward's bit for bit
    (`rw.recorded`'s); a tcgen05 call's (an sm100's) solve fused with its application, which stores no
    inverse where nothing is kept (`mainloop._solved`'s `keeps`), so its inverse is the keeping build's."""
    with generation.assumed(target, interpreted=True):
        rw.assert_inference_witnessed(rw.interpreted_lane(), heads=2)


def test_two_bf16_parts_of_a_value_sum_within_its_staged_reach():
    """fp32 values split into two bf16 parts (`product.formed`, a 16-bit call's doubling merges' operands in
    fp32, `mainloop.merge_parts`) sum within the witness's reach of the value
    (`interval_certificates.staged`), subnormals among them, whose second part rounds below bf16's normals at
    an absolute step, or flushes on a lane that flushes (this CPU's): a value near 1e-40 dropped 2^-134
    where the reach held none, and Kimi's steep inverse lay one bf16 step outside its enclosure on a GH200
    (row 10007)."""
    rng = np.random.default_rng(0)
    x = np.concatenate([np.float32([1.0e-40, -6.43e-40, 1.956e-38, 2.5e-38, -1.4e-45, 0.0, 1.0, -3.7e-3]),
                        rng.standard_normal(256).astype(np.float32) * np.float32(2.0) ** rng.integers(
                            -149, 20, 256).astype(np.float32)])
    total = sum(np.asarray(part, np.float64) for part in product.formed(jnp.asarray(x), 2))
    found = staged(point(x), dataclasses.replace(rw.interpreted_lane(), held=2, wide=True))
    assert np.all((found.lo <= total) & (total <= found.hi)), x[(total < found.lo) | (total > found.hi)]


def test_a_key_decays_solve_is_enclosed_as_its_own_bundle_rounds_it():
    """An H100's Kimi delta attention solve issues by the intra walk's bundle (`intra.sub_chunked`:
    mma.sync's, one TF32 part rounded to nearest, a tie away), not its other launches' warpgroup MMA's two
    bf16 parts: a system whose one pair 0.9979 rounds to TF32 0.998046875, a bf16 midpoint stored as 1.0.
    The solve's lane (`Lane.solving`) encloses that inverse; the call's lane, taken for the solve, rounds
    0.9979 to bf16 0.99609375 and does not (head review of a61e73fc4)."""
    lane = dataclasses.replace(rw.interpreted_lane(), held=1, merged=2, tf32="read", solve=(1, "rna"))
    pairs = np.zeros((rw.SUB, rw.SUB))
    pairs[1, 0] = np.float32(0.9979)
    system = point(pairs)

    def stored(solving):
        return staged(rw.solved_blocks(system, solving), lane)

    native = np.full(pairs.shape, -1.0)
    assert within(native, stored(lane.solving()))[1, 0]
    assert not within(native, stored(lane))[1, 0]


def test_the_decay_prefix_sums_its_terms_by_the_one_compensated_reduction():
    """The decay's base-2 prefix (`recurrence_algebra.prefix`), its terms fenced once and summed by the one
    compensated pair reduction's combine (`renormalized_add`), is, bit for bit, the sum whose combine fenced
    its operands at every step (`compensated_add`) over drawn decays of 1e-3 to 30 a token, jitted (row
    10722): one way of the same sum."""
    from dew.nn.kernels.recurrence_algebra import LOG2_E, prefix
    from dew.nn.precision import compensated_add, two_product

    rng = np.random.default_rng(0x10722)
    g = -jnp.asarray(np.exp(rng.uniform(np.log(1e-3), np.log(30.0), (3, 64, 4))), jnp.float32)

    def fenced_each_step(g):
        # The prefix as it was summed before (its parent's arithmetic, its combine fencing each step).
        scale = np.float32(LOG2_E)
        scaled, error = two_product(g, jnp.full_like(g, scale), bit_split=True)
        error = error + g * float(np.float32(LOG2_E - float(scale)))
        size = g.shape[-2]
        zero = jnp.zeros((), g.dtype)
        return jax.lax.reduce_window((scaled, error), (zero, zero), compensated_add, (1, size, 1), (1, 1, 1),
                                     ((0, 0), (size - 1, 0), (0, 0)))

    for got, wanted in zip(jax.jit(prefix)(g), jax.jit(fenced_each_step)(g), strict=True):
        np.testing.assert_array_equal(np.asarray(got), np.asarray(wanted))
