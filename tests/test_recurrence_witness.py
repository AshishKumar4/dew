"""The recurrence step witness (tests/recurrence_witness.py) on the forms and geometries its native cells
take (tests/test_native_certificates.py: two value heads on a key head, on mma.sync's bundle and warpgroup
MMA's), interpreted: each record the launches keep inside its step's enclosure (rows 5412, 5421)."""

import jax.numpy as jnp
import numpy as np
import pytest
import recurrence_witness as rw
from interval_certificates import within
from op_certificates import FLOAT32, exp2_rounded

from dew.nn.kernels import generation

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


def test_an_amperes_inference_forward_is_witnessed_as_its_states_and_output_launches():
    """On mma.sync's bundle an inference forward runs its states and output launches, no fused walk
    (`chunk_forward.walks_once`): the walk witness holds the records those launches keep."""
    with generation.assumed("sm80", interpreted=True):
        rw.assert_walk_witnessed(rw.interpreted_lane(), heads=2)


def test_an_fp32_forward_the_template_does_not_train_is_witnessed_as_its_inference_forward():
    """An fp32 call on mma.sync's bundle, whose fast backward holds no A100 thread (264 to 328 registers):
    a training step takes another kernel, so the witness holds the template's inference forward
    (`template.Mosaic.trains`), every record inside its step's enclosure."""
    with generation.assumed("sm80", interpreted=True):
        run = rw.recorded(*rw.draw(0, "mild", heads=2, dtype=jnp.float32))
        held = rw.witness(run, rw.interpreted_lane())
    for name, inside in held.items():
        assert np.all(inside), (name, float(np.mean(inside)))
