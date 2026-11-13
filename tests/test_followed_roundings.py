"""Check 1 of a quantized computation (`assert_as_exact`'s `followed`, docs/design/testing.md): each arm held
to the float64 computation that rounds where that arm rounded, every rounding's input a leaf of its own.

The site here is the shape of a quantization-aware layer: a product, its result rounded to a coarse grid (8
bits of significand, bfloat16's), and a second product of the rounded values. Two float32 programs of it (the
first product summed whole, or in two halves of its depth) round about one of its 16,384 values apart a draw,
each a quantum of the grid, which no float64 truth that rounds for itself can hold both arms to."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from arm_arithmetic import summed_as_run
from reference_error import (
    Followed,
    GateFailure,
    Inconclusive,
    Rounded,
    UnrecordedTruth,
    UnroundedDecisions,
    assert_as_exact,
    indexed_recording,
    widened,
)

ROWS, DEPTH, WIDTH, OUT = 64, 256, 256, 64

SEEDS = 8
"""The sites' declared draws, sized by `seeds_for(..., followed=...)` paired on armada."""


def draw(seed: int):
    """A draw of the site: its inputs and its two weights, each at its fan-in's scale."""
    rng = np.random.default_rng([0x5EED, seed])
    return (jnp.asarray(rng.standard_normal((ROWS, DEPTH)), jnp.float32),
            jnp.asarray(rng.standard_normal((DEPTH, WIDTH)) / np.sqrt(DEPTH), jnp.float32),
            jnp.asarray(rng.standard_normal((WIDTH, OUT)) / np.sqrt(WIDTH), jnp.float32))


def rounded(values):
    """The site's rounding: to 8 significant bits, to nearest even (`lax.reduce_precision`)."""
    return jax.lax.reduce_precision(values, exponent_bits=8, mantissa_bits=7)


def layer(first, *, split: bool = False, decided=None):
    """The site's float32 arm: `first(x, w1)` its first product (`split`: summed in two halves of its depth),
    rounded, then its second product; with `decided`, its decisions are `decided` of its rounded values
    instead, a fault in its rounding. Its leaves are its output and the values it rounded."""
    def run(x, w1, w2):
        half = DEPTH // 2
        h = (first(x[:, :half], w1[:half]) + first(x[:, half:], w1[half:])) if split else first(x, w1)
        q = rounded(h) if decided is None else decided(rounded(h))
        return Rounded({"output": q @ w2, "rounded": h}, {"q": q})
    return run


def followed_truth(decisions, x, w1, w2):
    """The site's float64 computation rounding where an arm rounded: its rounded values the arm's
    `decisions`, every other value its own."""
    return {"output": jnp.asarray(decisions["q"], jnp.float64) @ w2, "rounded": x @ w1}


def rounds(judged):
    """The site's rounding of the values an arm rounded, its judged leaf `rounded`."""
    return {"q": rounded(judged["rounded"])}


followed = Followed(followed_truth, rounds)
"""The site's truth, each arm's own, and its rounding of the values an arm rounded."""


def product(x, w):
    return x @ w


def decided_apart(one, other, seeds: int) -> int:
    """How many of `seeds` draws two arms round some value of otherwise, each compiled and run as the gate
    compiles and runs a site's arms (`reference_error._arms`, `_ynn_decided`, `summed_as_run`)."""
    import reference_error

    first, second = reference_error._arms(one, other)
    with summed_as_run():
        reference_error._ynn_decided("decided apart", [first, second], draw(0))
        return sum(not np.array_equal(*(np.asarray(arm(*draw(seed)).decisions["q"])
                                        for arm in (first, second))) for seed in range(seeds))


def test_arms_that_round_apart_pass_against_the_truths_of_their_own_roundings():
    """The null: Dew's arm summing its first product in two halves of its depth, the reference whole, round
    some value otherwise on some of the declared draws, and pass, each held to the truth that rounds where
    it rounded; a further reference, rounding as the first does, follows its own decisions alike."""
    halves, whole = layer(product, split=True), layer(product)
    assert decided_apart(halves, whole, SEEDS) > 0
    assert_as_exact(halves, whole, draw, "rounded apart", seeds=SEEDS, followed=followed,
                    also={"whole again": layer(product)})


def moved(q):
    """`q` four quanta up `rounded`'s grid (bfloat16's)."""
    for _ in range(4):
        q = jnp.nextafter(q.astype(jnp.bfloat16), jnp.asarray(jnp.inf, jnp.bfloat16)).astype(q.dtype)
    return q


FAULTS = {"zeroed": jnp.zeros_like, "moved four quanta": moved,
          "of another dtype": lambda q: q.astype(jnp.bfloat16)}
"""Decisions an arm's rounding did not make, equal in value or not: their bits are not its rounding's."""


@pytest.mark.parametrize("fault", FAULTS)
def test_a_decision_its_rounding_did_not_make_is_refused(fault):
    """A decision is followed only where the arm's rounding made it: an arm whose decisions are zeroed,
    moved four quanta up the grid, or held at another dtype, computes every other value as the reference
    does and its truth would follow it exactly, so the gate holds each decision, in structure, shape, dtype
    and bits, to the site's rounding of the values the arm rounded, its judged leaf (`Followed.rounds`),
    and refuses it."""
    with pytest.raises(UnroundedDecisions):
        assert_as_exact(layer(product, decided=FAULTS[fault]), layer(product), draw, f"a decision {fault}",
                        seeds=SEEDS, followed=followed)


def test_values_rounded_that_are_not_the_arms_are_charged():
    """The values a decision is held to are the arm's judged leaf, which check 1 holds to the truth: an arm
    stating it rounded zeros, and so deciding zeros, is legal to the site's rounding and charged on the
    values it rounded."""
    def forged(x, w1, w2):
        h = jnp.zeros((x.shape[0], w1.shape[1]), x.dtype)
        return Rounded({"output": rounded(h) @ w2, "rounded": h}, {"q": rounded(h)})

    with pytest.raises(GateFailure) as failed:
        assert_as_exact(forged, layer(product), draw, "forged values rounded", seeds=SEEDS, followed=followed)
    assert not isinstance(failed.value, (Inconclusive, UnroundedDecisions))


def test_a_live_arm_marked_as_a_recording_is_held_as_any_live_arm():
    """A recording's decisions are the reference's own (`indexed_recording`'s arm); a live reference that
    marks itself as recorded is still a live arm, its zeroed decisions refused."""
    zeroed = layer(product, decided=jnp.zeros_like)
    zeroed.recorded = True
    with pytest.raises(UnroundedDecisions):
        assert_as_exact(layer(product), zeroed, draw, "a live arm marked recorded", seeds=SEEDS,
                        followed=followed)


def test_an_error_before_a_rounding_is_charged_though_its_decision_is_followed():
    """The values a computation rounds are leaves of their own: an arm whose first product is sixteen
    float32 units off (a relative 2**-20) fails on them, though its truth follows whatever its rounding
    decided from them."""
    def off(x, w):
        return (x @ w) * (1 + 2.0 ** -20)

    with pytest.raises(GateFailure) as failed:
        assert_as_exact(layer(off), layer(product), draw, "off before rounding", seeds=SEEDS,
                        followed=followed)
    assert not isinstance(failed.value, Inconclusive)
    assert {finding.leaf.rpartition("[")[2] for finding in failed.value.findings} == {"'rounded']"}


def test_a_float64_path_that_rounds_for_itself_is_no_recorded_followed_truth():
    """A recorded reference's truth followed its own decisions, which the recording keeps
    (`record_case(..., followed=True)`): a float64 path that follows them is held to the recorded truth at
    every seed and judges Dew's arm by its own; one that rounds for itself is refused on the first draw
    whose roundings it decides otherwise."""
    from tool_support import load

    recorder = load("recorded_references")
    reference = layer(product)

    def drawn(seed):
        return dict(zip(("x", "w1", "w2"), (np.asarray(value) for value in draw(seed)), strict=True))

    def evaluate(inputs):
        ran = reference(*(jnp.asarray(inputs[name]) for name in ("x", "w1", "w2")))
        with jax.enable_x64(new_val=True):
            truth = followed_truth(ran.decisions, *(jnp.asarray(inputs[name], jnp.float64)
                                                     for name in ("x", "w1", "w2")))
        return (jax.tree.map(np.asarray, ran.judged), jax.tree.map(np.asarray, truth),
                jax.tree.map(np.asarray, ran.decisions))

    recorded = recorder.record_case("site", drawn, evaluate, summary=True, store_inputs=False, followed=True)
    seeds = int(recorded["site.seeds"])

    def by_name(fn):
        return lambda decisions, inputs: fn(decisions, *(inputs[name] for name in ("x", "w1", "w2")))

    draws, recorded_reference, truth = indexed_recording(recorded, "site", seeds, inputs_of=drawn,
                                                         truth_fn=by_name(followed_truth))
    dew = layer(product, split=True)
    assert_as_exact(lambda index, inputs: dew(*(jnp.asarray(inputs[name]) for name in ("x", "w1", "w2"))),
                    recorded_reference, draws, "recorded and followed", seeds=seeds,
                    followed=Followed(truth, rounds))

    def for_itself(decisions, x, w1, w2):
        return {"output": rounded(x @ w1) @ w2, "rounded": x @ w1}

    def own_rounding(seed):
        x, w1, _ = draw(seed)
        with jax.enable_x64(new_val=True):
            return np.asarray(rounded(jnp.asarray(x, jnp.float64) @ jnp.asarray(w1, jnp.float64)))

    flipped = next(seed for seed in range(seeds) if not np.array_equal(
        np.asarray(reference(*draw(seed)).decisions["q"], np.float64), own_rounding(seed)))
    _, _, own = indexed_recording(recorded, "site", seeds, inputs_of=drawn, truth_fn=by_name(for_itself))
    with jax.enable_x64(new_val=True), pytest.raises(UnrecordedTruth):
        own(jax.device_get(reference(*draw(flipped)).decisions), np.int32(flipped), widened(drawn(flipped)))


def near_one(seed: int):
    """A draw whose rounded values all sit within a quarter quantum of 1 (`rounded`'s grid there is 2**-7
    apart), so every arm that rounds for itself rounds each to 1, and a second product's weights."""
    rng = np.random.default_rng([0x0E1, seed])
    return (jnp.asarray(1 + rng.uniform(-2.0 ** -9, 2.0 ** -9, (ROWS, WIDTH)), jnp.float32),
            jnp.asarray(rng.standard_normal((WIDTH, OUT)) / np.sqrt(WIDTH), jnp.float32))


def excess(*, raised: bool):
    """A site whose leaf `excess` is its rounded values less 1, the rest as `layer`'s: rounding for itself,
    every value is 1 and the leaf zero; `raised`, its first row's values are decided one quantum up, the
    grid's next value, so that row of the leaf is not zero."""
    def run(h, w2):
        q = rounded(h)
        if raised:
            up = jnp.nextafter(q[0].astype(jnp.bfloat16), jnp.asarray(jnp.inf, jnp.bfloat16)).astype(q.dtype)
            q = q.at[0].set(up)
        return Rounded({"excess": q - 1, "output": q @ w2, "rounded": h}, {"q": q})
    return run


def followed_excess(decisions, h, w2):
    q = jnp.asarray(decisions["q"], jnp.float64)
    return {"excess": q - 1, "output": q @ w2, "rounded": h}


def test_a_zero_row_one_arms_decisions_make_is_no_row_the_recording_must_hold():
    """Rows pair only where two arms' truths are the same values: Dew's arm, every value rounded to 1, has
    a truth whose `excess` is zero on every row; the recorded reference decided its first row a quantum up,
    so its truth's row is not (a recording's decisions are the reference's own, held by its recorded
    truth). Each arm is exact against its own truth there, and the site passes; holding Dew's zero rows to
    the recording's, as a recorded leaf's rows otherwise are, refused it."""
    from tool_support import load

    recorder = load("recorded_references")
    reference = excess(raised=True)

    def drawn(seed):
        return dict(zip(("h", "w2"), (np.asarray(value) for value in near_one(seed)), strict=True))

    def evaluate(inputs):
        ran = reference(*(jnp.asarray(inputs[name]) for name in ("h", "w2")))
        with jax.enable_x64(new_val=True):
            truth = followed_excess(ran.decisions, *(jnp.asarray(inputs[name], jnp.float64)
                                                     for name in ("h", "w2")))
        return (jax.tree.map(np.asarray, ran.judged), jax.tree.map(np.asarray, truth),
                jax.tree.map(np.asarray, ran.decisions))

    recorded = recorder.record_case("raised", drawn, evaluate, summary=True, store_inputs=False,
                                    followed=True)
    seeds = int(recorded["raised.seeds"])
    draws, recorded_reference, truth = indexed_recording(
        recorded, "raised", seeds, inputs_of=drawn,
        truth_fn=lambda decisions, inputs: followed_excess(decisions, inputs["h"], inputs["w2"]))
    dew = excess(raised=False)
    assert_as_exact(lambda index, inputs: dew(jnp.asarray(inputs["h"]), jnp.asarray(inputs["w2"])),
                    recorded_reference, draws, "a zero row of Dew's decisions", seeds=seeds,
                    followed=Followed(truth, rounds))

