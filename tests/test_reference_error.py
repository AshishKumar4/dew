"""The float64 accuracy rule (tests/reference_error.py) on comparisons whose answer is known: what it
must refuse, and the rounding it must admit."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from reference_error import (
    PARITY,
    REDRAW,
    Inconclusive,
    assert_as_exact,
    assert_as_exact_as_the_reference,
    declared_seeds,
    distance,
    indexed_recording,
    input_digest,
    seeds_for,
    widened,
)


def test_a_small_row_beside_large_ones_is_held_to_its_own_size():
    """Dew doubles a row of 5s beside rows of 1e8 that the reference rounds by 8. Every
    tensor-wide distance passes it: its RMS error, 3.5, sits under PARITY times the reference's,
    5.7. Held to its own size, the row is 100% wrong against the reference's 8e-8."""
    truth = np.array([[1e8, 1e8], [5.0, 5.0]])
    reference = np.array([[1e8 + 8, 1e8 + 8], [5.0, 5.0]])
    dew = np.array([[1e8, 1e8], [10.0, 10.0]])
    assert distance(dew, truth) <= PARITY * distance(reference, truth)
    with pytest.raises(AssertionError):
        assert_as_exact_as_the_reference(dew, reference, truth, "small row")


def test_a_row_that_cancels_is_held_to_what_it_sums():
    """A query that keeps one key has a zero gradient: its terms, about 1 each, cancel. Both
    sides round them by about a float32 unit, which over the row's near-zero float64 value is a
    ratio near 1e10 and over its terms (`float64_terms`) one unit, so Dew rounding as the
    reference does passes. Dew ten times further in that row alone moves the RMS over 512 rows
    by 9%, which the tensor-wide rule admits; the row, held to its terms, does not pass."""
    rng = np.random.default_rng(0)
    truth = rng.normal(size=(512, 64))
    truth[7] = 1e-17
    terms = np.abs(truth)
    terms[7] = 1.0
    unit = float(np.finfo(np.float32).eps)

    def rounded(cancelling=1.0):
        noise = unit * terms * rng.choice([-1.0, 1.0], size=truth.shape)
        noise[7] *= cancelling
        return truth + noise

    reference, further = rounded(), rounded(10.0)
    assert_as_exact_as_the_reference(rounded(), reference, truth, "cancelling row", terms=terms)
    assert distance(further, truth) <= PARITY * distance(reference, truth)
    with pytest.raises(AssertionError):
        assert_as_exact_as_the_reference(further, reference, truth, "cancelling row", terms=terms)


def test_a_row_of_float64_zero_is_held_to_the_references_own_row():
    """A sum that cancels exactly before anything its terms see, as a query that keeps one key
    has a zero gradient through the flash formula's delta, has no size of its own: its distance
    is held to the reference's on that same row. Dew at the reference's residue passes; Dew with a
    residue where the reference computes the float64 zero exactly does not, though the tensor-wide
    RMS admits it."""
    rng = np.random.default_rng(1)
    truth = rng.normal(size=(64, 16))
    truth[0] = 0.0
    unit = float(np.finfo(np.float32).eps)
    reference = truth + unit * rng.normal(size=truth.shape)
    dew = truth + unit * rng.normal(size=truth.shape)
    assert_as_exact_as_the_reference(dew, reference, truth, "zero row")
    reference[0], dew[0] = 0.0, unit
    assert distance(dew, truth) <= PARITY * distance(reference, truth)
    with pytest.raises(AssertionError):
        assert_as_exact_as_the_reference(dew, reference, truth, "zero row")


def _draw(seed):
    """A product's operands, fresh for each seed and the same on every run."""
    rng = np.random.default_rng(seed)
    return rng.normal(size=(64, 256)).astype(np.float32), rng.normal(size=(256, 32)).astype(np.float32)


@jax.jit
def _product(x, w):
    return {"y": jnp.dot(x, w, precision="highest")}


@jax.jit
def _reordered(x, w):
    """The same sum in another order: its terms reversed, so it rounds at other partial sums."""
    return {"y": jnp.dot(x[:, ::-1], w[::-1], precision="highest")}


@jax.jit
def _lost_term(x, w):
    return {"y": jnp.dot(x[:, :-1], w[:-1], precision="highest")}


@jax.jit
def _bf16_operands(x, w):
    return {"y": jnp.dot(x.astype(jnp.bfloat16), w.astype(jnp.bfloat16),
                         preferred_element_type=jnp.float32)}


def test_check_one_passes_an_equally_exact_sum_over_the_seeds_its_own_reference_resolves():
    """The product summed in reverse rounds as exactly as the reference, by other partial sums: over the
    seeds the reference compared with itself needs (`seeds_for`), its pooled RMS error sits within
    PARITY of the reference's, every run alike."""
    seeds = seeds_for(_product, _draw, "product")
    assert_as_exact(_reordered, _product, _draw, "reordered", seeds=seeds)
    assert_as_exact(_reordered, _product, _draw, "reordered", seeds=seeds)


@pytest.mark.parametrize("wrong", [_lost_term, _bf16_operands], ids=["lost term", "bf16 operands"])
def test_check_one_fails_a_sum_that_differs_rather_than_rounds(wrong):
    """A lost term, and operands rounded to bf16, put the pooled ratio far past PARITY, and drawing
    REDRAW times as many seeds again does not bring it back."""
    with pytest.raises(AssertionError) as raised:
        assert_as_exact(wrong, _product, _draw, "wrong", seeds=seeds_for(_product, _draw, "product"))
    assert raised.type is AssertionError


def test_check_one_is_inconclusive_where_its_seeds_cannot_resolve_the_factor():
    """Errors set seed by seed, alike on every host: Dew's squared error is 4 on even seeds and 0.01 on
    odd ones, the reference's 1 on each, so the pooled ratio (1.42) is under PARITY but its interval
    over 8 seeds straddles it and is wider than it: inconclusive, never a pass. Fewer seeds than
    FEWEST_SEEDS are inconclusive whatever they read."""
    def errors(seed):
        return (np.zeros(64, np.float32), np.full(64, 2.0 if seed % 2 == 0 else 0.1, np.float32),
                np.ones(64, np.float32))

    def dew(x, mine, theirs):
        return {"y": x + mine}

    def reference(x, mine, theirs):
        return {"y": x + theirs}

    def truth(x, mine, theirs):
        return {"y": x}

    with pytest.raises(Inconclusive):
        assert_as_exact(dew, reference, errors, "straddling", seeds=8, truth_fn=truth)
    with pytest.raises(Inconclusive):
        assert_as_exact(_reordered, _product, _draw, "two seeds", seeds=2)
    # And seeds_for never declares fewer, whatever it is asked to start from.
    assert seeds_for(_product, _draw, "product", start=1) >= 8


def test_check_one_holds_dew_exact_where_the_reference_is_exact_on_every_seed():
    """Integer-valued operands make the reference's product exact on every seed: Dew must be too."""
    def integral(seed):
        rng = np.random.default_rng(seed)
        return (rng.integers(-4, 5, (16, 8)).astype(np.float32),
                rng.integers(-4, 5, (8, 4)).astype(np.float32))

    shifted = jax.jit(lambda x, w: {"y": jnp.dot(x, w) + 2.0 ** -20})
    exact = jax.jit(lambda x, w: {"y": jnp.dot(x, w)})
    with pytest.raises(AssertionError) as raised:
        assert_as_exact(shifted, exact, integral, "shifted", seeds=8)
    assert raised.type is AssertionError


@pytest.mark.parametrize("rank", [1, 2])
def test_check_one_fails_a_value_that_is_not_finite(rank):
    """A NaN in Dew's leaf fails, whatever the pooled errors would read, a vector's (which has no rows
    to catch it) as a matrix's."""
    def shaped(fn):
        return jax.jit(lambda x, w: {"y": fn(x, w)["y"].reshape(-1) if rank == 1 else fn(x, w)["y"]})

    poisoned = shaped(lambda x, w: {"y": jnp.dot(x, w, precision="highest").at[0, 0].set(jnp.nan)})
    with pytest.raises(AssertionError) as raised:
        assert_as_exact(poisoned, shaped(_product), _draw, "poisoned", seeds=8)
    assert raised.type is AssertionError


def test_check_one_holds_a_row_of_float64_zero_to_the_references_own():
    """A row the float64 truth holds at exactly zero, the reference too, is no further from zero in Dew
    than ROW_FACTOR times the reference on that row: a corruption too small to move the pooled error
    still fails there."""
    def zero_row(offset):
        return jax.jit(lambda x, w: {"y": jnp.concatenate(
            [jnp.dot(x, w, precision="highest"), jnp.full((1, w.shape[1]), offset, jnp.float32)])})

    with pytest.raises(AssertionError) as raised:
        assert_as_exact(zero_row(1e-12), zero_row(0.0), _draw, "zero row", seeds=8)
    assert raised.type is AssertionError


def test_a_recorded_reference_is_drawn_by_its_index_and_covers_the_redraw(tmp_path):
    """A reference recorded over (1 + REDRAW)·K seeds: each draw reads its seed's inputs, the reference
    and truth read their recorded outputs at it, and check 1 judges Dew against them; a recording too
    short for the redraw is refused before anything is judged."""
    rng = np.random.default_rng(9)
    count, seeds = 40, 8
    x = rng.normal(size=(count, 32)).astype(np.float32)
    # x·x - 1, rounded twice at fp32 alike by NumPy and XLA (XLA's tanh is about 1.6 times libm's error
    # on a CPU, so a transcendental would compare the two libraries, not the recording).
    truth = x.astype(np.float64) ** 2 - 1
    path = tmp_path / "recorded.npz"
    np.savez(path, **{"site.input.x": x, "site.reference.y": x * x - np.float32(1),
                      "site.truth.y": truth, "site.seeds": np.int64(seeds),
                      "site.redraw": np.int64(REDRAW)})
    assert declared_seeds(path) == {"site": seeds}
    with np.load(path) as stored:
        draw, reference, truth_fn = indexed_recording(dict(stored), "site", seeds)
        squared = jax.jit(lambda index, inputs: {"y": inputs["x"] * inputs["x"] - 1})
        assert_as_exact(squared, reference, draw, "recorded", seeds=seeds, truth_fn=truth_fn)
        with pytest.raises(AssertionError):
            indexed_recording(dict(stored), "site", 16)
        for narrow in (np.float32, jnp.bfloat16):
            with pytest.raises(AssertionError):
                indexed_recording({**stored, "site.truth.y": truth.astype(narrow)}, "site", seeds)


def test_a_recording_declares_the_count_its_recorded_reference_resolves(tmp_path):
    """tools/recorded_references.py records a reference over the seeds its own recording resolves, and
    their redraw; check 1 then judges Dew against the recording by its index."""
    from tool_support import load

    recorder = load("recorded_references")

    def draw(seed):
        return {"x": np.random.default_rng(100 + seed).normal(size=(64,)).astype(np.float32)}

    def evaluate(inputs):
        x = inputs["x"]
        return {"y": x * x - np.float32(1)}, {"y": x.astype(np.float64) ** 2 - 1}

    def squared(inputs):
        return {"y": inputs["x"] * inputs["x"] - 1}

    recorded = recorder.record_case("site", draw, evaluate, squared)
    seeds = int(recorded["site.seeds"])
    assert seeds >= 8 and len(recorded["site.input.x"]) == 5 * seeds
    path = tmp_path / "recorded.npz"
    np.savez(path, **recorded)
    assert declared_seeds(path) == {"site": seeds}
    with np.load(path) as stored:
        draw_recorded, reference, truth = indexed_recording(dict(stored), "site", seeds)
        assert_as_exact(jax.jit(lambda index, inputs: squared(inputs)), reference, draw_recorded, "recorded",
                        seeds=seeds, truth_fn=truth)


def test_check_one_holds_an_empty_leaf_exact_by_construction():
    """A tree with an empty leaf beside a real one (a one-tap convolution's history): the empty leaf adds
    no error and resolves nothing, so the tree is judged by its real leaf; an empty leaf whose shape or
    dtype is not the reference's fails."""
    def tree(fn):
        return jax.jit(lambda x, w: {"y": fn(x, w)["y"], "history": jnp.zeros((16, 0, 48), jnp.float32)})

    assert_as_exact(tree(_reordered), tree(_product), _draw, "with an empty leaf",
                    seeds=seeds_for(tree(_product), _draw, "with an empty leaf"))
    narrower = jax.jit(lambda x, w: {"y": _reordered(x, w)["y"],
                                     "history": jnp.zeros((16, 0, 48), jnp.bfloat16)})
    with pytest.raises(AssertionError):
        assert_as_exact(narrower, tree(_product), _draw, "a narrower empty leaf", seeds=8)

    def fuller(x, w):
        return {"y": _product(x, w)["y"], "history": jnp.zeros((16, 1, 48), x.dtype)}

    with pytest.raises(AssertionError):
        assert_as_exact(tree(_reordered), tree(_product), _draw, "a truth that is not empty", seeds=8,
                        truth_fn=fuller)


def test_a_recording_keeps_the_confirmed_count_where_the_proxy_planned_more():
    """A proxy whose one element needs more seeds than the recorded reference's 64 do: the recording
    keeps (1 + REDRAW) times the count the recorded reference confirms, not what the proxy planned."""
    from tool_support import load

    recorder = load("recorded_references")

    def draw(seed):
        return {"x": np.random.default_rng(200 + seed).normal(size=(64,)).astype(np.float32)}

    def evaluate(inputs):
        x = inputs["x"]
        return {"y": x * x - np.float32(1)}, {"y": x.astype(np.float64) ** 2 - 1}

    def one_element(inputs):
        return {"y": jnp.sum(inputs["x"] * inputs["x"])[None]}

    planned = seeds_for(jax.jit(lambda inputs: one_element(inputs)), lambda seed: (draw(seed),), "proxy")
    recorded = recorder.record_case("site", draw, evaluate, one_element)
    seeds = int(recorded["site.seeds"])
    assert planned > seeds and len(recorded["site.input.x"]) == 5 * seeds


def test_an_input_digest_tells_apart_a_scalar_from_its_vector_and_two_formats_of_one_width():
    """Inputs drawn again in another shape or format with the same bytes are not the recorded ones."""
    assert input_digest({"x": np.float32(1)}) != input_digest({"x": np.ones(1, np.float32)})
    one = np.array([1.0, 2.0], jnp.float8_e4m3fn)
    assert input_digest({"x": one}) != input_digest({"x": one.view(jnp.float8_e4m3fnuz)})
    strided = np.arange(12, dtype=np.float32).reshape(3, 4)[:, ::2]
    assert input_digest({"x": strided}) == input_digest({"x": np.ascontiguousarray(strided)})


def test_a_recording_of_outputs_alone_draws_its_inputs_again(tmp_path):
    """A generator's seeded draw, called again by the test (`inputs_of`), stands in for recorded inputs:
    the recording holds the outputs alone and is judged as one that holds both. A draw that changed since
    the recording is refused by its digest, as is `inputs_of` on a recording with no digest."""
    from tool_support import load

    recorder = load("recorded_references")

    def draw(seed):
        return {"x": np.random.default_rng(300 + seed).normal(size=(64,)).astype(np.float32)}

    def evaluate(inputs):
        x = inputs["x"]
        return {"y": x * x - np.float32(1)}, {"y": x.astype(np.float64) ** 2 - 1}

    def squared(inputs):
        return {"y": inputs["x"] * inputs["x"] - 1}

    recorded = recorder.record_case("site", draw, evaluate, squared, store_inputs=False)
    assert not any(name.startswith("site.input.") for name in recorded)
    seeds = int(recorded["site.seeds"])
    with pytest.raises(AssertionError):
        indexed_recording(recorded, "site", seeds)
    drawn, reference, truth = indexed_recording(recorded, "site", seeds, inputs_of=draw)
    assert_as_exact(jax.jit(lambda index, inputs: squared(inputs)), reference, drawn, "drawn again",
                    seeds=seeds, truth_fn=truth)
    changed, _, _ = indexed_recording(recorded, "site", seeds, inputs_of=lambda seed: draw(seed + 1))
    with pytest.raises(AssertionError):
        changed(0)
    undigested = {name: value for name, value in recorded.items() if name != "site.input_digest"}
    with pytest.raises(AssertionError):
        indexed_recording(undigested, "site", seeds, inputs_of=draw)


def test_a_recording_of_the_references_evidence_alone_holds_the_recomputed_truth_to_its_sketch():
    """A recording that keeps the reference's evidence (`leaf_summary`) and its truth's sketch, no
    outputs: check 1 judges Dew against it with the test's own float64 path as truth, held at every seed
    to the recorded sketch, so a float64 path that computes one element a millionth off is refused, and
    a recording without that path is refused outright."""
    from tool_support import load

    recorder = load("recorded_references")

    def draw(seed):
        rng = np.random.default_rng(400 + seed)
        return {"x": rng.normal(size=(16, 8)).astype(np.float32),
                "w": rng.normal(size=(8, 32)).astype(np.float32)}

    def evaluate(inputs):
        x, w = inputs["x"], inputs["w"]
        return ({"y": np.dot(x, w)},
                {"y": np.dot(x.astype(np.float64), w.astype(np.float64))})

    def product(inputs):
        return {"y": jnp.dot(inputs["x"], inputs["w"], precision="highest")}

    def wide(inputs):
        return {"y": jnp.dot(jnp.asarray(inputs["x"], jnp.float64), jnp.asarray(inputs["w"], jnp.float64))}

    recorded = recorder.record_case("site", draw, evaluate, product, summary=True)
    assert not any(name.startswith(("site.reference.", "site.truth.", "site.input.")) for name in recorded)
    seeds = int(recorded["site.seeds"])
    drawn, reference, truth = indexed_recording(recorded, "site", seeds, inputs_of=draw, truth_fn=wide)
    assert_as_exact(jax.jit(lambda index, inputs: product(inputs)), reference, drawn, "evidence", seeds=seeds,
                    truth_fn=truth)

    def off(inputs):
        y = wide(inputs)["y"]
        return {"y": y.at[3, 5].multiply(1 + 1e-6)}

    _, _, wrong = indexed_recording(recorded, "site", seeds, inputs_of=draw, truth_fn=off)
    with pytest.raises(AssertionError), jax.enable_x64(new_val=True):
        wrong(*widened(drawn(0)))
    _, _, overflowing = indexed_recording(recorded, "site", seeds, inputs_of=draw,
                                          truth_fn=lambda inputs: {"y": wide(inputs)["y"] * 1e300})
    with pytest.raises(AssertionError), jax.enable_x64(new_val=True):
        overflowing(*widened(drawn(0)))
    assert recorded["site.sketch.y"].shape == (len(recorded["site.mse.y"]), 32)
    with pytest.raises(AssertionError):
        indexed_recording(recorded, "site", seeds, inputs_of=draw)


def test_a_truths_sketch_rounds_alike_in_any_order_and_refuses_an_overflowing_or_empty_truth_apart():
    """The sketch of one truth agrees with itself summed in another order within `sketch_rounding`, each
    block of SKETCH_BLOCK summed from its other end as another host may; a
    truth recomputed far past float64's range is refused, not passed on an infinite bound; and an empty
    leaf's evidence is exact, its reference's dtype kept."""
    from reference_error import SKETCH, _sign_bits, leaf_summary, sketch_rounding, truth_sketch

    rng = np.random.default_rng(7)
    truth = rng.normal(size=4096)
    signs = ((_sign_bits(truth.size, 3)[:, None] >> np.arange(SKETCH, dtype=np.uint64)) & 1) * 2.0 - 1
    # Another host: each block summed one by one from its end, the blocks' sums summed exactly.
    blocks = np.flip((truth[:, None] * signs).reshape(-1, 64, SKETCH), axis=1).cumsum(axis=1)[:, -1]
    other_order = np.asarray([math.fsum(blocks[:, index].tolist()) for index in range(SKETCH)])
    allowed = 2 * sketch_rounding(float(np.sum(np.abs(truth))))
    assert np.all(np.abs(truth_sketch(truth, 3) - other_order) <= allowed)
    empty = leaf_summary(jnp.zeros((16, 0), jnp.bfloat16), np.zeros((16, 0)))
    assert (empty.mse, empty.worst, empty.finite, empty.dtype) == (0.0, 0.0, True, "bfloat16")
    with pytest.raises(AssertionError):
        leaf_summary(np.zeros(0), np.ones(3))
    # Two entries swapped are seen at a seed unless their signs agree in all SKETCH places.
    swapped = truth.copy()
    swapped[[0, 1]] = swapped[[1, 0]]
    bits = _sign_bits(truth.size, 3)
    assert (bits[0] == bits[1]) or np.any(truth_sketch(swapped, 3) != truth_sketch(truth, 3))
