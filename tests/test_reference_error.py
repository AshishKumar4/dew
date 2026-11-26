"""The float64 accuracy rule (tests/reference_error.py) on comparisons whose answer is known: what it
must refuse, and the rounding it must admit."""

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from arm_arithmetic import Written
from arm_forms import ROUNDED, over_windows, run_to_run
from check1_wire import CENSUS
from gate_cases import a_product_draw, cancelling, cancelling_draw, product_draws
from one_draw import assert_as_exact_as_the_reference
from reference_error import (
    FACTOR,
    PARITY,
    GateFailure,
    Inconclusive,
    as_given,
    assert_as_exact,
    assert_moves,
    assert_same_artifact,
    dew_path,
    distance,
    indexed_recording,
    input_digest,
    null_rate_bound,
    null_runs,
    seeds_for,
    widened,
)

from dew.nn.precision import accumulated
from dew.parity import MOST_SEEDS

pytestmark = pytest.mark.usefixtures("unqualified")
"""The gate's own tests: their sites are its subjects, declared at the counts they test it at."""


@pytest.fixture
def switched(monkeypatch):
    """The gate's YNNPACK switch on, whatever the run (a census pass sets `CENSUS`): the switch's own tests
    hold the switch."""
    monkeypatch.delenv(CENSUS, raising=False)


@pytest.mark.skipif(jax.default_backend() != "cpu", reason="WITHOUT_YNN is XLA:CPU's")
@pytest.mark.usefixtures("switched")
def test_without_ynn_computes_what_xla_cpu_computes_with_every_ynnpack_fusion_off(tmp_path):
    """Each of tests/ynn_programs.py's programs under `without_ynn` gives, bit for bit, what the same
    program gives jitted in a process whose XLA:CPU runs with every YNNPACK fusion off."""

    import ynn_programs
    from reference_error import without_ynn

    out = tmp_path / "theirs.npz"
    flags = os.environ.get("XLA_FLAGS", "") + " --xla_cpu_experimental_ynn_fusion_type="
    subprocess.run([sys.executable, str(Path(ynn_programs.__file__)), str(out)], check=True,
                   env={**os.environ, "XLA_FLAGS": flags})
    with np.load(out) as theirs:
        for name, value in ynn_programs.outputs(without_ynn).items():
            np.testing.assert_array_equal(np.asarray(value), theirs[name], err_msg=name)


@pytest.mark.skipif(jax.default_backend() != "cpu", reason="WITHOUT_YNN is XLA:CPU's")
def test_the_census_runs_the_gates_arms_as_xla_cpu_compiles_them_for_users(tmp_path):
    """Under the census (`CENSUS` set), `without_ynn` compiles a program as plain `jax.jit` does, YNNPACK's
    fusions on: each of tests/ynn_programs.py's programs gives, bit for bit, what it gives jitted in a
    process of XLA:CPU's defaults (row 4471)."""
    import ynn_programs
    out, ours = tmp_path / "theirs.npz", tmp_path / "ours.npz"
    subprocess.run([sys.executable, str(Path(ynn_programs.__file__)), str(out)], check=True)
    census = "import sys, numpy as np, ynn_programs; from reference_error import without_ynn; " \
        "np.savez(sys.argv[1], **{k: np.asarray(v) for k, v in ynn_programs.outputs(without_ynn).items()})"
    subprocess.run([sys.executable, "-c", census, str(ours)], check=True,
                   cwd=Path(ynn_programs.__file__).parent, env={**os.environ, CENSUS: "1"})
    with np.load(out) as theirs, np.load(ours) as found:
        for name in theirs:
            np.testing.assert_array_equal(found[name], theirs[name], err_msg=name)


def one_key_draw(seed):
    """bf16 queries [2, 5, 4, 32] over a single key and value of two heads."""
    rng = np.random.default_rng([1698, 1, seed])
    return tuple(jnp.asarray(rng.standard_normal(shape), jnp.bfloat16)
                 for shape in ((2, 5, 4, 32), (2, 1, 2, 32), (2, 1, 2, 32)))


def pulled_by_itself(attention):
    """`attention`'s output and the gradients of its inputs under that output as the cotangent."""
    def run(query, key, value):
        out, pull = jax.vjp(attention, query, key, value)
        return out, *pull(out)
    return run


def xla_attention(query, key, value):
    """jax.nn's attention, its XLA implementation."""
    return jax.nn.dot_product_attention(query, key, value, implementation="xla")


def written_attention(query, key, value):
    """The same attention written out: grouped heads, logits in at least fp32, a softmax there rounded to the
    operands' dtype."""
    group = query.shape[2] // key.shape[2]
    keys, values = jnp.repeat(key, group, axis=2), jnp.repeat(value, group, axis=2)
    wide = jnp.promote_types(query.dtype, jnp.float32)
    logits = jnp.einsum("btnh,bsnh->bnts", query, keys, preferred_element_type=wide)
    weights = jax.nn.softmax(logits / np.sqrt(query.shape[-1]), axis=-1).astype(query.dtype)
    return jnp.einsum("bnts,bsnh->btnh", weights, values)


@pytest.mark.skipif(jax.default_backend() != "cpu", reason="WITHOUT_YNN is XLA:CPU's")
@pytest.mark.usefixtures("switched")
def test_a_check_whose_program_xla_cpu_compiles_only_with_ynnpacks_fusions_runs_both_arms_with_them(
        tmp_path, monkeypatch):
    """XLA:CPU refuses jax.nn's XLA attention's backward over a single key without YNNPACK's fusions, at
    every matmul precision; check 1 decides, once before its draws, that both arms of that site take the
    fusions (`_ynn_decided`), the two still of one kind, names that decision in its stage's line
    (STAGES_LOG) and in the report CI reads (`SWITCHED`), and judges them."""

    from check1_judge import STAGES_LOG, SWITCHED
    from reference_error import without_ynn

    with pytest.raises(jax.errors.JaxRuntimeError):
        without_ynn(pulled_by_itself(xla_attention))(*one_key_draw(0))
    log = tmp_path / "stages.jsonl"
    monkeypatch.setenv(STAGES_LOG, str(log))
    assert_as_exact(pulled_by_itself(xla_attention), pulled_by_itself(written_attention), one_key_draw,
                    "attention over one key", seeds=16)
    lines = [json.loads(line) for line in log.read_text().splitlines()]
    assert lines and all(entry["ynn"] is not None for entry in lines)
    assert [entry["label"] for entry in SWITCHED] == ["attention over one key"]
    SWITCHED.clear()
    # An arm declared to vary run to run takes the site's decision as any arm does, through its declaration.
    from reference_error import run_to_run

    assert_as_exact(run_to_run(pulled_by_itself(xla_attention), "atomic adds"),
                    pulled_by_itself(written_attention), one_key_draw, "attention over one key, declared",
                    seeds=16)
    SWITCHED.clear()
    # A window loop's paired arms run each window's program, decided at their own calls.
    from reference_error import over_windows, windows

    assert_as_exact(over_windows(pulled_by_itself(written_attention)),
                    over_windows(pulled_by_itself(written_attention)), windows(one_key_draw, 3),
                    "attention over one key, windowed", seeds=8)
    SWITCHED.clear()


@pytest.mark.skipif(jax.default_backend() != "cpu", reason="WITHOUT_YNN is XLA:CPU's")
@pytest.mark.usefixtures("switched")
def test_a_process_whose_xla_refuses_the_ynn_options_moves_no_site_onto_the_fusions(monkeypatch):
    """Where XLA:CPU refuses WITHOUT_YNN's options themselves, a site of two Dew paths raises before
    deciding (`_ynn_option_held`), so it moves onto the fusions and passes nowhere."""
    import arm_forms
    import reference_error

    arm_forms._ynn_option_held.cache_clear()
    monkeypatch.setattr(arm_forms, "WITHOUT_YNN", {"xla_cpu_no_such_option": "none"})
    try:
        with pytest.raises(RuntimeError):
            assert_as_exact(pulled_by_itself(xla_attention), pulled_by_itself(written_attention),
                            one_key_draw, "attention over one key, its options refused", seeds=16)
        assert not reference_error.SWITCHED
    finally:
        arm_forms._ynn_option_held.cache_clear()


@pytest.mark.skipif(jax.default_backend() != "cpu", reason="WITHOUT_YNN is XLA:CPU's")
def test_without_ynn_takes_a_callers_compiler_options_over_its_own():
    """A caller who asks for YNNPACK's dot fusion gets the dot the default XLA:CPU computes, bit for
    bit."""
    from reference_error import without_ynn
    from ynn_programs import w, x

    dot_fusion = {"xla_cpu_experimental_ynn_fusion_type": "LIBRARY_FUSION_TYPE_DOT"}
    np.testing.assert_array_equal(without_ynn(jnp.dot, compiler_options=dot_fusion)(x, w),
                                  jax.jit(jnp.dot)(x, w))


def test_over_jax_inputs_an_arm_runs_only_compiled_and_a_numpy_one_is_refused_there():
    """A draw holding no NumPy array calls no arm before the gate compiles it: a JAX arm over JAX inputs
    runs traced alone, where called as given once first a model arm ran eagerly, a minute of a check
    (ledger 3774). A NumPy arm over them is refused where the gate's compile traces it, its NumPy call on a
    traced input refused by JAX; over NumPy inputs, where operators alone would trace it into float32
    JAX, it is refused before it runs compiled
    (`test_a_numpy_arm_is_refused_unless_it_is_given_and_windows_keep_their_arms_policy`)."""
    from reference_error import assert_as_exact

    called = []

    def dew(x):
        called.append(isinstance(x, jax.core.Tracer))
        return x * 2

    def numpy(x):
        return np.asarray(x) * 2

    def draw(seed):
        return (jnp.asarray(np.random.default_rng(seed).standard_normal(8), jnp.float32),)

    assert_as_exact(dew, dew, draw, "a JAX arm over JAX inputs", seeds=8)
    assert called and all(called)
    with pytest.raises(jax.errors.TracerArrayConversionError):
        assert_as_exact(dew, numpy, draw, "a NumPy arm over JAX inputs", seeds=8)


def noised_draw(seed):
    """A draw of standard normal values beside three noises of 1e-7 of them, which arms add at chosen scales
    (the further references' and the library arm's sites)."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal(4096), *(1e-7 * rng.standard_normal(4096) for _ in range(3))


def test_check_1_sizes_a_cancelling_row_by_its_terms_and_admits_nothing_else():
    """Term sizing (`assert_as_exact`'s `terms`, owner 10-29): a row whose truth cancels exactly, Dew's an
    ulp or so off its zero where the reference's sum cancels to it, is judged at its terms' size by the
    row rule and admitted, by default too, its truth holding a row of float64 zero (`_cancelling_terms`,
    ledger 2851), where without terms (False) it fails as a row of float64 zero against the reference's
    exact zero; and nothing else is: a dropped term in that row is refused, a residue on rows whose terms
    are zero too (no size) is refused as a row of float64 zero, and a truth with no terms (run as given)
    is refused at the site."""
    reference = cancelling(split=False)
    assert_as_exact(cancelling(split=True), reference, cancelling_draw, "a cancelling dot", seeds=16,
                    terms=True)
    assert_as_exact(cancelling(split=True), reference, cancelling_draw, "by default", seeds=16)
    with pytest.raises(GateFailure):
        assert_as_exact(cancelling(split=True), reference, cancelling_draw, "untermed", seeds=16,
                        terms=False)
    with pytest.raises(GateFailure):
        assert_as_exact(cancelling(split=True, leak=True), reference, cancelling_draw, "a leaked term",
                        seeds=16, terms=True)
    with pytest.raises(GateFailure):
        assert_as_exact(cancelling(split=True, residue=1e-20), reference, cancelling_draw,
                        "a residue on rows of no terms", seeds=16, terms=True)
    with pytest.raises(ValueError):
        assert_as_exact(cancelling(split=True), reference, cancelling_draw, "untraceable", seeds=16,
                        terms=True, truth_fn=as_given(lambda x, w: {"rows": np.zeros((64, 4))}))


@pytest.mark.parametrize("cancels", [False, True])
def test_a_sites_terms_decide_by_its_zero_leaves_beside_others(cancels):
    """A truth whose zero rows sit in one leaf beside a leaf of none (`_cancelling_terms`, which traces the
    terms of the leaves holding zero rows alone): a zero row that cancels sizes the site by its terms, and
    an empty sum leaves it the reference's residue rule, whatever the other leaves hold."""
    from reference_error import _cancelling_terms

    def truth(x):
        total = jnp.sum(x, axis=-1, keepdims=True)
        zero = total - total if cancels else jnp.sum(jnp.zeros_like(x), axis=-1, keepdims=True)
        return {"nonzero": total + 100, "zero": jnp.concatenate([zero, total + 100])}

    def draw(seed):
        return (jnp.arange(32, dtype=jnp.float32).reshape(4, 8),)

    assert (_cancelling_terms(draw, truth) is truth) == cancels


MIX = np.random.default_rng(2442).standard_normal((4, 4)).astype(np.float32)
"""A fixed [4, 4] product past `cancelling`'s rows (`mixed`)."""


def mixed(arm):
    """`arm`'s rows through MIX: a row that cancels before that product sums zero terms into it, in
    float64, however far each arm is off its zero."""
    def run(x, w):
        return {"rows": arm(x, w)["rows"] @ jnp.asarray(MIX)}
    return run


def mixed_terms(x, w):
    """`mixed`'s terms carried through its cancellation (`float64_terms`' sizes): each dot's entry the
    root sum of squares of its products, the cancelling rows' two sums both, then MIX's products of
    those sizes."""
    sizes = np.sqrt(np.square(np.asarray(x)) @ np.square(np.asarray(w)))
    sizes[:8] *= np.sqrt(2.0)
    return {"rows": np.sqrt(np.square(sizes) @ np.square(MIX.astype(np.float64)))}


def test_the_tail_tests_floor_is_derived_from_its_faults_at_the_leafs_noise():
    """The tail test's null share (`_tail_floor`, Main 2026-10-31) is ½ at the counts EXAMPLE_POWER asks,
    where no fault of FAULTS clears it with power to spare; at a large count over a heavy symmetric noise it
    rises above ½, so equally exact arms whose tails are a little uneven pass, and twice off on a tenth, the
    smallest fault, is still refused there; a fault in the leaf's own noise lowers its floor."""
    from dew.parity import LOWER_TAIL, _tail_floor, _tail_sign

    rng = np.random.default_rng([2469, 3])
    small = 0.4 * rng.standard_t(3, 1024) / np.sqrt(3)
    assert _tail_floor(small, LOWER_TAIL) == 0.5
    heavy = 0.5 * rng.standard_t(3, 82000) / np.sqrt(3)
    uneven = np.where(rng.random(82000) < 0.02, np.abs(heavy), heavy)  # a little more of the tail above
    assert _tail_floor(uneven, LOWER_TAIL) > 0.55 and _tail_sign(uneven) >= LOWER_TAIL
    twice = heavy + np.where(rng.random(82000) < 0.1, np.log(2.0), 0.0)
    assert _tail_sign(twice) < LOWER_TAIL
    tenfold = heavy + np.where(rng.random(82000) < 0.1, np.log(10.0), 0.0)
    assert _tail_floor(tenfold, LOWER_TAIL) < _tail_floor(heavy, LOWER_TAIL)
    assert _tail_sign(tenfold) < LOWER_TAIL


def test_a_row_both_arms_hold_exactly_is_not_carried_and_the_site_judges_as_before():
    """A row of float64 zero both arms hold exactly is no evidence (`_paired`): it is not carried in a
    seed's zero rows, which a window's gradient fills with every row it never touches (ledger 2469), and
    a site of such rows passes as it did."""
    from reference_error import _drawn

    def zeroed_draw(seed):
        x, w = a_product_draw(seed)
        return x.at[:4].set(0.0), w

    [draws] = _drawn(jnp.matmul, jnp.matmul, zeroed_draw, range(2), repeats=True)
    for rows, mine, theirs in (seed[0] for seed in draws.zero_rows):
        assert not np.any((mine == 0) & (theirs == 0)) and not np.isin(np.arange(4), rows).any()
    assert_as_exact(jnp.matmul, jnp.matmul, zeroed_draw, "rows held exactly", seeds=8)


def test_check_1_reads_a_discrete_leaf_in_its_own_representation():
    """A discrete leaf's errors are taken before the gate's float64 (row 13543): an int64 count of 2^53 the
    truth and the reference hold, which Dew's arm holds as 2^53 + 1 (float64 rounds both to 2^53), fails
    Dew's arm, where its own exact value passes; and the exact difference reaches the evidence, two leaves
    one apart holding a squared error of one."""
    from dew.parity import _error_rows

    big = np.int64(2**53)
    [(log_mse, *_)] = _error_rows([np.asarray([big + 1])], [np.asarray([big])], xp=np)
    assert log_mse == 0.0
    # The extremes apart by the whole range, never wrapped to a one-count error; signed against unsigned
    # refused.
    least, most = np.iinfo(np.int64).min, np.iinfo(np.int64).max
    [(log_mse, *_)] = _error_rows([np.asarray([most])], [np.asarray([least])], xp=np)
    assert np.isclose(log_mse, 2 * np.log(2.0**64))
    with pytest.raises(ValueError):
        _error_rows([np.asarray([1], np.uint64)], [np.asarray([1], np.int64)], xp=np)

    def counted(value):
        return as_given(lambda seed: {"count": np.asarray([value], np.int64)})

    def draw(seed):
        return (np.float32(seed),)

    assert_as_exact(counted(big), counted(big), draw, "an exact count", seeds=8, truth_fn=counted(big))
    with pytest.raises(GateFailure):
        assert_as_exact(counted(big + 1), counted(big), draw, "a count one off", seeds=8,
                        truth_fn=counted(big))


def test_check_1_refuses_a_draw_its_seed_does_not_fix_and_an_arm_of_more_than_its_inputs():
    """Planted controls (ledger 2641): a draw from a generator its seed does not fix, and an arm drawing a
    noise of its own, each compare draws no seed pairs, and check 1 refuses both before judging them
    (`UnpairedDraw`); a planted non-finite entry in Dew's arm fails it."""
    from reference_error import UnpairedDraw

    unseeded = np.random.default_rng()

    def unfixed_draw(seed):
        x, w = a_product_draw(seed)
        return x + jnp.asarray(unseeded.standard_normal(x.shape) * 1e-3, jnp.float32), w

    with pytest.raises(UnpairedDraw):
        assert_as_exact(jnp.matmul, jnp.matmul, unfixed_draw, "a draw its seed does not fix", seeds=16)
    noise = iter(range(10**6))

    def noisy(x, w):
        return x @ w + 1e-6 * jax.random.normal(jax.random.key(next(noise)), (16, 8))

    with pytest.raises(UnpairedDraw):
        assert_as_exact(as_given(noisy), jnp.matmul, a_product_draw, "an arm of its own noise", seeds=16)

    def planted(x, w):
        return (x @ w).at[3, 5].set(jnp.nan)

    with pytest.raises(GateFailure):
        assert_as_exact(planted, jnp.matmul, a_product_draw, "a planted non-finite entry", seeds=16)


def test_the_paired_worst_row_refuses_a_row_far_off_on_some_examples_and_passes_equally_exact_arms():
    """At a site that stacks examples, each example's worst row is held to the reference's on the same
    example (`_paired_rows`, ledger 2469): equally exact arms, their worst rows' log ratios symmetric however
    wide (a spread of 1.0 and 2.0), pass; Dew ten times the reference's on a fifth of 512 examples of such
    spread is refused at LOWER_TAIL; and two examples past the factor, both Dew's, are no finding."""
    from reference_error import LOWER_TAIL, ROW_FACTOR

    from dew.parity import _paired_rows

    rng = np.random.default_rng([2469, 512])
    for spread in (1.0, 2.0):
        null = rng.normal(0.0, spread, 512)
        assert _paired_rows(null, ROW_FACTOR) >= LOWER_TAIL
        off = null.copy()
        off[::5] += math.log(10.0)
        assert _paired_rows(off, ROW_FACTOR) < LOWER_TAIL
    assert _paired_rows(np.asarray([3.0, 3.0, 0.0]), ROW_FACTOR) >= LOWER_TAIL


def test_check_1_refuses_a_draw_and_an_arm_that_turn_stateful_after_their_first_block():
    """Planted controls of a later block (ledger 2641): a draw fixed by its seed in a site's first block that
    reads a state its calls leave from then on, and an arm of no noise on its first block that adds one
    after, are refused at the next block's first draw (`UnpairedDraw`), as every block checks its first
    draw and its arms on it; the first block alone passes."""
    from reference_error import UnpairedDraw

    asked = []

    def later_stateful(seed):
        x, w = a_product_draw(seed)
        asked.append(seed)
        return (x if seed == asked[0] else x + 1e-3 * len(asked)), w

    with pytest.raises(UnpairedDraw):
        null_runs(jnp.matmul, later_stateful, "a draw stateful after its first block", seeds=8, runs=2)
    calls = []

    def noisy_later(x, w):
        calls.append(len(calls))
        return x @ w + (0 if len(calls) <= 2 else 1e-6 * len(calls))

    # Its truth a pure product of its own, so the arm's first two calls are its first block's evaluation and
    # that check's replay: the first block passes, and the second's replay refuses it.
    [(verdict, _, _)] = null_runs(as_given(noisy_later), a_product_draw, "an arm pure on its first block",
                                  seeds=8, runs=1, truth_fn=jnp.matmul)
    assert verdict == "pass" and len(calls) > 2
    calls.clear()
    with pytest.raises(UnpairedDraw):
        null_runs(as_given(noisy_later), a_product_draw, "an arm of noise after its first block", seeds=8,
                  runs=2, truth_fn=jnp.matmul)


def test_check_1_sizes_rows_by_their_terms_beside_an_output_held_as_none():
    """Term sizing reads a truth's terms by the truth's own structure (`_term_leaves`): an output held as
    None (a gradient of an input given as None, as `with_gradients` returns one) takes no leaf of the
    terms, so the leaves after it are sized by their own, not their neighbour's."""
    def holding_none(arm):
        def run(x, w):
            return {"before": arm(x, w)["rows"], "none": None, "after": arm(x, w)["rows"][:, :2]}
        return run

    assert_as_exact(holding_none(cancelling(split=True)), holding_none(cancelling(split=False)),
                    cancelling_draw, "terms beside a None", seeds=16, terms=True)


def test_check_1_sizes_a_row_that_cancels_behind_its_final_sum_by_the_sites_own_terms():
    """A site's own terms program (`assert_as_exact`'s `terms`): the traced terms are a leaf's final
    sums' alone, so rows that cancel before a further product (`mixed`, as an attention query that keeps
    one key cancels its logit gradient before dQ sums it) have none and fail as rows of float64 zero; the
    site's own terms, carried through the cancellation, size them, and nothing else is admitted: a
    dropped term there is refused, a residue on rows whose terms are zero too is refused, and terms of
    another tree are refused at the site."""
    reference = mixed(cancelling(split=False))
    with pytest.raises(GateFailure):
        assert_as_exact(mixed(cancelling(split=True)), reference, cancelling_draw, "traced terms", seeds=16,
                        terms=True)
    assert_as_exact(mixed(cancelling(split=True)), reference, cancelling_draw, "the site's terms", seeds=16,
                    terms=mixed_terms)
    with pytest.raises(GateFailure):
        assert_as_exact(mixed(cancelling(split=True, leak=True)), reference, cancelling_draw, "a leaked term",
                        seeds=16, terms=mixed_terms)
    with pytest.raises(GateFailure):
        assert_as_exact(mixed(cancelling(split=True, residue=1e-20)), reference, cancelling_draw,
                        "a residue on rows of no terms", seeds=16, terms=mixed_terms)
    with pytest.raises(ValueError):
        assert_as_exact(mixed(cancelling(split=True)), reference, cancelling_draw, "another tree", seeds=16,
                        terms=lambda x, w: mixed_terms(x, w)["rows"])


long_dot_draw = product_draws(1698, 4096)
"""A [16, 4096] by [4096, 8] fp32 dot's operands: sums long enough that fp32's own rounding of them is
several times the half ulp of a float64 sum rounded once."""


def dews_product(x, w):
    """Dew's product (`accumulated`): in the suite's wide CPU arithmetic, float64 sums rounded once."""
    return accumulated(jnp.matmul, x, w)


def summed_wide(x, w):
    """Dew's product stating a precision of its own inside its arm, so it keeps the float64 sums."""
    with jax.default_matmul_precision("highest"):
        return dews_product(x, w)


@pytest.mark.skipif(jax.default_backend() != "cpu", reason="the suite's wide arithmetic is a CPU's")
def test_check_1_runs_each_arm_as_its_users_sum_their_products():
    """On the CPU lane check 1 runs each arm with fp32 products summed in fp32 (`summed_as_run`), the
    truth outside that: Dew's product, which sums in float64 only under the suite's wide arithmetic and
    as XLA's own float32 dot as its users run it (`accumulated`, row 5448), is judged as they run it, so it is
    on par with XLA's plain fp32 dot but not with the same product kept to its float64 sums; and against
    that product the plain dot fails too, so the truth is still float64's."""
    assert_as_exact(dews_product, jnp.matmul, long_dot_draw, "as its users sum", seeds=16)
    for arm, label in ((dews_product, "as run against summed wide"), (jnp.matmul, "summed wide")):
        with pytest.raises(GateFailure):
            assert_as_exact(arm, summed_wide, long_dot_draw, label, seeds=16)


def test_further_references_judge_one_dew_arm_each_by_its_own_check_once_a_draw():
    """`also`: one Dew arm against several references on the same draws, each judged by its own check 1
    under its own label, on the same draws for them all. A Dew arm on par with one
    reference and three times another's error fails against that one alone; a site with every
    reference on par passes; beside a plain Dew arm, a further reference of another kind than the
    reference's is refused."""
    from test_gate_arms import noised as arm, noiseless as truth

    draw = noised_draw
    assert_as_exact(arm(1.0, "mine"), arm(1.0, "theirs"), draw, "two references", seeds=16, truth_fn=truth,
                    also={"another": arm(1.0, "other")})
    with pytest.raises(GateFailure) as failed:
        assert_as_exact(arm(3.0, "mine"), arm(3.0, "theirs"), draw, "one far", seeds=16, truth_fn=truth,
                        also={"another": arm(1.0, "other")})
    assert {finding.leaf for finding in failed.value.findings} == {"one far, against another['value']"}
    def plain(x, *_):
        return {"value": x}

    with pytest.raises(ValueError):
        assert_as_exact(plain, plain, draw, "two kinds", seeds=16, truth_fn=truth,
                        also={"given": arm(1.0, "other")})


def test_a_library_arm_holds_dew_to_the_larger_of_its_error_and_the_references():
    """With the owner's library arm (`library_fn`), Dew is held within PARITY of the larger of the
    reference's error and the library arm's, seed by seed, on the leaves the exception covers: a Dew
    1.7 times the reference's error passes beside a library arm 1.8 times it, fails without it, fails
    at 3 times (past the library arm's own by 1.67), and fails on a leaf the exception does not cover;
    a library arm without its ledger row or its leaves is refused, as is one in NumPy the gate would
    compile. Null runs judge as the site does."""
    from reference_error import null_runs

    draw = noised_draw

    def arm(scale, noise):
        @as_given
        def run(x, mine, theirs, library):
            value = x + scale * {"mine": mine, "theirs": theirs, "library": library}[noise]
            return {"covered": value, "other": value}
        return run

    @as_given
    def truth(x, *_):
        return {"covered": x, "other": x}

    reference, library, both = arm(1.0, "theirs"), arm(1.8, "library"), ["['covered']", "['other']"]
    assert_as_exact(arm(1.7, "mine"), reference, draw, "library arm", seeds=16, truth_fn=truth,
                    library_fn=library, library_row=1543, library_leaves=both)
    with pytest.raises(GateFailure):
        assert_as_exact(arm(1.7, "mine"), reference, draw, "no library arm", seeds=16, truth_fn=truth)
    with pytest.raises(GateFailure):
        assert_as_exact(arm(3.0, "mine"), reference, draw, "past the library", seeds=16, truth_fn=truth,
                        library_fn=library, library_row=1543, library_leaves=both)
    with pytest.raises(GateFailure) as failed:
        assert_as_exact(arm(1.7, "mine"), reference, draw, "one leaf", seeds=16, truth_fn=truth,
                        library_fn=library, library_row=1543, library_leaves=["['covered']"])
    assert {finding.leaf for finding in failed.value.findings} == {"one leaf['other']"}
    for unnamed in ({"library_leaves": both}, {"library_row": 1543}, {"library_row": 1543,
                                                                      "library_leaves": ["['third']"]}):
        with pytest.raises(ValueError):
            assert_as_exact(arm(1.7, "mine"), reference, draw, "unrecorded", seeds=16, truth_fn=truth,
                            library_fn=library, **unnamed)
    with pytest.raises(TypeError):
        assert_as_exact(arm(1.7, "mine"), reference, draw, "a NumPy library arm", seeds=16, truth_fn=truth,
                        library_fn=lambda x, *_: {"covered": np.full(np.shape(x), 1 / 3),
                                                  "other": np.full(np.shape(x), 1 / 3)},
                        library_row=1543, library_leaves=both)
    verdicts = null_runs(reference, draw, "library arm", seeds=16, runs=2, truth_fn=truth,
                         dew_fn=arm(1.7, "mine"), paired=True, library_fn=library, library_leaves=both)
    assert [(verdict, stage) for verdict, stage, _ in verdicts] == [("pass", 1), ("pass", 1)], verdicts


def test_a_reference_compiled_with_options_of_its_own_runs_as_given():
    """A reference compiled with compiler options of its own (a trainer step's, `as_a_step`) runs as
    given: the gate would nest it in its own jit, which a jit with options of its own refuses (row
    1800)."""
    from arm_forms import _arms

    own = jax.jit(lambda x: {"y": x * 2}, compiler_options={"xla_cpu_enable_fast_math": False})
    def doubled(x):
        return {"y": jnp.asarray(x) + jnp.asarray(x)}

    mine, theirs = _arms(doubled, as_given(own))
    assert theirs.as_given and not getattr(mine, "without_ynn", False)
    # Unmarked, the gate nests it in its own jit, which a jit with options of its own refuses.
    _, nested = _arms(doubled, own)
    with pytest.raises(ValueError):
        nested(np.ones(4, np.float32))
    assert_as_exact(doubled, as_given(own),
                    lambda seed: (np.random.default_rng(seed).standard_normal(64).astype(np.float32),),
                    "a reference with options", seeds=8)


def test_an_arm_compiled_outside_the_gates_policy_is_refused():
    """An arm compiled without YNNPACK's fusions (`without_ynn`) beside an arm run as given, where the
    gate compiles Dew's arm with the fusions as its users run it, is refused rather than run unseen
    without them (row 2287); beside another Dew arm, where the gate compiles without them too, it runs
    as the gate would compile it. A plain jit is nested under the gate's own compile, whose options
    apply; a jit with options of its own refuses that nesting itself (row 1800). Every other arm kind
    runs: plain, as given, a window loop's."""
    from arm_forms import _arms
    from reference_error import over_windows, without_ynn

    def doubled(x):
        return {"y": jnp.asarray(x) + jnp.asarray(x)}

    given = as_given(lambda x: {"y": np.asarray(x) * 2})
    with pytest.raises(TypeError):
        _arms(without_ynn(doubled), given)
    with pytest.raises(TypeError):
        _arms(over_windows(without_ynn(doubled)), given)
    x = np.ones(4, np.float32)
    for dew, reference in ((without_ynn(doubled), doubled), (doubled, given), (jax.jit(doubled), given),
                           (jax.jit(doubled), doubled), (doubled, doubled)):
        mine, theirs = _arms(dew, reference)
        np.testing.assert_array_equal(np.asarray(mine(x)["y"]), np.asarray(theirs(x)["y"]))
    with pytest.raises(ValueError):
        _arms(jax.jit(doubled, compiler_options={"xla_cpu_enable_fast_math": False}), doubled)[0](x)


def test_a_float64_truth_runs_as_one_compiled_program():
    """The gate compiles a JAX truth (`_truth_program`) from its first call on, never running it eagerly, so
    a seed's truth is the same whichever seed a process draws first, and its values are the eager truth's
    own. A truth doing host work fails at its first call unless its site states it (`as_given`, ledger 3198);
    a truth `as_given`, one compiled already and a window loop's by its own run as they are; a check over a
    compiled truth judges as over the eager one."""
    from arm_arithmetic import Truth
    from reference_error import _truth_program, _Windowed, as_given, over_windows

    def truth(x, w):
        return {"y": jnp.dot(x, w, precision="highest")}

    with jax.enable_x64(new_val=True):
        program = _truth_program(truth)
        x, w = (np.asarray(value, np.float64) for value in _draw(0))
        first, later = program(x, w), program(x, w)
        np.testing.assert_array_equal(first["y"], later["y"])
        np.testing.assert_allclose(first["y"], truth(x, w)["y"], rtol=1e-14)
        # A truth that does host work, NumPy's or a widening on the host, fails at its first call, unstated.
        for hosted in (lambda x, w: {"y": np.asarray(x) @ np.asarray(w)},
                       lambda x, w: truth(np.asarray(x, np.float64), np.asarray(w, np.float64))):
            with pytest.raises(jax.errors.TracerArrayConversionError):
                _truth_program(hosted)(x, w)
    # A truth run as given, or compiled already, is itself, its programs unrewritten.
    given = as_given(truth)
    assert _truth_program(given).wrapped is given and _truth_program(_product).wrapped is _product
    assert _truth_program(None) is None
    windowed = _truth_program(over_windows(truth))
    assert isinstance(windowed, _Windowed) and isinstance(windowed.fn, Truth)
    assert_as_exact(_reordered, _product, _draw, "a compiled truth", seeds=8, truth_fn=truth)


@pytest.mark.usefixtures("switched")
def test_an_arm_of_dew_programs_compiles_them_as_the_gate_compiles_its_other_arm(monkeypatch):
    """Beside another Dew arm, an arm of Dew's own programs (`dew_programs`) runs with Dew's compile
    seam holding WITHOUT_YNN open (`compiled_with`), so a program it compiles through the seam
    (`dew.nn.kernels.options.program`) carries the option, and beside an arm `as_given` it runs as it
    is; the seam adds nothing once the arm returns, and a run's own XLA flag for an option wins."""
    from arm_forms import WITHOUT_YNN, _arms
    from reference_error import as_given, dew_programs

    from dew.nn.kernels import options

    seen = []

    @dew_programs
    def programs(x):
        seen.append(options.compiler_options())
        return options.program(lambda x: x * 2)(x)

    def written(x):
        return x + x

    mine, _ = _arms(programs, written)
    np.testing.assert_array_equal(mine(np.ones(3, np.float32)), 2 * np.ones(3, np.float32))
    expected = WITHOUT_YNN if jax.default_backend() == "cpu" else {}
    assert seen == [expected] and options.compiler_options() == {}
    mine, _ = _arms(programs, as_given(lambda x: np.asarray(x) * 2))
    mine(np.ones(3, np.float32))
    assert seen[-1] == {}
    with options.compiled_with({"xla_cpu_experimental_ynn_fusion_type": ""}):
        assert options.compiler_options() == {"xla_cpu_experimental_ynn_fusion_type": ""}
        monkeypatch.setenv("XLA_FLAGS", "--xla_cpu_experimental_ynn_fusion_type=dot")
        assert options.compiler_options() == {}
    assert options.compiler_options() == {}


def test_a_sizing_tells_each_count_it_draws_for_before_drawing_it():
    """A sizing given its `progress` (`seeds_for`'s, `parity.Sized`) is told each count it draws for, in
    order, each the one before it doubled from the start, with its place among them and their most (its
    doublings from the start to the cap), its last the count it declares: a long recording (a GPU's,
    chunked, cut short by its time) shows how far its sizing went, where it printed nothing until the end."""
    from reference_error import seeds_for

    from dew.parity import FEWEST_SEEDS, MOST_SEEDS

    told = []
    count = seeds_for(_product, _draw, "product", progress=lambda *each: told.append(each))
    doublings = len(told)
    assert [each for each, _, _ in told] == [FEWEST_SEEDS * 2 ** place for place in range(doublings)]
    assert [place for _, place, _ in told] == list(range(1, doublings + 1))
    assert {most for _, _, most in told} == {(MOST_SEEDS // FEWEST_SEEDS).bit_length()}
    assert told[-1][0] == count


def test_a_recordings_measured_errors_size_its_site_as_seeds_for_would():
    """`seeds_resolving` over the reference's errors measured apart (a recording's merged shards) gives
    `seeds_for`'s count for the reference compared with itself on the same draws, and refuses a
    recording of too few seeds."""
    from reference_error import _drawn, seeds_for

    from dew.parity import SIZING_FROM, seeds_resolving

    count = seeds_for(_product, _draw, "product")
    [draws] = _drawn(_product, _product, _draw, range(SIZING_FROM, SIZING_FROM + 4 * count), repeats=True)
    assert seeds_resolving(draws.theirs, draws.names, "product") == count
    with pytest.raises(AssertionError):
        seeds_resolving(draws.theirs[:count], draws.names, "product", start=count)
    # Past its ceiling it refuses, as seeds_for does, however many seeds are recorded.
    with pytest.raises(AssertionError):
        seeds_resolving(draws.theirs, draws.names, "product", most=count // 2)


def test_a_reference_sized_against_itself_draws_none_of_a_verdicts_seeds():
    """A reference sized against itself (`seeds_for` with no Dew arm) draws from SIZING_FROM on, as a paired
    sizing does: from seed 0 its count was chosen on the verdict's own inputs, its four blocks overlapping
    the whole first stage (spec alignment, 9005)."""
    from reference_error import seeds_for

    from dew.parity import SIZING_FROM

    drawn = []

    def draw(seed):
        drawn.append(seed)
        return _draw(seed)

    seeds_for(_product, draw, "product")
    assert min(drawn) >= SIZING_FROM


def _tripled(seed):
    """A draw of 64 normals and its seed, as JAX arrays (a NumPy draw makes `x * 3` a NumPy arm)."""
    return (jnp.asarray(np.random.default_rng(seed).standard_normal(64), jnp.float32), jnp.int32(seed))


def test_a_site_held_to_its_truth_rounded_once_evaluates_no_truth_of_its_own():
    """`ROUNDED`, the reference of draws whose truth float32 rounds once: float32's own product of x by three,
    one rounding of the exact product, passes against its float64 truth rounded once, the truth evaluated as
    often as at a site whose reference is a function (no second float64 run of it, rows 9481, 9515, 9532);
    an arm an ulp past it everywhere fails."""
    counted = []

    def truth(x, seed):
        counted.append(int(seed))
        return np.asarray(x, np.float64) * 3.0

    def calls(reference_fn) -> int:
        counted.clear()
        assert_as_exact(lambda x, seed: x * 3, reference_fn, _tripled, "x thrice", seeds=8,
                        truth_fn=as_given(truth))
        return len(counted)

    assert calls(ROUNDED) == calls(lambda x, seed: x * 3)
    with pytest.raises(AssertionError):
        assert_as_exact(lambda x, seed: jnp.nextafter(x * 3, jnp.inf), ROUNDED, _tripled,
                        "an ulp past x thrice", seeds=8, truth_fn=as_given(truth))


def test_rounded_is_refused_where_it_would_run_or_round_no_one_truth():
    """`ROUNDED` is never run: a sizing or null run of it against itself (no arm of Dew's beside it), a site
    without the truth it rounds, a further reference beside no truth, a weighed sizing (its reference's own
    rounding), and ROUNDED adapted (a run-to-run declaration, a window loop, which would call it), are each
    refused before any arm is called."""
    def truth(x, seed):
        return np.asarray(x, np.float64) * 3.0

    given = as_given(truth)
    for refused in (
            lambda: seeds_for(ROUNDED, _tripled, "itself", truth_fn=given),
            lambda: null_runs(ROUNDED, _tripled, "null runs", seeds=8, runs=2, truth_fn=given),
            lambda: assert_as_exact(lambda x, seed: x * 3, ROUNDED, _tripled, "no truth", seeds=8),
            lambda: assert_as_exact(lambda x, seed: x * 3, lambda x, seed: x * 3, _tripled, "further",
                                    seeds=8, also={"rounded": ROUNDED}),
            lambda: seeds_for(ROUNDED, _tripled, "weighed", truth_fn=given, dew_fn=lambda x, seed: x * 3,
                              conditioning=True),
            lambda: assert_as_exact(lambda x, seed: x * 3, run_to_run(ROUNDED, "a declaration"), _tripled,
                                    "declared", seeds=8, truth_fn=given),
            lambda: assert_as_exact(lambda x, seed: x * 3, over_windows(ROUNDED), _tripled, "windowed",
                                    seeds=8, truth_fn=given)):
        with pytest.raises(ValueError):
            refused()


def test_a_sites_held_condition_meets_every_result_of_dews_arm_and_fails_it():
    """A site's `held(outputs, inputs)` is called on each result of Dew's arm the gate draws, its draws'
    every seed among them, and its failure is the site's."""
    seen = set()

    def held(outputs, inputs):
        np.testing.assert_array_equal(np.asarray(outputs), np.asarray(inputs[0]) * 3)
        seen.add(int(inputs[1]))

    assert_as_exact(lambda x, seed: x * 3, lambda x, seed: x * 3, _tripled, "x thrice, held", seeds=8,
                    held=held)
    assert seen == set(range(8))

    class Refused(AssertionError):
        """The condition's failure."""

    def refusing(outputs, inputs):
        raise Refused

    with pytest.raises(Refused):
        assert_as_exact(lambda x, seed: x * 3, lambda x, seed: x * 3, _tripled, "x thrice, refused",
                        seeds=8, held=refusing)


def test_a_torch_arm_pulls_back_the_cotangent_it_is_given():
    """`torch_with_gradients` gives, bit for bit, a torch function's output and the gradients torch's
    autograd pulls back from the cotangent passed last, cast to the output's dtype, at float32 and at
    float64; a module's parameters are inputs through `functional_call`; an input the output does not
    reach has a zero gradient."""
    from reference_error import torch_with_gradients

    rng = np.random.default_rng(0)
    for dtype in (torch.float32, torch.float64):
        layer = torch.nn.Linear(3, 4).to(dtype)
        x, unused = (rng.normal(size=shape) for shape in ((5, 3), (2,)))
        params = {name: value.detach().numpy() for name, value in layer.named_parameters()}
        x, unused = x.astype(params["weight"].dtype), unused.astype(params["weight"].dtype)
        # A float64 cotangent, cast to the output's dtype.
        cotangent = rng.normal(size=(5, 4))

        def forward(x, params, unused, layer=layer):
            return torch.tanh(torch.func.functional_call(layer, params, (x,)))

        out, grad_x, grads, grad_unused = torch_with_gradients(forward)(x, params, unused, cotangent)
        tensors = [torch.from_numpy(x).requires_grad_(), *(torch.from_numpy(v).requires_grad_()
                                                           for v in params.values())]
        expected = torch.tanh(torch.func.functional_call(layer, dict(zip(params, tensors[1:], strict=True)),
                                                         (tensors[0],)))
        wanted = torch.autograd.grad(expected, tensors, torch.from_numpy(cotangent).to(dtype))
        assert out.dtype == grad_x.dtype == params["weight"].dtype
        np.testing.assert_array_equal(out, expected.detach().numpy())
        for got, want in zip((grad_x, *(grads[name] for name in params)), wanted, strict=True):
            np.testing.assert_array_equal(got, want.numpy())
        np.testing.assert_array_equal(grad_unused, np.zeros_like(unused))


def test_a_torch_arm_pulls_a_scalar_loss_back_from_its_scalar_one():
    """A scalar input takes a scalar gradient, its shape and dtype kept."""
    from reference_error import torch_with_gradients

    loss, gradient = torch_with_gradients(lambda a: 3 * a * a)(np.float32(2), np.float32(1))
    assert loss.dtype == gradient.dtype == np.float32 and np.shape(gradient) == ()
    np.testing.assert_array_equal(loss, np.float32(12))
    np.testing.assert_array_equal(gradient, np.float32(12))


def test_a_torch_arm_pulls_nothing_back_from_a_constant():
    """A constant output pulls nothing back, alone, or beside a reached one whose own cotangent it then
    takes, by its place in the tree."""
    from reference_error import torch_with_gradients

    x = np.array([2.0, 3.0], np.float32)
    _, gradient = torch_with_gradients(lambda x: torch.ones_like(x))(x, np.ones(2, np.float32))
    assert gradient.dtype == np.float32
    np.testing.assert_array_equal(gradient, np.zeros(2, np.float32))
    _, gradient = torch_with_gradients(lambda x: (torch.ones_like(x), x * 2))(
        x, (np.full(2, 7, np.float32), np.full(2, 100, np.float32)))
    np.testing.assert_array_equal(gradient, np.full(2, 200, np.float32))


def test_a_torch_arms_integer_input_has_no_gradient():
    """An integer input takes no gradient, where a float one would."""
    from reference_error import torch_with_gradients

    _, gradient = torch_with_gradients(lambda i: i.float() * 2)(np.array([1, 2], np.int32),
                                                                 np.ones(2, np.float32))
    assert gradient is None


def test_a_torch_arm_refuses_a_cotangent_of_another_tree():
    """A cotangent of another tree than the output's, a dict for a tuple or a list for one, is refused,
    where pairing by order would pull back the wrong leaf."""
    from reference_error import torch_with_gradients

    pulled = torch_with_gradients(lambda v: (v * 1, v * 2))
    x = np.ones(2, np.float32)
    for cotangent in ({"z": np.ones(2, np.float32), "a": np.ones(2, np.float32)}, [x, x]):
        with pytest.raises(ValueError):
            pulled(x, cotangent)


def test_a_torch_arm_pairs_each_cotangent_with_its_own_output():
    """Each cotangent pulls back through its own output, the first through the first."""
    from reference_error import torch_with_gradients

    _, gradient = torch_with_gradients(lambda v: (v * 1, v * 2))(
        np.ones(2, np.float32), (np.full(2, 10, np.float32), np.full(2, 100, np.float32)))
    assert gradient.dtype == np.float32
    np.testing.assert_array_equal(gradient, np.full(2, 210, np.float32))


def test_a_torch_arms_narrow_floats_go_in_through_float32():
    """A bfloat16 input and cotangent, as JAX holds them on the host, and a float8 input reach the arm
    in torch's format of their name, through the float32 that holds them exactly, and come out in
    float32."""
    from reference_error import torch_with_gradients

    def doubled(x):
        assert x.dtype == torch.bfloat16
        return x * 2

    halves = np.array([2.0, 3.0], jnp.bfloat16)
    out, gradient = torch_with_gradients(doubled)(halves, np.ones(2, jnp.bfloat16))
    assert out.dtype == gradient.dtype == np.float32
    np.testing.assert_array_equal(out, np.array([4.0, 6.0], np.float32))
    np.testing.assert_array_equal(gradient, np.full(2, 2, np.float32))
    # Torch computes no float8 on the CPU, so a float8 input is widened inside, as a quantized arm's is.

    def widened_inside(x):
        assert x.dtype == torch.float8_e4m3fn
        return x.float() * 2

    out, gradient = torch_with_gradients(widened_inside)(np.array([2.0, 3.0], jnp.float8_e4m3fn),
                                                         np.ones(2, np.float32))
    assert gradient.dtype == np.float32
    np.testing.assert_array_equal(gradient, np.full(2, 2, np.float32))


def test_a_torch_arms_narrow_integer_input_has_no_gradient():
    """An int4 input, which NumPy holds as ml_dtypes' too, is no float: it enters as int32, its values
    kept, and takes no gradient."""
    from reference_error import torch_with_gradients

    out, gradient = torch_with_gradients(lambda i: i.float() * 2)(np.array([1, -2], jnp.int4),
                                                                   np.ones(2, np.float32))
    assert gradient is None
    np.testing.assert_array_equal(out, np.array([2.0, -4.0], np.float32))


def test_a_torch_arms_bfloat16_output_comes_out_through_float32():
    """A bfloat16 output of float32 inputs comes back as the float32 that holds it, its values exactly."""
    from reference_error import torch_with_gradients

    x = np.arange(4, dtype=np.float32) / 3
    out, gradient = torch_with_gradients(lambda x: (x.bfloat16() * 3).bfloat16())(x, np.ones(4, np.float32))
    assert out.dtype == gradient.dtype == np.float32
    np.testing.assert_array_equal(out, (torch.from_numpy(x).bfloat16() * 3).float().numpy())
    np.testing.assert_array_equal(gradient, np.full(4, 3, np.float32))


NORMAL_SEEDS = 8
"""Draws of 4,096 recorded-site normals each, their deciles' counts pooled."""


def test_a_recorded_sites_normals_follow_the_normal_law():
    """`normals`, the draw a recorded site takes on every host alike, follows the standard normal law: the
    counts of its values in each tenth of the law's mass (the normal CDF of each, by `math.erf`), pooled
    over `NORMAL_SEEDS` draws, are as a fair sampler of ten equal chances draws them (the sampling gate)."""
    import math

    from recorded_inputs import normals
    from sampling_gate import assert_draws_as_declared

    def counts_of(seed):
        mass = 0.5 * (1 + np.vectorize(math.erf)(normals(seed, (4096,), np.float64) / math.sqrt(2)))
        return np.bincount(np.minimum((mass * 10).astype(np.int64), 9), minlength=10)

    assert_draws_as_declared(counts_of, np.full(10, 0.1), "recorded-site normals' tenths", seeds=NORMAL_SEEDS)


def test_a_small_row_beside_large_ones_is_held_to_its_own_size():
    """Dew doubles a row of 5s beside rows of 1e8 that the reference rounds by 8. Every
    tensor-wide distance passes it: its RMS error, 3.5, sits under twice the reference's, 5.7.
    Held to its own size, the row is 100% wrong against the reference's 8e-8."""
    truth = np.array([[1e8, 1e8], [5.0, 5.0]])
    reference = np.array([[1e8 + 8, 1e8 + 8], [5.0, 5.0]])
    dew = np.array([[1e8, 1e8], [10.0, 10.0]])
    assert distance(dew, truth) <= FACTOR * distance(reference, truth)
    with pytest.raises(AssertionError):
        assert_as_exact_as_the_reference(dew, reference, truth, "small row")


VECTOR_SEEDS = 512
"""Draws of a three-value vector: one draw in them erring eight units of float32 on every value moves the
pooled RMS ratio by sqrt(575 / 512) alone, where its row is eight times the reference's worst."""


@pytest.mark.parametrize("shape", [(3,), (1, 3)], ids=["a vector", "one row of a matrix"])
def test_a_vectors_values_are_one_row_held_by_the_worst_row_rule(shape):
    """A leaf of rank one is one row along its last axis, as a matrix's rows are (testing.md: a leaf's rows
    run along its last axis): three values tripled, Dew erring eight units of float32 on every value of
    the one draw of `VECTOR_SEEDS` whose values sum largest, fail check 1's worst-row rule laid as a vector
    as laid as one row of a matrix, the pooled ratio within PARITY; a leaf of rank one had no rows, and
    passed (row 6688)."""
    def draw(seed):
        return (np.random.default_rng(seed).standard_normal(shape).astype(np.float32),)

    sums = sorted(float(np.sum(np.asarray(draw(seed)[0], np.float64))) for seed in range(VECTOR_SEEDS))
    largest = (sums[-1] + sums[-2]) / 2

    # NumPy arms, as given: the erring draw is chosen on the host by its float64 sum.
    @as_given
    def tripled(x):
        return np.asarray(x, np.float32) * np.float32(3)

    @as_given
    def erring(x):
        made = np.asarray(x, np.float32) * np.float32(3)
        off = np.float32(1 + 8 * 2.0 ** -23)
        return made * off if np.sum(np.asarray(x, np.float64)) > largest else made

    @as_given
    def exact(x):
        return np.asarray(x, np.float64) * 3

    with pytest.raises(GateFailure):
        assert_as_exact(erring, tripled, draw, f"three values laid {shape}", seeds=VECTOR_SEEDS,
                        truth_fn=exact)


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
    assert_as_exact_as_the_reference(rounded(), reference, truth, "dq", terms=terms)
    assert distance(further, truth) <= FACTOR * distance(reference, truth)
    with pytest.raises(AssertionError):
        assert_as_exact_as_the_reference(further, reference, truth, "dq", terms=terms)


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
    assert distance(dew, truth) <= FACTOR * distance(reference, truth)
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


@jax.jit
def _broadcast(x, w):
    """The product with a singleton axis before it: the reference's values, not its shape."""
    return {"y": jnp.dot(x, w, precision="highest")[None]}


@jax.jit
def _renamed(x, w):
    return {"z": jnp.dot(x, w, precision="highest")}


def test_check_one_holds_each_leaf_to_its_truths_path_and_shape():
    """A result with a singleton axis added, whose values broadcast against the truth, or with its key
    renamed, whose values pair with the truth's by position, is not the reference's output: check 1
    fails it before any arithmetic. So does a reference whose shape is not its truth's."""
    for dew_fn in (_broadcast, _renamed):
        with pytest.raises(AssertionError):
            assert_as_exact(dew_fn, _product, _draw, "another tree", seeds=8)
    with pytest.raises(AssertionError):
        assert_as_exact(_product, _broadcast, _draw, "another reference", seeds=8, truth_fn=_product)


def test_a_committed_fixture_is_its_references_output_within_twice_a_fresh_runs_error():
    """An artifact check, not a parity (`assert_same_artifact`): a committed fixture whose largest error
    from float64 is up to twice a fresh run's is the reference's output; one further is not, even where
    only one element moves, which an RMS would let through, nor one not finite, or held to a fresh run
    that is not."""
    f64 = np.linspace(-1, 1, 101)
    fresh = f64 + 1e-7
    assert_same_artifact(fresh, f64 - 2e-7, f64, "twice as far")
    with pytest.raises(AssertionError):
        assert_same_artifact(fresh, f64 + 3e-7, f64, "three times as far")
    corrupted = f64.copy()
    corrupted[50] += 5e-7
    assert np.sqrt(np.mean(np.square(corrupted - f64))) < 1e-7
    with pytest.raises(AssertionError):
        assert_same_artifact(fresh, corrupted, f64, "one element moved")
    # Neither a NaN nor an infinity on either side is an evaluation of the reference.
    for committed, recomputed in ((np.full_like(f64, np.nan), fresh), (f64 + 1e-7, np.full_like(f64, np.inf)),
                                  (np.full_like(f64, np.inf), np.full_like(f64, np.inf))):
        with pytest.raises(AssertionError):
            assert_same_artifact(recomputed, committed, f64, "not finite")


def test_a_runs_learning_rates_part_from_the_base_at_any_float32_difference(monkeypatch):
    """compare.py's conditions read each run's learning rates as float32s, which must be the base run's
    exactly: a float64 schedule that rounds to the base's float32 matches it, and a value at the next
    float32 parts, which a relative 1e-6 let through."""
    from tool_support import REPO_ROOT, load

    monkeypatch.syspath_prepend(str(REPO_ROOT / "tools" / "reference_runs"))
    compare = load("reference_runs/compare")
    config = dict.fromkeys(compare.CONDITIONS, 1)
    base = np.float32([1e-3, 5e-3, 1e-2])
    rounding = base.astype(np.float64) + 1e-12
    next_up = np.nextafter(base, np.float32(1)).astype(np.float64)
    records = {"base": {"config": config, "lr": base.tolist()}}
    assert not compare.conditions({**records, "run": {"config": config, "lr": rounding.tolist()}})
    assert len(compare.conditions({**records, "run": {"config": config, "lr": next_up.tolist()}})) == 1


@pytest.mark.parametrize("wrong", [_lost_term, _bf16_operands], ids=["lost term", "bf16 operands"])
def test_check_one_fails_a_sum_that_differs_rather_than_rounds(wrong):
    """A lost term, and operands rounded to bf16, put the pooled ratio far past PARITY, and drawing
    REDRAW times as many seeds again does not bring it back."""
    with pytest.raises(AssertionError) as raised:
        assert_as_exact(wrong, _product, _draw, "wrong", seeds=seeds_for(_product, _draw, "product"))
    assert raised.type is GateFailure


def test_a_failing_site_raises_its_findings_as_data():
    """A lost term fails check 1's pooled ratio: its finding names the leaf, the rule and a measure past
    PARITY."""
    with pytest.raises(GateFailure) as raised:
        assert_as_exact(_lost_term, _product, _draw, "lost", seeds=seeds_for(_product, _draw, "product"))
    pooled = [finding for finding in raised.value.findings if finding.rule == "pooled"]
    assert pooled and all(finding.leaf.startswith("lost") and finding.measure > math.log(PARITY)
                          for finding in pooled)


def test_check_one_is_inconclusive_where_its_seeds_cannot_resolve_the_factor():
    """Errors set seed by seed, alike on every host: Dew's squared error is 4 on even seeds and 0.01 on
    odd ones, the reference's 1 on each, so the pooled ratio (1.42) is under PARITY but its interval
    straddles it and is wider than it over 8 seeds and over the second stage's 40 alike: inconclusive,
    never a pass. Its draws' paired ratios do not pass it either: their log ratios spread past
    QUALIFYING_SPREAD (`_typical_status`, ledger 2469). Fewer seeds than FEWEST_SEEDS are inconclusive
    whatever they read."""
    from reference_error import _interval

    from dew.parity import _status, _typical_status, _Undecided
    def errors(seed):
        return (np.zeros(64, np.float32), np.full(64, 2.0 if seed % 2 == 0 else 0.1, np.float32),
                np.ones(64, np.float32))

    # NumPy arms, as given: the error law is set on the host, seed by seed.
    @as_given
    def dew(x, mine, theirs):
        return {"y": x + mine}

    @as_given
    def reference(x, mine, theirs):
        return {"y": x + theirs}

    def truth(x, mine, theirs):
        return {"y": x}

    # Each seed's squared errors' logs (`dew.parity.log_errors`), as the gate carries them.
    mine, theirs = np.log(np.asarray([4.0, 0.01] * 20)), np.zeros(40)
    assert _status(math.sqrt(2.005), *_interval(mine, theirs)) == "inconclusive"
    # The typical draw's ratio passes: it is the draws' spread that leaves the leaf undecided.
    status, *typical, _, undecided = _typical_status(mine, theirs)
    assert status == "inconclusive" and _status(*typical) == "pass" and undecided is _Undecided.SPREAD
    with pytest.raises(Inconclusive):
        assert_as_exact(dew, reference, errors, "straddling", seeds=8, truth_fn=truth)
    with pytest.raises(Inconclusive):
        assert_as_exact(_reordered, _product, _draw, "two seeds", seeds=2)
    # And seeds_for never declares fewer, whatever it is asked to start from.
    assert seeds_for(_product, _draw, "product", start=1) >= 8


def _set_errors(even: float, odd: float):
    """A draw whose Dew's squared error is `even` on even seeds and `odd` on odd ones, the reference's 1
    on each: a site's error law set seed by seed, alike on every host."""
    def errors(seed):
        return (np.zeros(64, np.float32), np.full(64, np.sqrt(even if seed % 2 == 0 else odd), np.float32),
                np.ones(64, np.float32))

    # NumPy arms, as given: the error law is set on the host, seed by seed.
    @as_given
    def dew(x, mine, theirs):
        return {"y": x + mine}

    @as_given
    def reference(x, mine, theirs):
        return {"y": x + theirs}

    def truth(x, mine, theirs):
        return {"y": x}

    return dew, reference, errors, truth


def test_check_one_decides_at_its_second_stage_what_its_first_leaves_inconclusive(tmp_path, monkeypatch):
    """Dew's squared error 2.6 on even seeds and 0.1 on odd ones, the reference's 1: over 8 seeds the
    pooled ratio's interval straddles PARITY and is wider than it; the second stage's 40 resolve it
    under PARITY, so the site passes, and its stage is logged, as every judgement's is."""

    from check1_judge import STAGES_LOG
    from reference_error import _interval, _pooled

    from dew.parity import _status

    dew, reference, errors, truth = _set_errors(2.6, 0.1)
    first = np.log(np.array([2.6 if seed % 2 == 0 else 0.1 for seed in range(8)]))
    assert _status(_pooled(first, np.zeros(8)), *_interval(first, np.zeros(8))) == "inconclusive"
    log = tmp_path / "stages.jsonl"
    monkeypatch.setenv(STAGES_LOG, str(log))
    assert_as_exact(dew, reference, errors, "second stage", seeds=8, truth_fn=truth)
    assert_as_exact(_reordered, _product, _draw, "first stage", seeds=8)
    logged = {entry["label"]: (entry["stage"], entry["verdict"])
              for entry in map(json.loads, log.read_text().splitlines())}
    assert logged["second stage"] == (2, "pass") and logged["first stage"] == (1, "pass")


def test_a_site_draws_its_second_stage_for_every_reference_it_compares(tmp_path, monkeypatch):
    """Dew's squared error 2.6 on even seeds and 0.1 on odd ones against a reference of the same errors (its
    ratio 1 at every seed, a pass at the first stage) and a further one of error 1 (inconclusive over 8
    seeds): the further reference draws the second stage, and the site draws it for both, each judged on
    the 40 seeds, where the first comparison stopped at its 8 and its added rows went unjudged (spec
    alignment, 9006)."""
    from check1_judge import STAGES_LOG

    dew, reference, errors, truth = _set_errors(2.6, 0.1)

    @as_given
    def alike(x, mine, theirs):
        return {"y": x + mine}

    log = tmp_path / "stages.jsonl"
    monkeypatch.setenv(STAGES_LOG, str(log))
    assert_as_exact(dew, alike, errors, "staged together", seeds=8, truth_fn=truth,
                    also={"constant": reference})
    assert [json.loads(line)["stage"] for line in log.read_text().splitlines()] == [2, 2]


def test_a_leafs_interval_is_widened_to_hold_its_ratio():
    """A percentile interval that leaves out its own ratio is widened to hold it before any verdict: a
    ratio over PARITY under an interval wholly at or under it is no pass, and a ratio that is not a
    number is inconclusive."""
    from dew.parity import _status

    assert _status(1.6, 1.0, 1.4) != "pass"
    assert _status(0.9, 1.1, 1.4) == "pass"
    assert _status(1.7, 1.6, 1.8) == "over"
    assert _status(float("nan"), 1.0, 1.2) == "inconclusive"
    # The rule before the two stages read the interval as it was.
    assert _status(1.6, 1.0, 1.4, contained=False) == "over"
    assert _status(1.2, 0.5, 1.6, contained=False) == "inconclusive"


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
    assert raised.type is GateFailure


@pytest.mark.parametrize("shape", [(), (3,), (2, 3)], ids=["scalar", "vector", "matrix"])
def test_check_one_holds_dew_exact_where_the_reference_is_exact_below_squares_float64_holds(shape):
    """A float64 reference exact on a truth of 1e-200, and a Dew arm that returns zero there: Dew's error
    squares to zero in float64 and was read as exact (row 4498); it misses the exact reference, so the
    site fails, at a leaf of every rank."""
    def draw(seed):
        return (np.full(shape, 1e-200 * (1 + seed)),)

    with pytest.raises(AssertionError) as raised:
        assert_as_exact(as_given(lambda x: {"y": np.zeros_like(x)}), as_given(lambda x: {"y": x}), draw,
                        "below squares", seeds=8, truth_fn=as_given(lambda x: {"y": x}))
    assert raised.type is GateFailure


@pytest.mark.parametrize("rank", [1, 2])
def test_check_one_fails_a_value_that_is_not_finite(rank):
    """A NaN in Dew's leaf fails, whatever the pooled errors would read, a vector's (which has no rows
    to catch it) as a matrix's."""
    def shaped(fn):
        return jax.jit(lambda x, w: {"y": fn(x, w)["y"].reshape(-1) if rank == 1 else fn(x, w)["y"]})

    poisoned = shaped(lambda x, w: {"y": jnp.dot(x, w, precision="highest").at[0, 0].set(jnp.nan)})
    with pytest.raises(AssertionError) as raised:
        assert_as_exact(poisoned, shaped(_product), _draw, "poisoned", seeds=8)
    assert raised.type is GateFailure


def test_check_one_holds_a_row_of_float64_zero_to_the_references_own():
    """A row the float64 truth holds at exactly zero, the reference too, is no further from zero in Dew
    than ROW_FACTOR times the reference on that row: a corruption too small to move the pooled error
    still fails there."""
    def zero_row(offset):
        return jax.jit(lambda x, w: {"y": jnp.concatenate(
            [jnp.dot(x, w, precision="highest"), jnp.full((1, w.shape[1]), offset, jnp.float32)])})

    with pytest.raises(AssertionError) as raised:
        assert_as_exact(zero_row(1e-12), zero_row(0.0), _draw, "zero row", seeds=8)
    assert raised.type is GateFailure


def _tiny_row(dew_ulps):
    """A product's rows and one row of inputs scaled by 2^-60, about 1e-18 of the product: exact in float32
    and float64 alike, so the reference holds it exactly and float64 to its own precision. Dew moves it
    by `dew_ulps` float32 units relative."""
    unit = float(np.finfo(np.float32).eps)

    def run(ulps):
        def product(x, w):
            y = jnp.dot(x, w, precision="highest")
            tiny = x[:1, :w.shape[1]] * 2.0 ** -60
            if ulps:
                # One rounding of the exact product, in either arm's dtype (1 + ulps units is exact in both).
                tiny = tiny * jnp.asarray(1 + ulps * unit, tiny.dtype)
            return {"y": jnp.concatenate([y, tiny])}
        return product

    return jax.jit(run(dew_ulps)), jax.jit(run(0))


def test_a_small_row_the_reference_measures_is_judged_at_its_own_size():
    """A row 1e-18 of its leaf, which float64 holds to its own precision, lies below RESOLUTION of the
    leaf; the reference carries it exactly, so it is no row of float64 zero: Dew four float32 units off
    there is ordinary rounding judged at the row's own size, as at every row, and passes; four thousand
    units off fails the worst-row rule. The row is held against the reference's worst row over the leaf,
    not its own error on the row, which is exact: the zero rows' rule failed Dew at four units, the
    worst-row rule admits Dew's row up to ROW_FACTOR times the leaf's worst."""
    dew, reference = _tiny_row(4)
    assert_as_exact(dew, reference, _draw, "tiny row", seeds=8)
    further, reference = _tiny_row(4000)
    with pytest.raises(AssertionError) as raised:
        assert_as_exact(further, reference, _draw, "tiny row", seeds=8)
    assert raised.type is GateFailure


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


def test_an_input_digest_tells_apart_a_scalar_from_its_vector_and_two_formats_of_one_width():
    """Inputs drawn again in another shape or format with the same bytes are not the recorded ones."""
    assert input_digest({"x": np.float32(1)}) != input_digest({"x": np.ones(1, np.float32)})
    one = np.array([1.0, 2.0], jnp.float8_e4m3fn)
    assert input_digest({"x": one}) != input_digest({"x": one.view(jnp.float8_e4m3fnuz)})
    strided = np.arange(12, dtype=np.float32).reshape(3, 4)[:, ::2]
    assert input_digest({"x": strided}) == input_digest({"x": np.ascontiguousarray(strided)})


def test_null_runs_judge_the_reference_against_itself_on_draws_no_other_run_takes():
    """Each null run is check 1 on the reference against itself, its two sides on disjoint draws and
    every run on draws of its own; an equally exact pair passes, and the red rate's bound is the
    one-sided Clopper-Pearson one, not one over the runs."""
    taken = []

    def draw(seed):
        taken.append(seed)
        return _draw(seed)

    verdicts = null_runs(_product, draw, "null", seeds=8, runs=3)
    assert [verdict for verdict, _, _ in verdicts] == ["pass"] * 3

    # Draws of one truth (an orbit's) pair their rows of float64 zero: Dew's residue on a row the
    # reference holds at zero is no pass for drawing the reference elsewhere.
    def one_truth(seed):
        return np.ones(4, np.float32), np.float32(seed % 2)

    def exact(x, s):
        return {"y": jnp.stack([x * (1 + 1e-7), 0 * x])}

    def residue(x, s):
        return {"y": jnp.stack([x * (1 + 1e-7), 1e-12 * s * x])}

    def truth(x, s):
        return {"y": jnp.stack([x, 0 * x])}

    with pytest.raises(AssertionError):
        assert_as_exact(residue, exact, one_truth, "orbit", seeds=8, truth_fn=truth)
    [(verdict, _, _)] = null_runs(exact, one_truth, "orbit", seeds=8, runs=1, truth_fn=truth, dew_fn=residue)
    assert verdict == "over"
    # Each run draws its first seed again, the check that its seed fixes it (`_of_its_seed`); its draws are
    # otherwise its own.
    drawn = [seed for at, seed in enumerate(taken) if not at or taken[at - 1] != seed]
    assert len(drawn) == len(set(drawn)) and len(taken) - len(drawn) == 3
    assert null_rate_bound(0, 1400) == pytest.approx(1 - 0.001 ** (1 / 1400), rel=1e-9)
    assert null_rate_bound(0, 200) > 0.03
    assert null_rate_bound(3, 3) == 1.0


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

    recorded = recorder.record_case("site", draw, evaluate, summary=True)
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
    from truth_sketch import SKETCH, _sign_bits, sketch_rounding, truth_sketch

    from dew.parity import leaf_summary

    rng = np.random.default_rng(7)
    truth = rng.normal(size=4096)
    signs = ((_sign_bits(truth.size, 3)[:, None] >> np.arange(SKETCH, dtype=np.uint64)) & 1) * 2.0 - 1
    # Another host: each block summed one by one from its end, the blocks' sums summed exactly.
    blocks = np.flip((truth[:, None] * signs).reshape(-1, 64, SKETCH), axis=1).cumsum(axis=1)[:, -1]
    other_order = np.asarray([math.fsum(blocks[:, index].tolist()) for index in range(SKETCH)])
    allowed = 2 * sketch_rounding(float(np.sum(np.abs(truth))))
    assert np.all(np.abs(truth_sketch(truth, 3) - other_order) <= allowed)
    empty = leaf_summary(jnp.zeros((16, 0), jnp.bfloat16), np.zeros((16, 0)))
    assert (empty.log_mse, empty.worst, empty.finite, empty.dtype) == (-np.inf, 0.0, True, "bfloat16")
    with pytest.raises(AssertionError):
        leaf_summary(np.zeros(0), np.ones(3))
    # Two entries swapped are seen at a seed unless their signs agree in all SKETCH places.
    swapped = truth.copy()
    swapped[[0, 1]] = swapped[[1, 0]]
    bits = _sign_bits(truth.size, 3)
    assert (bits[0] == bits[1]) or np.any(truth_sketch(swapped, 3) != truth_sketch(truth, 3))


def test_the_procedure_before_the_two_stages_draws_again_as_it_did():
    """Compared against, the procedure before check 1's two stages (`_staged(two_stage=False)`) draws
    the second set wherever a leaf's ratio is over PARITY, a value not finite included, and only there:
    an inconclusive leaf is final at its declared seeds."""
    from dew.parity import NO_ZERO_ROWS, _Draws, _staged

    def law(dew):
        def drawn(start, stop):
            count = stop - start
            # Each seed's squared errors' logs: Dew's `dew`, the reference's one.
            return _Draws(["leaf"], np.full((count, 1), np.log(dew)), np.zeros((count, 1)),
                          np.zeros((count, 1, 2)), [[NO_ZERO_ROWS]] * count,
                          np.full((count, 1), np.isfinite(dew)), (0,))
        return drawn

    with np.errstate(invalid="ignore"):
        assert _staged(law(np.inf), 8, "infinite", two_stage=False)[2] == 2
        assert _staged(law(np.inf), 8, "infinite")[2] == 2
    assert _staged(law(1.0), 8, "equal", two_stage=False)[2] == 1


def test_a_change_moves_the_result_only_past_rounding():
    """`assert_moves`: a product scaled by 1 + 2^-10 moves it far past its rounding and passes; the same
    product summed in another order, a change rounding accounts for, does not; a leaf the original holds
    exactly moves on any departure."""
    def scaled(x, w):
        return {"y": jnp.dot(x, w, precision="highest") * (1 + 2.0 ** -10)}

    assert_moves(_product, jax.jit(scaled), _draw, "scaled", seeds=8)
    with pytest.raises(AssertionError):
        assert_moves(_product, _reordered, _draw, "reordered", seeds=8)

    def zero(x, w):
        return {"y": jnp.zeros(3, x.dtype)}

    def off_zero(x, w):
        return {"y": jnp.full(3, 1e-30, x.dtype)}

    assert_moves(jax.jit(zero), jax.jit(off_zero), _draw, "off zero", seeds=8)


def test_a_change_whose_error_ratio_float64_cannot_hold_moves_and_is_over():
    """A scalar the original misses by 1e-320 and a change misses by 1e-5 on every seed: their RMS ratio
    passes float64's range (its log about 725), which the gate holds at float64's largest, so the change
    moves (`assert_moves`) and a Dew arm so far off is over (check 1), where the ratio read as an infinity
    made every bootstrap quantile a NaN and the change no move."""
    def off_by(offset):
        return as_given(lambda x: {"y": x + offset})

    exact = as_given(lambda x: {"y": x})

    # A truth of zero, so 1e-320 off it is a subnormal float64 keeps.
    def at_zero(seed):
        del seed
        return (np.zeros(()),)

    assert_moves(off_by(1e-320), off_by(1e-5), at_zero, "past float64's ratios", seeds=8, truth_fn=exact)
    with pytest.raises(AssertionError) as raised:
        assert_as_exact(off_by(1e-5), off_by(1e-320), at_zero, "past float64's ratios", seeds=8,
                        truth_fn=exact)
    assert raised.type is GateFailure


def test_a_change_moves_by_its_intervals_lower_end_and_one_leaf_is_enough():
    """Errors set seed by seed: a changed RMS error of 1.51 times the original's on every seed (its
    interval that one value) moves; 1.49 does not, and is no move however its seeds are drawn; with two
    leaves, one moved is enough; a leaf not finite moves nothing."""
    for factor, moves in ((1.51, True), (1.49, False)):
        changed, original, errors, truth = _set_errors(factor ** 2, factor ** 2)
        if moves:
            assert_moves(original, changed, errors, "constant", seeds=8, truth_fn=truth)
        else:
            with pytest.raises(AssertionError):
                assert_moves(original, changed, errors, "constant", seeds=8, truth_fn=truth)

    dew, reference, errors, truth = _set_errors(4.0, 4.0)

    @as_given
    def two(x, mine, theirs):
        return {"moved": dew(x, mine, theirs)["y"], "kept": reference(x, mine, theirs)["y"]}

    @as_given
    def two_original(x, mine, theirs):
        return {"moved": reference(x, mine, theirs)["y"], "kept": reference(x, mine, theirs)["y"]}

    def two_truth(x, mine, theirs):
        return {"moved": x, "kept": x}

    assert_moves(two_original, two, errors, "one leaf", seeds=8, truth_fn=two_truth)

    @as_given
    def infinite(x, mine, theirs):
        return {"y": x + np.inf}

    with pytest.raises(AssertionError):
        assert_moves(reference, infinite, errors, "infinite", seeds=8, truth_fn=truth)


def test_a_row_of_float64_zero_is_judged_over_the_runs_seeds():
    """A row of float64 zero is held to the reference's residue on it pooled over the run's seeds
    (`pooled_zero_rows`): one seed's residue far past the reference's own, where the other seeds' are
    alike, passes; Dew ten times the reference's residue on every seed fails, and so does a residue
    where the reference holds the row at exactly zero on every seed."""
    from reference_error import ROW_FACTOR

    from dew.parity import pooled_zero_rows

    rows = np.asarray([3, 7])

    def seed(mine, theirs):
        return rows, np.asarray(mine, np.float64), np.asarray(theirs, np.float64)

    alike = [seed([1.0, 2.0], [1.0, 2.0]) for _ in range(7)]
    assert pooled_zero_rows([*alike, seed([1.0, 2.0 * ROW_FACTOR * 1.5], [1.0, 2.0])]) == []
    tenfold = [seed([10.0, 2.0], [1.0, 2.0]) for _ in range(8)]
    assert [row for row, _, _ in pooled_zero_rows(tenfold)] == [3]
    exact = [seed([0.0, 1e-30], [0.0, 0.0]) for _ in range(8)]
    assert [row for row, _, _ in pooled_zero_rows(exact)] == [7]
    # A row zero on some seeds alone pools over those.
    assert pooled_zero_rows([seed([1.0, 1.0], [1.0, 1.0]), (np.asarray([3]), np.asarray([5.0]),
                                                             np.asarray([1.0]))]) == []


@pytest.mark.parametrize("size", [1, 63, 64, 65, 3000, (1 << 16) + 70])
def test_a_truths_sketch_is_its_exact_signed_sums_within_its_rounding(size):
    """`truth_sketch` is within `sketch_rounding` of its definition's exact value, each entry times its
    seed's SKETCH signs (bit k of its word, +1 where set) and every product summed exactly: the most a
    block summed in any order may round by, which the gate allows each of two sketches (`_summarized`).
    Over magnitudes across 60 decades, signed zeros and float64 subnormals, at sizes off and on a block's
    edge and past a host chunk, where the blocks round (tests/test_truth_sketch.py holds the definition
    bit for bit where they do not)."""
    import math

    from truth_sketch import SKETCH, _sign_bits, sketch_rounding, truth_sketch

    rng = np.random.default_rng(size)
    values = rng.standard_normal(size) * 10.0 ** rng.uniform(-30, 30, size)
    values[::7] = -0.0
    values[1::11] = 2.0 ** -1060
    signs = ((_sign_bits(size, 3)[:, None] >> np.arange(SKETCH, dtype=np.uint64)) & 1) * 2.0 - 1
    exact = np.asarray([math.fsum(column) for column in (values[:, None] * signs).T.tolist()])
    allowed = sketch_rounding(math.fsum(np.abs(values).tolist()))
    assert np.all(np.abs(truth_sketch(values, 3) - exact) <= allowed)


def _bits(value):
    """`value` held to its bits: an array's or a float's dtype, shape and bytes (a NaN's and a zero's sign
    included), a summary's field by field, a tuple's item by item."""
    import dataclasses

    if dataclasses.is_dataclass(value):
        return [(field.name, _bits(getattr(value, field.name))) for field in dataclasses.fields(value)]
    if isinstance(value, (tuple, list)):
        return [_bits(item) for item in value]
    if value is None or isinstance(value, (bool, str)):
        return value
    array = np.asarray(value)
    return array.dtype.str, array.shape, array.tobytes()


def test_a_leaf_in_any_layout_reads_as_its_c_ordered_copy():
    """Check 1 reads a leaf laid out otherwise, F-ordered or strided, Dew's, the reference's and the
    truth's, as its C-ordered copy, bit for bit (`_evidence`, `leaf_summary`): every leaf lies in C order
    beside the others of its width (`_LaidTruths`), so its sums run alike whatever its layout. The leaves
    hold rows of float64 zero and leaves of every rank."""
    from dew.parity import _evidence, _prepared, leaf_summary

    rng = np.random.default_rng(2201)
    shapes = [(70, 129), (9, 33), (5, 4, 9), (40,), (3, 2, 1), (12, 9), ()]
    truth = {f"leaf{at}": rng.standard_normal(shape) * 10.0 ** rng.uniform(-3, 3)
             for at, shape in enumerate(shapes)}
    truth["leaf1"][2] = 0.0

    def arm():
        return {name: (value * (1 + 2.0 ** -20 * rng.standard_normal(value.shape))).astype(np.float32)
                for name, value in truth.items()}

    def strided(array):
        wide = np.zeros((*array.shape[:-1], 2 * array.shape[-1]), array.dtype)
        wide[..., ::2] = array
        return wide[..., ::2]

    def otherwise(tree):
        return {name: (np.asfortranarray, strided)[at % 2](value) if np.ndim(value) else value
                for at, (name, value) in enumerate(tree.items())}

    dew, reference, laid = otherwise(arm()), otherwise(arm()), otherwise(truth)
    assert not any(np.asarray(value).flags.c_contiguous for tree in (dew, reference, laid)
                   for value in tree.values() if np.ndim(value) >= 2)
    # Each copy its leaf's own shape: numpy lays a scalar out as one value of rank one.
    ordered = [{name: np.ascontiguousarray(value).reshape(np.shape(value)) for name, value in tree.items()}
               for tree in (dew, reference, laid)]

    def evidence(dew, reference, truth):
        return _evidence(_prepared(dew, truth), reference, truth, shared=True)

    assert _bits(evidence(dew, reference, laid)) == _bits(evidence(*ordered))
    for name in truth:
        assert _bits(leaf_summary(dew[name], laid[name])) == _bits(
            leaf_summary(ordered[0][name], ordered[2][name])), name


def test_a_leaf_read_from_its_rows_distances_is_the_leaf_read_whole():
    """A leaf recorded as each row's RMS distance from its float64 truth (`rows_summary`) reads as the leaf
    whole does (`leaf_summary`): its worst row, the rows its truth sizes zero, their errors and own sizes,
    its finiteness and dtype; and a leaf of no rows is exact, as `leaf_summary` reads one."""
    from dew.parity import _rows_rms, leaf_summary, rows_summary

    rng = np.random.default_rng(1848)
    truth = rng.standard_normal((6, 8))
    truth[2] = 0.0  # a row the truth sizes zero
    reference = (truth + rng.standard_normal((6, 8)) * 1e-6).astype(np.float32)
    whole = leaf_summary(reference, truth)
    read = rows_summary(_rows_rms(reference.astype(np.float64) - truth, 8), truth, finite=True,
                        dtype="float32", log_mse=whole.log_mse)
    for field in ("log_mse", "worst", "zero_rows", "zero_errors", "finite", "dtype", "zero_sizes",
                  "worst_log"):
        np.testing.assert_array_equal(getattr(read, field), getattr(whole, field), err_msg=field)
    empty, exact = (rows_summary(np.zeros(0), np.zeros((0, 4)), finite=True, dtype="float32"),
                    leaf_summary(np.zeros((0, 4), np.float32), np.zeros((0, 4))))
    for field in ("log_mse", "worst", "zero_rows", "finite"):
        np.testing.assert_array_equal(getattr(empty, field), getattr(exact, field), err_msg=field)


def rounding_apart(seed):
    """A draw whose two arms round apart, as two programs under fp32 sums do (ledger 2469): a value, its
    scale (lognormal, sigma 3, as a draw's conditioning sets it) and each arm's own rounding's size on it
    (lognormal, sigma 0.2, each arm's own, so the draws' log RMS ratios spread about 0.28)."""
    rng = np.random.default_rng([2469, seed])
    return (rng.standard_normal(64).astype(np.float32), np.float32(np.exp(rng.normal(0, 3.0))),
            np.exp(rng.normal(0, 0.2, 2)).astype(np.float32))


def rounded_apart(arm: int, factor: float = 1.0, *, every: int = 1):
    """`rounding_apart`'s arm `arm`: its value plus its own rounding, the draw's scale times its size times a
    fixed pattern of its own; Dew's `factor` times further on every `every`th draw's value (by its first
    element's bits)."""
    @as_given
    def run(x, scale, sizes):
        pattern = np.random.default_rng([arm, int(np.asarray(x).view(np.uint32)[0])]).standard_normal(x.shape)
        hit = int(np.asarray(x).view(np.uint32)[0]) % every == 0
        return np.asarray(x, np.float64) + scale * sizes[arm] * 1e-7 * pattern * (factor if hit else 1.0)
    return run


@as_given
def exact_value(x, scale, sizes):
    return np.asarray(x, np.float64)


def test_arms_rounding_apart_are_judged_by_the_typical_draw_and_a_fault_on_it_or_on_a_minority_is_refused():
    """Where two arms round apart (each draw's error the draw's scale, heavy-tailed, times each arm's own
    rounding), the pooled ratio does not resolve within MOST_SEEDS, and a final verdict it leaves
    inconclusive asks the draws' paired ratios (`_typical_status`, ledger 2469): an equally exact pair,
    its log ratios spread within QUALIFYING_SPREAD, passes at the count it sizes; a Dew twice off on every
    draw is refused by its typical draw; one ten times off on a tenth of its draws, which the typical
    draw's ratio leaves under PARITY, is refused by the spread its fault adds."""
    from dew.parity import _interval, _resolves, _status, _typical_status, _Undecided, log_errors

    reference = rounded_apart(1)
    seeds = seeds_for(reference, rounding_apart, "rounding apart", truth_fn=exact_value,
                      dew_fn=rounded_apart(0))
    # The pooled ratio does not resolve even at MOST_SEEDS: its sums are ruled by their largest draws.
    def squared(arm, seed):
        x, scale, sizes = rounding_apart(seed)
        return float(np.mean((arm(x, scale, sizes) - exact_value(x, scale, sizes)) ** 2))

    most = [[squared(arm, seed) for seed in range(MOST_SEEDS)] for arm in (rounded_apart(0), reference)]
    low, high = _interval(*(log_errors(errors) for errors in most))
    assert high > PARITY * low and seeds <= MOST_SEEDS
    assert_as_exact(rounded_apart(0), reference, rounding_apart, "rounding apart", seeds=seeds,
                    truth_fn=exact_value)
    with pytest.raises(GateFailure):
        assert_as_exact(rounded_apart(0, 2.0), reference, rounding_apart, "twice off", seeds=seeds,
                        truth_fn=exact_value)
    with pytest.raises(GateFailure):
        assert_as_exact(rounded_apart(0, 10.0, every=10), reference, rounding_apart, "a tenth ten times off",
                        seeds=seeds, truth_fn=exact_value)
    # Dew a hundred times the reference's squared error on 13 of 256 draws, equal on the rest: the pooled
    # ratio 2.46, unresolved, the typical draw's ratio under PARITY; the spread of its log ratios refuses it.
    mine, theirs = np.log(np.where(np.arange(256) < 13, 100.0, 1.0)), np.zeros(256)
    assert not _resolves(*_interval(mine, theirs))
    status, *typical, _, undecided = _typical_status(mine, theirs)
    assert status == "inconclusive" and _status(*typical) == "pass" and undecided is _Undecided.SPREAD


def test_a_leaf_the_second_stage_finds_over_is_over_whatever_its_first_stages_typical_draw():
    """The second stage is drawn by the pooled ratio alone, and its typical draw asked only of a final
    verdict the pooled ratio leaves inconclusive: a leaf whose first 32 seeds straddle PARITY (Dew's squared
    error 4 and 0.01 by turns, the reference's 1), whose typical draw would pass them, and whose further
    seeds read 4, is over at the second stage."""
    from dew.parity import NO_ZERO_ROWS, _Draws, _staged

    def drawn(start, stop):
        count = stop - start
        dew = np.asarray([4.0 if seed >= 32 or seed % 2 == 0 else 0.01 for seed in range(start, stop)])
        # Each seed's squared errors' logs: Dew's `dew`, the reference's one.
        return _Draws(["leaf"], np.log(dew)[:, None], np.zeros((count, 1)), np.zeros((count, 1, 2)),
                      [[NO_ZERO_ROWS]] * count, np.ones((count, 1), bool), (0,))

    over, unresolved, stage = _staged(drawn, 32, "over later")
    assert stage == 2 and [finding.leaf for finding in over] == ["over laterleaf"] and not unresolved


def stacked_examples(seed):
    """A draw of 16 examples whose two arms round apart, each example its own heavy-tailed scale (lognormal,
    sigma 3, shared by the arms) and each arm's own rounding's size on it (lognormal, sigma 0.4 on its log
    RMS error): the draw's RMS error is ruled by its worst example (ledger 2469)."""
    rng = np.random.default_rng([2469, 16, seed])
    return (rng.standard_normal((16, 32)).astype(np.float32),
            np.exp(rng.normal(0, 3.0, 16)).astype(np.float32),
            np.exp(rng.normal(0, 0.4, (2, 16))).astype(np.float32))


def stacked_arm(arm: int, factor: float = 1.0, every: int = 0):
    """`stacked_examples`' arm `arm`: each example's value plus its own rounding, the example's scale times
    its size times a fixed pattern of its own; Dew's `factor` times further on every `every`th example."""
    @as_given
    def run(x, scale, sizes):
        rng = np.random.default_rng([arm, int(np.asarray(x).view(np.uint32)[0, 0])])
        noise = rng.standard_normal(x.shape) * (scale * sizes[arm] * 1e-7)[:, None]
        if every:
            noise[::every] *= factor
        return {"rows": np.asarray(x, np.float64) + noise}
    return run


@as_given
def exact_rows(x, scale, sizes):
    return {"rows": np.asarray(x, np.float64)}


def test_a_site_that_stacks_examples_is_judged_by_its_examples_paired_ratios_and_refuses_a_fault_on_some():
    """At a site whose leaves stack independent examples (`assert_as_exact`'s `examples`), a leaf the pooled
    ratio cannot size, its draws ruled by their worst example, is judged by its examples' paired ratios
    (ledger 2469): an equally exact pair passes at the count it sizes; Dew three times off on a fifth of
    each draw's examples, which their draws' medians barely see, is refused by their tail about those
    medians."""
    from reference_error import _interval

    from dew.parity import log_errors

    reference = stacked_arm(1)
    seeds = seeds_for(reference, stacked_examples, "stacked", truth_fn=exact_rows, dew_fn=stacked_arm(0),
                      examples=True)
    # The pooled ratio does not resolve at that count: its draws are ruled by their worst examples.
    def squared(arm, seed):
        x, scale, sizes = stacked_examples(seed)
        return float(np.mean((arm(x, scale, sizes)["rows"] - np.asarray(x, np.float64)) ** 2))

    pooled = [[squared(arm, seed) for seed in range(seeds)] for arm in (stacked_arm(0), reference)]
    low, high = _interval(*(log_errors(errors) for errors in pooled))
    assert high > PARITY * low and seeds <= MOST_SEEDS
    assert_as_exact(stacked_arm(0), reference, stacked_examples, "stacked", seeds=seeds, truth_fn=exact_rows,
                    examples=True)
    with pytest.raises(GateFailure):
        assert_as_exact(stacked_arm(0, 3.0, every=5), reference, stacked_examples, "a fifth three times off",
                        seeds=seeds, truth_fn=exact_rows, examples=True)


def test_a_recorded_site_that_stacks_examples_is_judged_by_its_recorded_examples():
    """A site whose leaves stack examples, recorded as the reference's evidence alone (`record_case`'s
    summary), keeps each example's evidence, so check 1 over the recording judges Dew by its examples as
    it does live: an equally exact pair passes, and Dew three times off on a fifth of each draw's examples
    is refused."""
    from tool_support import load

    recorder = load("recorded_references")

    def draw(seed):
        return dict(zip(("x", "scale", "sizes"), stacked_examples(seed), strict=True))

    def evaluate(inputs):
        return stacked_arm(1)(**inputs), exact_rows(**inputs)

    recorded = recorder.record_case("stacked", draw, evaluate, summary=True, examples=True,
                                    dew=lambda inputs: stacked_arm(0)(**inputs))
    seeds = int(recorded["stacked.seeds"])
    drawn, reference, truth = indexed_recording(
        dict(recorded), "stacked", seeds, inputs_of=draw,
        truth_fn=lambda inputs: {"rows": jnp.asarray(inputs["x"], jnp.float64)})

    def dew(factor):
        return as_given(lambda index, inputs: stacked_arm(0, factor, every=5)(**inputs))

    assert_as_exact(dew(1.0), reference, drawn, "recorded stacked", seeds=seeds, truth_fn=truth,
                    examples=True)
    with pytest.raises(GateFailure):
        assert_as_exact(dew(3.0), reference, drawn, "recorded, a fifth three times off", seeds=seeds,
                        truth_fn=truth, examples=True)


def bits_apart():
    """A product whose last bits alternate from call to call, as a GPU reduction that adds with atomics in
    arrival order does: odd calls exact, even calls an ulp up (its calls counted in a list it closes over)."""
    calls = []

    def product(x, w):
        calls.append(len(calls))
        out = np.asarray(x, np.float32) @ np.asarray(w, np.float32)
        return out if len(calls) % 2 else np.nextafter(out, np.float32(np.inf))
    return as_given(product)


@pytest.mark.parametrize(("backend", "deterministic", "declared", "refused"), [
    ("gpu", False, False, False), ("gpu", True, False, True), ("cpu", False, False, True),
    ("gpu", True, True, False), ("cpu", False, True, True)])
@pytest.mark.parametrize("windowed", [False, True], ids=["one draw", "windowed"])
def test_check_1_holds_an_arm_to_a_bitwise_repeat_only_where_one_is_promised(
        monkeypatch, request, backend, deterministic, declared, refused, windowed):
    """Check 1 replays a site's arms on their first draw and requires their outputs again bit for bit only
    where the run promises a repeat (`_repeat_promised`, ledger 2842): on a GPU by default, whose reductions
    add with atomics in arrival order, an arm whose last bits move is not refused; on a GPU in deterministic
    mode (`with_deterministic_ops`), and on a CPU, it is (`UnpairedDraw`). An arm declared to vary run to run
    (`run_to_run`, row 4448) is judged by each draw's one run on a GPU in deterministic mode too, and held
    to its repeat on a CPU, whose interpreted programs are functions of their inputs; through a window loop
    (`over_windows`) and a null run's unrewritten reference alike."""
    from reference_error import UnpairedDraw, over_windows, run_to_run, windows

    monkeypatch.setattr(jax, "default_backend", lambda: backend)
    request.getfixturevalue("with_deterministic_ops" if deterministic else "without_deterministic_ops")
    label = f"bits apart on a {backend}" + (", deterministic" if deterministic else "")
    arm = run_to_run(bits_apart(), "atomic adds") if declared else bits_apart()
    truth, draw = jnp.matmul, a_product_draw
    if windowed:
        # Three windows a draw, so a rerun's calls alternate out of phase with the first run's.
        arm, truth, draw = over_windows(arm), over_windows(jnp.matmul), windows(a_product_draw, 3)
    if refused:
        with pytest.raises(UnpairedDraw):
            null_runs(arm, draw, label, seeds=8, runs=1, truth_fn=truth)
    else:
        null_runs(arm, draw, label, seeds=8, runs=1, truth_fn=truth)


def test_errors_whose_squares_float64_holds_as_zero_keep_their_ratio():
    """A float64 reference 1e-200 off a scalar truth and a Dew arm 1e-180 off it, over eight seeds: each
    squared error is below float64's least and read as one floor, so the two passed at ratio 1 where Dew is
    1e20 times the reference (row 6687). Each seed's error is carried as its log, so the site fails on its
    pooled ratio, the one rule a scalar has; a Dew arm as far off as the reference passes."""
    def draw(seed):
        # A scalar, which has no rows to judge (its pooled ratio alone does), far under either offset, so
        # each seed's error is its offset to ten digits.
        return (np.full((), (1 + seed) * 1e-211),)

    def off_by(offset):
        return as_given(lambda x: {"y": x + offset})

    with pytest.raises(AssertionError) as raised:
        assert_as_exact(off_by(1e-180), off_by(1e-200), draw, "far below squares", seeds=8,
                        truth_fn=as_given(lambda x: {"y": x}))
    assert raised.type is GateFailure
    assert_as_exact(off_by(1e-200), off_by(1e-200), draw, "as far below squares", seeds=8,
                    truth_fn=as_given(lambda x: {"y": x}))


def test_one_of_dews_own_paths_as_the_reference_is_one_program_with_dews_arm():
    """A reference declared one of Dew's own paths (`dew_path`), both arms one jitted function run as given
    (a decision head's `decide`, packed against alone): the gate runs it under the rewrite as Dew's arm, so
    the function is traced and compiled once for both, where an unrewritten reference traced it again under
    the pass switched off."""
    traced = []

    @jax.jit
    def decide(x):
        traced.append(x.shape)
        return jnp.tanh(x) * 2

    def draw(seed):
        return (np.random.default_rng(seed).standard_normal(8).astype(np.float32),)

    assert_as_exact(as_given(lambda x: decide(x)), dew_path(as_given(lambda x: decide(x))), draw,
                    "one function as both arms, a path of Dew's own", seeds=8,
                    truth_fn=as_given(lambda x: np.tanh(np.asarray(x, np.float64)) * 2))
    assert len(traced) == 1


class _Compiling:
    """A twin's TPU program whose compile raises `error`, or compiles where it is None, its run the TPU's."""

    def __init__(self, error):
        self.error = error

    def lower(self, *inputs, **options):
        return self

    def compile(self):
        if self.error is not None:
            raise self.error
        return self

    def __call__(self, *inputs, **options):
        return "the TPU's"


class _OnATpu(Written):
    """A twin as written (`Written`) whose TPU program compiles as `_Compiling` does."""

    def __init__(self, error):
        super().__init__(jnp.negative)
        self.error = error

    def program(self, platform: str):
        return _Compiling(self.error) if platform == "tpu" else super().program(platform)


def test_a_tpu_twin_as_written_runs_on_its_host_only_where_the_tpu_lowers_no_transpose():
    """A TPU call's twin as written runs on the TPU, its own arithmetic (rows 12813-12841: on its host, IEEE
    fp32 against a TPU's default-precision products, it failed 29 float32 cells), and on the host only where
    the TPU's compile raises XLA's UNIMPLEMENTED status for an HLO transpose (`TPU_TRANSPOSE`, rows
    12356-12361, 12118-12123, 12127); any other refusal is raised (`Written.ran_on`). A NumPy operand or
    option keys its call by its type, as a JAX array does (train 1285's v5e: a NumPy leaf was unhashable)."""
    from generation_support import on_a_v5e

    rows = jnp.ones((3,), jnp.float32)
    with on_a_v5e():
        assert _OnATpu(None)(rows) == "the TPU's"
    from arm_arithmetic import TPU_TRANSPOSE

    with on_a_v5e():
        assert _OnATpu(None).ran_on(rows) == "tpu"
        assert _OnATpu(jax.errors.JaxRuntimeError(f"{TPU_TRANSPOSE}.")).ran_on(rows) == "cpu"
        other = jax.errors.JaxRuntimeError("UNIMPLEMENTED: another operation is not implemented on TPU.")
        with pytest.raises(jax.errors.JaxRuntimeError) as raised:
            _OnATpu(other).ran_on(rows)
        assert raised.value is other
        # A NumPy operand and a NumPy scalar option, keyed by their types as JAX's arrays are: one compile.
        twin = _OnATpu(None)
        assert twin(np.ones((3,), np.float32), scale=np.float32(2)) == "the TPU's"
        assert twin(np.zeros((3,), np.float32), scale=np.float32(3)) == "the TPU's"
        assert len(twin._compiled) == 1
