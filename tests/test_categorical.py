"""The categorical distribution's one owner (`dew.nn.categorical`): every log-likelihood Dew takes
computes in at least float32 and keeps float64 logits in float64, masks the classes a row does not
hold, a class of no probability adding no term in value or gradient, and each form agrees with the
others; a label's log probability through the token owner (`dew.objectives.base.token_log_probs`) is
float64's own for a float64 model, and a ragged row's segments normalize as each segment alone."""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from confident_support import confident, label_derivative, log_softmax
from reference_error import assert_as_exact, with_gradients, without_ynn

from dew.nn import categorical
from dew.objectives.base import token_log_probs

MASKED_SEEDS = 8
"""Check 1's declared seeds for a masked row against the row of its held classes alone."""
COLLAPSE_SEEDS = {20: 8, 40: 8, 80: 8}
"""Check 1's declared seeds for a confident label's derivative at each margin, sized by `seeds_for`."""
CANCELLATION_SEEDS = 256
"""Check 1's declared seeds for the divergence's derivative near its reference, sized by `seeds_for`."""


def logits(dtype, shape=(3, 7), seed=0):
    return jnp.asarray(np.random.default_rng(seed).standard_normal(shape) * 4, dtype)


@pytest.mark.parametrize(("given", "computed"), [(jnp.bfloat16, jnp.float32), (jnp.float16, jnp.float32),
                                                 (jnp.float32, jnp.float32), (jnp.float64, jnp.float64)])
def test_each_form_computes_in_at_least_float32_and_keeps_float64(given, computed):
    with jax.enable_x64(new_val=True):
        rows = logits(given)
        labels = jnp.asarray([0, 3, 6])
        forms = [categorical.log_probs(rows), categorical.probabilities(rows),
                 categorical.cross_entropy(rows, labels), categorical.entropy(rows),
                 categorical.soft_cross_entropy(rows, jax.nn.one_hot(labels, 7, dtype=given)),
                 categorical.kl_divergence(rows, logits(given, seed=1)), token_log_probs(rows, labels),
                 categorical.rescaled(rows, jnp.exp(rows), rows + 1),
                 categorical.target_loss(rows, rows, rows),
                 *categorical.segment_log_partition(rows[0], jnp.asarray([0, 0, 1, 2, 1, 3, 3]), 3)[:1]]
        assert {form.dtype for form in forms} == {jnp.dtype(computed)}


def test_a_float64_models_label_log_probability_is_float64s_own():
    """A float64 model's token log probabilities are computed in float64, not rounded to float32 first:
    the label's log partition less its logit in float64 (Optax's form), which float32 would not give."""
    import optax

    with jax.enable_x64(new_val=True):
        rows = logits(jnp.float64) * 1e3 + 0.1
        labels = jnp.asarray([1, 2, 5])
        got = np.asarray(token_log_probs(rows, labels))
        wide = -np.asarray(optax.softmax_cross_entropy_with_integer_labels(rows, labels))
        np.testing.assert_array_equal(got, wide)
        narrow = -np.asarray(optax.softmax_cross_entropy_with_integer_labels(rows.astype(jnp.float32),
                                                                             labels))
        assert not np.array_equal(got, narrow.astype(np.float64))


def test_the_forms_agree_and_a_masked_class_takes_no_probability():
    """A one-hot soft target's cross entropy is minus the label's log probability, the entropy minus
    the expected log probability, the divergence Optax's of the two normalizations; a class `where`
    excludes takes no
    probability while the others hold it, and target mass on an excluded class is an infinite
    charge."""
    rows, labels = logits(jnp.float32), jnp.asarray([0, 3, 6])
    log_p = np.asarray(categorical.log_probs(rows))
    picked = np.take_along_axis(log_p, np.asarray(labels)[:, None], 1)[:, 0]
    np.testing.assert_array_equal(categorical.soft_cross_entropy(rows, jax.nn.one_hot(labels, 7)), -picked)
    np.testing.assert_array_equal(categorical.entropy(rows), -(jnp.exp(log_p) * log_p).sum(-1))
    # The divergence is Optax's of the two normalizations; from oneself it is a rounding's worth off
    # zero in fp32 (the softmax's cancellation), which the owner's divergence form takes up apart.
    np.testing.assert_array_equal(categorical.kl_divergence(rows, rows), optax.kl_divergence(
        categorical.log_probs(rows), categorical.probabilities(rows)))
    held = jnp.arange(7) < 4
    masked = np.asarray(categorical.probabilities(rows, held))
    np.testing.assert_array_equal(masked[:, 4:], 0)
    assert np.all(masked[:, :4] > 0)
    stray = jnp.zeros((3, 7)).at[:, 5].set(1.0)
    assert np.all(np.isinf(np.asarray(categorical.soft_cross_entropy(rows, stray, held))))


FORMS = {
    "cross entropy": lambda rows, where: categorical.cross_entropy(rows, jnp.asarray([0, 3, 2]), where),
    "divergence": lambda rows, where: categorical.kl_divergence(rows, rows[::-1] * 0.5, where),
    "entropy": lambda rows, where: categorical.entropy(rows, where),
}


@pytest.mark.parametrize("form", sorted(FORMS))
def test_a_masked_class_adds_no_term_in_value_or_gradient(form):
    """A form over a row whose last classes `where` excludes is the form over the held classes alone
    (check 1, both arms compiled alike), and its gradient is the held classes' with zeros at the
    excluded ones, where `0 * log 0` would put NaN; the entropy holds a -inf logit's class at no term
    as it holds an excluded one."""
    held = jnp.arange(7) < 4
    of = FORMS[form]
    masked = without_ynn(lambda rows: of(rows, held))
    alone = without_ynn(lambda rows: of(rows[:, :4], None))
    assert_as_exact(masked, alone, lambda seed: (logits(jnp.float32, seed=seed),), form, seeds=MASKED_SEEDS)
    rows = logits(jnp.float32)
    gradient = np.asarray(jax.grad(lambda rows: of(rows, held).sum())(rows))
    assert np.all(np.isfinite(gradient)) and np.all(gradient[:, 4:] == 0)
    if form == "entropy":
        filtered = jnp.where(held, rows, -jnp.inf)
        np.testing.assert_array_equal(categorical.entropy(filtered), categorical.entropy(rows, held))
        through = jax.grad(lambda rows: categorical.entropy(rows).sum())(filtered)
        assert np.all(np.isfinite(np.asarray(through)))


def test_a_ragged_rows_segments_each_normalize_as_the_segment_alone():
    """Each segment's log partition is the logsumexp of its entries alone (check 1), a segment of no
    entry's 0 and absent, and an entry of no segment (`count`) enters none."""
    segments = jnp.asarray([0, 2, 0, 4, 2, 2, 0])

    def partition(row):
        return categorical.segment_log_partition(row, segments, 4)[0][jnp.asarray([0, 2])]

    def alone(row):
        return jnp.stack([jax.nn.logsumexp(row[jnp.asarray([0, 2, 6])]),
                          jax.nn.logsumexp(row[jnp.asarray([1, 4, 5])])])

    assert_as_exact(without_ynn(partition), without_ynn(alone),
                    lambda seed: (logits(jnp.float32, (7,), seed),), "segment log partition",
                    seeds=MASKED_SEEDS)
    log_z, present = categorical.segment_log_partition(logits(jnp.float32, (7,)), segments, 4)
    np.testing.assert_array_equal(present, [True, False, True, False])
    np.testing.assert_array_equal(np.asarray(log_z)[[1, 3]], 0)


@pytest.mark.parametrize("margin", sorted(COLLAPSE_SEEDS))
def test_a_confident_labels_derivative_survives_and_sums_to_zero(margin):
    """Past the margin where the label's `p - 1` rounds to 0 in float32 (arXiv:2610.31642, eq. 4), the
    label loss's derivative, and a one-hot target's, is on par with the paper's eq. 7 from float64
    (check 1), every row of it summing to zero as float64's does, the label's own entry below zero; the
    loss is Optax's."""
    labels = jnp.zeros(3, jnp.int32)
    losses = {"label": lambda rows: categorical.cross_entropy(rows, labels),
              "one-hot target": lambda rows: categorical.soft_cross_entropy(rows, jax.nn.one_hot(labels, 7))}
    (rows,) = confident(margin, 0)
    for name, loss in losses.items():
        gradient = jax.jit(jax.grad(lambda rows, loss=loss: loss(rows).sum()))
        assert_as_exact(gradient, label_derivative, lambda seed: confident(margin, seed),
                        f"{name} at margin {margin}", seeds=COLLAPSE_SEEDS[margin])
        assert np.all(np.asarray(gradient(rows))[:, 0] < 0)
    np.testing.assert_array_equal(categorical.cross_entropy(rows, labels),
                                  optax.softmax_cross_entropy_with_integer_labels(jnp.asarray(rows), labels))


def written_softmax(rows: jax.Array) -> jax.Array:
    """The softmax written out: each row moved to its largest logit, exponentiated and normalized, its
    derivative autodiff's, through the maximum too."""
    moved = jnp.exp(rows - jnp.max(rows, axis=-1, keepdims=True))
    return moved / moved.sum(axis=-1, keepdims=True)


@pytest.mark.parametrize("margin", sorted(COLLAPSE_SEEDS))
def test_a_confident_rows_probabilities_differentiate_as_the_softmax_written_out(margin):
    """Past the margin where the most probable class's `p - 1` rounds to 0 in float32 (arXiv:2610.31642,
    eq. 4), the probabilities and their derivative under a drawn cotangent are on par from float64 with
    the softmax written out (check 1), whose derivative through the row's maximum gives the most
    probable class the others' sum, negated."""
    def draw(seed):
        (rows,) = confident(margin, seed)
        return rows, np.random.default_rng([margin, seed]).standard_normal(rows.shape).astype(np.float32)

    assert_as_exact(without_ynn(with_gradients(categorical.probabilities)),
                    without_ynn(with_gradients(written_softmax)), draw, f"probabilities at margin {margin}",
                    seeds=COLLAPSE_SEEDS[margin])


def near_reference(seed: int) -> tuple[np.ndarray, np.ndarray]:
    """A row of float32 logits a thousandth of a unit from its reference row's, and the reference."""
    rng = np.random.default_rng(seed)
    reference = rng.standard_normal((3, 7)) * 4
    return ((reference + 1e-3 * rng.standard_normal((3, 7))).astype(np.float32),
            reference.astype(np.float32))


def test_a_confident_soft_targets_derivative_keeps_its_shift_invariance():
    """Soft targets as confident as the logits, `[1, a]` against logits `[20, 0]` whose second probability
    is `a`, and the logits' own distribution: the derivative's entries sum to exactly 0, its derivative
    along all-ones, which shift invariance makes 0, and against its own distribution it is exactly 0,
    where the dominant class's other target mass taken as `T - t` rounds its entry to `-a`."""
    logits = jnp.asarray([20.0, 0.0], jnp.float32)
    own = jax.nn.softmax(logits)
    for targets in (jnp.asarray([1.0, own[1]]), own):
        def loss(row, targets=targets):
            return categorical.soft_cross_entropy(row, targets)

        _, along = jax.jvp(loss, (logits,), (jnp.ones(2, jnp.float32),))
        assert float(along) == 0.0
    np.testing.assert_array_equal(jax.grad(lambda row: categorical.soft_cross_entropy(row, own))(logits),
                                  np.zeros(2, np.float32))


def divergence_derivative(logits: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """`KL(q || p)`'s derivative in `logits` by arXiv:2610.31642's eq. 9 in NumPy, in the inputs' own
    dtype, its `p` the reference's softmax and its `q` the logits'."""
    log_q = log_softmax(reference)
    q, p = np.exp(log_q), np.exp(log_softmax(logits))
    delta = logits - reference
    d = delta - (q * delta).sum(axis=-1, keepdims=True)
    s = d - np.log1p((q * np.expm1(d)).sum(axis=-1, keepdims=True))
    u = q * np.expm1(s)
    return u - p * u.sum(axis=-1, keepdims=True)


def test_the_divergences_derivative_near_its_reference_keeps_the_differences_precision():
    """The divergence from a reference row is differentiated through the two rows' difference
    (arXiv:2610.31642, eq. 9): at the reference itself its derivative is exactly zero, and a thousandth
    of a unit away it is on par with the paper's form from float64 (check 1), where two softmaxes
    subtracted would keep only their own rounding."""
    gradient = jax.jit(jax.grad(lambda rows, reference: categorical.kl_divergence(rows, reference).sum()))
    rows = logits(jnp.float32)
    np.testing.assert_array_equal(gradient(rows, rows), np.zeros(rows.shape, np.float32))
    assert_as_exact(gradient, divergence_derivative, near_reference, "near the reference",
                    seeds=CANCELLATION_SEEDS)


def test_the_divergences_derivative_stays_finite_wherever_its_rows_are():
    """Rows each finite but far apart in offset have the derivative of the distributions they hold, here
    two uniform rows' zero; rows far apart in shape have a finite Hessian, and rows at opposite extrema,
    whose difference less its mean overflows, or so far apart that the difference's rounding lifts a
    reconstructed log probability above 0, the derivative `p - q` with one, and a uniform offset whose
    mean rounds the difference off 0 the derivative 0 with one; logits -inf where the
    reference is not take `p - q` in forward mode as in reverse; and a row whose classes `where`
    excludes entirely has derivative 0 beside a held row's."""
    far = jnp.asarray([[2e38, 2e38]], jnp.float32), jnp.asarray([[-2e38, -2e38]], jnp.float32)
    offset = jax.grad(lambda rows: categorical.kl_divergence(rows, far[1]).sum())(far[0])
    np.testing.assert_array_equal(offset, np.zeros((1, 2), np.float32))
    extrema = jnp.asarray([1e38, -1e38], jnp.float32), jnp.asarray([-1e38, 1e38], jnp.float32)
    # A difference so large that its rounding (512 at 6.4e9) lifts a reconstructed log probability above 0,
    # and a uniform offset of 26 classes whose mean under q rounds the difference off 0.
    coarse = jnp.asarray([0.0, 6442450432.0], jnp.float32), jnp.asarray([0.0, -384.0], jnp.float32)
    uniform = jnp.full((26,), 2.0 ** 26, jnp.float32), jnp.zeros((26,), jnp.float32)
    for (row, reference), expected in ((extrema, [1.0, -1.0]), (coarse, [-1.0, 1.0]), (uniform, [0.0] * 26)):
        for derivative in (jax.grad, jax.jacfwd):
            np.testing.assert_array_equal(
                derivative(lambda row, reference=reference: categorical.kl_divergence(row, reference))(row),
                expected)
    for row, reference in ((jnp.asarray([200.0, 0.0]), jnp.asarray([-200.0, 0.0])), extrema, coarse, uniform):
        def divergence(row, reference=reference):
            return categorical.kl_divergence(row, reference)

        for hessian in (jax.hessian, lambda f: jax.jacfwd(jax.jacrev(f)),
                        lambda f: jax.jacrev(jax.jacrev(f))):
            assert np.all(np.isfinite(np.asarray(hessian(divergence)(row))))
    student, reference = jnp.asarray([-jnp.inf, 0.0]), jnp.asarray([0.0, 0.0])
    _, forward = jax.jvp(lambda row: categorical.kl_divergence(row, reference), (student,),
                         (jnp.asarray([0.0, 1.0]),))
    backward = jax.grad(lambda row: categorical.kl_divergence(row, reference))(student)
    assert forward == backward[1] == 0.5
    rows, held = logits(jnp.float32, (2, 7)), jnp.asarray([[True] * 7, [False] * 7])
    jacobian = np.asarray(jax.jacfwd(lambda rows: categorical.kl_divergence(rows, rows[::-1], held))(rows))
    assert np.all(np.isfinite(jacobian)) and np.all(jacobian[1] == 0)


GROUP_SEEDS = 16
"""Check 1's declared seeds for a group's divergence and its gradient."""


def drawn(seed: int):
    """Two windows of four question slots of five options each: two groups in the first window (one of
    three members, one of two whose second member maps one option to no canonical key), one in the
    second, an ungrouped question and a padded slot, logits wide enough to set the answers apart."""
    rng = np.random.default_rng([0xC0, seed])
    logits = jnp.asarray(rng.standard_normal((2, 4, 5)) * 3, jnp.float32)
    options = jnp.asarray([[[1, 1, 1, 0, 0], [1, 1, 1, 0, 0], [1, 1, 1, 1, 0], [1, 1, 1, 0, 0]],
                           [[1, 1, 0, 0, 0], [1, 1, 0, 0, 0], [1, 1, 1, 0, 0], [0, 0, 0, 0, 0]]], bool)
    groups = jnp.asarray([[0, 0, 1, 0], [0, 0, -1, -1]], jnp.int32)
    canonical = jnp.asarray([[[0, 1, 2, -1, -1], [2, 0, 1, -1, -1], [1, 0, -1, 2, -1], [1, 2, 0, -1, -1]],
                             [[0, 1, -1, -1, -1], [1, 0, -1, -1, -1], [-1, -1, -1, -1, -1],
                              [-1, -1, -1, -1, -1]]], jnp.int32)
    cotangent = tuple(jnp.asarray(rng.standard_normal((2, 4)), jnp.float32) for _ in range(2))
    return logits, options, groups, canonical, cotangent


def written_out(logits, options, groups, canonical):
    """The divergence written out: each member's softmax over its mapped slots summed onto the
    canonical options by a one-hot product, the group's mean, and each member's KL from it over its
    group's size; each member's share of its group its inverse size."""
    mapped = options & (canonical >= 0)
    any_mapped = jnp.any(mapped, -1, keepdims=True)
    peak = jnp.max(jnp.where(mapped, logits, -jnp.inf), -1, keepdims=True)
    p = jnp.where(mapped, jnp.exp(jnp.where(mapped, logits, 0.0) - jnp.where(any_mapped, peak, 0.0)), 0.0)
    p = p / jnp.where(any_mapped, jnp.sum(p, -1, keepdims=True), 1.0)
    onehot = (canonical[..., None] == jnp.arange(logits.shape[-1])) & mapped[..., None]
    answers = jnp.einsum("bqk,bqkc->bqc", p, onehot.astype(p.dtype))
    member = (groups >= 0) & any_mapped[..., 0]
    same = (groups[..., :, None] == groups[..., None, :]) & member[..., :, None] & member[..., None, :]
    size = jnp.sum(same, -1).astype(p.dtype)
    mean = jnp.einsum("bqr,brc->bqc", same.astype(p.dtype), answers) / jnp.maximum(size, 1)[..., None]
    counted = (answers > 0) & member[..., None]
    terms = jnp.where(counted, answers * (jnp.log(jnp.where(counted, answers, 1))
                                          - jnp.log(jnp.where(counted, mean, 1))), 0.0)
    return (jnp.where(member, jnp.sum(terms, -1) / jnp.maximum(size, 1), 0.0),
            jnp.where(member, 1 / jnp.maximum(size, 1), 0.0))


def test_the_group_divergence_is_its_groups_jensen_shannon_divergence():
    """Each member's share of its group's generalized Jensen-Shannon divergence, and its share of the
    group, are on par from float64 with the divergence written out (check 1), their gradients under
    drawn cotangents too; every member's shares of a group sum to its divergence and to one."""
    def dew(logits, options, groups, canonical, cotangent):
        return with_gradients(lambda logits: divergence_of(logits, options, groups, canonical))(
            logits, cotangent)

    def reference(logits, options, groups, canonical, cotangent):
        return with_gradients(lambda logits: written_out(logits, options, groups, canonical))(
            logits, cotangent)

    assert_as_exact(without_ynn(dew), without_ynn(reference), drawn, "group divergence",
                    seeds=GROUP_SEEDS)
    logits, options, groups, canonical, _ = drawn(0)
    divergence, members = (np.asarray(value) for value in categorical.group_divergence(logits, groups,
                                                                                       canonical, options))
    for window, group in ((0, 0), (0, 1), (1, 0)):
        held = np.asarray(groups[window]) == group
        count = int(held.sum())
        assert np.array_equal(members[window][held], np.full(count, np.float32(1) / np.float32(count)))
        assert divergence[window][held].sum() >= 0
    assert np.all(members[np.asarray(groups) < 0] == 0) and np.all(divergence[np.asarray(groups) < 0] == 0)


def agreeing(seed: int):
    """A group of two members that give one answer over the canonical options: the second shows the
    first's options in another order, its slots mapped back onto the first's canonical options."""
    rng = np.random.default_rng([0xA6, seed])
    first = rng.standard_normal(4) * 3
    order = rng.permutation(4)
    logits = jnp.asarray(np.stack([first, first[order]])[None], jnp.float32)
    canonical = jnp.asarray(np.stack([np.arange(4), order])[None], jnp.int32)
    return logits, jnp.ones((1, 2, 4), bool), jnp.asarray([[0, 0]], jnp.int32), canonical


def divergence_of(logits, options, groups, canonical):
    """`group_divergence` over the draws' arguments, in their order."""
    return categorical.group_divergence(logits, groups, canonical, options)


def test_members_that_agree_have_no_divergence_and_extreme_logits_stay_finite():
    """Members that give one answer over the canonical options, in other slot orders, have a divergence
    whose float64 truth is zero, met on par (check 1); logits of a thousand keep the divergence and its
    gradient finite."""
    assert_as_exact(without_ynn(divergence_of), without_ynn(written_out), agreeing,
                    "agreeing members", seeds=GROUP_SEEDS)
    logits, options, groups, canonical = agreeing(0)
    apart = canonical.at[0, 1].set(jnp.arange(4))
    divergence, _ = categorical.group_divergence(logits * 1000, groups, apart, options)
    gradient = jax.grad(lambda x: divergence_of(x, options, groups, apart)[0].sum())(logits * 1000)
    assert bool(jnp.all(jnp.isfinite(divergence))) and bool(jnp.all(jnp.isfinite(gradient)))


def test_a_member_of_no_group_or_of_no_mapped_class_takes_no_part():
    """A member in no group whose classes map, and a grouped member none of whose classes map, take no
    part: the group's other member alone has divergence zero and the whole share, the outsider none,
    and the divergence's derivative is finite and zero in both modes."""
    logits = jnp.zeros((1, 2, 2), jnp.float32)
    cases = ((jnp.asarray([[0, -1]]), jnp.asarray([[[0, 1], [0, 1]]])),
             (jnp.asarray([[0, 0]]), jnp.asarray([[[0, 1], [-1, -1]]])))
    for groups, classes in cases:
        divergence, shares = categorical.group_divergence(logits, groups, classes)
        np.testing.assert_array_equal(np.asarray(divergence), np.zeros((1, 2), np.float32))
        np.testing.assert_array_equal(np.asarray(shares), np.asarray([[1.0, 0.0]], np.float32))
        def total(x, groups=groups, classes=classes):
            return categorical.group_divergence(x, groups, classes)[0].sum()

        np.testing.assert_array_equal(np.asarray(jax.grad(total)(logits)), np.zeros((1, 2, 2), np.float32))
        _, tangent = jax.jvp(total, (logits,), (jnp.ones_like(logits),))
        assert float(tangent) == 0.0
