"""`_dew_torch.rounding.rounding_scale`: a torch program's rounding scale, read as
tests/rounding_bound.py reads a JAX program's."""

import numpy as np
import pytest

from _dew_torch.rounding import rounding_scale

torch = pytest.importorskip("torch")

UNIT = 2.0 ** -24


def test_a_scale_is_zero_where_no_rounding_reaches_and_set_where_one_does():
    """Values moved and selected (a transpose, a concatenation, a maximum) round nothing, their scale 0;
    a product rounds where its value is not 0, and a softmax, decomposed into the steps it rounds, rounds
    every element; a reduction's scale reaches every element of its result, a mean's too, and an embedding's
    gradient rounds the rows its places read more than once and no row none read, and a convolution's
    gradients round every element."""
    rng = np.random.default_rng(0)
    x = torch.from_numpy(rng.standard_normal((8, 16)))
    y = torch.from_numpy(np.where(rng.random((8, 16)) < 0.5, 0.0, rng.standard_normal((8, 16))))
    _, moved = rounding_scale(
        lambda: torch.maximum(torch.cat([x.T, y.T]), torch.zeros(32, 8, dtype=torch.float64)), UNIT)
    assert not moved.any()
    value, product = rounding_scale(lambda: x * y, UNIT)
    np.testing.assert_array_equal(product > 0, value != 0)
    for program in (lambda: torch.softmax(x, -1), lambda: x @ y.T, lambda: x.sum(-1), lambda: x.mean(-1)):
        assert (rounding_scale(program, UNIT)[1] > 0).all()
    # An embedding's gradient sums the cotangents of the places that read a row: a row read twice rounds.
    table = torch.nn.Embedding(4, 3, dtype=torch.float64)
    ids = torch.tensor([0, 0, 2])

    def gradient():
        return torch.autograd.grad((table(ids) * x[:3, :3]).sum(), table.weight)[0]

    value, scale = rounding_scale(gradient, UNIT)
    assert (scale[0] > 0).all() and not scale[[1, 3]].any()
    # A convolution's gradients, its input's, kernel's and bias's, each a sum, every element rounds.
    conv = torch.nn.Conv1d(4, 4, 3, groups=4, padding=2, dtype=torch.float64)
    signal = torch.from_numpy(rng.standard_normal((2, 4, 9))).requires_grad_()

    def conv_gradients():
        return torch.autograd.grad((conv(signal) * conv(signal)).sum(), (signal, conv.weight, conv.bias))

    for scale in rounding_scale(conv_gradients, UNIT)[1]:
        assert (scale > 0).all()


def test_the_same_draw_weighs_alike_and_another_seed_draws_again():
    """A scale is a deterministic weight of its program, inputs and seed: the same seed gives the same
    scale, another seed another draw of it."""
    x = torch.from_numpy(np.random.default_rng(1).standard_normal((4, 64)))
    def scale(seed):
        return rounding_scale(lambda: torch.softmax(x, -1) @ x.T, UNIT, seed=seed)[1]

    first, again, other = scale(0), scale(0), scale(1)
    np.testing.assert_array_equal(first, again)
    assert not np.array_equal(first, other)


def test_a_power_rounds_as_torchs_kernel_computes_it_whatever_its_exponent():
    """torch's power (aten.pow) is a library power at any exponent but a few its kernel computes as products
    (row 10157: an integer exponent was weighed as an integer power's chain, a rounding more each unit of
    it): at 4, whose powers here are powers of two (each relative scale computed alike to the bit), its scale
    relative to its value is one number at exponents 7 and 50, a library power's one rounding of at most 2u,
    and half of it at 2, the one product `x * x` its kernel computes."""
    four = torch.full((1,), 4.0, dtype=torch.float64)

    def relative(exponent):
        value, scale = rounding_scale(lambda: torch.pow(four, exponent), UNIT)
        return float(scale[0] / value[0])

    assert relative(7) == relative(50) == 2 * relative(2)


def test_an_operation_of_no_rule_is_refused():
    """An aten operation the scale has no rule for is refused, not passed as exact."""
    with pytest.raises(ValueError):
        rounding_scale(lambda: torch.special.digamma(torch.ones(3, dtype=torch.float64)),
                                       UNIT)


def test_an_operation_of_several_floating_results_and_no_rule_is_refused():
    """An operation returning several floating results that no rule weighs apart is refused, not passed
    as exact."""
    symmetric = torch.eye(3, dtype=torch.float64) * 2 + 1
    with pytest.raises(ValueError):
        rounding_scale(lambda: torch.linalg.eigh(symmetric), UNIT)


@pytest.mark.parametrize("form", ["default", "dim", "correction"])
def test_a_variance_is_the_one_its_overload_computes(form):
    """`aten.var` read by its overload (`_dew_torch.rounding`): each form's value at no rounding is torch's
    own (a population variance of the whole for `default`'s `unbiased=False`, never per column), and a
    correction past the count divides by ATen's nonnegative degrees of freedom (the confirm review: every
    overload read as `dim`'s)."""
    import torch

    # Consecutive integers: every partial mean and sum exact, whatever order the variance accumulates in,
    # and each denominator a power of two (16 states, 5 less one) or none (4 less a correction of 5).
    square = torch.arange(16, dtype=torch.float64).reshape(4, 4)
    rows = torch.arange(5, dtype=torch.float64) + torch.arange(4, dtype=torch.float64)[:, None]
    computed = {"default": lambda: torch.ops.aten.var.default(square, unbiased=False),
                "dim": lambda: torch.ops.aten.var.dim(rows, [1], unbiased=True, keepdim=True),
                "correction": lambda: torch.ops.aten.var.correction(square, [0], correction=5)}[form]
    value, _ = rounding_scale(computed, 0.0)
    np.testing.assert_array_equal(value, computed().numpy())


def test_a_complex_variance_is_refused():
    """A complex tensor's variance (each deviation's squared magnitude, which squaring the deviation is not:
    `[1j, -1j]`'s is 1, its squared deviations' mean -1) is refused rather than read as a real one's (the
    confirm review)."""
    import torch

    z = torch.tensor([1j, -1j], dtype=torch.complex128)
    with pytest.raises(ValueError):
        rounding_scale(lambda: torch.ops.aten.var.correction(z, correction=0), UNIT)


def test_a_nearest_backward_at_a_scale_apart_from_its_shapes_ratio_is_refused():
    """A nearest upsampling's backward whose stated scale is not its shapes' integral factor (3 to 9 rows at
    a scale of 3.2, which maps no block of three) is refused rather than summed as blocks; at its factor it
    is torch's own sum (the confirm review)."""
    import torch

    gradient = torch.ones(1, 1, 9, 4, dtype=torch.float64)

    def backward(scale):
        return lambda: torch.ops.aten.upsample_nearest2d_backward(gradient, [9, 4], [1, 1, 3, 2], scale, 2.0)

    with pytest.raises(ValueError):
        rounding_scale(backward(3.2), UNIT)
    value, _ = rounding_scale(backward(3.0), 0.0)
    np.testing.assert_array_equal(value, backward(3.0)().numpy())
