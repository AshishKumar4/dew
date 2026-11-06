"""Diagnostic witness for the backward evaluator's product boundary."""

import pytest

from dew.algebra import forms
from dew.nn.kernels.templates import attention_backward_algebra, attention_portable
from test_attention_dropout import (
    test_identical_keys_give_an_exactly_zero_query_gradient_on_the_portable_mainloop as identical,
)


@pytest.mark.parametrize("issue_inner_product", [False, True, "declaration"])
def test_identical_keys_through_product_boundary(monkeypatch, issue_inner_product):
    original = forms.float32_dot_general
    einsum = attention_backward_algebra.float32_einsum

    def declared(subscripts, *operands, **keywords):
        if subscripts == "qd,kd->qk":
            return einsum("qd,dk->qk", operands[0], operands[1].T, **keywords)
        return einsum(subscripts, *operands, **keywords)

    def inner(left, right, dimensions, **keywords):
        if (issue_inner_product is True and left.ndim == right.ndim == 2
                and dimensions == (((1,), (1,)), ((), ()))):
            return attention_portable.interpreted_dot(left, right.T)
        return original(left, right, dimensions, **keywords)

    monkeypatch.setattr(forms, "float32_dot_general", inner)
    if issue_inner_product == "declaration":
        monkeypatch.setattr(attention_backward_algebra, "float32_einsum", declared)
        attention_backward_algebra.derived_backward.cache_clear()
    try:
        identical()
    finally:
        attention_backward_algebra.derived_backward.cache_clear()
