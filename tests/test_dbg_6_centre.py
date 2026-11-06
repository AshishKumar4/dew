"""Diagnostic witness for the backward evaluator's product boundary."""

import pytest

from dew.algebra import forms
from dew.nn.kernels.templates import attention_portable
from test_attention_dropout import (
    test_identical_keys_give_an_exactly_zero_query_gradient_on_the_portable_mainloop as identical,
)


@pytest.mark.parametrize("issue_inner_product", [False, True])
def test_identical_keys_through_product_boundary(monkeypatch, issue_inner_product):
    original = forms.float32_dot_general

    def inner(left, right, dimensions, **keywords):
        if (issue_inner_product and left.ndim == right.ndim == 2
                and dimensions == (((1,), (1,)), ((), ()))):
            return attention_portable.interpreted_dot(left, right.T)
        return original(left, right, dimensions, **keywords)

    monkeypatch.setattr(forms, "float32_dot_general", inner)
    identical()
