"""The unchanged three failing gradient arms under the scratch one-token forward regrouping."""
import functools

import pytest
import test_recurrence_stability as stability
from test_forward_aligned_probe import prototype

from dew.nn.kernels import recurrence_chunk


@pytest.mark.parametrize("name", ["scalar delta", "vector delta", "gated"])
def test_aligned_forward_and_gradient(name, monkeypatch):
    monkeypatch.setattr(recurrence_chunk, "_chunk_delta_forward", functools.partial(prototype, "tail1"))
    stability.test_the_chunked_ops_gradient_is_as_exact_as_the_recurrences(name, "aligned keys")
