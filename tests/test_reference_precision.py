"""A reference truth keeps the softmax's requested float32 arithmetic wide."""

import numpy as np
import torch

from tools.torch_precision import float64


def test_float64_truth_widens_a_tensor_softmax():
    """DiffusionGemma's self-conditioning uses Tensor.softmax, whose explicit
    float32 request must be widened just as functional.softmax's is.
    """
    logits = torch.tensor([[0.17, -1.32, 0.91, 2.14]], dtype=torch.float64)
    expected = logits.softmax(dim=-1, dtype=torch.float64)
    with float64():
        actual = logits.softmax(dim=-1, dtype=torch.float32)
    assert actual.dtype == torch.float64
    np.testing.assert_array_equal(actual.numpy(), expected.numpy())
