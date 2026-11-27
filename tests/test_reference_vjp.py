"""The autoencoder recorders' torch walk (tools/reference_vjp.py `walked`), check 1's own torch arm behind
the recordings' names and layout (row 13711)."""

import numpy as np
import pytest
import tool_support

torch = pytest.importorskip("torch")


def test_a_recorders_walk_is_the_models_own_gradients_under_the_recordings_names():
    """`walked` over a convolution keeps the names and channels-first layout its recordings hold
    (`output_<i>`, `grad_operand`, `grad_param.<name>` for each parameter named), each array the model's
    own autograd gives of the probed sum, bit for bit: a rename or a moved axis in the arm it runs
    (`interop_support.torch_coder_walk`) fails here, not as a silent mismatch against the store."""
    reference_vjp = tool_support.load("reference_vjp")
    torch.manual_seed(13711)
    model = torch.nn.Conv2d(3, 4, 3, padding=1)
    rng = np.random.default_rng(13711)
    inputs = {"operand": rng.standard_normal((2, 3, 8, 8)).astype(np.float32),
              "probe_0": rng.standard_normal((2, 4, 8, 8)).astype(np.float32)}
    walk = reference_vjp.walked(model, lambda held, x: (held(x),), inputs, torch.float32, ["weight"])
    assert sorted(walk) == ["grad_operand", "grad_param.weight", "output_0"]
    operand = torch.from_numpy(inputs["operand"]).requires_grad_()
    output = model(operand)
    grad_operand, grad_weight = torch.autograd.grad((output * torch.from_numpy(inputs["probe_0"])).sum(),
                                                    [operand, model.weight])
    for name, expected in (("output_0", output), ("grad_operand", grad_operand),
                           ("grad_param.weight", grad_weight)):
        expected = expected.detach().numpy()
        assert walk[name].shape == expected.shape, name
        np.testing.assert_array_equal(walk[name], expected, err_msg=name)
