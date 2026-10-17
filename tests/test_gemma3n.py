"""Gemma 3n's AltUp, LAuReL block and sparse MLP against transformers 5.16.1.

The block fixtures come from tools/hf_reference_b.py: one `Gemma3nTextAltUp`
predicting and correcting a stream of four copies, one
`Gemma3nTextLaurelBlock` and one `Gemma3nTextMLP` at sparsity 0.95, each on
random weights. Everything runs at fp32 on CPU, and each parity test states
its tolerance and the largest difference observed.
"""

from pathlib import Path
from statistics import NormalDist

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import linen as nn
from reference_error import assert_leaves_as_exact

from dew.nn.gemma3n import AltUp, AltUpLayer, LaurelBlock, gaussian_topk
from dew.nn.mlp import GatedMLP

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "gemma3n"
HIDDEN, COPIES = 32, 4


def fixture() -> dict:
    with np.load(FIXTURES / "blocks.npz") as data:
        return {key: np.asarray(value) for key, value in data.items()}


class Steps(nn.Module):
    """Both AltUp steps in one call, so one init holds every leaf."""
    spec: AltUp

    def setup(self):
        self.altup = AltUpLayer(spec=self.spec, emb_features=HIDDEN, name="altup")

    def __call__(self, stream, activated, train: bool = False):
        predictions = self.altup.predict(stream, train=train)
        corrected = self.altup.correct(predictions, activated, train=train)
        return predictions, corrected, self.altup.scale_corrected_output(
            corrected[self.spec.active_idx])


def steps(spec: AltUp, variables: dict, stream, activated, train: bool = False) -> tuple:
    outputs = Steps(spec).apply(variables, stream, activated, train=train)
    return tuple(np.asarray(output) for output in outputs)


def altup_variables(tensors: dict) -> dict:
    return {"params": {"altup": {
        "correct_output_scale": jnp.asarray(tensors["altup.correct_output_scale"]),
        "correction_coefs": {"kernel": jnp.asarray(tensors["altup.correction_coefs.weight"].T)},
        "prediction_coefs": {"kernel": jnp.asarray(tensors["altup.prediction_coefs.weight"].T)},
        "modality_router": {"kernel": jnp.asarray(tensors["altup.modality_router.weight"].T)},
        "router_norm": {"scale": jnp.asarray(tensors["altup.router_norm.weight"])},
    }}}


def test_altup_predicts_and_corrects_like_the_reference():
    """`predict`: the copies mixed by a per-token matrix of the active
    copy's modalities, added back; `correct`: the innovation of the block's
    output over the active prediction, scaled per copy by one plus a
    coefficient, added to every prediction; then the output scale on the
    active copy. Tolerance 1e-5; observed 1.2e-07, 4.8e-07 and 1.5e-08."""
    tensors = fixture()
    predictions, corrected, scaled = steps(
        AltUp(num_inputs=COPIES), altup_variables(tensors),
        jnp.asarray(tensors["stream"]), jnp.asarray(tensors["activated"]))
    assert float(np.max(np.abs(predictions - tensors["predictions"]))) < 1e-5
    assert float(np.max(np.abs(corrected - tensors["corrected"]))) < 1e-5
    assert float(np.max(np.abs(scaled - tensors["scaled"]))) < 1e-5


def test_the_laurel_block_matches_the_reference():
    """`x + post_laurel_norm(linear_right(linear_left(x)))`. Tolerance
    1e-5; observed 4.8e-07."""
    tensors = fixture()
    variables = {"params": {
        "linear_left": {"kernel": jnp.asarray(tensors["laurel.linear_left.weight"].T)},
        "linear_right": {"kernel": jnp.asarray(tensors["laurel.linear_right.weight"].T)},
        "post_laurel_norm": {"scale": jnp.asarray(tensors["laurel.post_laurel_norm.weight"])}}}
    output = LaurelBlock(rank=8, emb_features=HIDDEN).apply(
        variables, jnp.asarray(tensors["activated"]))
    assert float(np.max(np.abs(np.asarray(output) - tensors["laurel_output"]))) < 1e-5


def test_the_sparse_mlp_matches_the_reference():
    """`Gemma3nTextMLP` at sparsity 0.95: the gate's gaussian top-k before
    its tanh-gelu. Tolerance 1e-5; observed 3.0e-07; the same weights with
    the gate left dense disagree."""
    tensors = fixture()
    variables = {"params": {
        name: {"kernel": jnp.asarray(tensors[f"mlp.{name}.weight"].T)}
        for name in ("gate_proj", "up_proj", "down_proj")}}
    hidden = jnp.asarray(tensors["activated"])
    sparse = GatedMLP(hidden_features=48, out_features=HIDDEN, activation="geglu",
                      activation_sparsity=0.95).apply(variables, hidden)
    assert float(np.max(np.abs(np.asarray(sparse) - tensors["mlp_output"]))) < 1e-5
    dense = GatedMLP(hidden_features=48, out_features=HIDDEN, activation="geglu").apply(
        variables, hidden)
    assert float(np.max(np.abs(np.asarray(dense) - tensors["mlp_output"]))) > 1e-2


def test_the_sparse_mlp_trains_on_padded_rows_as_the_reference_does():
    """A padded row of zeros has a constant gate row, whose deviation is
    zero. The reference's deviation takes a zero gradient there (torch's std
    backward masks it), and so must the port's: sqrt's infinite derivative
    times the masked loss's zero cotangent made every weight's gradient NaN.
    The output and the gradients of the input and the three kernels, the
    padding masked out of the loss, are as exact as the reference's, both
    measured from the reference in float64."""
    import torch
    from transformers import Gemma3nTextConfig
    from transformers.models.gemma3n.modeling_gemma3n import Gemma3nTextMLP

    tensors = fixture()
    names = ("gate_proj", "up_proj", "down_proj")
    hidden = np.array(tensors["activated"], np.float32)
    hidden[:, 4:] = 0
    real = np.arange(hidden.shape[1]) < 4
    cotangent = np.random.default_rng(0).standard_normal(hidden.shape).astype(np.float32) * real[:, None]
    config = Gemma3nTextConfig(
        vocab_size=64, hidden_size=HIDDEN, intermediate_size=48, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8, sliding_window=4,
        hidden_size_per_layer_input=8, altup_num_inputs=COPIES, laurel_rank=8,
        activation_sparsity_pattern=[0.95, 0.0], num_kv_shared_layers=0)

    def reference(dtype):
        mlp = Gemma3nTextMLP(config, layer_idx=0).to(dtype)
        with torch.no_grad():
            for name in names:
                getattr(mlp, name).weight.copy_(torch.from_numpy(tensors[f"mlp.{name}.weight"]))
        x = torch.tensor(hidden, dtype=dtype, requires_grad=True)
        output = mlp(x)
        (output * torch.tensor(cotangent, dtype=dtype)).sum().backward()
        return {"output": output.detach().numpy(), "input": x.grad.numpy(),
                **{name: getattr(mlp, name).weight.grad.numpy().T for name in names}}

    def loss(params, x):
        output = GatedMLP(hidden_features=48, out_features=HIDDEN, activation="geglu",
                          activation_sparsity=0.95).apply({"params": params}, x)
        return jnp.sum(output * cotangent), output

    params = {name: {"kernel": jnp.asarray(tensors[f"mlp.{name}.weight"].T)} for name in names}
    (_, output), (kernels, inputs) = jax.value_and_grad(loss, (0, 1), has_aux=True)(
        params, jnp.asarray(hidden))
    dew = {"output": output, "input": inputs, **{name: kernels[name]["kernel"] for name in names}}
    assert_leaves_as_exact(jax.tree.map(np.asarray, dew), reference(torch.float32), reference(torch.float64),
                           reductions=hidden.shape[0] * hidden.shape[1])


@pytest.mark.parametrize("sparsity", [0.5, 0.95])
def test_gaussian_topk_has_the_population_cutoff_jacobian(sparsity):
    # Both active and inactive coordinates, separated from the cutoff.
    x = np.array([-2, -1, 0, 1, 5], dtype=np.float64)
    multiplier = NormalDist().inv_cdf(sparsity)
    mean, std = x.mean(), x.std()
    margin = x - mean - multiplier * std
    assert np.min(np.abs(margin)) > 0.05
    cutoff_gradient = (1 + multiplier * (x - mean) / std) / x.size
    expected = (np.eye(x.size) - cutoff_gradient) * (margin > 0)[:, None]
    def apply(row):
        return gaussian_topk(row, sparsity)
    actual = jax.jit(jax.jacrev(apply))(jnp.asarray(x, jnp.float32))
    # Five-term population moments and the normal quantile at fp32.
    # A stopped cutoff gradient or sample deviation misses by >1e-2.
    np.testing.assert_allclose(actual, expected, atol=5e-7, rtol=0)
    # Forward cancellation at the cutoff is bounded in the input scale,
    # not by relative error in the small positive remainder.
    unit = np.finfo(np.float32).eps / 2
    bound = 8 * unit * (np.abs(x) + abs(mean) + abs(multiplier) * std)
    output = np.asarray(apply(jnp.asarray(x, jnp.float32)), dtype=np.float64)
    assert np.all(np.abs(output - np.maximum(margin, 0)) <= bound)
