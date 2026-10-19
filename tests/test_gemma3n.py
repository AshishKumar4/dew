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

from dew.nn.gemma3n import AltUp, AltUpLayer, LaurelBlock
from dew.nn.mlp import GatedMLP, gaussian_topk

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


def sparse_mlps(sparsity: float, width: int, rows: int, dtype):
    """The reference's public `Gemma3nTextMLP` and Dew's `GatedMLP` at `sparsity` over the
    same random weights, `width` gate activations a row, and `rows` random inputs."""
    import torch
    from transformers import Gemma3nTextConfig
    from transformers.models.gemma3n.modeling_gemma3n import Gemma3nTextMLP

    draws = np.random.default_rng(0)
    names = ("gate_proj", "up_proj", "down_proj")
    shapes = {"gate_proj": (width, HIDDEN), "up_proj": (width, HIDDEN), "down_proj": (HIDDEN, width)}
    weights = {name: (draws.standard_normal(shape) / np.sqrt(shape[1])).astype(np.float32)
               for name, shape in shapes.items()}
    hidden = draws.standard_normal((rows, HIDDEN)).astype(np.float32)
    config = Gemma3nTextConfig(hidden_size=HIDDEN, intermediate_size=width, num_hidden_layers=1,
                               activation_sparsity_pattern=[sparsity])

    def reference(torch_dtype, cotangent=None):
        mlp = Gemma3nTextMLP(config, layer_idx=0).to(torch_dtype)
        with torch.no_grad():
            for name in names:
                getattr(mlp, name).weight.copy_(torch.from_numpy(weights[name]))
        x = torch.tensor(hidden).to(torch_dtype).requires_grad_()
        output = mlp(x)
        if cotangent is None:
            return {"output": output.detach().float().numpy()}
        (output * torch.tensor(cotangent, dtype=torch_dtype)).sum().backward()
        return {"output": output.detach().numpy(), "input": x.grad.numpy(),
                **{name: getattr(mlp, name).weight.grad.numpy().T for name in names}}

    model = GatedMLP(hidden_features=width, out_features=HIDDEN, activation="geglu",
                     activation_sparsity=sparsity, dtype=dtype)
    params = {name: {"kernel": jnp.asarray(weights[name].T, dtype)} for name in names}
    return reference, model, params, jnp.asarray(hidden, dtype)


@pytest.mark.parametrize("sparsity", [0.9, 0.95, 0.99])
def test_the_sparse_mlp_is_as_exact_as_the_reference_at_each_sparsity(sparsity):
    """Gemma 3n's sparse MLP through the reference's public forward and backward, 256 rows of
    2048 gate activations, at each sparsity a released pattern uses or bounds: Dew's fp32
    output and gradients are within tests/reference_error.py's rule of the reference's fp32,
    both from the reference in float64. The gate's cutoff multiplier is the reference's (its
    fp32 icdf); jax's own fp32 erfinv lands two ulps from it at 0.95."""
    import torch

    reference, model, params, hidden = sparse_mlps(sparsity, 2048, 256, jnp.float32)
    cotangent = np.random.default_rng(1).standard_normal((256, HIDDEN)).astype(np.float32)

    def loss(params, x):
        output = model.apply({"params": params}, x)
        return jnp.sum(output * cotangent), output

    (_, output), (kernels, inputs) = jax.jit(jax.value_and_grad(loss, (0, 1), has_aux=True))(params, hidden)
    dew = {"output": output, "input": inputs, **{name: kernel["kernel"] for name, kernel in kernels.items()}}
    ours, theirs, truth = (jax.tree.map(np.asarray, dew), reference(torch.float32, cotangent),
                           reference(torch.float64, cotangent))
    # The output and the input's gradient sum over the 2048 activations; a kernel's gradient over the rows.
    for names, reductions in ((("output", "input"), 2048), (("gate_proj", "up_proj", "down_proj"), 256)):
        assert_leaves_as_exact(*({name: tree[name] for name in names} for tree in (ours, theirs, truth)),
                               reductions=reductions)


@pytest.mark.parametrize("sparsity", [1e-9, 1 - 1e-9])
def test_the_sparse_mlp_takes_the_references_cutoff_at_the_ends_of_its_domain(sparsity):
    """A sparsity whose fp32 argument 2p - 1 rounds to -1 or 1 takes the reference's infinite
    cutoff multiplier (torch's icdf there): above, the gate keeps nothing and the MLP outputs
    zeros; below, it keeps every activation as infinity, which the down projection sums to
    NaN. Dew's MLP outputs the reference's, bit for bit, in fp32 and bf16, which both take the
    fp32 argument. A float64 model forms its own float64 argument, which these do not end."""
    import torch

    for torch_dtype, dtype in ((torch.float32, jnp.float32), (torch.bfloat16, jnp.bfloat16)):
        reference, model, params, hidden = sparse_mlps(sparsity, 64, 4, dtype)
        dew = jax.jit(lambda params, x, model=model: model.apply({"params": params}, x))(params, hidden)
        np.testing.assert_array_equal(np.asarray(dew, np.float32), reference(torch_dtype)["output"])


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
