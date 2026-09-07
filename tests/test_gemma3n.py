"""Gemma 3n's AltUp, LAuReL block and sparse MLP against transformers 5.16.1.

The block fixtures come from tools/hf_reference_b.py: one `Gemma3nTextAltUp`
predicting and correcting a stream of four copies, one
`Gemma3nTextLaurelBlock` and one `Gemma3nTextMLP` at sparsity 0.95, each on
random weights. Everything runs at fp32 on CPU, and each parity test states
its tolerance and the largest difference observed.
"""

import functools
from pathlib import Path
from statistics import NormalDist

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import linen as nn

from dew.nn.backbones.causal_transformer import BlockWiring, DecoderBlock, GatedMLP
from dew.nn.gemma3n import AltUp, AltUpLayer, LaurelBlock, gaussian_topk
from dew.nn.mixers import AttentionMixer, MixerContext

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


def test_the_prediction_matrix_is_transposed_as_the_reference_permutes_it():
    """The reference reshapes the coefficients to [n, n] and transposes
    before the matmul; mixing along the other axis disagrees on a stream
    whose copies differ."""
    tensors = fixture()
    kernel = tensors["altup.prediction_coefs.weight"].T.reshape(COPIES, COPIES, COPIES)
    swapped = {"params": {"altup": {
        **altup_variables(tensors)["params"]["altup"],
        "prediction_coefs": {"kernel": jnp.asarray(
            np.swapaxes(kernel, 1, 2).reshape(COPIES, COPIES * COPIES))}}}}
    predictions, _, _ = steps(AltUp(num_inputs=COPIES), swapped,
                              jnp.asarray(tensors["stream"]), jnp.asarray(tensors["activated"]))
    assert float(np.max(np.abs(predictions - tensors["predictions"]))) > 1e-2


def test_the_coefficient_clip_binds_in_the_training_pass_alone():
    """With a clip below the fixture's weights, the training pass predicts
    differently from the eval pass, which reads the weights as stored."""
    tensors = fixture()
    largest = float(np.max(np.abs(tensors["altup.prediction_coefs.weight"])))
    spec = AltUp(num_inputs=COPIES, coef_clip=largest / 4)
    variables = altup_variables(tensors)
    stream, activated = jnp.asarray(tensors["stream"]), jnp.asarray(tensors["activated"])
    evaluated, _, _ = steps(spec, variables, stream, activated)
    trained, _, _ = steps(spec, variables, stream, activated, train=True)
    assert float(np.max(np.abs(evaluated - tensors["predictions"]))) < 1e-5
    assert float(np.max(np.abs(trained - tensors["predictions"]))) > 1e-3


def test_zero_altup_scale_still_learns_the_per_layer_residual():
    features, per_layer_features = 8, 3
    mixer = AttentionMixer().build(MixerContext(
        emb_features=features, num_heads=2, num_kv_heads=2, head_dim=4, max_seq_len=1))
    block = DecoderBlock(
        mixer=mixer, feedforward=functools.partial(
            GatedMLP, hidden_features=12, out_features=features,
            activation="geglu", activation_sparsity=0.95),
        emb_features=features, wiring=BlockWiring(output_norms=True),
        norm_eps=1e-6, per_layer_input_dim=per_layer_features,
        altup=AltUp(num_inputs=COPIES))
    stream = jnp.linspace(-1, 1, COPIES * features).reshape(COPIES, 1, 1, features)
    per_layer = jnp.linspace(0.2, 1, per_layer_features).reshape(1, 1, per_layer_features)
    variables = block.init(jax.random.key(0), stream, per_layer_input=per_layer)
    params = variables["params"]
    params["post_per_layer_input_norm"]["scale"] = jnp.linspace(0.7, 1.3, features)

    def forward(scale):
        current = {**params, "altup": {**params["altup"], "correct_output_scale": scale}}
        return block.apply({"params": current}, stream, per_layer_input=per_layer)

    scale = params["altup"]["correct_output_scale"]
    corrected = block.apply(variables, stream, per_layer_input=None)
    np.testing.assert_array_equal(forward(scale), corrected)
    derivative = np.asarray(jax.jit(jax.jacrev(forward))(scale), dtype=np.float64)

    # At zero scale, GELU'(0)=1/2 and RMSNorm'(0)=diag(weight)/sqrt(eps).
    # Differentiate only the PLE term; the unscaled corrected copies do not
    # depend on this parameter. Float64 products give an independent chain
    # rule oracle; normalize out the known 1000x gain before comparing fp32.
    # CPU error is 1.31e-7; stopping the scale gradient misses by 0.457.
    active = np.asarray(corrected[0, 0, 0], dtype=np.float64)
    gate = np.asarray(params["per_layer_input_gate"]["kernel"], dtype=np.float64)
    projection = np.asarray(params["per_layer_projection"]["kernel"], dtype=np.float64)
    weight = np.asarray(params["post_per_layer_input_norm"]["scale"], dtype=np.float64)
    ple = np.asarray(per_layer[0, 0], dtype=np.float64)
    scaled_jacobian = (0.5 * active[:, None] * gate * ple) @ projection * weight
    expected = np.stack([np.zeros_like(scaled_jacobian)]
                        + [scaled_jacobian.T] * (COPIES - 1))
    np.testing.assert_allclose(derivative[:, 0, 0] * np.sqrt(block.norm_eps),
                               expected, atol=5e-7, rtol=0)


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


def test_gaussian_topk_keeps_about_the_stated_fraction():
    """On Gaussian rows a sparsity of 0.95 keeps about 5% of the entries,
    each as its distance above the cutoff, and 0.5 about half."""
    rows = jax.random.normal(jax.random.PRNGKey(0), (64, 4096))
    for sparsity in (0.95, 0.5):
        kept = gaussian_topk(rows, sparsity) > 0
        assert abs(float(kept.mean()) - (1 - sparsity)) < 0.01
    assert float(gaussian_topk(rows, 0.95).min()) == 0.0


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


def test_gaussian_topk_uses_zero_relu_derivative_at_the_cutoff():
    # At sparsity 1/2 the quantile is exactly zero, so the middle entry
    # equals the cutoff exactly; perturbations select opposite branches.
    def apply(row):
        return gaussian_topk(row, 0.5)
    x = jnp.array([-1., 0., 1.])
    derivative = jax.jit(jax.jacrev(apply))(x)
    np.testing.assert_allclose(derivative, [[0, 0, 0], [0, 0, 0],
                                          [-1/3, -1/3, 2/3]], atol=5e-8, rtol=0)
    delta = 2 ** -12
    assert float(apply(x.at[1].set(-delta))[1]) == 0
    assert float(apply(x.at[1].set(delta))[1]) == pytest.approx(2 * delta / 3)


@pytest.mark.parametrize("field,value", [("num_inputs", 1), ("active_idx", 4), ("coef_clip", 0.0)])
def test_an_altup_spec_out_of_range_is_refused(field, value):
    with pytest.raises(ValueError, match=f"altup_{field}"):
        AltUp(**{field: value})
