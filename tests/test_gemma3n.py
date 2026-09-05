"""Gemma 3n's AltUp, LAuReL block and sparse MLP against transformers 5.16.1.

The block fixtures come from tools/hf_reference_b.py: one `Gemma3nTextAltUp`
predicting and correcting a stream of four copies, one
`Gemma3nTextLaurelBlock` and one `Gemma3nTextMLP` at sparsity 0.95, each on
random weights. Everything runs at fp32 on CPU, and each parity test states
its tolerance and the largest difference observed.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import linen as nn

from dew.nn.backbones.causal_transformer import GatedMLP
from dew.nn.gemma3n import AltUp, AltUpLayer, LaurelBlock, gaussian_topk

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
    assert isinstance(outputs, tuple) and len(outputs) == 3
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


@pytest.mark.parametrize("field,value", [("num_inputs", 1), ("active_idx", 4), ("coef_clip", 0.0)])
def test_an_altup_spec_out_of_range_is_refused(field, value):
    with pytest.raises(ValueError, match=f"altup_{field}"):
        AltUp(**{field: value})
