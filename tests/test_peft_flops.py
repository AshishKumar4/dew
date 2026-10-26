"""Frozen-base and trainable-factor work, including the actual gradient matrix shapes."""


import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import linen as nn
from test_model_flops import decoder, forward_products

from dew.batch import FROZEN
from dew.diffusion import presets
from dew.inputs import Field, InputSpec
from dew.lora import LoRA
from dew.nn.attention_residuals import AttentionResiduals
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.backbones.dit import SimpleDiT
from dew.nn.backbones.layer_plan import LayerKind
from dew.nn.flops import MissingFlopsCount, dense_training_flops, projection_flops, ssd_flops
from dew.nn.gemma3n import AltUp
from dew.nn.hyper_connections import HyperConnections
from dew.nn.mixers.mamba import Mamba
from dew.nn.mixers.mamba2 import Mamba2, Mamba2Mixer
from dew.nn.mixers.mlp import MLPMixer
from dew.objectives.diffusion import DiffusionObjective
from dew.objectives.lm import TEXT_KEY, LMObjective


class DenseCount(nn.Module):
    inputs: int
    outputs: int

    @nn.compact
    def __call__(self, x):
        return nn.Dense(self.outputs, use_bias=False, name="layer")(x)

    @nn.nowrap
    def model_flops(self, lengths):
        return projection_flops(lengths, self.inputs, self.outputs)

    @nn.nowrap
    def parameter_flops(self, lengths):
        return self.model_flops(lengths)

    @nn.nowrap
    def matrix_usage(self, path, lengths, *, logits=False):
        return np.zeros(np.shape(lengths), np.int64) if logits else np.asarray(lengths)


def test_frozen_dense_rank_factors_equal_the_actual_forward_and_gradient_dots():
    model = DenseCount(8, 16)
    x = jnp.ones((2, 3, 8))
    variables = model.init(jax.random.key(401), x)
    adapted = LoRA(rank=2, alpha=2, modules=("layer",)).apply(model, variables, key=501)
    lengths = np.array([3, 3])
    expected = dense_training_flops(lengths, 8, 16, trainable=False, adapter_rank=2)
    assert adapted.model.training_flops(lengths) == expected

    def forward_backward(params, frozen, x):
        def forward(params, x):
            return adapted.model.apply({"params": params, FROZEN: frozen}, x)
        y, pullback = jax.vjp(forward, params, x)
        return y, *pullback(jnp.ones_like(y))

    assert expected == forward_products(forward_backward, adapted.variables["params"],
                                        adapted.variables[FROZEN], x)


def test_adapted_prediction_depths_count_their_maps_and_the_reused_head():
    model = decoder(num_nextn_predict_layers=2, attention_impl="reference", tie_embeddings=False)
    tokens = jnp.ones((1, 6), jnp.int32)
    variables = model.init(jax.random.key(409), tokens)
    adapted = LoRA(rank=2, alpha=4, modules=("q_proj", "eh_proj", "lm_head")).apply(
        model, variables, key=509)
    lengths = np.array([6], np.int64)
    # The input projection exists once per depth and the head is shared across both.
    predicted = 5 + 4
    factors = 2 * 2 * predicted * ((8 + 8) + (16 + 8) + (8 + model.vocab_size))
    native = model.prediction_flops(lengths)
    learned = model.prediction_parameter_flops(lengths)
    assert adapted.model.prediction_flops(lengths) == native + factors
    assert adapted.model.prediction_training_flops(lengths) == 3 * native - learned + 3 * factors

    def forward(variables, tokens):
        hidden = adapted.model.apply(variables, tokens, method="hidden_states")
        logits = adapted.model.apply(variables, hidden, method="logits_from_hidden")
        return logits, adapted.model.apply(variables, hidden, tokens, method="mtp_logits")

    objective = LMObjective(adapted.model, variables=adapted.variables, seq_len=6, mtp_weight=1.0)
    batch = {TEXT_KEY: np.ones((1, 7), np.int32)}
    upper = 2 * model.num_heads * model.head_dim * (model.num_layers * 6**2 + 5**2 + 4**2)
    assert objective.model_flops(None, batch) == forward_products(forward, adapted.variables, tokens) - upper
    assert objective.training_flops(None, batch) == (adapted.model.training_flops(lengths)
        + adapted.model.logits_training_flops(lengths) + adapted.model.prediction_training_flops(lengths))


def test_adapted_image_geometry_refuses_inherited_native_work_without_matrix_usage():
    model = SimpleDiT(emb_features=8, num_layers=1, num_heads=2, patch_size=2)
    image, time = jnp.ones((1, 4, 4, 3)), jnp.ones((1,))
    variables = model.init(jax.random.key(410), image, time)
    adapted = LoRA(rank=2, alpha=4, modules=("final_proj",)).apply(model, variables, key=510)
    objective = DiffusionObjective(adapted.model, presets.Flow(), InputSpec(Field("images", (4, 4, 3))),
                                   variables=adapted.variables, autoencoder=None, ema_decay=None)
    for count in (objective.model_flops, objective.training_flops):
        with pytest.raises(MissingFlopsCount):
            count(None, {"images": np.asarray(image)})


@pytest.mark.parametrize("fields", [{"altup": AltUp(num_inputs=2)},
                                    {"hyper_connections": HyperConnections(hc_mult=2, head="weighted")},
                                    {"attention_residuals": AttentionResiduals(block_size=2)}])
def test_residual_streams_separate_learned_work_from_mixing_for_peft(fields):
    model = decoder(causal=False, attention_impl="reference", **fields)
    tokens = jnp.ones((1, 4), jnp.int32)
    variables = model.init(jax.random.key(411), tokens)
    adapted = LoRA(rank=2, alpha=4, modules=("q_proj",)).apply(model, variables, key=511)
    lengths = np.array([4])
    factors = model.num_layers * 2 * 2 * (8 + 8) * 4
    native, learned = model.model_flops(lengths), model.parameter_flops(lengths)
    assert 0 < learned < native
    assert adapted.model.parameter_flops(lengths) == learned + factors
    assert adapted.model.training_flops(lengths) == 3 * native - learned + 3 * factors
    complete = adapted.model.model_flops(lengths) + adapted.model.logits_flops(lengths)
    assert complete == forward_products(adapted.model.apply, adapted.variables, tokens)


def test_mamba2_layers_learn_their_maps_and_conv_and_not_their_scan():
    """A Mamba-2 mixer's learned work is its two maps and its depthwise conv:
    in_proj's [z, x, B, C, dt], out_proj back to the residual and the conv's
    taps over x, B and C; the scan's state contractions are the rest. An
    adapted Nemotron-H-shaped hybrid, Mamba-2 and MLP mixers with no second
    feed-forward, charges its frozen work twice, its scan three times, and
    its rank-2 factors three times, as many as its in_proj kernels' widths
    make them."""
    lengths = np.array([4])
    mixer = Mamba2(8, 2, 4, state_size=3, chunk_size=1)
    width, groups_state = 2 * 4, 3
    per_token = (2 * 8 * (2 * width + 2 * groups_state + 2) + 2 * width * 8
                 + 2 * 4 * (width + 2 * groups_state))
    assert mixer.parameter_flops(lengths) == 4 * per_token
    assert mixer.model_flops(lengths) == 4 * per_token + ssd_flops(lengths, 2, 3, 4)
    # Mamba's in_proj [x, z], x_proj [dt, B, C], dt_proj, out_proj and conv over x.
    first = Mamba(8, 16, state_size=3, chunk_size=1)
    rank, inner = first.time_step_rank, 16
    maps = 2 * 8 * 2 * inner + 2 * inner * (rank + 2 * 3) + 2 * rank * inner + 2 * inner * 8 + 2 * 4 * inner
    assert first.parameter_flops(lengths) == 4 * maps
    assert first.model_flops(lengths) == 4 * maps + ssd_flops(lengths, 1, 3, inner)
    model = CausalTransformer(
        vocab_size=64, emb_features=8, num_layers=3, num_heads=2, num_kv_heads=1, head_dim=4, mlp_features=0,
        layer_types=("linear_attention", "mlp", "linear_attention"),
        kinds={"linear_attention": LayerKind(mixer=Mamba2Mixer(num_heads=2, head_dim=4, state_size=3,
                                                                n_groups=1, norm_groups=1, chunk_size=1)),
               "mlp": LayerKind(mixer=MLPMixer(intermediate_size=16))})
    tokens = jnp.ones((1, 4), jnp.int32)
    variables = model.init(jax.random.key(413), tokens)
    adapted = LoRA(rank=2, alpha=4, modules=("in_proj",)).apply(model, variables, key=513)
    kernels = [leaf.shape for path, leaf in jax.tree_util.tree_leaves_with_path(variables["params"])
               if "in_proj" in jax.tree_util.keystr(path) and leaf.ndim == 2]
    assert len(kernels) == 2
    factors = sum(2 * 2 * (inputs + outputs) * 4 for inputs, outputs in kernels)
    native, learned = model.model_flops(lengths), model.parameter_flops(lengths)
    assert 0 < learned < native
    assert adapted.model.training_flops(lengths) == 3 * native - learned + 3 * factors


def test_altup_adapter_router_is_used_for_both_prediction_and_correction():
    model = decoder(altup=AltUp(num_inputs=2), attention_impl="reference", causal=False)
    tokens = jnp.ones((1, 4), jnp.int32)
    variables = model.init(jax.random.key(412), tokens)
    adapted = LoRA(rank=2, alpha=4, modules=("modality_router",)).apply(model, variables, key=512)
    extra = model.num_layers * 2 * 2 * 2 * (8 + 2) * 4
    assert adapted.model.model_flops(np.array([4])) == model.model_flops(np.array([4])) + extra
    complete = adapted.model.model_flops(np.array([4])) + adapted.model.logits_flops(np.array([4]))
    assert complete == forward_products(adapted.model.apply, adapted.variables, tokens)
