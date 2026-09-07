"""Run Google's actual SFTDiffusion on a tiny model and record loss/update parity.

Requires gemma at bf0b49901a428d13e9c2b2629f0eb9c153d3cbd3,
hackable-diffusion 1.0.1 and kauldron 1.4.4 in an isolated reference environment.
Run with JAX_PLATFORMS=cpu. No released weights are downloaded. The experiment
uses the unmodified official model, full-response circular cache overlays,
self-conditioning, target selection and both official loss implementations.
Only random-key transport and the uniform vocabulary sampler are paired: the
latter uses randint rather than weighted choice for equal probabilities.
Dew uses those same independent draws without allocating a categorical table.
"""

from __future__ import annotations

import json
from pathlib import Path

from flax import linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from safetensors.numpy import save_file

from gemma.diffusion import _models, _transformer
from gemma.diffusion.hackable_diffusion_adapter.hd import hd_gemma_network, sft_model
from gemma.gm.nn.gemma4 import _config, _modules
from hackable_diffusion.lib.corruption.discrete import CategoricalProcess
from hackable_diffusion.lib.corruption.schedules import LinearDiscreteSchedule
from hackable_diffusion.lib.jax_helpers import SafeSpan
from hackable_diffusion.lib.training.discrete_loss import NoWeightDiscreteLoss
from hackable_diffusion.lib.training.time_sampling import UniformTimeSampler

FIXTURE = Path(__file__).resolve().parents[1] / "tests/fixtures/hf/diffusion-gemma-sft"
SEED = 2


class UniformDraws(CategoricalProcess):
    def sample_from_invariant(self, key, data_spec):
        return jax.random.randint(key, data_spec.shape, 0, self.num_categories)


class PairedKeysSFT(sft_model.SFTDiffusion):
    def make_rng(self, name="params"):
        original = super().make_rng(name)
        if name != "sampling":
            return original
        if self.scope is None:
            raise ValueError("the reference RNG must have a bound scope")
        step_key = jax.random.fold_in(jax.random.split(jax.random.key(SEED))[1], 0)
        return jax.random.split(step_key, 4)[self.scope.rng_counters[name] - 1]


def reference_model(*, self_cond_prob=0.5, stop_encoder=False):
    config = _config.TransformerConfig(
        num_embed=32, embed_dim=8, num_heads=2, num_kv_heads=1, head_dim=4,
        hidden_dim=16, attention_types=[_modules.AttentionType.LOCAL_SLIDING, _modules.AttentionType.GLOBAL],
        sliding_window_size=4, kv_cache_sharing_config=None, use_post_attn_norm=True,
        use_post_ffw_norm=True, final_logit_softcap=30.0, global_rope_proportion=1.0, local_rope_proportion=1.0,
        k_eq_v_global=True, global_key_size=4, num_global_kv_heads=1,
        global_base_frequency=10000, local_base_frequency=10000)
    model = _models.DiffusionGemma_26B_A4B(
        config=config, dtype=jnp.float32,
        self_conditioning_config=_transformer.SelfConditioningConfig(features=8, hidden_dim=16))
    network = hd_gemma_network.WrappedDiffusionGemmaNetwork(gemma_model=model)
    return PairedKeysSFT(
        gemma_network=network,
        corruption_process=UniformDraws.uniform_process(num_categories=32, schedule=LinearDiscreteSchedule()),
        time_sampler=UniformTimeSampler(span=SafeSpan(safety_epsilon=1e-4)),
        prompt_len=4, canvas_size=4, num_canvases=2,
        x0="batch.canvas", prompt="batch.prompt", canvas_id="batch.canvas_id",
        canvas_mask="batch.canvas_mask", encoder_target="batch.encoder_target",
        encoder_target_mask="batch.encoder_target_mask", self_cond_prob=self_cond_prob,
        stop_gradient_from_denoiser_to_encoder=stop_encoder)


def to_hf(tree):
    model = tree["params"]["gemma_network"]["gemma_model"]
    result = {
        "model.decoder.embed_tokens.weight": np.asarray(model["embedder"]["input_embedding"]),
        "model.decoder.norm.weight": np.asarray(model["final_norm"]["scale"]),
    }
    for index in range(2):
        layer = model[f"layer_{index}"]
        stem = f"model.decoder.layers.{index}."
        attention = layer["attn"]
        query = np.asarray(attention["q_einsum"]["w"])
        result[stem + "self_attn.q_proj.weight"] = query.transpose(0, 2, 1).reshape(8, 8)
        output = np.asarray(attention["attn_vec_einsum"]["w"])
        result[stem + "self_attn.o_proj.weight"] = output.reshape(8, 8).T
        if "k_einsum" in attention:
            key = np.asarray(attention["k_einsum"]["w"])
            result[stem + "self_attn.k_proj.weight"] = key.transpose(0, 2, 1).reshape(4, 8)
        else:
            kv = np.asarray(attention["kv_einsum"]["w"])
            for part, name in enumerate(("k_proj", "v_proj")):
                result[stem + f"self_attn.{name}.weight"] = kv[part].transpose(0, 2, 1).reshape(4, 8)
        result[stem + "self_attn.q_norm.weight"] = np.asarray(attention["query_norm"]["scale"])
        result[stem + "self_attn.k_norm.weight"] = np.asarray(attention["key_norm"]["scale"])
        for source, target in (("pre_attention_norm", "input_layernorm"),
                               ("post_attention_norm", "post_attention_layernorm"),
                               ("pre_ffw_norm", "pre_feedforward_layernorm"),
                               ("post_ffw_norm", "post_feedforward_layernorm")):
            result[stem + target + ".weight"] = np.asarray(layer[source]["scale"])
        gate, up = np.asarray(layer["mlp"]["gating_einsum"])
        result[stem + "mlp.gate_proj.weight"] = gate
        result[stem + "mlp.up_proj.weight"] = up
        result[stem + "mlp.down_proj.weight"] = np.asarray(layer["mlp"]["linear"]).T
        result[stem + "layer_scalar"] = np.asarray(layer["skip_scale"])
    sc = model["self_conditioner"]
    stem = "model.decoder.self_conditioning."
    result[stem + "pre_norm.weight"] = np.asarray(sc["pre_norm"]["scale"])
    result[stem + "gate_proj.weight"], result[stem + "up_proj.weight"] = np.asarray(sc["ffw"]["gating_einsum"])
    result[stem + "down_proj.weight"] = np.asarray(sc["ffw"]["linear"]).T
    return {name: np.ascontiguousarray(value) for name, value in result.items()}


def main():
    prompt = jnp.array([[2, 5, 0, 0], [2, 6, 7, 0]], jnp.int32)
    response = jnp.array([[3, 4, 5, 6, 7, 8, 0, 0], [11, 12, 13, 0, 0, 0, 0, 0]], jnp.int32)
    canvas_mask = response != 0
    full = jnp.concatenate([prompt, response], axis=-1)
    valid = jnp.concatenate([prompt != 0, canvas_mask], axis=-1)
    targets = jnp.concatenate([full[:, 1:], jnp.zeros((2, 1), jnp.int32)], axis=-1)
    target_mask = valid & jnp.concatenate([valid[:, 1:], jnp.zeros((2, 1), bool)], axis=-1)
    kwargs = dict(x0=response[..., None], prompt=prompt, canvas_mask=canvas_mask,
                  canvas_id=jnp.broadcast_to(jnp.repeat(jnp.arange(2), 4), (2, 8)),
                  encoder_target=targets, encoder_target_mask=target_mask)
    model = reference_model()
    variables = model.init({"params": jax.random.key(701), "sampling": jax.random.key(SEED)},
                           kwargs, method=lambda module, data: module(**data))
    paths, structure = jax.tree.flatten_with_path(variables)
    scattered = []
    for index, (path, value) in enumerate(paths):
        noise = jax.random.normal(jax.random.fold_in(jax.random.key(731), index), value.shape)
        name = jax.tree_util.keystr(path)
        scattered.append(1 + 0.1 * noise if name.endswith(("['scale']", "['skip_scale']")) else 0.25 * noise)
    variables = structure.unflatten(scattered)
    loss_fn = NoWeightDiscreteLoss(use_mask=True, mask_key="target_mask")
    ar_loss = sft_model.EncoderARLoss()

    def objective(values, source_model=model):
        out = source_model.apply(values, kwargs, method=lambda module, data: module(**data),
                                 rngs={"sampling": jax.random.key(SEED)})
        if not isinstance(out, dict):
            raise TypeError("official SFTDiffusion must return its prediction dictionary")
        canvas = loss_fn(out["output"], out["target"], out["noise_info"]["time"])
        encoder = ar_loss.get_values(out["encoder_logits"], targets, target_mask)
        return canvas.mean() + encoder.mean(), (canvas, encoder, out)

    (loss, (canvas_loss, encoder_loss, out)), gradient = jax.value_and_grad(objective, has_aux=True)(variables)
    detached = reference_model(stop_encoder=True)
    (_, _), detached_grad = jax.value_and_grad(lambda v: objective(v, detached), has_aux=True)(variables)
    off = objective(variables, reference_model(self_cond_prob=0.0))[0]
    on = objective(variables, reference_model(self_cond_prob=1.0))[0]
    updated = jax.tree.map(lambda value, grad: value - 0.001 * grad, variables, gradient)
    updated_loss = objective(updated)[0]
    FIXTURE.mkdir(parents=True, exist_ok=True)
    save_file(to_hf(variables), FIXTURE / "model.safetensors")
    references = FIXTURE / "reference"
    references.mkdir(exist_ok=True)
    save_file(to_hf(gradient), references / "gradient.safetensors")
    save_file(to_hf(detached_grad), references / "detached_gradient.safetensors")
    save_file(to_hf(updated), references / "updated.safetensors")
    config = {"model_type": "diffusion_gemma", "canvas_length": 4, "text_config": {
        "model_type": "diffusion_gemma_text", "vocab_size": 32, "hidden_size": 8,
        "intermediate_size": 16, "num_hidden_layers": 2, "num_attention_heads": 2,
        "num_key_value_heads": 1, "head_dim": 4, "global_head_dim": 4,
        "num_global_key_value_heads": 1, "hidden_activation": "gelu_pytorch_tanh",
        "layer_types": ["sliding_attention", "full_attention"], "sliding_window": 4,
        "rope_parameters": {"sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
                            "full_attention": {"rope_type": "proportional", "partial_rotary_factor": 1.0,
                                               "rope_theta": 10000.0}},
        "max_position_embeddings": 32, "rms_norm_eps": 1e-6, "tie_word_embeddings": True}}
    (FIXTURE / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    np.savez(FIXTURE / "reference.npz", tokens=full, canvas_mask=canvas_mask,
             encoder_target_mask=target_mask, loss=loss, canvas_loss=canvas_loss,
             encoder_loss=encoder_loss, updated_loss=updated_loss, sc_off_loss=off, sc_on_loss=on,
             noisy=out["xt"], time=out["noise_info"]["time"], selected_mask=out["target"]["target_mask"],
             encoder_logits=out["encoder_logits"], decoder_logits=out["output"]["logits"],
             step_key=jax.random.key_data(jax.random.fold_in(jax.random.split(jax.random.key(SEED))[1], 0)),
             run_seed=np.asarray(SEED))
    print(json.dumps({"loss": float(loss), "canvas": canvas_loss.tolist(), "encoder": encoder_loss.tolist(),
                      "updated_loss": float(updated_loss), "selected_mask": out["target"]["target_mask"].tolist()}))


if __name__ == "__main__":
    main()
