"""Google's published DiffusionGemma fine-tuning loss, not the original SD·RL stage.

Reference: gemma bf0b49901a428d13e9c2b2629f0eb9c153d3cbd3,
``diffusion/hackable_diffusion_adapter/hd/sft_model.py`` and the Sudoku config.
The two losses normalize tokens within each row and then average rows. Their
sufficient statistics remain separate so accumulation does not token-weight
the composite loss.
"""

from __future__ import annotations

import math

from flax import struct
import jax
import jax.numpy as jnp
import optax

from dew.inputs import Field, InputSpec
from dew.nn.diffusion_gemma import DiffusionGemma
from dew.objectives.base import Aux, Batch, EMASpec, Mean, Objective, Step, Variables, mean_loss
from dew.registry import objectives


@struct.dataclass
class BlockSFTStatistics:
    canvas: Mean
    encoder: Mean
    support: jax.Array


def _positions(valid: jax.Array) -> jax.Array:
    counts = jnp.cumsum(valid, axis=-1)
    return counts - (counts >= 1)


def _cache_geometry(valid: jax.Array, selected: jax.Array, prompt_length: int, canvas_size: int):
    """Represent the reference's circular overlay without mutating encoder K/V.

    Google evaluates the whole response, not only the selected canvas. Its
    noisy keys overwrite physical slots modulo the full prompt+response cache
    length. Dew's cache packs valid encoder keys; mapping physical slots to
    their packed indices preserves that source behavior, including padding.
    Concatenated old/new keys with an exclusion mask are the same attention
    operation as the reference's overwritten buffer, up to reduction order.
    """
    batch, total = valid.shape
    response = total - prompt_length
    physical = jnp.arange(total)
    ordered = jnp.argsort(~valid, axis=-1, stable=True)
    packed_valid = jnp.take_along_axis(valid, ordered, axis=-1)
    positions = _positions(valid)
    packed_positions = jnp.take_along_axis(positions, ordered, axis=-1)
    encoder_mask = (ordered[:, None, :] <= physical[None, :, None]) & packed_valid[:, None, :]
    write_slots = (prompt_length + selected[:, None] * canvas_size
                   + jnp.arange(response)[None, :]) % total
    overwritten = jnp.any(physical[None, :, None] == write_slots[:, None, :], axis=-1)
    allowed = valid & ((physical[None, :] < prompt_length)
                      | ((physical[None, :] - prompt_length) // canvas_size <= selected[:, None]))
    old_keys = jnp.take_along_axis(allowed & ~overwritten, ordered, axis=-1) & packed_valid
    new_keys = jnp.take_along_axis(allowed, write_slots, axis=-1)
    decoder_mask = jnp.broadcast_to(jnp.concatenate([old_keys, new_keys], axis=-1)[:, None, :],
                                    (batch, response, total + response))
    decoder_positions = positions[:, prompt_length:]
    key_positions = jnp.concatenate([packed_positions, decoder_positions], axis=-1)
    return positions, encoder_mask, packed_positions, decoder_mask, key_positions


def _row_mean(losses: jax.Array, mask: jax.Array) -> Mean:
    mass = mask.sum(axis=-1)
    row_losses = jnp.sum(jnp.where(mask != 0, losses, 0) * mask, axis=-1) / jnp.maximum(mass, 1)
    return Mean(row_losses.sum(), jnp.asarray(losses.shape[0], jnp.int32))


@objectives("block_diffusion")
class BlockDiffusionObjective(Objective[BlockSFTStatistics]):
    """Official post-release SFT over clean ``text`` rows split into prompt and canvases.

    A row has ``prompt_length + canvas_size * num_canvases`` tokens. Optional
    ``canvas_mask`` and ``encoder_target_mask`` select valid targets; absent
    masks are derived from the pad ID and adjacent valid encoder positions,
    matching Google's SequenceTargetShift. Every response token is corrupted,
    but only a uniformly selected valid canvas contributes diffusion CE.
    """

    def __init__(self, model: DiffusionGemma, *, prompt_length: int,
                 num_canvases: int = 1, canvas_size: int | None = None,
                 pretrained: Variables | None = None, pad_token_id: int = 0,
                 self_cond_prob: float = 0.5, safety_epsilon: float = 1e-4,
                 stop_gradient_from_denoiser_to_encoder: bool = False,
                 encoder_loss_weight: float = 1.0, decoder_loss_weight: float = 1.0,
                 ema_decay: float | None = None):
        canvas_size = model.canvas_length if canvas_size is None else canvas_size
        for name, value in (("prompt_length", prompt_length), ("num_canvases", num_canvases),
                            ("canvas_size", canvas_size)):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not 0 <= self_cond_prob <= 1 or not math.isfinite(self_cond_prob):
            raise ValueError("self_cond_prob must be in [0, 1]")
        if not 0 <= safety_epsilon < 0.5:
            raise ValueError("safety_epsilon must define a nonempty SafeSpan")
        if any(not math.isfinite(weight) or weight < 0
               for weight in (encoder_loss_weight, decoder_loss_weight)):
            raise ValueError("encoder_loss_weight and decoder_loss_weight must be nonnegative")
        if not encoder_loss_weight + decoder_loss_weight:
            raise ValueError("at least one SFT loss must be active")
        if model.text.layer_scalar not in ("frozen", "trainable"):
            raise ValueError("official SFT requires the model layer_scalar value")
        self._initial_scalar_mode = model.text.layer_scalar
        self.model = model.clone(text=model.text.clone(layer_scalar="trainable"))
        self.prompt_length = prompt_length
        self.canvas_size = canvas_size
        self.num_canvases = num_canvases
        self.sequence_length = prompt_length + canvas_size * num_canvases
        if self.sequence_length > model.max_seq_len:
            raise ValueError("the SFT sequence exceeds model.max_seq_len")
        # The reference allocates one full-sequence cache for training, not the
        # model's potentially much larger serving capacity. Weights are unchanged.
        self.training_model = self.model.clone(text=self.model.text.clone(max_seq_len=self.sequence_length))
        self.pretrained = pretrained
        self.pad_token_id = pad_token_id
        self.self_cond_prob = self_cond_prob
        self.safety_epsilon = safety_epsilon
        self.stop_gradient_from_denoiser_to_encoder = stop_gradient_from_denoiser_to_encoder
        self.encoder_loss_weight = encoder_loss_weight
        self.decoder_loss_weight = decoder_loss_weight
        self.inputs = InputSpec(sample=Field("text", (self.sequence_length,)))
        self.ema = None if ema_decay is None else EMASpec(optax.constant_schedule(ema_decay))

    def init(self, key: jax.Array) -> Variables:
        if self.pretrained is not None:
            if "params" not in self.pretrained:
                raise ValueError("pretrained must contain the params collection")
            if self._initial_scalar_mode == "trainable":
                return self.pretrained
            # Google makes skip_scale a parameter; Transformers declares the
            # same tensor a buffer. Move references once, under an explicit
            # model policy, without copying any parameter arrays.
            values = dict(self.pretrained)
            params = dict(values["params"])
            text = dict(params["text"])
            constants = dict(values["constants"])
            text_constants = dict(constants["text"])
            for index in range(self.model.text.num_layers):
                layer = f"layers_{index}"
                fixed = dict(text_constants[layer])
                learned = dict(text[layer])
                if "layer_scalar" in learned:
                    raise ValueError("frozen source unexpectedly contains a trainable layer_scalar")
                learned["layer_scalar"] = fixed.pop("layer_scalar")
                text[layer] = learned
                if fixed:
                    text_constants[layer] = fixed
                else:
                    del text_constants[layer]
            params["text"] = text
            values["params"] = params
            if text_constants:
                constants["text"] = text_constants
            else:
                del constants["text"]
            if constants:
                values["constants"] = constants
            else:
                del values["constants"]
            return values
        if self.model.conditioner is not None:
            raise ValueError("multimodal SFT initialization requires the complete pretrained variables")
        return self.model.init(key, jnp.zeros((1, self.canvas_size), jnp.int32))

    def loss(self, params: Variables, batch: Batch, step: Step):
        tokens = jnp.asarray(batch["text"])
        if tokens.ndim != 2 or tokens.shape[1] != self.sequence_length or not jnp.issubdtype(tokens.dtype, jnp.integer):
            raise ValueError(f"block SFT expects integer [B, {self.sequence_length}] token rows")
        tokens = tokens.astype(jnp.int32)
        response = tokens[:, self.prompt_length:]
        canvas_mask = jnp.asarray(batch.get("canvas_mask", response != self.pad_token_id), bool)
        if canvas_mask.shape != response.shape:
            raise ValueError("canvas_mask must align with all response tokens")
        full_valid = jnp.concatenate([tokens[:, :self.prompt_length] != self.pad_token_id, canvas_mask], axis=-1)
        time_key, corruption_key, canvas_key, sc_key = jax.random.split(step.key, 4)
        time = jax.random.uniform(time_key, (tokens.shape[0], 1),
                                  minval=self.safety_epsilon, maxval=1 - self.safety_epsilon)
        keep_key, noise_key = jax.random.split(corruption_key)
        keep = jax.random.bernoulli(keep_key, jnp.broadcast_to(1 - time, response.shape), mode="high")
        noise = jax.random.randint(noise_key, response.shape, 0, self.model.vocab_size)
        noisy = jnp.where(keep, response, noise)
        valid_canvases = canvas_mask[:, ::self.canvas_size].sum(axis=-1)
        selected = jax.random.randint(canvas_key, valid_canvases.shape, 0, jnp.maximum(valid_canvases, 1))
        positions, encoder_mask, encoder_keys, decoder_mask, decoder_keys = _cache_geometry(
            full_valid, selected, self.prompt_length, self.canvas_size)
        model = self.training_model
        cache = model.apply(params, tokens.shape[0], method=model.init_cache, mutable=["cache"])[1]["cache"]
        encoder_logits, mutated = model.apply(
            {**params, "cache": cache}, tokens, positions=positions, attention_mask=full_valid,
            attention_pairwise_mask=encoder_mask, attention_key_positions=encoder_keys,
            method=model.encode, train=True, mutable=["cache"])
        cache = mutated["cache"]
        if self.stop_gradient_from_denoiser_to_encoder:
            cache = jax.lax.stop_gradient(cache)
        def denoise(sc_logits):
            predicted = model.apply(
                {**params, "cache": cache}, noisy, self_conditioning_logits=sc_logits,
                train=True, positions=positions[:, self.prompt_length:],
                attention_pairwise_mask=decoder_mask, attention_key_positions=decoder_keys)
            if not isinstance(predicted, jax.Array):
                raise TypeError("a diffusion forward must return logits")
            return predicted

        zero_logits = jnp.zeros((*response.shape, self.model.vocab_size), jnp.float32)
        first = denoise(zero_logits)
        use_sc = jax.random.uniform(sc_key, (tokens.shape[0],)) < self.self_cond_prob
        sc_logits = jnp.where(use_sc[:, None, None], jax.lax.stop_gradient(first), zero_logits)
        logits = denoise(sc_logits)
        chosen = jnp.arange(response.shape[1])[None, :] // self.canvas_size == selected[:, None]
        target_mask = canvas_mask & chosen
        canvas_losses = optax.softmax_cross_entropy_with_integer_labels(logits.astype(jnp.float32), response)
        shifted = jnp.concatenate([tokens[:, 1:], jnp.full((tokens.shape[0], 1), self.pad_token_id, jnp.int32)], axis=-1)
        adjacent = full_valid & jnp.concatenate([full_valid[:, 1:], jnp.zeros((tokens.shape[0], 1), bool)], axis=-1)
        encoder_target_mask = jnp.asarray(batch.get("encoder_target_mask", adjacent), jnp.float32)
        if encoder_target_mask.shape != tokens.shape:
            raise ValueError("encoder_target_mask must align with the full sequence")
        encoder_losses = optax.softmax_cross_entropy_with_integer_labels(encoder_logits.astype(jnp.float32), shifted)
        canvas_stats, encoder_stats = _row_mean(canvas_losses, target_mask), _row_mean(encoder_losses, encoder_target_mask)
        support = self.decoder_loss_weight * target_mask.sum() + self.encoder_loss_weight * encoder_target_mask.sum()
        stats = BlockSFTStatistics(canvas_stats, encoder_stats, support)
        return stats, Aux(metrics={"canvas_ce": mean_loss(canvas_stats)[0],
                                  "encoder_ce": mean_loss(encoder_stats)[0]})

    def reduce_loss(self, stats: BlockSFTStatistics):
        canvas, _ = mean_loss(stats.canvas)
        encoder, _ = mean_loss(stats.encoder)
        return self.decoder_loss_weight * canvas + self.encoder_loss_weight * encoder, stats.support > 0
