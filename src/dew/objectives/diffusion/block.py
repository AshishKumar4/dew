"""Supervised fine-tuning for DiffusionGemma: a clean encoder pass and a denoising pass.

A row is a prompt followed by fixed-size canvases. The encoder scores the
whole row with next-token cross entropy; the denoiser scores one uniformly
chosen canvas from its corrupted tokens. Both losses average within a row
before averaging rows, and their sufficient statistics stay separate so
gradient accumulation does not token-weight the pair.

This is the published post-release SFT loss, not the earlier SD·RL stage.

Reference: gemma bf0b49901a428d13e9c2b2629f0eb9c153d3cbd3,
``diffusion/hackable_diffusion_adapter/hd/sft_model.py`` and the Sudoku config.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import optax
from flax import struct

from dew.artifacts import TokenScores
from dew.inputs import Field, InputSpec
from dew.nn.diffusion_gemma import DiffusionGemma
from dew.nn.inputs import ModelInputs
from dew.nn.sharding import LOGITS, constrain
from dew.objectives.base import (
    Aux,
    Batch,
    EMASpec,
    Mean,
    Objective,
    PathFilter,
    Step,
    Variables,
    freeze,
    mean_loss,
    thaw,
)
from dew.objectives.lm.chunked import chunked_cross_entropy, head_logits
from dew.registry import objectives

if TYPE_CHECKING:
    from dew.inference.tasks import BlockGeneration, Processor
    from dew.nn.backbones.causal_transformer import DecoderBank
    from dew.training.state import TrainState


@struct.dataclass
class BlockSFTStatistics:
    """Carry the two SFT losses separately, with the mass that supports them.

    Each is already a row mean, so the pair is not token-weighted when
    microbatches accumulate. `support` is the weighted token count behind
    both, which is what says the step scored anything at all.
    """

    canvas: Mean
    encoder: Mean
    support: jax.Array


def _positions(valid: jax.Array) -> jax.Array:
    """Number the valid tokens of each row from zero, padding included."""
    counts = jnp.cumsum(valid, axis=-1)
    return counts - (counts >= 1)


def _cache_geometry(valid: jax.Array, selected: jax.Array, prompt_length: int, canvas_size: int,
                    positions: jax.Array | None = None, image_groups: jax.Array | None = None):
    """Build the attention masks and positions of one SFT step.

    They represent the reference's circular overlay without mutating the
    encoder's K/V.

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
    positions = _positions(valid) if positions is None else positions
    packed_positions = jnp.take_along_axis(positions, ordered, axis=-1)
    encoder_mask = ordered[:, None, :] <= physical[None, :, None]
    if image_groups is not None:
        key_groups = jnp.take_along_axis(image_groups, ordered, axis=-1)
        encoder_mask |= ((image_groups[:, :, None] == key_groups[:, None, :])
                         & (image_groups[:, :, None] >= 0))
    encoder_mask &= packed_valid[:, None, :]
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
    """Average the masked losses within each row, then sum the rows.

    The mass is the row count, so accumulation weighs rows equally however
    many tokens each one counted.
    """
    mass = mask.sum(axis=-1)
    row_losses = jnp.sum(jnp.where(mask != 0, losses, 0) * mask, axis=-1) / jnp.maximum(mass, 1)
    return Mean(row_losses.sum(), jnp.asarray(losses.shape[0], jnp.int32))


@objectives("block_diffusion")
class BlockDiffusionObjective(Objective[BlockSFTStatistics]):
    """Fine-tune DiffusionGemma on clean ``text`` rows split into prompt and canvases.

    A row has ``prompt_length + canvas_size * num_canvases`` tokens. ``text``
    accepts token arrays or ModelInputs; media conditions only the clean encoder.
    Supplied attention validity controls cache occupancy, otherwise the pad ID
    and canvas mask do. Optional ``canvas_mask`` and ``encoder_target_mask``
    select text targets; media placeholders are never labels. Default encoder
    targets require adjacent valid slots, matching Google's SequenceTargetShift.
    Every response token is corrupted, but only a uniformly selected valid
    canvas contributes diffusion CE.

    `trainable` selects the parameter leaves the optimizer moves, by their
    full path (`dew.objectives.base.PathFilter`), the way `LMObjective`
    takes it; the rest of the tree is kept under `frozen`, which `init`
    returns and a checkpoint stores. An adapter's own filter
    (`dew.lora.LoRA.trainable`) goes here. None trains every leaf.

    Both cross-entropies score the final states through the bounded head
    (`dew.objectives.lm.chunked.chunked_cross_entropy`), `head_chunks`
    vocabulary tiles at a time, so no vocabulary-sized fp32 logits or
    softmax of a whole row is held for the backward pass. The first
    denoising pass, whose logits condition the second and carry no
    gradient, is the one place a full row of logits exists.
    """

    def __init__(self, model: DiffusionGemma, *, prompt_length: int,
                 num_canvases: int = 1, canvas_size: int | None = None,
                 pretrained: Variables | None = None, pad_token_id: int = 0,
                 self_cond_prob: float = 0.5, safety_epsilon: float = 1e-4,
                 stop_gradient_from_denoiser_to_encoder: bool = False,
                 encoder_loss_weight: float = 1.0, decoder_loss_weight: float = 1.0,
                 ema_decay: float | None = None, trainable: PathFilter | None = None,
                 head_chunks: int = 4):
        canvas_size = model.canvas_length if canvas_size is None else canvas_size
        for name, value in (("prompt_length", prompt_length), ("num_canvases", num_canvases),
                            ("canvas_size", canvas_size), ("head_chunks", head_chunks)):
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
        self.trainable = trainable
        self.head_chunks = head_chunks

    def pipeline(self, state: TrainState, *, ema: bool = True, processor: Processor | None = None) -> BlockGeneration:
        """Publish the state's weights as a `BlockGeneration` task.

        The sampler keeps the published defaults, and the tokenizer's EOS
        ids are the caller's to set.
        """
        from dew.diffusion.block import BlockProcess
        from dew.inference.tasks import BlockGeneration

        process = BlockProcess(canvas_length=self.model.canvas_length, vocab_size=self.model.vocab_size)
        return BlockGeneration(self.model, thaw(self._pipeline_weights(state, ema)), process, processor,
                               pad_token_id=self.pad_token_id)

    @property
    def bank_sites(self) -> tuple[DecoderBank, ...]:
        """Name the shared text stack, as the training model declares it."""
        return self.training_model.bank_sites

    def held_variables(self) -> Variables | None:
        """Return the SFT source this objective starts from."""
        return self.pretrained

    def init(self, key: jax.Array, variables: Variables | None = None) -> Variables:
        tree = self._whole_tree(key, variables)
        return tree if self.trainable is None else freeze(tree, self.trainable)

    def _whole_tree(self, key: jax.Array, variables: Variables | None) -> Variables:
        """Return the model's variables in one `params` collection.

        Either the source with its frozen split undone and the layer
        scalars moved, or a fresh init.
        """
        pretrained = self.pretrained if variables is None else variables
        if pretrained is not None:
            if "params" not in pretrained:
                raise ValueError("pretrained must contain the params collection")
            pretrained = thaw(pretrained)
            if self._initial_scalar_mode == "trainable":
                return pretrained
            # Google makes skip_scale a parameter; Transformers declares the
            # same tensor a buffer. Move references once, under an explicit
            # model policy, without copying any parameter arrays.
            values = dict(pretrained)
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
        canvas_losses, target_mask, encoder_losses, encoder_target_mask = self._token_losses(
            params, batch, step.key, train=True)
        canvas_stats, encoder_stats = _row_mean(canvas_losses, target_mask), _row_mean(encoder_losses, encoder_target_mask)
        support = self.decoder_loss_weight * target_mask.sum() + self.encoder_loss_weight * encoder_target_mask.sum()
        stats = BlockSFTStatistics(canvas_stats, encoder_stats, support)
        return stats, Aux(metrics={"canvas_ce": mean_loss(canvas_stats)[0],
                                  "encoder_ce": mean_loss(encoder_stats)[0]})

    def evaluate(self, params: Variables, batch: Batch, step: Step) -> TokenScores:
        """Score the denoiser's cross entropy on every canvas target of the batch.

        One noise level and one canvas per row are drawn from the pass's key,
        as training draws them, with dropout off and the averaged weights
        when the run keeps them, so `perplexity` over a validation pass is
        exp of the denoising loss per target."""
        params = params if step.ema is None else step.ema
        canvas_losses, target_mask, _, _ = self._token_losses(params, batch, step.key, train=False)
        return TokenScores(losses=canvas_losses, weights=target_mask.astype(canvas_losses.dtype))

    def _row(self, batch: Batch):
        """Read one batch's rows and the masks every later phase reads.

        Returns the `ModelInputs` it came from, the int32 `tokens`, the
        `response` half of them, the supplied `validity` or None, the
        `full_valid` occupancy of the whole row, the `canvas_mask` of
        response tokens that may be targets, and `text_slots`, which is
        None unless media placeholders have to be kept out.
        """
        value = batch["text"]
        prepared = value if isinstance(value, ModelInputs) else ModelInputs(jnp.asarray(value))
        tokens = prepared.tokens
        if tokens.ndim != 2 or tokens.shape[1] != self.sequence_length or not jnp.issubdtype(tokens.dtype, jnp.integer):
            raise ValueError(f"block SFT expects integer [B, {self.sequence_length}] token rows")
        tokens = tokens.astype(jnp.int32)
        fields = prepared.token_fields
        response = tokens[:, self.prompt_length:]
        validity = fields.get("attention_mask")
        response_valid = response != self.pad_token_id if validity is None else validity[:, self.prompt_length:]
        canvas_mask = jnp.asarray(batch.get("canvas_mask", response_valid), bool)
        if canvas_mask.shape != response.shape:
            raise ValueError("canvas_mask must align with all response tokens")
        full_valid = (jnp.concatenate([tokens[:, :self.prompt_length] != self.pad_token_id, canvas_mask], axis=-1)
                      if validity is None else jnp.asarray(validity, bool))
        if full_valid.shape != tokens.shape:
            raise ValueError("attention_mask must align with the full sequence")
        canvas_mask &= full_valid[:, self.prompt_length:]
        image_indices = fields.get("image_indices")
        text_slots = None if image_indices is None else image_indices < 0
        if text_slots is not None:
            # Zero-weight targets still reach embeddings and integer-label CE.
            response = jnp.where(text_slots[:, self.prompt_length:], response, 0)
        return prepared, tokens, response, validity, full_valid, canvas_mask, text_slots

    def _corrupted(self, response: jax.Array, canvas_mask: jax.Array, key: jax.Array):
        """Draw one noise level and corrupt the response at it.

        Returns the corrupted tokens, the canvas each row scores, and the
        key the self-conditioning draw still needs.
        """
        time_key, corruption_key, canvas_key, sc_key = jax.random.split(key, 4)
        time = jax.random.uniform(time_key, (response.shape[0], 1),
                                  minval=self.safety_epsilon, maxval=1 - self.safety_epsilon)
        keep_key, noise_key = jax.random.split(corruption_key)
        keep = jax.random.bernoulli(keep_key, jnp.broadcast_to(1 - time, response.shape), mode="high")
        noise = jax.random.randint(noise_key, response.shape, 0, self.model.vocab_size)
        noisy = jnp.where(keep, response, noise)
        valid_canvases = canvas_mask[:, ::self.canvas_size].sum(axis=-1)
        selected = jax.random.randint(canvas_key, valid_canvases.shape, 0, jnp.maximum(valid_canvases, 1))
        return noisy, selected, sc_key

    def _encoder_targets(self, batch: Batch, tokens: jax.Array, validity, full_valid: jax.Array,
                         text_slots) -> tuple[jax.Array, jax.Array]:
        """Shift the row by one and weight the targets the encoder scores.

        A default target needs its neighbour valid too, which is Google's
        SequenceTargetShift; a supplied mask is taken as given, except that
        media placeholders never become labels.
        """
        shifted = jnp.concatenate([tokens[:, 1:], jnp.full((tokens.shape[0], 1), self.pad_token_id, jnp.int32)], axis=-1)
        adjacent = full_valid & jnp.concatenate([full_valid[:, 1:], jnp.zeros((tokens.shape[0], 1), bool)], axis=-1)
        encoder_target_mask = jnp.asarray(batch.get("encoder_target_mask", adjacent), jnp.float32)
        if encoder_target_mask.shape != tokens.shape:
            raise ValueError("encoder_target_mask must align with the full sequence")
        if validity is not None:
            encoder_target_mask *= adjacent
        if text_slots is not None:
            encoder_target_mask *= jnp.concatenate([text_slots[:, 1:], jnp.zeros((tokens.shape[0], 1), bool)], axis=-1)
        return jnp.where(encoder_target_mask != 0, shifted, 0), encoder_target_mask

    def _token_losses(self, params: Variables, batch: Batch, key: jax.Array, *, train: bool
                      ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        """Score both SFT passes over one batch.

        Returns the denoiser's per-token cross entropies over the response
        and the encoder's over the full row, each with the mask of the
        targets it counts.
        """
        params = thaw(params)
        prepared, tokens, response, validity, full_valid, canvas_mask, text_slots = self._row(batch)
        noisy, selected, sc_key = self._corrupted(response, canvas_mask, key)
        fields = prepared.token_fields
        positions, encoder_mask, encoder_keys, decoder_mask, decoder_keys = _cache_geometry(
            full_valid, selected, self.prompt_length, self.canvas_size,
            fields.get("positions"), fields.get("image_groups"))
        model = self.training_model
        softcap, precision = model.text.final_logit_softcap, model.text.precision
        # The head as stored, so what the losses keep for their backward is
        # the table itself and not a transposed copy of it.
        head, stored = model.apply(params, params["params"], method=type(model).head_table)
        vocab_major = bool(stored)
        cache = model.apply(params, tokens.shape[0], method=model.init_cache, mutable=["cache"])[1]["cache"]
        encoder_kwargs = prepared.kwargs()
        encoder_kwargs.update(positions=positions, attention_mask=full_valid)
        encoder_states, mutated = model.apply(
            {**params, "cache": cache}, tokens, **encoder_kwargs,
            attention_pairwise_mask=encoder_mask, attention_key_positions=encoder_keys,
            method=model.encode, train=train, states=True, mutable=["cache"], rngs=None,
            capture_intermediates=False)
        cache = mutated["cache"]
        if self.stop_gradient_from_denoiser_to_encoder:
            cache = jax.lax.stop_gradient(cache)
        def denoise(sc_logits):
            predicted = model.apply(
                {**params, "cache": cache}, noisy, self_conditioning_logits=sc_logits,
                train=train, positions=positions[:, self.prompt_length:],
                attention_pairwise_mask=decoder_mask, attention_key_positions=decoder_keys,
                states=True)
            if not isinstance(predicted, jax.Array):
                raise TypeError("a diffusion forward must return its final states")
            return predicted

        zero_logits = jnp.zeros((*response.shape, self.model.vocab_size), jnp.float32)
        # The conditioning pass carries no gradient, so its states stop it
        # before the head and nothing of that pass is kept for the backward.
        first = jax.lax.stop_gradient(constrain(head_logits(
            jax.lax.stop_gradient(denoise(zero_logits)), head, softcap=softcap,
            precision=precision, vocab_major=vocab_major), LOGITS))
        use_sc = jax.random.uniform(sc_key, (tokens.shape[0],)) < self.self_cond_prob
        sc_logits = jnp.where(use_sc[:, None, None], first, zero_logits)
        states = denoise(sc_logits)
        chosen = jnp.arange(response.shape[1])[None, :] // self.canvas_size == selected[:, None]
        target_mask = canvas_mask & chosen
        if text_slots is not None:
            target_mask &= text_slots[:, self.prompt_length:]
        canvas_losses, _, _ = chunked_cross_entropy(
            states, head, response, self.head_chunks, softcap=softcap, precision=precision,
            vocab_major=vocab_major, predict=False)
        shifted, encoder_target_mask = self._encoder_targets(batch, tokens, validity, full_valid, text_slots)
        encoder_losses, _, _ = chunked_cross_entropy(
            encoder_states, head, shifted, self.head_chunks, softcap=softcap, precision=precision,
            vocab_major=vocab_major, predict=False)
        return canvas_losses, target_mask, encoder_losses, encoder_target_mask

    def reduce_loss(self, stats: BlockSFTStatistics):
        canvas, _ = mean_loss(stats.canvas)
        encoder, _ = mean_loss(stats.encoder)
        return self.decoder_loss_weight * canvas + self.encoder_loss_weight * encoder, stats.support > 0
