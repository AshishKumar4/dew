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
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from flax import linen as nn, struct

from dew.artifacts import TokenScores
from dew.inference.tasks import BlockGeneration
from dew.inputs import Field, InputSpec
from dew.nn.inputs import ModelInputs
from dew.nn.protocols import BlockDenoiser, LayerScalars
from dew.nn.sharding import LOGITS, constrain
from dew.objectives.base import (
    FROZEN,
    OMITTED,
    Aux,
    Batch,
    EMASpec,
    Objective,
    Omitted,
    ProgramModule,
    Ratio,
    Source,
    Step,
    Variables,
    thaw,
)
from dew.objectives.lm.chunked import head_cross_entropy, model_logits
from dew.records import JSON
from dew.registry import objectives

if TYPE_CHECKING:
    from dew.inference.tasks import Processor
    from dew.nn.backbones.decoder_stack import DecoderBank
    from dew.nn.diffusion_gemma import DiffusionGemma


@struct.dataclass
class BlockSFTStatistics:
    """Carry the two SFT losses separately, with the mass that supports them.

    Each is already a row mean, so the pair is not token-weighted when
    microbatches accumulate. `support` is the weighted token count behind
    both, which is what says the step scored anything at all.
    """

    canvas: Ratio
    encoder: Ratio
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


def _row_losses(losses: jax.Array, mask: jax.Array, batch: Batch) -> Ratio:
    """Average the masked losses within each row, then sum the rows (`Objective.row_mean`).

    The mass is the row count, so accumulation weighs rows equally however
    many tokens each one counted.
    """
    mass = mask.sum(axis=-1)
    return Objective.row_mean(jnp.sum(jnp.where(mask != 0, losses, 0) * mask, axis=-1) / jnp.maximum(mass, 1),
                              batch)


@objectives("block_diffusion")
class BlockDiffusionObjective(Objective[BlockSFTStatistics]):
    """Fine-tunes DiffusionGemma on clean ``text`` rows split into a prompt and canvases.

    A row has ``prompt_length + canvas_size * num_canvases`` tokens. ``text``
    holds token arrays or `ModelInputs`, and any media in it conditions only
    the clean encoder. When the batch supplies attention validity, that
    decides which cache slots are occupied; otherwise the pad ID and the
    canvas mask decide. The optional ``canvas_mask`` and
    ``encoder_target_mask`` select the text targets, and media placeholders
    are never labels. By default an encoder target requires adjacent valid
    slots, as Google's SequenceTargetShift does. Every response token is
    corrupted, but only one valid canvas, chosen uniformly, contributes the
    diffusion cross entropy.

    `model` may be the loaded SFT source, and `variables` and `processor`
    override its own (`Objective.bind_model`).

    Both cross entropies score the final states through the bounded head
    (`dew.objectives.lm.chunked.head_cross_entropy`), `head_chunks`
    vocabulary tiles at a time. The backward pass therefore never holds
    vocabulary-sized fp32 logits or the softmax of a whole row, unless an
    adapter on the head leaves no matrix to tile. The one place a full row
    of logits always exists is the first denoising pass, whose logits
    condition the second pass and receive no gradient.
    """

    saved_task = BlockGeneration

    def __init__(self, model: DiffusionGemma | Source, *, prompt_length: int,
                 num_canvases: int = 1, canvas_size: int | None = None,
                 variables: Variables | None | Omitted = OMITTED, pad_token_id: int = 0,
                 self_cond_prob: float = 0.5, safety_epsilon: float = 1e-4,
                 stop_gradient_from_denoiser_to_encoder: bool = False,
                 encoder_loss_weight: float = 1.0, decoder_loss_weight: float = 1.0,
                 ema_decay: float | None = None, head_chunks: int = 4,
                 processor: Processor | None | Omitted = OMITTED):
        model = self.bind_model(model, variables=variables, processor=processor)
        if not (isinstance(model, BlockDenoiser) and isinstance(model, LayerScalars)):
            raise TypeError(
                f"block diffusion encodes a clean prefix into a model's cache and denoises canvases "
                f"against it, with the layer scalars the published SFT trains, as DiffusionGemma "
                f"does, and a {type(model).__name__} does none of that")
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
        # The source reads its own tree; the model trained reads it with the
        # scalars moved (`held_variables`).
        self._source_model = model
        self.model = model.with_trainable_layer_scalars()
        self.prompt_length = prompt_length
        self.canvas_size = canvas_size
        self.num_canvases = num_canvases
        self.sequence_length = prompt_length + canvas_size * num_canvases
        if self.sequence_length > model.max_seq_len:
            raise ValueError("the SFT sequence exceeds model.max_seq_len")
        # The reference allocates one full-sequence cache for training, not the
        # model's potentially much larger serving capacity. Weights are unchanged.
        self.training_model = self.model.with_cache_capacity(self.sequence_length)
        self.pad_token_id = pad_token_id
        self.self_cond_prob = self_cond_prob
        self.safety_epsilon = safety_epsilon
        self.stop_gradient_from_denoiser_to_encoder = stop_gradient_from_denoiser_to_encoder
        self.encoder_loss_weight = encoder_loss_weight
        self.decoder_loss_weight = decoder_loss_weight
        self.inputs = InputSpec(sample=Field("text", (self.sequence_length,)))
        self.ema = EMASpec.constant(ema_decay)
        self.head_chunks = head_chunks

    def task_record(self) -> Mapping[str, JSON]:
        """The sequence length, canvas budget, tokenizer and block process."""
        from dew.diffusion.block import BlockProcess
        from dew.inference.tasks import recorded_tokenizer
        return {'seq_len': self.sequence_length, 'sample_tokens': self.canvas_size,
                'tokenizer': recorded_tokenizer(self.processor),
                'process': BlockProcess(self.canvas_size, self.model.vocab_size).to_json()}

    def build_task(self, variables: Variables, *,
                   processor: Processor | None | Omitted = OMITTED) -> BlockGeneration:
        """Return the model over `variables` as a `BlockGeneration` task.

        The sampler keeps the published defaults, and the caller sets the
        tokenizer's EOS ids.
        """
        from dew.diffusion.block import BlockProcess
        from dew.inference.tasks import BlockGeneration

        process = BlockProcess(canvas_length=self.model.canvas_length, vocab_size=self.model.vocab_size)
        return BlockGeneration(self.model, variables, process,
                               self.processor if processor is OMITTED else processor,
                               pad_token_id=self.pad_token_id)

    @property
    def bank_sites(self) -> tuple[DecoderBank, ...]:
        """The shared text stack, as the training model declares it."""
        return self.training_model.bank_sites

    def held_variables(self) -> Variables | None:
        """The SFT source this objective starts from, its split kept, read as
        the trained model reads it (`trainable_variables`); or None for a
        fresh init.

        A split tree's scalars go under `frozen` beside the rest of the base,
        so an adapter's run moves its factors alone, as the source's frozen
        scalars did not move either. The move rearranges the tree and copies
        no array.
        """
        pretrained = self.variables
        if pretrained is None:
            return None
        if "params" not in pretrained:
            raise ValueError("variables must contain the params collection")
        if FROZEN not in pretrained:
            return self._source_model.trainable_variables(pretrained)
        moved = self._source_model.trainable_variables({**pretrained, "params": pretrained[FROZEN]})
        return {**moved, "params": pretrained["params"], FROZEN: moved["params"]}

    def program_key(self) -> tuple[ProgramModule, ...]:
        """The model the losses run, sized to the SFT sequence, trained."""
        return (ProgramModule(self.training_model, None, trained=True),)

    def substitute(self, modules: Sequence[nn.Module]) -> None:
        (self.training_model,) = modules

    def fresh_variables(self, key: jax.Array, held: Variables | None) -> Variables:
        return self.model.init(key, jnp.zeros((1, self.canvas_size), jnp.int32))

    def loss(self, variables: Variables, batch: Batch, step: Step):
        canvas_losses, target_mask, encoder_losses, encoder_target_mask, _ = self._token_losses(
            variables, batch, step.key, train=True)
        canvas_stats, encoder_stats = (
            _row_losses(canvas_losses, target_mask, batch),
            _row_losses(encoder_losses, encoder_target_mask, batch),
        )
        support = (
            self.decoder_loss_weight * self.row_mean(target_mask, batch).total
            + self.encoder_loss_weight * self.row_mean(encoder_target_mask, batch).total
        )
        stats = BlockSFTStatistics(canvas_stats, encoder_stats, support)
        return stats, Aux(metrics={"canvas_ce": canvas_stats.mean()[0],
                                  "encoder_ce": encoder_stats.mean()[0]})

    def _evaluation_scores(self, params: Variables, batch: Batch, key: jax.Array) -> TokenScores:
        """Return the denoiser's cross entropy on every canvas target of the batch.

        One noise level and one canvas per row are drawn from the pass's key,
        as in training, with dropout off. So `perplexity` over a validation
        pass is the exponential of the denoising loss per target."""
        canvas_losses, target_mask, _, _, correct = self._token_losses(params, batch, key, train=False)
        assert correct is not None
        return TokenScores(canvas_losses, target_mask.astype(canvas_losses.dtype), correct)

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
        if (
            tokens.ndim != 2
            or tokens.shape[1] != self.sequence_length
            or not jnp.issubdtype(tokens.dtype, jnp.integer)
        ):
            raise ValueError(f"block SFT expects integer [B, {self.sequence_length}] token rows")
        tokens = tokens.astype(jnp.int32)
        fields = prepared.token_fields
        response = tokens[:, self.prompt_length:]
        validity = fields.get("attention_mask")
        response_valid = (
            response != self.pad_token_id if validity is None else validity[:, self.prompt_length :]
        )
        canvas_mask = jnp.asarray(batch.get("canvas_mask", response_valid), bool)
        if canvas_mask.shape != response.shape:
            raise ValueError("canvas_mask must align with all response tokens")
        full_valid = (
            jnp.concatenate([tokens[:, : self.prompt_length] != self.pad_token_id, canvas_mask], axis=-1)
            if validity is None
            else jnp.asarray(validity, bool)
        )
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
        shifted = jnp.concatenate(
            [tokens[:, 1:], jnp.full((tokens.shape[0], 1), self.pad_token_id, jnp.int32)], axis=-1
        )
        adjacent = full_valid & jnp.concatenate(
            [full_valid[:, 1:], jnp.zeros((tokens.shape[0], 1), bool)], axis=-1
        )
        encoder_target_mask = jnp.asarray(batch.get("encoder_target_mask", adjacent), jnp.float32)
        if encoder_target_mask.shape != tokens.shape:
            raise ValueError("encoder_target_mask must align with the full sequence")
        if validity is not None:
            encoder_target_mask *= adjacent
        if text_slots is not None:
            encoder_target_mask *= jnp.concatenate(
                [text_slots[:, 1:], jnp.zeros((tokens.shape[0], 1), bool)], axis=-1
            )
        return jnp.where(encoder_target_mask != 0, shifted, 0), encoder_target_mask

    def _token_losses(self, params: Variables, batch: Batch, key: jax.Array, *, train: bool
                      ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array | None]:
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
            return model.apply(
                {**params, "cache": cache}, noisy, self_conditioning_logits=sc_logits,
                train=train, positions=positions[:, self.prompt_length:],
                attention_pairwise_mask=decoder_mask, attention_key_positions=decoder_keys,
                states=True)

        zero_logits = jnp.zeros((*response.shape, self.model.vocab_size), jnp.float32)
        # The conditioning pass carries no gradient, so its states stop it
        # before the head and nothing of that pass is kept for the backward.
        first = jax.lax.stop_gradient(constrain(model_logits(
            model, params, jax.lax.stop_gradient(denoise(zero_logits))), LOGITS))
        use_sc = jax.random.uniform(sc_key, (tokens.shape[0],)) < self.self_cond_prob
        sc_logits = jnp.where(use_sc[:, None, None], first, zero_logits)
        states = denoise(sc_logits)
        chosen = jnp.arange(response.shape[1])[None, :] // self.canvas_size == selected[:, None]
        target_mask = canvas_mask & chosen
        if text_slots is not None:
            target_mask &= text_slots[:, self.prompt_length:]
        canvas_losses, predicted, _ = head_cross_entropy(model, params, states, response, self.head_chunks,
                                                         predict=not train)
        shifted, encoder_target_mask = self._encoder_targets(batch, tokens, validity, full_valid, text_slots)
        encoder_losses, _, _ = head_cross_entropy(model, params, encoder_states, shifted, self.head_chunks,
                                                  predict=False)
        correct = None if predicted is None else predicted == response
        return canvas_losses, target_mask, encoder_losses, encoder_target_mask, correct

    def reduce_loss(self, stats: BlockSFTStatistics):
        canvas, _ = stats.canvas.mean()
        encoder, _ = stats.encoder.mean()
        return self.decoder_loss_weight * canvas + self.encoder_loss_weight * encoder, stats.support > 0
