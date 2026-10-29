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
from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

import jax
import jax.numpy as jnp
from flax import linen as nn, struct
from flax.traverse_util import flatten_dict, unflatten_dict

from dew.artifacts import TokenScores
from dew.batch import FROZEN, POSITIONS_FIELD, Batch, Variables, real_places, shifted_span, thaw, token_layout
from dew.inference.tasks import BlockGeneration, BlockSettings, recorded_processor, recorded_tokenizer
from dew.inputs import Field, InputSpec
from dew.nn.inputs import ModelInputs
from dew.nn.precision import at_least_fp32, weighted_total
from dew.nn.protocols import BlockDenoiser, CacheCapacity, Lacks, LayerScalars
from dew.nn.sharding import LOGITS, constrain
from dew.nn.visibility import same_group
from dew.objectives.base import (
    OMITTED,
    Aux,
    EMASpec,
    Objective,
    Omitted,
    ProgramModule,
    Ratio,
    Source,
    Step,
    ended,
    model_rngs,
)
from dew.objectives.lm.chunked import model_logits
from dew.objectives.lm.head import HeadedObjective

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
    positions = real_places(valid) if positions is None else positions
    packed_positions = jnp.take_along_axis(positions, ordered, axis=-1)
    encoder_mask = ordered[:, None, :] <= physical[None, :, None]
    if image_groups is not None:
        key_groups = jnp.take_along_axis(image_groups, ordered, axis=-1)
        encoder_mask |= same_group(image_groups[:, :, None], key_groups[:, None, :])
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
    total, mass = weighted_total(losses, mask.astype(losses.dtype), axis=-1)
    return Objective.row_mean(total / jnp.maximum(mass, 1), batch)


def _observed_maxima(*passes: Variables) -> Variables:
    """The attention logit maxima of several passes, merged by module path (`Aux.qk_stats`).

    A module that more than one pass observed keeps the elementwise maximum of what each sowed: a logit is
    bounded by the largest any of them reads. A module only one pass observed keeps what it sowed, as an
    image's vision tower does, since the encoder pass alone runs it.
    """
    found: dict[tuple[str, ...], list[object]] = {}
    for tree in passes:
        for path, sown in flatten_dict(tree).items():
            found.setdefault(path, []).append(sown)
    return unflatten_dict({path: jax.tree.map(lambda *leaves: jnp.max(jnp.stack(leaves), axis=0), *sown)
                           for path, sown in found.items()})


class BlockDiffusionObjective(HeadedObjective[BlockSFTStatistics]):
    """Fine-tunes DiffusionGemma on clean ``text`` rows split into a prompt and canvases.

    A row has ``prompt_length + canvas_size * num_canvases`` tokens. ``text``
    holds token arrays or `ModelInputs`, and any media in it conditions only
    the clean encoder. When the batch supplies attention validity, that
    decides which cache slots are occupied; otherwise the ``prompt_mask`` and
    ``canvas_mask`` a canvas row carries decide (`dew.data.steps.chat.ResponseCanvases`),
    and the pad ID marks a bare row's context. The optional ``canvas_mask`` and
    ``encoder_target_mask`` select the text targets, and media placeholders
    are never labels. By default an encoder target requires adjacent valid
    slots, as Google's SequenceTargetShift does. Every response token is
    corrupted, but only one valid canvas, chosen uniformly, contributes the
    diffusion cross entropy.

    `model` may be the loaded SFT source, and `variables` and `processor`
    override its own (`Objective.bind_model`), as `eos_token_ids`, the ids its
    task's responses stop at, overrides the source's (`ended`).

    Both cross entropies score the final states through the head by the one
    policy every token objective takes (`dew.objectives.lm.head`, `head_chunks`
    and `head_tile` as an LM's): its whole logits kept where they fit, and
    tiled where the fit ladder moves it, whose backward then holds no
    vocabulary-sized fp32 logits, unless an adapter on the head leaves no
    matrix to tile. The one place a full row of logits always exists is the
    first denoising pass, whose logits condition the second pass and receive
    no gradient.

    A step whose optimizer clips by the attention logit maxima (`Step.qk_stats`, a QK-Clip) gets the
    maxima of the attention its passes run through the trained projections, merged by module path
    (`_observed_maxima`): a module several passes observe keeps their elementwise maximum, and one only
    the encoder reaches keeps its own (`Aux.qk_stats`).
    """

    saved_task = BlockGeneration
    settings = BlockSettings

    def __init__(self, model: DiffusionGemma | Source, *, prompt_length: int,
                 num_canvases: int = 1, canvas_size: int | None = None,
                 variables: Variables | None | Omitted = OMITTED, pad_token_id: int = 0,
                 self_cond_prob: float = 0.5, safety_epsilon: float = 1e-4,
                 stop_gradient_from_denoiser_to_encoder: bool = False,
                 encoder_loss_weight: float = 1.0, decoder_loss_weight: float = 1.0,
                 ema_decay: float | None = None, head_chunks: int = 4,
                 head_tile: tuple[int, int] | Literal['whole', 'tiled'] | None = None,
                 processor: Processor | None | Omitted = OMITTED,
                 eos_token_ids: Sequence[int] | Omitted = OMITTED):
        self.eos_token_ids = ended(model, eos_token_ids)
        model = self.bind_model(model, variables=variables, processor=processor)
        if not (isinstance(model, BlockDenoiser) and isinstance(model, LayerScalars)):
            raise Lacks("canvas",
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
        self.use_head(head_chunks, head_tile)

    def data(self, tokenizer=None):
        from dew.data.pipeline import ObjectiveData
        from dew.data.steps.text import FixedTokens

        geometry = FixedTokens(self.sequence_length)
        return ObjectiveData(train=(geometry,), evaluate=(geometry,))

    def _process(self):
        """The task's process, over the canvas the model generates on: the one `build_task` samples
        with and `task_settings` saves."""
        from dew.diffusion.block import BlockProcess

        return BlockProcess(canvas_length=self.model.canvas_length, vocab_size=self.model.vocab_size)

    def task_settings(self) -> BlockSettings:
        """The sequence length, canvas budget, block process, pad id, end-of-sequence ids, and
        tokenizer or source processor: the task's own."""
        return BlockSettings(seq_len=self.sequence_length, max_new_tokens=self.canvas_size,
                             process=self._process(), pad_token_id=self.pad_token_id,
                             eos_token_ids=self.eos_token_ids,
                             tokenizer=recorded_tokenizer(self.processor),
                             processor=recorded_processor(self.processor))

    def bank_sites(self, store: Variables) -> tuple[DecoderBank, ...]:
        """The shared text stack, as the training model declares it over `store`."""
        return self.training_model.bank_sites(store)

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
        """The model the losses run, sized to the SFT sequence, trained, with its head."""
        return (self.head_program(self.training_model),)

    def substitute(self, modules: Sequence[nn.Module]) -> None:
        (trained,) = modules
        if not (isinstance(trained, BlockDenoiser) and isinstance(trained, CacheCapacity)):
            raise TypeError(f"the block objective trains a block denoiser with a cache capacity, not a "
                            f"{type(trained).__name__}")
        self.training_model = trained
        # The record and the task read the substitute (a run's quantization, a
        # remat rung) at the serving capacity.
        self.model = trained.with_cache_capacity(self.model.max_seq_len)

    def fresh_variables(self, key: jax.Array, held: Variables | None) -> Variables:
        return self.model.init(key, jnp.zeros((1, self.canvas_size), jnp.int32))

    def loss(self, variables: Variables, batch: Batch, step: Step):
        stats, _, qk = self._scored(variables, batch, step, train=step.training)
        return stats, Aux(metrics={"canvas_ce": stats.canvas.mean()[0],
                                  "encoder_ce": stats.encoder.mean()[0]}, qk_stats=qk)

    def _evaluation_scores(self, params: Variables, batch: Batch, step: Step) -> TokenScores:
        """Return the denoiser's cross entropy on every canvas target of the batch.

        One noise level and one canvas per row are drawn by each example, as
        in training (`Step.sample`), with dropout off. So `perplexity` over a validation
        pass is the exponential of the denoising loss per target."""
        _, scores, _ = self._scored(params, batch, step, train=False)
        assert scores is not None
        return scores

    def _validation_scores(self, params: Variables, batch: Batch, step: Step
                           ) -> tuple[BlockSFTStatistics, TokenScores]:
        stats, scores, _ = self._scored(params, batch, step, train=False)
        assert scores is not None
        return stats, scores

    def _scored(self, params: Variables, batch: Batch, step: Step, *, train: bool
                ) -> tuple[BlockSFTStatistics, TokenScores | None, Variables | None]:
        """The SFT statistics, out of training the canvas targets' scores, and the attention logit maxima
        the step asks for (`Step.qk_stats`, else None), from one scoring of the batch."""
        canvas_losses, target_mask, encoder_losses, encoder_target_mask, correct, qk = self._token_losses(
            params, batch, step, train=train)
        support = (
            self.decoder_loss_weight * self.row_mean(target_mask, batch).total
            + self.encoder_loss_weight * self.row_mean(encoder_target_mask, batch).total
        )
        stats = BlockSFTStatistics(_row_losses(canvas_losses, target_mask, batch),
                                   _row_losses(encoder_losses, encoder_target_mask, batch), support)
        return stats, None if correct is None else TokenScores(
            canvas_losses, target_mask.astype(canvas_losses.dtype), correct), qk

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
        validity, _ = token_layout(fields)
        response_valid = (
            response != self.pad_token_id if validity is None else validity[:, self.prompt_length :]
        )
        canvas_mask = jnp.asarray(batch.get("canvas_mask", response_valid), bool)
        if canvas_mask.shape != response.shape:
            raise ValueError("canvas_mask must align with all response tokens")
        # A context's own ids as its row marks them (`ResponseCanvases`' prompt_mask), so an id the
        # vocabulary also pads with is read; bare rows mark none, and read the pad id as padding.
        unmarked = tokens[:, : self.prompt_length] != self.pad_token_id
        prompt_valid = jnp.asarray(batch.get("prompt_mask", unmarked), bool)
        if prompt_valid.shape != (tokens.shape[0], self.prompt_length):
            raise ValueError("prompt_mask must align with the prompt's tokens")
        full_valid = (
            jnp.concatenate([prompt_valid, canvas_mask], axis=-1)
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

    def _corrupted(self, response: jax.Array, canvas_mask: jax.Array, step: Step):
        """Draw a noise level for each row and corrupt the response at it, each row's draws its
        example's own (`Step.sample`).

        Returns the corrupted tokens, the canvas each row scores, and whether
        each row conditions on its first pass.
        """
        rows, length = response.shape
        low, high = self.safety_epsilon, 1 - self.safety_epsilon
        time = step.sample("times", lambda key: jax.random.uniform(key, (1,), jnp.float32, low, high),
                           rows=rows)
        keep = step.sample("kept", jax.random.uniform, (length,), dtype=jnp.float32, rows=rows) < 1 - time
        vocab = self.model.vocab_size
        noise = step.sample("noise", lambda key: jax.random.randint(key, (length,), 0, vocab, jnp.int32),
                            rows=rows)
        noisy = step.boundary("noised input", jnp.where(keep, response, noise))
        eligible = canvas_mask.reshape(rows, self.num_canvases, self.canvas_size).any(axis=-1)
        valid_canvases = eligible.sum(axis=-1)
        rank = (step.sample("canvas uniform", jax.random.uniform, (), dtype=jnp.float32, rows=rows)
                * jnp.maximum(valid_canvases, 1)).astype(jnp.int32)
        selected = jnp.argmax(jnp.cumsum(eligible, axis=-1) > rank[:, None], axis=-1)
        use_sc = step.sample("self conditioning uniform", jax.random.uniform, (), dtype=jnp.float32,
                             rows=rows) < self.self_cond_prob
        return noisy, step.boundary("canvas", selected), step.boundary("self conditioning", use_sc)

    def _encoder_targets(self, batch: Batch, tokens: jax.Array, validity, full_valid: jax.Array,
                         text_slots, dtype) -> tuple[jax.Array, jax.Array]:
        """Shift the row by one and weight the targets the encoder scores.

        A default target needs its neighbour valid too, which is Google's
        SequenceTargetShift; a supplied mask is taken as given, except that
        media placeholders never become labels. The weights are `dtype`'s.
        """
        shifted = jnp.concatenate(
            [tokens[:, 1:], jnp.full((tokens.shape[0], 1), self.pad_token_id, jnp.int32)], axis=-1
        )
        adjacent = jnp.pad(shifted_span(tokens.shape, 1, valid=full_valid), ((0, 0), (0, 1)))
        encoder_target_mask = jnp.asarray(batch.get("encoder_target_mask", adjacent), dtype)
        if encoder_target_mask.shape != tokens.shape:
            raise ValueError("encoder_target_mask must align with the full sequence")
        if validity is not None:
            encoder_target_mask *= adjacent
        if text_slots is not None:
            encoder_target_mask *= jnp.concatenate(
                [text_slots[:, 1:], jnp.zeros((tokens.shape[0], 1), bool)], axis=-1
            )
        return jnp.where(encoder_target_mask != 0, shifted, 0), encoder_target_mask

    def _token_losses(self, params: Variables, batch: Batch, step: Step, *, train: bool
                      ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array | None,
                                 Variables | None]:
        """Score both SFT passes over one batch.

        Returns the denoiser's per-token cross entropies over the response
        and the encoder's over the full row, each with the mask of the
        targets it counts, the canvas's argmax hits where the head reports
        them, and the attention logit maxima of the passes that run the trained projections, merged by
        module path, where the step asks for them (`Step.qk_stats`).
        """
        params = thaw(params)
        prepared, tokens, response, validity, full_valid, canvas_mask, text_slots = self._row(batch)
        noisy, selected, use_sc = self._corrupted(response, canvas_mask, step)
        fields = prepared.token_fields
        positions, encoder_mask, encoder_keys, decoder_mask, decoder_keys = _cache_geometry(
            full_valid, selected, self.prompt_length, self.canvas_size,
            fields.get(POSITIONS_FIELD), fields.get("image_groups"))
        model = self.training_model
        rngs = model_rngs(step.key, training=train)
        observed = ["qk"] if step.qk_stats else []
        cache = model.apply(params, tokens.shape[0], method=model.init_cache, mutable=["cache"])[1]["cache"]
        encoder_kwargs = prepared.kwargs()
        encoder_kwargs.update(positions=positions, attention_mask=full_valid)
        encoder_states, mutated = model.apply(
            {**params, "cache": cache}, tokens, **encoder_kwargs,
            attention_pairwise_mask=encoder_mask, attention_key_positions=encoder_keys,
            method=model.encode, train=train, states=True, mutable=["cache", *observed], rngs=rngs,
            capture_intermediates=False)
        cache = mutated["cache"]
        if self.stop_gradient_from_denoiser_to_encoder:
            cache = jax.lax.stop_gradient(cache)

        def denoise(sc_logits):
            read = model.apply(
                {**params, "cache": cache}, noisy, self_conditioning_logits=sc_logits,
                train=train, rngs=rngs, positions=positions[:, self.prompt_length:],
                attention_pairwise_mask=decoder_mask, attention_key_positions=decoder_keys,
                states=True, mutable=observed or False)
            return read if observed else (read, {})

        # The zero signal in the dtype the model's states compute in, at least fp32, so a float64 model's
        # first pass mixes its table in float64 (`soft_embeddings`), as its second does.
        zero_logits = jnp.zeros((*response.shape, self.model.vocab_size), at_least_fp32(encoder_states.dtype))
        # The conditioning pass carries no gradient, so its states stop it before the head and nothing of
        # that pass is kept for the backward. Its maxima still count: it runs the trained projections, so
        # its logits are the trained weights' too, and the clip bounds them with the other passes'.
        first_states, first_sown = denoise(zero_logits)
        first = jax.lax.stop_gradient(constrain(model_logits(
            model, params, jax.lax.stop_gradient(first_states)), LOGITS))
        sc_logits = jnp.where(use_sc[:, None, None], first, zero_logits)
        states, second_sown = denoise(sc_logits)
        qk = None
        if step.qk_stats:
            qk = _observed_maxima(mutated["qk"], first_sown["qk"], second_sown["qk"])
        chosen = jnp.arange(response.shape[1])[None, :] // self.canvas_size == selected[:, None]
        target_mask = canvas_mask & chosen
        if text_slots is not None:
            target_mask &= text_slots[:, self.prompt_length:]
        canvas_losses, predicted, _ = self.head_terms(model, params, states, response, predict=not train)
        # The targets' weights at the precision the encoder's states compute at, at least fp32.
        shifted, encoder_target_mask = self._encoder_targets(batch, tokens, validity, full_valid, text_slots,
                                                             at_least_fp32(encoder_states.dtype))
        encoder_losses, _, _ = self.head_terms(model, params, encoder_states, shifted, predict=False)
        correct = None if predicted is None else predicted == response
        return canvas_losses, target_mask, encoder_losses, encoder_target_mask, correct, qk

    def reduce_loss(self, stats: BlockSFTStatistics):
        canvas, _ = stats.canvas.mean()
        encoder, _ = stats.encoder.mean()
        return self.decoder_loss_weight * canvas + self.encoder_loss_weight * encoder, stats.support > 0
