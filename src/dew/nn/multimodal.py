"""Image and audio conditioning around the shared causal decoder."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import TYPE_CHECKING, Self

import jax
import jax.numpy as jnp
from flax import linen as nn, struct
from flax.typing import Dtype, PrecisionLike

from dew.nn.backbones.causal_transformer import CausalTransformer, DecoderBank
from dew.nn.protocols import ProjectionGroup
from dew.nn.vision import Gemma3nProjectorModule, Gemma3nVision, ProjectorBase, TowerBase
from dew.registry import models

if TYPE_CHECKING:
    from dew.nn.mla import MLAMixer
    from dew.records import JSON


@struct.dataclass
class Fusion:
    """Decoder token identities, their scaled, conditioned embeddings, and
    which positions media fill."""

    tokens: jax.Array
    embeddings: jax.Array
    media: jax.Array


class VisionConditioner(nn.Module):
    """An image encoder and projector with row-aligned media inputs.

    Pixels have shape [batch, images, channels, height, width]. All images
    in a numeric batch share their processed resolution; the processor keeps
    per-image lengths and does not substitute preprocessing inside the model.
    `train` reaches the one tower with stochastic layers, Gemma 3n's
    MobileNet and its drop-path.
    """

    vision: TowerBase
    projection: ProjectorBase
    dtype: Dtype | None = None
    precision: PrecisionLike = None

    def setup(self):
        self.tower = self.vision.build().clone(dtype=self.dtype, precision=self.precision)
        self.projector = self.projection.build().clone(dtype=self.dtype, precision=self.precision)

    def __call__(self, conditioning: Mapping[str, jax.Array], train: bool = False) -> jax.Array:
        pixels = conditioning["pixel_values"]
        positions = conditioning.get("image_position_ids")
        grid = conditioning.get("image_grid_thw")
        if positions is not None and grid is not None:
            raise ValueError("image positions and grid_thw belong to different tower inputs")
        if pixels.ndim != (4 if positions is not None or grid is not None else 5):
            raise ValueError("pixel_values must be row-aligned NCHW images or positioned patch pixels")
        batch, images = pixels.shape[:2]
        flat = pixels.reshape(batch * images, *pixels.shape[2:])
        if grid is not None:
            features = self.tower(flat, grid_thw=grid.reshape(batch * images, 3))
        elif isinstance(self.vision, Gemma3nVision):
            features = self.tower(flat, train=train)
        elif positions is None:
            features = self.tower(flat)
        else:
            features = self.tower(
                flat, pixel_position_ids=positions.reshape(batch * images, *positions.shape[2:])
            )
        projected = self.projector(features)
        return projected.reshape(batch, images * projected.shape[1], projected.shape[-1])

    def initialize_parameters(self) -> None:
        """Create the media leaves during an ordinary token-only model init.

        Fixed-resolution towers use their configured image size. Other towers
        need only one pooling/merge block to create resolution-independent
        parameters; real batches supply their processed geometry later.
        """
        geometry = self.vision.geometry()
        side = geometry.image_size
        if side is None:
            side = (geometry.patch_size or 16) * (geometry.block_size or 2)
        channels = geometry.channels or 3
        self({"pixel_values": jnp.zeros((1, 1, channels, side, side), self.dtype or jnp.float32)})


    def fuse(self, tokens: jax.Array, embeddings: jax.Array,
             image_indices: jax.Array, conditioning: Mapping[str, jax.Array],
             train: bool = False) -> Fusion:
        """Replace marked text slots with the corresponding soft feature."""
        if image_indices.shape != tokens.shape:
            raise ValueError("image_indices must align with the token rows")
        return Fusion(tokens, _place(embeddings, self(conditioning, train=train), image_indices),
                      image_indices >= 0)


class AudioConditioner(nn.Module):
    """An audio encoder and projector over row-aligned clips.

    Features have shape [batch, clips, frames, mel] with a True-for-valid
    frame mask. The table returned holds every clip's encoded frames back to
    back, ``clips * capacity`` per row, where capacity is the encoded frame
    count, or ``soft_tokens`` when the source fixes the slot count per clip
    (Gemma 3n): there, padded frames and the slots past the encoded frames
    carry the embedder's padding token, as modeling_gemma3n.py does.
    """

    audio: TowerBase
    projection: ProjectorBase
    soft_tokens: int | None = None
    padding_id: int | None = None
    dtype: Dtype | None = None
    precision: PrecisionLike = None

    def setup(self):
        # Named for the shared root scope beside the vision tower and projector.
        self.audio_tower = self.audio.build().clone(dtype=self.dtype, precision=self.precision)
        self.audio_projector = self.projection.build().clone(dtype=self.dtype, precision=self.precision)

    def __call__(self, conditioning: Mapping[str, jax.Array]) -> jax.Array:
        features = conditioning["input_features"]
        mask = conditioning["input_features_mask"]
        if features.ndim != 4 or mask.shape != features.shape[:3]:
            raise ValueError("input_features must be [B, A, T, F] with a [B, A, T] frame mask")
        batch, clips = features.shape[:2]
        encoding = self.audio_tower(features.reshape(batch * clips, *features.shape[2:]),
                              mask.reshape(batch * clips, features.shape[2]))
        projected = self.audio_projector(encoding.features)
        if self.soft_tokens is not None:
            if not isinstance(self.audio_projector, Gemma3nProjectorModule) or self.padding_id is None:
                raise ValueError("fixed audio slots require the Gemma 3n embedder and its padding token")
            padding = self.audio_projector.embed_hard(jnp.full((1, 1), self.padding_id, jnp.int32)).astype(
                projected.dtype
            )
            projected = jnp.where(encoding.mask[..., None], projected, padding)
            missing = self.soft_tokens - projected.shape[1]
            if missing < 0:
                raise ValueError(
                    f"{projected.shape[1]} encoded audio frames exceed the {self.soft_tokens} slots per clip"
                )
            projected = jnp.concatenate(
                [projected, jnp.broadcast_to(padding, (projected.shape[0], missing, projected.shape[-1]))],
                axis=1,
            )
        return projected.reshape(batch, clips * projected.shape[1], projected.shape[-1])

    def initialize_parameters(self) -> None:
        """Create the audio leaves during a token-only model init."""
        mel = self.audio.geometry().mel_features or 128
        frames = 16 if self.soft_tokens is None else 16 * self.soft_tokens
        self({"input_features": jnp.zeros((1, 1, frames, mel), self.dtype or jnp.float32),
              "input_features_mask": jnp.ones((1, 1, frames), bool)})


def _place(embeddings: jax.Array, table: jax.Array, indices: jax.Array) -> jax.Array:
    """Replace the marked slots of embeddings with the indexed table rows."""
    picked = table[jnp.arange(embeddings.shape[0])[:, None], jnp.maximum(indices, 0)]
    return jnp.where((indices >= 0)[..., None], picked.astype(embeddings.dtype), embeddings)



@models("multimodal_transformer")
class MultimodalTransformer(nn.Module):
    """Conditions the shared decoder on images and audio, using the ordinary decoder cache and head.

    ``image_indices`` and ``audio_indices`` give, for each token slot, the
    index of the soft feature that replaces it, or -1 for a text token. The
    towers run during conditioned forward and prefill calls; later decode
    steps read the cached language states and do not run the encoders again.
    The parameters keep the existing `language_model`, tower and projector
    names, and audio adds `audio_tower` and `audio_projector`. Gemma 3n also
    embeds its hard vision and audio vocabulary ranges through the embedders,
    and keeps placeholder ids for its per-layer inputs, as modeling_gemma3n.py
    does.
    """

    language_model: CausalTransformer
    vision: TowerBase
    projection: ProjectorBase
    family: str
    image_token_id: int
    pad_token_id: int = 0
    extra_placeholder_ids: tuple[int, ...] = ()
    audio: TowerBase | None = None
    audio_projection: ProjectorBase | None = None
    audio_soft_tokens: int | None = None
    dtype: Dtype | None = None
    precision: PrecisionLike = None

    def setup(self):
        self.conditioner = VisionConditioner(
            self.vision, self.projection,
            dtype=self.dtype, precision=self.precision)
        nn.share_scope(self.conditioner, self)
        if (self.audio is None) != (self.audio_projection is None):
            raise ValueError("an audio tower and its projector arrive together")
        if self.audio is not None and self.audio_projection is not None:
            self.audio_conditioner = AudioConditioner(
                self.audio, self.audio_projection, self.audio_soft_tokens,
                self.vocab_size - 1 if self.family == "gemma3n" else None,
                dtype=self.dtype, precision=self.precision)
            nn.share_scope(self.audio_conditioner, self)

    @property
    def vocab_size(self) -> int:
        return self.language_model.vocab_size

    @property
    def bank_sites(self) -> tuple[DecoderBank, ...]:
        return tuple(DecoderBank(("language_model", *site.namespace), site.view, site.scanned)
                     for site in self.language_model.bank_sites)

    @property
    def max_seq_len(self) -> int:
        return self.language_model.max_seq_len

    @property
    def emb_features(self) -> int:
        return self.language_model.emb_features

    @property
    def final_logit_softcap(self) -> float | None:
        return self.language_model.final_logit_softcap

    @property
    def causal(self) -> bool:
        return self.language_model.causal

    @property
    def num_nextn_predict_layers(self) -> int:
        return self.language_model.num_nextn_predict_layers

    def _conditioned_embeddings(self, tokens, image_indices, conditioning, train=False,
                                audio_indices=None) -> Fusion:
        """Decoder token identities and their embeddings with media slots filled.

        Marked slots read the towers; the rest read the embedding table. Gemma
        3n keeps placeholder ids for its per-layer inputs and embeds its hard
        media vocabulary ranges, so it runs this path with no payloads too.
        """
        if conditioning is not None and image_indices is None and audio_indices is None:
            raise ValueError("conditioned inputs require image_indices or audio_indices")
        media = jnp.zeros(tokens.shape, bool)
        for indices in (image_indices, audio_indices):
            if indices is not None:
                if indices.shape != tokens.shape:
                    raise ValueError("media indices must align with the token rows")
                media = media | (indices >= 0)
        if self.family == "gemma3n":
            # Placeholder ids feed the per-layer inputs; the hard vocabulary
            # ranges above the per-layer table read the embedders instead, on
            # every call because sampling can emit them.
            decoder_tokens = jnp.where(
                (tokens >= 0) & (tokens < self.language_model.per_layer_vocab), tokens, 0
            )
        else:
            decoder_tokens = jnp.where(media, 0, tokens)
        embeddings = self.language_model.scaled_embeddings(
            self.language_model.embed_tokens(decoder_tokens))
        if self.family == "gemma3n":
            embedders = [self.conditioner.projector]
            if self.audio is not None:
                embedders.append(self.audio_conditioner.audio_projector)
            for embedder in embedders:
                if not isinstance(embedder, Gemma3nProjectorModule):
                    raise TypeError("Gemma 3n media embedders carry the hard vocabulary")
                embeddings = embedder.merge_hard_embeddings(embeddings, tokens)
        if conditioning is not None and image_indices is not None:
            embeddings = self.conditioner.fuse(decoder_tokens, embeddings, image_indices,
                                               conditioning, train=train).embeddings
        if conditioning is not None and audio_indices is not None:
            if self.audio is None:
                raise ValueError("audio_indices require an audio tower")
            embeddings = _place(embeddings, self.audio_conditioner(conditioning), audio_indices)
        return Fusion(decoder_tokens, embeddings, media)

    def mtp_hidden_states(self, hidden, tokens, train: bool = False, positions=None,
                          segment_ids=None, image_indices=None, conditioning=None,
                          attention_mask=None, image_groups=None, rotary_positions=None):
        """Run the multi-token prediction layers over the same media embeddings as the main decoder."""
        embeddings = None
        if conditioning is not None:
            fused = self._conditioned_embeddings(tokens, image_indices, conditioning, train=train)
            tokens, embeddings = fused.tokens, fused.embeddings
        elif image_indices is not None:
            raise ValueError("image_indices require conditioning payloads")
        return self.language_model.mtp_hidden_states(
            hidden, tokens, train=train, positions=positions, segment_ids=segment_ids,
            input_embeddings=embeddings, attention_mask=attention_mask,
            image_groups=image_groups, rotary_positions=rotary_positions)

    def mtp_logits(self, hidden, tokens, **kwargs):
        """Return the shared language head's logits for each media-aware prediction depth."""
        return [
            self.language_model._logits(state) for state in self.mtp_hidden_states(hidden, tokens, **kwargs)
        ]

    def mtp_step(self, hidden, tokens, *, image_indices=None, conditioning=None,
                 input_embeddings=None, **kwargs):
        """Run one candidate prediction step with the decoder's separate MTP cache.

        It returns the step's logits and hidden state. Pass either prepared
        `input_embeddings` or media `conditioning`, not both."""
        if conditioning is not None:
            if input_embeddings is not None:
                raise ValueError("MTP receives either prepared embeddings or media conditioning")
            fused = self._conditioned_embeddings(tokens, image_indices, conditioning)
            tokens, input_embeddings = fused.tokens, fused.embeddings
        elif image_indices is not None:
            raise ValueError("image_indices require conditioning payloads")
        return self.language_model.mtp_step(hidden, tokens, input_embeddings=input_embeddings, **kwargs)

    def token_embeddings(self, tokens):
        """Return the decoder's own embeddings of `tokens`; a sampled token is always text, never media."""
        return self.language_model.token_embeddings(tokens)

    def init_mtp_cache(self, batch_size: int):
        self.language_model.init_mtp_cache(batch_size)


    @nn.compact
    def hidden_states(self, tokens, train: bool = False, decode: bool = False,
                      positions=None, segment_ids=None, image_indices=None,
                      conditioning: Mapping[str, jax.Array] | None = None,
                      attention_mask=None, image_groups=None, rotary_positions=None,
                      audio_indices=None, routed_experts=None, routed=None):
        """Run the decoder over text with the media embeddings spliced in.

        `conditioning` holds the towers' payloads, and `image_indices` and
        `audio_indices` say which token positions each one replaces. A
        decode step tracks the next position in the `cache` collection,
        because a media span advances a row by more than one token.
        `routed_experts` and `routed` replay a routing record and go to the
        language model unchanged. Engines record a row for every placeholder
        position too, so the `[B, S, layers, top_k]` layout matches the text.
        """
        if self.family == "gemma4":
            placeholder = tokens == self.image_token_id
            for token_id in self.extra_placeholder_ids:
                placeholder = placeholder | (tokens == token_id)
            tokens = jnp.where(placeholder, self.pad_token_id, tokens)
        if decode:
            allocated = self.has_variable("cache", "next_position")
            next_position = self.variable("cache", "next_position", jnp.zeros, (tokens.shape[0],), jnp.int32)
            valid = jnp.ones(tokens.shape, bool) if attention_mask is None else attention_mask
            logical = rotary_positions if rotary_positions is not None else positions
            if logical is None:
                logical = next_position.value[:, None] + jnp.cumsum(valid, axis=1) - 1
                positions = logical
            mask = valid if logical.ndim == 2 else valid[..., None]
            maximum = jnp.max(jnp.where(mask, logical, -1), axis=tuple(range(1, logical.ndim)))
            if allocated:
                next_position.value = jnp.where(valid.any(axis=1), maximum + 1, next_position.value)
        if conditioning is None and self.is_initializing():
            self.conditioner.initialize_parameters()
            if self.audio is not None:
                self.audio_conditioner.initialize_parameters()
        if conditioning is None and (image_indices is not None or audio_indices is not None):
            raise ValueError("media indices require conditioning payloads")
        if conditioning is None and self.family != "gemma3n":
            return self.language_model.hidden_states(
                tokens, train=train, decode=decode, positions=positions, segment_ids=segment_ids,
                attention_mask=attention_mask, image_groups=image_groups, rotary_positions=rotary_positions,
                routed_experts=routed_experts, routed=routed)
        fused = self._conditioned_embeddings(tokens, image_indices, conditioning,
                                             train=train, audio_indices=audio_indices)
        return self.language_model.hidden_states(
            fused.tokens, train=train, decode=decode, positions=positions, segment_ids=segment_ids,
            input_embeddings=fused.embeddings, attention_mask=attention_mask,
            image_groups=image_groups, rotary_positions=rotary_positions, media_mask=fused.media,
            routed_experts=routed_experts, routed=routed)

    def __call__(self, tokens, train: bool = False, decode: bool = False,
                 positions=None, segment_ids=None, image_indices=None,
                 conditioning: Mapping[str, jax.Array] | None = None,
                 attention_mask=None, image_groups=None, rotary_positions=None,
                 audio_indices=None):
        hidden = self.hidden_states(tokens, train=train, decode=decode, positions=positions,
                                    segment_ids=segment_ids, image_indices=image_indices,
                                    conditioning=conditioning, attention_mask=attention_mask,
                                    image_groups=image_groups, rotary_positions=rotary_positions,
                                    audio_indices=audio_indices)
        if self.is_initializing() and self.language_model.dspark is not None:
            self.language_model.reach_drafter(tokens, hidden.dtype)
        if self.is_initializing() and self.num_nextn_predict_layers:
            self.mtp_hidden_states(hidden, tokens, train=train, positions=positions,
                                   segment_ids=segment_ids, image_indices=image_indices,
                                   conditioning=conditioning, attention_mask=attention_mask,
                                   image_groups=image_groups, rotary_positions=rotary_positions)
        return self.language_model._logits(hidden)

    def states_and_logits(self, tokens, **kwargs):
        """Return the final hidden states and their logits from one media-aware forward pass."""
        hidden = self.hidden_states(tokens, **kwargs)
        return hidden, self.language_model._logits(hidden)

    def states_and_logits_at(self, tokens, slots, **kwargs):
        """Return the final hidden states, and the logits of one slot per row.

        A prefill needs logits only at the position the first sample reads.
        Running the head over every prompt position would take most of a
        long request's transient memory, so this scores only `slots`.
        """
        hidden = self.hidden_states(tokens, **kwargs)
        return hidden, self.language_model._logits(hidden[jnp.arange(hidden.shape[0]), slots])

    def head_weight(self, params):
        """Return the decoder's shared fp32 head matrix, for an objective's chunked scoring."""
        return self.language_model.head_weight(params["language_model"])

    def vocabulary_bias(self, params):
        """The decoder's vocabulary bias, for the same affine head its forward scores."""
        return self.language_model.vocabulary_bias(params['language_model'])

    @nn.compact
    def init_cache(self, batch_size: int):
        """Allocate the language model's cache and the next-position counter, without running the towers."""
        self.variable("cache", "next_position", jnp.zeros, (batch_size,), jnp.int32)
        self.language_model.init_cache(batch_size)

    # What a task, a server and the trainer read off the model, answered by
    # its language model (`dew.nn.protocols`). A clone keeps the towers, and
    # every variable keeps its path.

    @nn.nowrap
    def with_cache_capacity(self, capacity: int) -> Self:
        """This model with its language model's cache at `capacity` slots
        (`CacheCapacity`): the wrapper's own `max_seq_len` is the language
        model's, and Qwen3.5-0.8B served at 384 slots held caches of 8192 when
        only the wrapper was asked."""
        return self.clone(language_model=self.language_model.with_cache_capacity(capacity))

    @nn.nowrap
    def recompute_record(self) -> JSON:
        """Its language model's rung (`Recomputing`); the towers keep their own remat."""
        return self.language_model.recompute_record()

    @nn.nowrap
    def recompute_more(self) -> Self | None:
        """This model with its language model one rung up (`Recomputing`), or None at its top."""
        language_model = self.language_model.recompute_more()
        return None if language_model is None else self.clone(language_model=language_model)

    @nn.nowrap
    def restore_recompute(self, record: JSON) -> Self:
        """This model with its language model at `record`'s rung where that is
        above its own (`Recomputing`)."""
        return self.clone(language_model=self.language_model.restore_recompute(record))

    @nn.nowrap
    def inference_projection_groups(self, variables: Mapping[str, Mapping]) -> tuple[ProjectionGroup, ...]:
        """Its language model's packed groups at their paths in this model's
        variables (`PackedProjections`); the towers run once per prefill and
        keep their layout."""
        text = {collection: tree["language_model"] for collection, tree in variables.items()
                if "language_model" in tree}
        return tuple(dataclasses.replace(group, path=("language_model", *group.path))
                     for group in self.language_model.inference_projection_groups(text))

    @property
    def cache_rebuild_position(self) -> int | None:
        """Its language model's (`CacheRebuilding`)."""
        return self.language_model.cache_rebuild_position

    @property
    def indexed_mixers(self) -> tuple[MLAMixer, ...]:
        """Its language model's (`Indexed`)."""
        return self.language_model.indexed_mixers

    @property
    def keeps_triton_gemm(self) -> bool:
        """Its language model's (`TritonGemm`)."""
        return self.language_model.keeps_triton_gemm


__all__ = ["MultimodalTransformer"]
