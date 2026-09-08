"""Native vision conditioning around the shared causal decoder."""

from __future__ import annotations

import math
from collections.abc import Mapping

import jax
import jax.numpy as jnp
from flax import linen as nn, struct
from flax.typing import Dtype, PrecisionLike

from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.vision import Gemma3nProjectorModule, ProjectorBase, TowerBase
from dew.registry import models


@struct.dataclass
class Fusion:
    """Decoder token identities and their scaled, conditioned embeddings."""

    tokens: jax.Array
    embeddings: jax.Array


class VisionConditioner(nn.Module):
    """An image encoder and projector with row-aligned media inputs.

    Pixels have shape [batch, images, channels, height, width]. All images
    in a numeric batch share their processed resolution; the processor keeps
    per-image lengths and does not substitute preprocessing inside the model.
    """

    family: str
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
        elif positions is None:
            features = self.tower(flat)
        else:
            features = self.tower(flat, pixel_position_ids=positions.reshape(batch * images, *positions.shape[2:]))
        projected = self.projector(features)
        return projected.reshape(batch, images * projected.shape[1], projected.shape[-1])

    def initialize_parameters(self) -> None:
        """Create the media leaves during an ordinary token-only model init.

        Fixed-resolution towers use their configured image size. Other towers
        need only one pooling/merge block to create resolution-independent
        parameters; real batches supply their processed geometry later.
        """
        side = getattr(self.vision, "image_size", None)
        if side is None:
            patch = getattr(self.vision, "patch_size", 16)
            block = getattr(self.vision, "pooling_kernel_size", getattr(self.vision, "spatial_merge_size", 2))
            side = patch * block
        channels = getattr(self.vision, "num_channels", getattr(self.vision, "in_channels", getattr(self.vision, "in_chans", 3)))
        self({"pixel_values": jnp.zeros((1, 1, channels, side, side), self.dtype or jnp.float32)})


    def fuse(self, tokens: jax.Array, embeddings: jax.Array,
             image_indices: jax.Array, conditioning: Mapping[str, jax.Array],
             train: bool = False) -> Fusion:
        """Replace marked text slots with the corresponding soft feature."""
        if image_indices.shape != tokens.shape:
            raise ValueError("image_indices must align with the token rows")
        return Fusion(tokens, _place(embeddings, self(conditioning, train=train), image_indices))


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
            padding = self.audio_projector.embed_hard(jnp.full((1, 1), self.padding_id, jnp.int32)).astype(projected.dtype)
            projected = jnp.where(encoding.mask[..., None], projected, padding)
            missing = self.soft_tokens - projected.shape[1]
            if missing < 0:
                raise ValueError(f"{projected.shape[1]} encoded audio frames exceed the {self.soft_tokens} slots per clip")
            projected = jnp.concatenate(
                [projected, jnp.broadcast_to(padding, (projected.shape[0], missing, projected.shape[-1]))], axis=1)
        return projected.reshape(batch, clips * projected.shape[1], projected.shape[-1])

    def initialize_parameters(self) -> None:
        """Create the audio leaves during a token-only model init."""
        mel = getattr(self.audio, "input_feat_size", getattr(self.audio, "subsampling_conv_channels", (128,))[0])
        frames = 16 if self.soft_tokens is None else 16 * self.soft_tokens
        self({"input_features": jnp.zeros((1, 1, frames, mel), self.dtype or jnp.float32),
              "input_features_mask": jnp.ones((1, 1, frames), bool)})


def _place(embeddings: jax.Array, table: jax.Array, indices: jax.Array) -> jax.Array:
    """Replace the marked slots of embeddings with the indexed table rows."""
    picked = table[jnp.arange(embeddings.shape[0])[:, None], jnp.maximum(indices, 0)]
    return jnp.where((indices >= 0)[..., None], picked.astype(embeddings.dtype), embeddings)



@models("multimodal_transformer")
class MultimodalTransformer(nn.Module):
    """A media-conditioned decoder using the ordinary decoder cache and head.

    ``image_indices`` and ``audio_indices`` identify the soft feature for each
    text slot, or -1 for text. Media are evaluated during conditioned
    forward/prefill calls; subsequent decoding reads the cached language
    states without rerunning the encoders. Parameters retain the existing
    language_model, tower and projector names; audio adds audio_tower and
    audio_projector. Gemma 3n also embeds its hard vision and audio vocabulary
    ranges through the embedders and keeps placeholder ids for its per-layer
    inputs, as modeling_gemma3n.py does.
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
    attention_impl: str | None = None

    def setup(self):
        self.conditioner = VisionConditioner(
            self.family, self.vision, self.projection,
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
            decoder_tokens = jnp.where((tokens >= 0) & (tokens < self.language_model.per_layer_vocab), tokens, 0)
        else:
            decoder_tokens = jnp.where(media, 0, tokens)
        embeddings = self.language_model.embed_tokens(decoder_tokens)
        if self.language_model.embedding_scale:
            embeddings = (embeddings * jnp.asarray(
                math.sqrt(self.emb_features), self.language_model.embed_tokens.embedding.dtype)).astype(embeddings.dtype)
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
        return Fusion(decoder_tokens, embeddings)

    def mtp_hidden_states(self, hidden, tokens, train: bool = False, positions=None,
                          segment_ids=None, image_indices=None, conditioning=None,
                          attention_mask=None, image_groups=None, rotary_positions=None):
        """Prediction layers over the same media embeddings as the main decoder."""
        embeddings = slots = None
        if conditioning is not None:
            fused = self._conditioned_embeddings(tokens, image_indices, conditioning, train=train)
            tokens, embeddings = fused.tokens, fused.embeddings
            slots = jnp.broadcast_to(jnp.arange(tokens.shape[1]), tokens.shape)
        elif image_indices is not None:
            raise ValueError("image_indices require conditioning payloads")
        return self.language_model.mtp_hidden_states(
            hidden, tokens, train=train, positions=positions, segment_ids=segment_ids,
            input_embeddings=embeddings, embedding_positions=slots, attention_mask=attention_mask,
            image_groups=image_groups, rotary_positions=rotary_positions)

    def mtp_logits(self, hidden, tokens, **kwargs):
        """The shared language head over each media-aware prediction depth."""
        return [self.language_model._logits(state) for state in self.mtp_hidden_states(hidden, tokens, **kwargs)]

    def mtp_step(self, hidden, tokens, *, image_indices=None, conditioning=None,
                 input_embeddings=None, **kwargs):
        """One candidate prediction step using the decoder's independent MTP cache."""
        if conditioning is not None:
            if input_embeddings is not None:
                raise ValueError("MTP receives either prepared embeddings or media conditioning")
            fused = self._conditioned_embeddings(tokens, image_indices, conditioning)
            tokens, input_embeddings = fused.tokens, fused.embeddings
        elif image_indices is not None:
            raise ValueError("image_indices require conditioning payloads")
        return self.language_model.mtp_step(hidden, tokens, input_embeddings=input_embeddings, **kwargs)

    def init_mtp_cache(self, batch_size: int):
        self.language_model.init_mtp_cache(batch_size)


    @nn.compact
    def hidden_states(self, tokens, train: bool = False, decode: bool = False,
                      positions=None, segment_ids=None, image_indices=None,
                      conditioning: Mapping[str, jax.Array] | None = None,
                      attention_mask=None, image_groups=None, rotary_positions=None,
                      audio_indices=None):
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
        if conditioning is not None and image_indices is None and audio_indices is None:
            raise ValueError("conditioned inputs require image_indices or audio_indices")
        if conditioning is None and self.family != "gemma3n":
            return self.language_model.hidden_states(
                tokens, train=train, decode=decode, positions=positions, segment_ids=segment_ids,
                attention_mask=attention_mask, image_groups=image_groups, rotary_positions=rotary_positions)
        fused = self._conditioned_embeddings(tokens, image_indices, conditioning,
                                             train=train, audio_indices=audio_indices)
        decoder_tokens, embeddings = fused.tokens, fused.embeddings
        slots = jnp.broadcast_to(jnp.arange(tokens.shape[1]), tokens.shape)
        return self.language_model.hidden_states(
            decoder_tokens, train=train, decode=decode, positions=positions, segment_ids=segment_ids,
            input_embeddings=embeddings, embedding_positions=slots, attention_mask=attention_mask,
            image_groups=image_groups, rotary_positions=rotary_positions)

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
        if self.is_initializing() and self.num_nextn_predict_layers:
            self.mtp_hidden_states(hidden, tokens, train=train, positions=positions,
                                   segment_ids=segment_ids, image_indices=image_indices,
                                   conditioning=conditioning, attention_mask=attention_mask,
                                   image_groups=image_groups, rotary_positions=rotary_positions)
        return self.language_model._logits(hidden)

    def head_weight(self, params):
        """The decoder's shared fp32 head matrix for chunked objective scoring."""
        return self.language_model.head_weight(params["language_model"])

    @nn.compact
    def init_cache(self, batch_size: int):
        """Allocate the nested language cache without evaluating media."""
        self.variable("cache", "next_position", jnp.zeros, (batch_size,), jnp.int32)
        self.language_model.init_cache(batch_size)
