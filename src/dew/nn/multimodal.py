"""Image and audio conditioning around the shared causal decoder."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import TYPE_CHECKING, Protocol, Self, overload, runtime_checkable

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn, struct
from flax.typing import Dtype, PrecisionLike

from dew._model_types import GEMMA4_TYPES
from dew.batch import Coordinates, coordinates as build_coordinates
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.flops import MissingFlopsCount, counted_flops, parameter_flops
from dew.nn.inputs import CrossContext, next_position
from dew.nn.kv_cache import POSITION
from dew.nn.protocols import OutputTable, ProjectionGroup, held_groups
from dew.nn.sharding import explicit_spec
from dew.nn.vision import Gemma3nVision, ProjectorBase, TowerBase, TowerGeometry
from dew.nn.wrapping import DecoderWrapper

if TYPE_CHECKING:
    from dew.nn.cross_attention import CrossAttention
    from dew.nn.hyper_connections import HyperConnections
    from dew.nn.mla import MLAMixer


@runtime_checkable
class HardVocabularyEmbedder(Protocol):
    """A media embedder that also embeds a range of the text vocabulary, the
    hard tokens the decoder's own table does not hold, as Gemma 3n's vision
    and audio embedders do (modeling_gemma3n.py, Gemma3nMultimodalEmbedder)."""

    def embed_hard(self, ids: jax.Array) -> jax.Array: ...

    def merge_hard_embeddings(self, token_embeddings: jax.Array, ids: jax.Array) -> jax.Array: ...


@struct.dataclass
class Fusion:
    """Decoder token identities, their scaled, conditioned embeddings, and
    which positions media fill; both None where the decoder looks its tokens
    up itself."""

    tokens: jax.Array
    embeddings: jax.Array | None
    media: jax.Array | None


@overload
def _conditioner_modules(tower: TowerBase, projector: ProjectorBase, dtype, precision
                          ) -> tuple[nn.Module, ProjectorBase]: ...


@overload
def _conditioner_modules(tower: None, projector: ProjectorBase, dtype, precision
                          ) -> tuple[None, ProjectorBase]: ...


def _conditioner_modules(tower: TowerBase | None, projector: ProjectorBase, dtype, precision):
    """A media tower and native projector in the conditioner's compute policy."""
    return (None if tower is None else tower.build().clone(dtype=dtype, precision=precision),
            projector.clone(dtype=dtype, precision=precision))


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
        self.tower, self.projector = _conditioner_modules(self.vision, self.projection,
                                                         self.dtype, self.precision)

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


    def fuse(self, embeddings: jax.Array, image_indices: jax.Array, conditioning: Mapping[str, jax.Array],
             train: bool = False) -> jax.Array:
        """`embeddings` `[B, S, D]` with the slots `image_indices` marks replaced by their soft features."""
        if image_indices.shape != embeddings.shape[:-1]:
            raise ValueError("image_indices must align with the token rows")
        return _place(embeddings, self(conditioning, train=train), image_indices)


class AudioConditioner(nn.Module):
    """An audio encoder and projector over row-aligned clips.

    Features have shape [batch, clips, frames, mel] with a True-for-valid
    frame mask. The table returned holds every clip's encoded frames back to
    back, ``clips * capacity`` per row, where capacity is the encoded frame
    count, or ``soft_tokens`` when the source fixes the slot count per clip
    (Gemma 3n): there, padded frames and the slots past the encoded frames
    carry the embedder's padding token, as modeling_gemma3n.py does.
    """

    audio: TowerBase | None
    projection: ProjectorBase
    soft_tokens: int | None = None
    padding_id: int | None = None
    dtype: Dtype | None = None
    precision: PrecisionLike = None

    def setup(self):
        self.audio_tower, self.audio_projector = _conditioner_modules(
            self.audio, self.projection, self.dtype, self.precision)

    def __call__(self, conditioning: Mapping[str, jax.Array]) -> jax.Array:
        features = conditioning["input_features"]
        mask = conditioning["input_features_mask"]
        if features.ndim != 4 or mask.shape != features.shape[:3]:
            raise ValueError("input_features must be [B, A, T, F] with a [B, A, T] frame mask")
        batch, clips = features.shape[:2]
        flat = features.reshape(batch * clips, *features.shape[2:])
        valid = mask.reshape(batch * clips, features.shape[2])
        if self.audio_tower is not None:
            encoding = self.audio_tower(flat, valid)
            flat, valid = encoding.features, encoding.mask
        projected = self.audio_projector(flat)
        if self.soft_tokens is not None:
            if not isinstance(self.audio_projector, HardVocabularyEmbedder) or self.padding_id is None:
                raise ValueError("fixed audio slots require the Gemma 3n embedder and its padding token")
            padding = self.audio_projector.embed_hard(jnp.full((1, 1), self.padding_id, jnp.int32)).astype(
                projected.dtype
            )
            projected = jnp.where(valid[..., None], projected, padding)
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
        geometry = TowerGeometry() if self.audio is None else self.audio.geometry()
        # Without an audio tower, raw frames are the projector's input.
        mel = self.projection.input_width if self.audio is None else geometry.mel_features or 128
        if mel is None:
            raise ValueError("a raw-audio projector declares its input width")
        # Fixed slots take no more encoded frames than they hold: one token's frames a slot.
        frames = 16 if self.soft_tokens is None else (geometry.frames_per_token or 16) * self.soft_tokens
        self({"input_features": jnp.zeros((1, 1, frames, mel), self.dtype or jnp.float32),
              "input_features_mask": jnp.ones((1, 1, frames), bool)})


def _place(embeddings: jax.Array, table: jax.Array, indices: jax.Array) -> jax.Array:
    """Replace the marked slots of embeddings with the indexed table rows."""
    rows = (jnp.arange(embeddings.shape[0])[:, None], jnp.maximum(indices, 0))
    picked = table.at[rows].get(out_sharding=explicit_spec(indices, table.ndim - 2))
    return jnp.where((indices >= 0)[..., None], picked.astype(embeddings.dtype), embeddings)



class MultimodalTransformer(DecoderWrapper):
    """Conditions the shared decoder on images and audio, using the ordinary decoder cache and head.

    ``image_indices`` and ``audio_indices`` give, for each token slot, the
    index of the soft feature that replaces it, or -1 for a text token. The
    towers run during conditioned forward and prefill calls; later decode
    steps read the cached language states and do not run the encoders again.
    The parameters keep the existing `language_model`, tower and projector
    names, and audio adds `audio_tower` and `audio_projector`. Gemma 3n also
    embeds its hard vision and audio vocabulary ranges through the embedders,
    and keeps placeholder ids for its per-layer inputs, as modeling_gemma3n.py
    does. The serving and training hooks are its language model's (`DecoderWrapper`); a
    clone keeps the towers and every variable's path. So is its cross-attention
    (`CrossAttending`): a call's context entries (`dew.nn.inputs.CONTEXT`) go to the
    language model's layers, beside the media the towers read.
    """

    wraps = "language_model"

    language_model: CausalTransformer
    vision: TowerBase | None
    projection: ProjectorBase | None
    family: str
    image_token_id: int
    pad_token_id: int = 0
    extra_placeholder_ids: tuple[int, ...] = ()
    audio: TowerBase | None = None
    audio_projection: ProjectorBase | None = None
    audio_soft_tokens: int | None = None
    dtype: Dtype | None = None
    precision: PrecisionLike = None

    @nn.nowrap
    def model_flops(self, lengths) -> float:
        """Text-only decoder work. Media inputs require their own tower geometry."""
        if isinstance(self.audio_projection, HardVocabularyEmbedder):
            raise MissingFlopsCount("MultimodalTransformer needs its hard-audio-vocabulary matrix geometry")
        return counted_flops(self.language_model, lengths)

    @nn.nowrap
    def parameter_flops(self, lengths) -> float:
        return parameter_flops(self.language_model, lengths)

    @property
    def cross_attention(self) -> CrossAttention | None:
        return self.language_model.cross_attention

    @property
    def context_width(self) -> int:
        return self.language_model.context_width

    @property
    def context_dtype(self) -> jnp.dtype:
        return self.language_model.context_dtype

    @nn.nowrap
    def cross_flops(self, lengths, context_lengths) -> float:
        return self.language_model.cross_flops(lengths, context_lengths)

    @nn.nowrap
    def with_cross_attention(self, cross: CrossAttention | None) -> Self:
        return self._over(self.language_model.with_cross_attention(cross))

    @nn.nowrap
    def logits_flops(self, lengths) -> float:
        return self.language_model.logits_flops(lengths)

    @nn.nowrap
    def matrix_usage(self, path: tuple[str, ...], lengths, *, logits: bool = False):
        """The same native language scope the wrapper actually invokes without media."""
        relative = path[1:] if path and path[0] == 'params' else path
        if relative and relative[0] == 'language_model':
            return self.language_model.matrix_usage(relative[1:], lengths, logits=logits)
        if relative and relative[0] in ('conditioner', 'audio_conditioner'):
            return np.zeros(np.shape(lengths), np.int64)
        raise MissingFlopsCount(f"MultimodalTransformer has no matrix usage for {'/'.join(relative)}")

    def setup(self):
        if (self.vision is None) != (self.projection is None):
            raise ValueError("a vision embedding and its projector arrive together")
        if self.vision is not None and self.projection is not None:
            self.conditioner = VisionConditioner(
                self.vision, self.projection,
                dtype=self.dtype, precision=self.precision)
            nn.share_scope(self.conditioner, self)
        if self.audio is not None and self.audio_projection is None:
            raise ValueError("an audio tower requires its projector")
        if self.audio_projection is not None:
            self.audio_conditioner = AudioConditioner(
                self.audio, self.audio_projection, self.audio_soft_tokens,
                self.vocab_size - 1 if self.family == "gemma3n" else None,
                dtype=self.dtype, precision=self.precision)
            nn.share_scope(self.audio_conditioner, self)

    @property
    def num_nextn_predict_layers(self) -> int:
        return self.language_model.num_nextn_predict_layers

    def _fused(self, tokens, image_indices=None, audio_indices=None, conditioning=None,
               train: bool = False) -> Fusion:
        """Decoder token identities and their embeddings with media slots filled,
        which the main forward and every prediction depth read alike.

        Gemma 4's placeholder ids read as padding. Marked slots read the
        towers; the rest read the embedding table. Gemma 3n keeps placeholder
        ids for its per-layer inputs and embeds its hard media vocabulary
        ranges, so it runs this path with no payloads too; any other call
        without payloads leaves the decoder its own lookup.
        """
        if self.family in GEMMA4_TYPES:
            placeholder = tokens == self.image_token_id
            for token_id in self.extra_placeholder_ids:
                placeholder = placeholder | (tokens == token_id)
            tokens = jnp.where(placeholder, self.pad_token_id, tokens)
        if conditioning is None and (image_indices is not None or audio_indices is not None):
            raise ValueError("media indices require conditioning payloads")
        if conditioning is None and self.family != "gemma3n":
            return Fusion(tokens, None, None)
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
                if not isinstance(embedder, HardVocabularyEmbedder):
                    raise TypeError("Gemma 3n media embedders carry the hard vocabulary")
                embeddings = embedder.merge_hard_embeddings(embeddings, tokens)
        if conditioning is not None and image_indices is not None:
            if self.vision is None:
                raise ValueError("image_indices require a vision embedding")
            embeddings = self.conditioner.fuse(embeddings, image_indices, conditioning, train=train)
        if conditioning is not None and audio_indices is not None:
            if self.audio_projection is None:
                raise ValueError("audio_indices require an audio projector")
            embeddings = _place(embeddings, self.audio_conditioner(conditioning), audio_indices)
        return Fusion(decoder_tokens, embeddings, media)

    def mtp_hidden_states(self, hidden, tokens, train: bool = False, positions=None,
                          segment_ids=None, image_indices=None, conditioning=None,
                          attention_mask=None, image_groups=None, block_groups=None, rotary_positions=None,
                          audio_indices=None, coordinates: Coordinates | None = None):
        """Run the multi-token prediction layers over the same media embeddings and
        token fields as the main decoder (`hidden_states`)."""
        _, media = CrossContext.parted(conditioning)
        fused = self._fused(tokens, image_indices, audio_indices, media, train=train)
        return self.language_model.mtp_hidden_states(
            hidden, fused.tokens, train=train, positions=positions, segment_ids=segment_ids,
            input_embeddings=fused.embeddings, attention_mask=attention_mask,
            image_groups=image_groups, block_groups=block_groups, rotary_positions=rotary_positions,
            coordinates=coordinates)

    def mtp_logits(self, hidden, tokens, **kwargs):
        """Return the shared language head's logits for each media-aware prediction depth."""
        return [self.language_model.logits_from_hidden(state)
                for state in self.mtp_hidden_states(hidden, tokens, **kwargs)]

    def mtp_step(self, hidden, tokens, *, image_indices=None, audio_indices=None, conditioning=None,
                 input_embeddings=None, **kwargs):
        """Run one candidate prediction step with the decoder's separate MTP cache.

        It returns the step's logits and hidden state. Pass either prepared
        `input_embeddings` or media `conditioning`, not both."""
        if conditioning is not None and input_embeddings is not None:
            raise ValueError("MTP receives either prepared embeddings or media conditioning")
        fused = self._fused(tokens, image_indices, audio_indices, conditioning)
        prepared = fused.embeddings if input_embeddings is None else input_embeddings
        return self.language_model.mtp_step(hidden, fused.tokens, input_embeddings=prepared, **kwargs)

    def token_embeddings(self, tokens):
        """Return the decoder's own embeddings of `tokens`; a sampled token is always text, never media."""
        return self.language_model.token_embeddings(tokens)

    def init_mtp_cache(self, batch_size: int):
        self.language_model.init_mtp_cache(batch_size)


    @nn.compact
    def hidden_states(self, tokens, train: bool = False, decode: bool = False,
                      positions=None, segment_ids=None, image_indices=None,
                      conditioning: Mapping[str, jax.Array] | None = None,
                      attention_mask=None, image_groups=None, block_groups=None, rotary_positions=None,
                      audio_indices=None, routed_experts=None, routed=None, right_padded: bool = False,
                      coordinates: Coordinates | None = None):
        """Run the decoder over text with the media embeddings spliced in.

        `conditioning` holds the towers' payloads, and `image_indices` and
        `audio_indices` say which token positions each one replaces. A
        decode step tracks the next position in the `cache` collection,
        because a media span advances a row by more than one token.
        `routed_experts` and `routed` replay a routing record and go to the
        language model unchanged. Engines record a row for every placeholder
        position too, so the `[B, S, layers, top_k]` layout matches the text.
        `right_padded` and `coordinates` are the language model's
        (`CausalTransformer.hidden_and_mtp_inputs`).
        """
        # The tokens as the decoder reads them (`Coordinates`), whichever fields the caller gave
        # them in, so the cursor counts the places the layers do.
        if coordinates is None and (segment_ids is not None or attention_mask is not None
                                    or positions is not None or right_padded):
            coordinates = build_coordinates(tokens.shape, segments=segment_ids, valid=attention_mask,
                                            positions=positions, right_padded=right_padded)
        if decode:
            allocated = self.has_variable("cache", POSITION)
            cursor = self.variable("cache", POSITION, jnp.zeros, (tokens.shape[0],), jnp.int32)
            valid = (jnp.ones(tokens.shape, bool) if coordinates is None or coordinates.valid is None
                     else coordinates.valid)
            positions = None if coordinates is None else coordinates.positions
            if rotary_positions is None and positions is None:
                positions = cursor.value[:, None] + jnp.cumsum(valid, axis=1) - 1
                coordinates = (build_coordinates(tokens.shape, positions=positions) if coordinates is None
                               else dataclasses.replace(coordinates, positions=positions))
            # The response's rule (`next_position`): past the last real token's position, or past
            # the largest coordinate of a multi-axis one.
            fields = {name: value for name, value in (("rotary_positions", rotary_positions),
                                                      ("positions", positions)) if value is not None}
            following = next_position(fields, valid)
            assert following is not None
            if allocated:
                cursor.value = jnp.where(valid.any(axis=1), following, cursor.value)
        # The towers read the media; the language model reads the context, and checks it.
        context, media = CrossContext.parted(conditioning)
        if media is None and self.is_initializing():
            if self.vision is not None:
                self.conditioner.initialize_parameters()
            if self.audio_projection is not None:
                self.audio_conditioner.initialize_parameters()
        fused = self._fused(tokens, image_indices, audio_indices, media, train=train)
        return self.language_model.hidden_states(
            fused.tokens, train=train, decode=decode, input_embeddings=fused.embeddings,
            image_groups=image_groups, block_groups=block_groups, rotary_positions=rotary_positions,
            media_mask=fused.media,
            routed_experts=routed_experts, routed=routed, coordinates=coordinates, conditioning=context)

    def __call__(self, tokens, train: bool = False, decode: bool = False,
                 positions=None, segment_ids=None, image_indices=None,
                 conditioning: Mapping[str, jax.Array] | None = None,
                 attention_mask=None, image_groups=None, block_groups=None, rotary_positions=None,
                 audio_indices=None, coordinates: Coordinates | None = None):
        hidden = self.hidden_states(tokens, train=train, decode=decode, positions=positions,
                                    segment_ids=segment_ids, image_indices=image_indices,
                                    conditioning=conditioning, attention_mask=attention_mask,
                                    image_groups=image_groups, block_groups=block_groups,
                                    rotary_positions=rotary_positions, audio_indices=audio_indices,
                                    coordinates=coordinates)
        if self.is_initializing() and self.language_model.dspark is not None:
            self.language_model.reach_drafter(tokens, hidden.dtype)
        if self.is_initializing() and self.num_nextn_predict_layers:
            self.mtp_hidden_states(hidden, tokens, train=train, positions=positions,
                                   segment_ids=segment_ids, image_indices=image_indices,
                                   conditioning=conditioning, attention_mask=attention_mask,
                                   image_groups=image_groups, block_groups=block_groups,
                                   rotary_positions=rotary_positions, audio_indices=audio_indices,
                                   coordinates=coordinates)
        return self.language_model.logits_from_hidden(hidden)

    def logits(self, tokens, *, train: bool = False, **fields):
        """Return the `[B, S, vocab]` fp32 logits of the whole sequence, media
        fused as `hidden_states` fuses them from `fields`. Without `decode`
        the call writes no cache."""
        return self.language_model.logits_from_hidden(self.hidden_states(tokens, train=train, **fields))

    def logits_from_hidden(self, hidden):
        """Return the decoder's logits of final states `hidden`."""
        return self.language_model.logits_from_hidden(hidden)

    def states_and_logits(self, tokens, **kwargs):
        """Return the final hidden states and their logits from one media-aware forward pass."""
        hidden = self.hidden_states(tokens, **kwargs)
        return hidden, self.language_model.logits_from_hidden(hidden)

    def states_and_logits_at(self, tokens, slots, **kwargs):
        """Return the final hidden states, and the logits of one slot per row.

        A prefill needs logits only at the position the first sample reads.
        Running the head over every prompt position would take most of a
        long request's transient memory, so this scores only `slots`.
        """
        hidden = self.hidden_states(tokens, **kwargs)
        return hidden, self.language_model.logits_from_hidden(hidden[jnp.arange(hidden.shape[0]), slots])

    def output_table(self) -> OutputTable | None:
        """Return the decoder's head as the matrix its final states contract,
        or None where none does (`CausalTransformer.output_table`)."""
        return self.language_model.output_table()

    def vocabulary_rows(self) -> jax.Array:
        """The decoder's head rows (`CausalTransformer.vocabulary_rows`)."""
        return self.language_model.vocabulary_rows()

    @nn.compact
    def init_cache(self, batch_size: int):
        """Allocate the language model's cache and the next-position counter, without running the towers."""
        self.variable("cache", POSITION, jnp.zeros, (batch_size,), jnp.int32)
        self.language_model.init_cache(batch_size)

    @nn.nowrap
    def with_passes(self, steps: int | None = None, layers: tuple[int, int] | None = None, *,
                    backprop_steps: int | None = None, remat: bool | None = None) -> Self | None:
        language_model = self.language_model.with_passes(steps, layers, backprop_steps=backprop_steps,
                                                         remat=remat)
        return None if language_model is None else self._over(language_model)

    @nn.nowrap
    def inference_projection_groups(self, variables: Mapping[str, Mapping]) -> tuple[ProjectionGroup, ...]:
        """The language model's (`held_groups`); the towers run once per prefill and keep their layout."""
        return held_groups(self.language_model, "language_model", variables)

    @property
    def cache_rebuild_position(self) -> int | None:
        return self.language_model.cache_rebuild_position

    def mixed_admission_refusal(self) -> str | None:
        return "media fuse into a prefill's embeddings, which a mixed call does not carry"

    def split_prefill_refusal(self) -> str | None:
        return self.language_model.split_prefill_refusal()

    @property
    def mtp_hyper_connections(self) -> HyperConnections | None:
        return self.language_model.mtp_hyper_connections

    @property
    def indexed_mixers(self) -> tuple[MLAMixer, ...]:
        return self.language_model.indexed_mixers


__all__ = ["MultimodalTransformer"]
