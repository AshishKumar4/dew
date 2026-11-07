"""MiMo-V2.6's audio path: the tokenizer's encoder from mel frames to residual codes, and the audio
encoder from codes to the decoder's embeddings (XiaomiMiMo/MiMo-V2.6-Pro-MOPD's `modeling_mimo_v2.py`:
AudioTokenizerEncoder, MiMoAudioEncoder).

Both are Dew's decoder blocks. The tokenizer's are Whisper-style (layer norms, a biased GELU
feed-forward) with a rotary, causal attention, every `swa_per_block`-th layer of each block of them
full and the others a window of keys back; the encoder's are Qwen 2's (RMS norms, SwiGLU), bidirectional
over each group of `group_size` code frames. The codes come from `dew.nn.quantizers`.
"""

import dataclasses
import functools
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.typing import Dtype, PrecisionLike

from dew.batch import coordinates
from dew.nn.conv import Conv
from dew.nn.dense import Dense
from dew.nn.precision import float32_dot_general

from .activations import gelu_exact
from .audio import AudioEncoding
from .backbones.decoder_block import BlockWiring, DecoderBlock
from .blocks import TokenEmbedding
from .inputs import AttentionMetadata, sequence_blocks
from .mixer_base import MixerContext
from .mixers.attention import CausalSelfAttention
from .mlp import GatedMLP
from .norms import LayerNorm, RMSNorm
from .quantizers import residual_quantize
from .vision import TowerBase, TowerGeometry

if TYPE_CHECKING:
    from dew.data.processors import LogMel


@dataclasses.dataclass(frozen=True)
class MiMoAudioTokenizer:
    """The audio tokenizer's encoder geometry and its mel features, under its config's names."""

    n_mels: int = 128
    sampling_rate: int = 24000
    nfft: int = 960
    hop_length: int = 240
    window_size: int = 960
    fmin: float = 0.0
    fmax: float | None = None
    d_model: int = 1024
    kernel_size: int = 3
    stride_size: int = 2
    avg_pooler: int = 2
    encoder_layers: int = 24
    encoder_attention_heads: int = 16
    encoder_ffn_dim: int = 4096
    encoder_causal: bool = True
    encoder_attn_window_size: tuple[int, int] = (128, 0)
    encoder_skip_layer_id: int | None = 3
    hybrid_attention: bool = True
    swa_per_block: int = 2
    rope_theta: float = 10000.0
    codebook_size: tuple[int, ...] = (1024, 1024, 256) + (128,) * 17

    def mel(self) -> "LogMel":
        """The log mel frames the encoder reads: torchaudio's MelSpectrogram at the tokenizer's rate,
        window and hop with magnitudes, floored at 1e-7 (MiMo-Audio-Tokenizer's `mel_spectrogram`)."""
        from dew.data.processors import LogMel

        return LogMel(self.sampling_rate, self.nfft, self.hop_length, self.window_size, self.n_mels,
                      self.fmin, self.fmax)

    def windowed(self, index: int) -> bool:
        """Whether layer `index` keeps a window of keys (AudioTokenizerEncoder's attn_window_sizes)."""
        return self.encoder_attn_window_size[0] > 0 and (
            not self.hybrid_attention or index % self.swa_per_block < self.swa_per_block - 1)


@dataclasses.dataclass(frozen=True)
class MiMoAudio(TowerBase):
    """MiMo-V2.6's audio tower: mel frames through the tokenizer's encoder to codes, in segments of
    `audio_segment_size` frames, then the codes through the audio encoder; audio_config's names."""

    tokenizer: MiMoAudioTokenizer = MiMoAudioTokenizer()
    audio_channels: int = 20
    group_size: int = 4
    input_local_dim: int = 1024
    input_local_layers: int = 6
    input_local_attn_heads: int = 16
    input_local_intermediate_size: int = 4096
    input_full_attention: bool = True
    rope_theta: float = 640000.0
    rms_norm_eps: float = 1e-6
    add_post_norm: bool = True
    projection_layers: int = 2
    out_hidden_size: int = 6144
    speech_vocab_size: tuple[int, ...] = (1280,) * 20
    audio_segment_size: int = 6000
    tokenizer_batch_frames: int = 256000
    """The mel frames the reference tokenizes in one batch (tokenize_audio_batch's `_at_group_by_length`)."""

    def build(self) -> nn.Module:
        return MiMoAudioTower(self)

    def geometry(self) -> TowerGeometry:
        return TowerGeometry(mel_features=self.tokenizer.n_mels)

    def encoded(self, mask: np.ndarray) -> np.ndarray:
        """Which of each clip's encoded tokens `MiMoAudioTower` gives hold one, `[clips, capacity]`, from
        its valid frames `mask` `[clips, frames]`: the frames cut into segments of `audio_segment_size`,
        each convolved and pooled to its codes (`AudioTokenizerEncoder`), a clip's codes in groups of
        `group_size`, as many places as a clip of every frame valid fills."""
        size = self.audio_segment_size
        whole = tower_counts(np.asarray(size), self.tokenizer)[1]
        valid = np.sum(mask, axis=1)
        codes = valid // size * whole + tower_counts(valid % size, self.tokenizer)[1]
        groups = grouped(codes, self.group_size)
        capacity = grouped(-(-mask.shape[1] // size) * whole, self.group_size)
        return np.arange(capacity)[None, :] < groups[:, None]


class AudioTokenizerEncoder(nn.Module):
    """The tokenizer's encoder up to its codes (AudioTokenizerEncoder.get_features, then the
    quantizer). Each row is one sequence of `lengths` mel frames: two GELU convolutions, the second
    halving time; the blocks, the output of layer `encoder_skip_layer_id - 1` added back after the
    last; a layer norm; then a bias-free convolution pooling `avg_pooler` frames, GELU and a layer norm.
    Returns the features `[B, frames, d_model]` the quantizer reads and which frames hold one.

    The reference runs its segments in batches and pads each to the batch's longest, which reaches
    the frames at a row's end: `batch_lengths` is the longest of each row's batch, every row's own
    batch where None (`MiMoAudioTower` groups them as tokenize_audio_batch does)."""

    config: MiMoAudioTokenizer
    dtype: Dtype | None = None
    precision: PrecisionLike = None

    @nn.compact
    def __call__(self, mels, lengths, batch_lengths=None):
        cfg = self.config
        batch_lengths = jnp.max(lengths) if batch_lengths is None else batch_lengths
        conv = functools.partial(Conv, cfg.d_model, dtype=self.dtype, precision=self.precision)
        place = jnp.arange(mels.shape[1])[None, :]
        x = jnp.where((place < lengths[:, None])[..., None], mels, 0)
        x = gelu_exact(conv((cfg.kernel_size,), padding=((1, 1),), name="conv1")(x))
        # The reference's first convolution ends where its batch does; the second reads zeros past it.
        ends = jnp.broadcast_to(batch_lengths + 3 - cfg.kernel_size, lengths.shape)
        x = jnp.where((jnp.arange(x.shape[1])[None, :] < ends[:, None])[..., None], x, 0)
        x = gelu_exact(conv((cfg.kernel_size,), strides=(cfg.stride_size,), padding=((1, 1),),
                            name="conv2")(x))

        (counts, codes), widest = (tower_counts(lengths, cfg),
                                   tower_counts(jnp.broadcast_to(batch_lengths, lengths.shape), cfg)[0])
        valid = jnp.arange(x.shape[1])[None, :] < counts[:, None]
        heads = cfg.encoder_attention_heads
        skip = None
        for index in range(cfg.encoder_layers):
            attention = functools.partial(
                CausalSelfAttention, ctx=MixerContext(
                    emb_features=cfg.d_model, num_heads=heads, num_kv_heads=heads,
                    head_dim=cfg.d_model // heads, max_seq_len=1, causal=cfg.encoder_causal, qk_norm=False,
                    attention_bias=True, rope_theta=cfg.rope_theta,
                    # The reference masks keys past `window` places back: |q - k| < window + 1
                    # (`window_sides`).
                    sliding_window=cfg.encoder_attn_window_size[0] + 1 if cfg.windowed(index) else None,
                    bidirectional_window=not cfg.encoder_causal, dtype=self.dtype, precision=self.precision))
            mlp = functools.partial(GatedMLP, hidden_features=cfg.encoder_ffn_dim, out_features=cfg.d_model,
                                    activation="gelu_exact", use_bias=True, dtype=self.dtype,
                                    precision=self.precision)
            block = DecoderBlock(attention, mlp, cfg.d_model, BlockWiring(), norm_eps=1e-5, norm_type="layer",
                                 norm_bias=True, dtype=self.dtype, precision=self.precision,
                                 name=f"layers_{index}")
            # The padding runs as a document of its own; each frame keeps its index as its position.
            places = jnp.broadcast_to(jnp.arange(valid.shape[1]), valid.shape)
            x = block(x, attention_metadata=AttentionMetadata(
                coordinates=coordinates(segments=valid.astype(jnp.int32), positions=places)))
            if cfg.encoder_skip_layer_id is not None and index == cfg.encoder_skip_layer_id - 1:
                skip = x
        if skip is not None:
            x = x + skip
        x = LayerNorm(epsilon=1e-5, dtype=self.dtype, name="layer_norm")(x)
        pool = cfg.avg_pooler
        if pool > 1:
            # The reference pools its batch as it unpacks it: past a row's frames, up to the batch's
            # longest, each place repeats the row's last frame, and past the longest it is zero.
            last = jnp.take_along_axis(x, jnp.maximum(counts - 1, 0)[:, None, None], axis=1)
            spot = jnp.arange(x.shape[1])[None, :, None]
            x = jnp.where(spot < counts[:, None, None], x, jnp.where(spot < widest[:, None, None], last, 0))
            x = jnp.pad(x, ((0, 0), (0, -x.shape[1] % pool), (0, 0)))
            x = gelu_exact(conv((pool,), strides=(pool,), padding="VALID", use_bias=False,
                                name="down_sample_layer")(x))
            x = LayerNorm(epsilon=1e-5, dtype=self.dtype, name="down_sample_norm")(x)
        return x, jnp.arange(x.shape[1])[None, :] < codes[:, None]


class ResidualQuantizer(nn.Module):
    """The tokenizer's frozen codebooks, `codebook_size` codewords each, and the codes of features
    under them (`dew.nn.quantizers.residual_quantize`)."""

    sizes: tuple[int, ...]
    width: int

    @nn.compact
    def __call__(self, features) -> jax.Array:
        return residual_quantize(features, [
            self.param(f"codebook_{stage}", nn.initializers.normal(1.0), (size, self.width), jnp.float32)
            for stage, size in enumerate(self.sizes)])


class MiMoAudioCodeEncoder(nn.Module):
    """The audio encoder over codes (MiMoAudioEncoder._process_audio_codes): each code frame the sum
    of its stages' embeddings, `group_size` frames a group, a row's last group filled with its last
    frame; the groups through Qwen 2 blocks, bidirectional where `input_full_attention`, its final
    norm where `add_post_norm`; then each group's frames side by side through the projection.
    Returns `[B, groups, out_hidden_size]`."""

    config: MiMoAudio
    dtype: Dtype | None = None
    precision: PrecisionLike = None

    @nn.compact
    def __call__(self, codes, counts):
        cfg = self.config
        batch, frames, _ = codes.shape
        groups = grouped(frames, cfg.group_size)
        codes = jnp.pad(codes, ((0, 0), (0, groups * cfg.group_size - frames), (0, 0)))
        last = jnp.take_along_axis(codes, jnp.maximum(counts - 1, 0)[:, None, None], axis=1)
        place = jnp.arange(codes.shape[1])[None, :, None]
        codes = jnp.where(place < counts[:, None, None], codes, last)
        x = functools.reduce(jnp.add, [
            TokenEmbedding(size, cfg.input_local_dim, dtype=self.dtype, name=f"speech_embeddings_{channel}")(
                codes[..., channel])
            for channel, size in enumerate(cfg.speech_vocab_size[:cfg.audio_channels])])
        x = x.reshape(batch * groups, cfg.group_size, cfg.input_local_dim)
        heads = cfg.input_local_attn_heads
        attention = functools.partial(
            CausalSelfAttention, ctx=MixerContext(
                emb_features=cfg.input_local_dim, num_heads=heads, num_kv_heads=heads,
                head_dim=cfg.input_local_dim // heads, max_seq_len=cfg.group_size,
                causal=not cfg.input_full_attention, qk_norm=False, attention_bias=True, o_proj_bias=False,
                rope_theta=cfg.rope_theta, dtype=self.dtype, precision=self.precision))
        mlp = functools.partial(GatedMLP, hidden_features=cfg.input_local_intermediate_size,
                                out_features=cfg.input_local_dim, dtype=self.dtype, precision=self.precision)
        # Qwen2RMSNorm casts the normalized activations back before its weight
        # multiplies them (modeling_qwen2.py:247-252), at every block norm and the final one.
        for index in range(cfg.input_local_layers):
            x = DecoderBlock(attention, mlp, cfg.input_local_dim, BlockWiring(), norm_eps=cfg.rms_norm_eps,
                             scale_after_cast=True, dtype=self.dtype, precision=self.precision,
                             name=f"layers_{index}")(x)
        if cfg.add_post_norm:
            x = RMSNorm(epsilon=cfg.rms_norm_eps, scale_after_cast=True, dtype=self.dtype, name="norm")(x)
        x = x.reshape(batch, groups, cfg.group_size * cfg.input_local_dim)
        dense = functools.partial(Dense, use_bias=False, dtype=self.dtype, precision=self.precision,
                                  dot_general=float32_dot_general)
        if cfg.projection_layers == 1:
            return dense(cfg.out_hidden_size, name="projection")(x)
        mlp = GatedMLP(4 * cfg.group_size * cfg.input_local_dim, cfg.out_hidden_size,
                        in_features=x.shape[-1], activation=gelu_exact,
                        names=("projection_fc1", "projection_fc2"), dtype=self.dtype,
                        precision=self.precision)
        nn.share_scope(self, mlp)
        return mlp(x)


class MiMoAudioTower(nn.Module):
    """Mel frames `[clips, frames, n_mels]` with their valid-frame mask to the decoder's embeddings:
    each clip cut into segments of `audio_segment_size` frames, every segment tokenized as its own
    sequence (tokenize_audio_batch), a clip's codes joined, then the code encoder. The mask marks the
    groups a clip's codes fill."""

    config: MiMoAudio
    dtype: Dtype | None = None
    precision: PrecisionLike = None

    @nn.compact
    def __call__(self, input_features, input_features_mask) -> AudioEncoding:
        cfg = self.config
        clips, frames, mel = input_features.shape
        size = cfg.audio_segment_size
        segments = -(-frames // size)
        lengths = jnp.sum(jnp.asarray(input_features_mask, jnp.int32), axis=1)
        mels = sequence_blocks(input_features, size).reshape(clips * segments, size, mel)
        spans = jnp.clip(lengths[:, None] - jnp.arange(segments)[None, :] * size, 0, size).reshape(-1)
        features, valid = AudioTokenizerEncoder(cfg.tokenizer, dtype=self.dtype, precision=self.precision,
                                                name="tokenizer")(mels, spans, batch_longest(spans, cfg))
        codes = ResidualQuantizer(cfg.tokenizer.codebook_size, cfg.tokenizer.d_model,
                                  name="quantizer")(features)
        # Only a clip's last segment is short, so its codes are a prefix of the joined rows.
        codes = codes.reshape(clips, -1, codes.shape[-1])
        counts = jnp.sum(valid.reshape(clips, -1), axis=1)
        features = MiMoAudioCodeEncoder(cfg, dtype=self.dtype, precision=self.precision, name="encoder")(
            codes, counts)
        groups = grouped(counts, cfg.group_size)
        return AudioEncoding(features, jnp.arange(features.shape[1])[None, :] < groups[:, None])


def tower_counts(frames, tokenizer: MiMoAudioTokenizer):
    """The tokens a segment of `frames` mel frames (an integer or an integer array, on the host or traced)
    holds at each stage of the audio tokenizer's encoder: its two convolutions' (`AudioTokenizerEncoder`'s
    conv1 and its strided conv2), then its pooling's by `avg_pooler` rounding up, its codes. The one count
    the tower's masks and the host's placement read (`MiMoAudio.encoded`). An empty segment, a clip's
    padding, holds none at any stage, as the release tokenizes none (`batch_longest`), whatever count the
    convolutions' arithmetic gives zero frames."""
    kernel, stride = tokenizer.kernel_size, tokenizer.stride_size
    convolved = (frames > 0) * ((frames + 3 - kernel + 2 - kernel) // stride + 1)
    return convolved, -(-convolved // tokenizer.avg_pooler)


def grouped(codes, group_size: int):
    """A clip's encoded tokens of its `codes`: one a group of `group_size`, the last group short
    (`MiMoAudioCodeEncoder`)."""
    return -(-codes // group_size)


def batch_longest(spans, config: MiMoAudio) -> jax.Array:
    """Each segment's reference batch's longest: tokenize_audio_batch takes the segments in order into
    batches of at most `tokenizer_batch_frames` frames (`_at_group_by_length`), a segment that would
    pass the bound opening the next batch unless its batch is empty; empty segments join none."""

    def take(carry, span):
        total, batch = carry
        opens = (total + span > config.tokenizer_batch_frames) & (total > 0)
        return (jnp.where(opens, span, total + span), batch + opens), batch + opens

    _, batches = jax.lax.scan(take, (jnp.zeros((), spans.dtype), jnp.zeros((), jnp.int32)), spans)
    longest = jax.ops.segment_max(spans, batches, num_segments=spans.shape[0])
    return longest[batches]




















