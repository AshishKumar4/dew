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
from collections.abc import Mapping
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.typing import Dtype, PrecisionLike

from dew import records
from dew.interop.weights import translate_parameters
from dew.nn.conv import Conv
from dew.nn.dense import Dense
from dew.nn.precision import float32_dot_general
from dew.registry import Record

from .activations import gelu_exact
from .attention import LayerNorm, RMSNorm
from .audio import AudioEncoding
from .backbones.decoder_block import BlockWiring, DecoderBlock
from .blocks import TokenEmbedding
from .inputs import AttentionMetadata, coordinates, sequence_blocks
from .mixer_base import MixerContext
from .mixers.attention import CausalSelfAttention
from .mlp import GatedMLP
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

        def convolved(frames):
            return (frames + 3 - cfg.kernel_size + 2 - cfg.kernel_size) // cfg.stride_size + 1

        counts, widest = convolved(lengths), convolved(jnp.broadcast_to(batch_lengths, lengths.shape))
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
            counts = -(-counts // pool)
        return x, jnp.arange(x.shape[1])[None, :] < counts[:, None]


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
        groups = -(-frames // cfg.group_size)
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
        groups = -(-counts // cfg.group_size)
        return AudioEncoding(features, jnp.arange(features.shape[1])[None, :] < groups[:, None])


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


def translate_mimo_audio_config(audio_config: Mapping[str, object],
                                tokenizer_config: Mapping[str, object]) -> Record:
    """A MiMo-V2 audio_config and its audio tokenizer's config into a MiMoAudio value's fields."""
    if str(tokenizer_config.get("ln_type", "LayerNorm")) != "LayerNorm":
        raise ValueError(f"ln_type {tokenizer_config.get('ln_type')!r} is not the tokenizer's layer norm")
    if str(tokenizer_config.get("activation_function", "gelu")) != "gelu":
        raise ValueError("the tokenizer's feed-forward is GELU")
    if records.number(audio_config.get("partial_rotary_factor", 1.0), "partial_rotary_factor") != 1.0:
        raise ValueError("the audio encoder rotates every head dimension")
    channels = records.integer(audio_config["audio_channels"], "audio_channels")
    # The release writes the vocabularies as text: one size for every channel, or each one's, dashed.
    vocab = str(audio_config.get("speech_vocab_size", 1280))
    sizes = [records.integer(int(size), "speech_vocab_size") for size in vocab.split("-")]
    sizes = sizes if len(sizes) > 1 else sizes * channels
    stages = records.integer(tokenizer_config.get("num_quantizers", 20), "num_quantizers")
    books = tokenizer_config.get("codebook_size", 1024)
    books = records.integers(books, "codebook_size") if isinstance(books, (list, tuple)) else (
        records.integer(books, "codebook_size"),) * stages
    books = (*books, *(books[-1],) * (stages - len(books)))[:stages]
    skip = tokenizer_config.get("encoder_skip_layer_id")
    tokenizer = {
        "n_mels": records.integer(tokenizer_config["n_mels"], "n_mels"),
        "sampling_rate": records.integer(tokenizer_config["sampling_rate"], "sampling_rate"),
        "nfft": records.integer(tokenizer_config["nfft"], "nfft"),
        "hop_length": records.integer(tokenizer_config["hop_length"], "hop_length"),
        "window_size": records.integer(tokenizer_config["window_size"], "window_size"),
        "fmin": records.number(tokenizer_config.get("fmin", 0.0), "fmin"),
        "fmax": (None if tokenizer_config.get("fmax") is None
                 else records.number(tokenizer_config["fmax"], "fmax")),
        "d_model": records.integer(tokenizer_config["d_model"], "d_model"),
        "kernel_size": records.integer(tokenizer_config["kernel_size"], "kernel_size"),
        "stride_size": records.integer(tokenizer_config["stride_size"], "stride_size"),
        "avg_pooler": records.integer(tokenizer_config["avg_pooler"], "avg_pooler"),
        "encoder_layers": records.integer(tokenizer_config["encoder_layers"], "encoder_layers"),
        "encoder_attention_heads": records.integer(tokenizer_config["encoder_attention_heads"],
                                                   "encoder_attention_heads"),
        "encoder_ffn_dim": records.integer(tokenizer_config["encoder_ffn_dim"], "encoder_ffn_dim"),
        "encoder_causal": records.boolean(tokenizer_config.get("encoder_causal", False), "encoder_causal"),
        "encoder_attn_window_size": records.integers(
            tokenizer_config.get("encoder_attn_window_size", [-1, -1]), "encoder_attn_window_size"),
        "encoder_skip_layer_id": None if skip is None else records.integer(skip, "encoder_skip_layer_id"),
        "hybrid_attention": records.boolean(tokenizer_config.get("hybrid_attention", False),
                                            "hybrid_attention"),
        "swa_per_block": records.integer(tokenizer_config.get("swa_per_block", 2), "swa_per_block"),
        "rope_theta": records.number(tokenizer_config.get("rope_theta", 10000.0), "rope_theta"),
        "codebook_size": books,
    }
    return {"class": "mimo_audio", "fields": {
        "tokenizer": tokenizer,
        "audio_channels": channels,
        "group_size": records.integer(audio_config["group_size"], "group_size"),
        "input_local_dim": records.integer(audio_config["input_local_dim"], "input_local_dim"),
        "input_local_layers": records.integer(audio_config["input_local_layers"], "input_local_layers"),
        "input_local_attn_heads": records.integer(audio_config["input_local_attn_heads"],
                                                  "input_local_attn_heads"),
        "input_local_intermediate_size": records.integer(audio_config["input_local_intermediate_size"],
                                                         "input_local_intermediate_size"),
        "input_full_attention": records.boolean(audio_config.get("input_full_attention", True),
                                                "input_full_attention"),
        "rope_theta": records.number(audio_config.get("rope_theta", 640000.0), "rope_theta"),
        "add_post_norm": records.boolean(audio_config.get("add_post_norm", True), "add_post_norm"),
        "projection_layers": records.integer(audio_config.get("projection_layers", 2), "projection_layers"),
        "out_hidden_size": records.integer(audio_config["out_hidden_size"], "out_hidden_size"),
        "speech_vocab_size": tuple(sizes),
        "audio_segment_size": records.integer(audio_config.get("audio_segment_size", 6000),
                                              "audio_segment_size"),
    }}


_TOKENIZER_NORMS = {"self_attn_layer_norm": "input_layernorm", "final_layer_norm": "post_attention_layernorm"}
_TOKENIZER_MAPS = {("self_attn", "q_proj"): ("self_attn", "q_proj"),
                   ("self_attn", "k_proj"): ("self_attn", "k_proj"),
                   ("self_attn", "v_proj"): ("self_attn", "v_proj"),
                   ("self_attn", "out_proj"): ("self_attn", "o_proj"), ("fc1",): ("mlp", "up_proj"),
                   ("fc2",): ("mlp", "down_proj")}


def mimo_audio_tokenizer_path(name: str) -> tuple[str, ...]:
    """One audio tokenizer tensor, its `encoder.` prefix off, into the tower's tree; anything else
    raises ValueError."""
    parts = name.split(".")
    leaf = parts[-1]
    kind = "kernel" if leaf == "weight" else "bias"
    path: tuple[str, ...] | None = None
    if parts[0] in ("conv1", "conv2"):
        path = ("tokenizer", parts[0], kind)
    elif parts[:2] == ["down_sample_layer", "0"]:
        path = ("tokenizer", "down_sample_layer", kind)
    elif parts[0] in ("layer_norm", "down_sample_norm"):
        path = ("tokenizer", parts[0], "scale" if leaf == "weight" else "bias")
    elif parts[:2] == ["quantizer", "vq"] and parts[-1] == "embed" and parts[3].isdigit():
        path = ("quantizer", f"codebook_{parts[3]}")
    elif parts[0] == "layers" and parts[1].isdigit():
        layer, rest = f"layers_{parts[1]}", tuple(parts[2:-1])
        if rest[0] in _TOKENIZER_NORMS and len(rest) == 1:
            path = ("tokenizer", layer, _TOKENIZER_NORMS[rest[0]], "scale" if leaf == "weight" else "bias")
        elif rest in _TOKENIZER_MAPS:
            path = ("tokenizer", layer, *_TOKENIZER_MAPS[rest], kind)
    if path is None:
        raise ValueError(f"unknown tensor name {name!r}")
    return path


def mimo_audio_encoder_path(name: str) -> tuple[str, ...]:
    """One audio encoder tensor (`speech_embeddings.*`, or `audio_encoder.*` its prefix off) into
    the tower's tree; anything else raises ValueError."""
    parts = name.split(".")
    leaf = parts[-1]
    path: tuple[str, ...] | None = None
    if parts[0] == "speech_embeddings" and parts[1].isdigit():
        path = ("encoder", f"speech_embeddings_{parts[1]}", "embedding")
    elif parts[0] == "projection":
        site = {"mlp.0": "projection_fc1", "mlp.2": "projection_fc2"}.get(".".join(parts[1:-1]), "projection")
        path = ("encoder", site, "kernel")
    elif parts[:2] == ["input_local_transformer", "norm"]:
        path = ("encoder", "norm", "scale")
    elif parts[:2] == ["input_local_transformer", "layers"] and parts[2].isdigit():
        layer, rest = f"layers_{parts[2]}", tuple(parts[3:-1])
        if rest in (("input_layernorm",), ("post_attention_layernorm",)):
            path = ("encoder", layer, rest[0], "scale")
        elif len(rest) == 2 and rest[0] in ("self_attn", "mlp"):
            path = ("encoder", layer, *rest, "kernel" if leaf == "weight" else "bias")
    if path is None:
        raise ValueError(f"unknown tensor name {name!r}")
    return path


def translate_mimo_audio_weights(tokenizer: Mapping[str, np.ndarray], encoder: Mapping[str, np.ndarray], *,
                                 param_dtype: str = "float32") -> dict:
    """The tower's parameters from the tokenizer's tensors (`encoder.` prefix off) and the audio
    encoder's. The codebooks are the quantizer's frozen `embed` buffers, its running statistics
    left behind; the tokenizer's key maps carry no bias, so theirs is zero."""
    statistics = ("inited", "cluster_size", "embed_avg")
    tensors = {name: tensor for name, tensor in tokenizer.items() if name.split(".")[-1] not in statistics}
    for name, tensor in list(tensors.items()):
        if name.endswith("k_proj.weight"):
            bias = name.removesuffix("weight") + "bias"
            tensors.setdefault(bias, np.zeros(np.shape(tensor)[0], np.asarray(tensor).dtype))
    params = translate_parameters(tensors, mimo_audio_tokenizer_path, param_dtype)
    params.update(translate_parameters(encoder, mimo_audio_encoder_path, param_dtype))
    return {"params": params}
