"""Engram: n-gram hash lookups written into the residual streams.

DeepSeek-V4.1 (arXiv 2609.19969, section 2.4.2) adds Engram (Cheng et al.
2026) at a few layers; the reference is the release's `engram.py` and
`model.py` (DEEPSEEK_V41_REVISION in tools/deepseek_v41_reference.py). Per
position:

1. Every token id maps through a compressed vocabulary of normalized ids
   (`compressed_token_map`, engram.py:17-55).
2. The position and the `max_ngram_size - 1` compressed ids before it make
   one n-gram of each size from 2 up; look-back stops at the sequence start
   and at a dead token (an image span), and missing slots hold the pad id's
   compressed id (engram.py:155-170).
3. Each layer multiplies the look-back ids by its own odd multipliers and
   XORs them, so after `i` steps the value hashes the `(i + 1)`-gram; each
   (n-gram size, head) reduces it modulo its prime and offsets it into its
   bucket range (engram.py:172-180).
4. The layer projects the looked-up rows into one key per residual stream
   and a shared value, gated into each stream by a sigmoid of the
   sign-preserving square root of the stream's normalized dot product with
   its key (model.py:328-365).

The reference hashes in 64-bit integers; here 8-bit limbs in uint32 stay
exact without x64, since every compressed id stays under 2**24 and every
prime under 2**28.

Qwen4-Exp's n-gram per-layer embedding (`Qwen4ExpTextNGramEmbedding` and
`Qwen4ExpTextPLELayer`, modeling_qwen4_exp.py:971-1190 in transformers
5.16.1) hashes the same way over raw token ids: its multipliers come from a
SplitMix64 stream (`Engram.multiplier_seed`), its look-back stops at an
end-of-sequence id, which fills the slots it stops (`stop_token_id`,
`pad_token_id`), and each table's rows round up to a multiple (`row_multiple`).
Its layer (`PLELayer`) gates one shared value into every stream by the same
sign-preserving root, then adds a dilated causal conv of the gated values.
"""

from __future__ import annotations

import dataclasses
import functools

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.typing import Dtype, PrecisionLike

from dew.batch import Coordinates, document_ranks, prefix_sum, real_places, valid_order
from dew.nn.dense import Dense

from .inputs import PLAIN_CALL, MixerCall, coordinates_of
from .linear import DepthwiseConv1d, held_conv1d
from .norms import RMSNorm
from .precision import at_least_fp32, float32_dot_general, rms_statistics
from .sharding import logical_axes

__all__ = ["Engram", "PLELayer", "compressed_token_map"]

_DEAD = -1
"""A compressed id no n-gram reaches across (engram.py:130)."""
_LIMB = 8
_LIMBS = 8


@dataclasses.dataclass(frozen=True)
class Engram:
    """The engram layers and their hash layout, by the release's config names.

    `layer_ids` are the decoder layers that add a lookup before their
    attention; `num_embeddings` their table rows (none stated: the primes'
    layout), `vocab_size` the bucket
    count each prime search starts above, `compressed_vocab_size` the size of
    the tokenizer's compressed vocabulary every multiplier derives from, and
    `pad_token_id` the vocabulary id whose compressed id fills missing
    look-back (engram_pad_token_id).
    """

    layer_ids: tuple[int, ...]
    num_embeddings: tuple[int, ...]
    max_ngram_size: int
    vocab_size: int
    n_heads: int
    head_dim: int
    compressed_vocab_size: int
    pad_token_id: int = 2
    multiplier_seed: int | None = None
    """Qwen4-Exp's `seed`: each table's multipliers drawn from SplitMix64 at `seed + 10007 * index`, the
    table's index among the lookups (`_build_layer_multipliers`, modeling_qwen4_exp.py:986-996); None draws
    them as V4.1 does (`multipliers`)."""
    stop_token_id: int | None = None
    """A token id a position's look-back stops at, filling that slot and every one past it with the pad
    id: Qwen4-Exp's end of sequence (`_shift_right_ignore_eos`, :1053-1067); None for none."""
    row_multiple: int = 1
    """Each table's rows, the primes' sum rounded up to this multiple (make_ngram_vocab_size_divisible_by)."""
    ple_conv_kernel_size: int | None = None
    """Qwen4-Exp's per-layer embedding: each lookup a `PLELayer`, its conv of this many taps
    `max_ngram_size` apart; None for V4.1's `EngramLayer`."""
    ple_bias: bool = False
    """Whether a `PLELayer`'s key and value projections add biases: DeepSeek's demo Engram's and
    MaxText's (maxtext/layers/engram.py), whose layer the PLE layer computes; Qwen4-Exp's add none."""
    table_parts: int = 1
    """The equal row shards each table is stored in (Qwen4-Exp's `split_ngram_parts`), a PLE layer's
    table one leaf of them stacked (`PLELayer`), each written back as stored."""
    media_stops: bool = True
    """Whether a media position is dead, its look-back stopped there (V4.1, engram.py:164-166); Qwen4-Exp
    hashes an image's placeholder ids as it does any token's (`ple_input_ids`, modeling_qwen4_exp.py:1361)."""

    def __post_init__(self):
        object.__setattr__(self, "layer_ids", tuple(int(layer) for layer in self.layer_ids))
        object.__setattr__(self, "num_embeddings", tuple(int(rows) for rows in self.num_embeddings))
        if self.num_embeddings and len(self.num_embeddings) != len(self.layer_ids):
            raise ValueError(
                f"engram_num_embeddings names one table per engram layer, got "
                f"{len(self.num_embeddings)} for layers {self.layer_ids}")
        if self.max_ngram_size < 2 or self.n_heads < 1 or self.head_dim < 1 or self.row_multiple < 1:
            raise ValueError("an engram hashes n-grams of two tokens or more over one head or more")
        if self.ple_bias and self.ple_conv_kernel_size is None:
            raise ValueError("ple_bias biases a PLE layer's projections, which V4.1's engram layer has "
                             "none of")
        if not 0 < self.compressed_vocab_size < 2 ** 24:
            raise ValueError(
                f"compressed ids are hashed exactly below 2**24, got a compressed "
                f"vocabulary of {self.compressed_vocab_size}")
        for layer, primes in zip(self.layer_ids, self.primes, strict=True):
            if max(p for per in primes for p in per) >= 2 ** 28:
                raise ValueError(f"engram layer {layer}'s bucket primes pass 2**28")
        declared = tuple(-(-sum(sum(per) for per in primes) // self.row_multiple) * self.row_multiple
                         for primes in self.primes)
        if not self.num_embeddings:
            # Qwen4-Exp's config states no table sizes: they are its primes' layout.
            object.__setattr__(self, "num_embeddings", declared)
        if declared != self.num_embeddings:
            raise ValueError(
                f"engram_num_embeddings {self.num_embeddings} are not the tables the "
                f"bucket primes lay out, {declared}")
        if any(rows % self.table_parts for rows in self.num_embeddings) or self.table_parts < 1:
            raise ValueError(f"each table's rows {self.num_embeddings} split into {self.table_parts} "
                             "equal shards")

    @functools.cached_property
    def primes(self) -> tuple[tuple[tuple[int, ...], ...], ...]:
        """`[layer][n-gram size][head]` bucket moduli: the next unused prime
        above `vocab_size - 1`, drawn in order (engram.py:9-14, :107-118).
        Each search restarts there and skips the primes drawn before it, so
        the draws are the successive primes from `vocab_size` on."""
        from sympy import nextprime

        current = self.vocab_size - 1
        layers = []
        for _ in self.layer_ids:
            per_ngram = []
            for _ in range(self.max_ngram_size - 1):
                sizes = []
                for _ in range(self.n_heads):
                    current = nextprime(current)
                    sizes.append(current)
                per_ngram.append(tuple(sizes))
            layers.append(tuple(per_ngram))
        return tuple(layers)

    @functools.cached_property
    def multipliers(self) -> np.ndarray:
        """`[layers, max_ngram_size]` odd multipliers, bounded so a product fits int64: from a per-layer
        RNG seeded 10007 * layer (engram.py:58-75), or under `multiplier_seed` from SplitMix64 at
        `seed + 10007 * index` (modeling_qwen4_exp.py:986-996)."""
        bound = max(1, (np.iinfo(np.int64).max // self.compressed_vocab_size) // 2)
        if self.multiplier_seed is not None:
            return np.asarray([[2 * (_splitmix64(self.multiplier_seed + 10007 * index + _GAMMA * (size + 1))
                                     % bound) + 1 for size in range(self.max_ngram_size)]
                               for index in range(len(self.layer_ids))], np.int64)
        rows = [np.random.default_rng(10007 * layer).integers(
            low=0, high=bound, size=(self.max_ngram_size,), dtype=np.int64) * 2 + 1
            for layer in self.layer_ids]
        return np.stack(rows)

    def hash_ids(self, compressed, blocked, pad):
        """Every engram layer's bucket ids, `[B, S, layers, columns]` int32,
        a column per (n-gram size, head).

        `compressed` `[B, S, max_ngram_size]` holds each position's compressed
        id and its look-back, nearest first, with `blocked` marking the slots
        past the sequence start or a dead token; those hash as `pad`, the pad
        token's compressed id.
        """
        tokens = jnp.where(blocked, jnp.asarray(pad, jnp.uint32), compressed.astype(jnp.uint32))
        limbs = _multiplier_limbs(self.multipliers)  # [layers, n, limbs]
        products = _multiply(tokens[:, :, None, :], limbs)  # [B, S, layers, n, 8]
        primes = np.asarray(self.primes, np.uint32)  # [layers, n - 1, heads]
        offsets = np.concatenate(
            [np.zeros((len(self.layer_ids), 1), np.int64),
             np.cumsum(primes.reshape(len(self.layer_ids), -1), axis=1)[:, :-1]], axis=1)
        rolling = products[..., 0, :]
        hashes = []
        for size in range(1, self.max_ngram_size):
            rolling = jnp.bitwise_xor(rolling, products[..., size, :])
            hashes.append(_modulo(rolling[..., None, :], primes[None, None, :, size - 1, :, None]))
        ids = jnp.concatenate(hashes, axis=-1).astype(jnp.int32)
        return ids + jnp.asarray(offsets, jnp.int32)


_MASK64 = (1 << 64) - 1
_GAMMA = 0x9E3779B97F4A7C15


def _splitmix64(value: int) -> int:
    """SplitMix64's step from `value` (modeling_qwen4_exp.py:979-984), in Python's exact integers."""
    value = (value + _GAMMA) & _MASK64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _MASK64
    return (value ^ (value >> 31)) & _MASK64


def _multiplier_limbs(multipliers: np.ndarray) -> np.ndarray:
    values = multipliers.astype(np.uint64)
    return np.stack([(values >> np.uint64(_LIMB * index)) & np.uint64(0xFF)
                     for index in range(_LIMBS)], axis=-1).astype(np.uint32)


def _multiply(tokens, limbs):
    """`tokens * multiplier` as eight little-endian 8-bit limbs; a token is
    under 2**24 and a limb under 2**8, so every column sum fits uint32."""
    columns = tokens[..., None] * jnp.asarray(limbs)
    out, carry = [], jnp.zeros(columns.shape[:-1], jnp.uint32)
    for index in range(_LIMBS):
        total = columns[..., index] + carry
        out.append(total & 0xFF)
        carry = total >> _LIMB
    return jnp.stack(out, axis=-1)


def _modulo(limbs, prime):
    """The 64-bit value the limbs hold modulo `prime` (< 2**28): Horner over
    the bytes' 4-bit halves from the top, every partial below 2**32."""
    prime = jnp.asarray(prime, jnp.uint32)
    remainder = jnp.zeros(jnp.broadcast_shapes(limbs.shape[:-1], prime.shape[:-1]), jnp.uint32)
    for index in reversed(range(_LIMBS)):
        for half in (limbs[..., index] >> 4, limbs[..., index] & 0xF):
            remainder = (remainder * 16 + half) % prime[..., 0]
    return remainder


def _lookback(compressed, valid, positions, size: int, history=None, stop=None):
    """Each position's compressed id and its `size - 1` predecessors.

    `compressed` `[B, S]` int32 (_DEAD for a dead token), `valid` `[B, S]` the
    tokens that are part of the sequence (padding is skipped, not dead),
    `positions` `[B, S]` their sequence positions, `history` `[B, size - 1]`
    the ids before this call, nearest last, _DEAD where none, and `stop` the
    compressed id a predecessor stops the look-back at, or None. Returns
    `([B, S, size] ids, [B, S, size] blocked, [B, size - 1] new history)`.
    """
    batch = compressed.shape[0]
    if history is None:
        history = jnp.full((batch, size - 1), _DEAD, jnp.int32)
    # The valid tokens packed to the front, after the history they follow.
    order, _ = valid_order(valid)
    packed = jnp.concatenate(
        [history, jnp.take_along_axis(jnp.where(valid, compressed, _DEAD), order, axis=1)], axis=1)
    rank = prefix_sum(valid, dtype=jnp.int32) - 1 + (size - 1)
    shifts = jnp.arange(size)
    at = rank[..., None] - shifts
    ids = jnp.take_along_axis(packed[:, None, :], jnp.maximum(at, 0), axis=2)
    dead = (at < 0) | (ids == _DEAD) | (positions[..., None] < shifts)
    if stop is not None:
        dead = dead | ((ids == stop) & (shifts > 0))
    blocked = jnp.cumsum(dead, axis=-1) > 0
    count = jnp.sum(valid, axis=1)
    tail = jnp.arange(size - 1) + count[:, None]
    history = jnp.take_along_axis(packed, tail, axis=1)
    return ids, blocked, history


class EngramHashes(nn.Module):
    """Every engram layer's bucket ids for a call's tokens,
    `[B, S, layers, columns]` (engram.py:153-180).

    Tokens map through the `constants` collection's `token_map`, which a loaded
    checkpoint derives from its tokenizer and a fresh model fills with each id
    modulo the compressed vocabulary. Each position hashes with the valid tokens
    before it, skipping padding and stopping at a row's start, a packed
    document's (its rank in its document, `Coordinates.starts`, whatever logical
    positions the call gives), a `media` position (engram.py:164-166) and, while
    decoding, where the cached history runs out.
    """

    spec: Engram
    vocab_size: int

    @nn.compact
    def __call__(self, tokens, coordinates: Coordinates | None, decode: bool, media=None):
        spec, vocab = self.spec, self.vocab_size
        table = self.variable('constants', 'token_map',
                              lambda: jnp.arange(vocab, dtype=jnp.int32) % spec.compressed_vocab_size).value
        compressed = jnp.take(table, tokens, axis=0)
        if media is not None and spec.media_stops:
            compressed = jnp.where(media, _DEAD, compressed)
        # The call's layout (`Coordinates`): its real tokens, and their places a decode's history reads.
        laid = Coordinates() if coordinates is None else coordinates
        valid = jnp.ones(tokens.shape, bool) if laid.valid is None else jnp.asarray(laid.valid, bool)
        positions = laid.positions
        size = spec.max_ngram_size
        history = held = None
        allocated = False
        if decode:
            allocated = self.has_variable('cache', 'history')
            held = self.variable('cache', 'history', jnp.full, (tokens.shape[0], size - 1), _DEAD, jnp.int32)
            history = held.value
            if positions is None:
                # The history knows where a row starts; a call's positions do not.
                positions = jnp.full(tokens.shape, size, jnp.int32)
        else:
            # Each token's rank in its document bounds its lookback, the packed documents' by their starts.
            real = valid if laid.segments is None else valid & (jnp.asarray(laid.segments) != 0)
            positions = real_places(valid) if laid.starts is None else document_ranks(real, laid.starts)
        stop = None if spec.stop_token_id is None else table[spec.stop_token_id]
        ids, blocked, history = _lookback(compressed, valid, jnp.asarray(positions), size, history, stop)
        if held is not None and allocated:
            held.value = history
        return spec.hash_ids(ids, blocked, table[spec.pad_token_id])


@logical_axes({
    # The table's rows, 384M a layer at release size, shard as a
    # vocabulary's do; the key weights are model-width rows per stream.
    ("engram", "embed"): ("vocab", None),
    ("engram",): (None, "embed"),
    ("wkv",): (None, "embed"),
})
class EngramLayer(nn.Module):
    """One layer's lookup gated into the residual streams (model.py:328-365).

    `embed` `[rows, head_dim]` is the release's `engram.embed.weight`; `wkv`
    projects a position's looked-up rows to `hc_mult` keys and one value; `q_weight`
    and `k_weight` `[hc_mult, D]` only ever act as their product. A media position of
    the call (`call.metadata.media`) takes no contribution (model.py:351-365).
    """

    rows: int
    head_dim: int
    hc_mult: int
    emb_features: int
    norm_eps: float
    dtype: Dtype | None = None
    precision: PrecisionLike = None

    @nn.compact
    def __call__(self, streams, hash_ids, call: MixerCall = PLAIN_CALL):
        media = None if call.metadata is None else call.metadata.media
        token_mask = None if media is None else ~media
        # The rows are gathered from the table in its storage dtype and cast
        # after, so the table itself is never cast whole.
        table = nn.Embed(self.rows, self.head_dim, embedding_init=nn.initializers.normal(1.0), name='embed')
        dtype = streams.dtype if self.dtype is None else self.dtype
        looked = table(hash_ids).astype(dtype).reshape(*hash_ids.shape[:2], -1)
        kv = Dense(self.emb_features * (self.hc_mult + 1), use_bias=False, dtype=self.dtype,
                   precision=self.precision, name='wkv', dot_general=float32_dot_general)(looked)
        q_weight = self.param(
            "q_weight", nn.initializers.ones, (self.hc_mult, self.emb_features), jnp.float32
        )
        k_weight = self.param(
            "k_weight", nn.initializers.ones, (self.hc_mult, self.emb_features), jnp.float32
        )
        wide = at_least_fp32(kv.dtype)
        key = kv[..., :self.hc_mult * self.emb_features].astype(wide).reshape(
            *kv.shape[:2], self.hc_mult, self.emb_features)
        value = kv[..., self.hc_mult * self.emb_features:].astype(wide)
        h = streams.astype(wide)
        rstd = (rms_statistics(h, self.norm_eps, keepdims=False)[1]
                * rms_statistics(key, self.norm_eps, keepdims=False)[1])
        dot = jnp.sum(h * (q_weight * k_weight) * key, -1) * rstd * self.emb_features ** -0.5
        gate = jax.nn.sigmoid(jnp.copysign(jnp.sqrt(jnp.maximum(jnp.abs(dot), 1e-6)), dot))
        if token_mask is not None:
            gate = jnp.where(token_mask[..., None], gate, 0.0)
        return (h + gate[..., None] * value[..., None, :]).astype(streams.dtype)


class ShardedTable(nn.Module):
    """A table of `rows` stored in `parts` equal row shards, one `[parts, rows / parts, features]` leaf at
    `embed/embedding`, which a load stacks from the shards and an export writes back shard by shard
    (`decoder_parts.expert_slot`); a lookup reads the rows in order. The rows are gathered in the
    table's storage dtype and cast after, so the table itself is never cast whole."""

    parts: int
    rows: int
    features: int

    @nn.compact
    def __call__(self, ids):
        table = Embedded(self.parts, self.rows // self.parts, self.features, name='embed')()
        return jnp.take(table.reshape(self.rows, self.features), ids, axis=0)


class Embedded(nn.Module):
    """The `[parts, rows, features]` leaf `embedding` of a `ShardedTable`, drawn as `nn.Embed` draws."""

    parts: int
    rows: int
    features: int

    @nn.compact
    def __call__(self):
        return self.param('embedding', nn.initializers.normal(1.0), (self.parts, self.rows, self.features),
                          jnp.float32)


class PLELayer(nn.Module):
    """Qwen4-Exp's n-gram per-layer embedding, added to the residual streams
    (`Qwen4ExpTextPLELayer`, modeling_qwen4_exp.py:1117-1190), which is DeepSeek's demo Engram as
    MaxText computes it (maxtext/layers/engram.py:496-695), its projections biased there (`use_bias`).

    The looked-up rows of `shards` (`ShardedTable`) project to one key per
    stream (`key_proj`) and one shared value (`value_proj`). Each stream's
    normed activations meet its normed key, and the sign-preserving root of
    their scaled product gates the value into it; the gated values, normed
    again, run through a causal depthwise conv of `kernel_size` taps
    `dilation` apart and a SiLU, which the gated values themselves join. The
    three norms take each stream's own statistics under one `(1 + w)` weight
    (`RMSNorm(groups=hc_mult)`). The call's padded slots (`call.metadata`) add
    nothing and the conv steps over them and across no document's start
    (modeling_qwen4_exp.py:1176-1190); under `call.decode` the conv's last
    `(kernel_size - 1) * dilation` inputs wait in the `cache` collection's
    `conv_state`, which the first decode call only allocates.
    """

    rows: int
    head_dim: int
    hc_mult: int
    emb_features: int
    norm_eps: float
    kernel_size: int
    dilation: int
    parts: int = 1
    use_bias: bool = False
    dtype: Dtype | None = None
    precision: PrecisionLike = None

    @nn.compact
    def __call__(self, streams, hash_ids, call: MixerCall = PLAIN_CALL):
        batch, length, hc, width = streams.shape
        tokens = coordinates_of(call.metadata, batch, length)
        token_mask, segments = tokens.valid, tokens.segments
        table = ShardedTable(self.parts, self.rows, self.head_dim, name='shards')
        dtype = streams.dtype if self.dtype is None else self.dtype
        looked = table(hash_ids).astype(dtype).reshape(batch, length, -1)
        dense = functools.partial(Dense, use_bias=self.use_bias, dtype=self.dtype, precision=self.precision,
                                  dot_general=float32_dot_general)
        norm = functools.partial(RMSNorm, epsilon=self.norm_eps, scale_offset=True, groups=hc, dtype=dtype)
        key = norm(name='norm_key')(dense(hc * width, name='key_proj')(looked))
        key = key.reshape(batch, length, hc, width)
        value = dense(width, name='value_proj')(looked)
        query = norm(name='norm_query')(streams.reshape(batch, length, hc * width)).reshape(key.shape)
        dot = jnp.sum(key * query, axis=-1) / np.sqrt(width).astype(np.float32)
        gate = jnp.sign(dot) * jnp.sqrt(jnp.maximum(jnp.abs(dot), 1e-6))
        gated = (jax.nn.sigmoid(gate)[..., None] * value[:, :, None, :]).reshape(batch, length, hc * width)
        normed = norm(name='norm_conv')(gated)
        if token_mask is not None:
            gated = jnp.where(token_mask[..., None], gated, 0).astype(gated.dtype)
            normed = jnp.where(token_mask[..., None], normed, 0).astype(normed.dtype)
        taps, _ = DepthwiseConv1d(features=hc * width, kernel=self.kernel_size, name='conv1d')()
        # The dilated taps as one causal kernel `(kernel_size - 1) * dilation + 1` wide, zero between them.
        span = (self.kernel_size - 1) * self.dilation + 1
        dilated = jnp.zeros((hc * width, span), taps.dtype).at[:, ::self.dilation].set(taps)
        held = None
        if call.decode:
            allocated = self.has_variable('cache', 'conv_state')
            held = self.variable('cache', 'conv_state', jnp.zeros, (batch, span - 1, hc * width),
                                 normed.dtype)
            if not allocated:
                return streams
        convolved, history = held_conv1d(normed, dilated.astype(normed.dtype), token_mask,
                                         None if held is None else held.value, segments=segments)
        if held is not None:
            held.value = history.astype(held.value.dtype)
        return streams + (gated + jnp.asarray(convolved)).reshape(streams.shape).astype(streams.dtype)


def compressed_token_map(tokenizer) -> tuple[np.ndarray, int]:
    """Every token id onto the compressed vocabulary, and its size
    (engram.py:17-55): a token's text through NFKC, NFD, accent stripping,
    lowercasing, whitespace runs to one space and a strip that leaves a lone
    space standing; a partial UTF-8 token is keyed by its raw form. Ids
    number the keys in first-seen order. `tokenizer` is a transformers fast
    tokenizer; the raw Rust tokenizer decodes, as training did."""
    from tokenizers import Regex, normalizers

    sentinel = "\ue000"
    normalizer = normalizers.Sequence([
        normalizers.NFKC(), normalizers.NFD(), normalizers.StripAccents(), normalizers.Lowercase(),
        normalizers.Replace(Regex(r"[ \t\r\n]+"), " "), normalizers.Replace(Regex(r"^ $"), sentinel),
        normalizers.Strip(), normalizers.Replace(sentinel, " ")])
    backend = tokenizer.backend_tokenizer
    keys: dict[str, int] = {}
    lookup = np.zeros(len(tokenizer), np.int32)
    for token_id in range(len(tokenizer)):
        text = backend.decode([token_id], skip_special_tokens=False)
        if "\ufffd" in text:
            key = backend.id_to_token(token_id)
        else:
            normalized = normalizer.normalize_str(text)
            key = normalized if normalized else text
        lookup[token_id] = keys.setdefault(key, len(keys))
    return lookup, len(keys)
