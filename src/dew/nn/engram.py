"""Engram: n-gram hash lookups written into the residual streams.

DeepSeek-V4.1 (arXiv 2609.19969, section 2.4.2) adds Engram (Cheng et al.
2026) at a few layers. The reference is the release's inference code,
`engram.py` and `Engram` in `model.py` (DEEPSEEK_V41_REVISION in
tools/deepseek_v41_reference.py). Per position:

1. Every token id maps through a compressed vocabulary in which tokens that
   normalize alike share an id (`compressed_token_map`, engram.py:17-55).
2. The position and the `max_ngram_size - 1` compressed ids before it make
   one n-gram of each size from 2 up. Look-back stops at the sequence start
   and at a dead token (an image span); the missing slots hold the pad id's
   compressed id (engram.py:155-170).
3. Each engram layer multiplies the look-back ids by its own odd multipliers
   and XORs them: after `i` steps the value hashes the `(i + 1)`-gram. Each
   (n-gram size, head) pair reduces it modulo its own prime and offsets it
   into its bucket range of the layer's table (engram.py:172-180).
4. The layer looks the rows up, projects them into one key per residual
   stream and a shared value, and gates the value into each stream by the
   sign-preserving square root of the stream's normalized dot product with
   its key, through a sigmoid (model.py:328-365).

The hashes are 64-bit integer arithmetic in the reference. Here they run on
8-bit limbs in uint32 so they stay exact without x64: a compressed id and
every prime stay under 2**24, which the spec checks.
"""

from __future__ import annotations

import dataclasses
import functools

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.typing import Dtype, PrecisionLike

from .sharding import logical_axes

DEAD = -1
"""A compressed id no n-gram reaches across (engram.py:130)."""
_LIMB = 8
_LIMBS = 8


def _is_prime(n: int) -> bool:
    """Deterministic Miller-Rabin for every n below 3.3e24."""
    if n < 2:
        return False
    small = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41)
    for p in small:
        if n % p == 0:
            return n == p
    d, r = n - 1, 0
    while d % 2 == 0:
        d, r = d // 2, r + 1
    for a in small:
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(r - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


@dataclasses.dataclass(frozen=True)
class Engram:
    """The engram layers and their hash layout, by the release's config names.

    `layer_ids` are the decoder layers that add a lookup before their
    attention; `num_embeddings` their table rows, `vocab_size` the bucket
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

    def __post_init__(self):
        object.__setattr__(self, "layer_ids", tuple(int(layer) for layer in self.layer_ids))
        object.__setattr__(self, "num_embeddings", tuple(int(rows) for rows in self.num_embeddings))
        if len(self.num_embeddings) != len(self.layer_ids):
            raise ValueError(
                f"engram_num_embeddings names one table per engram layer, got "
                f"{len(self.num_embeddings)} for layers {self.layer_ids}")
        if self.max_ngram_size < 2 or self.n_heads < 1 or self.head_dim < 1:
            raise ValueError("an engram hashes n-grams of two tokens or more over one head or more")
        if not 0 < self.compressed_vocab_size < 2 ** 24:
            raise ValueError(
                f"compressed ids are hashed exactly below 2**24, got a compressed "
                f"vocabulary of {self.compressed_vocab_size}")
        for layer, primes in zip(self.layer_ids, self.primes, strict=True):
            if max(p for per in primes for p in per) >= 2 ** 24:
                raise ValueError(f"engram layer {layer}'s bucket primes pass 2**24")
        declared = tuple(sum(sum(per) for per in primes) for primes in self.primes)
        if declared != self.num_embeddings:
            raise ValueError(
                f"engram_num_embeddings {self.num_embeddings} are not the tables the "
                f"bucket primes lay out, {declared}")

    @property
    def columns(self) -> int:
        """Hash ids per position and layer: one per (n-gram size, head)."""
        return (self.max_ngram_size - 1) * self.n_heads

    @functools.cached_property
    def primes(self) -> tuple[tuple[tuple[int, ...], ...], ...]:
        """`[layer][n-gram size][head]` bucket moduli: the next unused prime
        above `vocab_size - 1`, drawn in order (engram.py:9-14, :107-118)."""
        seen: set[int] = set()
        layers = []
        for _ in self.layer_ids:
            per_ngram = []
            for _ in range(self.max_ngram_size - 1):
                sizes, current = [], self.vocab_size - 1
                for _ in range(self.n_heads):
                    current += 1
                    while not _is_prime(current) or current in seen:
                        current += 1
                    seen.add(current)
                    sizes.append(current)
                per_ngram.append(tuple(sizes))
            layers.append(tuple(per_ngram))
        return tuple(layers)

    @functools.cached_property
    def multipliers(self) -> np.ndarray:
        """`[layers, max_ngram_size]` odd multipliers from a per-layer RNG
        seeded 10007 * layer, bounded so a product fits int64
        (engram.py:58-75)."""
        bound = max(1, (np.iinfo(np.int64).max // self.compressed_vocab_size) // 2)
        rows = [np.random.default_rng(10007 * layer).integers(
            low=0, high=bound, size=(self.max_ngram_size,), dtype=np.int64) * 2 + 1
            for layer in self.layer_ids]
        return np.stack(rows)

    def hash_ids(self, compressed, blocked, pad):
        """Every engram layer's bucket ids, `[B, S, layers, columns]` int32.

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
    """The 64-bit value the limbs hold modulo `prime` (< 2**24): Horner over
    the bytes from the top, every partial below 2**32."""
    prime = jnp.asarray(prime, jnp.uint32)
    remainder = jnp.zeros(jnp.broadcast_shapes(limbs.shape[:-1], prime.shape[:-1]), jnp.uint32)
    for index in reversed(range(_LIMBS)):
        remainder = (remainder * 256 + limbs[..., index]) % prime[..., 0]
    return remainder


def lookback(compressed, valid, positions, size: int, history=None):
    """Each position's compressed id and its `size - 1` predecessors.

    `compressed` `[B, S]` int32 (DEAD for a dead token), `valid` `[B, S]` the
    tokens that are part of the sequence (padding is skipped, not dead),
    `positions` `[B, S]` their sequence positions, `history` `[B, size - 1]`
    the ids before this call, nearest last, DEAD where none. Returns
    `([B, S, size] ids, [B, S, size] blocked, [B, size - 1] new history)`.
    """
    batch = compressed.shape[0]
    if history is None:
        history = jnp.full((batch, size - 1), DEAD, jnp.int32)
    # The valid tokens packed to the front, after the history they follow.
    order = jnp.argsort(~valid, axis=1, stable=True)
    packed = jnp.concatenate(
        [history, jnp.take_along_axis(jnp.where(valid, compressed, DEAD), order, axis=1)], axis=1)
    rank = jnp.cumsum(valid, axis=1) - 1 + (size - 1)
    shifts = jnp.arange(size)
    at = rank[..., None] - shifts
    ids = jnp.take_along_axis(packed[:, None, :], jnp.maximum(at, 0), axis=2)
    dead = (at < 0) | (ids == DEAD) | (positions[..., None] < shifts)
    blocked = jnp.cumsum(dead, axis=-1) > 0
    count = jnp.sum(valid, axis=1)
    tail = jnp.arange(size - 1) + count[:, None]
    history = jnp.take_along_axis(packed, tail, axis=1)
    return ids, blocked, history


class EngramHashes(nn.Module):
    """Every engram layer's bucket ids for a call's tokens,
    `[B, S, layers, columns]` (engram.py:153-180).

    Tokens map through the compressed vocabulary, the `constants`
    collection's `token_map`: a loaded checkpoint derives it from its
    tokenizer (`compressed_token_map`), and a fresh model maps each id to
    itself modulo the compressed vocabulary's size. Each position hashes with the valid tokens before it: padding
    is skipped, and look-back stops at a row's start, at a packed document's
    (`positions`) and, while decoding, where the cached history of the row's
    earlier calls runs out. Padding positions get ids nothing reads.
    """

    spec: Engram
    vocab_size: int

    @nn.compact
    def __call__(self, tokens, valid, positions, decode: bool):
        spec, vocab = self.spec, self.vocab_size
        table = self.variable('constants', 'token_map',
                              lambda: jnp.arange(vocab, dtype=jnp.int32) % spec.compressed_vocab_size).value
        compressed = jnp.take(table, tokens, axis=0)
        valid = jnp.ones(tokens.shape, bool) if valid is None else jnp.asarray(valid, bool)
        size = spec.max_ngram_size
        history = held = None
        allocated = False
        if decode:
            allocated = self.has_variable('cache', 'history')
            held = self.variable('cache', 'history', jnp.full, (tokens.shape[0], size - 1), DEAD, jnp.int32)
            history = held.value
            if positions is None:
                # The history knows where a row starts; a call's positions do not.
                positions = jnp.full(tokens.shape, size, jnp.int32)
        elif positions is None:
            positions = jnp.maximum(jnp.cumsum(valid, axis=1) - 1, 0)
        ids, blocked, history = lookback(compressed, valid, jnp.asarray(positions), size, history)
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
    projects the `columns` rows to `hc_mult` keys and one value; `q_weight`
    and `k_weight` `[hc_mult, D]` only ever act as their product.
    """

    rows: int
    columns: int
    head_dim: int
    hc_mult: int
    emb_features: int
    norm_eps: float
    dtype: Dtype | None = None
    precision: PrecisionLike = None

    @nn.compact
    def __call__(self, streams, hash_ids, token_mask=None):
        # The rows are gathered from the table in its storage dtype and cast
        # after, so the table itself is never cast whole.
        table = nn.Embed(self.rows, self.head_dim, embedding_init=nn.initializers.normal(1.0), name='embed')
        dtype = streams.dtype if self.dtype is None else self.dtype
        looked = table(hash_ids).astype(dtype).reshape(*hash_ids.shape[:2], -1)
        kv = nn.Dense(self.emb_features * (self.hc_mult + 1), use_bias=False, dtype=self.dtype,
                      precision=self.precision, name='wkv')(looked)
        q_weight = self.param('q_weight', nn.initializers.ones, (self.hc_mult, self.emb_features), jnp.float32)
        k_weight = self.param('k_weight', nn.initializers.ones, (self.hc_mult, self.emb_features), jnp.float32)
        key = kv[..., :self.hc_mult * self.emb_features].astype(jnp.float32).reshape(
            *kv.shape[:2], self.hc_mult, self.emb_features)
        value = kv[..., self.hc_mult * self.emb_features:].astype(jnp.float32)
        h = streams.astype(jnp.float32)
        rstd = (jax.lax.rsqrt(jnp.mean(jnp.square(h), -1) + self.norm_eps)
                * jax.lax.rsqrt(jnp.mean(jnp.square(key), -1) + self.norm_eps))
        dot = jnp.sum(h * (q_weight * k_weight) * key, -1) * rstd * self.emb_features ** -0.5
        gate = jax.nn.sigmoid(jnp.copysign(jnp.sqrt(jnp.maximum(jnp.abs(dot), 1e-6)), dot))
        if token_mask is not None:
            gate = jnp.where(token_mask[..., None], gate, 0.0)
        return (h + gate[..., None] * value[..., None, :]).astype(streams.dtype)


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
