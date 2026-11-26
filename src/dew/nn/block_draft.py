"""A block drafter over a target's fused taps, with a recurrent Markov head: Inkling 2's multi-token
prediction draft (the release's `mtp_config` and `mtp.safetensors`), as vLLM runs it (vllm-project/vllm,
model_executor/models/inkling2/nvidia/dspark.py, its ops/dspark.py kernels and its speculator.py).

The target records the residual each tapped visit of its layers is handed (`MarkovBlockDraft.taps`: a visit
is one run of a layer in the forward's order, a looped block's layers visited once a pass; under depth
residuals the partial block) and, as its last tap, its final norm's output. Each draft layer reads the taps
mixed by its own softmax weights and normed (`fused_taps`, vLLM's `mix_rmsnorm`), projected to its context
keys and values at each token's place. A draft is a block of `block_size` places after the context: the
drawn token (the anchor, embedded as the target's first layer reads it) and `block_size - 1` copies of a
learned mask embedding, through pre-norm layers whose queries read the whole block and the context before
it, a local layer its last `window` places alone. Each mask place's final states are scored by the target's
head beside a correction the Markov head adds: a gated cell over `markov_rank` features, stepped from zero
along the block, fed the token drafted before (the anchor first) and the place's states beside the mix of
the last context token's taps, its latent read out through a projection to the stream (`hidden_out`) and
the target's head. vLLM folds that projection into the head's rows once (`Inkling2MarkovHead.fold`); Dew
scores the corrected states, the head being linear in them.
"""

from __future__ import annotations

import dataclasses
import functools
from collections.abc import Callable, Sequence
from typing import NamedTuple

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.typing import Dtype, PrecisionLike

from dew.batch import prefix_sum
from dew.nn.attention import reference_attention
from dew.nn.dense import Dense
from dew.nn.kv_cache import (
    CURSOR,
    CacheSpec,
    KVCache,
    KVSpec,
    KVStore,
    Opened,
    cache_positions,
    cache_variables,
    compute_dtype,
    cursor,
    kept,
)
from dew.nn.norms import RMSNorm
from dew.nn.precision import at_least_fp32, float32_dot_general
from dew.nn.rope import rotary_freqs, rotate

CARRIED = 'taps'
"""The drafter's cache leaf holding each row's mix of its last context token's taps (the Markov head's
context, vLLM's `f_rnn`), under its holder (`Carried`)."""


class Draft(NamedTuple):
    """A drafting pass's block: `tokens` `[B, K + 1]` the drawn token then the K drafted, `logits` `[B, K,
    vocab]` the scores each drafted token was chosen from, and `confidence` `[B, K]` each one's acceptance
    score where the drafter has a confidence head (DSpark's), else None."""

    tokens: jax.Array
    logits: jax.Array
    confidence: jax.Array | None


@dataclasses.dataclass(frozen=True)
class MarkovBlockDraft:
    """The drafter, by the release's `mtp_config`: `layers` (num_hidden_layers) draft layers of `num_heads`
    query and `num_kv_heads` key heads `head_dim` wide under a NeoX rotary at `rope_theta`, their gated MLPs
    `mlp_features` wide, their norms at `norm_eps`; `local_layers` (local_layer_ids) reading their last
    `window` (sliding_window_size) context places, the others all; a block of `block_size` places, the
    anchor and its masks; `taps` (target_layer_ids), the target's visits it reads in order, the last the
    target's visit count, standing for its final norm's output; and a Markov head of `markov_rank`."""

    layers: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    mlp_features: int
    block_size: int
    taps: tuple[int, ...]
    markov_rank: int
    rope_theta: float
    norm_eps: float
    local_layers: tuple[int, ...] = ()
    window: int | None = None

    def __post_init__(self):
        object.__setattr__(self, 'taps', tuple(int(tap) for tap in self.taps))
        object.__setattr__(self, 'local_layers', tuple(int(layer) for layer in self.local_layers))
        if self.layers < 1 or self.block_size < 2 or len(self.taps) < 2 or self.markov_rank < 1:
            raise ValueError("a Markov block drafter drafts a block of the drawn token and a mask or more "
                             "through a layer or more, from a target visit or more beside the target's final "
                             "states, through a Markov head of a rank of one or more")
        if list(self.taps) != sorted(set(self.taps)) or self.taps[0] < 0:
            raise ValueError(f"a block drafter's taps are distinct visits in the target's order, got "
                             f"{self.taps}")
        outside = sorted(set(self.local_layers) - set(range(self.layers)))
        if outside:
            raise ValueError(f"the block drafter's local layers {outside} are outside its {self.layers}")
        if bool(self.local_layers) != (self.window is not None and self.window > 0):
            raise ValueError("a block drafter's local layers read a window of places, and its window is "
                             "its local layers'")
        if self.num_heads % self.num_kv_heads or self.head_dim % 2:
            raise ValueError("the block drafter's query heads group over its key heads, each rotated in "
                             "pairs of channels")

    @property
    def visits(self) -> tuple[int, ...]:
        """The tapped visits whose residual the target records; the last tap is its final states."""
        return self.taps[:-1]


def tapped(order: Sequence[int], visits: Sequence[int]) -> tuple[tuple[int, int], ...]:
    """Each of `visits`' layer and its run among that layer's, by `order`, the layer each visit runs
    (`CausalTransformer.visits`): where its record lies among those the layer appends."""
    return tuple((order[visit], order[:visit].count(order[visit])) for visit in visits)


def fused_taps(taps, weights, norm: nn.Module):
    """The target's taps `[..., T * D]` summed by `weights` `[T]` and normed (`norm`), in at least fp32
    and rounded once to the taps' dtype: vLLM's `mix_rmsnorm` (ops/dspark.py)."""
    wide = at_least_fp32(taps.dtype)
    split = taps.reshape(*taps.shape[:-1], weights.shape[0], -1).astype(wide)
    return norm(jnp.sum(split * weights.astype(wide)[:, None], axis=-2)).astype(taps.dtype)


class Context(NamedTuple):
    """A draft layer's context: its keys and values `[B, M, kv_heads, head_dim]`, each slot's place and
    whether it holds a token `[B, M]`, and each row's count of context tokens `[B]`, the anchor's place."""

    keys: jax.Array
    values: jax.Array
    places: jax.Array
    real: jax.Array
    count: jax.Array


class ContextStore(nn.Module):
    """A draft layer's cached context: each row's keys and values in a store of its own (`KVSpec`), its
    cursor each row's count of context tokens."""

    kv_cache: KVCache
    capacity: int
    kv_heads: int
    head_dim: int

    @nn.nowrap
    def spec(self, rows: int, dtype: Dtype) -> KVSpec:
        return KVSpec(self.kv_cache, rows, self.capacity, self.kv_heads, self.head_dim, dtype)

    @nn.nowrap
    def cache_spec(self, rows: int, dtype: Dtype) -> CacheSpec:
        """The store for `rows` rows of `dtype`, with its cursor."""
        return {**self.spec(rows, dtype).leaves(), **cursor(rows)}

    @nn.compact
    def __call__(self, mixed, valid, rows: int, dtype: Dtype, project: Callable) -> Context:
        """The context of `rows` rows (of `dtype`, the drafter's) after writing `mixed`'s real tokens
        (`valid`, every one where None) after those the store holds, each projected at its count
        (`project(mixed, places)`, its keys and values); `mixed` None writes none."""
        store = KVStore.open(self, self.spec(rows, dtype))
        if mixed is not None:
            slots = cache_positions(self, rows, mixed.shape[1], valid)
            store.write(*project(mixed, jnp.maximum(slots, 0)), slots)
        else:
            cache_variables(self, cursor(rows))
        keys, values = store.read()
        count = self.get_variable('cache', CURSOR)
        places = jnp.broadcast_to(jnp.arange(keys.shape[1])[None], (rows, keys.shape[1]))
        return Context(store.plain(keys), values, places, places < count[:, None], count)


class Carried(nn.Module):
    """The Markov head's carried taps `[rows, D]` a cache holds (`CARRIED`): each row's last real context
    token's mix."""

    features: int

    @nn.nowrap
    def cache_spec(self, rows: int, dtype: Dtype) -> CacheSpec:
        return {CARRIED: Opened((rows, self.features), dtype)}

    @nn.compact
    def __call__(self, fresh, real, rows: int, dtype: Dtype) -> jax.Array:
        """The carried taps (of `dtype`, the drafter's) after writing `fresh` `[B, D]` where `real` `[B]`
        holds (a row with no real token keeps what it carried); `fresh` None writes none."""
        held = cache_variables(self, self.cache_spec(rows, dtype))[CARRIED]
        if fresh is not None:
            kept(held, jnp.where(real[:, None], fresh.astype(dtype), held.value), CARRIED)
        return held.value


class DraftAttention(nn.Module):
    """A draft layer's attention (vLLM's `Inkling2DraftSelfAttention`): queries and keys normed per head
    and rotated at their places, reading the block both ways and the context's keys and values before it,
    projected from the layer's mix of the target's taps (`kv_proj_ctx`, its keys normed apart); cached, each
    row's context in a store of its own (`held`), uncached as given (`given`)."""

    spec: MarkovBlockDraft
    emb_features: int
    local: bool
    kv_cache: KVCache
    max_seq_len: int
    dtype: Dtype | None = None
    precision: PrecisionLike = None

    def setup(self):
        spec = self.spec
        dense = functools.partial(Dense, use_bias=False, dtype=self.dtype, precision=self.precision)
        keys = spec.num_kv_heads * spec.head_dim
        self.q_proj = dense(spec.num_heads * spec.head_dim)
        self.k_proj, self.v_proj = dense(keys), dense(keys)
        self.k_proj_ctx, self.v_proj_ctx = dense(keys), dense(keys)
        self.o_proj = dense(self.emb_features)
        norm = functools.partial(RMSNorm, epsilon=spec.norm_eps, dtype=self.dtype)
        self.q_norm, self.k_norm, self.k_norm_ctx = norm(), norm(), norm()
        self.held_context = ContextStore(self.kv_cache, self.max_seq_len, spec.num_kv_heads, spec.head_dim)

    def _heads(self, x, count: int):
        return x.reshape(*x.shape[:-1], count, self.spec.head_dim)

    def _rotated(self, x, places):
        cos, sin = rotary_freqs(places, self.spec.head_dim, self.spec.rope_theta,
                                dtype=at_least_fp32(x.dtype))
        return rotate(x, cos, sin)

    def context(self, mixed, places):
        """The context's rotated keys and values `[B, M, kv_heads, head_dim]` from the layer's mix of the
        target's taps `mixed` `[B, M, D]` at `places` `[B, M]` (`store_context_kv`)."""
        heads = self.spec.num_kv_heads
        keys = self._rotated(self.k_norm_ctx(self._heads(self.k_proj_ctx(mixed), heads)), places)
        return keys, self._heads(self.v_proj_ctx(mixed), heads)

    def held(self, mixed, valid, rows: int, dtype: Dtype) -> Context:
        """The cached context of `rows` rows of `dtype` after writing `mixed`'s real tokens (`valid`, every
        one where None) after those the store holds, each at its count (`ContextStore`); `mixed` None writes
        none."""
        return self.held_context(mixed, valid, rows, dtype, self.context)

    def given(self, mixed, valid) -> Context:
        """The context `mixed` uncached, each real token (`valid`) at its rank among its row's, as a cache
        holds it."""
        rows, length = mixed.shape[:2]
        real = jnp.ones((rows, length), bool) if valid is None else jnp.asarray(valid, bool)
        places = prefix_sum(real, dtype=jnp.int32) - 1
        keys, values = self.context(mixed, jnp.maximum(places, 0))
        return Context(keys, values, places, real, jnp.sum(real, axis=1, dtype=jnp.int32))

    def __call__(self, x, context: Context):
        """The block `x` `[B, K, D]` at the places after `context`'s, read both ways beside it."""
        spec = self.spec
        rows, length = x.shape[:2]
        anchor = context.count[:, None]
        places = anchor + jnp.arange(length)[None]
        query = self._rotated(self.q_norm(self._heads(self.q_proj(x), spec.num_heads)), places)
        keys = self._rotated(self.k_norm(self._heads(self.k_proj(x), spec.num_kv_heads)), places)
        values = self._heads(self.v_proj(x), spec.num_kv_heads)
        seen = context.real & (context.places < anchor)
        if self.local:
            assert spec.window is not None
            seen = seen & (context.places >= anchor - spec.window)
        allowed = jnp.concatenate([jnp.broadcast_to(seen[:, None], (rows, length, seen.shape[1])),
                                   jnp.ones((rows, length, length), bool)], axis=-1)
        attended = reference_attention(query, jnp.concatenate([context.keys, keys], axis=1),
                                       jnp.concatenate([context.values, values], axis=1),
                                       mask=allowed[:, None], precision=self.precision)
        return self.o_proj(attended.reshape(rows, length, spec.num_heads * spec.head_dim))


class DraftMLP(nn.Module):
    """A draft layer's gated MLP: `down(silu(gate(x)) * up(x))`, its product in at least fp32 rounded once
    (vLLM's `silu_and_mul`, its sigmoid `gate / (1 + exp(-gate))`)."""

    features: int
    emb_features: int
    dtype: Dtype | None = None
    precision: PrecisionLike = None

    def setup(self):
        dense = functools.partial(Dense, use_bias=False, dtype=self.dtype, precision=self.precision)
        self.gate_proj, self.up_proj = dense(self.features), dense(self.features)
        self.down_proj = dense(self.emb_features)

    def __call__(self, x):
        gate, up = self.gate_proj(x), self.up_proj(x)
        wide = at_least_fp32(gate.dtype)
        gated = gate.astype(wide) / (1.0 + jnp.exp(-gate.astype(wide)))
        return self.down_proj((gated * up.astype(wide)).astype(gate.dtype))


class DraftLayer(nn.Module):
    """A pre-norm draft layer (vLLM's `Inkling2DraftLayer`): its norm before the attention takes the
    stream's sum with the layer before's output, as before its MLP (`RMSNorm.seam`); its taps' norm
    (`dflare_norm`) normalizes its mix of them."""

    spec: MarkovBlockDraft
    emb_features: int
    local: bool
    kv_cache: KVCache
    max_seq_len: int
    dtype: Dtype | None = None
    precision: PrecisionLike = None

    def setup(self):
        spec = self.spec
        norm = functools.partial(RMSNorm, epsilon=spec.norm_eps, dtype=self.dtype)
        self.input_layernorm, self.post_attention_layernorm = norm(), norm()
        # The mix's norm in its own wide arithmetic, rounded to the taps' dtype after (`fused_taps`).
        self.dflare_norm = RMSNorm(epsilon=spec.norm_eps)
        self.self_attn = DraftAttention(spec, self.emb_features, self.local, self.kv_cache, self.max_seq_len,
                                        dtype=self.dtype, precision=self.precision)
        self.mlp = DraftMLP(spec.mlp_features, self.emb_features, dtype=self.dtype, precision=self.precision)

    def __call__(self, x, residual, context: Context):
        """The layer's MLP output and the stream before it, from the layer before's output `x` and its
        stream `residual` (None into the first layer, whose stream `x` is)."""
        residual, normed = ((x, self.input_layernorm(x)) if residual is None
                            else self.input_layernorm.seam(residual, x))
        residual, normed = self.post_attention_layernorm.seam(residual, self.self_attn(normed, context))
        return self.mlp(normed), residual


class MarkovHead(nn.Module):
    """The recurrent correction (vLLM's `Inkling2MarkovHead`, the release's `rnn_hidden`): its `rnn_proj`
    `[3r, 2r + 2D]` read as its step's columns (`rnn_step`, over the state and the token before's
    embedding `markov_w1`) and its context's (`rnn_context`, over a place's states and the carried taps,
    with the bias), each product accumulated and kept in at least fp32; the gate, candidate and output each
    rounded to the states' dtype, and every op of the cell after them, as the reference's torch ops round
    (`markov_gate`); its latent read out through `hidden_out` to the stream."""

    rank: int
    rows: int
    emb_features: int
    norm_eps: float
    dtype: Dtype | None = None
    precision: PrecisionLike = None

    def setup(self):
        wide = at_least_fp32(self.dtype)
        self.markov_w1 = self.param('markov_w1', nn.initializers.normal(1.0), (self.rows, self.rank),
                                    jnp.float32)
        self.rnn_step = Dense(3 * self.rank, use_bias=False, dtype=wide, precision=self.precision,
                              dot_general=float32_dot_general)
        self.rnn_context = Dense(3 * self.rank, dtype=wide, precision=self.precision,
                                 dot_general=float32_dot_general)
        self.hidden_out = Dense(self.emb_features, dtype=self.dtype, precision=self.precision)
        self.dflare_norm = RMSNorm(epsilon=self.norm_eps)

    def context(self, hidden, carried):
        """The context's part of each step's gates `[B, K, 3r]` (`Inkling2MarkovHead.context`): over each
        place's states `hidden` `[B, K, D]` beside the carried taps `carried` `[B, D]`."""
        return self.rnn_context(jnp.concatenate([hidden, jnp.broadcast_to(carried[:, None], hidden.shape)],
                                                axis=-1))

    def step(self, state, previous, context):
        """One step of the cell from `state` `[B, r]` after the token `previous` `[B]` under its place's
        `context` `[B, 3r]`: the next state and its correction to the place's states `[B, D]`."""
        narrow, wide = state.dtype, at_least_fp32(state.dtype)

        def rounded(x):
            return x.astype(narrow).astype(wide)

        embedded = jnp.take(self.markov_w1, previous, axis=0).astype(narrow)
        raw = context + self.rnn_step(jnp.concatenate([state, embedded], axis=-1))
        gate, candidate, out = jnp.split(rounded(raw), 3, axis=-1)
        gate, candidate = rounded(jax.nn.sigmoid(gate)), rounded(jnp.tanh(candidate))
        kept_state = rounded(gate * state.astype(wide))
        update = rounded(rounded(1.0 - gate) * candidate)
        return (kept_state + update).astype(narrow), self.hidden_out(jnp.tanh(out).astype(narrow))


class BlockDrafter(nn.Module):
    """The drafter's layers, final norm, Markov head, the taps' mixing weights (`fuse_weight`, a row a
    layer and the Markov head's last) and the mask embedding (`mask_embed`), as the release's
    `mtp.safetensors` holds them; the target's embedding and head are the drafter's (`draft`)."""

    spec: MarkovBlockDraft
    emb_features: int
    vocab_rows: int
    kv_cache: KVCache
    max_seq_len: int
    dtype: Dtype | None = None
    precision: PrecisionLike = None

    def setup(self):
        spec = self.spec
        self.layers = [DraftLayer(spec, self.emb_features, index in spec.local_layers, self.kv_cache,
                                  self.max_seq_len, dtype=self.dtype, precision=self.precision)
                       for index in range(spec.layers)]
        self.norm = RMSNorm(epsilon=spec.norm_eps, dtype=self.dtype)
        self.markov_head = MarkovHead(spec.markov_rank, self.vocab_rows, self.emb_features, spec.norm_eps,
                                      dtype=self.dtype, precision=self.precision)
        self.fuse_weight = self.param('fuse_weight', nn.initializers.zeros, (spec.layers + 1, len(spec.taps)),
                                      jnp.float32)
        self.mask_embed = self.param('mask_embed', nn.initializers.normal(0.02), (self.emb_features,),
                                     jnp.float32)
        self.carried = Carried(self.emb_features)

    @nn.nowrap
    def cache_spec(self, rows: int) -> CacheSpec:
        """Each layer's context store for `rows` rows (`ContextStore`), and each row's carried taps
        (`Carried`), each under its holder's name."""
        spec, dtype = self.spec, compute_dtype(self, self.dtype)
        store = ContextStore(self.kv_cache, self.max_seq_len, spec.num_kv_heads, spec.head_dim, parent=None)
        carried = Carried(self.emb_features, parent=None)
        return {**{f'layers_{index}': {'self_attn': {'held_context': store.cache_spec(rows, dtype)}}
                   for index in range(spec.layers)},
                'carried': carried.cache_spec(rows, dtype)}

    def contexts(self, taps, valid, rows: int, decode: bool) -> tuple[list[Context], jax.Array]:
        """Each layer's context and the Markov head's carried taps `[B, D]`: cached, after writing the
        real tokens of `taps` `[B, M, T * D]` (`valid`, every one where None; None writes none) and carrying
        each row's last; uncached, of `taps` as given."""
        weights = jax.nn.softmax(self.fuse_weight.astype(at_least_fp32(self.fuse_weight.dtype)), axis=-1)
        # The drafter's activations' dtype, which its cache keeps them in (`cache_spec`).
        dtype = compute_dtype(self, self.dtype)
        if not decode and taps is None:
            raise ValueError("uncached, a block drafter reads its whole context, given")
        contexts = []
        for index, layer in enumerate(self.layers):
            mixed = None if taps is None else fused_taps(taps, weights[index], layer.dflare_norm)
            contexts.append(layer.self_attn.held(mixed, valid, rows, dtype) if decode
                            else layer.self_attn.given(mixed, valid))
        carried = None if taps is None else self._carried(taps, valid, weights[-1])
        if not decode:
            assert carried is not None
            fresh, real = carried
            return contexts, jnp.where(real[:, None], fresh, jnp.zeros_like(fresh))
        fresh, real = (None, None) if carried is None else carried
        return contexts, self.carried(fresh, real, rows, dtype)

    def _carried(self, taps, valid, weights) -> tuple[jax.Array, jax.Array]:
        """Each row's last real context token's taps mixed for the Markov head (`compute_f_rnn` over vLLM's
        `prepare_context_anchor` rows) `[B, D]`, and whether the row has any real token `[B]`."""
        rows, length = taps.shape[:2]
        real = jnp.ones((rows, length), bool) if valid is None else jnp.asarray(valid, bool)
        at = length - 1 - jnp.argmax(real[:, ::-1], axis=1)
        return (fused_taps(taps[jnp.arange(rows), at], weights, self.markov_head.dflare_norm),
                jnp.any(real, axis=1))

    def states(self, embedded, contexts: Sequence[Context]):
        """The block's final states `[B, K, D]` from the anchor's embedding `embedded` `[B, D]` and the
        mask embedding after it, through every layer over its context."""
        rows, width = embedded.shape
        x = jnp.concatenate([embedded[:, None], jnp.broadcast_to(self.mask_embed.astype(embedded.dtype),
                                                                 (rows, self.spec.block_size - 1, width))],
                            axis=1)
        residual = None
        for layer, context in zip(self.layers, contexts, strict=True):
            x, residual = layer(x, residual, context)
        return self.norm.seam(residual, x)[1]

    def markov_context(self, hidden, carried):
        """The Markov head's context part of every step (`MarkovHead.context`)."""
        return self.markov_head.context(hidden, carried)

    def markov_step(self, state, previous, context):
        """One step of the Markov head (`MarkovHead.step`)."""
        return self.markov_head.step(state, previous, context)


def draft(drafter: BlockDrafter, taps, valid, tokens, embed: Callable, logits_of: Callable, *, decode: bool,
          choose: Callable | None = None) -> Draft | None:
    """One drafting pass of `drafter` (vLLM's `Inkling2DSparkSpeculator`): its context from `taps` `[B, M,
    T * D]`, the real tokens `valid` marks (every one where None), written to its cache where `decode` (None
    drafts after what the cache holds), and with `tokens` `[B]` drawn after it a block drafted, each token
    chosen by `choose(index, logits)` or greedily where None, its logits the target's head (`logits_of`) over
    the place's states and its correction; `tokens` None writes the context alone and returns None. `embed`
    embeds the anchor as the target's first layer reads it."""
    rows = (tokens if tokens is not None else taps).shape[0]
    contexts, carried = drafter.contexts(taps, valid, rows, decode)
    if tokens is None:
        return None
    hidden = drafter.states(embed(tokens), contexts)[:, 1:]
    context = drafter.markov_context(hidden, carried)
    state = jnp.zeros((rows, drafter.spec.markov_rank), hidden.dtype)
    wide = at_least_fp32(hidden.dtype)
    drafted, scored = [tokens.astype(jnp.int32)], []
    for index in range(drafter.spec.block_size - 1):
        state, correction = drafter.markov_step(state, drafted[-1], context[:, index])
        logits = logits_of(hidden[:, index].astype(wide) + correction.astype(wide))
        chosen = jnp.argmax(logits, axis=-1) if choose is None else choose(index, logits)
        drafted.append(chosen.astype(jnp.int32))
        scored.append(logits)
    return Draft(jnp.stack(drafted, 1), jnp.stack(scored, 1), None)
