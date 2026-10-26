"""In-context memory: a layer whose inner network's parameters update over the sequence (ledger 1645).

A token's views of the stream, `k_t`, `v_t`, `q_t`, train the inner network `f(W; .)` on an objective
of its own, and the token reads the updated network:

    g_t = grad_W objective(f(W_t'; k_t), target_t)      gradients at the chunk's start state W_t'
    S_t = mu_t S_{t-1} - theta_t g_t                     momentum (mu = 0 without it)
    W_t = a_t W_{t-1} + S_t                              retention (a = 1 - alpha; 1 without it)
    y_t = f(W_t; q_t)

(Titans, arXiv 2501.00663, Eq 8-15; TTT, arXiv 2407.04620, Eq 2-6; Miras, arXiv 2504.13173, Eq 10, 13,
28; Nested Learning, arXiv 2512.24695, Eq 71, 84-90.) The target is the value view (`Given`) or the
network's own image of it at the chunk's start, `f(W_t'; v_t)` (`SelfGenerated`, Nested Learning
Eq 84). A chunk of `chunk` tokens takes every gradient at the state it starts from, counted from its
document's first real token; `chunk=1` is online gradient descent. A document's first token starts
from W_0, the network's meta-learned parameters, with no momentum.

The state (W, S and the chunk's start) is held in at least float32 whatever the compute dtype, and
the inner gradients and the scans run there: a bfloat16 state drops every step below its ulp. The
readout runs the network at the layer's dtype over the state cast to it, so at a zero step size a
network reads exactly as its W_0 does.

Two computations give the one function. A row of one document runs chunk by chunk: its real tokens
compacted to the front (`valid_order`), each chunk's gradients one `vmap` at the chunk's start, the
momentum and retention across the chunk one associative scan of their affine steps, and the chunk's
readouts one `vmap` over its states. A packed row, and a cached call, run token by token (`_stepped`),
which is the definition. The state is the network's parameters, a traced carry: the network is
applied unbound (`network.apply({"params": W}, x)`) and W_0 is this layer's own parameter, so the layer
composes with a scanned layer stack and remat; the chunk's carry is named (`CARRY`) for a remat policy.
"""

from __future__ import annotations

import dataclasses
import functools
from typing import ClassVar, Literal, NamedTuple

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.typing import Dtype, PrecisionLike
from jax.ad_checkpoint import checkpoint_name

from dew.nn.dense import Dense
from dew.optim import newton_schulz

from .attention import LayerNorm
from .inputs import PLAIN_CALL, MixerCall, coordinates_of, counted_places, valid_order
from .precision import at_least_fp32, l2norm

CARRY = "memory_carry"
"""The name of the state a chunk hands the next (`checkpoint_name`), for a remat policy to save."""

STATE, MOMENTUM, START, COUNT = "memory_state", "memory_momentum", "memory_start", "memory_count"
"""The decode cache's variables: the network's parameters now, the momentum, the parameters at the
chunk's start, and each row's real tokens seen, rows first."""


@dataclasses.dataclass(frozen=True)
class Objective:
    """The loss a token's prediction `f(W; k)` takes against its target, summed over the features:
    `dot`, -<p, v> (linear attention's, Nested Learning Eq 17); `l2`, 1/2 ||p - v||^2 (Titans Eq 12
    with Appendix C's 1/2); `lp`, ||p - v||_p^p (Miras Eq 10); `huber`, Huber's loss of each feature's
    error at `delta` (Miras Eq 13)."""

    kind: Literal["dot", "l2", "lp", "huber"] = "l2"
    p: float = 3.0
    delta: float = 1.0

    def __post_init__(self):
        if self.kind not in ("dot", "l2", "lp", "huber"):
            raise ValueError(f"an objective is dot, l2, lp or huber, not {self.kind!r}")
        if self.kind == "lp" and self.p < 1:
            raise ValueError(f"an lp objective takes p >= 1, got {self.p}")
        if self.kind == "huber" and self.delta <= 0:
            raise ValueError(f"a huber objective takes a positive delta, got {self.delta}")

    def __call__(self, prediction: jax.Array, target: jax.Array) -> jax.Array:
        error = prediction - target
        if self.kind == "dot":
            return -jnp.sum(prediction * target, axis=-1)
        if self.kind == "l2":
            return 0.5 * jnp.sum(jnp.square(error), axis=-1)
        if self.kind == "lp":
            return jnp.sum(jnp.abs(error) ** self.p, axis=-1)
        size = jnp.abs(error)
        return jnp.sum(jnp.where(size <= self.delta, 0.5 * jnp.square(error),
                                 self.delta * (size - 0.5 * self.delta)), axis=-1)


@dataclasses.dataclass(frozen=True)
class Gate:
    """A per-token scalar a head: `value` itself, or, without one, `scale * sigmoid(x w + b)` of the
    token's stream (TTT's eta(x) = eta_base sigmoid(theta_lr x), Sec 2.7), `b` starting at `bias`."""

    value: float | None = None
    scale: float = 1.0
    bias: float = 0.0


class LinearMemory(nn.Module):
    """`W k`, `W` square over the view's width: a linear memory (TTT-Linear's f_lin without its norm and
    residual, Titans' linear case)."""

    dtype: Dtype | None = None
    precision: PrecisionLike = None

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        return Dense(x.shape[-1], use_bias=False, dtype=self.dtype, precision=self.precision,
                     name="kernel")(x)


class ResidualMemory(nn.Module):
    """`x + N(f_res(x))`: f_res linear (`hidden` None) or `Dense(D) . activation . Dense(hidden)`, N a
    layer norm or none. TTT's f (Sec 2.7: LN and residual, f_lin or a 4x GELU MLP), Nested Learning's
    memories (Eq 89, no norm) and Miras' (Sec 5.3)."""

    hidden: int | None = None
    activation: Literal["gelu", "silu", "relu"] = "gelu"
    norm: bool = True
    dtype: Dtype | None = None
    precision: PrecisionLike = None

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        dense = functools.partial(Dense, dtype=self.dtype, precision=self.precision)
        branch = x
        if self.hidden is not None:
            branch = getattr(nn, self.activation)(dense(self.hidden, name="up")(branch))
        branch = dense(x.shape[-1], name="down")(branch)
        if self.norm:
            branch = LayerNorm(epsilon=1e-6, dtype=self.dtype, name="norm")(branch)
        return x + branch


class MLPMemory(nn.Module):
    """Titans' deep memory: dense layers of the `hidden` widths, then one back to the view's width,
    `activation` between them (Sec 3.1)."""

    hidden: tuple[int, ...]
    activation: Literal["gelu", "silu", "relu"] = "silu"
    dtype: Dtype | None = None
    precision: PrecisionLike = None

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        for index, width in enumerate((*self.hidden, x.shape[-1])):
            if index:
                x = getattr(nn, self.activation)(x)
            x = Dense(width, dtype=self.dtype, precision=self.precision, name=f"layers_{index}")(x)
        return x


class InContextMemory(nn.Module):
    """The in-context memory layer `[B, S, D] -> [B, S, D]` (the module docstring).

    `network` is the inner network, any module that maps one vector of the view's width to one of the
    same width and takes a `dtype` (`LinearMemory`, `ResidualMemory`, `MLPMemory`, an existing block
    such as `GatedMLP`); it runs unbound, at the dtype each use asks, and its parameters, drawn at
    init, are W_0. `heads` memories run side by side over the views split into heads of
    `head_features`. A view is projected from the stream, or the stream itself where its `project_*`
    is off, as a continuum-memory level reads its block's input (one head). `gates` are the step size
    theta (`learning_rate`), the momentum mu (None: none) and the retention alpha (None: none).
    `output` projects the heads' readouts back to the stream's width.
    """

    network: nn.Module
    heads: int = 1
    head_features: int | None = None
    objective: Objective = Objective()
    target: Literal["given", "self_generated"] = "given"
    learning_rate: Gate = Gate()
    momentum: Gate | None = None
    retention: Gate | None = None
    chunk: int = 16
    orthogonalize: int = 0
    """Newton-Schulz steps orthogonalizing each matrix of a token's gradient before its step (`dew.optim.
    newton_schulz`, Nested Learning Sec 4.2's Muon-style inner update); 0 steps the gradient as it is.
    Vectors (biases, norm scales) step as they are."""
    project_query: bool = True
    project_key: bool = True
    project_value: bool = True
    normalize: bool = False
    output: bool = True
    dtype: Dtype | None = None
    precision: PrecisionLike = None

    reads: ClassVar[frozenset[str]] = frozenset()

    def __post_init__(self):
        if self.chunk < 1:
            raise ValueError(f"a memory updates every chunk of one token or more, got {self.chunk}")
        if self.orthogonalize < 0:
            raise ValueError(f"orthogonalize counts Newton-Schulz steps, got {self.orthogonalize}")
        if self.target not in ("given", "self_generated"):
            raise ValueError(f"a memory's target is given or self_generated, not {self.target!r}")
        if self.heads > 1 and not (self.project_query and self.project_key and self.project_value):
            raise ValueError("a view read as the stream itself is one head's; project it to split heads")
        super().__post_init__()

    def _inner(self, *, wide: bool) -> nn.Module:
        """The inner network, at the state's precision where `wide` (the gradients), else the layer's."""
        return self.network.clone(dtype=at_least_fp32(self.dtype) if wide else self.dtype, parent=None)

    def _views(self, x: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array, int]:
        """The queries, keys and values `[B, S, H, F]` and the views' width F."""
        features = (self.head_features or x.shape[-1] // self.heads) if self.project_key else x.shape[-1]

        def view(name: str, projected: bool) -> jax.Array:
            if not projected:
                return x[:, :, None, :]
            out = Dense(self.heads * features, use_bias=False, dtype=self.dtype, precision=self.precision,
                        name=name)(x)
            return out.reshape(*x.shape[:2], self.heads, features)

        query, key, value = (view(name, projected) for name, projected in (
            ("q_proj", self.project_query), ("k_proj", self.project_key), ("v_proj", self.project_value)))
        if self.normalize:
            query, key = l2norm(query), l2norm(key)
        return query, key, value, features

    def _gate(self, name: str, gate: Gate, x: jax.Array) -> jax.Array:
        """The gate's value a token and head `[B, S, H]`, in at least float32."""
        wide = at_least_fp32(self.dtype)
        if gate.value is not None:
            return jnp.full((*x.shape[:2], self.heads), gate.value, wide)
        logit = Dense(self.heads, dtype=self.dtype, precision=self.precision, name=name,
                      bias_init=nn.initializers.constant(gate.bias))(x)
        return gate.scale * jax.nn.sigmoid(logit.astype(wide))

    def _initial(self, features: int) -> dict:
        """W_0, a head's network parameters stacked over the heads `[H, ...]`."""
        network = self._inner(wide=False)

        def drawn(key: jax.Array) -> dict:
            keys = jax.random.split(key, self.heads)
            example = jnp.zeros((1, features), jnp.float32)
            return jax.vmap(lambda one: dict(network.init(one, example)["params"]))(keys)

        return self.param("initial", drawn)

    @nn.compact
    def __call__(self, x: jax.Array, call: MixerCall = PLAIN_CALL) -> jax.Array:
        batch, length, _ = x.shape
        query, key, value, features = self._views(x)
        theta = self._gate("learning_rate", self.learning_rate, x)
        mu = (jnp.zeros_like(theta) if self.momentum is None else self._gate("momentum", self.momentum, x))
        a = (jnp.ones_like(theta) if self.retention is None
             else 1 - self._gate("retention", self.retention, x))
        initial = self._initial(features)
        wide = at_least_fp32(self.dtype)
        initial_wide = jax.tree.map(lambda leaf: leaf.astype(wide), initial)
        tokens = coordinates_of(call.metadata, batch, length)
        real = jnp.ones((batch, length), bool) if tokens.valid is None else jnp.asarray(tokens.valid, bool)
        if tokens.segments is not None:
            real = real & (tokens.segments != 0)
        places = counted_places(tokens.segments, tokens.valid, tokens.positions)
        places = jnp.broadcast_to(jnp.arange(length)[None], (batch, length)) if places is None else places
        firsts = (real & (jnp.cumsum(real, axis=1) == 1)) | (
            jnp.zeros_like(real) if tokens.starts is None else jnp.asarray(tokens.starts, bool) & real)
        steps = _Steps(query.astype(wide), key.astype(wide), value.astype(wide), theta, mu, a, real,
                       firsts, places % self.chunk == 0)
        if call.decode:
            read = self._decoded(initial_wide, steps)
        elif tokens.max_segments > 1:
            read = self._stepped(initial_wide, steps)[0]
        else:
            read = self._chunked(initial_wide, steps)
        read = read.astype(self.dtype or x.dtype)
        if self.output:
            read = Dense(x.shape[-1], use_bias=False, dtype=self.dtype, precision=self.precision,
                         name="o_proj")(read.reshape(batch, length, -1))
        else:
            read = read.reshape(batch, length, -1)
        return read

    # The function, token by token and chunk by chunk.

    def _applied(self, state: dict, x: jax.Array, *, wide: bool) -> jax.Array:
        """The inner network at parameters `state` over one token's vector `x`."""
        out = self._inner(wide=wide).apply({"params": state}, x)
        if not isinstance(out, jax.Array):
            raise TypeError(f"an in-context memory's network maps a vector to one, got {type(out).__name__}")
        return out

    def _gradient(self, state: dict, key: jax.Array, target: jax.Array) -> dict:
        """One head's gradient of the objective at `state` for one token, in the state's precision, each
        matrix orthogonalized where `orthogonalize` asks."""
        gradient = jax.grad(lambda held: self.objective(self._applied(held, key, wide=True), target))(state)
        if not self.orthogonalize:
            return gradient
        return jax.tree.map(lambda leaf: newton_schulz(leaf, self.orthogonalize) if leaf.ndim == 2 else leaf,
                            gradient)

    def _target(self, state: dict, value: jax.Array) -> jax.Array:
        """A token's target: its value view, or the network's image of it at `state` (Nested Learning
        Eq 84), held fixed in the inner gradient."""
        if self.target == "given":
            return value
        return jax.lax.stop_gradient(self._applied(state, value, wide=True))

    def _read(self, state: dict, query: jax.Array) -> jax.Array:
        """One head's readout of a token at `state`, at the layer's dtype."""
        dtype = self.dtype or query.dtype
        cast = jax.tree.map(lambda leaf: leaf.astype(dtype) if self.dtype is not None else leaf, state)
        return self._applied(cast, query.astype(dtype), wide=False)

    def _token(self, carry: tuple, step: tuple) -> tuple:
        """One token's update and readout over every row and head: the definition."""
        start, state, momentum = carry
        query, key, value, theta, mu, a, real, first, boundary, initial = step

        def reset(fresh: dict, held: dict) -> dict:
            return jax.tree.map(lambda f, h: jnp.where(_lead(first, h), f, h), fresh, held)

        def keep(new: dict, old: dict) -> dict:
            return jax.tree.map(lambda n, o: jnp.where(_lead(real, n), n, o), new, old)

        state = reset(initial, state)
        momentum = reset(jax.tree.map(jnp.zeros_like, initial), momentum)
        start = reset(initial, jax.tree.map(lambda s, w: jnp.where(_lead(boundary, s), w, s), start, state))
        per_head = jax.vmap(jax.vmap(lambda held, k, v: self._gradient(held, k, self._target(held, v))))
        gradient = per_head(start, key, value)
        new_momentum = jax.tree.map(lambda s, g: _lead(mu, s) * s - _lead(theta, g) * g, momentum, gradient)
        new_state = jax.tree.map(lambda w, s: _lead(a, w) * w + s, state, new_momentum)
        state, momentum = keep(new_state, state), keep(new_momentum, momentum)
        read = jax.vmap(jax.vmap(self._read))(state, query)
        return (start, state, momentum), read

    def _stepped(self, initial: dict, steps: _Steps, carry: tuple | None = None) -> tuple[jax.Array, tuple]:
        """The readouts `[B, S, H, F]` token by token, and the carry after the last token."""
        batch = steps.real.shape[0]
        rows = jax.tree.map(lambda leaf: jnp.broadcast_to(leaf, (batch, *leaf.shape)), initial)
        if carry is None:
            carry = (rows, rows, jax.tree.map(jnp.zeros_like, rows))
        sequence = tuple(jnp.moveaxis(field, 1, 0) for field in steps)
        carry, reads = jax.lax.scan(lambda held, step: self._token(held, (*step, rows)), carry, sequence)
        return jnp.moveaxis(reads, 0, 1), carry

    def _chunked(self, initial: dict, steps: _Steps) -> jax.Array:
        """The readouts `[B, S, H, F]` of rows of one document, chunk by chunk (the module docstring)."""
        batch, length = steps.real.shape
        order, _ = valid_order(steps.real)
        gathered = _Steps(*(jnp.take_along_axis(field, order.reshape(*order.shape, *(1,) * (field.ndim - 2)),
                                                axis=1) for field in steps))
        count = -(-length // self.chunk)
        padding = count * self.chunk - length

        def chunks(field: jax.Array) -> jax.Array:
            padded = jnp.pad(field, ((0, 0), (0, padding)) + ((0, 0),) * (field.ndim - 2))
            return jnp.moveaxis(padded.reshape(batch, count, self.chunk, *field.shape[2:]), 1, 0)

        laid = _Steps(*(chunks(field) for field in gathered))
        rows = jax.tree.map(lambda leaf: jnp.broadcast_to(leaf, (batch, *leaf.shape)), initial)
        carry = (rows, jax.tree.map(jnp.zeros_like, rows))
        _, reads = jax.lax.scan(self._chunk, carry, laid)
        reads = jnp.moveaxis(reads, 0, 1).reshape(batch, count * self.chunk, *reads.shape[3:])[:, :length]
        inverse = jnp.argsort(order, axis=1)
        return jnp.take_along_axis(reads, inverse.reshape(*inverse.shape, 1, 1), axis=1)

    def _chunk(self, carry: tuple, step: _Steps) -> tuple:
        """One chunk of every row: its gradients at its start, its states, its readouts `[B, C, H, F]`."""
        state, momentum = carry
        state, momentum = checkpoint_name((state, momentum), CARRY)
        per_token = jax.vmap(jax.vmap(jax.vmap(
            lambda held, k, v: self._gradient(held, k, self._target(held, v)),
            in_axes=(0, 0, 0)), in_axes=(None, 0, 0)), in_axes=(0, 0, 0))
        gradient = per_token(state, step.key, step.value)
        real = step.real[..., None]
        theta, mu, a = (jnp.where(real, field, fill) for field, fill in
                        ((step.theta, 0.0), (step.mu, 1.0), (step.a, 1.0)))
        # Each token's affine step of (S, W): S' = mu S + u, W' = a W + c S + u, with c = mu for a
        # real token and 0 for a padding one, which leaves both as they are.
        c = jnp.where(real, mu, 0.0)
        update = jax.tree.map(lambda g: -_lead(theta, g) * g, gradient)
        matrices = jnp.stack([jnp.stack([mu, jnp.zeros_like(mu)], -1), jnp.stack([c, a], -1)], -2)
        composed, offsets = jax.lax.associative_scan(_composed, (matrices, (update, update)), axis=1)
        moment = jax.tree.map(lambda s0, bs: _scaled(composed[..., 0, 0], s0) + bs, momentum, offsets[0])
        states = jax.tree.map(lambda s0, w0, bw: _scaled(composed[..., 1, 0], s0)
                              + _scaled(composed[..., 1, 1], w0) + bw, momentum, state, offsets[1])
        reads = jax.vmap(jax.vmap(jax.vmap(self._read)))(states, step.query)
        last = jax.tree.map(lambda leaf: leaf[:, -1], (states, moment))
        return last, reads

    def _decoded(self, initial: dict, steps: _Steps) -> jax.Array:
        """A cached call: the cache's state continued token by token, its first call allocating it."""
        batch = steps.real.shape[0]
        rows = jax.tree.map(lambda leaf: jnp.zeros((batch, *leaf.shape), leaf.dtype), initial)
        allocated = self.has_variable("cache", STATE)
        start = self.variable("cache", START, lambda: rows)
        state = self.variable("cache", STATE, lambda: rows)
        momentum = self.variable("cache", MOMENTUM, lambda: rows)
        seen = self.variable("cache", COUNT, jnp.zeros, (batch,), jnp.int32)
        # A row's place continues from the tokens it has seen; a fresh row starts from W_0.
        places = seen.value[:, None] + jnp.cumsum(steps.real, axis=1) - steps.real
        first = steps.real & (places == 0)
        boundary = steps.real & (places % self.chunk == 0)
        reads, carry = self._stepped(initial, steps._replace(first=first, boundary=boundary),
                                     (start.value, state.value, momentum.value))
        if allocated:
            start.value, state.value, momentum.value = carry
            seen.value = seen.value + jnp.sum(steps.real, axis=1, dtype=jnp.int32)
        return reads


class _Steps(NamedTuple):
    """A call's per-token fields, rows then tokens first: the views `[B, S, H, F]`, the gates
    `[B, S, H]`, and the real, first and chunk-boundary masks `[B, S]`."""

    query: jax.Array
    key: jax.Array
    value: jax.Array
    theta: jax.Array
    mu: jax.Array
    a: jax.Array
    real: jax.Array
    first: jax.Array
    boundary: jax.Array


def _lead(scalars: jax.Array, leaf: jax.Array) -> jax.Array:
    """`scalars` broadcast against `leaf`, whose leading dimensions they are."""
    return scalars.reshape(*scalars.shape, *(1,) * (leaf.ndim - scalars.ndim))


def _scaled(scalars: jax.Array, leaf: jax.Array) -> jax.Array:
    """`leaf` `[B, H, ...]` times `scalars` `[B, C, H]`, broadcast to `[B, C, H, ...]`."""
    return scalars.reshape(*scalars.shape, *(1,) * (leaf.ndim - 2)) * leaf[:, None]


def _composed(first: tuple, second: tuple) -> tuple:
    """Two affine steps of (S, W) as one, the first applied first: `(M2 M1, M2 b1 + b2)`."""
    matrix_1, (s_1, w_1) = first
    matrix_2, (s_2, w_2) = second
    matrix = jnp.einsum("...ij,...jk->...ik", matrix_2, matrix_1)

    def weigh(row: int, column: int, leaf: jax.Array) -> jax.Array:
        scalars = matrix_2[..., row, column]
        return scalars.reshape(*scalars.shape, *(1,) * (leaf.ndim - scalars.ndim)) * leaf

    s = jax.tree.map(lambda s1, w1, s2: weigh(0, 0, s1) + weigh(0, 1, w1) + s2, s_1, w_1, s_2)
    w = jax.tree.map(lambda s1, w1, w2: weigh(1, 0, s1) + weigh(1, 1, w1) + w2, s_1, w_1, w_2)
    return matrix, (s, w)


__all__ = ["CARRY", "Gate", "InContextMemory", "LinearMemory", "MLPMemory", "Objective", "ResidualMemory"]
