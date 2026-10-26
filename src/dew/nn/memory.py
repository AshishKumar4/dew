"""In-context memory: a layer whose inner network's parameters update over the sequence (ledger 1645).

A token's views of the stream, `k_t`, `v_t`, `q_t`, train the inner network `f(W; .)` on an objective
of its own, and the token reads the updated network:

    g_t = grad_W objective(f(W_t'; k_t), target_t)      gradients at the chunk's start state W_t'
    S_t = mu_t S_{t-1} - theta_t g_t                     momentum (mu = 0 without it)
    W_t = a_t W_{t-1} + S_t                              retention (a = 1 - alpha; 1 without it)
    y_t = f(W_t; q_t)                                    or f(W_t'; q_t) where a chunk commits at its end

(Titans, arXiv 2501.00663, Eq 8-15; TTT, arXiv 2407.04620, Eq 2-6; Miras, arXiv 2504.13173, Eq 10, 13,
28; Nested Learning, arXiv 2512.24695, Eq 71, 84-90.) The target is the value view (`given`) or the
network's own image of it at the chunk's start, `f(W_t'; v_t)` (`self_generated`, Nested Learning
Eq 84). A chunk of `chunk` tokens takes every gradient at the state it starts from, counted from its
document's first real token; `chunk=1` is online gradient descent. Each token reads the network its
own step updated, or, where the chunk commits its steps at its end (`commit`), the network its chunk
started from (a continuum-memory level, Nested Learning Eq 71). A document's first token starts from
W_0, the network's meta-learned parameters, with no momentum; the parameters the inner loop never
steps (`slow`) stay at W_0 throughout.

The state (W, S and the chunk's start) is held in at least float32 whatever the compute dtype, and
the inner gradients and the scans run there: a bfloat16 state drops every step below its ulp. The
readout runs the network at the layer's dtype over the state cast to it, so at a zero step size a
network reads exactly as its W_0 does.

Two computations give the one function. A row of one document runs chunk by chunk: its real tokens
compacted to the front (`valid_order`), each chunk's gradients one `vmap` at the chunk's start, the
momentum and retention across the chunk one associative scan of their affine steps, and the chunk's
readouts one `vmap` over its states. A packed row, and a cached call, run token by token (`_stepped`),
which is the definition. The state is the network's parameters, a traced carry. Each head's network
is this layer's child (`head_0`, ...), whose parameters are its W_0, and every application runs that
child at its own scope over the weights the step holds, read-only (`_substituted`), so the layer
composes with a scanned layer stack and remat, and a method interceptor the model computes under finds
each head's sites at their paths, as at any child; the chunk's carry is named (`CARRY`) for a remat
policy. The heads run one after another, each vectorized over its rows and tokens, and may hold
different trees (an adapter on one head alone).

An adapter on an inner kernel (`dew.lora`) adds its branch, s drop(x) A B, at every application, as it
does at any Dense, its dropout drawing from each token's key in a forward that carries the dropout
stream. Its update is W_0's: the B factor rides the state with no step of its own, so a retention
forgets it as it forgets W_0, and the memory computes what the adapter's merge computes. A checkpoint's
input quantization at an inner site (`dew.quantization`) rounds that site's input at every application
alike, and Qwix's quantized training quantizes every product of the moving weights, its stochastic
rounding drawing from each token's key. A frozen FP8 record's codes at an inner kernel stand for W_0's
kernel, as they decode: in a memory only W_0 is frozen, and each application multiplies the state the
steps move from it. Qwix's serving quantization, which stores a weight quantized for its products to
read, is refused for the inner weights, which move with every token (`QuantizedServingRefused`).

A linear memory (`LinearMemory`) on its given target, with no momentum, no orthogonalized or slow
weights and each token reading its own step, whose heads each compute the linear map of their input by
their kernel (no wrapper rounds or drops their inputs or quantizes their products, `nonlinear_input`),
is a row of the shared linear recurrence (`dew.nn.kernels.recurrence`), and runs there: on the dot
objective its step writes theta k v^T, its retention the decay, at any chunk (the gradient is the same
at every state); on the l2 objective at chunk 1 without retention it is the delta rule with beta theta
(Titans takes the gradient before the retention, the gated delta rule after it, so those differ). A
call runs the recurrence's chunked op (`recurrence_chunk.chunk`, on its kernels where they take the
call) and a cached token its decode step (`recurrence.step`). The recurrence starts each document from
zero, so W_0 enters as its own term: the delta rule's state is W - W_0, written with the values
v - W_0^T k, and the dot rule's W - P_t W_0, P_t the product of the shares the document kept; each
readout adds P_t W_0^T q (P_t one for the delta rule). That readout is computed in at least float32 and
rounded once to the layer's dtype.
"""

from __future__ import annotations

import dataclasses
import functools
from typing import ClassVar, Literal, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn, traverse_util
from flax.core import lift
from flax.linen.transforms import lift_direct_transform
from flax.typing import Dtype, PrecisionLike
from jax.ad_checkpoint import checkpoint_name
from jax.interpreters.ad import JVPTracer

from dew.nn import linear
from dew.nn.dense import Dense
from dew.nn.kernels import recurrence, recurrence_chunk
from dew.nn.mixer_base import DELTA_RESIDUALS, MEMORY_RESIDUALS
from dew.nn.sharding import LOW_RANK_FACTORS, rows_like
from dew.optim import _orthogonalized
from dew.quantization import decoded_frozen_fp8, quantizes

from .attention import FORWARD_MODE, LayerNorm
from .inputs import PLAIN_CALL, MixerCall, coordinates_of, counted_places, sequence_blocks, valid_order
from .precision import at_least_fp32, l2norm

CARRY = MEMORY_RESIDUALS[0]
"""The name of the state a chunk hands the next (`checkpoint_name`), for a remat policy to save
(`dew.nn.mixer_base.RESIDUALS`)."""

STATE, MOMENTUM, START, COUNT = "memory_state", "memory_momentum", "memory_start", "memory_count"
"""The decode cache's variables: the network's fast parameters now, the momentum, the parameters at the
chunk's start, and the real tokens of each row's current document seen, rows first. A row that has
seen none starts from W_0."""
GRADIENT, TARGET, READ = range(3)
"""The uses of a token's network, which its dropout key folds in apart: the gradient's prediction at
its key, a self-generated target at its value, and its readout at its query."""

ADAPTER_A, ADAPTER_B = LOW_RANK_FACTORS
"""An adapter's factors beside an inner kernel (`dew.lora`): A held as W_0 holds it, B riding the state
with no step of its own (`InContextMemory._gradient`)."""

KEPT = "memory_kept"
"""A linear memory's decode cache beside its recurrence state (`STATE`): the log of the share of W_0
each row's current document keeps, the sum of its log retentions, rows first."""


@dataclasses.dataclass(frozen=True)
class Objective:
    """The loss a token's prediction `f(W; k)` takes against its target, summed over the features:
    `dot`, -<p, v> (linear attention's, Nested Learning Eq 17); `l2`, 1/2 ||p - v||^2 (Titans Eq 12
    with Appendix C's 1/2); `lp`, ||p - v||_p^p (Miras Eq 10); `huber`, Huber's loss of each feature's
    error at `delta` (Miras Eq 13); `l2_or_l1`, the l2 loss while the whole error's norm is at most
    `delta`, else `delta` times its l1 norm (Miras Eq 16, Yaad's Eq 26), so past `delta` every feature
    steps `delta` times its error's sign, however small its own error."""

    kind: Literal["dot", "l2", "lp", "huber", "l2_or_l1"] = "l2"
    p: float = 3.0
    delta: float = 1.0

    def __post_init__(self):
        if self.kind not in ("dot", "l2", "lp", "huber", "l2_or_l1"):
            raise ValueError(f"an objective is dot, l2, lp, huber or l2_or_l1, not {self.kind!r}")
        if self.kind == "lp" and self.p < 1:
            raise ValueError(f"an lp objective takes p >= 1, got {self.p}")
        if self.kind in ("huber", "l2_or_l1") and self.delta <= 0:
            raise ValueError(f"a {self.kind} objective takes a positive delta, got {self.delta}")

    def __call__(self, prediction: jax.Array, target: jax.Array) -> jax.Array:
        error = prediction - target
        if self.kind == "dot":
            return -jnp.sum(prediction * target, axis=-1)
        squares = 0.5 * jnp.sum(jnp.square(error), axis=-1)
        if self.kind == "l2":
            return squares
        size = jnp.abs(error)
        if self.kind == "lp":
            return jnp.sum(size ** self.p, axis=-1)
        if self.kind == "l2_or_l1":
            # ||p - v|| <= delta, squared on both sides.
            return jnp.where(2 * squares <= self.delta ** 2, squares, self.delta * jnp.sum(size, axis=-1))
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
    such as `GatedMLP`); it runs as each head's scope stores it, at the dtype each use asks, and its
    parameters, drawn at init, are W_0. `heads` memories run side by side over the views split into
    heads of `head_features`. A view is projected from the stream, or the stream itself where its
    `project_*` is off, as a continuum-memory level reads its block's input (one head). `gates` are the
    step size theta (`learning_rate`), the momentum mu (None: none) and the retention alpha (None: none).
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
    commit: Literal["token", "chunk"] = "token"
    """When a step becomes the network a token reads: each token's at once (`token`, TTT and Titans,
    which read the state their own step updated), or a chunk's at its end (`chunk`): the network stays
    as it is between those scheduled updates, so every token of a chunk reads the network its chunk
    started from, a continuum-memory level's (Nested Learning Eq 71)."""
    orthogonalize: int = 0
    """Newton-Schulz steps orthogonalizing each matrix of a token's gradient before its step, Muon's own
    iteration (`dew.optim._orthogonalized`, Nested Learning Sec 4.2's Muon-style inner update); 0 steps
    the gradient as it is. Vectors (biases, norm scales) step as they are."""
    slow: tuple[str, ...] = ()
    """The network's parameters, by their top-level names (a submodule's or a parameter's), that the
    inner loop reads at W_0 and never steps: meta-learned by the outer loop alone, as TTT's layer norm
    is (the authors' ttt.py updates only `ttt_param_names`, W1 and b1, and W2 and b2). The rest are
    the fast weights the sequence updates."""
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
        if self.commit not in ("token", "chunk"):
            raise ValueError(f"a memory commits its steps every token or chunk, not {self.commit!r}")
        if self.heads > 1 and not (self.project_query and self.project_key and self.project_value):
            raise ValueError("a view read as the stream itself is one head's; project it to split heads")
        super().__post_init__()

    def setup(self):
        # Each head's network, a child at its own scope (`head_0`, ...) whose parameters are that head's
        # W_0, computing at the dtype each use's operands carry (`_applied`).
        for head in range(self.heads):
            setattr(self, f"head_{head}", self.network.copy(dtype=None))

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

    def _logit(self, name: str, gate: Gate, x: jax.Array) -> jax.Array:
        """A projected gate's logit a token and head `[B, S, H]`, in at least float32."""
        logit = Dense(self.heads, dtype=self.dtype, precision=self.precision, name=name,
                      bias_init=nn.initializers.constant(gate.bias))(x)
        return logit.astype(at_least_fp32(self.dtype))

    def _gate(self, name: str, gate: Gate, x: jax.Array) -> jax.Array:
        """The gate's value a token and head `[B, S, H]`, in at least float32."""
        if gate.value is not None:
            return rows_like(jnp.full((*x.shape[:2], self.heads), gate.value, at_least_fp32(self.dtype)), x)
        return gate.scale * jax.nn.sigmoid(self._logit(name, gate, x))

    def _kept(self, x: jax.Array) -> tuple[jax.Array, jax.Array]:
        """The share `a = 1 - alpha` of the network each token keeps `[B, S, H]` and its log, in at least
        float32: one and zero without a retention."""
        wide = at_least_fp32(self.dtype)
        gate = self.retention
        if gate is None:
            nothing = rows_like(jnp.zeros((*x.shape[:2], self.heads), wide), x)
            return nothing + 1, nothing
        if gate.value is not None:
            alpha = rows_like(jnp.full((*x.shape[:2], self.heads), gate.value, wide), x)
            return 1 - alpha, jnp.log1p(-alpha)
        logit = self._logit("retention", gate, x)
        # log(1 - s sigmoid(z)) = log((1 - s) + e^-z) - log(1 + e^-z): finite wherever s is at most one,
        # where the sigmoid rounds to one included.
        log_kept = jnp.logaddexp(jnp.log1p(-gate.scale), -logit) - jax.nn.softplus(-logit)
        return 1 - gate.scale * jax.nn.sigmoid(logit), log_kept

    def _heads(self, features: int) -> tuple[_Head, ...]:
        """Each head's network as this call computes it (`_Head`).

        A head's network is its own child, so its parameters are its network's own (`head_0/...`): a
        matrix is one an optimizer steps on its own (Muon orthogonalizes it apart from the other heads')
        and a vector a vector, as its network declares them, and the heads may hold different trees (an
        adapter or an input quantization on one head alone). The first call of an initialization runs
        each network once at its scope over a zero vector of the view's width, so whatever draws there
        draws as at any child: W_0, an adapter's factors, a quantization's annotations. A head's
        frozen FP8 kernel is W_0's as its codes decode (`dew.quantization.decoded_frozen_fp8`): in a
        memory only W_0 is frozen, and every application multiplies the state the steps move from it.

        A weight stored quantized for serving (Qwix's, `dew.quantization.quantize_for_serving`) is
        refused (`QuantizedServingRefused`): a served product reads the stored value, and an inner
        weight moves with every token, so there is none to store."""
        networks = [getattr(self, f"head_{head}") for head in range(self.heads)]
        if self.is_initializing():
            for network in networks:
                network(jnp.zeros((features,), jnp.float32))
        wide = at_least_fp32(self.dtype)
        heads = []
        for network in networks:
            variables = decoded_frozen_fp8(dict(nn.meta.unbox(network.variables)))
            flat = traverse_util.flatten_dict(dict(variables.pop("params", {})))
            served = sorted("/".join(path) for path, leaf in flat.items()
                            if not isinstance(leaf, jax.Array | np.ndarray))
            if served:
                raise QuantizedServingRefused("/".join(network.path), served)
            unknown = sorted(set(self.slow) - {path[0] for path in flat})
            if unknown:
                raise ValueError(f"slow names {unknown}, which are none of the network's parameters "
                                 f"{sorted({path[0] for path in flat})}")
            held = {path for path in flat if path[0] in self.slow or path[-1] == ADAPTER_A}
            heads.append(_Head(
                network,
                traverse_util.unflatten_dict({path: flat[path].astype(wide) for path in held}),
                variables,
                traverse_util.unflatten_dict({path: leaf.astype(wide) for path, leaf in flat.items()
                                              if path not in held})))
        return tuple(heads)

    def nonlinear_input(self, name: str) -> bool:
        """Whether this call computes the submodule `name` (a head's network), or a module under it,
        other than as the linear map of its input by its stored kernel, an adapter's branch added: as
        the Qwix quantization the call runs under selects it (`dew.quantization.quantizes`), and as a
        method interceptor answers for itself (`dew.lora`'s dropout on a training forward, a
        checkpoint's input quantization). A linear memory runs the shared recurrence only where every
        head computes that map (`_route`)."""
        return quantizes(self, name)

    @nn.compact
    def __call__(self, x: jax.Array, call: MixerCall = PLAIN_CALL) -> jax.Array:
        batch, length, _ = x.shape
        query, key, value, features = self._views(x)
        theta = self._gate("learning_rate", self.learning_rate, x)
        mu = (jnp.zeros_like(theta) if self.momentum is None else self._gate("momentum", self.momentum, x))
        a, log_kept = self._kept(x)
        wide = at_least_fp32(self.dtype)
        heads = self._heads(features)
        tokens = coordinates_of(call.metadata, batch, length)
        # Every array of rows this call makes is placed as the stream's rows are (`rows_like`), so an
        # Explicit axis that splits them splits these alike, a scan's carry and a cache among them.
        real = (rows_like(jnp.ones((batch, length), bool), x) if tokens.valid is None
                else jnp.asarray(tokens.valid, bool))
        if tokens.segments is not None:
            real = real & (tokens.segments != 0)
        # Each token's rank among its document's real tokens, which the chunks count from whatever
        # logical positions the call gives, as the chunked path compacts the real tokens from 0.
        places = counted_places(tokens.segments, tokens.valid)
        if places is None:
            places = rows_like(jnp.broadcast_to(jnp.arange(length)[None], (batch, length)), x)
        steps = _Steps(query.astype(wide), key.astype(wide), value.astype(wide), theta, mu, a, real, places,
                       self._keys(x))
        hooks, segments = self._route(), tokens.max_segments
        if hooks is not None:
            read = (self._recurrent_decoded(hooks, heads, steps, log_kept, segments) if call.decode
                    else self._recurrent(hooks, heads, steps, log_kept, max_segments=segments)[0])
        elif call.decode:
            read = self._decoded(heads, steps)
        elif tokens.max_segments > 1:
            read = self._stepped(heads, steps)[0]
        else:
            read = self._chunked(heads, steps)
        read = read.astype(self.dtype or x.dtype)
        if self.output:
            read = Dense(x.shape[-1], use_bias=False, dtype=self.dtype, precision=self.precision,
                         name="o_proj")(read.reshape(batch, length, -1))
        else:
            read = read.reshape(batch, length, -1)
        return read

    def _keys(self, x: jax.Array) -> dict[str, jax.Array] | None:
        """Each token and head's key data `[B, S, H, ...]` of every RNG stream the call carries but
        `params` (a training forward's `dropout` and `stochastic_rounding`, `training_rngs`), each split
        from its own stream, so the streams stay as apart as the call holds them; None without any. An
        application folds its use into its token's key (`_applied`)."""
        assert self.scope is not None, "a bound memory has a scope"
        streams = sorted(name for name in self.scope.rngs if name != "params")
        batch, length = x.shape[:2]
        return {name: rows_like(jax.random.key_data(jax.random.split(self.make_rng(name),
                                                                     (batch, length, self.heads))), x)
                for name in streams} or None

    @property
    def residuals(self) -> tuple[str, ...]:
        """The residuals this layer may mark for a remat policy (`dew.nn.mixer_base.RESIDUALS`): those of
        every path its configuration can take, as the call's wrappers decide between them and the
        configuration does not see the wrappers (`_route`). A linear memory on the shared recurrence
        marks its rule's (the delta rule's port saves its decay, inverse and memory; the dot rule's
        portable math marks none), and on the general path, as any other memory, its chunk scan's
        carry."""
        hooks = self._recurrence()
        return (*(DELTA_RESIDUALS if hooks is not None and hooks.delta else ()), *MEMORY_RESIDUALS)

    def _route(self) -> recurrence.Hooks | None:
        """This call's row of the shared linear recurrence (`_recurrence`), or None where its
        configuration is none or a head computes other than the linear map of its input by its kernel
        (`nonlinear_input`): an input's quantizer, Qwix's quantized products, an adapter's dropout on a
        training forward."""
        hooks = self._recurrence()
        if hooks is None or any(self.nonlinear_input(f"head_{head}") for head in range(self.heads)):
            return None
        return hooks

    # A linear memory on the shared recurrence.

    def _recurrence(self) -> recurrence.Hooks | None:
        """This memory's form as a row of the shared linear recurrence (the module docstring), or None
        where it is none: a linear network on its given target, stepping each matrix as its gradient
        with no momentum and every weight fast, each token reading its own step, on the dot objective
        (any chunk, its gradient the same at every state) or on the l2 objective at chunk 1 without
        retention (the delta rule, whose gradient Titans takes before the retention and the gated rule
        after it), a retention keeping a share of the network in (0, 1]."""
        gate = self.retention
        kept = gate is None or (0 <= gate.scale <= 1 if gate.value is None else 0 <= gate.value < 1)
        rows = {"dot": True, "l2": self.chunk == 1 and gate is None}
        linear = (isinstance(self.network, LinearMemory) and self.target == "given" and self.momentum is None
                  and not self.orthogonalize and not self.slow and self.commit == "token")
        if not (linear and kept and rows.get(self.objective.kind, False)):
            return None
        # The l2 rule's log decay is zero: with a decay the delta rule takes the port's kernels.
        return recurrence.Hooks(decay="scalar", delta=self.objective.kind == "l2", write_gate=True, scale=1.0)

    def _recurrent(self, hooks: recurrence.Hooks, heads: tuple[_Head, ...], steps: _Steps,
                   log_kept: jax.Array, *, max_segments: int, held: jax.Array | None = None,
                   kept: jax.Array | None = None) -> tuple[jax.Array, jax.Array, jax.Array]:
        """The readouts `[B, S, H, F]` of a linear memory through the shared recurrence, its state after
        them `[B, H, F, F]` and each token's log of the share of W_0 its document keeps `[B, S, H]`: the
        recurrence's chunked op, or its decode step for a cached call of one token, continuing the state
        `held` and the share `kept` `[B, H]` a cache holds. The recurrence starts each document from
        zero, so W_0 enters as its own term (the module docstring), each head's network at its W_0 over
        a token's views: an adapter's branch on its kernel included, which the recurrence's state never
        holds."""
        real = steps.real
        # A padding token neither steps nor decays the memory.
        beta = jnp.where(real[..., None], steps.theta, 0.0)
        log_decay = jnp.where(real[..., None], log_kept, 0.0)

        def read_w0(view: jax.Array, use: int) -> jax.Array:
            return jnp.stack([jax.vmap(jax.vmap(functools.partial(
                self._applied, head, head.initial, wide=True, use=use)))(
                    view[:, :, index], _at(steps.keys, index, 2))
                for index, head in enumerate(heads)], axis=2)

        value = steps.value - read_w0(steps.key, GRADIENT) if hooks.delta else steps.value
        begins, continued = _documents(steps)
        if held is not None and real.shape[1] == 1:
            # The shared decode dispatch: its step kernel where it takes the state, else the portable
            # step in the memory's own precision.
            out, state = linear.decode_step(
                hooks, held, steps.query[:, 0], steps.key[:, 0], value[:, 0], log_decay[:, 0], beta[:, 0],
                real[:, 0], tangent=FORWARD_MODE.get() or isinstance(steps.query, JVPTracer))
            out = out[:, None]
        else:
            out, state = recurrence_chunk.chunk(hooks, recurrence_chunk.Chunk(
                steps.query, steps.key, value, log_decay=log_decay, beta=beta, state=held,
                starts=begins & ~continued), max_segments=max_segments)
        share = _within(log_decay, begins)
        if kept is not None:
            share = share + jnp.where(continued[..., None], kept[:, None], 0.0)
        wide = at_least_fp32(self.dtype)
        return out.astype(wide) + jnp.exp(share)[..., None] * read_w0(steps.query, READ), state, share

    def _recurrent_decoded(self, hooks: recurrence.Hooks, heads: tuple[_Head, ...], steps: _Steps,
                           log_kept: jax.Array, max_segments: int) -> jax.Array:
        """A linear memory's cached call (`_recurrent`), the cache holding the recurrence's state
        (`STATE`) and the log of W_0's share each row's document keeps (`KEPT`), zero for a fresh row;
        its first call allocates it."""
        batch, features, wide = steps.real.shape[0], steps.key.shape[-1], at_least_fp32(self.dtype)
        allocated = self.has_variable("cache", STATE)
        held = self.variable("cache", STATE, lambda: rows_like(
            jnp.zeros((batch, self.heads, features, features), wide), steps.real))
        kept = self.variable("cache", KEPT, lambda: rows_like(
            jnp.zeros((batch, self.heads), wide), steps.real))
        reads, state, share = self._recurrent(hooks, heads, steps, log_kept, max_segments=max_segments,
                                              held=held.value, kept=kept.value)
        if allocated:
            held.value = state
            kept.value = _at_last(share, steps.real, kept.value)
        return reads

    # The function, token by token and chunk by chunk.

    def _applied(self, head: _Head, fast: dict, x: jax.Array, keys: dict[str, jax.Array] | None = None, *,
                 wide: bool, use: int) -> jax.Array:
        """`head`'s network at its moving weights `fast` and the ones it holds over one token's vector
        `x`, at the state's precision where `wide` (the gradients) and else the layer's dtype (a
        readout), run at its own scope (`_substituted`): so a method interceptor the model computes
        under finds its sites there, an adapter's branch and a quantization among them, each RNG stream
        drawing from the token's `keys` (key data, None without any) folded with this `use`."""
        params = _joined(head.held, fast)
        if not wide:
            dtype = self.dtype or x.dtype
            params = jax.tree.map(lambda leaf: leaf.astype(dtype) if self.dtype is not None else leaf, params)
            x = x.astype(dtype)
        rngs = {} if keys is None else {name: jax.random.fold_in(jax.random.wrap_key_data(data), use)
                                        for name, data in keys.items()}
        out = _substituted(head.network, {**head.collections, "params": params}, rngs, x)
        if not isinstance(out, jax.Array):
            raise TypeError(f"an in-context memory's network maps a vector to one, got {type(out).__name__}")
        return out

    def _gradient(self, head: _Head, state: dict, key: jax.Array, target: jax.Array,
                  keys: dict[str, jax.Array] | None) -> dict:
        """One head's gradient of the objective at the moving weights `state` for one token, in the
        state's precision, each matrix orthogonalized where `orthogonalize` asks. An adapter's B factor
        rides the state, so a retention forgets the adapter's update s A B as it forgets W_0's, and the
        memory computes what the adapter's merge computes; it takes no step of its own, which goes to the
        kernel it updates, whose gradient is the whole map's."""
        gradient = jax.grad(lambda fast: self.objective(
            self._applied(head, fast, key, keys, wide=True, use=GRADIENT), target))(state)

        def stepped(path: tuple, leaf: jax.Array) -> jax.Array:
            if path[-1].key == ADAPTER_B:
                return jnp.zeros_like(leaf)
            return (_orthogonalized(leaf, self.orthogonalize, None) if self.orthogonalize and leaf.ndim == 2
                    else leaf)

        return jax.tree_util.tree_map_with_path(stepped, gradient)

    def _target(self, head: _Head, state: dict, value: jax.Array, keys: dict[str, jax.Array] | None
                ) -> jax.Array:
        """A token's target: its value view, or the network's image of it at `state` (Nested Learning
        Eq 84). The inner gradient holds it fixed (`_gradient` differentiates the prediction alone); the
        outer one runs through it, to the value projection and the state that made it."""
        if self.target == "given":
            return value
        return self._applied(head, state, value, keys, wide=True, use=TARGET)

    def _step(self, head: _Head, state: dict, key: jax.Array, value: jax.Array,
              keys: dict[str, jax.Array] | None) -> dict:
        """One head's gradient for one token at `state`, of the objective against its target there."""
        return self._gradient(head, state, key, self._target(head, state, value, keys), keys)

    def _read(self, head: _Head, state: dict, query: jax.Array, keys: dict[str, jax.Array] | None
              ) -> jax.Array:
        """One head's readout of a token at `state`, at the layer's dtype."""
        return self._applied(head, state, query, keys, wide=False, use=READ)

    def _token(self, carry: tuple, step: _Steps, heads: tuple[_Head, ...]) -> tuple:
        """One token's update and readout over every row and head: the definition. The carry holds each
        head's start, state and momentum, rows first."""
        start, state, momentum = carry
        query, key, value, theta, mu, a, real, place, keys = step
        first, boundary = real & (place == 0), real & (place % self.chunk == 0)

        def reset(fresh, held):
            return jax.tree.map(lambda f, h: jnp.where(_lead(first, h), f, h), fresh, held)

        def keep(new, old):
            return jax.tree.map(lambda n, o: jnp.where(_lead(real, n), n, o), new, old)

        initial = tuple(_rows(head.initial, real) for head in heads)
        state = reset(initial, state)
        momentum = reset(jax.tree.map(jnp.zeros_like, initial), momentum)
        start = reset(initial, jax.tree.map(lambda s, w: jnp.where(_lead(boundary, s), w, s), start, state))
        gradient = tuple(jax.vmap(functools.partial(self._step, head))(
            start[index], key[:, index], value[:, index], _at(keys, index, 1))
            for index, head in enumerate(heads))
        moved = tuple(jax.tree.map(functools.partial(_momentum_step, mu[:, index], theta[:, index]),
                                   momentum[index], gradient[index]) for index in range(self.heads))
        stepped = tuple(jax.tree.map(functools.partial(_state_step, a[:, index]), state[index], moved[index])
                        for index in range(self.heads))
        state, momentum = keep(stepped, state), keep(moved, momentum)
        read = state if self.commit == "token" else start
        reads = [jax.vmap(functools.partial(self._read, head))(
            read[index], query[:, index], _at(keys, index, 1)) for index, head in enumerate(heads)]
        return (start, state, momentum), jnp.stack(reads, axis=1)

    def _stepped(self, heads: tuple[_Head, ...], steps: _Steps, carry: tuple | None = None
                 ) -> tuple[jax.Array, tuple]:
        """The readouts `[B, S, H, F]` token by token, and the carry after the last token."""
        if carry is None:
            rows = tuple(_rows(head.initial, steps.real) for head in heads)
            carry = (rows, rows, jax.tree.map(jnp.zeros_like, rows))
        sequence = jax.tree.map(lambda field: jnp.moveaxis(field, 1, 0), steps)
        carry, reads = jax.lax.scan(lambda held, step: self._token(held, step, heads), carry, sequence)
        return jnp.moveaxis(reads, 0, 1), carry

    def _chunked(self, heads: tuple[_Head, ...], steps: _Steps) -> jax.Array:
        """The readouts `[B, S, H, F]` of rows of one document, chunk by chunk (the module docstring)."""
        batch, length = steps.real.shape
        order, _ = valid_order(steps.real)
        gathered = jax.tree.map(lambda field: jnp.take_along_axis(
            field, order.reshape(*order.shape, *(1,) * (field.ndim - 2)), axis=1), steps)
        laid = jax.tree.map(lambda field: sequence_blocks(field, self.chunk, leading=True), gathered)
        count = laid.real.shape[0]
        rows = tuple(_rows(head.initial, steps.real) for head in heads)
        carry = (rows, jax.tree.map(jnp.zeros_like, rows))
        _, reads = jax.lax.scan(functools.partial(self._chunk, heads=heads), carry, laid)
        reads = jnp.moveaxis(reads, 0, 1).reshape(batch, count * self.chunk, *reads.shape[3:])[:, :length]
        inverse = jnp.argsort(order, axis=1)
        return jnp.take_along_axis(reads, inverse.reshape(*inverse.shape, 1, 1), axis=1)

    def _chunk(self, carry: tuple, step: _Steps, *, heads: tuple[_Head, ...]) -> tuple:
        """One chunk of every row: its gradients at its start, its states, its readouts `[B, C, H, F]`.
        The carry holds each head's state and momentum, rows first."""
        state, momentum = carry
        state, momentum = checkpoint_name((state, momentum), CARRY)
        real = step.real[..., None]
        theta, mu, a = (jnp.where(real, field, fill) for field, fill in
                        ((step.theta, 0.0), (step.mu, 1.0), (step.a, 1.0)))
        # Each token's affine step of (S, W): S' = mu S + u, W' = a W + c S + u, with c = mu for a
        # real token and 0 for a padding one, which leaves both as they are.
        c = jnp.where(real, mu, 0.0)
        matrices = jnp.stack([jnp.stack([mu, jnp.zeros_like(mu)], -1), jnp.stack([c, a], -1)], -2)
        laid = [self._chunk_head(head, state[index], momentum[index], step, theta[:, :, index],
                                 matrices[:, :, index], index) for index, head in enumerate(heads)]
        lasts, moments, reads = zip(*laid, strict=True)
        return (lasts, moments), jnp.stack(reads, axis=2)

    def _chunk_head(self, head: _Head, state: dict, momentum: dict, step: _Steps, theta: jax.Array,
                    matrices: jax.Array, index: int) -> tuple[dict, dict, jax.Array]:
        """One head's chunk of every row (`_chunk`), from its state and momentum `[B, ...]`, its step
        sizes `[B, C]` and affine steps `[B, C, 2, 2]`: its state and momentum after the chunk and its
        readouts `[B, C, F]`."""
        keys = _at(step.keys, index, 2)
        gradient = jax.vmap(jax.vmap(functools.partial(self._step, head), in_axes=(None, 0, 0, 0)))(
            state, step.key[:, :, index], step.value[:, :, index], keys)
        update = jax.tree.map(lambda g: -_lead(theta, g) * g, gradient)
        composed, offsets = jax.lax.associative_scan(_composed, (matrices, (update, update)), axis=1)
        moment = jax.tree.map(lambda s0, bs: _scaled(composed[..., 0, 0], s0) + bs, momentum, offsets[0])
        states = jax.tree.map(lambda s0, w0, bw: _scaled(composed[..., 1, 0], s0)
                              + _scaled(composed[..., 1, 1], w0) + bw, momentum, state, offsets[1])
        # Each token reads its own state, or with commit per chunk the state the chunk entered.
        read, mapped = (states, 0) if self.commit == "token" else (state, None)
        reads = jax.vmap(jax.vmap(functools.partial(self._read, head), in_axes=(mapped, 0, 0)))(
            read, step.query[:, :, index], keys)
        last = jax.tree.map(lambda leaf: leaf[:, -1], (states, moment))
        return *last, reads

    def _decoded(self, heads: tuple[_Head, ...], steps: _Steps) -> jax.Array:
        """A cached call: the cache's state continued token by token, its first call allocating it. The
        call's first document continues the row's, whose real tokens the cache has counted; a document
        that starts in the call starts from W_0, and the cache then counts that document's."""
        rows = jax.tree.map(jnp.zeros_like, tuple(_rows(head.initial, steps.real) for head in heads))
        allocated = self.has_variable("cache", STATE)
        start = self.variable("cache", START, lambda: rows)
        state = self.variable("cache", STATE, lambda: rows)
        momentum = self.variable("cache", MOMENTUM, lambda: rows)
        seen = self.variable("cache", COUNT, lambda: jnp.zeros_like(steps.real, jnp.int32)[:, 0])
        places = steps.places + jnp.where(_documents(steps)[1], seen.value[:, None], 0)
        reads, carry = self._stepped(heads, steps._replace(places=places),
                                     (start.value, state.value, momentum.value))
        if allocated:
            start.value, state.value, momentum.value = carry
            seen.value = _at_last(places + 1, steps.real, seen.value)
        return reads


class QuantizedServingRefused(ValueError):
    """An in-context memory refusing inner weights stored quantized for serving (Qwix's, at the head
    network `network`'s `leaves`): a served product reads a stored quantized weight, and an inner weight
    moves with every token, so there is none to store. Its projections serve; quantized training
    (`dew.quantization.Quantization.apply`) quantizes every product of the moving weights."""

    def __init__(self, network: str, leaves: list[str]):
        super().__init__(f"{network}'s weights {leaves} are stored quantized for serving, which an "
                         "in-context memory's inner weights cannot be: they move with every token, so there "
                         "is no stored value to serve. Select its projections for serving, or train it "
                         "quantized")
        self.network, self.leaves = network, leaves

    def __reduce__(self):
        return type(self), (self.network, self.leaves)


class _Head(NamedTuple):
    """One head's network as a call computes it: the child at its scope (`network`), the parameters it
    holds at W_0 (`held`: the `slow` ones and an adapter's A factors), its other collections as stored
    (an input's scale), and the W_0 of its moving weights (`initial`), all in the state's precision."""

    network: nn.Module
    held: dict
    collections: dict
    initial: dict


def _substituted(network: nn.Module, variables: dict, rngs: dict[str, jax.Array], x: jax.Array) -> jax.Array:
    """The bound `network` over `x` at its own scope, its variables `variables` and its RNG streams
    `rngs` in place of the ones it stores, every collection read-only: one lifted transform on Flax's
    `lift.pack`. A method interceptor finds the network at its path, as at any child. `nn.map_variables`
    substitutes the variables alone: its `rngs` filters the enclosing scope's streams, whose keys Flax
    draws by trace-time counters, so inside the memory's scans and vmaps every token and row would draw
    one key; here each call's streams are the token's own (`InContextMemory._keys`)."""
    def transform(fn, scopes, *args):
        def run(scope_fn, repack, variable_groups, rng_groups, *args):
            (group,) = variable_groups
            if len(group) != 1:
                raise ValueError("an in-context memory's network runs at its one scope, sharing no module")
            inner = scope_fn(((variables,),), ((rngs,),), mutable_filter=False)
            return fn(inner, *args), repack(inner)
        return lift.pack(run, (True,), (False,), (True,), name="memory")(scopes, *args)

    return lift_direct_transform(transform, (_call,), network, x)


def _call(network: nn.Module, x: jax.Array) -> jax.Array:
    return network(x)


class _Steps(NamedTuple):
    """A call's per-token fields, rows then tokens first: the views `[B, S, H, F]`, the gates
    `[B, S, H]`, the real tokens `[B, S]`, each one's rank among its document's real tokens `[B, S]`,
    its first at 0, from which its chunks count, and each token and head's key data of every RNG stream
    `[B, S, H, ...]`, None without any (`InContextMemory._keys`)."""

    query: jax.Array
    key: jax.Array
    value: jax.Array
    theta: jax.Array
    mu: jax.Array
    a: jax.Array
    real: jax.Array
    places: jax.Array
    keys: dict[str, jax.Array] | None


def _documents(steps: _Steps) -> tuple[jax.Array, jax.Array]:
    """Each document's first real token `[B, S]`, and the tokens of a call's first document `[B, S]`,
    every one up to its second document's first: in a cached call, the document its row holds."""
    begins = steps.real & (steps.places == 0)
    return begins, jnp.cumsum(begins, axis=1) <= 1


def _at_last(values: jax.Array, real: jax.Array, held: jax.Array) -> jax.Array:
    """`values` `[B, S, ...]` at each row's last real token, and `held` `[B, ...]` in a row with none."""
    last = jnp.max(jnp.where(real, jnp.arange(real.shape[1]), 0), axis=1)
    at = jnp.take_along_axis(values, _lead(last, values), axis=1)[:, 0]
    return jnp.where(_lead(jnp.any(real, axis=1), held), at, held)


def _rows(initial: dict, real: jax.Array) -> dict:
    """A head's W_0 for every row `[B, ...]`, placed as the rows of `real` `[B, S]` are."""
    return jax.tree.map(lambda leaf: rows_like(jnp.broadcast_to(leaf, (real.shape[0], *leaf.shape)), real),
                        initial)


def _within(values: jax.Array, begins: jax.Array) -> jax.Array:
    """Each token's sum of `values` `[B, S, ...]` over its document up to it, `begins` `[B, S]` marking
    each document's first token: a sum that restarts at each, scanned as one associative scan."""
    def combined(first: tuple, second: tuple) -> tuple:
        (opened, summed), (opens, adds) = first, second
        return opened | opens, jnp.where(_lead(opens, adds), adds, summed + adds)

    return jax.lax.associative_scan(combined, (begins, values), axis=1)[1]


def _at(keys: dict[str, jax.Array] | None, head: int, axis: int) -> dict[str, jax.Array] | None:
    """Each stream's key data in `keys` at head `head` of its `axis`; None stays None."""
    return None if keys is None else {name: jnp.take(data, head, axis=axis) for name, data in keys.items()}


def _joined(held: dict, moving: dict) -> dict:
    """One network's parameters from its held leaves and its moving ones, which name different leaves."""
    return traverse_util.unflatten_dict({**traverse_util.flatten_dict(held),
                                         **traverse_util.flatten_dict(moving)})


def _lead(scalars: jax.Array, leaf: jax.Array) -> jax.Array:
    """`scalars` broadcast against `leaf`, whose leading dimensions they are."""
    return scalars.reshape(*scalars.shape, *(1,) * (leaf.ndim - scalars.ndim))


def _momentum_step(mu: jax.Array, theta: jax.Array, momentum: jax.Array, gradient: jax.Array) -> jax.Array:
    """`mu S - theta g` for a head's leaf `[B, ...]` and its rows' gates `[B]`."""
    return _lead(mu, momentum) * momentum - _lead(theta, gradient) * gradient


def _state_step(a: jax.Array, state: jax.Array, momentum: jax.Array) -> jax.Array:
    """`a W + S` for a head's leaf `[B, ...]` and its rows' kept shares `[B]`."""
    return _lead(a, state) * state + momentum


def _scaled(scalars: jax.Array, leaf: jax.Array) -> jax.Array:
    """`leaf` `[B, ...]` times `scalars` `[B, C]`, broadcast to `[B, C, ...]`."""
    return scalars.reshape(*scalars.shape, *(1,) * (leaf.ndim - 1)) * leaf[:, None]


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


__all__ = ["CARRY", "Gate", "InContextMemory", "LinearMemory", "MLPMemory", "Objective",
           "QuantizedServingRefused", "ResidualMemory"]
