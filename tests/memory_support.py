"""The in-context memory checks' shared draws and oracles (`dew.nn.memory`, ledger 1645): the
configurations each paper writes (`CONFIGURATIONS`, `layer`); the token-by-token reference of their
update (`written`), each loss as its paper writes it (`paper_loss`); a seed's stream and its own weights
(`drawn`); the calls the checks compare, the full pass (`applied`), a packed call (`packed`) and a cached
decode (`decoded`); and where a head's inner network sits in the weights (`head_of`, `inner_sites`)."""

import functools
import re

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn, traverse_util

from dew.batch import coordinates
from dew.nn.inputs import PLAIN_CALL, AttentionMetadata, MixerCall
from dew.nn.memory import Gate, InContextMemory, LinearMemory, MLPMemory, Objective, ResidualMemory
from dew.nn.mlp import GatedMLP
from dew.nn.precision import L2NORM_EPS

WIDTH, HEADS, FEATURES, LENGTH, CHUNK, ROWS = 32, 2, 16, 12, 4, 8
MEMORY_SEEDS = 8
"""The declared seeds for check 1 of the memories (`assert_as_exact`): `seeds_for` on armada sizes every
configuration, its token-by-token, decoded, packed, cached and recurrence paths included, at 8 over the
drawn weights, but the sites a check module's `SEEDS` names."""


CONFIGURATIONS = {
    # TTT-Linear (Sec 2.7): f(x) = x + LN(W x), the L2 reconstruction (Eq 4), eta(x) = eta_base sigmoid(.).
    # Its inner loop updates W and b alone; the layer norm's scale and bias are outer-loop parameters
    # (the authors' ttt.py, `TTTCache.ttt_param_names`).
    "ttt_linear": {"network": ResidualMemory(hidden=None, norm=True), "slow": ("norm",)},
    # TTT-MLP: a 4x GELU MLP inside the residual and norm, the norm outer-loop only.
    "ttt_mlp": {"network": ResidualMemory(hidden=4 * FEATURES, activation="gelu"), "slow": ("norm",)},
    # Titans' deep memory with momentum and forgetting (Eq 12-14).
    "titans": {"network": MLPMemory(hidden=(2 * FEATURES,)),
               "momentum": Gate(), "retention": Gate(bias=-2.0)},
    # Moneta: the lp bias with p = 3 (Miras Eq 10, 24) and retention. Its gradient grows with the
    # error's square, so it steps a two-hundredth where the others step a tenth.
    "moneta": {"network": ResidualMemory(hidden=4 * FEATURES),
               "objective": Objective("lp", p=3.0), "retention": Gate(bias=-2.0),
               "learning_rate": Gate(scale=0.005)},
    # Yaad: Miras Eq 16's switch between l2 and l1 by the whole error's norm, as its Eq 26 selects.
    "yaad": {"network": ResidualMemory(hidden=4 * FEATURES),
             "objective": Objective("l2_or_l1", delta=0.5), "retention": Gate(bias=-2.0)},
    # Miras' first Huber bias, each coordinate's Huber loss (Eq 13-14), on Yaad's memory.
    "huber": {"network": ResidualMemory(hidden=4 * FEATURES),
              "objective": Objective("huber", delta=0.5), "retention": Gate(bias=-2.0)},
    # Nested Learning's self-modifying memory: a residual MLP (Eq 89), its own target (Eq 84), retention.
    "self_modifying": {"network": ResidualMemory(hidden=2 * FEATURES, norm=False),
                       "target": "self_generated", "retention": Gate(bias=-2.0)},
    # Linear attention: a linear memory on the dot-product bias, every token (Nested Learning Eq 14-18).
    "linear_attention": {"network": LinearMemory(),
                         "objective": Objective("dot"), "chunk": 1},
    # The delta rule: a linear memory on the l2 objective, every token (Miras Eq 9, DeltaNet).
    "delta_rule": {"network": LinearMemory(), "chunk": 1},
    # Gated linear attention: the dot-product bias with a retention gate (Miras Eq 8), whose gradient
    # is the same at every state, so its chunk changes nothing.
    "gated_linear_attention": {"network": LinearMemory(), "objective": Objective("dot"),
                               "retention": Gate(bias=-2.0)},
    # A continuum-memory level (Nested Learning Sec 7.1): an existing gated MLP over its block's input,
    # reconstructing it, its parameters stepped at the end of every chunk (Eq 71), at a hundredth.
    "cms": {"network": GatedMLP(hidden_features=2 * WIDTH, out_features=WIDTH), "commit": "chunk", "heads": 1,
            "head_features": None, "project_query": False, "project_key": False, "project_value": False,
            "output": False, "normalize": False, "learning_rate": Gate(scale=0.01)},
}
"""Each configuration with its paper's queries and keys l2-normalized (Titans Sec 4.4, TTT's views) and
a step size of `STEP` times a sigmoid, the stable range the papers train in: an unnormalized key under
a unit step drives the inner gradient descent off, where no precision computes the same update."""


STEP = 0.1
"""Heads 16 wide: a layer norm over a head's few features divides by a spread that a draw can bring
near zero, so narrower heads leave the norm's rounding too heavy-tailed for any count to resolve."""


def layer(name: str, dtype=jnp.float32) -> InContextMemory:
    fields = {"heads": HEADS, "head_features": FEATURES, "chunk": CHUNK, "normalize": True,
              "learning_rate": Gate(scale=STEP), **CONFIGURATIONS[name]}
    return InContextMemory(dtype=dtype, **fields)


def huber(error, delta):
    """Huber's loss of each coordinate of `error` (Miras Eq 13)."""
    size = jnp.abs(error)
    return jnp.where(size <= delta, error * error / 2, delta * (size - delta / 2))


def paper_loss(kind: str, objective: Objective):
    """The loss its paper writes for `kind`, of one token's prediction and target `[F]` at `objective`'s
    `p` and `delta`, each written here from its equation: Nested Learning Eq 17's dot product, Titans
    Eq 12's l2 (with Appendix C's half), Miras Eq 10's lp, Miras Eq 13's Huber of each coordinate, and
    Miras Eq 16, the l2 objective while the whole error's norm is within delta and delta times the l1
    objective past it."""
    losses = {
        "dot": lambda prediction, target: -jnp.dot(prediction, target),
        "l2": lambda prediction, target: jnp.dot(prediction - target, prediction - target) / 2,
        "lp": lambda prediction, target: jnp.sum(jnp.abs(prediction - target) ** objective.p),
        "huber": lambda prediction, target: jnp.sum(huber(prediction - target, objective.delta)),
        "l2_or_l1": lambda prediction, target: jnp.where(
            jnp.sqrt(jnp.dot(prediction - target, prediction - target)) <= objective.delta,
            jnp.dot(prediction - target, prediction - target) / 2,
            objective.delta * jnp.linalg.norm(prediction - target, 1)),
    }
    return losses[kind]


def written(module: InContextMemory, *, dtype, segments=None, gradient=None, inputs=None):
    """The paper's update as a function of the stream, token by token, in `dtype` (float64 for the
    truth): the views and gates from the layer's projections (or the stream itself), the state in at
    least float32, each gradient at its chunk's start counted from its document's first token
    (`segments` `[S]`, one document where None) of the loss its paper writes for the layer's objective
    (`paper_loss`), the network's `slow` parameters left at W_0, and the readout at `dtype` of the
    state each token updated, or, where a chunk commits its steps at its end, of the state its chunk
    started from (Nested Learning Eq 71). `inputs` maps each vector the network takes (a quantizer's
    round trip), none where None."""
    wide = jnp.promote_types(dtype, jnp.float32)
    fed = (lambda view: view) if inputs is None else inputs
    grad_net, read_net = module.network.clone(dtype=wide), module.network.clone(dtype=dtype)
    loss = paper_loss(module.objective.kind, module.objective)
    firsts = np.arange(LENGTH) == 0 if segments is None else np.r_[True, segments[1:] != segments[:-1]]

    def gate(name, spec, x, p):
        if spec is None:
            return None
        if spec.value is not None:
            return jnp.full((x.shape[0], module.heads), spec.value, wide)
        return spec.scale * jax.nn.sigmoid((x @ p[name]["kernel"] + p[name]["bias"]).astype(wide))

    def one_head(network0, q, k, v, theta, mu, alpha):
        """One row's and head's tokens `[S, F]`, `[S]`, scanned token by token."""
        fixed = {name: leaf for name, leaf in network0.items() if name in module.slow}
        head0 = {name: leaf for name, leaf in network0.items() if name not in module.slow}

        def inner(network, params, x):
            """`network` at the updated `params` and the held ones."""
            return network.apply({"params": {**fixed, **params}}, x)

        def token(carry, step):
            state, momentum, start, place = carry
            q_t, k_t, v_t, theta_t, mu_t, alpha_t, first = step
            state, momentum, start, place = jax.tree.map(
                lambda fresh, kept: jnp.where(first, fresh, kept),
                (head0, jax.tree.map(jnp.zeros_like, head0), head0, 0), (state, momentum, start, place))
            start = jax.tree.map(lambda w, s: jnp.where(place % module.chunk == 0, w, s), state, start)
            target = v_t if module.target == "given" else inner(grad_net, start, fed(v_t))
            apply = functools.partial(inner, grad_net)
            g = (jax.grad(lambda params: loss(apply(params, fed(k_t)), target))(start)
                 if gradient is None else gradient(loss, apply, start, fed(k_t), target))
            momentum = jax.tree.map(lambda s, gl: mu_t * s - theta_t * gl, momentum, g)
            state = jax.tree.map(lambda w, s: (1 - alpha_t) * w + s, state, momentum)
            read = start if module.commit == "chunk" else state
            cast = jax.tree.map(lambda leaf: leaf.astype(dtype), {**fixed, **read})
            return (state, momentum, start, place + 1), read_net.apply({"params": cast}, fed(q_t))

        carry = (head0, jax.tree.map(jnp.zeros_like, head0), head0, 0)
        return jax.lax.scan(token, carry, (q, k, v, theta, mu, alpha, jnp.asarray(firsts)))[1]

    @jax.jit
    def run(x, params):
        # Cast here, under the caller's float64 mode where the truth runs.
        p = jax.tree.map(lambda leaf: jnp.asarray(leaf, dtype), params["params"])
        heads = [params["params"][f"head_{head}"] for head in range(module.heads)]
        initial = jax.tree.map(lambda *leaves: jnp.stack(leaves).astype(wide), *heads)
        x = jnp.asarray(x, dtype)
        batch = x.shape[0]
        q, k, v = ((x @ p[f"{name}_proj"]["kernel"]).reshape(batch, LENGTH, module.heads, -1) if projected
                   else x[:, :, None] for name, projected in (("q", module.project_query),
                                                              ("k", module.project_key),
                                                              ("v", module.project_value)))
        if module.normalize:
            q, k = (unit(view) for view in (q, k))
        q, k, v = q.astype(dtype), k.astype(wide), v.astype(wide)

        def gates(row):
            theta = gate("learning_rate", module.learning_rate, row, p)
            mu = gate("momentum", module.momentum, row, p)
            alpha = gate("retention", module.retention, row, p)
            zeros = jnp.zeros_like(theta)
            return theta, zeros if mu is None else mu, zeros if alpha is None else alpha

        theta, mu, alpha = jax.vmap(gates)(x)
        per_head = jax.vmap(one_head, in_axes=(0, 1, 1, 1, 1, 1, 1), out_axes=1)
        reads = jax.vmap(per_head, in_axes=(None, 0, 0, 0, 0, 0, 0))(initial, q, k, v, theta, mu, alpha)
        reads = reads.reshape(batch, LENGTH, -1)
        return reads @ p["o_proj"]["kernel"] if module.output else reads

    return run


def unit(view):
    """`view` over its length on its last axis, eps under the root (FLA's l2norm), in at least float32
    and rounded once to its dtype."""
    wide = view.astype(jnp.promote_types(view.dtype, jnp.float32))
    return (wide / jnp.sqrt(jnp.sum(wide * wide, axis=-1, keepdims=True) + L2NORM_EPS)).astype(view.dtype)


def drawn(module: InContextMemory):
    """A seed's stream and its own weights for `module`: each draw a fresh model, so no one draw of
    the weights is shared by every seed's error."""
    init = jax.jit(lambda key: module.init(key, jnp.zeros((1, LENGTH, WIDTH))))

    def draw(seed: int) -> tuple[np.ndarray, dict]:
        x = np.random.default_rng([1645, seed]).standard_normal((ROWS, LENGTH, WIDTH)).astype(np.float32)
        return x, jax.tree.map(np.asarray, init(jax.random.key(seed)))
    return draw


def applied(module: InContextMemory, call: MixerCall = PLAIN_CALL):
    """`module` over a draw's stream under the draw's weights."""
    return jax.jit(lambda x, params: module.apply(params, x, call))


def head_of(path: tuple[str, ...]) -> int | None:
    """The memory head whose network the scope `path` lies in (a scope `head_<h>`), or None."""
    return next((int(name[5:]) for name in path if re.fullmatch(r"head_\d+", name)), None)


def inner_sites(params: dict, heads: tuple[int, ...] | None = None) -> list[str]:
    """The `/`-joined scopes in `params` of every inner Dense (a module holding a kernel) of a memory
    head's network, of the `heads` given, or every head where None."""
    return sorted({"/".join(path[:-1]) for path in traverse_util.flatten_dict(params)
                   if path[-1] == "kernel" and head_of(path) is not None
                   and (heads is None or head_of(path) in heads)})


def packed(segments: np.ndarray, positions: np.ndarray | None = None) -> MixerCall:
    """A call over rows packed with the documents `segments` numbers, at the logical `positions` given
    (each document's real tokens counted from 0 where None)."""
    laid = coordinates(segments=jnp.asarray(segments), positions=positions, max_segments=2)
    return MixerCall(metadata=AttentionMetadata(coordinates=laid))


def decoded(module: nn.Module):
    """`module` over a draw's stream through its decode cache: a first call allocating it, a prefill of
    six tokens, then one token a call."""
    decode = MixerCall(decode=True)

    @jax.jit
    def run(x, params):
        _, cache = module.apply(params, x[:, :1], decode, mutable=["cache"])
        prefill, cache = module.apply({**params, **cache}, x[:, :6], decode, mutable=["cache"])
        reads = [prefill]
        for t in range(6, LENGTH):
            token, cache = module.apply({**params, **cache}, x[:, t:t + 1], decode, mutable=["cache"])
            reads.append(token)
        return jnp.concatenate(reads, axis=1)

    return run
