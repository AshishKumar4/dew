"""The in-context memory (`dew.nn.memory`, ledger 1645): each configuration against its paper's update
equations, written out token by token below (`written`), at check 1 against float64; the chunk-parallel
computation against the token-by-token one; packed documents as each alone; a decode as the full pass;
and a zero step size reading the initial network."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from reference_error import assert_as_exact

from dew.nn.inputs import AttentionMetadata, MixerCall, coordinates
from dew.nn.memory import Gate, InContextMemory, LinearMemory, MLPMemory, Objective, ResidualMemory

WIDTH, HEADS, FEATURES, LENGTH, CHUNK = 8, 2, 4, 12, 4
MEMORY_SEEDS = 32
"""The declared seeds for check 1 of the memories (`assert_as_exact`), sized by `seeds_for` on armada."""

CONFIGURATIONS = {
    # TTT-Linear (Sec 2.7): f(x) = x + LN(W x), the L2 reconstruction (Eq 4), eta(x) = sigmoid(theta_lr x).
    "ttt_linear": {"network": functools.partial(ResidualMemory, hidden=None, norm=True)},
    # TTT-MLP: a 4x GELU MLP inside the residual and norm, eta_base 0.1.
    "ttt_mlp": {"network": functools.partial(ResidualMemory, hidden=4 * FEATURES, activation="gelu"),
                "learning_rate": Gate(scale=0.1)},
    # Titans' deep memory with momentum and forgetting (Eq 12-14).
    "titans": {"network": functools.partial(MLPMemory, widths=(2 * FEATURES, FEATURES)),
               "momentum": Gate(), "retention": Gate(bias=-2.0), "learning_rate": Gate(scale=0.5)},
    # Moneta: the lp bias with p = 3 (Miras Eq 10, 24) and retention.
    "moneta": {"network": functools.partial(ResidualMemory, hidden=4 * FEATURES),
               "objective": Objective("lp", p=3.0), "retention": Gate(bias=-2.0),
               "learning_rate": Gate(scale=0.1)},
    # Yaad: Huber's bias (Miras Eq 13, 26).
    "yaad": {"network": functools.partial(ResidualMemory, hidden=4 * FEATURES),
             "objective": Objective("huber", delta=0.5), "retention": Gate(bias=-2.0)},
    # Nested Learning's self-modifying memory: a residual MLP (Eq 89), its own target (Eq 84), retention.
    "self_modifying": {"network": functools.partial(ResidualMemory, hidden=2 * FEATURES, norm=False),
                       "target": "self_generated", "retention": Gate(bias=-2.0)},
    # Linear attention: a linear memory on the dot-product bias, every token (Nested Learning Eq 14-18).
    "linear_attention": {"network": functools.partial(LinearMemory, features=FEATURES),
                         "objective": Objective("dot"), "chunk": 1},
}


def layer(name: str, dtype=jnp.float32) -> InContextMemory:
    fields = {"chunk": CHUNK, **CONFIGURATIONS[name]}
    return InContextMemory(heads=HEADS, head_features=FEATURES, dtype=dtype, **fields)


def written(module: InContextMemory, params: dict, *, dtype, segments=None):
    """The paper's update as a function of the stream, token by token, in `dtype` (float64 for the
    truth): the views and gates from the layer's projections, the state in at least float32, each
    gradient at its chunk's start counted from its document's first token (`segments` `[S]`, one
    document where None), and the readout at `dtype`."""
    wide = jnp.promote_types(dtype, jnp.float32)
    p = jax.tree.map(lambda leaf: jnp.asarray(leaf, dtype), params["params"])
    grad_net, read_net = module.network(dtype=wide), module.network(dtype=dtype)
    initial = jax.tree.map(lambda leaf: jnp.asarray(leaf, wide), params["params"]["initial"])
    firsts = np.ones(LENGTH, bool) if segments is None else np.r_[True, segments[1:] != segments[:-1]]

    def gate(name, spec, x):
        if spec is None:
            return None
        if spec.value is not None:
            return jnp.full((x.shape[0], HEADS), spec.value, wide)
        return spec.scale * jax.nn.sigmoid((x @ p[name]["kernel"] + p[name]["bias"]).astype(wide))

    def one_head(head0, q, k, v, theta, mu, alpha):
        """One row's and head's tokens `[S, F]`, `[S]`, scanned token by token."""
        def token(carry, step):
            state, momentum, start, place = carry
            q_t, k_t, v_t, theta_t, mu_t, alpha_t, first = step
            state, momentum, start, place = jax.tree.map(
                lambda fresh, held: jnp.where(first, fresh, held),
                (head0, jax.tree.map(jnp.zeros_like, head0), head0, 0), (state, momentum, start, place))
            start = jax.tree.map(lambda w, s: jnp.where(place % module.chunk == 0, w, s), state, start)
            target = v_t if module.target == "given" else grad_net.apply({"params": start}, v_t)
            g = jax.grad(lambda held: module.objective(grad_net.apply({"params": held}, k_t), target))(start)
            momentum = jax.tree.map(lambda s, gl: mu_t * s - theta_t * gl, momentum, g)
            state = jax.tree.map(lambda w, s: (1 - alpha_t) * w + s, state, momentum)
            cast = jax.tree.map(lambda leaf: leaf.astype(dtype), state)
            return (state, momentum, start, place + 1), read_net.apply({"params": cast}, q_t)

        carry = (head0, jax.tree.map(jnp.zeros_like, head0), head0, 0)
        return jax.lax.scan(token, carry, (q, k, v, theta, mu, alpha, jnp.asarray(firsts)))[1]

    @jax.jit
    def run(x):
        x = jnp.asarray(x, dtype)
        batch = x.shape[0]
        q, k, v = ((x @ p[f"{name}_proj"]["kernel"]).reshape(batch, LENGTH, HEADS, FEATURES).astype(
            wide if name != "q" else dtype) for name in ("q", "k", "v"))

        def gates(row):
            theta = gate("learning_rate", module.learning_rate, row)
            mu = gate("momentum", module.momentum, row)
            alpha = gate("retention", module.retention, row)
            zeros = jnp.zeros_like(theta)
            return theta, zeros if mu is None else mu, zeros if alpha is None else alpha

        theta, mu, alpha = jax.vmap(gates)(x)
        per_head = jax.vmap(one_head, in_axes=(0, 1, 1, 1, 1, 1, 1), out_axes=1)
        reads = jax.vmap(per_head, in_axes=(None, 0, 0, 0, 0, 0, 0))(initial, q, k, v, theta, mu, alpha)
        return reads.reshape(batch, LENGTH, -1) @ p["o_proj"]["kernel"]

    return run


def _head(index: int, leaf):
    return leaf[index]


def draw(seed: int) -> tuple[np.ndarray]:
    return (np.random.default_rng([1645, seed]).standard_normal((2, LENGTH, WIDTH)).astype(np.float32),)


@pytest.mark.parametrize("name", sorted(CONFIGURATIONS))
def test_a_memory_follows_its_papers_update(name):
    """Each configuration computes its paper's update and readout token by token (`written`), on par
    with that written-out update in float32, against float64 (check 1)."""
    module = layer(name)
    params = module.init(jax.random.key(0), jnp.zeros((1, LENGTH, WIDTH)))
    assert_as_exact(jax.jit(lambda x: module.apply(params, x)), written(module, params, dtype=jnp.float32),
                    draw, f"{name} memory", seeds=MEMORY_SEEDS,
                    truth_fn=written(module, params, dtype=jnp.float64))


def packed(segments: np.ndarray) -> MixerCall:
    """A call over rows packed with the documents `segments` numbers."""
    laid = coordinates(segments=jnp.asarray(segments), max_segments=2)
    return MixerCall(metadata=AttentionMetadata(coordinates=laid))


@pytest.mark.parametrize("name", ["titans", "self_modifying"])
def test_the_chunk_parallel_memory_is_the_token_by_token_one(name):
    """A row of one document runs chunk by chunk; the same row as a packed call runs token by token,
    the definition. The two are on par against float64 (check 1)."""
    module = layer(name)
    params = module.init(jax.random.key(0), jnp.zeros((1, LENGTH, WIDTH)))
    one = np.ones((2, LENGTH), np.int32)
    stepped = jax.jit(lambda x: module.apply(params, x, packed(one)))
    assert_as_exact(jax.jit(lambda x: module.apply(params, x)), stepped, draw, f"{name} chunked",
                    seeds=MEMORY_SEEDS, truth_fn=written(module, params, dtype=jnp.float64))


def test_packed_documents_read_as_each_alone():
    """Two documents packed in a row each start from the initial network, their chunks counted from
    their own first token: each reads, bit for bit, as the document alone."""
    module = layer("titans")
    params = module.init(jax.random.key(0), jnp.zeros((1, LENGTH, WIDTH)))
    x = draw(0)[0]
    cut = 5
    segments = np.where(np.arange(LENGTH) < cut, 1, 2)[None].repeat(2, 0)
    together = np.asarray(module.apply(params, x, packed(segments)))
    first = np.asarray(module.apply(params, x[:, :cut], packed(np.ones((2, cut), np.int32))))
    second = np.asarray(module.apply(params, x[:, cut:], packed(np.ones((2, LENGTH - cut), np.int32))))
    np.testing.assert_array_equal(together, np.concatenate([first, second], axis=1))


def test_a_decode_continues_the_full_pass():
    """A prefill and then one token a call, through the cache, reads each token as the full pass
    token by token does, bit for bit, momentum and chunk boundaries carried across the calls."""
    module = layer("titans")
    params = module.init(jax.random.key(0), jnp.zeros((1, LENGTH, WIDTH)))
    x = draw(1)[0]
    full = np.asarray(module.apply(params, x, packed(np.ones((2, LENGTH), np.int32))))
    _, cache = module.apply(params, x[:, :1], MixerCall(decode=True), mutable=["cache"])
    reads = []
    prefill, cache = module.apply({**params, **cache}, x[:, :6], MixerCall(decode=True), mutable=["cache"])
    reads.append(np.asarray(prefill))
    for t in range(6, LENGTH):
        token, cache = module.apply({**params, **cache}, x[:, t:t + 1], MixerCall(decode=True),
                                    mutable=["cache"])
        reads.append(np.asarray(token))
    np.testing.assert_array_equal(np.concatenate(reads, axis=1), full)


def test_a_zero_step_reads_the_initial_network():
    """With no step, every token reads the initial network over its query, as that network computes
    it over the queries, on par against float64 (check 1): the update adds exact zeros."""
    module = InContextMemory(network=functools.partial(ResidualMemory, hidden=4 * FEATURES), heads=HEADS,
                             head_features=FEATURES, learning_rate=Gate(value=0.0), output=False,
                             chunk=CHUNK)
    params = module.init(jax.random.key(0), jnp.zeros((1, LENGTH, WIDTH)))

    def initial(x):
        kernel = params["params"]["q_proj"]["kernel"].astype(x.dtype)
        queries = (x @ kernel).reshape(-1, LENGTH, HEADS, FEATURES)
        network = ResidualMemory(hidden=4 * FEATURES, dtype=x.dtype)
        stored = params["params"]["initial"]
        heads = [network.apply({"params": jax.tree.map(functools.partial(_head, h), stored)},
                               queries[:, :, h]) for h in range(HEADS)]
        return jnp.stack(heads, axis=2).reshape(-1, LENGTH, HEADS * FEATURES)

    assert_as_exact(jax.jit(lambda x: module.apply(params, x)), jax.jit(initial), draw, "zero-step memory",
                    seeds=MEMORY_SEEDS, truth_fn=initial)
