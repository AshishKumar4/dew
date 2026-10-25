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


def written(module: InContextMemory, params: dict, x: np.ndarray, *, dtype, segments=None) -> np.ndarray:
    """The paper's update, token by token, in `dtype` (float64 for the truth): the views and gates from
    the layer's projections, the state in at least float32, each gradient at its chunk's start counted
    from its document's first token, and the readout at `dtype`."""
    wide = jnp.promote_types(dtype, jnp.float32)
    p = jax.tree.map(lambda leaf: jnp.asarray(leaf, dtype), params["params"])
    x = jnp.asarray(x, dtype)
    batch, length, _ = x.shape
    views = {name: (x @ p[f"{name}_proj"]["kernel"]).reshape(batch, length, HEADS, FEATURES)
             for name in ("q", "k", "v")}

    def gate(name, spec):
        if spec is None:
            return None
        if spec.value is not None:
            return jnp.full((batch, length, HEADS), spec.value, wide)
        logit = x @ p[name]["kernel"] + p[name]["bias"]
        return spec.scale * jax.nn.sigmoid(logit.astype(wide))

    theta = gate("learning_rate", module.learning_rate)
    mu = gate("momentum", module.momentum)
    alpha = gate("retention", module.retention)
    grad_net, read_net = module.network(dtype=wide), module.network(dtype=dtype)
    initial = jax.tree.map(lambda leaf: jnp.asarray(leaf, wide), params["params"]["initial"])
    segments = np.ones((batch, length), int) if segments is None else segments
    out = np.zeros((batch, length, HEADS, FEATURES), np.float64)

    def gradient(start, k, target):
        return jax.grad(lambda held: module.objective(grad_net.apply({"params": held}, k), target))(start)

    def stepped(momentum, g, kept, rate):
        return jax.tree.map(lambda s, gl: kept * s - rate * gl, momentum, g)

    def retained(state, momentum, a):
        return jax.tree.map(lambda w, s: a * w + s, state, momentum)

    for b in range(batch):
        for h in range(HEADS):
            head0 = jax.tree.map(functools.partial(_head, h), initial)
            place = 0
            for t in range(length):
                if t == 0 or segments[b, t] != segments[b, t - 1]:
                    state, momentum, place = head0, jax.tree.map(jnp.zeros_like, head0), 0
                if place % module.chunk == 0:
                    start = state
                k, v = (views[name][b, t, h].astype(wide) for name in ("k", "v"))
                target = v if module.target == "given" else grad_net.apply({"params": start}, v)
                momentum = stepped(momentum, gradient(start, k, target), 0.0 if mu is None else mu[b, t, h],
                                   theta[b, t, h])
                state = retained(state, momentum, 1.0 if alpha is None else 1 - alpha[b, t, h])
                cast = jax.tree.map(lambda leaf: leaf.astype(dtype), state)
                out[b, t, h] = np.asarray(read_net.apply({"params": cast}, views["q"][b, t, h]), np.float64)
                place += 1
    read = jnp.asarray(out.reshape(batch, length, -1), dtype)
    return np.asarray(read @ p["o_proj"]["kernel"])


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
    assert_as_exact(jax.jit(lambda x: module.apply(params, x)),
                    lambda x: written(module, params, x, dtype=jnp.float32), draw,
                    f"{name} memory", seeds=MEMORY_SEEDS,
                    truth_fn=lambda x: written(module, params, x, dtype=jnp.float64))


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
                    seeds=MEMORY_SEEDS, truth_fn=lambda x: written(module, params, x, dtype=jnp.float64))


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
    """With no step, every token reads the initial network over its query, bit for bit, as that
    network computes it: the update adds exact zeros."""
    module = InContextMemory(network=functools.partial(ResidualMemory, hidden=4 * FEATURES), heads=HEADS,
                             head_features=FEATURES, learning_rate=Gate(value=0.0), output=False,
                             chunk=CHUNK)
    params = module.init(jax.random.key(0), jnp.zeros((1, LENGTH, WIDTH)))
    x = draw(2)[0]
    read = np.asarray(module.apply(params, x)).reshape(2, LENGTH, HEADS, FEATURES)
    network = ResidualMemory(hidden=4 * FEATURES)
    queries = (x @ np.asarray(params["params"]["q_proj"]["kernel"])).reshape(2, LENGTH, HEADS, FEATURES)
    for h in range(HEADS):
        head = jax.tree.map(functools.partial(_head, h), params["params"]["initial"])
        expected = network.apply({"params": head}, queries[:, :, h])
        np.testing.assert_array_equal(read[:, :, h], np.asarray(expected))
