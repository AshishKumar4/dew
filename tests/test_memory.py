"""The in-context memory (`dew.nn.memory`, ledger 1645): each configuration against its paper's update
equations, written out token by token below (`written`), at check 1 against float64; the chunk-parallel
computation against the token-by-token one; packed documents as each alone; a decode as the full pass;
and a zero step size reading the initial network."""

import functools
import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import linen as nn
from flax.typing import Dtype
from reference_error import assert_as_exact, with_gradients

from dew.nn.inputs import PLAIN_CALL, AttentionMetadata, MixerCall, coordinates
from dew.nn.memory import Gate, InContextMemory, LinearMemory, MLPMemory, Objective, ResidualMemory
from dew.nn.mlp import GatedMLP
from dew.nn.precision import L2NORM_EPS

WIDTH, HEADS, FEATURES, LENGTH, CHUNK, ROWS = 32, 2, 16, 12, 4, 8
MEMORY_SEEDS = 8
"""The declared seeds for check 1 of the memories (`assert_as_exact`): `seeds_for` on armada sizes every
configuration, its token-by-token path included, at 8 over the drawn weights."""

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
            jnp.linalg.norm(prediction - target) <= objective.delta,
            jnp.dot(prediction - target, prediction - target) / 2,
            objective.delta * jnp.linalg.norm(prediction - target, 1)),
    }
    return losses[kind]


def written(module: InContextMemory, *, dtype, segments=None, gradient=None):
    """The paper's update as a function of the stream, token by token, in `dtype` (float64 for the
    truth): the views and gates from the layer's projections (or the stream itself), the state in at
    least float32, each gradient at its chunk's start counted from its document's first token
    (`segments` `[S]`, one document where None) of the loss its paper writes for the layer's objective
    (`paper_loss`), the network's `slow` parameters left at W_0, and the readout at `dtype` of the
    state each token updated, or, where a chunk commits its steps at its end, of the state its chunk
    started from (Nested Learning Eq 71)."""
    wide = jnp.promote_types(dtype, jnp.float32)
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
            target = v_t if module.target == "given" else inner(grad_net, start, v_t)
            apply = functools.partial(inner, grad_net)
            g = (jax.grad(lambda params: loss(apply(params, k_t), target))(start)
                 if gradient is None else gradient(loss, apply, start, k_t, target))
            momentum = jax.tree.map(lambda s, gl: mu_t * s - theta_t * gl, momentum, g)
            state = jax.tree.map(lambda w, s: (1 - alpha_t) * w + s, state, momentum)
            read = start if module.commit == "chunk" else state
            cast = jax.tree.map(lambda leaf: leaf.astype(dtype), {**fixed, **read})
            return (state, momentum, start, place + 1), read_net.apply({"params": cast}, q_t)

        carry = (head0, jax.tree.map(jnp.zeros_like, head0), head0, 0)
        return jax.lax.scan(token, carry, (q, k, v, theta, mu, alpha, jnp.asarray(firsts)))[1]

    @jax.jit
    def run(x, params):
        # Cast here, under the caller's float64 mode where the truth runs.
        p = jax.tree.map(lambda leaf: jnp.asarray(leaf, dtype), params["params"])
        initial = jax.tree.map(lambda leaf: jnp.asarray(leaf, wide), params["params"]["initial"])
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


def _head(index: int, leaf):
    return leaf[index]


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


@pytest.mark.parametrize("name", sorted(CONFIGURATIONS))
def test_a_memory_follows_its_papers_update(name):
    """Each configuration computes its paper's update and readout token by token (`written`), on par
    with that written-out update in float32, against float64 (check 1)."""
    module = layer(name)
    assert_as_exact(applied(module), written(module, dtype=jnp.float32), drawn(module), f"{name} memory",
                    seeds=MEMORY_SEEDS, truth_fn=written(module, dtype=jnp.float64))


def test_yaads_objective_switches_by_the_whole_errors_norm():
    """Miras Eq 16 steps the l2 gradient while the error's norm is within delta and delta times the l1
    gradient past it, over every coordinate alike: an error of (2, 0.1) steps (0.5, 0.5) at delta
    0.5, where each coordinate's Huber would step (0.5, 0.1), and steps the error itself at delta 3."""
    prediction, target = jnp.asarray([2.0, 0.1]), jnp.zeros(2)
    for delta, step in ((0.5, [0.5, 0.5]), (3.0, [2.0, 0.1])):
        gradient = jax.grad(Objective("l2_or_l1", delta=delta))(prediction, target)
        np.testing.assert_array_equal(np.asarray(gradient), np.asarray(step, np.float32))


def test_an_l2_linear_memory_runs_the_gated_delta_rule():
    """A linear memory on the l2 objective that steps every token is the delta rule: traced for an
    H100, its call takes the gated delta rule's kernels through the shared recurrence's dispatch, and
    the same memory chunked by four, whose gradients wait for their chunk's start, decides no such
    route."""
    from dew.nn.kernels import generation

    def routes(module: InContextMemory) -> list[tuple[str, str]]:
        x = jax.ShapeDtypeStruct((ROWS, LENGTH, WIDTH), jnp.float32)
        params = jax.eval_shape(module.init, jax.random.key(0), x)
        with generation.assumed("sm90"), generation.routes() as found:
            jax.eval_shape(module.apply, params, x)
        return [(route.choice, route.kernel) for route in found if route.choice == "gated_delta_rule"]

    assert routes(layer("delta_rule")) == [("gated_delta_rule", "fla")]
    assert routes(layer("delta_rule").clone(chunk=CHUNK)) == []


class Product(nn.Module):
    """`k W`, a linear memory's function as a network of this test's own, which the layer runs on its
    general path rather than the shared recurrence."""

    dtype: Dtype | None = None

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        kernel = self.param("kernel", nn.initializers.lecun_normal(), (x.shape[-1], x.shape[-1]))
        return jnp.dot(x, kernel.astype(self.dtype or x.dtype))


def on_the_general_path(params: dict) -> dict:
    """A linear memory's weights as the same memory over `Product` holds them."""
    initial = params["params"]["initial"]
    return {"params": {**params["params"], "initial": {"kernel": initial["kernel"]["kernel"]}}}


@pytest.mark.parametrize("name", ["linear_attention", "gated_linear_attention", "delta_rule"])
def test_a_linear_memory_on_the_recurrence_is_its_general_path(name):
    """Each linear memory the shared recurrence runs computes what the same memory computes on the
    general path, over rows of two packed documents, W_0 restarting each: on par against float64
    (check 1)."""
    module = layer(name)
    general = module.clone(network=Product())
    segments = np.where(np.arange(LENGTH) < 5, 1, 2).astype(np.int32)
    call = packed(np.broadcast_to(segments, (ROWS, LENGTH)))
    reference = jax.jit(lambda x, params: general.apply(on_the_general_path(params), x, call))
    assert_as_exact(applied(module, call), reference, drawn(module), f"{name} on the recurrence",
                    seeds=MEMORY_SEEDS, truth_fn=written(module, dtype=jnp.float64, segments=segments))


def test_a_self_generated_target_carries_the_outer_gradient():
    """The self-modifying memory trains through its own target: the gradients of its stream and of
    every parameter, the value projection's included, which reaches the output only through the target
    the network makes of each value, are the token-by-token reference's, on par against float64
    (check 1). The inner gradient holds the target fixed; the outer one does not, so the value
    projection's gradient is not zero."""
    module = layer("self_modifying")
    draw = drawn(module)

    def draw_with_cotangent(seed: int):
        cotangent = np.random.default_rng([1647, seed]).standard_normal((ROWS, LENGTH, WIDTH))
        return *draw(seed), cotangent.astype(np.float32)

    run = jax.jit(with_gradients(lambda x, params: module.apply(params, x)))
    assert_as_exact(run, jax.jit(with_gradients(written(module, dtype=jnp.float32))), draw_with_cotangent,
                    "self-generated gradients", seeds=MEMORY_SEEDS,
                    truth_fn=with_gradients(written(module, dtype=jnp.float64)))
    gradient = run(*draw_with_cotangent(0))[2]["params"]["v_proj"]["kernel"]
    assert np.abs(np.asarray(gradient)).max() > 0


def packed(segments: np.ndarray, positions: np.ndarray | None = None) -> MixerCall:
    """A call over rows packed with the documents `segments` numbers, at the logical `positions` given
    (each document's real tokens counted from 0 where None)."""
    laid = coordinates(segments=jnp.asarray(segments), positions=positions, max_segments=2)
    return MixerCall(metadata=AttentionMetadata(coordinates=laid))


@pytest.mark.parametrize("name", ["titans", "self_modifying"])
def test_the_chunk_parallel_memory_is_the_token_by_token_one(name):
    """A row of one document runs chunk by chunk; the same row as a packed call runs token by token,
    the definition. The two are on par against float64 (check 1)."""
    module = layer(name)
    stepped = applied(module, packed(np.ones((ROWS, LENGTH), np.int32)))
    assert_as_exact(applied(module), stepped, drawn(module), f"{name} chunked", seeds=MEMORY_SEEDS,
                    truth_fn=written(module, dtype=jnp.float64))


def test_packed_documents_read_as_each_alone():
    """Two documents packed in a row each start from the initial network, their chunks counted from
    their own first token: each reads, bit for bit, as the document alone."""
    module = layer("titans")
    x, params = drawn(module)(0)
    cut = 5
    segments = np.where(np.arange(LENGTH) < cut, 1, 2)[None].repeat(ROWS, 0)
    together = np.asarray(module.apply(params, x, packed(segments)))
    first = np.asarray(module.apply(params, x[:, :cut], packed(np.ones((ROWS, cut), np.int32))))
    second = np.asarray(module.apply(params, x[:, cut:], packed(np.ones((ROWS, LENGTH - cut), np.int32))))
    np.testing.assert_array_equal(together, np.concatenate([first, second], axis=1))


def test_logical_positions_leave_a_documents_chunks_in_place():
    """A memory counts its chunks over each document's real tokens from its first, whatever logical
    positions the call gives (a continuation's or a canvas's origin): a packed row whose positions
    start at 1 reads, bit for bit, as the same row with positions from 0."""
    module = layer("titans")
    x, params = drawn(module)(0)
    segments = np.ones((ROWS, LENGTH), np.int32)
    shifted = np.broadcast_to(np.arange(1, LENGTH + 1, dtype=np.int32), (ROWS, LENGTH))
    np.testing.assert_array_equal(np.asarray(module.apply(params, x, packed(segments, shifted))),
                                  np.asarray(module.apply(params, x, packed(segments))))


@pytest.mark.parametrize("name", ["titans", "cms", "gated_linear_attention", "delta_rule"])
def test_a_decode_continues_the_full_pass(name):
    """A prefill and then one token a call, through the cache, reads each token as the full pass does,
    momentum, chunk boundaries and a level's network as its last scheduled update committed it carried
    across the calls: on par against float64 (check 1)."""
    module = layer(name)
    decode = MixerCall(decode=True)

    @jax.jit
    def decoded(x, params):
        _, cache = module.apply(params, x[:, :1], decode, mutable=["cache"])
        prefill, cache = module.apply({**params, **cache}, x[:, :6], decode, mutable=["cache"])
        reads = [prefill]
        for t in range(6, LENGTH):
            token, cache = module.apply({**params, **cache}, x[:, t:t + 1], decode, mutable=["cache"])
            reads.append(token)
        return jnp.concatenate(reads, axis=1)

    full = applied(module, packed(np.ones((ROWS, LENGTH), np.int32)))
    assert_as_exact(decoded, full, drawn(module), f"decoded {name} memory", seeds=MEMORY_SEEDS,
                    truth_fn=written(module, dtype=jnp.float64))


@pytest.mark.parametrize("name", ["titans", "gated_linear_attention"])
def test_a_cached_prefill_starts_each_packed_document_afresh(name):
    """A cached prefill of packed documents starts each from the initial network, its chunks counted
    from its own first token, and the calls after it continue its last document: a prefill of a
    three-token document and five tokens of the next, then that document's last four one call each,
    reads each token as the full packed pass does, on par against float64 (check 1)."""
    module = layer(name)
    decode = MixerCall(decode=True)
    segments = np.where(np.arange(LENGTH) < 3, 1, 2).astype(np.int32)
    prefill = MixerCall(decode=True, metadata=AttentionMetadata(coordinates=coordinates(
        segments=jnp.broadcast_to(jnp.asarray(segments[:8]), (ROWS, 8)), max_segments=2)))

    @jax.jit
    def decoded(x, params):
        _, cache = module.apply(params, x[:, :1], decode, mutable=["cache"])
        read, cache = module.apply({**params, **cache}, x[:, :8], prefill, mutable=["cache"])
        reads = [read]
        for t in range(8, LENGTH):
            token, cache = module.apply({**params, **cache}, x[:, t:t + 1], decode, mutable=["cache"])
            reads.append(token)
        return jnp.concatenate(reads, axis=1)

    full = applied(module, packed(np.broadcast_to(segments, (ROWS, LENGTH))))
    assert_as_exact(decoded, full, drawn(module), f"packed cached {name} memory", seeds=MEMORY_SEEDS,
                    truth_fn=written(module, dtype=jnp.float64, segments=segments))


def test_a_zero_step_reads_the_initial_network():
    """With no step, every token reads the initial network over its query, as that network computes
    it over the queries, on par against float64 (check 1): the update adds exact zeros."""
    module = InContextMemory(network=ResidualMemory(hidden=4 * FEATURES), heads=HEADS,
                             head_features=FEATURES, learning_rate=Gate(value=0.0), output=False,
                             chunk=CHUNK)

    def initial(x, params):
        kernel = params["params"]["q_proj"]["kernel"].astype(x.dtype)
        queries = (x @ kernel).reshape(-1, LENGTH, HEADS, FEATURES)
        network = ResidualMemory(hidden=4 * FEATURES, dtype=x.dtype)
        stored = params["params"]["initial"]
        heads = [network.apply({"params": jax.tree.map(functools.partial(_head, h), stored)},
                               queries[:, :, h]) for h in range(HEADS)]
        return jnp.stack(heads, axis=2).reshape(-1, LENGTH, HEADS * FEATURES)

    assert_as_exact(applied(module), jax.jit(initial), drawn(module), "zero-step memory", seeds=MEMORY_SEEDS,
                    truth_fn=initial)


def test_an_orthogonalized_step_follows_the_paper_with_muons_iteration():
    """With `orthogonalize`, each matrix of a token's gradient steps as Muon's quintic Newton-Schulz of it
    (`dew.optim.newton_schulz`, Nested Learning Sec 4.2) and its vectors as they are: the layer is the
    paper's update with that map applied to its gradients, on par against float64 (check 1)."""
    from dew.optim import newton_schulz

    module = layer("titans").clone(orthogonalize=5)

    def gradient(loss, apply, start, k, target):
        """The paper's gradient with Muon's map on each matrix, in the reference's own precision."""
        g = jax.grad(lambda held: loss(apply(held, k), target))(start)
        return jax.tree.map(lambda leaf: newton_schulz(leaf, 5) if leaf.ndim == 2 else leaf, g)

    assert_as_exact(applied(module), written(module, dtype=jnp.float32, gradient=gradient), drawn(module),
                    "orthogonalized memory", seeds=MEMORY_SEEDS,
                    truth_fn=written(module, dtype=jnp.float64, gradient=gradient))


def decoder(**fields):
    """A tiny decoder alternating attention and Titans-style memory layers."""
    from model_support import TINY_DECODER

    from dew.nn.backbones.causal_transformer import CausalTransformer
    from dew.nn.backbones.layer_plan import LayerKind
    from dew.nn.mixers.memory import InContextMemoryMixer

    memory = InContextMemory(network=MLPMemory(hidden=(16,)), heads=2, head_features=8, chunk=4,
                             learning_rate=Gate(scale=0.5), momentum=Gate(), retention=Gate(bias=-2.0))
    return CausalTransformer(**{"vocab_size": 37, **TINY_DECODER, "num_layers": 4, "dtype": "float32",
                                "layer_types": ("memory", "attention") * 2,
                                "kinds": {"memory": LayerKind(mixer=InContextMemoryMixer(memory))}, **fields})


def test_a_memory_layer_kind_trains_scans_and_decodes_in_a_decoder():
    """A decoder whose layer kind is a memory: its scanned stack computes the plain loop's logits bit for
    bit from the same tree; one gradient step changes the memories' initial networks and projections;
    and its cached decode, token by token after a prefill, gives the full pass's last logits on par
    against float64 (check 1)."""
    plain, scanned = decoder(), decoder(scan_layers=True)
    ids = jax.random.randint(jax.random.key(1), (2, LENGTH), 0, 37)
    variables = plain.init(jax.random.key(0), ids)
    np.testing.assert_array_equal(np.asarray(plain.apply(variables, ids)),
                                  np.asarray(scanned.apply(variables, ids)))
    loss = jax.jit(jax.grad(lambda held: jnp.mean(plain.apply(held, ids) ** 2)))
    gradient = loss(variables)["params"]
    memory = gradient["layers_0"]["self_attn"]
    assert {"initial", "q_proj", "k_proj", "v_proj", "o_proj", "learning_rate", "momentum",
            "retention"} <= set(memory)
    assert all(np.abs(np.asarray(leaf)).max() > 0 for leaf in jax.tree.leaves(memory["initial"]))

    def decoded(tokens):
        cache = plain.apply(variables, tokens.shape[0], method=plain.init_cache, mutable=["cache"])[1]
        logits, cache = plain.apply({**variables, **cache}, tokens[:, :6], decode=True, mutable=["cache"])
        for t in range(6, LENGTH):
            logits, cache = plain.apply({**variables, **cache}, tokens[:, t:t + 1], decode=True,
                                        mutable=["cache"])
        return logits[:, -1]

    def full(tokens):
        return plain.apply(variables, tokens)[:, -1]

    def draw_ids(seed):
        return (np.random.default_rng([1646, seed]).integers(0, 37, (2, LENGTH), dtype=np.int32),)

    wide = decoder(dtype="float64")

    def truth(tokens):
        widened = jax.tree.map(lambda leaf: jnp.asarray(leaf, jnp.float64), variables)
        return wide.apply(widened, tokens)[:, -1]

    assert_as_exact(jax.jit(decoded), jax.jit(full), draw_ids, "memory decoder decode", seeds=MEMORY_SEEDS,
                    truth_fn=truth)


def test_muon_steps_each_memory_heads_matrices_on_their_own():
    """A memory decoder trains under Muon: each head's inner matrices are matrices Muon orthogonalizes on
    their own, by Linen's kernel convention (inputs into outputs), its biases take AdamW's update, and
    one step of the optimizer runs over the whole tree."""
    import optax

    from dew.config import OptimConfig
    from dew.optim import muon_weight_dimension_numbers

    model = decoder()
    params = model.init(jax.random.key(0), jnp.zeros((2, LENGTH), jnp.int32))["params"]
    numbers = muon_weight_dimension_numbers(params)["layers_0"]["self_attn"]["initial"]
    layers = {f"layers_{index}": {"kernel": optax.contrib.MuonDimensionNumbers(), "bias": None}
              for index in (0, 1)}
    assert numbers == {f"head_{head}": layers for head in range(2)}
    solver = OptimConfig(optimizer="muon", learning_rate=1e-3).build(10)
    updates, _ = solver.update(jax.tree.map(jnp.ones_like, params), solver.init(params), params)
    assert all(np.isfinite(np.asarray(leaf)).all() for leaf in jax.tree.leaves(updates))


def test_a_memory_layer_kind_records_and_rebuilds():
    """A memory layer kind records the memory's whole configuration: a decoder whose two kinds set every
    field a memory computes by away from its default (heads, an existing GatedMLP as a level's network
    over the raw stream with no output projection, an lp and a Huber objective, a self-generated
    target, constant and scaled gates, momentum, retention, Newton-Schulz steps, the chunk and its
    commit, weights the inner loop holds at W_0) writes a JSON record that rebuilds the same model,
    which computes the original's logits bit for bit."""
    from model_support import TINY_DECODER

    from dew.config import ModelConfig
    from dew.nn.backbones.causal_transformer import CausalTransformer
    from dew.nn.backbones.layer_plan import LayerKind
    from dew.nn.mixers.memory import InContextMemoryMixer
    from dew.registry import to_record

    heads = InContextMemory(network=MLPMemory(hidden=(24,), activation="gelu"), heads=4, head_features=8,
                            objective=Objective("lp", p=3.0), target="self_generated",
                            learning_rate=Gate(scale=0.05, bias=0.5), momentum=Gate(value=0.5),
                            retention=Gate(scale=0.5, bias=-1.0), chunk=2, orthogonalize=3, normalize=True)
    level = InContextMemory(network=GatedMLP(hidden_features=48, out_features=32), project_query=False,
                            project_key=False, project_value=False, output=False,
                            objective=Objective("huber", delta=0.5), learning_rate=Gate(value=0.01), chunk=3,
                            commit="chunk", slow=("gate_proj",))
    model = CausalTransformer(**{**TINY_DECODER, "vocab_size": 37, "num_layers": 3, "dtype": jnp.float32,
                                 "layer_types": ("heads", "level", "attention"),
                                 "kinds": {"heads": LayerKind(mixer=InContextMemoryMixer(heads)),
                                           "level": LayerKind(mixer=InContextMemoryMixer(level)),
                                           "attention": LayerKind()}})
    record = json.loads(json.dumps(to_record(ModelConfig.from_model(model), ModelConfig)))
    rebuilt = ModelConfig.from_dict(record).build()
    assert rebuilt == model
    ids = jax.random.randint(jax.random.key(1), (2, LENGTH), 0, 37)
    variables = model.init(jax.random.key(0), ids)
    np.testing.assert_array_equal(np.asarray(rebuilt.apply(variables, ids)),
                                  np.asarray(model.apply(variables, ids)))
