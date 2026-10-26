"""The in-context memory (`dew.nn.memory`, ledger 1645): each configuration against its paper's update
equations, written out token by token below (`written`), at check 1 against float64; the chunk-parallel
computation against the token-by-token one; the linear memories on the shared recurrence against their
general path; packed documents as each alone; a decode, and a cached packed prefill, as the full pass;
the outer gradients through a self-generated target; a zero step size reading the initial network; and
the memory as a decoder's layer kind, scanned, rematted, decoded, under Muon and recorded."""

import functools
import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import linen as nn, traverse_util
from flax.typing import Dtype
from reference_error import assert_as_exact, with_gradients

from dew.nn.inputs import PLAIN_CALL, AttentionMetadata, MixerCall, coordinates
from dew.nn.memory import CARRY, Gate, InContextMemory, LinearMemory, MLPMemory, Objective, ResidualMemory
from dew.nn.mlp import GatedMLP
from dew.nn.precision import L2NORM_EPS

WIDTH, HEADS, FEATURES, LENGTH, CHUNK, ROWS = 32, 2, 16, 12, 4, 8
MEMORY_SEEDS = 8
"""The declared seeds for check 1 of the memories (`assert_as_exact`): `seeds_for` on armada sizes every
configuration, its token-by-token, decoded, packed, cached and recurrence paths included, at 8 over the
drawn weights."""
SEEDS = {"self-generated gradients": 256, "memory decoder gradient": 256, "memory decoder decode": 32,
         "adapted memory": 8}
"""The sites `seeds_for` on armada sizes past `MEMORY_SEEDS`: the self-modifying memory's and the memory
decoder's gradients, every parameter's leaf, at 256, each reference compared with itself, and the
decoder's last logits after a cached decode at 32."""

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
        heads = [params["params"]["initial"][f"head_{head}"] for head in range(module.heads)]
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


@pytest.mark.parametrize(("dtype", "features", "chosen", "kernel", "verdict"), [
    ("float32", FEATURES, {}, "step", "kernel"),
    ("float64", FEATURES, {}, "xla", "inherent"),
    ("float32", 12, {}, "xla", "gap: roadmap: recurrence step widths"),
    ("float32", FEATURES, {"delta_decode": "fused"}, "step", "inherent")],
    ids=["fp32", "float64", "width_12", "fused_chosen"])
def test_a_cached_linear_memorys_token_takes_the_shared_decode_dispatch(dtype, features, chosen, kernel,
                                                                        verdict):
    """A linear memory's cached token, traced for an H100, steps through the shared decode dispatch: the
    recurrence's step kernel where it takes the state, fp32 of power-of-two widths, else the portable
    step, so a float64 memory keeps its own precision and a head width the kernel's blocks do not take
    never reaches its launch; with the fused step chosen, which a memory has nothing to fuse into, the
    step. Each route off the preferred kernel says why, as the kernel-route audit reads it."""
    from dew.nn.kernels import generation
    from tools import kernel_routes

    module = layer("delta_rule", dtype=jnp.dtype(dtype)).clone(head_features=features)
    decode = MixerCall(decode=True)
    with jax.enable_x64(new_val=dtype == "float64"):
        x = jax.ShapeDtypeStruct((ROWS, 1, WIDTH), jnp.dtype(dtype))
        params = jax.eval_shape(module.init, jax.random.key(0), x)
        _, cache = jax.eval_shape(lambda held, x: module.apply(held, x, decode, mutable=["cache"]),
                                  params, x)
        with generation.assumed("sm90"), generation.chosen(**chosen), generation.routes() as found:
            jax.eval_shape(lambda held, cached, x: module.apply({**held, **cached}, x, decode,
                                                                mutable=["cache"]), params, cache, x)
    [route] = [route for route in found if route.choice == "delta_decode"]
    assert (route.kernel, kernel_routes.verdict(route)) == (kernel, verdict)


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
    held = {name: {"kernel": head["kernel"]["kernel"]} for name, head in initial.items()}
    return {"params": {**params["params"], "initial": held}}


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
                    "self-generated gradients", seeds=SEEDS["self-generated gradients"],
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
        heads = [network.apply({"params": stored[f"head_{h}"]}, queries[:, :, h]) for h in range(HEADS)]
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


def decoder(memory: InContextMemory | None = None, **fields):
    """A tiny decoder of two memory layers, Titans-style where `memory` is None, which a scanned stack
    scans as one group, then two attention layers."""
    from model_support import TINY_DECODER

    from dew.nn.backbones.causal_transformer import CausalTransformer
    from dew.nn.backbones.layer_plan import LayerKind
    from dew.nn.mixers.memory import InContextMemoryMixer

    if memory is None:
        memory = InContextMemory(network=MLPMemory(hidden=(16,)), heads=2, head_features=8, chunk=4,
                                 normalize=True, learning_rate=Gate(scale=STEP), momentum=Gate(),
                                 retention=Gate(bias=-2.0))
    return CausalTransformer(**{"vocab_size": 37, **TINY_DECODER, "num_layers": 4, "dtype": jnp.float32,
                                "layer_types": ("memory", "memory", "attention", "attention"),
                                "kinds": {"memory": LayerKind(mixer=InContextMemoryMixer(memory))}, **fields})


def test_a_memory_decoder_trains_scans_remats_and_decodes():
    """A decoder whose first two layers are memories, which its scanned stack scans as one group, from
    the plain loop's tree: the scanned stack's logits and gradient, and both under a remat policy that
    keeps each memory's chunk carry (`CARRY`), are the plain loop's, on par against float64 (check 1),
    the gradient reaching every memory's initial network; and the scanned stack's cached decode, token
    by token after a prefill, gives the full pass's last logits (check 1)."""
    from model_support import model_draw

    from dew.nn.backbones.decoder_block import RematPolicy

    plain, scanned = decoder(), decoder(scan_layers=True)
    rematted = decoder(scan_layers=True, remat=RematPolicy(save=(CARRY,)))
    wide = decoder(dtype=jnp.float64)
    draw_model = model_draw(plain, (ROWS, LENGTH), 37)

    def trained(model):
        """The model's logits and its parameters' gradient under a cotangent."""
        @jax.jit
        def run(variables, ids, cotangent):
            logits, pull = jax.vjp(lambda held: model.apply(held, ids), variables)
            return logits, pull(cotangent)[0]["params"]
        return run

    def draw_with_cotangent(seed: int):
        cotangent = np.random.default_rng([1648, seed]).standard_normal((ROWS, LENGTH, 37)).astype(np.float32)
        return *draw_model(seed), cotangent

    for name, model in (("scanned", scanned), ("rematted", rematted)):
        assert_as_exact(trained(model), trained(plain), draw_with_cotangent,
                        f"{name} memory decoder gradient", seeds=SEEDS["memory decoder gradient"],
                        truth_fn=trained(wide))
    gradient = trained(rematted)(*draw_with_cotangent(0))[1]
    for index in (0, 1):
        assert all(np.abs(np.asarray(leaf)).max() > 0
                   for leaf in jax.tree.leaves(gradient[f"layers_{index}"]["self_attn"]["initial"]))

    @jax.jit
    def decoded(variables, ids):
        cache = scanned.apply(variables, ids.shape[0], method=scanned.init_cache, mutable=["cache"])[1]
        logits, cache = scanned.apply({**variables, **cache}, ids[:, :6], decode=True, mutable=["cache"])
        for t in range(6, LENGTH):
            logits, cache = scanned.apply({**variables, **cache}, ids[:, t:t + 1], decode=True,
                                          mutable=["cache"])
        return logits[:, -1]

    def last(model):
        return jax.jit(lambda variables, ids: model.apply(variables, ids)[:, -1])

    assert_as_exact(decoded, last(plain), draw_model, "memory decoder decode",
                    seeds=SEEDS["memory decoder decode"], truth_fn=last(wide))


ADAPTED = {"titans": ("layers_0", "layers_1"), "gated_linear_attention": ("kernel",),
           "delta_rule": ("kernel",)}
"""Each configuration's inner kernels by the module names an adapter targets (`dew.lora.LoRA.modules`):
Titans' two MLP layers and a linear memory's one map, at every head of both memory layers."""


def factor_draw(model, lora):
    """A draw of a seed's base weights for `model` as its init draws them, the factors of `lora`
    attached over them each drawn a tenth of a normal from zero, and its token rows."""
    ids = jnp.zeros((ROWS, LENGTH), jnp.int32)
    initialize = jax.jit(model.init)
    attached = traverse_util.flatten_dict(lora.apply(model, initialize(jax.random.key(0), ids), key=0)
                                          .variables["params"])

    def draw(seed: int):
        ids = np.random.default_rng([1649, seed]).integers(0, 37, (ROWS, LENGTH), dtype=np.int32)
        base = traverse_util.flatten_dict(initialize(jax.random.key(seed), ids)["params"])
        rng = np.random.default_rng([1650, seed])
        factors = {path: (0.1 * rng.standard_normal(leaf.shape)).astype(np.float32)
                   for path, leaf in attached.items() if path not in base}
        return {"params": traverse_util.unflatten_dict({**base, **factors})}, ids

    return draw


@pytest.mark.parametrize("name", sorted(ADAPTED))
def test_an_adapter_on_a_memorys_inner_kernels_trains_drops_and_merges(name):
    """LoRA on every head's inner kernels of two memory layers (`dew.lora`): fresh, the decoder's logits
    are the base's bit for bit; its loss's gradient reaches every inner B factor; a forward that carries
    the dropout stream drops the branches' inputs, so its logits move; and with drawn factors the
    adapted decoder computes its merge, on par against the merge in float64 (check 1): a retention
    forgets the adapter's update as it forgets W_0's, and a linear memory's recurrence reads the adapted
    map."""
    from dew.lora import LoRA

    model, wide = decoder(layer(name)), decoder(layer(name), dtype=jnp.float64)
    lora = LoRA(rank=2, modules=ADAPTED[name], dropout=0.5)
    draw = factor_draw(model, lora)
    ids = jnp.asarray(draw(0)[1])
    base = model.init(jax.random.key(0), ids)
    adapter = lora.apply(model, base, key=0)
    adapted = adapter.model
    np.testing.assert_array_equal(np.asarray(adapted.apply(adapter.variables, ids)),
                                  np.asarray(model.apply(base, ids)))

    gradient = jax.grad(lambda factors: jnp.sum(adapted.apply({**adapter.variables, "params": factors}, ids)
                                                ** 2))(adapter.variables["params"])
    inner = [leaf for path, leaf in traverse_util.flatten_dict(gradient).items()
             if path[-1] == "lora_B" and "initial" in path]
    assert len(inner) == 2 * HEADS * len(ADAPTED[name])
    assert all(np.abs(np.asarray(leaf)).max() > 0 for leaf in inner)

    drawn_factors, ids = draw(0)
    dropped = adapted.apply(drawn_factors, ids, rngs={"dropout": jax.random.key(3)})
    assert not np.array_equal(np.asarray(dropped), np.asarray(adapted.apply(drawn_factors, ids)))

    def merged(model):
        return jax.jit(lambda variables, ids: model.apply(adapter.merge(variables), ids))

    assert_as_exact(jax.jit(adapted.apply), merged(model), draw,
                    f"adapted {name} memory", seeds=SEEDS["adapted memory"], truth_fn=merged(wide))


SCALE = 0.125
"""The FP8 input scale of the quantized checks' inner sites: inputs of a unit's spread take E4M3's
coarse steps at 8 to 16 times it."""


def level() -> InContextMemory:
    """A continuum-memory level over its block's raw stream, one head of Titans' MLP updated every
    token: its network's input is the stream itself, the same values in every precision."""
    return layer("cms").clone(network=MLPMemory(hidden=(2 * WIDTH,)), commit="token")


def test_a_memorys_inner_network_rounds_its_inputs_at_their_declared_scale():
    """A checkpoint's input quantization at an inner site (`dew.quantization.InputQuantization`) rounds
    that site's input at its scale at every application of the network, the gradient's prediction and
    the readout: the layer computes its paper's update over the inputs E4M3 rounds at the scale its
    variables hold, on par against float64 (check 1)."""
    from dew.quantization import QUANTIZATION, InputQuantization, InputQuantizer

    module = level()
    quantized = InputQuantization(sites=(("initial/head_0/layers_0", "fp8"),)).apply(module)
    draw = drawn(module)

    def scaled(seed: int):
        x, params = draw(seed)
        scales = {"initial": {"head_0": {"layers_0": {"fp8_input_scale": np.float32(SCALE)}}}}
        return x, {**params, QUANTIZATION: scales}

    rounded = InputQuantizer.fp8(SCALE).qdq
    assert_as_exact(applied(quantized), written(module, dtype=jnp.float32, inputs=rounded), scaled,
                    "quantized memory", seeds=MEMORY_SEEDS,
                    truth_fn=written(module, dtype=jnp.float64, inputs=rounded))


def test_a_quantized_memory_decoder_reloads_from_its_record():
    """A decoder computing under input quantization at its memories' inner sites reads their scales:
    its logits move from the plain decoder's at the same weights; and its record rebuilds a decoder that
    computes them bit for bit."""
    from dew.config import ModelConfig
    from dew.quantization import QUANTIZATION, InputQuantization
    from dew.registry import to_record

    sites = tuple((f"layers_{layer}/self_attn/initial/head_{head}/layers_0", "fp8")
                  for layer in (0, 1) for head in range(2))
    plain = decoder()
    quantized = InputQuantization(sites=sites).apply(plain)
    ids = jax.random.randint(jax.random.key(1), (ROWS, LENGTH), 0, 37)
    variables = quantized.init(jax.random.key(0), ids)
    scales = jax.tree.map(lambda scale: np.float32(SCALE), variables[QUANTIZATION])
    variables = {**variables, QUANTIZATION: scales}
    logits = np.asarray(quantized.apply(variables, ids))
    assert not np.array_equal(logits, np.asarray(plain.apply({"params": variables["params"]}, ids)))
    record = json.loads(json.dumps(to_record(ModelConfig.from_model(quantized), ModelConfig)))
    reloaded = ModelConfig.from_dict(record).build()
    np.testing.assert_array_equal(np.asarray(reloaded.apply(variables, ids)), logits)


def test_a_frozen_fp8_inner_kernel_is_w0_as_its_codes_decode():
    """An inner kernel a frozen FP8 record packs (`InputQuantization.frozen_fp8`, a LoRA run's FP8 base)
    is the memory's W_0 as its codes decode (`frozen_fp8_kernel`, as a merge decodes it), which the
    in-context steps move from: the layer computes, bit for bit, the same layer whose kernels are those
    decoded ones."""
    from dew.quantization import QUANTIZATION, InputQuantization, frozen_fp8_kernel

    module = layer("titans")
    sites = tuple(f"initial/head_{head}/layers_0" for head in range(HEADS))
    record = InputQuantization(frozen_fp8=sites, measured_on="sm90")
    x, params = drawn(module)(0)
    codes = record.packed(params["params"])
    kernels = {("initial", f"head_{head}", "layers_0", "kernel"): frozen_fp8_kernel(
        codes["initial"][f"head_{head}"]["layers_0"]) for head in range(HEADS)}
    flat = traverse_util.flatten_dict(params["params"])
    held = traverse_util.unflatten_dict({path: leaf for path, leaf in flat.items() if path not in kernels})
    frozen = jax.jit(record.apply(module).apply)({"params": held, QUANTIZATION: codes}, x)
    decoded = jax.jit(module.apply)({"params": traverse_util.unflatten_dict({**flat, **kernels})}, x)
    np.testing.assert_array_equal(np.asarray(frozen), np.asarray(decoded))


def explicit_rows_run(module: InContextMemory):
    """`module`'s forward over whole rows, the gradients of its weights and stream under a cotangent,
    its forward over rows packing two documents (`segments`), and a prefill of six tokens continued one
    a cached call."""
    decode = MixerCall(decode=True)

    @jax.jit
    def run(x, params, cotangent, segments):
        full, pull = jax.vjp(lambda held, x: module.apply(held, x), params, x)
        laid = coordinates(segments=segments, max_segments=2)
        call = MixerCall(metadata=AttentionMetadata(coordinates=laid))
        _, cache = module.apply(params, x[:, :1], decode, mutable=["cache"])
        prefill, cache = module.apply({**params, **cache}, x[:, :6], decode, mutable=["cache"])
        reads = [prefill]
        for t in range(6, LENGTH):
            token, cache = module.apply({**params, **cache}, x[:, t:t + 1], decode, mutable=["cache"])
            reads.append(token)
        return full, pull(cotangent), module.apply(params, x, call), jnp.concatenate(reads, axis=1)

    return run


@pytest.mark.mesh(devices=2)
@pytest.mark.parametrize("name", ["titans", "gated_linear_attention", "cms"])
def test_a_memory_runs_its_rows_split_over_an_explicit_axis(name):
    """Rows split over an Explicit data axis, as a run's batch is: a memory's forward, the gradients of
    its weights and stream, its forward over packed documents and a prefill continued token by token
    through its cache, each on the general chunk scan, the token scan or a linear memory's recurrence,
    are what one device computes, on par against float64 (check 1)."""
    from jax.sharding import AxisType, NamedSharding, PartitionSpec

    module = layer(name)
    mesh = jax.make_mesh((2,), ("data",), axis_types=(AxisType.Explicit,), devices=jax.devices()[:2])
    run, draw = explicit_rows_run(module), drawn(module)
    segments = np.broadcast_to(np.where(np.arange(LENGTH) < 5, 1, 2), (ROWS, LENGTH)).astype(np.int32)

    def placed(seed: int):
        x, params = draw(seed)
        cotangent = np.random.default_rng([1651, seed]).standard_normal((*x.shape[:2], WIDTH))
        return x, params, cotangent.astype(np.float32), segments

    def split(x, params, cotangent, segments):
        with jax.set_mesh(mesh):
            rows, whole = NamedSharding(mesh, PartitionSpec("data")), NamedSharding(mesh, PartitionSpec())
            return jax.device_get(run(jax.device_put(x, rows), jax.device_put(params, whole),
                                      jax.device_put(cotangent, rows), jax.device_put(segments, rows)))

    assert_as_exact(split, run, placed, f"{name} memory rows split", seeds=MEMORY_SEEDS,
                    truth_fn=explicit_rows_run(module.clone(dtype=jnp.float64)))


def test_a_memory_refuses_a_bidirectional_decoder():
    """A memory reads each token's prefix alone, so a decoder that asks its layers for the whole row
    (`causal=False`, a masked diffusion model's) refuses to build one; the same decoder causal builds."""
    ids = jnp.zeros((2, LENGTH), jnp.int32)
    decoder().init(jax.random.key(0), ids)
    with pytest.raises(ValueError):
        decoder(causal=False).init(jax.random.key(0), ids)


def test_a_memory_decoder_names_and_climbs_to_the_residuals_its_path_marks(capsys):
    """A memory kind declares the residuals its path marks (`MixerBase.residuals`): a linear memory on
    the delta rule, which runs the rule's port, its decay, inverse and memory, each of which a policy
    saving it alone keeps for the backward, so its decoder's remat ladder takes the rung recomputing
    the rule's memory alone; Titans' memory its chunk scan's carry, kept alike, and a ladder without
    that rung."""
    from jax.ad_checkpoint import print_saved_residuals
    from model_support import TINY_DECODER

    from dew.nn.backbones.causal_transformer import CausalTransformer
    from dew.nn.backbones.decoder_block import RematPolicy
    from dew.nn.kernels.generation import chosen
    from dew.nn.mixer_base import DELTA_RESIDUALS
    from dew.nn.mixers.memory import InContextMemoryMixer

    def kept(memory: InContextMemory) -> tuple[tuple[str, ...], list]:
        kind = InContextMemoryMixer(memory)
        model = CausalTransformer(**{"vocab_size": 37, **TINY_DECODER, "dtype": jnp.float32, "mixer": kind})
        ids = jax.random.randint(jax.random.key(3), (2, LENGTH), 0, 37)
        variables = model.init(jax.random.key(4), ids)

        def saves(name: str) -> bool:
            capsys.readouterr()
            loss = jax.checkpoint(lambda held: jnp.sum(model.apply(held, ids) ** 2),
                                  policy=RematPolicy(save=(name,)).checkpoint_policy())
            print_saved_residuals(loss, variables)
            return f"named '{name}'" in capsys.readouterr().out

        assert all(saves(name) for name in kind.residuals)
        climbed = [model]
        while (stronger := climbed[-1].recompute_more()) is not None:
            climbed.append(stronger)
        return kind.residuals, [rung.recompute_record() for rung in climbed]

    with chosen(gated_delta_rule="fla"):
        delta, delta_rungs = kept(layer("delta_rule"))
    titans, titans_rungs = kept(layer("titans"))
    assert set(DELTA_RESIDUALS) <= set(delta) and "all_but_delta_memory" in delta_rungs
    assert CARRY in titans and "all_but_delta_memory" not in titans_rungs


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
