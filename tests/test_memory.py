"""The in-context memory (`dew.nn.memory`, ledger 1645) as a function: each configuration against its
paper's update equations, written out token by token (`memory_support.written`), at check 1 against
float64; the chunk-parallel computation against the token-by-token one; the linear memories on the
shared recurrence against their general path; packed documents as each alone; a decode, and a cached
packed prefill, as the full pass; the outer gradients through a self-generated target; a zero step size
reading the initial network; and an orthogonalized step. Its compositions are
`test_memory_composition`'s."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import linen as nn
from flax.typing import Dtype
from memory_support import (
    CHUNK,
    CONFIGURATIONS,
    FEATURES,
    HEADS,
    LENGTH,
    MEMORY_SEEDS,
    ROWS,
    WIDTH,
    applied,
    decoded,
    drawn,
    head_of,
    layer,
    packed,
    written,
)
from reference_error import assert_as_exact, with_gradients

from dew.batch import coordinates
from dew.nn.inputs import AttentionMetadata, MixerCall
from dew.nn.memory import Gate, InContextMemory, Objective, ResidualMemory

SEEDS = {"self-generated gradients": 512, "decoded gated_linear_attention": 32}
"""The sites' declared seeds where `seeds_for` on armada, each reference compared with itself, sizes them
past `MEMORY_SEEDS`: the self-modifying memory's gradients, every parameter's leaf, at 512, and a linear
memory with retention decoding token by token at 32."""


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
    return {"params": {name: {"kernel": tree["kernel"]["kernel"]} if head_of((name,)) is not None else tree
                       for name, tree in params["params"].items()}}


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
    full = applied(module, packed(np.ones((ROWS, LENGTH), np.int32)))
    assert_as_exact(decoded(module), full, drawn(module), f"decoded {name} memory",
                    seeds=SEEDS.get(f"decoded {name}", MEMORY_SEEDS),
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
        heads = [network.apply({"params": params["params"][f"head_{h}"]}, queries[:, :, h])
                 for h in range(HEADS)]
        return jnp.stack(heads, axis=2).reshape(-1, LENGTH, HEADS * FEATURES)

    assert_as_exact(applied(module), jax.jit(initial), drawn(module), "zero-step memory", seeds=MEMORY_SEEDS,
                    truth_fn=initial)


def test_an_orthogonalized_step_follows_the_paper_with_muons_iteration():
    """With `orthogonalize`, each matrix of a token's gradient steps as Muon's quintic Newton-Schulz of it
    (optax's iteration, `dew.optim._orthogonalized`, Nested Learning Sec 4.2) and its vectors as they are:
    the layer is the paper's update with that map applied to its gradients, on par against float64
    (check 1)."""
    from dew.optim import _orthogonalized

    module = layer("titans").clone(orthogonalize=5)

    def gradient(loss, apply, start, k, target):
        """The paper's gradient with Muon's map on each matrix, in the reference's own precision."""
        g = jax.grad(lambda held: loss(apply(held, k), target))(start)
        return jax.tree.map(lambda leaf: _orthogonalized(leaf, 5, None) if leaf.ndim == 2 else leaf, g)

    assert_as_exact(applied(module), written(module, dtype=jnp.float32, gradient=gradient), drawn(module),
                    "orthogonalized memory", seeds=MEMORY_SEEDS,
                    truth_fn=written(module, dtype=jnp.float64, gradient=gradient))
