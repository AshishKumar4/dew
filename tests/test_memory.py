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
    decode_passes,
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


@pytest.mark.parametrize("name", ["titans", "self_modifying", "ttt_linear", "titans_linear"])
def test_the_chunk_parallel_memory_is_the_token_by_token_one(name):
    """A row of one document runs chunk by chunk, TTT-Linear's and Titans' linear memories on their
    mini-batched kernels (`InContextMemory._mini_batch`); the same row as a packed call runs token by
    token, the definition. The two are on par against float64 (check 1)."""
    module = layer(name)
    stepped = applied(module, packed(np.ones((ROWS, LENGTH), np.int32)))
    assert_as_exact(applied(module), stepped, drawn(module), f"{name} chunked", seeds=MEMORY_SEEDS,
                    truth_fn=written(module, dtype=jnp.float64))


PADDED = ((np.arange(LENGTH)[None] >= np.asarray([0, 1, 3, 0, 5, 2, 0, 7])[:, None])
          & (np.arange(LENGTH)[None] < np.asarray([12, 12, 12, 10, 12, 12, 8, 12])[:, None]))
"""Each row's real tokens: rows left-padded by 1 to 7 tokens, two right-padded by 2 and 4, three whole."""


@pytest.mark.parametrize("name", ["ttt_linear", "titans_linear"])
def test_a_mini_batched_memory_reads_padded_rows_as_the_token_path_does(name):
    """Rows of one document padded left and right (`PADDED`, a `valid` mask) run on the mini-batched
    kernels, their real tokens compacted and their chunks counted from each row's first real token, and
    read every real token as the token path does over the same rows packed (pads a segment of none), on
    par against that path in float64 (check 1); a padding token's readout is no real token's."""
    module = layer(name)
    valid = jnp.asarray(PADDED)
    by_mask = MixerCall(metadata=AttentionMetadata(coordinates=coordinates(valid=valid)))
    by_tokens = packed(np.where(PADDED, 1, 0).astype(np.int32))

    def real(run):
        return jax.jit(lambda x, params: jnp.where(valid[..., None], run(x, params), 0))

    assert_as_exact(real(applied(module, by_mask)), real(applied(module, by_tokens)), drawn(module),
                    f"{name} padded", seeds=MEMORY_SEEDS,
                    truth_fn=real(applied(layer(name, dtype=jnp.float64), by_tokens)))


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
    across the calls: on par with the full passes, each step's at the width it reads (`decode_passes`),
    against float64 (check 1)."""
    module = layer(name)
    full = decode_passes(module, np.ones((ROWS, LENGTH), np.int32), 6)
    assert_as_exact(decoded(module), full, drawn(module), f"decoded {name} memory",
                    seeds=SEEDS.get(f"decoded {name}", MEMORY_SEEDS),
                    truth_fn=written(module, dtype=jnp.float64))


@pytest.mark.parametrize("name", ["titans", "gated_linear_attention"])
def test_a_cached_prefill_starts_each_packed_document_afresh(name):
    """A cached prefill of packed documents starts each from the initial network, its chunks counted
    from its own first token, and the calls after it continue its last document: a prefill of a
    three-token document and five tokens of the next, then that document's last four one call each,
    reads each token as the full packed passes do, each step's at the width it reads (`decode_passes`), on
    par against float64 (check 1)."""
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

    full = decode_passes(module, np.broadcast_to(segments, (ROWS, LENGTH)), 8)
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
    (Dew's iteration, held to optax's: `optim_support.optax_orthogonalized`, Nested Learning Sec 4.2) and
    its vectors as they are: the layer is the paper's update with that map applied to its gradients, on par
    against float64 (check 1)."""
    from optim_support import optax_orthogonalized

    module = layer("titans").clone(orthogonalize=5)

    def gradient(loss, apply, start, k, target):
        """The paper's gradient with Muon's map on each matrix, in the reference's own precision."""
        g = jax.grad(lambda held: loss(apply(held, k), target))(start)
        return jax.tree.map(lambda leaf: optax_orthogonalized(leaf, 5) if leaf.ndim == 2 else leaf, g)

    assert_as_exact(applied(module), written(module, dtype=jnp.float32, gradient=gradient), drawn(module),
                    "orthogonalized memory", seeds=MEMORY_SEEDS,
                    truth_fn=written(module, dtype=jnp.float64, gradient=gradient))


def test_a_memory_recomputes_its_chunks_only_where_its_rung_names_them():
    """An in-context memory's training step recomputes each chunk from its carry in the backward only where
    the rung it runs under names its chunks (`residuals.recomputing`, `memory_chunk`: the heaviest
    rung's, `decoder_block.recomputed_inside`); under any other it keeps their values, the step's work its
    forward's and backward's alone (row 6577: recomputed under every rung, a Titans step that fit ran
    31-41% slower)."""
    from dew.nn.inputs import PLAIN_CALL
    from dew.nn.residuals import recomputing

    module = layer("titans")
    params = jax.eval_shape(lambda: module.init(jax.random.key(0), jnp.zeros((1, LENGTH, WIDTH))))
    x = jax.ShapeDtypeStruct((ROWS, LENGTH, WIDTH), jnp.float32)

    def flops(names: tuple[str, ...]) -> float:
        def loss(params, x):
            with recomputing(names):
                return jnp.sum(module.apply(params, x, PLAIN_CALL))

        return jax.jit(jax.grad(loss)).lower(params, x).compile().cost_analysis()["flops"]

    assert flops(()) < flops(("memory_chunk",))


def test_a_steps_forward_is_read_in_the_control_flow_it_runs_in():
    """The bytes a step's forward made (`LiveSet.forward_bytes`) are those whose first scope under the step
    and the control flow it runs in is a `jvp(`, a replayed window's forward inside its conditional and
    loop among them (row 12884: a decision step's replay held 6,971 MiB there, which the climb read as no
    forward's), and never a backward's (`transpose(`)."""
    from dew.training.memory import LiveSet

    replayed = "jit(step)/cond/branch_1_fun/while/body"
    live = LiveSet(0, 0, ((100, f"{replayed}/jvp(Model)/dot_general", ()),
                          (50, f"{replayed}/transpose(jvp(Model))/dot_general", ()),
                          (30, "jit(step)/jvp(Model)/tanh", ()),
                          (7, "jit(step)/convert_element_type", ())))
    assert live.forward_bytes(lambda op_name, names: True) == 130


def test_a_host_placed_array_holds_none_of_its_devices_memory():
    """The bytes a step's arguments hold where they lie (`resident_on`), which its compile's limit gives
    back, are those in each device's own memory: an array placed in its host's pinned memory
    (`Layout(host=...)`'s banks) holds none (row 12883: counted, a host-offloaded run's limit gave back bytes
    its device never held)."""
    from dew.training.memory import resident_on

    device = jax.devices()[0]
    host = jax.sharding.SingleDeviceSharding(device, memory_kind="pinned_host")
    on_device = jax.device_put(jnp.ones((64,), jnp.float32), device)
    on_host = jax.device_put(jnp.ones((256,), jnp.float32), host)
    assert resident_on({"device": on_device, "host": on_host}) == {device: on_device.nbytes}
