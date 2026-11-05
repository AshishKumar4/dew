"""A commit's host-resident fields brought to the device a chunk at a time (`dew.training.staging`, row 3526).

The CPU backend exposes `pinned_host` beside `device`, so a staged commit runs here as written and is held to
the whole one bit for bit; what it saves of a device is a GPU's measurement, which one backend's single
memory cannot show (tests/test_host_placement.py): the GPU flow's cell
(`a_staged_commit_holds_two_layers_of_host_resident_moments_at_once`, tests/test_gpu_training.py)."""

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.sharding import SingleDeviceSharding

from dew.training.inspection import Inspection, stacked, summaries
from dew.training.staging import chunks, flattened, staged

SHAPES = {"embedding": (64, 32), "bias": (32,), "kernel": (128, 16)}


def drawn(seed: int) -> dict:
    rng = np.random.default_rng(seed)
    return {name: jnp.asarray(rng.standard_normal(shape), jnp.float32) for name, shape in SHAPES.items()}


def normalized_trace() -> optax.GradientTransformation:
    """A momentum whose update is its trace over the trace's global norm: each update reads every leaf."""
    def init(params):
        return optax.TraceState(trace=jax.tree.map(jnp.zeros_like, params))

    def update(updates, state, params=None):
        del params
        trace = jax.tree.map(lambda held, new: 0.9 * held + new, state.trace, updates)
        norm = optax.global_norm(trace)
        return jax.tree.map(lambda leaf: leaf / norm, trace), optax.TraceState(trace=trace)
    return optax.GradientTransformation(init, update)


def cut_of(optimizer: optax.GradientTransformation, sampled: bool = False) -> list[list[str]]:
    """The chunks `chunks` cuts `optimizer`'s state into for its update, each by its leaves' paths; a
    `sampled` update's commit reports its summaries too, as a sampled step's does (`Transaction.commit`)."""
    params = drawn(0)
    state = optimizer.init(params)
    paths, structure = jax.tree_util.tree_flatten_with_path(state)

    def update(fields, gradients, params):
        updates, state = optimizer.update(gradients, structure.unflatten(fields), params)
        reported = summaries(Inspection(every=1), params, gradients, updates,
                             optax.apply_updates(params, updates)) if sampled else {}
        return jax.tree.leaves(state), stacked(reported, jnp.asarray(a=True))

    traced = jax.make_jaxpr(update)([leaf for _, leaf in paths], params, params)
    equations, _ = flattened(traced.jaxpr, traced.jaxpr.invars, traced.consts)
    cut, _, _ = chunks(equations, traced.jaxpr.invars[:len(paths)],
                       [leaf.size * leaf.dtype.itemsize for _, leaf in paths])
    return [sorted(jax.tree_util.keystr(paths[index][0]) for index in chunk) for chunk in cut]


def test_an_elementwise_update_stages_each_parameters_moments_alone():
    """Adam reads each parameter's two moments into its own update alone, so each parameter's moments are a
    chunk, the largest the step holds of the state at once, and its count a scalar fetched first; clipping
    by the global norm reads the gradients together, and still each parameter's moments alone."""
    for optimizer in (optax.adamw(1e-3), optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(1e-3))):
        cut = cut_of(optimizer)
        assert len(cut) == len(SHAPES)
        for chunk in cut:
            names = {path.rsplit("'", 2)[1] for path in chunk}
            assert len(names) == 1 and len(chunk) == 2, cut


def test_a_sampled_commits_summaries_stage_each_parameters_moments_alone():
    """A sampled commit's summaries of its update, norms over every leaf stacked into one vector
    (`dew.training.inspection`), hold a few words, not a leaf: Adam's commit still stages each parameter's
    moments alone (row 3659), where summing its norms across the leaves joined every chunk into one."""
    assert cut_of(optax.adamw(1e-3), sampled=True) == cut_of(optax.adamw(1e-3))
    assert len(cut_of(optax.adamw(1e-3), sampled=True)) == len(SHAPES)


def spread_total() -> optax.GradientTransformation:
    """A trace whose every leaf is its update plus the whole last trace's sum, that scalar spread over the
    leaf's shape by stacking it once an entry: each new leaf reads every old one, though only a scalar of
    them meets each update."""
    def init(params):
        return optax.TraceState(trace=jax.tree.map(jnp.zeros_like, params))

    def update(updates, state, params=None):
        del params
        total = sum(jnp.sum(leaf) for leaf in jax.tree.leaves(state.trace))
        trace = jax.tree.map(lambda held, new: new + jnp.stack([total] * held.size).reshape(held.shape),
                             state.trace, updates)
        return trace, optax.TraceState(trace=trace)
    return optax.GradientTransformation(init, update)


def test_an_update_that_reads_its_whole_state_stages_it_in_one_chunk():
    """A trace scaled by its own global norm reads every leaf of the state into each update: one chunk. So
    does one that spreads a scalar of every leaf back over each leaf's shape (`spread_total`): a summary
    is no larger than the distinct scalars it is made of, and a leaf-sized value of one scalar is a leaf
    each update holds till the last chunk is read."""
    assert [len(chunk) for chunk in cut_of(normalized_trace())] == [len(SHAPES)]
    assert [len(chunk) for chunk in cut_of(spread_total())] == [len(SHAPES)]


def test_a_staged_commit_is_the_whole_commit_and_rests_where_it_rested():
    """A clipped AdamW commit over moments in pinned host memory, staged a chunk at a time, makes the
    parameters and moments the commit over the whole fetched state makes, bit for bit, and returns the
    moments to pinned host memory."""
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(1e-2))
    params, gradients = drawn(1), drawn(2)
    state = optimizer.update(drawn(3), optimizer.init(params), params)[1]
    device = jax.devices()[0]
    host = SingleDeviceSharding(device, memory_kind="pinned_host")
    resting, shardings = jax.device_put(state, host), jax.tree.map(lambda _: host, state)

    def commit(state, gradients, params):
        updates, new = optimizer.update(gradients, state, params)
        return new, optax.apply_updates(params, updates)

    whole = jax.jit(lambda resting, gradients, params: commit(
        jax.device_put(resting, SingleDeviceSharding(device, memory_kind="device")), gradients, params))
    chunked = jax.jit(lambda resting, gradients, params: staged(commit, resting, shardings,
                                                                (gradients, params), gradients))
    expected, found = whole(resting, gradients, params), chunked(resting, gradients, params)
    assert all(np.array_equal(left, right) for left, right in zip(jax.tree.leaves(expected),
                                                                  jax.tree.leaves(found), strict=True))
    assert {leaf.sharding.memory_kind for leaf in jax.tree.leaves(found[0])} == {"pinned_host"}


def rated() -> optax.GradientTransformation:
    """Updates scaled by a rate its state holds as a weakly typed scalar, as `optax.inject_hyperparams`
    holds a Python float's."""
    def update(updates, state, params=None):
        del params
        return jax.tree.map(lambda leaf: leaf * state["rate"], updates), state
    return optax.GradientTransformation(lambda params: {"rate": jnp.asarray(0.1)}, update)


def test_a_staged_weakly_typed_scalar_scales_as_it_does_unstaged():
    """A weakly typed rate in pinned host memory scales bfloat16 updates in bfloat16, staged as unstaged:
    traced strongly typed, the staged commit promoted them to float32."""
    optimizer = rated()
    params = jax.tree.map(lambda leaf: leaf.astype(jnp.bfloat16), drawn(4))
    state = optimizer.init(params)
    device = jax.devices()[0]
    host = SingleDeviceSharding(device, memory_kind="pinned_host")
    resting, shardings = jax.device_put(state, host), jax.tree.map(lambda _: host, state)

    def commit(state, gradients):
        updates, new = optimizer.update(gradients, state)
        return new, updates

    whole = jax.jit(lambda resting, gradients: commit(
        jax.device_put(resting, SingleDeviceSharding(device, memory_kind="device")), gradients))
    chunked = jax.jit(lambda resting, gradients: staged(commit, resting, shardings, (gradients,), gradients))
    (_, expected), (_, found) = whole(resting, params), chunked(resting, params)
    assert [leaf.dtype for leaf in jax.tree.leaves(found)] == [jnp.bfloat16] * len(SHAPES)
    assert all(np.array_equal(left, right) for left, right in zip(jax.tree.leaves(expected),
                                                                  jax.tree.leaves(found), strict=True))


def a_staged_commit_holds_two_layers_of_host_resident_moments_at_once():
    """On a GPU, whose pinned host memory is apart from its own: AdamW's commit over eight layers' moments in
    pinned host memory, staged a layer at a time, compiles to temporaries of at most two layers' moments, the
    two chunks its trace measures (`dew.training.staging.measured`). Its chunks' barriers hold it
    there: without them XLA fetched the moments ahead, as the whole fetch behind the gradient did, 11.5 MB of
    the 16.8 on an H100 against 3.2 staged, and 4.2 MB on an A100 (row 3526's residency probe)."""
    device = jax.devices()[0]
    host = SingleDeviceSharding(device, memory_kind="pinned_host")
    params = {f"layer{index}": jnp.ones((512, 512), jnp.float32) for index in range(8)}
    optimizer = optax.adamw(1e-3)
    state = jax.device_put(optimizer.init(params), host)
    shardings = jax.tree.map(lambda _: host, state)

    def commit(state, gradients, params):
        updates, new = optimizer.update(gradients, state, params)
        return new, optax.apply_updates(params, updates)

    def chunked(resting, gradients, params):
        return staged(commit, resting, shardings, (gradients, params), gradients)

    memory = jax.jit(chunked).lower(state, params, params).compile().memory_analysis()
    assert memory is not None
    assert memory.temp_size_in_bytes <= 2 * (2 * 512 * 512 * 4)


def test_flattening_runs_a_plain_call_in_its_caller_and_keeps_a_marked_span_a_call():
    """A staged commit runs an optax helper's `jit` in its caller (`flattened`) and keeps a marked span
    (`dew.nn.derivatives.marked`) a call of its own: the algebra pass reads its boundary, and a protected
    span's inside no rule touches, so flattened its sums would be rewritten together."""
    from dew.nn.derivatives import Protected, marked

    def two_sums(x):
        return jnp.sum(x) + jnp.sum(2 * x)

    closed = jax.make_jaxpr(lambda x: jax.jit(lambda y: marked(Protected(), two_sums)(y + 1))(x))(jnp.ones(4))
    equations, _ = flattened(closed.jaxpr, closed.jaxpr.invars, closed.consts)
    names = [equation.primitive.name for equation in equations]
    assert names.count("marked_call") == 1 and not {"jit", "pjit"} & set(names), names
