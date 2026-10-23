"""Scratch-only localization and regroupings of the aligned-key final-memory forward error."""
import functools
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import test_recurrence_chunk as token
import test_recurrence_stability as stability
from reference_error import widened

from dew.nn.kernels import recurrence as r, recurrence_algebra as a, recurrence_chunk as c

ORIGINAL = c._chunk_delta_forward
MODES = ("current", "wy-maps", "chrono-maps", "chrono-write", "chrono-transition",
         "wide-carry", "wide-prep", "wide-write", "wide-corrected", "tail1", "tail2", "tail4")


def cast64(x):
    return jax.tree.map(lambda x: x.astype(jnp.float64) if hasattr(x, "dtype")
                        and jnp.issubdtype(x.dtype, jnp.floating) else x, x)


def prototype(mode, query, key, value, g, beta, state, chunk_size, starts, delta, scale, dtype):
    if mode == "current":
        return ORIGINAL(query, key, value, g, beta, state, chunk_size, starts, delta, scale, dtype)
    args = query, key, value, g, beta
    steps = c.chunked(*(cast64(args) if mode == "wide-prep" else args), chunk_size,
                      starts=starts, delta=delta, scale=scale)
    if mode == "wide-prep":
        steps = jax.tree.map(lambda x: x.astype(query.dtype) if hasattr(x, "dtype")
                             and jnp.issubdtype(x.dtype, jnp.floating) else x, steps)
    resets = jnp.zeros(value.shape[:2], bool) if starts is None else starts
    resets = c._chunk_rows(jnp.broadcast_to(resets[..., None], value.shape[:3]), chunk_size)
    values, logs, writes = (c._chunk_rows(x, chunk_size) for x in (value, g, beta))
    state = jax.lax.optimization_barrier(state)

    if mode in ("chrono-maps", "chrono-write", "chrono-transition", "wy-maps"):
        transition, written = a.transition(steps.key, steps.w, steps.u, steps.grown, steps.last, steps.kept)
        if mode != "wy-maps":
            lead = steps.key.shape[:3]
            dk, dv = key.shape[-1], value.shape[-1]
            identity = jnp.broadcast_to(jnp.eye(dk, dtype=query.dtype), (*lead, dk, dk))
            zero = jnp.zeros((*lead, dk, dv), query.dtype)

            def one(maps, inputs):
                transition, written = maps
                key, value, g, beta, reset = inputs
                transition = jnp.where(reset[..., None, None], 0.0, transition)
                written = jnp.where(reset[..., None, None], 0.0, written)
                empty_query = jnp.zeros_like(key)
                if mode != "chrono-write":
                    _, transition = r.update(transition, empty_query, key, jnp.zeros_like(key),
                                              jnp.exp(g), beta[..., None], delta=delta)
                if mode != "chrono-transition":
                    _, written = r.update(written, empty_query, key, value, jnp.exp(g),
                                           beta[..., None], delta=delta)
                return (transition, written), None

            (ct, cw), _ = jax.lax.scan(one, (identity, zero), tuple(jnp.moveaxis(x, 3, 0)
                                                                   for x in (steps.key, values, logs,
                                                                             writes, resets)))
            if mode != "chrono-write":
                transition = ct
            if mode != "chrono-transition":
                written = cw

        def across(state, maps):
            transition, written = maps
            return a.portable("compose", transition, state) + written, state

        state, entered = jax.lax.scan(across, state, tuple(jnp.moveaxis(x, 2, 0)
                                                          for x in (transition, written)))
        entered = jnp.moveaxis(entered, 0, 2)
        corrected = steps.u - a.portable("corrected", steps.w, entered)
    elif mode.startswith("tail"):
        tail = min(int(mode[4:]), chunk_size)
        count = steps.key.shape[2]
        lengths = jnp.minimum(chunk_size, value.shape[1] - jnp.arange(count) * chunk_size)
        prefix = jnp.maximum(lengths - tail, 0)
        row = jnp.maximum(prefix - 1, 0)[None, None, :, None, None]
        grown = jnp.take_along_axis(steps.grown, jnp.broadcast_to(
            row, (*steps.grown.shape[:3], 1, steps.grown.shape[-1])), axis=3)
        grown = jnp.where(prefix[None, None, :, None, None] > 0, grown, 1.0)
        pair_row = jnp.broadcast_to(row[..., None], (*steps.pairs.shape[:3], 1,
                                                     chunk_size, steps.pairs.shape[-1]))
        last = jnp.take_along_axis(steps.pairs, pair_row, axis=3)[..., 0, :, :]
        last = jnp.where(prefix[None, None, :, None, None] > 0, last, 0.0)
        in_prefix = jnp.arange(chunk_size)[None, None, None, :] < prefix[None, None, :, None]
        kept = (~jnp.any(resets & in_prefix, axis=-1))[..., None, None].astype(query.dtype)
        prefix_step = steps.key, steps.w, steps.u, grown, last, kept

        def gathered(x):
            index = prefix[None, None, :, None] + jnp.arange(tail)[None, None, None, :]
            if x.ndim == 5:
                index = index[..., None]
            index = jnp.broadcast_to(index, (*x.shape[:3], tail, *x.shape[4:]))
            return jnp.take_along_axis(x, index, axis=3)

        tail_tokens = tuple(map(gathered, (steps.key, values, logs, writes, resets)))

        def across(state, inputs):
            *p, keys, values, logs, writes, resets = inputs
            carried, (entered, corrected) = a.carry(state, tuple(p))
            for index in range(tail):
                carried = jnp.where(resets[..., index, None, None], 0.0, carried)
                _, carried = r.update(carried, jnp.zeros_like(keys[..., index, :]), keys[..., index, :],
                                       values[..., index, :], jnp.exp(logs[..., index, :]),
                                       writes[..., index, None], delta=delta)
            return carried, (entered, corrected)

        inputs = (*prefix_step, *tail_tokens)
        state, (entered, corrected) = jax.lax.scan(
            across, state, tuple(jnp.moveaxis(x, 2, 0) for x in inputs))
        entered, corrected = (jnp.moveaxis(x, 0, 2) for x in (entered, corrected))
    else:
        def issue(name, left, right):
            if mode == "wide-" + name:
                return (left.astype(jnp.float64) @ right.astype(jnp.float64)).astype(left.dtype)
            return a.portable(name, left, right)

        def across(state, step):
            return a.carry(state, step, issue)

        operands = steps.key, steps.w, steps.u, steps.grown, steps.last, steps.kept
        if mode == "wide-carry":
            state, operands = cast64((state, operands))
        state, (entered, corrected) = jax.lax.scan(across, state, tuple(jnp.moveaxis(x, 2, 0)
                                                                      for x in operands))
        entered, corrected = (jnp.moveaxis(x, 0, 2).astype(query.dtype) for x in (entered, corrected))
        state = state.astype(query.dtype)
    core = a.readout(steps.query, steps.entering, entered, steps.key, steps.pairs, corrected)
    output = c._chunk_sequence(core, value.shape[1]).astype(dtype), state
    return output, (steps, entered)


def measure(mode, name, seeds, monkeypatch, *, first=0):
    hooks = stability.SITES[name]
    monkeypatch.setattr(c, "_chunk_delta_forward", functools.partial(prototype, mode))
    ours = jax.jit(functools.partial(c.chunk, hooks, chunk_size=token.CHUNK, ported=False))
    reference = jax.jit(functools.partial(token.token_by_token, hooks))
    errors = [[], []]
    controls = [[], []]
    with jax.enable_x64(True):
        for seed in range(first, first + seeds):
            with jax.enable_x64(False):
                operands, _ = stability.cotangents(hooks, "aligned keys", seed)
            mine, ref = ours(operands), reference(operands)
            truth = reference(*widened((operands,)))
            for i in range(2):
                errors[i].append(np.mean((np.asarray(mine[i], np.float64)-np.asarray(truth[i]))**2))
                controls[i].append(np.mean((np.asarray(ref[i], np.float64)-np.asarray(truth[i]))**2))
        for i, label in enumerate(("output", "final")):
            own, ref = np.sqrt(np.mean(errors[i])), np.sqrt(np.mean(controls[i]))
            print(f"FORWARD|{mode}|{name}|seeds={seeds}|{label}|ours={own:.12g}"
                  f"|reference={ref:.12g}|ratio={own/ref:.9g}", flush=True)
            assert np.isfinite(own)


def decompose(seed=0):
    with jax.enable_x64(False):
        operands, _ = stability.cotangents(stability.SITES["scalar delta"], "aligned keys", seed)

    def raw(op):
        h = op.value.shape[2]
        q, k = (jnp.repeat(x, h // x.shape[2], axis=2) for x in (op.query, op.key))
        return q, k, op.value, op.log_decay[..., None], op.beta

    def chronological(op):
        q, k, v, g, beta = raw(op)
        pad = (-k.shape[1]) % token.CHUNK
        xs = tuple(jnp.moveaxis(jnp.pad(x, [(0, 0), (0, pad)]
                                        + [(0, 0)] * (x.ndim-2)), 1, 0) for x in (q, k, v, g, beta))

        def one(state, values):
            q, k, v, g, beta = values
            _, state = r.update(state, q, k, v, jnp.exp(g), beta[..., None], delta=True)
            return state, state

        _, states = jax.lax.scan(one, op.state, xs)
        ends = states.reshape(-1, token.CHUNK, *states.shape[1:])[:, -1]
        ends = jnp.moveaxis(ends, 0, 2)
        entered = jnp.concatenate((op.state[:, :, None], ends[:, :, :-1]), axis=2)
        return entered, ends

    with jax.enable_x64(True):
        truth_enter, truth_end = jax.jit(chronological)(*widened((operands,)))
        q, k, v, g, beta = raw(operands)

        @jax.jit
        def chains(q, k, v, g, beta, state, truth_enter):
            steps = c.chunked(q, k, v, g, beta, token.CHUNK)
            inputs = tuple(jnp.moveaxis(x, 2, 0) for x in
                           (steps.key, steps.w, steps.u, steps.grown, steps.last, steps.kept))
            def one(state, step):
                state, _ = a.carry(state, step)
                return state, state

            _, accumulated = jax.lax.scan(one, state, inputs)

            def independent(unused, values):
                entered, *step = values
                return unused, a.carry(entered, tuple(step))[0]

            _, local = jax.lax.scan(independent, 0,
                                    (jnp.moveaxis(truth_enter.astype(jnp.float32), 2, 0), *inputs))
            return jnp.moveaxis(accumulated, 0, 2), jnp.moveaxis(local, 0, 2)

        accumulated, local = chains(q, k, v, g, beta, operands.state, truth_enter)
        rms = lambda x: np.sqrt(np.mean(np.asarray(x, np.float64)**2))
        for i in range(truth_end.shape[2]):
            exact = np.asarray(truth_end[:, :, i])
            ae = np.asarray(accumulated[:, :, i], np.float64) - exact
            le = np.asarray(local[:, :, i], np.float64) - exact
            last = min((i+1)*token.CHUNK, k.shape[1])-1
            key = np.asarray(k[:, last], np.float64)
            along = key[..., :, None] * np.sum(key[..., :, None]*ae, axis=-2, keepdims=True)
            print(f"CHUNK|{i}|accumulated={rms(ae):.12g}|local={rms(le):.12g}"
                  f"|propagated_delta={rms(ae-le):.12g}|along_key={rms(along):.12g}"
                  f"|orthogonal={rms(ae-along):.12g}", flush=True)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("name,seeds", [pytest.param("scalar delta", 40, id="scalar"),
                                       pytest.param("vector delta", 40, id="vector"),
                                       pytest.param("gated", 320, id="gated")])
def test_forward_probe(mode, name, seeds, monkeypatch):
    measure(mode, name, seeds, monkeypatch)


@pytest.mark.parametrize("mode", ("current", "wy-maps", "chrono-maps", "chrono-write",
                                  "tail1", "tail2", "tail4"))
def test_forward_cost(mode, monkeypatch):
    monkeypatch.setattr(c, "_chunk_delta_forward", functools.partial(prototype, mode))
    shape = (2, 4096, 16, 128)
    keys = jax.random.split(jax.random.key(20261023), 5)
    query, key, value = (jax.random.normal(k, shape) for k in keys[:3])
    key = key / jnp.linalg.norm(key, axis=-1, keepdims=True)
    g = -jax.nn.softplus(jax.random.normal(keys[3], shape[:-1] + (1,)) - 1.0)
    beta = jax.nn.sigmoid(jax.random.normal(keys[4], shape[:-1]))
    state = jnp.zeros((2, 16, 128, 128), jnp.float32)
    args = query, key, value, g, beta, state
    fn = jax.jit(functools.partial(c.chunk_delta_rule, chunk_size=64))
    compiled = fn.lower(*args).compile()
    memory = compiled.memory_analysis()
    peak = (memory.argument_size_in_bytes + memory.output_size_in_bytes
            + memory.temp_size_in_bytes - memory.alias_size_in_bytes)
    jax.block_until_ready(compiled(*args))
    times = []
    for _ in range(7):
        start = time.monotonic()
        jax.block_until_ready(compiled(*args))
        times.append(time.monotonic() - start)
    print(f"COST|{mode}|peak={peak}|seconds={np.median(times):.9g}|samples={times}", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "decompose":
        decompose(int(sys.argv[2]) if len(sys.argv) > 2 else 0)
    else:
        with pytest.MonkeyPatch.context() as patch:
            measure(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "scalar delta", 1, patch,
                    first=int(sys.argv[3]) if len(sys.argv) > 3 else 0)
