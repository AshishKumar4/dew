"""Diagnostic chunk-parallel recurrence adjoint; unchanged chunked primal."""
import jax
import jax.numpy as jnp

from dew.nn.kernels import recurrence as r, recurrence_algebra as a, recurrence_chunk as c

ORIGINAL = c.chunk_delta_rule


def chunk_adjoint(query, key, value, g, beta, state=None, chunk_size=c.CHUNK_SIZE, *, starts=None,
                  delta=True, scale=None, dtype=None, source_method="scan"):
    b, s, h, dk = key.shape
    dv = value.shape[-1]
    work = jnp.promote_types(jnp.result_type(query, key, value), jnp.float32)
    if state is None:
        state = jnp.zeros((b, h, dk, dv), work)
    query, key, value, g, beta, state = (x.astype(work) for x in (query, key, value, g, beta, state))
    result_dtype = jnp.result_type(query, key, value) if dtype is None else dtype
    scale = dk**-.5 if scale is None else scale
    pad = (-s) % chunk_size
    nc = (s + pad) // chunk_size

    def block(x):
        x = jnp.pad(x, [(0, 0), (0, pad)] + [(0, 0)] * (x.ndim - 2))
        return jnp.moveaxis(x, 2, 1).reshape(b, h, nc, chunk_size, *x.shape[3:])

    def unblock(x):
        return jnp.moveaxis(x.reshape(b, h, s + pad, *x.shape[4:]), 1, 2)[:, :s]

    resets = jnp.zeros((b, s), bool) if starts is None else starts
    resets = block(jnp.broadcast_to(resets[..., None], (b, s, h)))

    @jax.custom_vjp
    def run(query, key, value, g, beta, state):
        return ORIGINAL(query, key, value, g, beta, state, chunk_size, starts=starts,
                        delta=delta, scale=scale, dtype=result_dtype)

    def forward(query, key, value, g, beta, state):
        steps = c.chunked(query, key, value, g, beta, chunk_size, starts=starts, delta=delta, scale=scale)
        state = jax.lax.optimization_barrier(state)
        entered, corrected, final = c.xla_chunk_states(
            steps.key, steps.w, steps.u, steps.grown, steps.last, state, steps.kept)
        core = a.readout(steps.query, steps.entering, entered, steps.key, steps.pairs, corrected)
        out = unblock(core).astype(result_dtype), final
        transition, _ = a.transition(steps.key, steps.w, steps.u, steps.grown, steps.last, steps.kept)
        source_operands = ((steps.query * steps.entering, steps.w,
                            a.paired("scores", steps.query, steps.key, steps.pairs))
                           if source_method == "wy" else None)
        return out, (steps.query, steps.key, block(value), block(g), block(beta), entered, transition,
                     source_operands)

    def backward(residual, cotangent):
        query, key, value, g, beta, entered, transition, source_operands = residual
        cot, final_cot = cotangent
        cot = block(cot.astype(work))
        tokens = query, key, value, g, beta, resets
        time_axis = lambda x: jnp.moveaxis(x, 3, 0)

        def local(state, query, key, value, g, beta, reset):
            state = jnp.where(reset[..., None, None], 0.0, state)
            return r.update(state, query, key, value, jnp.exp(g), beta[..., None], delta=delta)

        def primal_step(state, token):
            _, final = local(state, *token)
            return final, state

        _, before = jax.lax.scan(primal_step, entered, tuple(map(time_axis, tokens)))

        def source_step(ds, inputs):
            state, cot, *token = inputs
            _, pull = jax.vjp(lambda state: local(state, *token), state)
            return pull((cot, ds))[0], None

        if source_method == "wy":
            decayed_query, w, scores = source_operands
            corrected_cot = a._remembered_transpose(scores, cot, a.portable, "local")
            source = (a._remembered_transpose(decayed_query, cot, a.portable, "memory")
                      - a.portable("corrected", jnp.swapaxes(w, -1, -2), corrected_cot))
        else:
            source, _ = jax.lax.scan(source_step, jnp.zeros_like(entered),
                                     (before, time_axis(cot), *map(time_axis, tokens)), reverse=True)

        def across(ds, inputs):
            transition, source = inputs
            return a.portable("compose", jnp.swapaxes(transition, -1, -2), ds) + source, ds

        _, ends = jax.lax.scan(across, final_cot, (jnp.moveaxis(transition, 2, 0),
                                                  jnp.moveaxis(source, 2, 0)), reverse=True)
        ends = jnp.moveaxis(ends, 0, 2)

        def adjoint_step(ds, inputs):
            state, cot, query, key, value, g, beta, reset = inputs
            _, pull = jax.vjp(lambda state, q, k, v, g, beta: local(state, q, k, v, g, beta, reset),
                              state, query, key, value, g, beta)
            ds, *gradients = pull((cot, ds))
            return ds, tuple(gradients)

        initial, gradients = jax.lax.scan(adjoint_step, ends,
                                          (before, time_axis(cot), *map(time_axis, tokens)), reverse=True)
        dq, dk, dv, dg, dbeta = (unblock(jnp.moveaxis(x, 0, 3)) for x in gradients)
        return dq * scale, dk, dv, dg, dbeta, initial[:, :, 0]

    run.defvjp(forward, backward)
    return run(query, key, value, g, beta, state)


def chunk_adjoint_wy_source(*args, **kwargs):
    return chunk_adjoint(*args, **kwargs, source_method="wy")
