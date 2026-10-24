"""Paired diagnostics for the JAX switch's parameter-layout site."""

import json

import jax
import jax.numpy as jnp
import numpy as np
import reference_error as gate
import test_attention_template as site

CASE = site.CASES["causal-documents-window"]


def _report(name, draws):
    print(json.dumps({
        "name": name,
        "jax": jax.__version__,
        "ratio_interval": gate._judged(draws, 0),
        "ours_rms": float(np.sqrt(np.mean(draws.mine))),
        "reference_rms": float(np.sqrt(np.mean(draws.theirs))),
        "per_seed": np.sqrt(np.stack([draws.mine[:8, 0], draws.theirs[:8, 0]], axis=-1)).tolist(),
        "judgement": gate._judgement(draws, name),
    }), flush=True)


def test_sized_layout_site():
    ours, reference, draw, label = site.layout_site(CASE, jnp.float32)
    seeds = gate.seeds_for(reference, draw, label)
    paired_seeds = gate.seeds_for(reference, draw, label, dew_fn=ours)
    print(json.dumps({"seeds_for": seeds, "paired_seeds_for": paired_seeds,
                      "input_digests": [int(gate.input_digest(dict(zip(
                          ("q", "k", "v"), draw(seed), strict=True)))) for seed in range(8)],
                      "devices": [str(device) for device in jax.devices()]}), flush=True)
    for count in sorted({8, 40, seeds, 5 * seeds, 256}):
        _report(f"native-{count}", gate._drawn(ours, reference, draw, range(count)))
    gate.assert_as_exact(ours, reference, draw, label, seeds=seeds)


def test_layout_site_with_matching_accumulation():
    ours, reference, draw, label = site.layout_site(CASE, jnp.float32)

    @jax.jit
    def same_sums(*inputs):
        with jax.default_matmul_precision("F32_F32_F32"):
            return reference(*inputs)

    seeds = gate.seeds_for(same_sums, draw, label)
    print(json.dumps({"same_sums_seeds_for": seeds}), flush=True)
    for name, fn in (("native", reference), ("same-sums", same_sums)):
        _report(name, gate._drawn(ours, fn, draw, range(5 * seeds), truth_fn=reference))
    gate.assert_as_exact(ours, same_sums, draw, label, seeds=seeds, truth_fn=reference)


def test_layout_truth_bias_rounding():
    ours, reference, draw, label = site.layout_site(CASE, jnp.float32)
    with jax.enable_x64(new_val=True):
        inputs = gate.widened(draw(0))
        mine, theirs = np.asarray(ours(*inputs)), np.asarray(reference(*inputs))
    print(json.dumps({"float64_arms_apart_rms": float(np.sqrt(np.mean((mine - theirs) ** 2))),
                      "float64_output_rms": float(np.sqrt(np.mean(mine ** 2)))}), flush=True)
    _report("own-float64-truth", gate._drawn(ours, reference, draw, range(256), truth_fn=ours))


def _pipeline(replace):
    queries, keys = site.geometry(CASE)
    spans = site.template.spans(queries, keys, site.ROWS, **site.arguments(CASE))
    seen = site.template.dense(spans)
    streams = jax.random.split(jax.random.key(5), 4)
    temperature = jax.random.uniform(streams[0], (site.ROWS, 4, queries), jnp.float32, 0.5, 1.5)
    key_bias = jax.random.normal(streams[1], (site.ROWS, 4, keys), jnp.float32)
    dense_bias = 0.5 * jax.random.normal(streams[2], (site.ROWS, 4, queries, keys), jnp.float32)
    gates = jax.random.normal(streams[3], (site.ROWS, queries, 4, 32), jnp.float32)

    def wider(fn, *inputs):
        with jax.enable_x64(new_val=True):
            return fn(*(x.astype(jnp.float64) for x in inputs)).astype(inputs[0].dtype)

    def run(query, key, value):
        wide = jnp.promote_types(query.dtype, jnp.float32)

        def one(row, head):
            q, k, v = query[row, :, head], key[row, :, head // 2], value[row, :, head // 2]

            def dot(q, k):
                return jnp.einsum("qd,kd->qk", q, k, preferred_element_type=q.dtype)

            s = wider(dot, q, k) if replace in ("qk", "dots") else dot(q, k)
            s = s * jnp.asarray(32 ** -0.5, wide)
            s = (s * temperature[row, head, :, None].astype(wide)
                 + key_bias[row, head, None, :].astype(wide) + dense_bias[row, head].astype(wide))
            s = jnp.where(seen[row], s, -jnp.inf)
            m = jnp.max(s, axis=-1, initial=-jnp.inf)
            safe = jnp.where(jnp.isfinite(m), m, 0)
            delta = s - safe[:, None]
            p = wider(jnp.exp, delta) if replace == "exp" else jnp.exp(delta)
            total = (wider(lambda x: jnp.sum(x, axis=-1), p) if replace == "sum"
                     else jnp.sum(p, axis=-1))

            def values(p, v):
                return jnp.einsum("qk,kd->qd", p, v, preferred_element_type=p.dtype)

            p = p.astype(value.dtype)
            acc = wider(values, p, v) if replace in ("pv", "dots") else values(p, v)
            o = jnp.where(total[:, None] > 0, acc / jnp.where(total > 0, total, 1)[:, None], 0)
            o = o * jax.nn.sigmoid(gates[row, :, head].astype(o.dtype))
            return o.astype(query.dtype)

        out = jax.vmap(jax.vmap(one, in_axes=(None, 0)), in_axes=(0, None))(
            jnp.arange(site.ROWS), jnp.arange(4))
        return jnp.swapaxes(out, 1, 2)

    return jax.jit(run)


def test_isolate_template_operations():
    ours, reference, draw, _ = site.layout_site(CASE, jnp.float32)
    inputs = draw(0)
    raw = _pipeline("none")
    print(json.dumps({"pipeline_replays_template_bitwise": bool(np.array_equal(
        np.asarray(raw(*inputs)), np.asarray(ours(*inputs))))}), flush=True)
    for replacement in ("none", "qk", "pv", "dots", "exp", "sum"):
        _report("widen-" + replacement, gate._drawn(
            _pipeline(replacement), reference, draw, range(40), truth_fn=ours))
