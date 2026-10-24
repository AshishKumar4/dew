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
                      "input_digests": [int(gate.input_digest(draw(seed))) for seed in range(8)],
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
