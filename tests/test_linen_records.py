"""A plain Flax Linen graph records itself and rebuilds, in a fresh process,
into a model that loads its original weights and computes the same bits:
nested layers' default initializers are left out of the record at every
level, as the root's are, and a layer the graph holds twice stays one layer
with one set of weights, while two equal layers stay two."""

import json
import os
import subprocess
import sys
from pathlib import Path

import flax.linen as nn
import flax.serialization
import jax
import numpy as np

from dew.config import ModelConfig
from dew.registry import to_record


class Residual(nn.Module):
    """A user module holding its submodules as fields."""

    inner: nn.Dense
    norm: nn.LayerNorm

    def __call__(self, x):
        return x + self.inner(self.norm(x))


class Tagged(nn.Module):
    """A user module whose `tags` are plain JSON data, beside one layer it holds twice."""

    later: nn.Dense
    earlier: nn.Dense
    tags: dict[str, object]

    def __call__(self, x):
        return self.earlier(self.later(x))


def models() -> dict[str, nn.Module]:
    shared = nn.Dense(8)
    return {"sequential": nn.Sequential([nn.Dense(16), nn.LayerNorm(), nn.Dense(8, use_bias=False)]),
            "residual": Residual(nn.Dense(8, kernel_init=nn.initializers.zeros), nn.LayerNorm(epsilon=1e-3)),
            "shared": nn.Sequential([shared, shared]),
            "separate": nn.Sequential([nn.Dense(8), nn.Dense(8)])}


REBUILD = """
import json, sys
import flax.serialization, jax, jax.numpy as jnp, numpy as np
import test_linen_records
from dew.config import ModelConfig
out = {}
for name, saved in json.loads(sys.stdin.read()).items():
    model = ModelConfig.from_dict(saved["record"]).build()
    x = jnp.asarray(saved["x"], jnp.float32)
    params = flax.serialization.from_bytes(jax.eval_shape(model.init, jax.random.key(0), x),
                                           bytes.fromhex(saved["params"]))
    out[name] = [np.asarray(model.apply(params, x)).tobytes().hex(), sorted(params["params"])]
print(json.dumps(out))
"""


def test_a_nested_linen_model_rebuilds_from_its_record_with_its_weights_in_a_fresh_process():
    x = jax.random.normal(jax.random.key(1), (2, 8))
    held, expected = {}, {}
    for name, model in models().items():
        params = model.init(jax.random.key(0), x)
        held[name] = {"record": to_record(ModelConfig.from_model(model), ModelConfig),
                      "params": flax.serialization.to_bytes(params).hex(), "x": np.asarray(x).tolist()}
        expected[name] = [np.asarray(model.apply(params, x)).tobytes().hex(), sorted(params["params"])]
    root = Path(__file__).resolve().parents[1]
    # The fresh process runs on the lane's backend, under its flags, as this one does: the same bits.
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(root / "src"), str(root / "tests")])}
    done = subprocess.run([sys.executable, "-c", REBUILD], input=json.dumps(held), capture_output=True,
                          text=True, env=env, timeout=300, check=False)
    assert done.returncode == 0, done.stderr[-2000:]
    assert json.loads(done.stdout.splitlines()[-1]) == expected
    assert expected["shared"][1] == ["layers_0"] and expected["separate"][1] == ["layers_0", "layers_1"]


def test_data_shaped_like_a_shared_module_stays_data_beside_a_real_one():
    """`tags` holds mappings shaped like shared-module definitions, the same
    number twice and the number the real shared layer takes. They are data,
    so the record reads them back as written; and with its keys sorted, the
    record refers to the shared layer (`earlier`) before defining it
    (`later`), and still builds one layer with one set of weights."""
    shared = nn.Dense(8)
    tags = {"a": {"class": "row", "share": 0}, "b": {"class": "row", "share": 0}}
    model = Tagged(later=shared, earlier=shared, tags=tags)
    x = jax.random.normal(jax.random.key(1), (2, 8))
    params = model.init(jax.random.key(0), x)
    record = json.loads(json.dumps(to_record(ModelConfig.from_model(model), ModelConfig), sort_keys=True))
    rebuilt = ModelConfig.from_dict(record).build()
    assert rebuilt.tags == tags
    assert sorted(params["params"]) == ["later"]
    np.testing.assert_array_equal(rebuilt.apply(params, x), model.apply(params, x))
