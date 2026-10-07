"""A plain Flax Linen graph records itself and rebuilds, in a fresh process,
into a model that loads its original weights and computes the same bits:
nested layers' default initializers are left out of the record at every
level, as the root's are."""

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
from dew.registry import imported, to_record


class Residual(nn.Module):
    """A user module holding its submodules as fields."""

    inner: nn.Dense
    norm: nn.LayerNorm

    def __call__(self, x):
        return x + self.inner(self.norm(x))


def models() -> dict[str, nn.Module]:
    return {"sequential": nn.Sequential([nn.Dense(16), nn.LayerNorm(), nn.Dense(8, use_bias=False)]),
            "residual": Residual(nn.Dense(8, kernel_init=nn.initializers.zeros), nn.LayerNorm(epsilon=1e-3))}


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
    out[name] = np.asarray(model.apply(params, x)).tobytes().hex()
print(json.dumps(out))
"""


def test_a_nested_linen_model_rebuilds_from_its_record_with_its_weights_in_a_fresh_process():
    x = jax.random.normal(jax.random.key(1), (2, 8))
    held, expected = {}, {}
    for name, model in models().items():
        params = model.init(jax.random.key(0), x)
        held[name] = {"record": to_record(ModelConfig.from_model(model), ModelConfig),
                      "params": flax.serialization.to_bytes(params).hex(), "x": np.asarray(x).tolist()}
        expected[name] = np.asarray(model.apply(params, x)).tobytes().hex()
    root = Path(__file__).resolve().parents[1]
    env = {**os.environ, "JAX_PLATFORMS": "cpu",
           "PYTHONPATH": os.pathsep.join([str(root / "src"), str(root / "tests")])}
    done = subprocess.run([sys.executable, "-c", REBUILD], input=json.dumps(held), capture_output=True,
                          text=True, env=env, timeout=300, check=False)
    assert done.returncode == 0, done.stderr[-2000:]
    assert json.loads(done.stdout.splitlines()[-1]) == expected


def test_a_nested_layers_default_initializer_is_left_out_and_a_given_one_is_kept():
    record = ModelConfig.from_model(models()["residual"]).fields
    assert "bias_init" not in record["inner"] and "scale_init" not in record["norm"]
    assert imported(record["inner"]["kernel_init"]["function"]) is nn.initializers.zeros
