"""Independent audit reproductions, with test-only controls removing each proposed cause."""

import contextlib
import json
import os
import subprocess
import sys

import jax
import jax.numpy as jnp
import pytest

from dew.nn.attention_blocks import NormalAttention
from dew.nn.sharding import boxed, declarations, declared_axes, parameter_path
from dew.optim import OptimConfig


MESH_PROBE = r"""
import json
import sys
import jax
import numpy as np
from jax.sharding import AxisType, Mesh, PartitionSpec as P
from dew.nn.kernels.options import OuterProgram

mesh_key, static = map(int, sys.argv[1:])

class WithConcreteMeshKey(OuterProgram):
    def _keyed(self, args, kwargs):
        key, dynamic, named = super()._keyed(args, kwargs)
        return (key, jax.sharding.get_mesh()), dynamic, named

def function(x, scale=1):
    return jax.lax.with_sharding_constraint(x, P("x")) * scale + 1

devices = jax.devices("cpu")
assert len(devices) == 4, devices
# Finish lazy imports and registration of trace-context config slots before measuring keys.
OuterProgram(lambda x: x + 1)(np.zeros(1, dtype=np.float32)).block_until_ready()
meshes = [Mesh(np.asarray(devices[start:start + 2]), ("x",), axis_types=(AxisType.Auto,))
          for start in (0, 2)]
assert meshes[0].abstract_mesh == meshes[1].abstract_mesh
arguments = {"static_argnames": ("scale",)} if static else {}
dew = (WithConcreteMeshKey if mesh_key else OuterProgram)(function, **arguments)
native = jax.jit(function, **arguments)
given = np.arange(8, dtype=np.float32)
kwargs = {"scale": 2} if static else {}
keys, executables, observed = [], [], []

def assignment(array):
    return [int(device.id) for device in array.sharding.mesh.devices.flat]

for mesh in meshes:
    with jax.set_mesh(mesh):
        keys.append(dew._keyed((given,), kwargs)[0])
        executables.append(dew.executable(given, **kwargs))
        actual, reference = dew(given, **kwargs), native(given, **kwargs)
        actual.block_until_ready()
        reference.block_until_ready()
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(reference))
        observed.append({"expected": [int(d.id) for d in mesh.devices.flat],
                         "dew": assignment(actual), "jax_jit": assignment(reference),
                         "jax_cache": native._cache_size(), "dew_cache": len(dew._ready)})

print(json.dumps({"jax": jax.__version__, "mesh_key": mesh_key, "static": static,
                  "same_dew_key": keys[0] == keys[1],
                  "same_dew_executable": executables[0] is executables[1],
                  "observed": observed}), flush=True)
assert [row["jax_jit"] for row in observed] == [row["expected"] for row in observed]
assert [row["jax_cache"] for row in observed] == [1, 2]
assert (keys[0] != keys[1]) == bool(mesh_key)
assert [row["dew"] for row in observed] == [row["expected"] for row in observed]
assert executables[0] is not executables[1]
"""


@pytest.mark.parametrize("static", [False, True], ids=["dynamic-key", "static-key"])
@pytest.mark.parametrize("mesh_key", [False, True], ids=["original-red", "mesh-key-control-green"])
def test_mesh_assignment_specializes_the_executable(mesh_key, static):
    """Exactly four CPU devices, disjoint equal-shape meshes, a retained callable and host inputs."""
    flags = [flag for flag in os.environ.get("XLA_FLAGS", "").split()
             if not flag.startswith("--xla_force_host_platform_device_count=")]
    environment = {**os.environ, "JAX_PLATFORMS": "cpu",
                   "XLA_FLAGS": " ".join([*flags, "--xla_force_host_platform_device_count=4"])}
    result = subprocess.run([sys.executable, "-c", MESH_PROBE, str(int(mesh_key)), str(int(static))],
                            capture_output=True, text=True, env=environment)
    print(result.stdout, flush=True)
    if result.returncode:
        pytest.fail(result.stdout + result.stderr, pytrace=False)


@pytest.mark.parametrize("mode", ["flat-original-red", "flat-boxed-control-green", "headed-control-green"])
def test_normal_attention_muon_initialization(mode):
    """Only the parameter-local axis declarations change between the flat red/green arms."""
    flat = mode != "headed-control-green"
    model = NormalAttention(query_dim=8, heads=2, dim_head=4, flat=flat, attention_impl="xla")
    params = model.init(jax.random.key(0), jnp.zeros((1, 2, 8), jnp.float32))["params"]
    declared = declarations(params)
    before = {"/".join(parameter_path(path)): {"shape": leaf.shape,
                                             "axes": declared_axes(path, leaf.ndim, declared)}
              for path, leaf in jax.tree_util.tree_leaves_with_path(params)}
    print(json.dumps({"mode": mode, "registry": before}), flush=True)
    table = {parameter_path(path): (("heads", "embed") if parameter_path(path)[0] == "to_out_0"
                                   else ("embed", "heads"))[-leaf.ndim:]
             for path, leaf in jax.tree_util.tree_leaves_with_path(params)}
    context = boxed(table) if mode == "flat-boxed-control-green" else contextlib.nullcontext()
    with context:
        effective = {"/".join(parameter_path(path)): declared_axes(path, leaf.ndim, declared)
                     for path, leaf in jax.tree_util.tree_leaves_with_path(params)}
        print(json.dumps({"mode": mode, "effective": effective}), flush=True)
        solver = OptimConfig(optimizer="muon", learning_rate=1e-3).build(4)
        solver.init(params)
