"""Programs compiled ahead for this GPU's generation by a process that sees no GPU load in a fresh process on
the GPU and run there with nothing compiled, bit for bit as the same programs jitted (ledger 2716: a GPU node
spends its time on its device's work alone; 2026-11-02, an H100 ran a CPU host's sm90 build of a dot and the
template attention's forward so)."""

import os
import pickle
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dew.nn.attention import scaled_dot_product_attention
from dew.training.ahead import BuildTarget, compiled_in, dumps

pytestmark = pytest.mark.gpu(reason="a program compiled ahead runs on the GPU it was built for")

PROGRAMS = """
import jax
import jax.numpy as jnp

from dew.nn.attention import scaled_dot_product_attention


def attention(q, k, v):
    return scaled_dot_product_attention(q, k, v, implementation="template", causal=True)


def programs(generation):
    square = jax.ShapeDtypeStruct((128, 128), jnp.float32)
    found = {"dot": (lambda x, y: x @ y, [square, square])}
    # The template's forward is Mosaic GPU's on Hopper alone, the route the pilot ran ahead.
    if generation == "sm90":
        query = jax.ShapeDtypeStruct((1, 256, 8, 128), jnp.bfloat16)
        key = jax.ShapeDtypeStruct((1, 256, 2, 128), jnp.bfloat16)
        found["attention"] = (attention, [query, key, key])
    return found
"""
BUILD = """
import sys

from dew.training.ahead import compile_programs
from programs import programs

compile_programs(programs(sys.argv[1]), sys.argv[1]).save(sys.argv[2])
"""
RUN = """
import pickle
import sys

import jax
import numpy as np

from dew.training.ahead import AheadPrograms, executable_only

loaded = AheadPrograms.load(sys.argv[1]).loaded()
with open(sys.argv[2], "rb") as file:
    arrays = pickle.load(file)
with executable_only():
    results = {name: np.asarray(jax.block_until_ready(program(*arrays[name])))
               for name, program in loaded.items()}
with open(sys.argv[3], "wb") as file:
    pickle.dump(results, file)
"""


def attention(q, k, v):
    return scaled_dot_product_attention(q, k, v, implementation="template", causal=True)


def test_programs_built_with_no_gpu_run_on_this_gpu_with_nothing_compiled_bit_for_bit_as_jitted(tmp_path):
    generation = BuildTarget.of(jax.devices()).generation
    for name, text in (("programs.py", PROGRAMS), ("build.py", BUILD), ("run.py", RUN)):
        (tmp_path / name).write_text(text)
    paths = os.pathsep.join([str(tmp_path), os.environ.get("PYTHONPATH", "")])
    environment = {**os.environ, "PYTHONPATH": paths}
    # Built where no GPU is visible, as a CPU host builds them.
    build = [sys.executable, str(tmp_path / "build.py"), generation, str(tmp_path / "programs.dew-aot")]
    subprocess.run(build, env={**environment, "CUDA_VISIBLE_DEVICES": ""}, check=True)
    keys = jax.random.split(jax.random.key(0), 5)
    arrays = {"dot": [np.asarray(jax.random.normal(key, (128, 128), jnp.float32)) for key in keys[:2]]}
    functions = {"dot": lambda x, y: x @ y}
    if generation == "sm90":
        arrays["attention"] = [np.asarray(jax.random.normal(keys[2], (1, 256, 8, 128), jnp.bfloat16)),
                               *(np.asarray(jax.random.normal(key, (1, 256, 2, 128), jnp.bfloat16))
                                 for key in keys[3:])]
        functions["attention"] = attention
    (tmp_path / "arguments.pkl").write_bytes(pickle.dumps(arrays))
    # Loaded and run in a fresh process whose compilers' dumps must stay empty.
    subprocess.run([sys.executable, str(tmp_path / "run.py"), str(tmp_path / "programs.dew-aot"),
                    str(tmp_path / "arguments.pkl"), str(tmp_path / "outputs.pkl")],
                   env=dumps(tmp_path / "dumps", environment), check=True)
    assert compiled_in(tmp_path / "dumps") == []
    ahead = pickle.loads((tmp_path / "outputs.pkl").read_bytes())
    for name, function in functions.items():
        jitted = np.asarray(jax.jit(function)(*arrays[name]))
        assert ahead[name].dtype == jitted.dtype and ahead[name].tobytes() == jitted.tobytes(), name
