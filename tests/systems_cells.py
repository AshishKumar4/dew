"""Two of a run's systems on the GPU, cells of its flow (tests/test_gpu_systems.py), each run in fresh
interpreters of its own:

- Rematerialization is invisible to everything but memory use: a compilation counts the memory kept alive
  outside its state, in a fresh 2 GiB allocator (`compilation_counts_memory_kept_alive_outside_its_state`).
- The same program from two checkouts of Dew shares one compiled executable: a Pallas kernel's Triton
  module carries its source locations, which the pinned JAX's cache key strips (constraints.txt;
  `two_checkouts_share_one_compiled_kernel`).
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

ROOT = Path(__file__).resolve().parents[1]

PROGRAM = """
import json, sys
import jax, jax.numpy as jnp
from dew.cache import enable_compilation_cache
from dew.nn.norms import rms_normalized
from dew.telemetry.instrumentation import compilation_cache_counts, count_compilation_cache
enable_compilation_cache(sys.argv[1])
count_compilation_cache()
x, scale = jnp.ones((4, 128, 1024), jnp.bfloat16), jnp.ones((1024,), jnp.float32)
jax.jit(lambda x, scale: rms_normalized(x, scale, 1e-6, jnp.bfloat16, False, False, True,
                                        implementation="pallas")).lower(x, scale).compile()
print(json.dumps(compilation_cache_counts()))
"""


def two_checkouts_share_one_compiled_kernel(tmp_path):
    """A second checkout (here the same tree under another path) finds the first's executable."""
    other = tmp_path / "checkout" / "src"
    other.mkdir(parents=True)
    (other / "dew").symlink_to(ROOT / "src" / "dew", target_is_directory=True)
    counts = []
    for source in (ROOT / "src", other):
        path = os.pathsep.join(filter(None, (str(source), os.environ.get("PYTHONPATH"))))
        done = subprocess.run([sys.executable, "-c", PROGRAM, str(tmp_path / "cache")], capture_output=True,
                              text=True, timeout=600, env={**os.environ, "PYTHONPATH": path})
        assert done.returncode == 0, done.stderr[-3000:]
        counts.append(json.loads(done.stdout.splitlines()[-1]))
    assert counts[0]["train/cache_misses"] >= 1
    assert counts[1]["train/cache_misses"] == 0 and counts[1]["train/cache_hits"] >= 1


def compilation_counts_memory_kept_alive_outside_its_state(tmp_path):
    """A fresh 2 GiB allocator isolates the accounting regression from
    fragmentation left by earlier GPU tests. Its real held buffers force
    the compiler to fit the remaining memory, then the step runs there.
    It compiles into a cache of its own: a rung recorded beside a shared
    cache (`dew.training.rungs`), as its own held buffers record one, would
    start the next run's first compile above the whole logits.
    """
    if jax.device_count() != 1 or shutil.which("nvidia-smi") is None:
        pytest.skip("needs one NVIDIA GPU with memory reporting")
    rows = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True, timeout=10).stdout.splitlines()
    if len(rows) != 1:
        pytest.skip("the isolated allocator experiment needs one physical GPU")
    name, total, free = (value.strip() for value in rows[0].split(","))
    if name != jax.devices()[0].device_kind:
        pytest.skip("cannot identify the current GPU's physical memory")
    pool_mib, runtime_mib = 2048, 1024
    if int(free) < pool_mib + runtime_mib:
        pytest.skip("needs 2 GiB for the child pool and 1 GiB for its CUDA runtime")
    root = Path(__file__).resolve().parents[1]
    environment = {**os.environ, "JAX_PLATFORMS": "cuda",
                   "PYTHONPATH": os.pathsep.join((str(root / "src"), str(root / "tests"))),
                   "XLA_PYTHON_CLIENT_ALLOCATOR": "bfc", "JAX_COMPILATION_CACHE_DIR": str(tmp_path),
                   "XLA_PYTHON_CLIENT_MEM_FRACTION": str(pool_mib / int(total)),
                   "XLA_PYTHON_CLIENT_PREALLOCATE": "true"}
    done = subprocess.run(
        [sys.executable, "-c", "from systems_cells import _resident_memory_step; _resident_memory_step()"],
        cwd=root, env=environment, capture_output=True, text=True, timeout=180)
    assert done.returncode == 0, done.stdout + done.stderr


def _resident_memory_step():
    """Compile and run with real external buffers in the bounded child pool."""
    import optax

    from dew.nn.backbones.causal_transformer import CausalTransformer
    from dew.objectives.lm import LMObjective
    from dew.training import Trainer

    device = jax.devices()[0]
    model = CausalTransformer(vocab_size=65536, emb_features=64, num_layers=1,
                              num_heads=2, mlp_features=128, max_seq_len=256,
                              dtype=jnp.bfloat16)
    objective = LMObjective(model, seq_len=256)
    trainer = Trainer(objective, optax.adam(1e-4), key=jax.random.key(0))
    state, _, _ = trainer.place()
    batch = {"text": jnp.zeros((4, 257), jnp.int32)}
    trainer.compile(state, batch)
    assert objective.head_tile is None, "the unconstrained step must keep the whole logits"

    def additional_bytes():
        assert trainer.executable is not None
        stats = trainer.executable.memory_analysis()
        assert stats is not None
        return stats.output_size_in_bytes - stats.alias_size_in_bytes + stats.temp_size_in_bytes

    memory = device.memory_stats()
    assert memory is not None
    reserve = memory["bytes_limit"] - memory["bytes_in_use"] - additional_bytes() // 2
    assert reserve > 0
    allocate = jax.jit(lambda value, size: jnp.broadcast_to(value, (size,)), static_argnums=1)
    block = 64 * 2**20
    zero = jnp.asarray(0, jnp.uint8)
    held = [allocate(zero, min(block, reserve - start)) for start in range(0, reserve, block)]
    jax.block_until_ready(held)
    step = trainer.compile(state, batch)
    memory = device.memory_stats()
    assert memory is not None
    assert additional_bytes() <= memory["bytes_limit"] - memory["bytes_in_use"]
    _, loss, _, finite, _, _ = jax.block_until_ready(step(state, batch))
    assert bool(finite), float(loss)
    jax.block_until_ready(held)
