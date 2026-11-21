"""A step that keeps its variables on the host, prepared where no GPU is visible (`prepare_host_step`), starts
on this GPU reading every program from JAX's persistent cache, compiling none, and steps with nothing traced,
lowered or compiled (row 7908; Main's choice: the shared cache, no file)."""

import json
import os
import subprocess
import sys
from pathlib import Path

import jax
import pytest
from cache_keys import unread

pytestmark = pytest.mark.gpu(reason="the host step's accelerator runs on the GPU it was prepared for")

COMMON = """
import jax.numpy as jnp
import numpy as np
from cache_keys import keyed

from test_host_training import HOST, streaming_decoder

TOKENS = np.stack([np.arange(row, row + 9) % 256 for row in range(1)]).astype(np.int32)
KEYED = keyed()
"""
"""What both processes build: the bf16 decoder whose layers bank on the host, its one-row batch, and the
record of each program's persistent-cache key (`cache_keys.keyed`)."""

PREPARE = COMMON + """
import json
import sys

from dew.training.ahead import host_gpus, prepare_host_step

generation, topology = sys.argv[1], sys.argv[2]
trainer = streaming_decoder(HOST, dtype=jnp.bfloat16)
trainer.device_mesh = trainer.mesh.build(list(host_gpus(generation, topology, 1)))
print(json.dumps({"startup": prepare_host_step(trainer, {"text": TOKENS}).record(), "keys": KEYED.programs}))
"""
"""The build: its trainer's mesh the host's GPU by its serialized topology, no GPU visible."""

RUN = COMMON + """
import json

import jax

from dew.distributed.batches import batch_shardings
from dew.telemetry.compiles import executable_only
from dew.telemetry.instrumentation import compilation_cache_counts, count_compilation_cache
from dew.training.ahead import step_batch

count_compilation_cache()
trainer = streaming_decoder(HOST, dtype=jnp.bfloat16)
state, _, _ = trainer.place()
# The batch as `fit` hands its step one (`step_batch`), as the prepare compiled for it (rows 8372, 8446: the
# raw batch made the start, the begun step and its end other programs than the prepared ones).
batch = step_batch(trainer, {"text": jnp.asarray(TOKENS)})
batch = jax.device_put(batch, batch_shardings(trainer.device_mesh, batch))
before, at = compilation_cache_counts(), len(KEYED.programs)
step = trainer.compile(state, batch)
after, keys = compilation_cache_counts(), KEYED.programs[at:]
for _ in range(3):
    with executable_only():
        state, *_ = step(state, batch)
print(json.dumps({"startup": trainer._compiler.host_startup.record(),
                  "counts": {name: after[name] - before[name] for name in after}, "keys": keys}))
"""
"""The run on this GPU: its startup over the cache the build filled, then three steps compiling nothing."""


def reported(script: Path, arguments: list[str], environment: dict[str, str]) -> dict:
    done = subprocess.run([sys.executable, str(script), *arguments], env=environment, capture_output=True,
                          text=True, check=True)
    return json.loads(done.stdout.splitlines()[-1])


def test_a_host_step_prepared_without_a_gpu_starts_here_reading_every_program_from_the_cache(tmp_path):
    """A host step's programs compiled where no GPU is visible for this GPU (its serialized topology,
    `host_gpus`), the CPU companion's on this host's CPUs, key JAX's persistent cache as this GPU's run
    compiles them: the run's startup misses the cache for none, and its steps then trace, lower and compile
    nothing (`executable_only`)."""
    from jax.experimental import topologies

    from dew.training.ahead import BuildTarget

    topology = tmp_path / "topology.bin"
    topology.write_bytes(topologies.get_attached_topology("cuda").serialize())
    tests = str(Path(__file__).resolve().parent)
    environment = {**os.environ, "PYTHONPATH": os.pathsep.join([tests, os.environ.get("PYTHONPATH", "")]),
                   "JAX_COMPILATION_CACHE_DIR": str(tmp_path / "cache"),
                   "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS": "0",
                   "JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES": "-1"}
    companion = f"{environment.get('XLA_FLAGS', '')} --xla_force_host_platform_device_count=1".strip()
    for name, text in (("prepare.py", PREPARE), ("run.py", RUN)):
        (tmp_path / name).write_text(text)
    generation = BuildTarget.of(jax.devices()[:1]).generation
    # Each side's programs dumped (`JAX_DUMP_IR_TO`), which a miss names how they differ by.
    prepared = reported(tmp_path / "prepare.py", [generation, str(topology)],
                        {**environment, "CUDA_VISIBLE_DEVICES": "", "JAX_PLATFORMS": "cpu",
                         "XLA_FLAGS": companion, "JAX_DUMP_IR_TO": str(tmp_path / "prepared")})
    ran = reported(tmp_path / "run.py", [],
                   {**environment, "JAX_PLATFORMS": "cuda,cpu", "XLA_FLAGS": companion,
                    "JAX_DUMP_IR_TO": str(tmp_path / "ran")})
    assert set(prepared["startup"]["by_backend"]) == {"cpu", "gpu"}, prepared["startup"]
    # Text whole: pytest cuts a dict's repr in an assertion's message at a few entries.
    missed = unread(ran["keys"], tmp_path / "ran", prepared["keys"], tmp_path / "prepared")
    assert ran["counts"]["train/cache_misses"] == 0 and ran["counts"]["train/cache_hits"] > 0, json.dumps(
        {"counts": ran["counts"], "missed": missed}, indent=1, default=str)
    started, built = ran["startup"], prepared["startup"]
    assert started["programs"] == built["programs"], (started, built)

