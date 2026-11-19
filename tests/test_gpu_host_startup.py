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

pytestmark = pytest.mark.gpu(reason="the host step's accelerator runs on the GPU it was prepared for")

COMMON = """
import logging
import re

import jax.numpy as jnp
import numpy as np

from test_host_training import HOST, streaming_decoder

TOKENS = np.stack([np.arange(row, row + 9) % 256 for row in range(1)]).astype(np.int32)


class Keyed(logging.Handler):
    \"\"\"Each program JAX keys its persistent cache by, as it logs them: its module's name, whether the cache
    held it, and the hash of each part of its key (the computation, the backend's version, the compile
    options, the accelerator's config, ...), gathered by the thread that keyed it. Its records are taken one
    at a time under the handler's own lock (`logging.Handler.handle`), which `emit` runs within.\"\"\"

    PART = re.compile(r"get_cache_key hash of serialized (.+): ([0-9a-f]+)$")
    READ = re.compile(r"(Persistent compilation cache hit|PERSISTENT COMPILATION CACHE MISS) for '(.+)' "
                      r"with key")

    def __init__(self):
        super().__init__(logging.DEBUG)
        self.programs, self.parts = [], {}

    def emit(self, record):
        message, thread = record.getMessage(), record.thread
        if part := self.PART.match(message):
            self.parts.setdefault(thread, {})[part[1]] = part[2]
        elif read := self.READ.match(message):
            self.programs.append([read[2], read[1].startswith("Persistent"), self.parts.pop(thread, {})])


KEYED = Keyed()
for name in ("jax._src.cache_key", "jax._src.compiler"):
    logging.getLogger(name).setLevel(logging.DEBUG)
    logging.getLogger(name).addHandler(KEYED)
"""
"""What both processes build: the bf16 decoder whose layers bank on the host, its one-row batch, and the
record of each program's persistent-cache key (`Keyed`)."""

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

count_compilation_cache()
trainer = streaming_decoder(HOST, dtype=jnp.bfloat16)
state, _, _ = trainer.place()
batch = {"text": jnp.asarray(TOKENS)}
batch = jax.device_put(batch, batch_shardings(trainer.device_mesh, batch))
before, keyed = compilation_cache_counts(), len(KEYED.programs)
step = trainer.compile(state, batch)
after, keys = compilation_cache_counts(), KEYED.programs[keyed:]
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
    prepared = reported(tmp_path / "prepare.py", [generation, str(topology)],
                        {**environment, "CUDA_VISIBLE_DEVICES": "", "JAX_PLATFORMS": "cpu",
                         "XLA_FLAGS": companion})
    ran = reported(tmp_path / "run.py", [],
                   {**environment, "JAX_PLATFORMS": "cuda,cpu", "XLA_FLAGS": companion})
    assert set(prepared["startup"]["by_backend"]) == {"cpu", "gpu"}, prepared["startup"]
    assert ran["counts"]["train/cache_misses"] == 0 and ran["counts"]["train/cache_hits"] > 0, (
        ran["counts"], unread(ran["keys"], prepared["keys"]))
    started, built = ran["startup"], prepared["startup"]
    assert started["programs"] == built["programs"], (started, built)


def unread(ran: list, prepared: list) -> dict[str, list[str]]:
    """Each program the run's startup missed the cache for, by its module's name, with the parts of its
    key no program of that name the build prepared shares (all of them where the build prepared none of
    that name): what keyed the run's compile apart from the build's."""
    built: dict[str, list[dict[str, str]]] = {}
    for module, _, parts in prepared:
        built.setdefault(module, []).append(parts)
    return {module: sorted(part for part, value in parts.items()
                           if all(other.get(part) != value for other in built.get(module, [])))
            for module, held, parts in ran if not held}
