"""Programs compiled ahead for this GPU's generation by a process that sees no GPU load in a fresh process on
the GPU and run there with nothing compiled, bit for bit as the same programs compiled here through Dew's
compile point for the generation's facts, as the build compiles them (ledger 2716: a GPU node spends its time
on its device's work alone; 2026-11-02, an H100 ran a CPU host's sm90 build of a dot and the template
attention's forward so)."""

import json
import os
import pickle
import subprocess
import sys
from pathlib import Path

import jax
import numpy as np
import pytest
from ahead_programs import arguments, programs

from dew.nn.kernels.generation import assumed, built_target
from dew.nn.kernels.options import owned_program
from dew.training.ahead import BuildTarget, compiled_in, dumps

pytestmark = pytest.mark.gpu(reason="a program compiled ahead runs on the GPU it was built for")

BUILD = """
import sys

from dew.training.ahead import compile_programs
from ahead_programs import programs

compile_programs(programs(sys.argv[1]), sys.argv[1], int(sys.argv[3])).save(sys.argv[2])
"""
RUN = """
import pickle
import sys

import jax
import numpy as np

from dew.telemetry.compiles import executable_only
from dew.training.ahead import AheadPrograms

loaded = AheadPrograms.load(sys.argv[1]).loaded()
with open(sys.argv[2], "rb") as file:
    arrays = pickle.load(file)
with executable_only():
    results = {name: np.asarray(jax.block_until_ready(program(*arrays[name])))
               for name, program in loaded.items()}
with open(sys.argv[3], "wb") as file:
    pickle.dump(results, file)
"""
REFUSED = """
import sys

from dew.training.ahead import AheadPrograms

try:
    AheadPrograms.load(sys.argv[1]).loaded()
except ValueError:
    sys.exit(0)
sys.exit("its programs loaded in a process whose XLA computes otherwise")
"""
"""A fresh process loading the programs at argv[1], which its loader must refuse."""
TRAIN_REFUSED = """
import sys

import jax
from test_compile_ahead import BATCH, SEQ_LEN, VOCAB, fine_tune
from token_support import token_rows

from dew.training import Inspection, MeshSpec
from dew.training.ahead import AheadOfTime

run = fine_tune(MeshSpec().build(jax.devices()[:1]), ahead=AheadOfTime.load(sys.argv[1]), xla=True, split=2,
                inspection=Inspection(every=2), dtype="bfloat16")
try:
    run.fit(token_rows(VOCAB, BATCH, SEQ_LEN + 1), steps=1, log_every=1)
except ValueError:
    sys.exit(0)
sys.exit("the step loaded in a process whose XLA computes otherwise")
"""
"""A fresh process whose trainer trains a step from the step at argv[1] (`QUIET_BUILD`'s), which its loader
must refuse as it loads it: a trainer handed a step loads it at its first step's compile, not as it is
made."""
QUIET_BUILD = """
import sys

from test_compile_ahead import BATCH, SEQ_LEN, VOCAB, fine_tune
from token_support import token_rows

from dew.data import Memory
from dew.training import Inspection, MeshSpec
from dew.nn.kernels.generation import compile_only_gpus
from dew.training.ahead import compile_ahead

generation, path, multiprocessors, memory = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
target = MeshSpec().build(compile_only_gpus(generation, 1, memory, multiprocessors=multiprocessors))
trainer = fine_tune(target, xla=True, split=2, inspection=Inspection(every=2), dtype="bfloat16")
batch = token_rows(VOCAB, BATCH, SEQ_LEN + 1)
validation = {"val": Memory(token_rows(VOCAB, 2 * BATCH, SEQ_LEN + 1, seed=1)).batch(BATCH)}
fit = {"steps": 6, "log_every": 5, "eval_every": 3, "validation": validation}
compile_ahead(trainer, batch, fit=fit).save(path)
"""
"""A step of two microbatches, sampled every second step, compiled ahead for the GPU argv[1] names, with
argv[3] multiprocessors and argv[4] bytes of memory, with the programs of its fit's evaluations, and saved
at argv[2], in a process that sees no GPU."""
QUIET = """
import sys

import jax
from test_compile_ahead import BATCH, SEQ_LEN, VOCAB, fine_tune
from token_support import token_rows

from dew.data import Memory
from dew.telemetry.compiles import Compiles
from dew.training import Inspection, MeshSpec
from dew.training.ahead import AheadOfTime

run = fine_tune(MeshSpec().build(jax.devices()[:1]), ahead=AheadOfTime.load(sys.argv[1]), xla=True, split=2,
                inspection=Inspection(every=2), dtype="bfloat16")
validation = {"val": Memory(token_rows(VOCAB, 2 * BATCH, SEQ_LEN + 1, seed=1)).batch(BATCH)}
with Compiles(None) as compiles:
    state = run.fit(token_rows(VOCAB, BATCH, SEQ_LEN + 1), steps=6, log_every=5, eval_every=3,
                    validation=validation)
assert int(state.step) == 6 and run.plan["narrow"]
compiled = [(program.name, program.where) for program in compiles.programs if program.kind == "compiled"]
assert not compiled, compiled
"""
"""A fresh process training six steps of two microbatches from the step at argv[1], sampled every second
step, two samples to a log, evaluated every third step, compiling nothing on its GPU: its bf16 step's plan
keeps its narrow copies, which the programs compiled with it make (row 4777)."""
COMPUTING = "--xla_gpu_enable_fast_min_max=true"
"""An XLA flag that changes what a program computes, which a process loading programs compiled without it is
refused for (`BuildTarget.held`)."""


ALIASED = """
import json
import sys

import jax

mode, cache, topology = sys.argv[1], sys.argv[2], sys.argv[3]
jax.config.update("jax_compilation_cache_dir", cache)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
read = []
jax.monitoring.register_event_listener(
    lambda event, **metadata: read.append(event) if event == "/jax/compilation_cache/cache_hits" else None)
import jax.numpy as jnp
from jax.experimental import topologies

from dew.training.ahead import BuildTarget, host_gpus


def program(values, weights):
    # A product XLA tunes on a GPU and leaves untuned where it has none: a 4096-square bf16 GEMM fusion.
    return jax.nn.relu(values @ weights) * 2 + 1


SHAPE = (4096, 4096)


if mode == "topology":
    # This host's GPU's own topology and generation, written for a process that sees no GPU.
    with open(topology, "wb") as file:
        file.write(topologies.get_attached_topology("cuda").serialize())
    print(json.dumps({"generation": BuildTarget.of(jax.devices()[:1]).generation}))
elif mode == "compile":
    # No GPU in this process (none visible, JAX_PLATFORMS=cpu): XLA compiles over the topology with none to
    # tune on.
    (compile_only,) = host_gpus(sys.argv[4], topology, 1)
    there = jax.sharding.SingleDeviceSharding(compile_only)
    placed = jax.ShapeDtypeStruct(SHAPE, jnp.bfloat16, sharding=there)
    jax.jit(program).lower(placed, placed).compile()
    print(json.dumps({"read": len(read)}))
else:
    values = jnp.ones(SHAPE, jnp.bfloat16)
    jax.block_until_ready(jax.jit(program)(values, values))
    print(json.dumps({"read": len(read)}))
"""
"""A fresh process, by its mode (argv[1]), over a persistent cache (argv[2]) and a topology file (argv[3]):
`topology` writes this host's GPU's topology and prints its generation; `compile` compiles one program over
that topology for the generation (argv[4]) with no GPU attached, outside Dew's compile point; `run` runs it
on the GPU. Each prints how many programs it read from the cache."""


def environment_of(tmp_path: Path) -> dict[str, str]:
    """This process's environment for a fresh one running a script under `tmp_path`, with this suite's
    modules importable."""
    tests = str(Path(__file__).resolve().parent)
    paths = os.pathsep.join([str(tmp_path), tests, os.environ.get("PYTHONPATH", "")])
    return {**os.environ, "PYTHONPATH": paths}


def computing_otherwise(environment: dict[str, str]) -> dict[str, str]:
    """`environment` with `COMPUTING` among its XLA flags."""
    return {**environment, "XLA_FLAGS": f"{environment.get('XLA_FLAGS', '')} {COMPUTING}".strip()}


def test_programs_built_with_no_gpu_run_on_this_gpu_with_nothing_compiled_bit_for_bit_as_compiled_here(
        tmp_path):
    own = BuildTarget.of(jax.devices()[:1])
    generation = own.generation
    # A target is the device's own, whatever generation this process assumes.
    with assumed("sm80" if generation != "sm80" else "sm90"):
        assert BuildTarget.of(jax.devices()[:1]) == own
    for name, text in (("build.py", BUILD), ("run.py", RUN), ("refused.py", REFUSED)):
        (tmp_path / name).write_text(text)
    environment = environment_of(tmp_path)
    # Built where no GPU is visible, as a CPU host builds them.
    # For this GPU: its generation and its own multiprocessors.
    build = [sys.executable, str(tmp_path / "build.py"), generation, str(tmp_path / "programs.dew-aot"),
             str(own.multiprocessors)]
    subprocess.run(build, env={**environment, "CUDA_VISIBLE_DEVICES": "", "JAX_PLATFORMS": "cpu"}, check=True)
    arrays = {name: [np.asarray(array) for array in drawn] for name, drawn in arguments(generation).items()}
    (tmp_path / "arguments.pkl").write_bytes(pickle.dumps(arrays))
    # Loaded and run in a fresh process whose compilers' dumps must stay empty; refused in one whose XLA
    # computes otherwise.
    run = [sys.executable, str(tmp_path / "run.py"), str(tmp_path / "programs.dew-aot"),
           str(tmp_path / "arguments.pkl"), str(tmp_path / "outputs.pkl")]
    subprocess.run(run, env=dumps(tmp_path / "dumps", environment), check=True)
    assert compiled_in(tmp_path / "dumps") == []
    subprocess.run([sys.executable, str(tmp_path / "refused.py"), str(tmp_path / "programs.dew-aot")],
                   env=computing_otherwise(environment), check=True)
    ahead = pickle.loads((tmp_path / "outputs.pkl").read_bytes())
    for name, (function, _) in programs(generation).items():
        # Under the facts the build used: this GPU's generation and its own multiprocessors.
        with assumed(built_target(generation, own.multiprocessors)):
            here = np.asarray(owned_program(function)(*arrays[name]))
        assert ahead[name].dtype == here.dtype and ahead[name].tobytes() == here.tobytes(), name


def test_a_run_compiled_ahead_for_this_gpu_compiles_nothing_on_it_and_is_refused_where_xla_computes_otherwise(
        tmp_path):
    """A run handed its step compiled ahead for this GPU where no GPU is visible, two microbatches a step and
    sampled every second step, trains six steps logged twice and evaluated twice in a fresh process
    compiling nothing on the GPU: its start, its window's allocation, its split step's whole, its counters,
    its latest sample, its log's copies, its logged tokens' sum and its evaluations' keys, loss, sums and
    reduction were compiled with the step (`AheadOfTime.starts`; row 4752), and its bf16 compute's narrow
    copies, which it makes, were too (row 4777). A fresh process whose XLA
    computes otherwise is refused that step by the trainer's loader: the one rule every loader holds (row
    3606)."""
    own = BuildTarget.of(jax.devices()[:1])
    for name, text in (("quiet_build.py", QUIET_BUILD), ("quiet.py", QUIET), ("refused.py", TRAIN_REFUSED)):
        (tmp_path / name).write_text(text)
    environment = environment_of(tmp_path)
    # Its memory stated as the GPU's allocator limit: a plan for no more than the GPU holds, which it loads.
    assert own.memory is not None
    build = [sys.executable, str(tmp_path / "quiet_build.py"), own.generation, str(tmp_path / "step.dew-aot"),
             str(own.multiprocessors), str(own.memory)]
    subprocess.run(build, env={**environment, "CUDA_VISIBLE_DEVICES": "", "JAX_PLATFORMS": "cpu"}, check=True)
    subprocess.run([sys.executable, str(tmp_path / "quiet.py"), str(tmp_path / "step.dew-aot")],
                   env=environment, check=True)
    subprocess.run([sys.executable, str(tmp_path / "refused.py"), str(tmp_path / "step.dew-aot")],
                   env=computing_otherwise(environment), check=True)


def test_a_deviceless_compile_stores_nothing_a_device_run_reads(tmp_path):
    """The contract between the persistent cache's key and the autotuner: a program compiled in a process that
    sees no GPU (none visible, `JAX_PLATFORMS=cpu`) over this host's own topology (`host_gpus`), as a user's
    compile-only process compiles one (a 4096-square bf16 GEMM, outside Dew's compile point, with no
    process settings handed down), is no program of the GPU's own: XLA had no device to tune it on. So the
    GPU's run of the same program on the same persistent cache reads nothing the deviceless compile wrote and
    compiles its own: an untuned program is never read under a tuned key (rows 4792 and 6046). JAX post87
    read it (UB-4792-AHEAD-TIME-gh200); since post90's key it does not, read 0 on GH200 on post94 and post95
    (MC-UNTUNEDKEY4-post94-f74e55896, MC-UNTUNEDKEY4-post95-16d207cc1), so a JAX that loses that key turns
    this red."""
    (tmp_path / "aliased.py").write_text(ALIASED)
    # None of this process's JAX settings or XLA flags handed down (a lane's name its own tuning and cache),
    # as a user's process runs; its client's memory settings kept (`XLA_PYTHON_CLIENT_*`: a packed job's
    # share of a GPU others share).
    environment = {name: value for name, value in environment_of(tmp_path).items()
                   if not name.startswith("JAX_") and name != "XLA_FLAGS"}
    cache, topology = str(tmp_path / "cache"), str(tmp_path / "topology.bin")

    def ran(mode: str, *arguments: str, **variables: str) -> dict:
        done = subprocess.run([sys.executable, str(tmp_path / "aliased.py"), mode, cache, topology,
                               *arguments], env={**environment, **variables}, capture_output=True, text=True,
                              check=True)
        return json.loads(done.stdout.splitlines()[-1])

    generation = ran("topology")["generation"]
    # The GPU hidden as well: a process that sees one of the topology's own compiles through it, tuned.
    assert ran("compile", generation, JAX_PLATFORMS="cpu", CUDA_VISIBLE_DEVICES="") == {"read": 0}
    assert ran("run") == {"read": 0}
