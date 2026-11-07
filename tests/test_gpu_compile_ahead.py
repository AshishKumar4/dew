"""Programs compiled ahead for this GPU's generation by a process that sees no GPU load in a fresh process on
the GPU and run there with nothing compiled, bit for bit as the same programs compiled here through Dew's
compile point for the generation's facts, as the build compiles them (ledger 2716: a GPU node spends its time
on its device's work alone; 2026-11-02, an H100 ran a CPU host's sm90 build of a dot and the template
attention's forward so)."""

import os
import pickle
import subprocess
import sys
from pathlib import Path

import jax
import numpy as np
import pytest
from ahead_programs import arguments, programs
from test_compile_ahead import BATCH, SEQ_LEN, VOCAB, fine_tune
from token_support import token_rows

from dew.nn.kernels.generation import assumed, built_target
from dew.nn.kernels.options import owned_program
from dew.training import MeshSpec
from dew.training.ahead import BuildTarget, compile_ahead, compile_only_gpus, compiled_in, dumps

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
TRAIN = """
import sys

import jax
from test_compile_ahead import BATCH, SEQ_LEN, VOCAB, fine_tune
from token_support import token_rows

from dew.training import MeshSpec
from dew.training.ahead import AheadOfTime

refused = sys.argv[2] == "refused"
try:
    run = fine_tune(MeshSpec().build(jax.devices()[:1]), ahead=AheadOfTime.load(sys.argv[1]), xla=True)
    assert int(run.fit(token_rows(VOCAB, BATCH, SEQ_LEN + 1), steps=1, log_every=1).step) == 1
except ValueError:
    if not refused:
        raise
    sys.exit(0)
if refused:
    sys.exit("the step loaded in a process whose XLA computes otherwise")
"""
"""A fresh process training from the step at argv[1], or, argv[2] "refused", one its loader must refuse."""
QUIET_BUILD = """
import sys

from test_compile_ahead import BATCH, SEQ_LEN, VOCAB, fine_tune
from token_support import token_rows

from dew.training import Inspection, MeshSpec
from dew.training.ahead import compile_ahead, compile_only_gpus

generation, path, multiprocessors = sys.argv[1], sys.argv[2], int(sys.argv[3])
target = MeshSpec().build(compile_only_gpus(generation, 1, multiprocessors=multiprocessors))
trainer = fine_tune(target, xla=True, split=2, inspection=Inspection(every=2))
compile_ahead(trainer, token_rows(VOCAB, BATCH, SEQ_LEN + 1)).save(path)
"""
"""A step of two microbatches, sampled every second step, compiled ahead for the GPU argv[1] names, with
argv[3] multiprocessors, and saved at argv[2], in a process that sees no GPU."""
QUIET = """
import sys

import jax
from test_compile_ahead import BATCH, SEQ_LEN, VOCAB, fine_tune
from token_support import token_rows

from dew.telemetry.compiles import Compiles
from dew.training import Inspection, MeshSpec
from dew.training.ahead import AheadOfTime

run = fine_tune(MeshSpec().build(jax.devices()[:1]), ahead=AheadOfTime.load(sys.argv[1]), xla=True, split=2,
                inspection=Inspection(every=2))
with Compiles(None) as compiles:
    assert int(run.fit(token_rows(VOCAB, BATCH, SEQ_LEN + 1), steps=6, log_every=5).step) == 6
compiled = [(program.name, program.where) for program in compiles.programs if program.kind == "compiled"]
assert not compiled, compiled
"""
"""A fresh process training six steps of two microbatches from the step at argv[1], sampled every second
step, two samples to a log, compiling nothing on its GPU."""
COMPUTING = "--xla_gpu_enable_fast_min_max=true"
"""An XLA flag that changes what a program computes, which a process loading programs compiled without it is
refused for (`BuildTarget.held`)."""


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


def test_a_run_compiled_ahead_for_this_gpu_compiles_nothing_on_it_through_its_logged_steps(tmp_path):
    """A run handed its step compiled ahead for this GPU where no GPU is visible, two microbatches a step and
    sampled every second step, trains six steps logged twice in a fresh process compiling nothing on the
    GPU: its start, its window's allocation, its split step's whole, its counters, its latest sample, its
    log's copies and its logged tokens' sum were compiled with the step (`AheadOfTime.starts`; row 4752)."""
    own = BuildTarget.of(jax.devices()[:1])
    for name, text in (("quiet_build.py", QUIET_BUILD), ("quiet.py", QUIET)):
        (tmp_path / name).write_text(text)
    environment = environment_of(tmp_path)
    build = [sys.executable, str(tmp_path / "quiet_build.py"), own.generation, str(tmp_path / "step.dew-aot"),
             str(own.multiprocessors)]
    subprocess.run(build, env={**environment, "CUDA_VISIBLE_DEVICES": "", "JAX_PLATFORMS": "cpu"}, check=True)
    subprocess.run([sys.executable, str(tmp_path / "quiet.py"), str(tmp_path / "step.dew-aot")],
                   env=environment, check=True)


def test_a_step_compiled_ahead_for_this_gpu_trains_here_and_is_refused_where_xla_computes_otherwise(
        tmp_path):
    """A trainer's step compiled ahead for this GPU's generation and saved trains from it in a fresh process
    of this environment, and a fresh process whose XLA computes otherwise is refused by the trainer's loader,
    naming the flag: the one rule every loader holds (row 3606)."""
    own = BuildTarget.of(jax.devices()[:1])
    batch = token_rows(VOCAB, BATCH, SEQ_LEN + 1)
    target = MeshSpec().build(compile_only_gpus(own.generation, 1, multiprocessors=own.multiprocessors))
    compile_ahead(fine_tune(target, xla=True), batch).save(tmp_path / "step.dew-aot")
    (tmp_path / "train.py").write_text(TRAIN)
    environment = environment_of(tmp_path)
    train = [sys.executable, str(tmp_path / "train.py"), str(tmp_path / "step.dew-aot")]
    subprocess.run([*train, "trained"], env=environment, check=True)
    subprocess.run([*train, "refused"], env=computing_otherwise(environment), check=True)
