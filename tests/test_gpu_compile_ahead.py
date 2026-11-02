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

from dew.nn.kernels.generation import assumed
from dew.nn.kernels.options import owned_program
from dew.training.ahead import BuildTarget, compiled_in, dumps

pytestmark = pytest.mark.gpu(reason="a program compiled ahead runs on the GPU it was built for")

BUILD = """
import sys

from dew.training.ahead import compile_programs
from ahead_programs import programs

compile_programs(programs(sys.argv[1]), sys.argv[1]).save(sys.argv[2])
"""
RUN = """
import pickle
import sys

import jax
import numpy as np

from dew.telemetry.instrumentation import executable_only
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


def test_programs_built_with_no_gpu_run_on_this_gpu_with_nothing_compiled_bit_for_bit_as_compiled_here(
        tmp_path):
    own = BuildTarget.of(jax.devices()[:1])
    generation = own.generation
    # A target is the device's own, whatever generation this process assumes.
    with assumed("sm80" if generation != "sm80" else "sm90"):
        assert BuildTarget.of(jax.devices()[:1]) == own
    for name, text in (("build.py", BUILD), ("run.py", RUN)):
        (tmp_path / name).write_text(text)
    tests = str(Path(__file__).resolve().parent)
    paths = os.pathsep.join([str(tmp_path), tests, os.environ.get("PYTHONPATH", "")])
    environment = {**os.environ, "PYTHONPATH": paths}
    # Built where no GPU is visible, as a CPU host builds them.
    build = [sys.executable, str(tmp_path / "build.py"), generation, str(tmp_path / "programs.dew-aot")]
    subprocess.run(build, env={**environment, "CUDA_VISIBLE_DEVICES": "", "JAX_PLATFORMS": "cpu"}, check=True)
    arrays = {name: [np.asarray(array) for array in drawn] for name, drawn in arguments(generation).items()}
    (tmp_path / "arguments.pkl").write_bytes(pickle.dumps(arrays))
    # Loaded and run in a fresh process whose compilers' dumps must stay empty.
    subprocess.run([sys.executable, str(tmp_path / "run.py"), str(tmp_path / "programs.dew-aot"),
                    str(tmp_path / "arguments.pkl"), str(tmp_path / "outputs.pkl")],
                   env=dumps(tmp_path / "dumps", environment), check=True)
    assert compiled_in(tmp_path / "dumps") == []
    ahead = pickle.loads((tmp_path / "outputs.pkl").read_bytes())
    for name, (function, _) in programs(generation).items():
        with assumed(generation):
            here = np.asarray(owned_program(function)(*arrays[name]))
        assert ahead[name].dtype == here.dtype and ahead[name].tobytes() == here.tobytes(), name
