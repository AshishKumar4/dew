"""tools/peak_memory.py reads a program's device high-water in a process that compiled nothing: a program
whose every compile the cache keeps is measured, and one with a program the cache keeps no entry for, which
compiles on every run, is refused, as a compile and its autotuning would join the high-water."""

import sys

import jax
import pytest
from tool_support import load

peak_memory = load("peak_memory")

PROGRAM = """import jax, jax.numpy as jnp
{setup}
x = jnp.ones((64, 64))
jax.jit(lambda a: a @ a + 1)(x).block_until_ready()
"""


def test_a_program_the_cache_holds_whole_is_measured_one_high_water_a_device(tmp_path):
    program = tmp_path / "program.py"
    program.write_text(PROGRAM.format(setup=""))
    peaks = peak_memory.measured([str(program)])
    assert len(peaks) == len(jax.local_devices())


def test_a_program_that_compiles_in_its_measured_run_is_refused(tmp_path):
    # Its program needs a compile on every run: the process keeps nothing in the cache it was given.
    program = tmp_path / "program.py"
    program.write_text(PROGRAM.format(setup="jax.config.update('jax_enable_compilation_cache', False)"))
    with pytest.raises(peak_memory.CompiledWhileMeasured):
        peak_memory.measured([str(program)])


def test_a_program_that_fails_is_refused_with_its_error(tmp_path):
    program = tmp_path / "program.py"
    program.write_text("raise SystemExit(3)\n")
    with pytest.raises(peak_memory.subprocess.CalledProcessError) as raised:
        peak_memory.measured([str(program)])
    assert raised.value.returncode == 3 and sys.executable in raised.value.cmd


def test_a_program_imports_the_modules_beside_it_as_it_would_alone(tmp_path):
    (tmp_path / "beside.py").write_text("VALUE = 1\n")
    program = tmp_path / "program.py"
    program.write_text("import beside\n" + PROGRAM.format(setup="assert beside.VALUE == 1"))
    assert len(peak_memory.measured([str(program)])) == len(jax.local_devices())
