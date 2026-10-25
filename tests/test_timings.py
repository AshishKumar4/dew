"""The choice layer's timing log (dew.nn.kernels.timings): passive where DEW_TIMINGS is unset, one whole
line per row where it names a directory, appended, never interleaved, and a held lock a stated wait."""

import dataclasses
import json
import subprocess
import sys
from typing import ClassVar

import filelock
import jax
import jax.numpy as jnp
import pytest
import tokamax

from dew.nn.kernels import timings, tuning

ROW = {'launch': 'gmm', 'call': '{}', 'config': '{}'}


@pytest.mark.parametrize("value", [None, ""])
def test_unset_or_empty_writes_nothing(monkeypatch, tmp_path, value):
    if value is None:
        monkeypatch.delenv(timings.ENV, raising=False)
    else:
        monkeypatch.setenv(timings.ENV, value)
    monkeypatch.chdir(tmp_path)
    timings.record([ROW])
    assert timings.directory() is None
    assert list(tmp_path.iterdir()) == []


def test_rows_append_one_line_each_into_a_created_directory(monkeypatch, tmp_path):
    where = tmp_path / "made" / "here"
    monkeypatch.setenv(timings.ENV, str(where))
    timings.record([ROW, dict(ROW, launch='tgmm')])
    timings.record([dict(ROW, launch='dense')])
    lines = (where / "timings.jsonl").read_text().splitlines()
    assert [json.loads(line)['launch'] for line in lines] == ['gmm', 'tgmm', 'dense']


def test_an_observation_names_its_outcome_and_an_errors_class_alone():
    row = timings.observation('gmm', '{"rows": 8}', '{"block_m": 64}', device=jax.devices()[0],
                              method=timings.CUPTI_EVICTED, status='resource_exhausted',
                              error=MemoryError("a long message about memory"))
    assert row['status'] == 'resource_exhausted' and row['error'] == 'MemoryError'
    assert row['run'] == timings.RUN and row['schema'] == timings.SCHEMA
    assert row['samples_ms'] == [] and row['samples_total'] == 0 and row['estimate_s'] is None
    assert row['platform'] == jax.devices()[0].platform
    assert "long message" not in json.dumps(row)


def test_an_observation_bounds_its_texts_and_samples():
    """A text past MAX_TEXT is kept as its digest and length; samples past MAX_SAMPLES are counted."""
    call = 'x' * (timings.MAX_TEXT + 1)
    row = timings.observation('gmm', call, '{}', device=jax.devices()[0], method=timings.CUPTI_EVICTED,
                              status='timed', samples_ms=tuple(range(timings.MAX_SAMPLES + 5)))
    assert row['call'].startswith('sha256:') and row['call'].endswith(f':{len(call)}')
    assert len(row['samples_ms']) == timings.MAX_SAMPLES and row['samples_total'] == timings.MAX_SAMPLES + 5


WRITER = (
    "import sys\n"
    "import filelock\n"
    "from dew.nn.kernels import timings\n"
    "try:\n"
    "    for i in range(int(sys.argv[2])):\n"
    "        row = {'launch': sys.argv[1], 'i': i, 'pad': 'x' * 20000}\n"
    "        timings.record([row] * 3, wait=float(sys.argv[3]))\n"
    "except filelock.Timeout:\n"
    "    sys.exit(3)\n"
)


def _writer(tmp_path, name: str, batches: int, wait: float):
    environment = {timings.ENV: str(tmp_path), 'JAX_PLATFORMS': 'cpu', 'PATH': '/usr/bin:/bin',
                   'PYTHONPATH': ':'.join(sys.path)}
    return subprocess.Popen([sys.executable, '-c', WRITER, name, str(batches), str(wait)], env=environment)


def test_another_process_cannot_append_while_the_lock_is_held(tmp_path):
    """A writer in another process times out on a lock this one holds, writing nothing, and appends
    once it is released."""
    with filelock.FileLock(tmp_path / "timings.jsonl.lock"):
        assert _writer(tmp_path, 'a', 1, 0.5).wait(timeout=600) == 3
        assert not (tmp_path / "timings.jsonl").exists()
    assert _writer(tmp_path, 'a', 1, 60).wait(timeout=600) == 0
    assert len((tmp_path / "timings.jsonl").read_text().splitlines()) == 3


def test_concurrent_writers_never_interleave_their_batches(tmp_path):
    """Two processes appending long batches at once: every line whole, every row there, each batch's
    three rows together."""
    writers = [_writer(tmp_path, name, 40, 60) for name in 'ab']
    assert [writer.wait(timeout=600) for writer in writers] == [0, 0]
    rows = [json.loads(line) for line in (tmp_path / "timings.jsonl").read_text().splitlines()]
    keys = [(row['launch'], row['i']) for row in rows]
    assert sorted(keys) == sorted((name, i) for name in 'ab' for i in range(40) for _ in range(3))
    assert all(keys[start] == keys[start + 1] == keys[start + 2] for start in range(0, len(keys), 3))


def test_a_held_lock_raises_after_its_stated_wait(monkeypatch, tmp_path):
    monkeypatch.setenv(timings.ENV, str(tmp_path))
    with filelock.FileLock(tmp_path / "timings.jsonl.lock"), pytest.raises(filelock.Timeout):
        timings.record([ROW], wait=0.1)


def _bound():
    """A grouped matmul's forward launch (a template with a costed record) and an RMS norm's (a kernel op
    with none), each bound to a small call as the tuner keys it."""
    from test_tuning import operands, step

    from dew.nn.kernels import grouped_matmul, norm_swiglu, tuning
    shape = jax.ShapeDtypeStruct
    gmm = grouped_matmul.Gmm().bind(shape((256, 128), jnp.bfloat16), shape((2, 128, 128), jnp.bfloat16),
                                    shape((2,), jnp.int32), (), (), out_dtype="bfloat16", trans_rhs=False,
                                    interpret=False)
    with tuning.launches():
        found = tokamax.autotuning.get_bound_args(step(), *operands(15))
    rms = next(ba for ba in found if isinstance(ba.op, norm_swiglu.RmsForward))
    return gmm, rms


def test_the_tuner_logs_each_measured_config_with_its_features(monkeypatch, tmp_path):
    """Each config the tuner timed is one row: a template's with its features and the estimate they give,
    a kernel op without a costed record with none, and one the device could not hold with its error's
    class, its samples empty; with DEW_TIMINGS unset nothing is written."""
    from dew.nn.kernels import generation, tuning

    gmm, rms = _bound()
    data = tokamax.BenchmarkData(compile_time_ms=12.0, lower_time_ms=3.0, metadata={},
                                 evaluation_times_ms=(1.0, 1.5, 1.25))
    device = jax.devices()[0]
    with generation.assumed("sm80"):
        tile, other = gmm.op.fitting(gmm, generation.gpu_target())[:2]
        monkeypatch.delenv(timings.ENV, raising=False)
        tuning.observed(gmm.op, gmm, {tile: data}, device)
        assert not list(tmp_path.iterdir())
        monkeypatch.setenv(timings.ENV, str(tmp_path))
        tuning.observed(gmm.op, gmm, {tile: data, other: MemoryError("held")}, device)
        tuning.observed(rms.op, rms, {next(iter(rms.autotuning_configs)): data}, device)
    rows = [json.loads(line) for line in (tmp_path / "timings.jsonl").read_text().splitlines()]
    timed, exhausted, norm = rows
    assert timed['status'] == 'timed' and timed['samples_ms'] == [1.0, 1.5, 1.25]
    assert timed['compile_ms'] == 12.0 and timed['features']['legacy']['programs'] > 0
    assert timed['estimate_s'] is not None and timed['launch'] == 'Gmm'
    assert exhausted['status'] == 'resource_exhausted' and exhausted['error'] == 'MemoryError'
    assert exhausted['samples_ms'] == [] and exhausted['compile_ms'] is None
    assert norm['features'] is None and norm['estimate_s'] is None and norm['launch'] == 'RmsForward'
    assert norm['features_status'] == 'unsupported' and timed['features_status'] == 'stated'



@dataclasses.dataclass(frozen=True)
class Factor:
    factor: int


@dataclasses.dataclass(frozen=True)
class Scaled(tuning.KernelOp):
    """A launch that runs anywhere: its input times its config's factor."""
    config_cls: ClassVar[type[Factor]] = Factor

    def _fwd(self, x, *, config: Factor, return_residuals: bool = False):
        return x * config.factor, None

    def _get_heuristics_config(self, ba) -> Factor:
        return Factor(1)

    def _get_autotuning_configs(self, ba) -> set[Factor]:
        return {Factor(1), Factor(2), Factor(3)}


class FailingProfiler:
    """A stand-in for Mosaic's profiler whose CUPTI times every run as one kernel of 1 ms (the flush's
    first, named apart), until run `fails`, which raises `error`."""

    def __init__(self, fails: int, error: Exception):
        self.runs, self.fails, self.error = 0, fails, error
        profiler = self

        class Cupti:
            def __init__(self, finalize: bool = True):
                del finalize

            def measure(self, executable, aggregate: bool = False):
                def timed(*args):
                    profiler.runs += 1
                    if profiler.runs == profiler.fails:
                        raise profiler.error
                    name = 'flush' if profiler.runs == 1 else 'scaled'
                    return executable(*args), [(name, 1.0)]
                return timed

        self.Cupti = Cupti


@pytest.mark.parametrize(("error", "raises"), [(RuntimeError("the device faulted"), True),
                                               (RuntimeError("another subscriber holds CUPTI"), False)])
def test_a_measurement_that_fails_part_way_logs_the_configs_it_timed(monkeypatch, tmp_path, error, raises):
    """A tune whose second config fails, by a fault that propagates or by another subscriber taking
    CUPTI, logs the config it timed first, with its samples, and nothing for the configs it did not."""
    monkeypatch.setenv(timings.ENV, str(tmp_path))
    ba = Scaled().bind(jax.ShapeDtypeStruct((8,), jnp.float32))
    # The flush's run, the first config's repeats, then the second config's first run fails.
    profiler = FailingProfiler(fails=2 + tuning.REPEATS, error=error)
    if raises:
        with pytest.raises(RuntimeError):
            tuning._measured(ba.op, ba, profiler)
    else:
        measured = tuning._measured(ba.op, ba, profiler)
        assert all(isinstance(timing, RuntimeError) for timing in measured.values())
    [row] = [json.loads(line) for line in (tmp_path / "timings.jsonl").read_text().splitlines()]
    assert row['status'] == 'timed' and row['samples_ms'] == [1.0] * tuning.REPEATS
    assert row['launch'] == 'Scaled' and row['features_status'] == 'unsupported'
