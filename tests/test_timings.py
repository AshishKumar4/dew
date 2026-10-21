"""The choice layer's timing log (dew.nn.kernels.timings): passive where DEW_TIMINGS is unset, one whole
line per row where it names a directory, appended, never interleaved, and a held lock a stated wait."""

import json
import subprocess
import sys

import filelock
import pytest

from dew.nn.kernels import timings

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
    row = timings.observation('gmm', '{"rows": 8}', '{"block_m": 64}', method=timings.CUPTI_EVICTED,
                              status='resource_exhausted', error=MemoryError("a long message about memory"))
    assert row['status'] == 'resource_exhausted' and row['error'] == 'MemoryError'
    assert row['run'] == timings.RUN and row['schema'] == timings.SCHEMA
    assert row['samples_ms'] == [] and row['estimate_s'] is None
    assert "long message" not in json.dumps(row)


def test_concurrent_writers_never_interleave_their_lines(tmp_path):
    """Two processes appending long rows at once: every line whole, every row there."""
    script = (
        "import sys\n"
        "from dew.nn.kernels import timings\n"
        "for i in range(40):\n"
        "    timings.record([{'launch': sys.argv[1], 'i': i, 'pad': 'x' * 20000}] * 3)\n"
    )
    environment = {timings.ENV: str(tmp_path), 'JAX_PLATFORMS': 'cpu', 'PATH': '/usr/bin:/bin'}
    environment['PYTHONPATH'] = ':'.join(sys.path)
    writers = [subprocess.Popen([sys.executable, '-c', script, name], env=environment) for name in 'ab']
    assert [writer.wait(timeout=600) for writer in writers] == [0, 0]
    rows = [json.loads(line) for line in (tmp_path / "timings.jsonl").read_text().splitlines()]
    assert sorted((row['launch'], row['i']) for row in rows) == sorted(
        (name, i) for name in 'ab' for i in range(40) for _ in range(3))


def test_a_held_lock_raises_after_its_stated_wait(monkeypatch, tmp_path):
    monkeypatch.setenv(timings.ENV, str(tmp_path))
    monkeypatch.setattr(timings, 'LOCK_SECONDS', 0.1)
    with filelock.FileLock(tmp_path / "timings.jsonl.lock"), pytest.raises(filelock.Timeout):
        timings.record([ROW])
