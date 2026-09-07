"""Reporting persists real results and releases owned sinks on failures."""

import json
import math

import jax
import numpy as np
import optax
import pytest
from PIL import Image

from dew.artifacts import ImageGrid, TextSamples, VideoGrid, Representations, TokenScores
from dew.data import Dataset
from dew.telemetry.records import RunRecord, FitEnded, json_value
from dew.training import LocalTracker, Trackers, Trainer, Checkpoints
from test_instrumentation import Regression, batches


def records(path):
    return [json.loads(line) for line in (path / 'records.jsonl').read_text().splitlines()]


def test_nonfinite_metrics_preserve_meaning_without_invalid_json(tmp_path):
    with LocalTracker(tmp_path) as tracker:
        tracker.log({'psnr': math.inf, 'diverged': math.nan, 'bound': -math.inf, 'loss': 1.5}, 2)
    row = json.loads((tmp_path / 'scalars.jsonl').read_text(),
                     parse_constant=lambda x: pytest.fail(f'invalid JSON: {x}'))
    assert row['scalars'] == {'psnr': '+Inf', 'diverged': 'NaN', 'bound': '-Inf', 'loss': 1.5}
    assert float(row['scalars']['psnr']) == math.inf
    with pytest.raises(RuntimeError, match='closed'):
        tracker.log({'loss': 1.}, 3)


def test_journals_do_not_touch_recipe_config_and_preserve_metadata(tmp_path):
    config = tmp_path / 'run.json'
    config.write_text('{"model":"original"}')
    with LocalTracker(tmp_path) as tracker:
        tracker.artifact(RunRecord('example', {'model': 'affine'}, {}, 2, {'jax': jax.__version__}), 0)
    assert config.read_text() == '{"model":"original"}'
    assert records(tmp_path)[0]['value']['config'] == {'model': 'affine'}
    with pytest.raises(TypeError):
        json_value({'device-array': np.ones(2)})


def test_all_builtin_previews_have_local_representations(tmp_path):
    with LocalTracker(tmp_path) as tracker:
        tracker.artifact(ImageGrid(np.zeros((1, 8, 8, 3)), ('image',)), 1)
        tracker.artifact(VideoGrid(np.zeros((1, 2, 8, 8, 3)), ('clip',)), 1)
        tracker.artifact(TextSamples(np.array([[1, 2]]), texts=('text',)), 1)
        tracker.artifact(Representations(np.ones((2, 3)), np.array([0, 1])), 1)
        tracker.artifact(TokenScores(np.ones((2, 3)), np.ones((2, 3))), 1)
    entries = records(tmp_path)
    assert {e['type'] for e in entries} == {
        'ImageGrid', 'VideoGrid', 'TextSamples', 'Representations', 'TokenScores'}
    image = Image.open(next(tmp_path.glob('*.png')))
    assert image.size == (8, 8) and np.asarray(image)[0, 0].tolist() == [127, 127, 127]
    payloads = [json.loads(p.read_text()) for p in tmp_path.glob('*.json')]
    assert {'prompt': '', 'texts': ['text'], 'tokens': [[1, 2]]} in payloads


def test_fanout_continues_to_local_sink_and_context_preserves_primary(tmp_path):
    primary = ValueError('bad objective')

    class Broken:
        def log(self, scalars, step):
            raise OSError('sink offline')

        def artifact(self, value, step):
            raise OSError('sink offline')

        def close(self):
            raise OSError('close offline')

    local = LocalTracker(tmp_path)
    with pytest.raises(ValueError) as raised:
        with Trackers(Broken(), local) as tracker:
            with pytest.raises(OSError):
                tracker.log({'loss': 3.}, 1)
            raise primary
    assert raised.value is primary
    assert 'close offline' in '\n'.join(primary.__notes__)
    assert json.loads((tmp_path / 'scalars.jsonl').read_text())['scalars'] == {'loss': 3.}
    with pytest.raises(RuntimeError):
        local.log({}, 2)


def test_fit_records_requests_not_durability_and_leaves_tracker_borrowed(tmp_path):
    local = LocalTracker(tmp_path / 'tracking')
    trainer = Trainer(Regression(), optax.sgd(0.01), key=jax.random.key(0), tracker=local,
                      checkpoints=Checkpoints(str(tmp_path / 'checkpoint')))
    state = trainer.fit(Dataset(batches, None, None, 8), steps=2, log_every=1)
    local.log({'after_fit': 1.}, 2)
    local.close()
    entries = records(tmp_path / 'tracking')
    assert int(state.step) == 2
    assert [e['type'] for e in entries] == ['FitStarted', 'CheckpointRequested', 'FitEnded']
    assert entries[-1]['value']['status'] == 'completed'


def test_failure_is_recorded_without_replacing_the_original(tmp_path):
    error = RuntimeError('rollout broke')

    def rollout(state, batch, key):
        raise error

    with LocalTracker(tmp_path) as local:
        trainer = Trainer(Regression(), optax.sgd(.01), key=jax.random.key(0),
                          tracker=local, rollout=rollout)
        with pytest.raises(RuntimeError) as raised:
            trainer.fit(Dataset(batches, None, None, 8), steps=2)
        assert raised.value is error
    outcome = records(tmp_path)[-1]['value']
    assert outcome['status'] == 'failed' and 'rollout broke' in outcome['traceback']


def test_plotting_is_explicit_and_infinity_remains_in_journal(tmp_path):
    pytest.importorskip('matplotlib')
    with LocalTracker(tmp_path, plots=True) as tracker:
        tracker.log({'psnr': 10.}, 1)
        tracker.log({'psnr': math.inf}, 2)
        assert not list(tmp_path.glob('*.png'))
    assert Image.open(tmp_path / 'metric-0.png').size[0] > 0
    assert '+Inf' in (tmp_path / 'scalars.jsonl').read_text()
