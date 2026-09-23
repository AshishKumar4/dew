"""Reporting persists real results and releases owned sinks on failures."""

import io
import json
import math

import jax
import numpy as np
import optax
import pytest
from PIL import Image
from test_instrumentation import Regression, batches

from dew.artifacts import ImageGrid, Representations, TextSamples, TokenScores, VideoGrid
from dew.data import Dataset
from dew.telemetry.records import RunRecord, json_value
from dew.training import Checkpoints, LocalTracker, Trackers, Trainer


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
    assert image.size == (8, 8) and np.asarray(image)[0, 0].tolist() == [128, 128, 128]
    payloads = [json.loads(p.read_text()) for p in tmp_path.glob('*.json')]
    assert {'prompt': '', 'texts': ['text'], 'tokens': [[1, 2]]} in payloads


def test_an_image_preview_holds_the_pixels_the_image_metrics_score(tmp_path):
    """Every [-1, 1] value lands on its nearest level and out-of-range
    samples clip: the bytes a tracker writes are the bytes `clip_score` and
    `fid` read, not a level darker where truncation would drop the fraction.
    Zero is 127.5 exactly, which rounds half to even (the 128 above)."""
    levels = np.array([0, 1, 2, 127, 128, 200, 254, 255, 0, 255], np.float32)
    offsets = np.array([0.0, 0.4, -0.4, 0.45, 0.3, 0.45, -0.45, -0.3, 0.0, 0.0], np.float32)
    values = (levels + offsets) / 127.5 - 1.0
    values[-2:] = (-1.5, 1.5)
    images = np.broadcast_to(values[None, None, :, None], (1, 1, 10, 3)).astype(np.float32)
    with LocalTracker(tmp_path) as tracker:
        tracker.artifact(ImageGrid(images, ('levels',)), 1)
    written = np.asarray(Image.open(next(tmp_path.glob('*.png'))))[0, :, 0]
    assert written.tolist() == levels.astype(int).tolist()


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
    state = trainer.fit(Dataset(lambda partition: batches(), None, None, 8), steps=2, log_every=1)
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
            trainer.fit(Dataset(lambda partition: batches(), None, None, 8), steps=2)
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


@pytest.mark.parametrize('preview', [False, True])
def test_local_scalar_sink_does_not_enable_preview_computation(tmp_path, preview):
    generated = []

    class Display(Regression):
        def preview(self, params, batch, step, *, scored=None):
            generated.append(int(step.step))
            return ImageGrid(np.zeros((1, 8, 8, 3)))

    with LocalTracker(tmp_path) as sink:
        trainer = Trainer(Display(), optax.sgd(.01), key=jax.random.key(0), tracker=sink)
        data = Dataset(lambda partition: batches(), lambda partition: iter([next(batches())]), None, 8)
        if preview:
            trainer.fit(data, steps=2, eval_every=1, preview=True)
        else:
            # With preview off and no metric, the scheduled pass has no
            # consumer and is refused before anything is generated.
            with pytest.raises(ValueError, match="nothing consumes"):
                trainer.fit(data, steps=2, eval_every=1, preview=False)
    assert generated == ([1, 2] if preview else [])
    assert len(list(tmp_path.glob('*.png'))) == (2 if preview else 0)


@pytest.mark.parametrize('body_failure', [False, True])
def test_close_failure_reaches_later_wandb_offline_outcome(tmp_path, monkeypatch, request, body_failure):
    import functools

    import wandb
    from wandb.proto import wandb_internal_pb2
    from wandb.sdk.internal.datastore import DataStore

    from dew.training import WandbTracker
    request.addfinalizer(wandb.teardown)
    monkeypatch.setenv('WANDB_DIR', str(tmp_path))
    monkeypatch.setenv('WANDB_MODE', 'offline')
    monkeypatch.setenv('WANDB_SILENT', 'true')
    # Explicit SDK directory avoids its process-wide cached environment from
    # sending this run's transport file to a preceding test's temporary path.
    monkeypatch.setattr(wandb, 'init', functools.partial(wandb.init, dir=str(tmp_path)))
    original = ValueError('training failed')
    cleanup = OSError('local journal flush failed')

    class FailedLocal(LocalTracker):
        def close(self):
            super().close()
            raise cleanup

    local = FailedLocal(tmp_path / 'local')
    later_local = LocalTracker(tmp_path / 'later-local')
    with pytest.raises((ValueError, OSError)) as caught:
        with Trackers(local, WandbTracker('dew-close-proof', offline=True), later_local) as sinks:
            sinks.log({'train/loss': 1.}, 1)
            if body_failure:
                raise original
    assert caught.value is (original if body_failure else cleanup)
    if body_failure:
        assert any('local journal flush failed' in note for note in original.__notes__)
    with pytest.raises(RuntimeError, match='closed'):
        later_local.log({}, 2)

    # Read the actual offline transport record a subsequent W&B sync would
    # upload. This checks the persisted run outcome, not a mock finish echo.
    journal = next((tmp_path / 'wandb').glob('offline-run-*/*.wandb'))
    store = DataStore()
    store.open_for_scan(str(journal))
    exit_codes = []
    try:
        while (payload := store.scan_data()) is not None:
            record = wandb_internal_pb2.Record()
            record.ParseFromString(payload)
            if record.HasField('exit'):
                exit_codes.append(record.exit.exit_code)
    finally:
        store.close()
    assert exit_codes == [1]


def fit(tracker, objective=None):
    """Two logged steps of the small regression through `tracker`."""
    trainer = Trainer(objective or Regression(), optax.sgd(0.01), key=jax.random.key(0),
                      tracker=tracker)
    return trainer.fit(Dataset(lambda partition: batches(), None, None, 8), steps=2, log_every=1)


def mlflow_store(tmp_path, monkeypatch):
    """A local MLflow store under `tmp_path`. MLflow 3.16 gates its file
    backend behind this switch, and a file store keeps the run's artifacts in
    the temporary directory instead of a cwd-relative artifact root."""
    pytest.importorskip('mlflow')
    monkeypatch.setenv('MLFLOW_ALLOW_FILE_STORE', 'true')
    return (tmp_path / 'mlruns').as_uri()


def test_mlflow_run_holds_the_fit_read_back_with_mlflows_client(tmp_path, monkeypatch):
    store = mlflow_store(tmp_path, monkeypatch)
    import mlflow.artifacts

    from dew.training import MLflowTracker

    with MLflowTracker('dew-reporting', 'tiny-fit', uri=store) as tracker:
        fit(tracker)
        tracker.artifact(ImageGrid(np.zeros((1, 8, 8, 3)), ('a caption',)), 2)
        client, run = tracker.run

    assert [(point.step, math.isfinite(point.value))
            for point in client.get_metric_history(run, 'train/loss')] == [(1, True), (2, True)]
    assert client.get_run(run).info.status == 'FINISHED'
    assert ({file.path for file in client.list_artifacts(run, 'records')}
            == {'records/FitStarted-0.json', 'records/FitEnded-2.json'})
    artifacts = client.get_run(run).info.artifact_uri
    outcome = mlflow.artifacts.load_dict(f'{artifacts}/records/FitEnded-2.json')
    assert outcome['step'] == 2 and outcome['value']['status'] == 'completed'
    assert ({file.path for file in client.list_artifacts(run, 'previews/step-2')}
            == {'previews/step-2/preview-2-0.png', 'previews/step-2/preview-2.json'})
    drawn = mlflow.artifacts.download_artifacts(f'{artifacts}/previews/step-2/preview-2-0.png')
    assert Image.open(drawn).size == (8, 8)


def test_mlflow_marks_the_run_of_a_failed_fit_failed(tmp_path, monkeypatch):
    store = mlflow_store(tmp_path, monkeypatch)
    from dew.training import MLflowTracker

    class Broken(Regression):
        def loss(self, params, batch, step):
            raise ValueError('objective broke')

    with MLflowTracker('dew-reporting', 'failed-fit', uri=store) as tracker:
        with pytest.raises(ValueError, match='objective broke'):
            fit(tracker, Broken())
        client, run = tracker.run
    assert client.get_run(run).info.status == 'FAILED'


def test_tensorboard_events_hold_the_fit_read_back_with_the_accumulator(tmp_path):
    pytest.importorskip('tensorboard')
    from tensorboard.backend.event_processing.event_accumulator import (
        IMAGES,
        SCALARS,
        TENSORS,
        EventAccumulator,
    )

    from dew.training import TensorBoardTracker

    with TensorBoardTracker(tmp_path / 'events') as tracker:
        fit(tracker)
        tracker.artifact(ImageGrid(np.zeros((1, 8, 8, 3)), ('a caption',)), 2)
        tracker.artifact(Representations(np.zeros((4, 3)), np.zeros((4,))), 2)

    reader = EventAccumulator(str(tmp_path / 'events'),
                              size_guidance={SCALARS: 0, IMAGES: 0, TENSORS: 0})
    reader.Reload()
    assert [(point.step, math.isfinite(point.value))
            for point in reader.Scalars('train/loss')] == [(1, True), (2, True)]
    outcome = json.loads(reader.Tensors('reporting/FitEnded')[0].tensor_proto.string_val[0])
    assert outcome['status'] == 'completed'
    assert json.loads(reader.Tensors('reporting/FitStarted')[0]
                      .tensor_proto.string_val[0])['target_steps'] == 2
    drawn = reader.Images('val/samples/0')
    assert [(image.step, Image.open(io.BytesIO(image.encoded_image_string)).size)
            for image in drawn] == [(2, (8, 8))]
    assert json.loads(reader.Tensors('val/samples/captions')[0]
                      .tensor_proto.string_val[0]) == ['a caption']
    assert reader.Histograms('val/representation_std')[0].histogram_value.num == 3


def test_the_previews_tensorboard_can_show_render_and_the_rest_raises(tmp_path):
    pytest.importorskip('tensorboard')
    from tensorboard.backend.event_processing.event_accumulator import IMAGES, TENSORS, EventAccumulator

    from dew.training import TensorBoardTracker

    with TensorBoardTracker(tmp_path / 'events') as tracker:
        # Distinct frames: a GIF drops one that does not differ from the last.
        clip = np.stack([np.full((8, 8, 3), -1.0), np.full((8, 8, 3), 1.0)])[None]
        tracker.artifact(VideoGrid(clip, ('a clip',)), 1)
        tracker.artifact(TextSamples(np.array([[1, 2]]), texts=('text',)), 1)
        with pytest.raises(TypeError, match='no renderer for TokenScores'):
            tracker.artifact(TokenScores(np.zeros((1, 2)), np.ones((1, 2))), 1)

    reader = EventAccumulator(str(tmp_path / 'events'), size_guidance={IMAGES: 0, TENSORS: 0})
    reader.Reload()
    clip = Image.open(io.BytesIO(reader.Images('val/samples/0')[0].encoded_image_string))
    assert (clip.format, clip.n_frames) == ('GIF', 2)
    assert json.loads(reader.Tensors('val/samples')[0]
                      .tensor_proto.string_val[0]) == {'prompt': '', 'texts': ['text']}
