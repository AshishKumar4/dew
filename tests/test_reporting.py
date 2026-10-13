"""Reporting persists real results and releases owned sinks on failures."""

import io
import json
import math
from contextlib import nullcontext

import jax
import numpy as np
import optax
import pytest
from affine_run import TimingRegression as Regression, batches
from PIL import Image

from dew.artifacts import Decisions, ImageGrid, Representations, TextSamples, TokenScores, VideoGrid
from dew.data.dataset import Loader
from dew.pool import PREEMPTED_EXIT
from dew.telemetry.records import FitEnded
from dew.training import Checkpoints, LocalTracker, Trackers, Trainer


def records(path):
    return [json.loads(line) for line in (path / 'records.jsonl').read_text().splitlines()]


def decisions() -> Decisions:
    """Two rows of two questions over three options: row 0 asks both, with known answers 2 and 0; row 1
    asks one with no known answer, and its second question has no options."""
    probabilities = np.array([[[0.1, 0.2, 0.7], [0.6, 0.3, 0.1]], [[0.3, 0.3, 0.4], [1 / 3] * 3]], np.float32)
    options = np.array([[[True] * 3, [True] * 3], [[True] * 3, [False] * 3]])
    return Decisions(probabilities=probabilities, options=options, labels=np.array([[2, 0], [1, 0]]),
                     ordinal=np.zeros((2, 2), bool), scored=np.array([[True, True], [False, True]]))


def test_all_builtin_previews_have_local_representations(tmp_path):
    with LocalTracker(tmp_path) as tracker:
        tracker.artifact(ImageGrid(np.zeros((1, 8, 8, 3)), ('image',)), 1)
        tracker.artifact(VideoGrid(np.zeros((1, 2, 8, 8, 3)), ('clip',)), 1)
        tracker.artifact(TextSamples(np.array([[1, 2]]), texts=('text',)), 1)
        tracker.artifact(Representations(np.ones((2, 3)), np.array([0, 1])), 1)
        tracker.artifact(
            TokenScores(np.ones((2, 3)), np.ones((2, 3)), correct=np.zeros_like(np.ones((2, 3)), dtype=bool)),
            1,
        )
        tracker.artifact(decisions(), 1)
    entries = records(tmp_path)
    assert {e['type'] for e in entries} == {
        'ImageGrid', 'VideoGrid', 'TextSamples', 'Representations', 'TokenScores', 'Decisions'}
    saved = {name: np.load(path) for name, path in ((p.name, p) for p in tmp_path.glob('*.npz'))}
    assert any('probabilities' in archive and archive['labels'].tolist() == [[2, 0], [1, 0]]
               for archive in saved.values())
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


def test_fit_records_requests_not_durability_and_leaves_tracker_borrowed(tmp_path):
    local = LocalTracker(tmp_path / 'tracking')
    trainer = Trainer(Regression(), optax.sgd(0.01), key=jax.random.key(0), tracker=local,
                      checkpoints=Checkpoints(str(tmp_path / 'checkpoint')))
    state = trainer.fit(Loader(lambda partition: batches(), None, None, 8), steps=2, log_every=1)
    local.log({'after_fit': 1.}, 2)
    local.close()
    entries = records(tmp_path / 'tracking')
    assert int(state.step) == 2
    assert [e['type'] for e in entries] == ['FitStarted', 'StepCompiled', 'CheckpointRequested',
                                            'FitEnded']
    # The regression model has no remat, and no rung was climbed.
    assert entries[1]['value']['remat'] is None
    # The fit check's arithmetic is the record's too: XLA's analysis of the
    # step, and the devices' stats, which a CPU's allocator does not report.
    memory = entries[1]['value']['memory']
    assert memory['analysis']['temp'] >= 0 and memory['analysis']['argument'] > 0
    assert [device['id'] for device in memory['devices']] == [device.id for device in jax.local_devices()]
    assert entries[-1]['value']['status'] == 'completed'


@pytest.mark.parametrize('preview', [True])
def test_local_scalar_sink_does_not_enable_preview_computation(tmp_path, preview):
    generated = []

    class Display(Regression):
        def preview(self, params, batch, step, *, scored=None):
            generated.append(int(step.step))
            return ImageGrid(np.zeros((1, 8, 8, 3)))

    with LocalTracker(tmp_path) as sink:
        trainer = Trainer(Display(), optax.sgd(.01), key=jax.random.key(0), tracker=sink)
        data = Loader(lambda partition: batches(), lambda partition: iter([next(batches())]), None, 8)
        trainer.fit(data, steps=2, eval_every=1, preview=preview)
    assert generated == [1, 2]
    assert len(list(tmp_path.glob('*.png'))) == 2


def fit(tracker, objective=None):
    """Two logged steps of the small regression through `tracker`."""
    trainer = Trainer(objective or Regression(), optax.sgd(0.01), key=jax.random.key(0),
                      tracker=tracker)
    return trainer.fit(Loader(lambda partition: batches(), None, None, 8), steps=2, log_every=1)


def mlflow_store(tmp_path, monkeypatch):
    """A local MLflow store under `tmp_path`. MLflow 3.16 gates its file
    backend behind this switch, and a file store keeps the run's artifacts in
    the temporary directory instead of a cwd-relative artifact root."""
    pytest.importorskip('mlflow')
    monkeypatch.setenv('MLFLOW_ALLOW_FILE_STORE', 'true')
    return (tmp_path / 'mlruns').as_uri()


@pytest.fixture
def tracked_runs(tmp_path, monkeypatch):
    import wandb
    from test_publish import TrackedRun

    from dew.training import MLflowTracker, WandbTracker

    run = TrackedRun()
    monkeypatch.setattr(wandb, 'init', lambda **_: run)
    mlflow = MLflowTracker('dew-outcomes', uri=mlflow_store(tmp_path, monkeypatch))
    return run, mlflow, Trackers(WandbTracker('dew-outcomes', offline=True), mlflow)


@pytest.mark.parametrize('unwind', [False, True])
@pytest.mark.parametrize('status,exit_code,terminal_status', [
    ('completed', 0, 'FINISHED'), ('lowered', 0, 'FINISHED'),
    ('preempted', PREEMPTED_EXIT, 'KILLED'), ('failed', 1, 'FAILED'), ('interrupted', 1, 'FAILED'),
])
def test_trackers_finish_with_the_recorded_outcome(tracked_runs, status, exit_code, terminal_status, unwind):
    run, mlflow, sinks = tracked_runs
    with (pytest.raises(SystemExit) if unwind else nullcontext()), sinks:
        sinks.artifact(FitEnded(status, 1.), 2)
        client, run_id = mlflow.run
        if unwind:
            raise SystemExit(exit_code)
    assert run.exit_codes == [exit_code]
    assert run.preempting is (status == 'preempted')
    assert client.get_run(run_id).info.status == terminal_status


@pytest.mark.parametrize('code,exit_code,terminal_status', [
    (0, 0, 'FINISHED'), (None, 0, 'FINISHED'), (2, 1, 'FAILED'), ('stopped', 1, 'FAILED'),
])
def test_trackers_use_the_contexts_process_exit_code(tracked_runs, code, exit_code, terminal_status):
    run, mlflow, sinks = tracked_runs
    with pytest.raises(SystemExit) as caught, sinks:
        sinks.artifact(FitEnded('completed', 1.), 2)
        client, run_id = mlflow.run
        raise SystemExit(code)
    assert caught.value.code == code
    assert run.exit_codes == [exit_code]
    assert client.get_run(run_id).info.status == terminal_status


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
            == {'records/FitStarted-0.json', 'records/StepCompiled-0.json',
                'records/FitEnded-2.json'})
    artifacts = client.get_run(run).info.artifact_uri
    outcome = mlflow.artifacts.load_dict(f'{artifacts}/records/FitEnded-2.json')
    assert outcome['step'] == 2 and outcome['value']['status'] == 'completed'
    assert ({file.path for file in client.list_artifacts(run, 'previews/step-2')}
            == {'previews/step-2/preview-2-0.png', 'previews/step-2/preview-2.json'})
    drawn = mlflow.artifacts.download_artifacts(f'{artifacts}/previews/step-2/preview-2-0.png')
    assert Image.open(drawn).size == (8, 8)


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


def test_tensorboard_renders_clips_texts_token_losses_and_decisions(tmp_path):
    pytest.importorskip('tensorboard')
    from tensorboard.backend.event_processing.event_accumulator import (
        HISTOGRAMS,
        IMAGES,
        TENSORS,
        EventAccumulator,
    )

    from dew.training import TensorBoardTracker

    with TensorBoardTracker(tmp_path / 'events') as tracker:
        # Distinct frames: a GIF drops one that does not differ from the last.
        clip = np.stack([np.full((8, 8, 3), -1.0), np.full((8, 8, 3), 1.0)])[None]
        tracker.artifact(VideoGrid(clip, ('a clip',)), 1)
        tracker.artifact(TextSamples(np.array([[1, 2]]), texts=('text',)), 1)
        scores = np.array([[0.5, 2.0]])
        tracker.artifact(TokenScores(scores, np.array([[1.0, 0.0]]), correct=np.zeros((1, 2), dtype=bool)), 1)
        tracker.artifact(decisions(), 1)

    reader = EventAccumulator(str(tmp_path / 'events'), size_guidance={IMAGES: 0, TENSORS: 0, HISTOGRAMS: 0})
    reader.Reload()
    clip = Image.open(io.BytesIO(reader.Images('val/samples/0')[0].encoded_image_string))
    assert (clip.format, clip.n_frames) == ('GIF', 2)
    assert json.loads(reader.Tensors('val/samples')[0]
                      .tensor_proto.string_val[0]) == {'prompt': '', 'texts': ['text']}
    losses = reader.Histograms('val/token_losses')[0].histogram_value
    assert (losses.num, losses.sum) == (1, 0.5)
    shown = json.loads(reader.Tensors('val/decisions')[0].tensor_proto.string_val[0])
    assert [row[:4] for row in shown['rows']] == [[0, 0, 2, 2], [0, 1, 0, 0]]
