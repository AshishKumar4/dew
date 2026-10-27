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
    labels = np.array([[2, 0], [1, 0]])
    return Decisions(probabilities=probabilities, options=options, labels=labels, targets=np.eye(3)[labels],
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


def test_a_journal_reads_back_what_a_fit_logged_its_final_validation_and_nonfinite_values(tmp_path):
    """A tracker over a tracking directory reads its scalars back as they were logged, in order:
    every report a fit sent it, the final validation's scores under the run's last step among them,
    and the spellings of a nonfinite value as that value."""
    class Recording(LocalTracker):
        def __init__(self, directory):
            super().__init__(directory)
            self.sent = []

        def log(self, scalars, step):
            self.sent.append((step, {name: float(value) for name, value in scalars.items()}))
            super().log(scalars, step)

    held = next(batches())
    tracker = Recording(tmp_path / 'tracking')
    # Checkpoints rank the run by its validation loss, which each evaluation scores.
    trainer = Trainer(Regression(), optax.sgd(0.01), key=jax.random.key(0), tracker=tracker,
                      checkpoints=Checkpoints(str(tmp_path / 'checkpoint')))
    state = trainer.fit(Loader(lambda partition: batches(), lambda partition: iter([held]), None, 8),
                        steps=2, log_every=1, eval_every=1)
    tracker.log({'psnr': math.inf, 'loss': math.nan, 'floor': -math.inf}, 3)
    tracker.close()
    with LocalTracker(tmp_path / 'tracking') as journal:
        read = journal.scalars()
    assert read[:-1] == tracker.sent[:-1]
    assert any(step == int(state.step) and 'val/loss' in scalars for step, scalars in read)
    step, nonfinite = read[-1]
    assert step == 3 and nonfinite['psnr'] == math.inf and nonfinite['floor'] == -math.inf
    assert math.isnan(nonfinite['loss'])


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
    # The pass that warms the evaluations before the first step previews too, and logs nothing.
    assert generated == [0, 1, 2]
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


def test_mlflow_takes_a_metric_named_in_free_text_under_a_name_it_accepts(tmp_path, monkeypatch):
    """A rollout source's free-text name in a metric's (`math[hard]`, `a/../b`) reaches MLflow as a
    name it accepts, two such names apart; a name MLflow accepts arrives as it is."""
    store = mlflow_store(tmp_path, monkeypatch)
    from dew.training import MLflowTracker

    names = ("rollout/source/openai/gsm8k/reward/mean", "rollout/source/math[hard]/reward/mean",
             "rollout/source/math(hard)/reward/mean", "rollout/source/a/../b/reward/mean", "..loss",
             ".../loss")
    with MLflowTracker('dew-reporting', 'free-text', uri=store) as tracker:
        tracker.log({name: float(index) for index, name in enumerate(names)}, 1)
        client, run = tracker.run
    logged = client.get_run(run).data.metrics
    from dew.training.tracker import mlflow_name

    assert len({mlflow_name(name) for name in names}) == len(names)
    assert {mlflow_name(name): index for index, name in enumerate(names)} == logged
    assert mlflow_name(names[0]) == names[0]
    # A name already of MLflow's form that another's repair reaches is refused, not merged with it.
    with MLflowTracker('dew-reporting', 'taken', uri=store) as tracker:
        tracker.log({"": 1.0}, 1)
        with pytest.raises(ValueError):
            tracker.log({mlflow_name(""): 2.0}, 2)


def test_the_panel_shows_the_metrics_its_last_interval_reported():
    """A metric an interval no longer reports, a rollout source absent from its batch, leaves the
    panel rather than keep showing its last value, and shows its trend again when it returns."""
    from dew.training.display import TrainingDisplay

    display = TrainingDisplay(total=10)
    display.interval(1, {"train/loss": 2.0, "rollout/source/a/reward": 1.0})
    display.interval(2, {"train/loss": 1.5})
    assert [row.name for _, row in display.rows()] == ["loss"]
    display.interval(3, {"train/loss": 1.0, "rollout/source/a/reward": 0.5})
    assert [(row.name, list(row.values)) for _, row in display.rows()] == [
        ("loss", [2.0, 1.5, 1.0]), ("source/a/reward", [1.0, 0.5])]


def test_the_panel_names_a_lower_bound_utilization_and_its_uncounted_kernels_within_its_name_column():
    """A step some of whose launches' work XLA's cost analysis cannot count reports its utilizations
    as lower bounds beside how many launches went uncounted: the panel shows them as `hfu ≥` and
    `hlo_flops_util ≥` and the count as `uncounted kernels`, each whole within the name column, where
    the metrics' own names would be cut."""
    from dew.training.display import TRAINER_SHOWN, TrainingDisplay

    display = TrainingDisplay(total=10, shown=TRAINER_SHOWN)
    display.interval(1, {"train/loss": 2.0, "train/hfu_lower_bound": 0.41,
                         "train/hlo_flops_util_lower_bound": 0.38, "train/unsized_kernels": 50.0})
    names = {row.name: group for group, row in display.rows()}
    assert names == {"loss": "train", "hfu ≥": "throughput", "hlo_flops_util ≥": "throughput",
                     "uncounted kernels": "throughput"}
    assert all(len(name) + 2 <= 24 for name in names)


def test_a_metrics_label_names_it_in_evaluations_and_the_summary_too():
    """A label (`Shown.label`) is the name everywhere the display shows the metric: an evaluation's
    scores, its rows and the run's summary, whose trackers keep the metric's own name."""
    from dew.objectives.base import Shown
    from dew.training.display import TrainingDisplay
    from dew.training.evaluation import Evaluation

    display = TrainingDisplay(total=2, first=0, shown={"loss": Shown(better="lower", label="nll"),
                                                       "accuracy": Shown(percent=True, label="top-1")})
    evaluation = Evaluation(step=2, split="val", scores={"val/accuracy": 0.5}, coordinated_batches=1,
                            records=4, event_key=(0,), elapsed_seconds=0.1, previews=())
    display.evaluation(evaluation)
    assert display.scores(evaluation).plain.startswith("top-1 ")
    assert [row.name for _, row in display.evaluation_rows()] == ["top-1"]
    # The summary's final loss is named as the display names it.
    assert display.named("loss") == "nll" and display.named("tokens") == "tokens"


def test_the_panel_shows_a_runs_data_as_one_line_and_its_rows_without_the_counters():
    """A run's data counters (`data/...`) reach the panel as one line, not as a row each: each
    label's rows as its reader counted them, named only where there is more than one label, the
    records rejected over every label, a reason only where something was rejected for it, and the
    share of the rows truncated, over the rows of the labels that truncate, only where any was."""
    from dew.training import display as shown
    from dew.training.display import TrainingDisplay

    display = TrainingDisplay(total=10)
    display.interval(1, {"train/loss": 2.0, "data/rows/tiny-shakespeare": 4824.0,
                         "data/rejected/tiny-shakespeare/empty": 0.0, "data/truncated/tiny-shakespeare": 0.0,
                         "data/chunks_per_record/tiny-shakespeare": 1.0})
    assert [row.name for _, row in display.rows()] == ["loss"]
    data_line = shown.data_line
    line = data_line(display.corpus_counts)
    assert line is not None and line.plain == "data 4,824 rows  ·  0 rejected"
    # mix(web, code).filter(f): the corpora count the rows they fed the mix, the mixture the rows
    # f kept, and f's drops are the mixture's.
    mixture = data_line({"data/rows/web": 3000.0, "data/rows/code": 1824.0, "data/rows/mixture": 4700.0,
                         "data/rejected/web/too_long": 2.0, "data/rejected/code/too_long": 1.0,
                         "data/rejected/code/empty": 0.0, "data/rejected/mixture/filtered": 124.0,
                         "data/truncated/web": 0.25, "data/truncated/code": 0.0})
    assert mixture is not None
    assert mixture.plain == ("data web 3,000  code 1,824  mixture 4,700 rows  ·  127 rejected (too_long 3, "
                             "filtered 124)  ·  15.5% truncated")
    assert data_line({"train/loss": 1.0}) is None


def test_tensorboard_events_hold_the_fit_read_back_with_the_accumulator(tmp_path):
    pytest.importorskip('tensorboard')
    from tensorboard.backend.event_processing.event_accumulator import (
        IMAGES,
        SCALARS,
        TENSORS,
        EventAccumulator,
    )
    from tensorboard.util import tensor_util

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
    # The histogram plugin's tensor (`histogram_pb`): each bucket's left edge, right edge and count.
    buckets = tensor_util.make_ndarray(reader.Tensors('val/representation_std')[0].tensor_proto)
    assert buckets.shape == (30, 3) and buckets[:, 2].sum() == 3


def test_tensorboard_renders_clips_texts_token_losses_and_decisions(tmp_path):
    pytest.importorskip('tensorboard')
    from tensorboard.backend.event_processing.event_accumulator import (
        HISTOGRAMS,
        IMAGES,
        TENSORS,
        EventAccumulator,
    )
    from tensorboard.util import tensor_util

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
    # The histogram plugin's tensor (`histogram_pb`): each bucket's left edge, right edge and count.
    losses = tensor_util.make_ndarray(reader.Tensors('val/token_losses')[0].tensor_proto)
    assert losses[:, 2].sum() == 1 and losses[:, 0].min() <= 0.5 <= losses[:, 1].max()
    shown = json.loads(reader.Tensors('val/decisions')[0].tensor_proto.string_val[0])
    assert [row[:4] for row in shown['rows']] == [[0, 0, 2, 2], [0, 1, 0, 0]]


def test_a_live_panel_keeps_redrawing_while_evaluations_of_new_splits_arrive():
    """Rich redraws the live panel on a thread of its own while training records each split's first
    evaluation; a redraw that walked the splits while one was added would raise there ("dictionary
    changed size during iteration"), which Rich does not catch, and the panel would stop updating.
    A redraw walking the splits without pause, the interpreter switching threads every microsecond,
    while thousands of splits arrive, never raises, and draws every split."""
    import sys
    import threading

    from dew.training.display import TrainingDisplay
    from dew.training.evaluation import Evaluation

    display = TrainingDisplay(total=10)
    failures: list[BaseException] = []
    arriving = threading.Event()

    def redraw():
        while not arriving.is_set():
            try:
                display.evaluation_rows()
            except RuntimeError as failure:
                failures.append(failure)
                return

    interval = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    redrawing = threading.Thread(target=redraw)
    redrawing.start()
    try:
        for split in range(3000):
            display.evaluation(Evaluation(
                step=1, split=f"split{split}", scores={f"split{split}/loss": float(split)},
                coordinated_batches=1, records=1, event_key=(0,), elapsed_seconds=0.0, previews=()))
    finally:
        arriving.set()
        redrawing.join()
        sys.setswitchinterval(interval)
    assert failures == []
    assert len(display.evaluation_rows()) == 3000


def paths_of(figure) -> list[list[tuple[float, ...]]]:
    """The points of each line a figure draws, in the order drawn."""
    return [[tuple(float(value) for value in point.split(","))
             for point in shape.attributes["d"][1:].split(" L")]
            for shape in figure.shapes if shape.tag == "path"]


def texts_of(figure) -> list[str]:
    return [shape.text for shape in figure.shapes if shape.tag == "text"]


def test_a_figures_label_and_text_read_back_as_written():
    """Markup characters in a figure's label and text are escaped: an SVG reader reads back what was
    written, quotes of either kind included."""
    from xml.etree import ElementTree

    from dew import figures

    written = "a < b & \"c\" 'd' > e"
    figure = figures.Figure(120, 40, label=written)
    figure.text(10, 20, written)
    read = ElementTree.fromstring(figure.svg())
    svg = "{http://www.w3.org/2000/svg}"
    assert read.get("aria-label") == written
    assert read.find(f"{svg}title").text == written
    assert read.find(f"{svg}text").text == written


def test_a_curve_draws_each_series_at_its_own_steps_from_its_finite_values():
    """`figures.curve` with steps per series: the axes span the finite values only, a nonfinite
    value is left out of its line (its neighbours joined) and counted on the figure, the running
    mean is taken over the finite values, the last value is the last window's mean, and a series
    with no finite value draws no line. Shared steps still give each series as many as it has
    values, as the tutorials' calls pass them."""
    from dew import figures

    drawn = figures.curve({"loss": [4.0, math.nan, 2.0, 1.0, math.inf], "other": [math.nan]},
                          x={"loss": [1, 2, 3, 4, 5], "other": [3]}, x_label="step", y_label="loss",
                          smooth=2)
    raw, mean = paths_of(drawn)
    # Finite values 4, 2, 1 at steps 1, 3, 4, and their running mean of two, 3 and 1.5, at steps 3
    # and 4. The plot area runs from x 50 to 440 over steps 1 to 4, the last finite step, and from
    # y 242 to 62 over values 1 to 4, the finite ones, under the legend's row.
    assert raw == pytest.approx([(50, 62), (310, 182), (440, 242)], abs=0.01)
    assert mean == pytest.approx([(310, 122), (440, 212)], abs=0.01)
    assert "3 nonfinite values left out" in texts_of(drawn)
    assert "loss 1.50" in texts_of(drawn) and "other none finite" in texts_of(drawn)

    # One series names its last value at its line's end: 3 at step 10, the plot area's top right.
    alone = figures.curve({"loss": [1.0, 3.0]}, x=[0, 10], x_label="step", y_label="loss")
    (end,) = [shape for shape in alone.shapes if shape.tag == "text" and shape.text == "3.00"]
    assert (float(end.attributes["x"]), float(end.attributes["y"])) == pytest.approx((436, 30), abs=0.01)

    shared = figures.curve({"long": [1.0, 2.0, 3.0], "short": [1.0, 3.0]}, x=[10, 20, 30],
                           x_label="step", y_label="value")
    long, short = paths_of(shared)
    assert [x for x, _ in short] == [x for x, _ in long][:2]


def test_a_journals_plot_draws_training_beside_validation_each_at_its_own_steps(tmp_path):
    """`LocalTracker.plot` draws the metrics it names, by default the training loss beside each
    split's validation loss, as one `dew.figures` curve whose lines sit at the steps each metric
    was logged: validation's two evaluations land where training's steps 5 and 10 do. The axis
    is named for the metrics' common last part unless the caller names it, even as nothing. A
    metric the journal never logged is refused by name; and `plots=True` writes each metric to a
    file of its own on close, two names that would read alike included."""
    tracker = LocalTracker(tmp_path / "run", plots=True)
    for step in range(1, 11):
        tracker.log({"train/loss": 2.0 / step, "train/grad_norm": 1.0 + step, "train-loss": 0.5}, step)
        if step % 5 == 0:
            tracker.log({"val/loss": 2.5 / step}, step)
    tracker.log({"train/loss": math.inf}, 11)
    figure = tracker.plot()
    train, validation = paths_of(figure)
    assert len(train) == 10 and len(validation) == 2
    assert validation[0][0] == train[4][0] and validation[1][0] == train[9][0]
    assert "1 nonfinite value left out" in texts_of(figure) and "loss" in texts_of(figure)
    assert all(math.isfinite(value) for line in (train, validation) for point in line for value in point)
    assert "value" in texts_of(tracker.plot("train/loss", "train/grad_norm"))
    assert "loss" not in texts_of(tracker.plot("train/loss", y_label=""))
    with pytest.raises(KeyError):
        tracker.plot("val/perplexity")
    tracker.close()
    written = sorted(path.name for path in (tmp_path / "run").glob("*.svg"))
    assert len(written) == 4 and len({name.rpartition("-")[0] for name in written}) == 3


TERMINAL = """
import logging
import sys

sys.path.insert(0, "tests")
from affine_run import fit_one_step
from rich.console import Console

from dew.logging import display_console

fit_one_step()
log = logging.getLogger("dew.terminal")
try:
    raise ValueError("the traceback's own words")
except ValueError:
    log.exception("a warning with its traceback")
with display_console(Console(file=sys.stdout, force_terminal=True, soft_wrap=True)):
    log.warning("a warning on the panel's console")
log.warning("a warning back on stderr")
"""


def test_a_run_in_a_terminal_logs_on_rich_handlers_and_the_panel_takes_them():
    """A run in a pseudo-terminal installs Dew's rich stderr handler: a warning with a traceback
    prints its line and then the traceback as rich renders it, and the live panel's console takes the
    handler's records while it shows (`display_console`), handing them back after."""
    import os
    import pty
    import subprocess
    import sys

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, (os.path.join(root, "src"),
                                                                     os.environ.get("PYTHONPATH")))),
           "COLUMNS": "200", "TERM": "xterm-256color"}
    leader, follower = pty.openpty()
    child = subprocess.Popen([sys.executable, "-c", TERMINAL], stdin=follower, stdout=follower,
                             stderr=follower, env=env, cwd=root)
    os.close(follower)
    printed = []
    while True:
        try:
            chunk = os.read(leader, 1 << 16)
        except OSError:
            # The terminal closes, as the child exits, with EIO.
            break
        if not chunk:
            break
        printed.append(chunk)
    os.close(leader)
    assert child.wait(timeout=300) == 0
    text = b"".join(printed).decode(errors="replace")
    for words in ("a warning with its traceback", "Traceback", "the traceback's own words",
                  "a warning on the panel's console", "a warning back on stderr"):
        assert words in text, text[-3000:]
