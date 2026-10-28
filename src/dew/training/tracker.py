"""Where a run's numbers and artifacts go.

A `Tracker` is the one capability the trainer logs through. `WandbTracker`
sends each artifact to a W&B run, `LocalTracker` writes journals, preview
files and optional plots in a directory, `MLflowTracker` opens an MLflow run
and `TensorBoardTracker` writes an event file.

Each renders artifact types with a `functools.singledispatch` function, so a
new artifact type registers a renderer, and MLflow uploads the files the
local renderers write. `Trackers` fans one report out to several sinks. A
backend is imported when the first value is logged into it.
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import io
import json
import posixpath
import re
import tempfile
import time
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, TextIO

import jax
import numpy as np

from dew._records import integer, record
from dew.artifacts import (
    Artifact,
    Decisions,
    ImageGrid,
    Representations,
    TextSamples,
    TokenScores,
    VideoGrid,
    uint8_pixels,
)
from dew.cleanup import attempted
from dew.telemetry.records import RECORD_TYPES, FitEnded, Record, RunRecord, json_value, reported_number

if TYPE_CHECKING:
    from mlflow.tracking import MlflowClient
    from tensorboard.compat.proto.summary_pb2 import Summary
    from tensorboard.summary.writer.event_file_writer import EventFileWriter

    from dew.figures import Figure


type Reported = Record | Artifact
"""What a tracker is handed: one of the run's own records, or an artifact an
objective's evaluation produces. A renderer is registered per artifact type
Dew defines, and a type with none is refused (`dew.artifacts.Artifact`)."""


class Tracker(Protocol):
    def log(self, scalars: Mapping[str, float], step: int) -> None: ...

    def artifact(self, value: Reported, step: int) -> None: ...

    def close(self) -> None: ...


def _home(array: jax.Array | np.ndarray) -> np.ndarray:
    """`array` as numpy. A shard of a global array is refused.

    A tracker draws on process zero alone, and completing a global array needs
    every process, so an artifact arrives here already brought home by
    `dew._coordination.collective_host`. The refusal names the missing call
    instead of hanging in a collective one process entered by itself.
    """
    if isinstance(array, jax.Array) and not array.is_fully_addressable:
        raise ValueError(
            "a tracker was handed a shard of a global array, which it cannot "
            "complete from one process: bring the artifact home with "
            "dew.distributed.collective_host on every process before drawing it")
    return np.asarray(array)

def _uint8(images: jax.Array | np.ndarray) -> np.ndarray:
    """[-1, 1] floats as the bytes an image viewer reads, the bytes a metric scores."""
    return uint8_pixels(_home(images))


def _png(image: np.ndarray) -> bytes:
    """One `[H, W, C]` uint8 image as PNG bytes."""
    from PIL import Image

    buffer = io.BytesIO()
    Image.fromarray(image.squeeze(-1) if image.shape[-1] == 1 else image).save(buffer, 'PNG')
    return buffer.getvalue()


def _gif(clip: np.ndarray) -> bytes:
    """One `[T, H, W, C]` uint8 clip as an animated GIF."""
    from PIL import Image

    buffer = io.BytesIO()
    frames = [Image.fromarray(frame) for frame in clip]
    frames[0].save(buffer, 'GIF', save_all=True, append_images=frames[1:], duration=100, loop=0)
    return buffer.getvalue()


DECISION_ROWS = 256
"""The most scored questions a tracker shows of one `Decisions` preview."""
DECISION_COLUMNS = ("row", "question", "label", "predicted", "p_label")
"""The columns of a decision preview's rows (`_decided`), as every sink shows them."""


def _texts(value: TextSamples) -> tuple[str, ...]:
    """A text preview's samples as text: its decoded texts, or each row's token ids where it has none."""
    return value.texts or tuple(str(row.tolist()) for row in _home(value.tokens))


def _decided(value: Decisions) -> list[list[float | int]]:
    """A decision preview as rows of [row, question, label, predicted, probability of the label], for the
    questions with a known answer and real options, the first `DECISION_ROWS` of them."""
    probabilities, labels = _home(value.probabilities), _home(value.labels)
    asked = _home(value.scored) & np.any(_home(value.options), axis=-1)
    predicted = np.argmax(probabilities, axis=-1)
    return [[int(row), int(question), int(labels[row, question]), int(predicted[row, question]),
             float(probabilities[row, question, labels[row, question]])]
            for row, question in np.argwhere(asked)[:DECISION_ROWS]]


def _counted_losses(value: TokenScores) -> np.ndarray:
    """The per-token losses of the targets that count."""
    return _home(value.losses)[_home(value.weights) > 0].astype(np.float64)


def _spread(value: Representations) -> np.ndarray:
    """The per-dimension spread of a batch of representations.

    It goes to zero when the encoder stops telling its inputs apart, which
    is the collapse a run watches this number for.
    """
    return np.std(_home(value.features).astype(np.float32), axis=0)


type Payload = Mapping[str, object]
"""One W&B log call: the metric names a renderer writes under and the wandb
values it built for them. wandb is an optional import, so the values are
read by wandb alone and named nowhere here."""


@functools.singledispatch
def render(value: Reported) -> Payload:
    """The W&B payload for an explicitly reported artifact."""
    raise TypeError(f"WandbTracker has no renderer for {type(value).__name__}")

@render.register
def _(value: ImageGrid) -> Payload:
    import wandb

    captions = list(value.captions) + [None] * (len(value.images) - len(value.captions))
    return {"val/samples": [wandb.Image(image, caption=caption)
                            for image, caption in zip(_uint8(value.images), captions, strict=True)]}


@render.register
def _(value: VideoGrid) -> Payload:
    import wandb

    # wandb reads clips as [N, T, C, H, W].
    clips = np.transpose(_uint8(value.videos), (0, 1, 4, 2, 3))
    return {"val/samples": wandb.Video(clips, fps=10, caption=" | ".join(value.captions))}


@render.register
def _(value: TextSamples) -> Payload:
    import wandb

    rows = [[index, value.prompt, text] for index, text in enumerate(_texts(value))]
    table = wandb.Table(columns=["sample", "prompt", "text"])
    for row in rows:
        table.add_data(*row)
    return {"val/samples": table}


@render.register
def _(value: Representations) -> Payload:
    import wandb

    return {"val/representation_std": wandb.Histogram(_spread(value).tolist())}


@render.register
def _(value: TokenScores) -> Payload:
    import wandb

    losses = _counted_losses(value)
    return {"val/token_losses": wandb.Histogram(losses.tolist())} if losses.size else {}


@render.register
def _(value: Decisions) -> Payload:
    import wandb

    table = wandb.Table(columns=list(DECISION_COLUMNS))
    for row in _decided(value):
        table.add_data(*row)
    return {"val/decisions": table}


class _OwnedTracker:
    def close(self) -> None:
        raise NotImplementedError

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        attempted((("Tracker close", self.close),), primary=exc)


class WandbTracker(_OwnedTracker):
    """A Weights & Biases run, opened on the first value logged into it."""

    render = staticmethod(render)

    def __init__(self, project: str, name: str | None = None, *,
                 entity: str | None = None, config: Mapping[str, object] | None = None,
                 offline: bool = False, id: str | None = None):
        self.project = project
        self.name = name
        self.entity = entity
        self.config = dict(config) if config is not None else None
        self.offline = offline
        self.id = id
        self._run = None
        self._closed = False
        self._exit_code = 0

    @property
    def run(self):
        if self._closed:
            raise RuntimeError("WandbTracker is closed")
        if self._run is None:
            import wandb

            self._run = wandb.init(
                project=self.project, name=self.name, entity=self.entity,
                config=self.config, id=self.id, resume="allow",
                mode="offline" if self.offline else None)
            self._run.define_metric("train/step")
            self._run.define_metric("*", step_metric="train/step")
        return self._run

    def log(self, scalars: Mapping[str, float], step: int) -> None:
        self.run.log({**{name: float(value) for name, value in scalars.items()}, "train/step": step})

    def artifact(self, value: Reported, step: int) -> None:
        if self._closed:
            raise RuntimeError("WandbTracker is closed")
        if isinstance(value, RunRecord):
            self.run.config.update({"run_record": json_value(value)}, allow_val_change=True)
            return
        if isinstance(value, RECORD_TYPES):
            self.run.summary[type(value).__name__] = json_value(value)
            self.run.log({"train/step": step,
                          "reporting/record": json.dumps(json_value(value), allow_nan=False),
                          "reporting/type": type(value).__name__})
            if isinstance(value, FitEnded):
                if value.status == "preempted":
                    # W&B keeps a preempting run resumable under its id.
                    self.run.mark_preempting()
                self._exit_code = value.exit_code
            return
        self.run.log({**self.render(value), "train/step": step})

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        run, self._run = self._run, None
        if run is not None:
            run.finish(exit_code=self._exit_code)

    def __exit__(self, exc_type, exc, tb) -> None:
        # A fit that ended preempted raises Preempted, and keeps its code.
        if self._exit_code == 0:
            self._exit_code = FitEnded.outcome(0., exc).exit_code
        super().__exit__(exc_type, exc, tb)


class LocalTracker(_OwnedTracker):
    """Writes each report to a tracking directory as it is logged.

    The reports are JSONL journals (`scalars.jsonl` and `records.jsonl`),
    preview files and, with `plots`, plots. Nonfinite metrics are written to
    the journal as the strings NaN, +Inf and -Inf. Plots are drawn only by
    `plot()`, or by `close` when `plots` is set, as `dew.figures` curves, and
    they read the journal, so the tracker keeps no history in memory.
    """

    def __init__(self, directory: str | Path, *, plots: bool = False):
        self.directory = Path(directory)
        self.plots = plots
        self._files: dict[str, TextIO] = {}
        self._closed = False
        self._sequence = 0

    def _check(self) -> None:
        if self._closed:
            raise RuntimeError('LocalTracker is closed')

    def _write(self, name: str, payload: Mapping[str, object]) -> None:
        self._check()
        line = json.dumps(json_value(payload), allow_nan=False)
        if name not in self._files:
            self.directory.mkdir(parents=True, exist_ok=True)
            self._files[name] = (self.directory / name).open('a', buffering=1)
        self._files[name].write(line + '\n')

    def log(self, scalars: Mapping[str, float], step: int) -> None:
        self._write('scalars.jsonl', {'step': step, 'time': time.time(),
                                    'scalars': {name: float(value) for name, value in scalars.items()}})

    def artifact(self, value: Reported, step: int) -> None:
        self._check()
        if isinstance(value, RECORD_TYPES):
            self._write('records.jsonl', {'type': type(value).__name__, 'step': step,
                                        'time': time.time(), 'value': value})
            return
        self.directory.mkdir(parents=True, exist_ok=True)
        self._sequence += 1
        prefix = self.directory / f'preview-{step}-{time.time_ns()}-{self._sequence}'
        files = _write_preview(value, prefix)
        self._write('records.jsonl', {'type': type(value).__name__, 'step': step,
                                    'time': time.time(), 'files': [path.name for path in files]})

    def scalars(self) -> list[tuple[int, dict[str, float]]]:
        """The scalars the journal holds, each report's step and values in the order they were
        logged, nonfinite values read back as floats. A tracker over a run's directory reads the
        run's reports this way, its final validation's scores among them (`Trainer.fit` logs them
        under the run's last step); a journal no report wrote yet holds none."""
        self._check()
        journal = self.directory / 'scalars.jsonl'
        if not journal.exists():
            return []
        reports = []
        with journal.open() as handle:
            for line in handle:
                row = record(json.loads(line), 'scalars journal row')
                values = record(row['scalars'], 'scalars')
                reports.append((integer(row['step'], 'step'),
                                {name: reported_number(value, name) for name, value in values.items()}))
        return reports

    def plot(self, *metrics: str, smooth: int = 1, y_label: str | None = None,
             y_range: tuple[float, float] | None = None) -> Figure:
        """The journal's `metrics` as one figure (`dew.figures.curve`), each line at the steps its
        metric was logged: by default the training loss beside every split's validation loss (each
        `<prefix>/loss` the journal holds). `smooth` draws each line's running mean over that many
        points over it, and `y_label` names the axis, by default the metrics' common last part
        (`loss`). A nonfinite value is left out of its line and counted on the figure; the journal
        keeps it. Raises `KeyError` for a metric the journal never logged, naming those it holds."""
        self._check()
        return _curves(self._history(), metrics, smooth=smooth, y_label=y_label, y_range=y_range)

    def plot_files(self) -> list[Path]:
        """Draw each metric in the journal to its own SVG file and return the paths. A file is named
        for its metric, each character outside `[A-Za-z0-9_.-]` written `-`, with a digest of the
        whole name, so two names never share a file. The journal is not changed."""
        self._check()
        history = self._history()
        paths = []
        for name in history:
            readable = re.sub(r'[^A-Za-z0-9_.-]', '-', name)[:64]
            path = self.directory / f'{readable}-{hashlib.sha256(name.encode()).hexdigest()[:8]}.svg'
            _curves(history, (name,)).save(path)
            paths.append(path)
        return paths

    def _history(self) -> dict[str, list[tuple[int, float]]]:
        """Each metric the journal holds, its steps and values in the order logged."""
        history: dict[str, list[tuple[int, float]]] = {}
        for step, values in self.scalars():
            for name, value in values.items():
                history.setdefault(name, []).append((step, value))
        return history

    def close(self) -> None:
        if self._closed:
            return

        def plotted() -> None:
            self.plot_files()

        def closed() -> None:
            self._closed = True

        try:
            plots = [("Journal plots", plotted)] if self.plots else []
            attempted([*plots, ("Journal state", closed),
                       *((f"Journal close of {name}", handle.close) for name, handle in self._files.items())])
        finally:
            self._files.clear()


def _curves(history: Mapping[str, list[tuple[int, float]]], metrics: tuple[str, ...], *, smooth: int = 1,
            y_label: str | None = None, y_range: tuple[float, float] | None = None) -> Figure:
    """`metrics` of a journal's `history` as one curve (`LocalTracker.plot`)."""
    from dew.figures import curve
    names = metrics or tuple(name for name in history if name.count('/') == 1 and name.endswith('/loss'))
    if missing := [name for name in names if name not in history] or ([] if names else ['train/loss']):
        raise KeyError(f'the journal logged no {", ".join(missing)}; it holds {sorted(history)}')
    if y_label is None:
        lasts = {name.rpartition('/')[2] for name in names}
        y_label = lasts.pop() if len(lasts) == 1 else 'value'
    return curve({name: [value for _, value in history[name]] for name in names},
                 x={name: [step for step, _ in history[name]] for name in names},
                 x_label='step', y_label=y_label, y_range=y_range, smooth=smooth)


@functools.singledispatch
def _write_preview(value: Reported, prefix: Path) -> list[Path]:
    raise TypeError(f'LocalTracker has no renderer for {type(value).__name__}')


def _write_frames(frames, encode, suffix: str, captions, prefix: Path) -> list[Path]:
    """Write one file per frame, and the captions beside them as JSON."""
    paths = []
    for index, frame in enumerate(frames):
        path = prefix.with_name(f'{prefix.name}-{index}{suffix}')
        path.write_bytes(encode(frame))
        paths.append(path)
    written = prefix.with_suffix('.json')
    written.write_text(json.dumps(list(captions)))
    return [*paths, written]


@_write_preview.register
def _(value: ImageGrid, prefix: Path) -> list[Path]:
    return _write_frames(_uint8(value.images), _png, '.png', value.captions, prefix)


@_write_preview.register
def _(value: VideoGrid, prefix: Path) -> list[Path]:
    return _write_frames(_uint8(value.videos), _gif, '.gif', value.captions, prefix)


@_write_preview.register
def _(value: TextSamples, prefix: Path) -> list[Path]:
    path = prefix.with_suffix('.json')
    path.write_text(json.dumps({'prompt': value.prompt, 'texts': list(value.texts),
                                'tokens': _home(value.tokens).tolist()}))
    return [path]


@_write_preview.register
def _(value: Representations, prefix: Path) -> list[Path]:
    return _arrays(prefix, features=value.features, labels=value.labels)


@_write_preview.register
def _(value: TokenScores, prefix: Path) -> list[Path]:
    return _arrays(prefix, losses=value.losses, weights=value.weights)


def _arrays(prefix: Path, **arrays) -> list[Path]:
    """`arrays`, brought home, saved together under `prefix` as one .npz preview."""
    path = prefix.with_suffix('.npz')
    np.savez(path, allow_pickle=False, **{name: _home(array) for name, array in arrays.items()})
    return [path]


@_write_preview.register
def _(value: Decisions, prefix: Path) -> list[Path]:
    return _arrays(prefix, probabilities=value.probabilities, options=value.options, labels=value.labels,
                   targets=value.targets, ordinal=value.ordinal, scored=value.scored)


MLFLOW_NAME = re.compile(r"[/\w.\- ]{0,250}")
"""The metric names MLflow takes on every platform: alphanumerics, underscores, periods, dashes,
spaces and slashes, at most 250 of them (mlflow/utils/validation.py, `validate_param_and_metric_name`
and `MAX_ENTITY_KEY_LENGTH`; a colon only off Windows)."""


def mlflow_name(name: str) -> str:
    """`name` as an MLflow metric's: as it is where MLflow takes it, else with each character MLflow
    refuses and each empty, `.` or `..` path segment as `_`, cut to fit, and the first 8 hex digits
    of `name`'s SHA-256 after a period. A metric's name can carry free text (a rollout's data
    source, `math[hard]`), which MLflow would refuse at the first log; the digest keeps two names
    that read alike apart."""
    taken = MLFLOW_NAME.fullmatch(name) and posixpath.normpath(name) == name
    if taken and not name.startswith(("/", "..")) and name not in ("", "."):
        return name
    digest = hashlib.sha256(name.encode()).hexdigest()[:8]
    segments = [segment if segment not in ("", ".", "..") else "_"
                for segment in re.sub(r"[^/\w.\- ]", "_", name).split("/")]
    path = "/".join(segments)
    # MLflow refuses a name whose normal form starts with `..` (`path_not_unique`), `..loss` too.
    return f"{'_' * path.startswith('..')}{path}"[:250 - len(digest) - 1] + f".{digest}"


class MLflowTracker(_OwnedTracker):
    """An MLflow run in `experiment`, opened on the first value logged into it.

    Scalars are logged as the run's metrics, and a preview is uploaded as the
    files `LocalTracker` writes for it. A record is uploaded as a JSON
    artifact and not as a parameter, because MLflow refuses a second value
    for a parameter and a run reports several records of the same type.
    """

    def __init__(self, experiment: str, name: str | None = None, *, uri: str | None = None):
        self.experiment = experiment
        self.name = name
        self.uri = uri
        self._run = None
        self._closed = False
        self._status = 'FINISHED'
        # Each name MLflow holds, by the name logged under it (`mlflow_name`).
        self._names: dict[str, str] = {}

    @property
    def run(self) -> tuple[MlflowClient, str]:
        """The MLflow client and the run id; the first access creates the run, and the experiment if
        it does not exist."""
        if self._closed:
            raise RuntimeError('MLflowTracker is closed')
        if self._run is None:
            from mlflow.tracking import MlflowClient

            client = MlflowClient(tracking_uri=self.uri)
            known = client.get_experiment_by_name(self.experiment)
            experiment = (client.create_experiment(self.experiment) if known is None
                          else known.experiment_id)
            self._run = client, client.create_run(experiment, run_name=self.name).info.run_id
        return self._run

    def log(self, scalars: Mapping[str, float], step: int) -> None:
        from mlflow.entities import Metric

        client, run = self.run
        moment = int(time.time() * 1000)
        client.log_batch(run, metrics=[Metric(self._held(name), float(value), moment, step)
                                       for name, value in scalars.items()])

    def _held(self, name: str) -> str:
        """The name MLflow holds `name`'s metric under, refused where another of the run's names took
        it, as a name already of MLflow's form and another's repair (`mlflow_name`) can, whose two
        series MLflow would merge into one."""
        held = mlflow_name(name)
        owner = self._names.setdefault(held, name)
        if owner != name:
            raise ValueError(f"metrics {owner!r} and {name!r} both reach MLflow as {held!r}; rename one")
        return held

    def artifact(self, value: Reported, step: int) -> None:
        client, run = self.run
        if isinstance(value, RECORD_TYPES):
            client.log_dict(run, {'step': step, 'value': json_value(value)},
                            f'records/{type(value).__name__}-{step}.json')
            if isinstance(value, FitEnded):
                self._status = value.terminal_status
            return
        with tempfile.TemporaryDirectory() as directory:
            for path in _write_preview(value, Path(directory) / f'preview-{step}'):
                client.log_artifact(run, str(path), f'previews/step-{step}')

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        opened, self._run = self._run, None
        if opened is not None:
            client, run = opened
            client.set_terminated(run, self._status)

    def __exit__(self, exc_type, exc, tb) -> None:
        # A fit that ended preempted raises Preempted, and keeps its status.
        if self._status == 'FINISHED':
            self._status = FitEnded.outcome(0., exc).terminal_status
        super().__exit__(exc_type, exc, tb)


class TensorBoardTracker(_OwnedTracker):
    """A TensorBoard event file in `directory`, opened on the first value
    logged into it.

    Scalars are written as scalar summaries. A record is written as the JSON
    `LocalTracker` journals for it, in a text summary under
    `reporting/<type>`. Image and video grids are written as image summaries,
    with a video as an animated GIF; text samples as text; and
    representations as a histogram of their spread. Other previews raise
    `TypeError`.
    """

    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        self._writer = None
        self._closed = False

    @property
    def writer(self) -> EventFileWriter:
        """The event-file writer, which the first access opens in `directory`."""
        if self._closed:
            raise RuntimeError('TensorBoardTracker is closed')
        if self._writer is None:
            from tensorboard.summary.writer.event_file_writer import EventFileWriter

            self._writer = EventFileWriter(str(self.directory))
        return self._writer

    def _add(self, summary: Summary, step: int) -> None:
        from tensorboard.compat.proto.event_pb2 import Event

        self.writer.add_event(Event(wall_time=time.time(), step=step, summary=summary))

    def log(self, scalars: Mapping[str, float], step: int) -> None:
        from tensorboard.compat.proto.summary_pb2 import Summary

        self._add(Summary(value=[Summary.Value(tag=name, simple_value=float(value))
                                 for name, value in scalars.items()]), step)

    def artifact(self, value: Reported, step: int) -> None:
        if isinstance(value, RECORD_TYPES):
            from tensorboard.compat.proto.summary_pb2 import Summary

            self._add(Summary(value=[_text(f'reporting/{type(value).__name__}',
                                           json.dumps(json_value(value), allow_nan=False))]), step)
            return
        self._add(_summarize(value), step)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        writer, self._writer = self._writer, None
        if writer is not None:
            writer.close()


def _text(tag: str, payload: str) -> Summary.Value:
    """A text-plugin summary value: the one-string tensor a text tab reads, its metadata and tensor
    built as TensorBoard's own TensorFlow-free `text_pb` builds them."""
    from tensorboard.compat.proto.summary_pb2 import Summary
    from tensorboard.plugins.text.metadata import create_summary_metadata
    from tensorboard.util.tensor_util import make_tensor_proto

    return Summary.Value(tag=tag, metadata=create_summary_metadata(display_name=None, description=None),
                         tensor=make_tensor_proto([payload], dtype=np.object_))


def _picture(tag: str, encoded: bytes, frame: np.ndarray) -> Summary.Value:
    """An image-plugin summary value from encoded bytes and the frame in them."""
    from tensorboard.compat.proto.summary_pb2 import Summary

    height, width, channels = frame.shape
    return Summary.Value(tag=tag, image=Summary.Image(
        height=height, width=width, colorspace=channels, encoded_image_string=encoded))


@functools.singledispatch
def _summarize(value: Reported) -> Summary:
    """The TensorBoard summary for an explicitly reported artifact."""
    raise TypeError(f'TensorBoardTracker has no renderer for {type(value).__name__}')


def _frames_summary(frames, encode, shown, captions):
    """A summary holding one image value per frame, and the captions as text.

    `shown` picks the frame whose height, width and channels describe the
    encoded bytes, which for a clip is its first frame.
    """
    from tensorboard.compat.proto.summary_pb2 import Summary

    pictures = [_picture(f'val/samples/{index}', encode(frame), shown(frame))
                for index, frame in enumerate(frames)]
    return Summary(value=[*pictures,
                          _text('val/samples/captions', json.dumps(list(captions)))])


# A registered renderer cannot annotate its return: singledispatch resolves a
# registration's annotations, and the proto module is imported where it is
# used rather than above.
@_summarize.register
def _(value: ImageGrid):
    return _frames_summary(_uint8(value.images), _png, lambda image: image, value.captions)


@_summarize.register
def _(value: VideoGrid):
    # TensorBoard has no video summary; its image plugin animates a GIF.
    return _frames_summary(_uint8(value.videos), _gif, lambda clip: clip[0], value.captions)


@_summarize.register
def _(value: TextSamples):
    from tensorboard.compat.proto.summary_pb2 import Summary

    return Summary(value=[_text('val/samples', json.dumps(
        {'prompt': value.prompt, 'texts': list(_texts(value))}))])


def _histogram(tag: str, values: np.ndarray) -> Summary:
    """A histogram-plugin summary of `values`, in 30 buckets, as TensorBoard's own TensorFlow-free
    `histogram_pb` builds it: the plugin's tensor of bucket edges and counts."""
    from tensorboard.plugins.histogram.summary_v2 import histogram_pb

    return histogram_pb(tag, values, buckets=30)


@_summarize.register
def _(value: Representations):
    return _histogram('val/representation_std', _spread(value).astype(np.float64))


@_summarize.register
def _(value: TokenScores):
    from tensorboard.compat.proto.summary_pb2 import Summary

    losses = _counted_losses(value)
    return _histogram('val/token_losses', losses) if losses.size else Summary(value=[])


@_summarize.register
def _(value: Decisions):
    from tensorboard.compat.proto.summary_pb2 import Summary

    return Summary(value=[_text('val/decisions', json.dumps(
        {'columns': list(DECISION_COLUMNS), 'rows': _decided(value)}))])


class Trackers(_OwnedTracker):
    """Sends each report to several trackers.

    Every tracker receives every report, even when an earlier tracker fails.
    Then the first failure is raised, with the later ones attached as notes.
    """

    def __init__(self, *trackers: Tracker):
        self.trackers = trackers
        self._closed = False

    def _each(self, operation) -> None:
        if self._closed:
            raise RuntimeError('Trackers is closed')
        attempted([(type(tracker).__name__, functools.partial(operation, tracker))
                   for tracker in self.trackers])

    def log(self, scalars: Mapping[str, float], step: int) -> None:
        self._each(lambda tracker: tracker.log(scalars, step))

    def artifact(self, value: Reported, step: int) -> None:
        self._each(lambda tracker: tracker.artifact(value, step))

    def close(self) -> None:
        if not self._closed:
            try:
                self._each(lambda tracker: tracker.close())
            finally:
                self._closed = True

    def __exit__(self, exc_type, exc, tb) -> None:
        # Not _each: a later owned sink must see an earlier sink's close
        # failure so it records the run as failed.
        if self._closed:
            return
        error = exc
        for tracker in self.trackers:
            try:
                if isinstance(tracker, _OwnedTracker):
                    tracker.__exit__(type(error) if error is not None else None, error,
                                     error.__traceback__ if error is not None else None)
                else:
                    tracker.close()
            except BaseException as failure:
                if error is None:
                    error = failure
                else:
                    error.add_note(f'Tracker close failed: {failure!r}')
        self._closed = True
        if exc is None and error is not None:
            raise error


@dataclasses.dataclass(frozen=True)
class Wandb:
    """Where a run reports to Weights & Biases.

    Setting it turns tracking on; the entity and the offline switch have no effect
    without a project. `name` is the run's display name there, apart from the name
    its checkpoints are kept under; None shows it as the run is named.
    """

    project: str
    entity: str | None = None
    offline: bool = False
    name: str | None = None
