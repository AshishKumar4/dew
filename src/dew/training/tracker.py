"""Where a run's numbers and artifacts go.

A `Tracker` is the one capability the trainer logs through. `WandbTracker`
sends each artifact to a W&B run, `LocalTracker` writes journals, preview
files and optional plots in a directory, `MLflowTracker` opens an MLflow run
and `TensorBoardTracker` writes an event file; each renders artifact types
with a `functools.singledispatch` function, so a new artifact type registers
a renderer, and MLflow uploads the files the local renderers write.
`Trackers` fans one report out to several sinks. A backend is imported when
the first value is logged into it.
"""

from __future__ import annotations

import functools
import io
import json
import tempfile
import time
from pathlib import Path
from collections.abc import Mapping
from typing import Protocol, TextIO, TypeAlias, TYPE_CHECKING

import jax
import numpy as np

from dew.artifacts import ImageGrid, Representations, TextSamples, VideoGrid, TokenScores
from dew.telemetry.records import RECORD_TYPES, RunRecord, FitEnded, json_value

if TYPE_CHECKING:
    from mlflow.tracking import MlflowClient
    from tensorboard.compat.proto.summary_pb2 import Summary
    from tensorboard.summary.writer.event_file_writer import EventFileWriter


class Tracker(Protocol):
    def log(self, scalars: Mapping[str, float], step: int) -> None: ...

    def artifact(self, value: object, step: int) -> None: ...

    def close(self) -> None: ...


def _home(array: jax.Array | np.ndarray) -> np.ndarray:
    """`array` as numpy. A shard of a global array is refused.

    A tracker draws on process zero alone, and completing a global array needs
    every process, so an artifact arrives here already brought home by
    `dew.artifacts.host`. The refusal names the missing call instead of
    hanging in a collective one process entered by itself.
    """
    if isinstance(array, jax.Array) and not array.is_fully_addressable:
        raise ValueError(
            "a tracker was handed a shard of a global array, which it cannot "
            "complete from one process: bring the artifact home with "
            "dew.artifacts.host on every process before drawing it")
    return np.asarray(array)

def _uint8(images: jax.Array | np.ndarray) -> np.ndarray:
    """[-1, 1] floats as the bytes an image viewer reads."""
    return np.clip((_home(images).astype(np.float32) + 1.0) * 127.5, 0, 255).astype(np.uint8)


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


Payload: TypeAlias = dict[str, object]


@functools.singledispatch
def render(value: object) -> Payload:
    """The W&B payload for an explicitly reported artifact."""
    raise TypeError(f"WandbTracker has no renderer for {type(value).__name__}")

@render.register
def _(value: ImageGrid) -> Payload:
    import wandb

    captions = list(value.captions) + [None] * (len(value.images) - len(value.captions))
    return {"val/samples": [wandb.Image(image, caption=caption)
                            for image, caption in zip(_uint8(value.images), captions)]}


@render.register
def _(value: VideoGrid) -> Payload:
    import wandb

    # wandb reads clips as [N, T, C, H, W].
    clips = np.transpose(_uint8(value.videos), (0, 1, 4, 2, 3))
    return {"val/samples": wandb.Video(clips, fps=10, caption=" | ".join(value.captions))}


@render.register
def _(value: TextSamples) -> Payload:
    import wandb

    texts = value.texts or tuple(str(row.tolist()) for row in _home(value.tokens))
    rows = [[index, value.prompt, text] for index, text in enumerate(texts)]
    table = wandb.Table(columns=["sample", "prompt", "text"])
    for row in rows:
        table.add_data(*row)
    return {"val/samples": table}


@render.register
def _(value: Representations) -> Payload:
    import wandb

    # The per-dimension spread across the batch, the collapse view of a
    # representation. It goes to zero when the encoder stops telling inputs
    # apart.
    spread = np.std(_home(value.features).astype(np.float32), axis=0)
    return {"val/representation_std": wandb.Histogram(spread.tolist())}


class _OwnedTracker:
    def close(self) -> None:
        raise NotImplementedError

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        try:
            self.close()
        except BaseException as failure:
            if exc is None:
                raise
            exc.add_note(f"Tracker close failed: {failure!r}")


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

    def artifact(self, value: object, step: int) -> None:
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
                self._exit_code = int(value.status != "completed")
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
        if exc is not None:
            self._exit_code = 1
        super().__exit__(exc_type, exc, tb)


class LocalTracker(_OwnedTracker):
    """Synchronous reports in a tracking directory: JSONL journals, preview files
    and optional plots.

    Nonfinite metrics are journaled as the strings NaN, +Inf and -Inf. Plotting
    runs only on plot() or close and reads the journal rather than retaining
    history.
    """

    def __init__(self, directory: str | Path, *, plots: bool = False):
        self.directory = Path(directory)
        self.plots = plots
        self._files: dict[str, TextIO] = {}
        self._closed = False
        self._sequence = 0
        if plots:
            import importlib.util
            if importlib.util.find_spec('matplotlib') is None:
                raise ImportError('LocalTracker plots need dew-ml[plots]')

    def _check(self) -> None:
        if self._closed:
            raise RuntimeError('LocalTracker is closed')

    def _write(self, name: str, value: object) -> None:
        self._check()
        payload = json.dumps(json_value(value), allow_nan=False)
        if name not in self._files:
            self.directory.mkdir(parents=True, exist_ok=True)
            self._files[name] = (self.directory / name).open('a', buffering=1)
        self._files[name].write(payload + '\n')

    def log(self, scalars: Mapping[str, float], step: int) -> None:
        self._write('scalars.jsonl', {'step': step, 'time': time.time(),
                                    'scalars': {name: float(value) for name, value in scalars.items()}})

    def artifact(self, value: object, step: int) -> None:
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

    def plot(self) -> list[Path]:
        """Render journal metrics with matplotlib Agg; never changes the journal."""
        self._check()
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        history: dict[str, list[tuple[int, float]]] = {}
        journal = self.directory / 'scalars.jsonl'
        if not journal.exists():
            return []
        with journal.open() as handle:
            for line in handle:
                row = json.loads(line)
                for key, value in row['scalars'].items():
                    history.setdefault(key, []).append((row['step'], float(value)))
        paths = []
        for index, (key, points) in enumerate(history.items()):
            figure = Figure(figsize=(7, 4))
            FigureCanvasAgg(figure)
            axes = figure.subplots()
            x, y = zip(*points)
            values = np.asarray(y)
            axes.plot(x, np.where(np.isfinite(values), values, np.nan), marker=".")
            axes.set(title=key, xlabel='step')
            nonfinite = sum(not np.isfinite(value) for value in y)
            if nonfinite:
                axes.text(0.02, 0.98, f'{nonfinite} nonfinite values (see journal)',
                          transform=axes.transAxes, va='top')
            path = self.directory / f'metric-{index}.png'
            figure.savefig(path)
            paths.append(path)
        return paths

    def close(self) -> None:
        if self._closed:
            return
        error = None
        try:
            if self.plots:
                self.plot()
        except BaseException as failure:
            error = failure
        self._closed = True
        for handle in self._files.values():
            try:
                handle.close()
            except BaseException as failure:
                if error is None:
                    error = failure
                else:
                    error.add_note(f'Journal close also failed: {failure!r}')
        self._files.clear()
        if error is not None:
            raise error


@functools.singledispatch
def _write_preview(value: object, prefix: Path) -> list[Path]:
    raise TypeError(f'LocalTracker has no renderer for {type(value).__name__}')


@_write_preview.register
def _(value: ImageGrid, prefix: Path) -> list[Path]:
    paths = []
    for index, image in enumerate(_uint8(value.images)):
        path = prefix.with_name(f'{prefix.name}-{index}.png')
        path.write_bytes(_png(image))
        paths.append(path)
    captions = prefix.with_suffix('.json')
    captions.write_text(json.dumps(list(value.captions)))
    return [*paths, captions]


@_write_preview.register
def _(value: VideoGrid, prefix: Path) -> list[Path]:
    paths = []
    for index, clip in enumerate(_uint8(value.videos)):
        path = prefix.with_name(f'{prefix.name}-{index}.gif')
        path.write_bytes(_gif(clip))
        paths.append(path)
    captions = prefix.with_suffix('.json')
    captions.write_text(json.dumps(list(value.captions)))
    return [*paths, captions]


@_write_preview.register
def _(value: TextSamples, prefix: Path) -> list[Path]:
    path = prefix.with_suffix('.json')
    path.write_text(json.dumps({'prompt': value.prompt, 'texts': list(value.texts),
                                'tokens': _home(value.tokens).tolist()}))
    return [path]


@_write_preview.register
def _(value: Representations, prefix: Path) -> list[Path]:
    path = prefix.with_suffix('.npz')
    np.savez(path, features=_home(value.features), labels=_home(value.labels))
    return [path]


@_write_preview.register
def _(value: TokenScores, prefix: Path) -> list[Path]:
    path = prefix.with_suffix('.npz')
    np.savez(path, losses=_home(value.losses), weights=_home(value.weights))
    return [path]


class MLflowTracker(_OwnedTracker):
    """An MLflow run in `experiment`, opened on the first value logged into it.

    Scalars are the run's metrics and a preview is uploaded as the files the
    local renderers write. A record becomes a JSON artifact rather than a
    parameter, because MLflow refuses a second value for a parameter and a
    run reports several records under one type.
    """

    def __init__(self, experiment: str, name: str | None = None, *, uri: str | None = None):
        self.experiment = experiment
        self.name = name
        self.uri = uri
        self._run = None
        self._closed = False
        self._status = 'FINISHED'

    @property
    def run(self) -> tuple[MlflowClient, str]:
        """The client and the run id, creating the experiment and the run once."""
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
        client.log_batch(run, metrics=[Metric(name, float(value), moment, step)
                                       for name, value in scalars.items()])

    def artifact(self, value: object, step: int) -> None:
        client, run = self.run
        if isinstance(value, RECORD_TYPES):
            client.log_dict(run, {'step': step, 'value': json_value(value)},
                            f'records/{type(value).__name__}-{step}.json')
            if isinstance(value, FitEnded):
                self._status = 'FINISHED' if value.status == 'completed' else 'FAILED'
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
        if exc is not None:
            self._status = 'FAILED'
        super().__exit__(exc_type, exc, tb)


class TensorBoardTracker(_OwnedTracker):
    """A TensorBoard event file in `directory`, opened on the first value
    logged into it.

    Scalars are scalar summaries and a record is the JSON its journal row
    holds, as a text summary under `reporting/<type>`. Previews render with
    `_summarize`.
    """

    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        self._writer = None
        self._closed = False

    @property
    def writer(self) -> EventFileWriter:
        """The event-file writer, opening `directory` once."""
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

    def artifact(self, value: object, step: int) -> None:
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
    """A text-plugin summary value: the string tensor a text tab reads."""
    from tensorboard.compat.proto import (summary_pb2, tensor_pb2, tensor_shape_pb2, types_pb2)

    return summary_pb2.Summary.Value(
        tag=tag,
        metadata=summary_pb2.SummaryMetadata(
            plugin_data=summary_pb2.SummaryMetadata.PluginData(plugin_name='text')),
        tensor=tensor_pb2.TensorProto(
            dtype=types_pb2.DT_STRING, string_val=[payload.encode()],
            tensor_shape=tensor_shape_pb2.TensorShapeProto(
                dim=[tensor_shape_pb2.TensorShapeProto.Dim(size=1)])))


def _picture(tag: str, encoded: bytes, frame: np.ndarray) -> Summary.Value:
    """An image-plugin summary value from encoded bytes and the frame in them."""
    from tensorboard.compat.proto.summary_pb2 import Summary

    height, width, channels = frame.shape
    return Summary.Value(tag=tag, image=Summary.Image(
        height=height, width=width, colorspace=channels, encoded_image_string=encoded))


@functools.singledispatch
def _summarize(value: object) -> Summary:
    """The TensorBoard summary for an explicitly reported artifact."""
    raise TypeError(f'TensorBoardTracker has no renderer for {type(value).__name__}')


# A registered renderer cannot annotate its return: singledispatch resolves a
# registration's annotations, and the proto module is imported where it is
# used rather than above.
@_summarize.register
def _(value: ImageGrid):
    from tensorboard.compat.proto.summary_pb2 import Summary

    images = [_picture(f'val/samples/{index}', _png(image), image)
              for index, image in enumerate(_uint8(value.images))]
    return Summary(value=[*images,
                          _text('val/samples/captions', json.dumps(list(value.captions)))])


@_summarize.register
def _(value: VideoGrid):
    from tensorboard.compat.proto.summary_pb2 import Summary

    # TensorBoard has no video summary; its image plugin animates a GIF.
    clips = [_picture(f'val/samples/{index}', _gif(clip), clip[0])
             for index, clip in enumerate(_uint8(value.videos))]
    return Summary(value=[*clips,
                          _text('val/samples/captions', json.dumps(list(value.captions)))])


@_summarize.register
def _(value: TextSamples):
    from tensorboard.compat.proto.summary_pb2 import Summary

    texts = value.texts or tuple(str(row.tolist()) for row in _home(value.tokens))
    return Summary(value=[_text('val/samples', json.dumps(
        {'prompt': value.prompt, 'texts': list(texts)}))])


@_summarize.register
def _(value: Representations):
    from tensorboard.compat.proto.summary_pb2 import HistogramProto, Summary

    # The per-dimension spread across the batch, as WandbTracker draws it.
    spread = np.std(_home(value.features).astype(np.float32), axis=0).astype(np.float64)
    counts, edges = np.histogram(spread, bins=30)
    return Summary(value=[Summary.Value(tag='val/representation_std', histo=HistogramProto(
        min=spread.min(), max=spread.max(), num=spread.size, sum=spread.sum(),
        sum_squares=(spread ** 2).sum(), bucket_limit=edges[1:].tolist(),
        bucket=counts.tolist()))])


class Trackers(_OwnedTracker):
    """Fan out without dropping reports; attempt every sink, raise the first failure."""

    def __init__(self, *trackers: Tracker):
        self.trackers = trackers
        self._closed = False

    def _each(self, operation) -> None:
        if self._closed:
            raise RuntimeError('Trackers is closed')
        error = None
        for tracker in self.trackers:
            try:
                operation(tracker)
            except BaseException as failure:
                if error is None:
                    error = failure
                else:
                    error.add_note(f'{type(tracker).__name__} also failed: {failure!r}')
        if error is not None:
            raise error

    def log(self, scalars: Mapping[str, float], step: int) -> None:
        self._each(lambda tracker: tracker.log(scalars, step))

    def artifact(self, value: object, step: int) -> None:
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

