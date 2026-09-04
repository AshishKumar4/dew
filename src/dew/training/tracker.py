"""Where a run's numbers and artifacts go.

A `Tracker` is the one capability the trainer logs through. `WandbTracker`
renders each artifact type with a `functools.singledispatch` function, which
is the whole mechanism by which objectives return typed values and the
tracker draws them: a new artifact type registers a renderer, nothing else
changes. wandb is imported when the first value is logged, not before.
"""

from __future__ import annotations

import functools
from collections.abc import Mapping
from typing import Any, Protocol

import numpy as np

from dew.artifacts import ImageGrid, Representations, TextSamples, VideoGrid


class Tracker(Protocol):
    def log(self, scalars: Mapping[str, float], step: int) -> None: ...

    def artifact(self, value: Any, step: int) -> None: ...


def _home(array) -> np.ndarray:
    """`array` as numpy, refusing a shard of a global array.

    A tracker draws on process zero alone, and completing a global array needs
    every process, so an artifact arrives here already brought home by
    `dew.artifacts.host`. Refusing names the missing call instead of hanging in
    a collective one process entered by itself.
    """
    if not getattr(array, "is_fully_addressable", True):
        raise ValueError(
            "a tracker was handed a shard of a global array, which it cannot "
            "complete from one process: bring the artifact home with "
            "dew.artifacts.host on every process before drawing it")
    return np.asarray(array)


def _uint8(images) -> np.ndarray:
    """[-1, 1] floats as the bytes an image viewer reads."""
    return np.clip((_home(images).astype(np.float32) + 1.0) * 127.5, 0, 255).astype(np.uint8)


@functools.singledispatch
def render(value, step: int):
    """The wandb payload for one artifact, or NotImplemented for a type
    nothing draws (a metric reads it instead)."""
    return NotImplemented


@render.register
def _(value: ImageGrid, step: int):
    import wandb

    captions = list(value.captions) + [None] * (len(value.images) - len(value.captions))
    return {"val/samples": [wandb.Image(image, caption=caption)
                            for image, caption in zip(_uint8(value.images), captions)]}


@render.register
def _(value: VideoGrid, step: int):
    import wandb

    # wandb reads clips as [N, T, C, H, W].
    clips = np.transpose(_uint8(value.videos), (0, 1, 4, 2, 3))
    return {"val/samples": wandb.Video(clips, fps=10, caption=" | ".join(value.captions))}


@render.register
def _(value: TextSamples, step: int):
    import wandb

    texts = value.texts or tuple(str(row.tolist()) for row in _home(value.tokens))
    rows = [[index, value.prompt, text] for index, text in enumerate(texts)]
    return {"val/samples": wandb.Table(columns=["sample", "prompt", "text"], data=rows)}


@render.register
def _(value: Representations, step: int):
    import wandb

    # The per-dimension spread across the batch: the collapse view of a
    # representation, which goes to zero when the encoder stops telling
    # inputs apart.
    return {"val/representation_std": wandb.Histogram(
        np.std(_home(value.features).astype(np.float32), axis=0))}


class WandbTracker:
    """A Weights & Biases run, opened on the first value logged into it."""

    render = staticmethod(render)

    def __init__(self, project: str, name: str | None = None, *,
                 entity: str | None = None, config: Mapping[str, Any] | None = None,
                 offline: bool = False, id: str | None = None):
        self.project = project
        self.name = name
        self.entity = entity
        self.config = dict(config) if config is not None else None
        self.offline = offline
        self.id = id
        self._run = None

    @property
    def run(self):
        if self._run is None:
            import wandb

            self._run = wandb.init(
                project=self.project, name=self.name, entity=self.entity,
                config=self.config, id=self.id, resume="allow",
                mode="offline" if self.offline else None)
            self._run.define_metric("train/step")
            self._run.define_metric("train/*", step_metric="train/step")
            self._run.define_metric("val/*", step_metric="train/step")
        return self._run

    def log(self, scalars: Mapping[str, float], step: int) -> None:
        self.run.log({name: float(value) for name, value in scalars.items()}, step=step)

    def artifact(self, value: Any, step: int) -> None:
        payload = self.render(value, step)
        if payload is not NotImplemented:
            self.run.log(payload, step=step)

    def finish(self) -> None:
        if self._run is not None:
            self._run.finish()
            self._run = None
