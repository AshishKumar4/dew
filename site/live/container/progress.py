"""Progress reports for the landing page's sampler, while a sampling cell runs.

The shared model service wraps its pinned pipeline in `Reporting`, which samples exactly as the pipeline does, with the solver it is handed
wrapped so each step also sends its clean prediction to the host. While the
compiled program runs, the kernel's main thread displays one output per solver
step, whose text is {"dew-progress": {"step": k, "steps": n}} with n the
cell's step count and whose PNG is that prediction mapped linearly from
latents to a small RGB image, then {"dew-progress": {"stage": "decode"}} while
the model's final clean prediction and the VAE decode run. The
page shows these as the run's progress; any other display is the cell's own.
"""

from __future__ import annotations

import base64
import io
import json
import queue
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import jax
import numpy as np
from PIL import Image

# Stable Diffusion's VAE latents (sd-vae-ft-mse, as the model is trained on)
# to RGB in [-1, 1], one row per latent channel: the least-squares linear fit
# that ComfyUI and Diffusers' previews use for SD 1.x latents.
LATENT_RGB = np.array([[0.3512, 0.2297, 0.3227],
                       [0.3250, 0.4974, 0.2350],
                       [-0.2829, 0.1762, 0.2721],
                       [-0.2120, -0.2616, -0.7177]], np.float32)

_steps: queue.SimpleQueue[np.ndarray] = queue.SimpleQueue()


def _report(denoised: Any) -> None:
    """Runs on the runtime's callback thread; the main thread displays."""
    _steps.put(np.asarray(denoised[0], np.float32))


@dataclass(frozen=True)
class ReportingSolver:
    """`inner`, sending each step's clean prediction to `_report`."""

    inner: Any

    def init(self, x, times, process, *, key):
        return self.inner.init(x, times, process, key=key)

    def step(self, x, t, t_next, denoised, eps, state, key, process, denoise, /):
        jax.debug.callback(_report, denoised)
        return self.inner.step(x, t, t_next, denoised, eps, state, key, process, denoise)


def preview_png(latent: np.ndarray) -> str | None:
    """A latent `[H, W, 4]` as a base64 PNG of the same size; None for other shapes."""
    if latent.ndim != 3 or latent.shape[-1] != LATENT_RGB.shape[0]:
        return None
    rgb = np.clip((latent @ LATENT_RGB + 1) * 127.5, 0, 255).astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(rgb).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def _show(report: dict[str, Any], png: str | None = None) -> None:
    from IPython.display import display  # the kernel's; nothing else needs IPython

    data = {"text/plain": json.dumps({"dew-progress": report})}
    if png is not None:
        data["image/png"] = png
    display(data, raw=True)


class Reporting:
    """`pipe`, reporting each sampling step while a call runs."""

    def __init__(self, pipe: Any) -> None:
        self.pipe = pipe

    def __getattr__(self, name: str) -> Any:
        return getattr(self.pipe, name)

    def __call__(self, prompts, *, steps: int | None = None, solver=None, decode: bool = True, **kw):
        count = self.pipe.steps if steps is None else steps
        # The solver steps between the grid's points; the model's last call, the
        # clean prediction at the final point, runs with the decode.
        process, times = self.pipe.prepared_process(count)
        walked = len(process.times(count) if times is None else times) - 1
        reporting = ReportingSolver(self.pipe.solver if solver is None else solver)
        while not _steps.empty():
            _steps.get_nowait()
        # A call's first run of a program with host callbacks returns only when the
        # program has finished, so the call runs on its own thread and this one displays.
        with ThreadPoolExecutor(1) as pool:
            call = pool.submit(self.pipe, prompts, steps=steps, solver=reporting, decode=decode, **kw)
            done = 0
            while done < walked:
                try:
                    latent = _steps.get(timeout=0.2)
                except queue.Empty:
                    if call.done() and (call.exception() is not None or _ready(call.result())):
                        break
                    continue
                done += 1
                _show({"step": done, "steps": count}, preview_png(latent))
            if decode and done == walked:
                _show({"stage": "decode"})
            return call.result()


def _ready(result: Any) -> bool:
    return all(leaf.is_ready() for leaf in jax.tree.leaves(result))


class StalePage(ValueError):
    """A page requests a model revision absent from the offline runtime."""

    def _render_traceback_(self) -> list[str]:
        return [str(self)]
