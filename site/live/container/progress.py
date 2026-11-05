"""Progress reports for the landing page's sampler, while a sampling cell runs.

The shared model service streams each solver step of a served image request
(`dew.inference.ImageRequest.on_step`), and sends the guest's kernel, per
step, {"progress": {"step": k, "steps": n}} with n the cell's step count and
a PNG of that step's clean prediction mapped linearly from latents to a small
RGB image (`preview_png`), then {"progress": {"stage": "decode"}} while the
model's final clean prediction and the VAE decode run. The kernel displays
each as {"dew-progress": ...}, which the page shows as the run's progress.
"""

from __future__ import annotations

import base64
import io

import numpy as np
from PIL import Image

# Stable Diffusion's VAE latents (sd-vae-ft-mse, as the model is trained on)
# to RGB in [-1, 1], one row per latent channel: the least-squares linear fit
# that ComfyUI and Diffusers' previews use for SD 1.x latents.
LATENT_RGB = np.array([[0.3512, 0.2297, 0.3227],
                       [0.3250, 0.4974, 0.2350],
                       [-0.2829, 0.1762, 0.2721],
                       [-0.2120, -0.2616, -0.7177]], np.float32)


def preview_png(latent: np.ndarray) -> str | None:
    """A latent `[H, W, 4]` as a base64 PNG of the same size; None for other shapes."""
    if latent.ndim != 3 or latent.shape[-1] != LATENT_RGB.shape[0]:
        return None
    rgb = np.clip((latent @ LATENT_RGB + 1) * 127.5, 0, 255).astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(rgb).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


class StalePage(ValueError):
    """A page requests a model revision absent from the offline runtime."""

    def _render_traceback_(self) -> list[str]:
        return [str(self)]
