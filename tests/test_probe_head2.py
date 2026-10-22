"""Probe for the second dedupe head: image resizing and cropping (1086), host and device augmentation
(1073) and prompt reward context (1087), each output hashed, printed as JSON for an old/new comparison."""
import hashlib
import json

import numpy as np

from dew.data import images


def digest(value) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    return hashlib.sha256(array.dtype.str.encode() + str(array.shape).encode() + array.tobytes()).hexdigest()[:16]


def test_probe():
    rng = np.random.default_rng(0)
    out = {}
    pictures = [rng.integers(0, 256, shape, np.uint8) for shape in
                [(37, 53, 3), (64, 64, 3), (20, 90, 3), (90, 20, 3), (13, 13, 3), (33, 47, 1), (48, 32, 3)]]
    for index, image in enumerate(pictures):
        row = {"image": image}
        for name, step in [("resize-none-24", images.Resize(24)), ("resize-none-100", images.Resize(100)),
                           ("resize-none-rect", images.Resize((30, 70))),
                           *((f"resize-{method}", images.Resize(28, method)) for method in
                             ("area", "bilinear", "bicubic", "nearest", "lanczos")),
                           ("fit-32", images.Fit(32)), ("fit-200", images.Fit(200)),
                           ("center-16", images.CenterCrop(16)), ("center-80", images.CenterCrop(80)),
                           ("bucket", images.Bucket(((24, 32), (32, 24), (28, 28))))]:
            out[f"{index}-{name}"] = {key: digest(value) for key, value in step.map(row).items()}
    from dew.data.video import fit_frames
    frames = rng.integers(0, 256, (3, 40, 30, 3), np.uint8)
    out["video"] = digest(fit_frames(frames, 16))
    from dew.data.image_augmentation import apply_host, augment_batch, draw_host, jitter_host
    for index, image in enumerate(pictures[:5]):
        generator = np.random.default_rng(index)
        parameters = draw_host(generator, image.shape, flip=True, jitter=True, crop_scale=(0.5, 1.0))
        out[f"host-draw-{index}"] = [digest(part) for part in parameters]
        pixels = apply_host(image, parameters, 24)
        out[f"host-apply-{index}"] = digest(pixels)
        out[f"host-apply-rounded-{index}"] = digest(np.rint(pixels).astype(np.uint8))
        jittered = images.ColorJitter().random_map({"image": image}, np.random.default_rng(index))
        out[f"colorjitter-{index}"] = digest(jittered["image"])
        out[f"jitter-host-{index}"] = digest(jitter_host(image.astype(np.float32), (1.3, 0.9, 1.6), (2, 0, 1)))
    batch = np.stack([rng.integers(0, 256, (32, 32, 3), np.uint8) for _ in range(4)])
    keys = rng.integers(0, 2**31, (4, 2), np.uint32)
    out["device"] = digest(augment_batch(batch, keys, size=16, flip=True, jitter=True, crop_scale=(0.5, 1.0)))
    from dew.data.prompts import RewardWidth
    from dew.data.steps.chat import RenderPrompt
    rows = [{"prompt": "hi", "data_source": "gsm8k", "ground_truth": "4"},
            {"prompt": "x", "reward_model": {"ground_truth": {"a": [1, 2]}}, "extra_info": {"k": "v"},
             "ability": "math"},
            {"prompt": "y", "data_source": None, "extra_info": "plain", "ability": "code"}]
    out["reward-width"] = RewardWidth()(rows)
    render = RenderPrompt("byte", max_prompt_len=8, reward_bytes=64)
    out["render"] = [{key: digest(value) for key, value in render.map(dict(row)).items()} for row in rows]
    print("PROBE " + json.dumps(out, sort_keys=True))
