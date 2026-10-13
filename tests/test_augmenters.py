"""Image transform tests: the augmentation field, mode by mode.

Ground truth, measured against torchvision 0.28 in a throwaway venv: the
previous torchvision v2 pipelines returned plain numpy arrays UNCHANGED
(`v2.RandomHorizontalFlip(p=1.0)(img) is img`), and both image sources feed
cv2-decoded uint8 HWC numpy arrays. Augmentation was therefore a silent
no-op before; it is live now, which is a training data distribution change,
not parity.

These tests run in the bare venv: a subprocess test proves nothing in the
module's import or construction path reaches torchvision.
"""

import os
import random
import struct
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
from absl import flags

# grain's worker processes read absl flags; a test that never ran absl.app
# would trip UnparsedFlagAccessError at any worker_count > 0.
if not flags.FLAGS.is_parsed():
    flags.FLAGS.mark_as_parsed()

from dew.config.data import Augment, ImageData, loader
from dew.data import Loading, Memory, images
from dew.data.dataset import DataPartition
from dew.data.sources.codec import unpack_dict_of_byte_arrays
from dew.data.steps.image import EncodeCaptions

REPO_ROOT = Path(__file__).resolve().parents[1]

LABELS = ("pink rose", "yellow tulip", "white lily", "blue orchid")
SCALE = 48
DRAWS = 32
def _write_labels(tmp_path):
    labels = tmp_path / "labels.txt"
    labels.write_text("\n".join(LABELS) + "\n")
    return labels


def _synthetic_image(seed=0):
    return np.random.RandomState(seed).randint(0, 256, (37, 53, 3), dtype=np.uint8)


def _pack_records(records):
    """Mirror of unpack_dict_of_byte_arrays' wire format."""
    packed = bytearray()
    for key, value in records.items():
        key_bytes = key.encode("utf-8")
        packed += struct.pack("I", len(key_bytes)) + key_bytes
        packed += struct.pack("I", len(value)) + value
    return bytes(packed)


CAPTION_TEMPLATES = ("a photo of a {}", "a photo of a {} flower", "This is a photo of a {}")
"""More than one, so a test can see which the record's rng drew."""

FLIP_JITTER = images.Augment(flip=True, jitter=True)


def _element_for(kind, seed=0):
    image = _synthetic_image(seed)
    if kind == "tfds":
        return {"image": image, "label": 2}
    encoded = cv2.imencode(".jpg", image)[1].tobytes()
    return _pack_records({"jpg": encoded, "txt": b"a yellow tulip"})


def _spec(kind, augment=FLIP_JITTER):
    """Labelled pixels or encoded ArrayRecord bytes through the host recipe."""
    return ImageData(source=Memory([_element_for(kind)] * DRAWS), image_size=SCALE,
                     caption_templates=CAPTION_TEMPLATES if kind == "tfds" else (),
                     classes=LABELS if kind == "tfds" else (),
                     augment=None if augment is None else Augment(augment.flip, augment.jitter,
                                                                  backend="host"))


def _read(spec, *, seed=0, batch=8):
    data = loader(spec, batch=batch, read="indexed", seed=seed,
                  loading=Loading(workers=0, threads=1, read_buffer=2),
                  host=(EncodeCaptions(lambda captions: {"caption": np.asarray(captions)}),))
    stream = data.train(DataPartition())
    try:
        return [{name: column[index] for name, column in row.items()}
                for _ in range(data.steps_per_epoch) for row in (next(stream),)
                for index in range(batch)]
    finally:
        stream.close()


def _resized(kind, element):
    """The deterministic prefix of random_map: decode, convert, resize.
    For arrayrecord, `element` is still the packed byte blob the transform unpacks."""
    if kind == "tfds":
        image = element["image"]
    else:
        records = unpack_dict_of_byte_arrays(element)
        decoded = cv2.imdecode(
            np.frombuffer(records["jpg"], dtype=np.uint8), cv2.IMREAD_UNCHANGED
        )
        image = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
    return cv2.resize(image, (SCALE, SCALE), interpolation=cv2.INTER_AREA)


# ---------------------------------------------------------------------------------
# torchvision is gone from the import and construction paths
# ---------------------------------------------------------------------------------

def test_module_imports_and_constructs_without_torchvision(tmp_path):
    """A base install without torchvision imports, constructs and augments."""
    labels_file = _write_labels(tmp_path)
    script = "\n".join(
        [
            "import sys",
            "sys.modules['torchvision'] = None",
            "sys.modules['transformers'] = None",
            "",
            "import numpy as np",
            "from dew.config.data import ImageData",
            "from dew.data import images",
            "from dew.data.images import Augment, augment_image",
            "",
            "labels = sys.argv[1]",
            "image = np.zeros((9, 11, 3), dtype=np.uint8)",
            "for augment in (None, Augment(flip=True, jitter=False), Augment(flip=True, jitter=True)):",
            "    ImageData(source=(), labels=labels)",
            "    images.random_resized_crop(16), images.flip(), images.color_jitter()",
            "    out = augment_image(augment, image, np.random.default_rng(0))",
            "    assert out.dtype == np.uint8 and out.shape == image.shape",
            "print('ok')",
        ]
    )
    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT / "src"), JAX_PLATFORMS="cpu")
    result = subprocess.run(
        [sys.executable, "-c", script, str(labels_file)],
        capture_output=True, text=True, env=env, cwd=REPO_ROOT,
    )
    assert result.returncode == 0, result.stderr


def test_collecting_augmenters_preserves_reference_imports():
    """Collecting this file in a shared run must not poison the modules the
    rest of the suite imports. The old sys.modules['torchvision'] = None at
    module scope survived collection and broke every later import of
    torchvision or transformers in the same process."""
    script = "\n".join([
        "import pytest",
        "assert pytest.main(['--collect-only', '-q', 'tests/test_augmenters.py']) == 0",
        "from torchvision.io import decode_image",
        "from transformers import MixtralForCausalLM",
        "print('ok')",
    ])
    _fresh_process(script)


def _fresh_process(script, *args):
    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT / "src"), JAX_PLATFORMS="cpu")
    result = subprocess.run([sys.executable, "-c", script, *map(str, args)],
                            capture_output=True, text=True, env=env, cwd=REPO_ROOT)
    assert result.returncode == 0, (
        result.stdout + result.stderr)


# The images a loader's reader threads decode together, first import of cv2
# included: cv2's import is not safe to share out, since one that fails in one
# thread leaves the others a half-built module (`module 'cv2' has no attribute
# 'INTER_AREA'`). What the loader promises is that OpenCV is in place before
# its readers start, in the process that opens it and in a spawned worker.
_READERS = "\n".join([
    "import dataclasses, io, sys",
    "import numpy as np",
    "from PIL import Image",
    "from dew.config.data import ImageData, loader",
    "from dew.data import Loading",
    "from dew.data.dataset import DataPartition",
    "buffer = io.BytesIO()",
    "Image.fromarray(np.arange(40 * 40 * 3, dtype=np.uint8).reshape(40, 40, 3)).save(",
    "    buffer, format='PNG')",
    "PNG = buffer.getvalue()",
    "",
    "SPEC = ImageData(source=[{'image': PNG, 'caption': 'a picture'}] * 64, image_size=16, augment=None)",
])


def test_opening_an_image_loader_imports_opencv_before_its_readers_start():
    _fresh_process("\n".join([
        _READERS,
        "loading = Loading(workers=0, threads=32, read_buffer=64)",
        "data = loader(SPEC, batch=32, read='indexed', loading=loading)",
        "batch = next(iter(data.train(DataPartition())))",
        "assert batch['image'].shape == (32, 16, 16, 3)",
        "print('ok')",
    ]))


# ---------------------------------------------------------------------------------
# The augmentation field
# ---------------------------------------------------------------------------------

@pytest.mark.parametrize("kind", ["tfds", "gcs"])
def test_none_mode_returns_the_resized_image_bit_identical(kind):
    element = _element_for(kind)
    for out in _read(_spec(kind, None)):
        assert out["image"].dtype == np.uint8
        np.testing.assert_array_equal(out["image"], _resized(kind, element))
        if kind == "gcs":
            assert out["caption"] == "a yellow tulip"
        else:
            assert out["caption"] in {template.format(LABELS[2]) for template in CAPTION_TEMPLATES}


@pytest.mark.parametrize("kind", ["tfds", "gcs"])
def test_flip_only_mode_returns_only_the_image_or_its_mirror(kind):
    element = _element_for(kind)
    base = _resized(kind, element)
    mirror = base[:, ::-1, :]
    outcomes = set()
    for row in _read(_spec(kind, images.Augment(flip=True, jitter=False))):
        out = row["image"]
        assert out.dtype == np.uint8
        assert out.shape == base.shape
        assert np.array_equal(out, base) or np.array_equal(out, mirror)
        outcomes.add("same" if np.array_equal(out, base) else "mirror")

    assert outcomes == {"same", "mirror"}


@pytest.mark.parametrize("kind", ["tfds", "gcs"])
def test_flip_jitter_mode_keeps_shape_and_dtype_and_changes_statistics(kind):
    element = _element_for(kind)
    base = _resized(kind, element)
    mirror = base[:, ::-1, :]
    outputs = [row["image"] for row in _read(_spec(kind))]

    for out in outputs:
        assert out.dtype == np.uint8  # ColorJitter must not promote to float
        assert out.shape == base.shape

    # the jitter is live: some draw is neither the image nor its mirror
    assert any(not np.array_equal(o, base) and not np.array_equal(o, mirror)
               for o in outputs)
    # and records are not all augmented the same way
    assert not all(np.array_equal(o, outputs[0]) for o in outputs)
    # pixel statistics actually move
    assert any(abs(int(o.mean()) - int(base.mean())) > 1 for o in outputs)


# ---------------------------------------------------------------------------------
# Seeding: augmentation follows the record rng alone
# ---------------------------------------------------------------------------------

@pytest.mark.parametrize("kind", ["tfds", "gcs"])
def test_augmentation_and_caption_repeat_from_the_same_record_rng(kind):
    """The same record must get the same augmentation and the same caption
    whatever the worker or process count, and neither global RNG may move."""
    first = _spec(kind)
    other = _spec(kind)

    numpy_state = np.random.get_state()[1]
    python_state = random.getstate()
    one = _read(first, seed=7)
    # an unrelated record in between must not disturb the next one
    _read(other, seed=999)
    two = _read(other, seed=7)

    for left, right in zip(one, two, strict=True):
        np.testing.assert_array_equal(left["image"], right["image"])
        assert left["caption"] == right["caption"]
    assert np.array_equal(numpy_state, np.random.get_state()[1])
    assert random.getstate() == python_state


def test_a_record_with_no_caption_keeps_an_empty_host_caption():
    row = {"image": _synthetic_image(), "url": "x"}
    out = _read(ImageData(source=Memory([row]), image_size=SCALE, augment=None), batch=1)[0]
    assert out["caption"] == "" and out["url"] == "x"
    np.testing.assert_array_equal(out["image"], images.resize_image(row["image"], SCALE))


def test_a_hub_record_captions_from_the_record_and_reads_no_label_file():
    """The same image transform serves a hub dataset: what changes is where the
    caption comes from, and that a caption dataset has no class index."""
    element = {"image": _synthetic_image(0), "caption": "a yellow tulip"}
    out = _read(ImageData(source=Memory([element]), image_size=SCALE, augment=None), batch=1)[0]

    np.testing.assert_array_equal(
        out["image"], cv2.resize(element["image"], (SCALE, SCALE),
                                 interpolation=cv2.INTER_AREA))
    assert out["caption"] == "a yellow tulip"
    assert "label" not in out


def test_a_host_crop_reaches_the_output_size_without_a_flip_or_a_jitter():
    """Crop geometry is its own: a staged image crops and resizes to the output size whether or
    not a flip or a jitter rides along, and a crop scale at equal sizes still crops."""
    image = np.random.default_rng(0).integers(0, 256, (2 * SCALE, 2 * SCALE, 3), dtype=np.uint8)
    for crop in ((1.0, 1.0), (0.8, 1.0)):
        spec = ImageData(source=Memory([{"image": image}]), image_size=SCALE, staging_size=2 * SCALE,
                         augment=Augment(flip=False, jitter=False, crop_scale=crop, backend="host"))
        out = _read(spec, batch=1)[0]
        assert out["image"].shape == (SCALE, SCALE, 3)
    small = image[:SCALE, :SCALE]
    spec = ImageData(source=Memory([{"image": small}]), image_size=SCALE,
                     augment=Augment(flip=False, jitter=False, crop_scale=(0.5, 0.5), backend="host"))
    cropped = _read(spec, batch=1)[0]
    assert cropped["image"].shape == (SCALE, SCALE, 3) and not np.array_equal(cropped["image"], small)
