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
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
from data_support import CAPTION_TEMPLATES, FLIP_JITTER, keep_captions, parse_grain_flags

from dew.data import Loading, Memory, images, presets
from dew.data.compile import compile
from dew.data.dataset import DataPartition
from dew.data.images import EncodeCaptions
from dew.data.sources.codec import pack_dict_of_byte_arrays, unpack_dict_of_byte_arrays

parse_grain_flags()

REPO_ROOT = Path(__file__).resolve().parents[1]

LABELS = ("pink rose", "yellow tulip", "white lily", "blue orchid")
SCALE = 48
DRAWS = 32


def _synthetic_image(seed=0):
    return np.random.RandomState(seed).randint(0, 256, (37, 53, 3), dtype=np.uint8)




def _element_for(kind, seed=0):
    image = _synthetic_image(seed)
    if kind == "tfds":
        return {"image": image, "label": 2}
    encoded = cv2.imencode(".jpg", image)[1].tobytes()
    return pack_dict_of_byte_arrays({"jpg": encoded, "txt": b"a yellow tulip"})


def _spec(kind):
    """Labelled pixels, captioned from CAPTION_TEMPLATES by their label, or encoded ArrayRecord bytes
    captioned by their own text: `_read`'s rows and caption."""
    rows = [_element_for(kind)] * DRAWS
    if kind == "tfds":
        return {"rows": rows, "caption": images.caption(templates=CAPTION_TEMPLATES, classes=LABELS)}
    return {"rows": rows}


def _read(rows, *, augment=FLIP_JITTER, caption=None, staging_size=None, seed=0, batch=8):
    """An epoch of `rows` resized to SCALE pixels, staged at `staging_size` where given, captioned
    by `caption`, else by their own caption or text, and through the host `augment` steps, read by
    index in a run's order under `seed`, a record at a time with its caption as words."""
    chain = Memory(rows).shuffle(seed).repeat().pipe(
        presets.images, SCALE, batch=batch, augment=augment, staging_size=staging_size, rejects=False,
        caption=caption or images.caption(), host=(EncodeCaptions(keep_captions),))
    data = compile(chain, read="indexed", seed=seed, loading=Loading(workers=0, threads=1, read_buffer=2))
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

def test_module_imports_and_constructs_without_torchvision():
    """A base install without torchvision imports, constructs and augments."""
    script = "\n".join(
        [
            "import sys",
            "sys.modules['torchvision'] = None",
            "sys.modules['transformers'] = None",
            "",
            "import numpy as np",
            "from dew.data import Memory, images, presets",
            "",
            "image = np.zeros((9, 11, 3), dtype=np.uint8)",
            "steps = (images.Flip(), images.ColorJitter())",
            "presets.images(Memory([{'image': image}]), 8, batch=1, augment=steps,",
            "               caption=images.caption(templates=('a {}',), classes=('rose',)))",
            "for stage in (images.Flip(), images.ColorJitter()):",
            "    out = stage.random_map({'image': image}, np.random.default_rng(0))['image']",
            "    assert out.dtype == np.uint8 and out.shape == image.shape",
            "print('ok')",
        ]
    )
    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT / "src"), JAX_PLATFORMS="cpu")
    result = subprocess.run(
        [sys.executable, "-c", script],
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
    "from dew.data import Loading, Memory, presets",
    "from dew.data.compile import compile",
    "from dew.data.dataset import DataPartition",
    "buffer = io.BytesIO()",
    "Image.fromarray(np.arange(40 * 40 * 3, dtype=np.uint8).reshape(40, 40, 3)).save(",
    "    buffer, format='PNG')",
    "PNG = buffer.getvalue()",
    "",
    "CHAIN = Memory([{'image': PNG, 'caption': 'a picture'}] * 64).shuffle(0).repeat().pipe(",
    "    presets.images, 16, batch=32, rejects=False)",
])


def test_opening_an_image_loader_imports_opencv_before_its_readers_start():
    _fresh_process("\n".join([
        _READERS,
        "loading = Loading(workers=0, threads=32, read_buffer=64)",
        "data = compile(CHAIN, read='indexed', loading=loading)",
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
    for out in _read(**_spec(kind), augment=False):
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
    for row in _read(**_spec(kind), augment=images.Flip()):
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
    outputs = [row["image"] for row in _read(**_spec(kind))]

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
    numpy_state = np.random.get_state()[1]
    python_state = random.getstate()
    one = _read(**_spec(kind), seed=7)
    # an unrelated record in between must not disturb the next one
    _read(**_spec(kind), seed=999)
    two = _read(**_spec(kind), seed=7)

    for left, right in zip(one, two, strict=True):
        np.testing.assert_array_equal(left["image"], right["image"])
        assert left["caption"] == right["caption"]
    assert np.array_equal(numpy_state, np.random.get_state()[1])
    assert random.getstate() == python_state


def test_a_record_with_no_caption_keeps_an_empty_host_caption():
    row = {"image": _synthetic_image(), "url": "x"}
    out = _read([row], augment=False, batch=1)[0]
    assert out["caption"] == "" and out["url"] == "x"
    np.testing.assert_array_equal(out["image"], images.resize_image(row["image"], SCALE))


def test_a_hub_record_captions_from_the_record_and_reads_no_label_file():
    """The same image transform serves a hub dataset: what changes is where the
    caption comes from, and that a caption dataset has no class index."""
    element = {"image": _synthetic_image(0), "caption": "a yellow tulip"}
    out = _read([element], augment=False, batch=1)[0]

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
        out = _read([{"image": image}], augment=images.RandomCrop(SCALE, scale=crop), staging_size=2 * SCALE,
                    batch=1)[0]
        assert out["image"].shape == (SCALE, SCALE, 3)
    small = image[:SCALE, :SCALE]
    cropped = _read([{"image": small}], augment=images.RandomCrop(SCALE, scale=(0.5, 0.5)), batch=1)[0]
    assert cropped["image"].shape == (SCALE, SCALE, 3) and not np.array_equal(cropped["image"], small)
