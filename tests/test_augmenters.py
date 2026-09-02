"""Image augmenter tests: the albumentations migration, mode by mode.

Ground truth, measured against torchvision 0.28 in a throwaway venv: the
previous torchvision v2 pipelines returned plain numpy arrays UNCHANGED
(`v2.RandomHorizontalFlip(p=1.0)(img) is img`), and both image sources feed
cv2-decoded uint8 HWC numpy arrays. Augmentation was therefore a silent
no-op before; it is live now, which is a training data distribution change,
not parity.

These tests run in the bare venv: torchvision is blocked in sys.modules to
prove nothing in the module's import or construction path reaches it.
"""

import hashlib
import os
import random
import struct
import subprocess
import sys
from pathlib import Path

from absl import flags
import cv2
import grain.python as pygrain
import numpy as np
import pytest

# grain's worker processes read absl flags; a test that never ran absl.app
# would trip UnparsedFlagAccessError at any worker_count > 0.
if not flags.FLAGS.is_parsed():
    flags.FLAGS.mark_as_parsed()

# Any import of torchvision from here on raises ImportError; the module under
# test must import, construct and augment regardless.
sys.modules["torchvision"] = None

from dew.data.sources import images  # noqa: E402
from dew.data.sources.images import ImageGCSAugmenter, ImageTFDSAugmenter  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]

LABELS = ("pink rose", "yellow tulip", "white lily", "blue orchid")
SCALE = 48
DRAWS = 32
RECORDS = 16


class _StubTokenizer:
    """Offline stand-in for the CLIP tokenizer the real transform builds.

    The ids are a digest of the caption, so a caption stays comparable after a
    trip through grain's worker processes.
    """

    def __init__(self, tensor_type="np"):
        self.tensor_type = tensor_type

    def __call__(self, caption):
        digest = hashlib.blake2s(caption.encode(), digest_size=8).digest()
        return {
            "input_ids": np.frombuffer(digest, np.int32)[None, :],
            "attention_mask": np.ones((1, 2), np.int32),
        }


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
        packed += struct.pack("=I", len(key_bytes)) + key_bytes
        packed += struct.pack("=I", len(value)) + value
    return bytes(packed)


def _element_for(kind, seed=0):
    image = _synthetic_image(seed)
    if kind == "tfds":
        return {"image": image, "label": 2}
    encoded = cv2.imencode(".jpg", image)[1].tobytes()
    return _pack_records({"jpg": encoded, "txt": b"a yellow tulip"})


def _make_transform(kind, labels_file, monkeypatch):
    monkeypatch.setattr(images, "AutoTextTokenizer", _StubTokenizer)
    if kind == "tfds":
        augmenter = ImageTFDSAugmenter(label_path=str(labels_file))
    else:
        augmenter = ImageGCSAugmenter()
    return augmenter.create_transform(image_scale=SCALE)()


def _resized(kind, element):
    """The deterministic prefix of random_map: decode, convert, resize.
    For GCS, `element` is still the packed byte blob the transform unpacks."""
    if kind == "tfds":
        image = element["image"]
    else:
        records = images.unpack_dict_of_byte_arrays(element)
        decoded = cv2.imdecode(
            np.frombuffer(records["jpg"], dtype=np.uint8), cv2.IMREAD_UNCHANGED
        )
        image = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
    return cv2.resize(image, (SCALE, SCALE), interpolation=cv2.INTER_AREA)


def _record_rng(key):
    """grain's samplers derive one Philox generator per record."""
    return np.random.Generator(np.random.Philox(key=key))


# ---------------------------------------------------------------------------------
# torchvision is gone from the import and construction paths
# ---------------------------------------------------------------------------------

def test_module_imports_and_constructs_without_torchvision(tmp_path):
    """The old create_transform imported torchvision at call time; the whole
    point of the migration is that a base install (no torchvision) works."""
    labels_file = _write_labels(tmp_path)
    script = "\n".join([
        "import sys",
        "sys.modules['torchvision'] = None",
        "",
        "import numpy as np",
        "import grain.python as pygrain",
        "from dew.data.sources.images import (",
        "    ImageGCSAugmenter, ImageTFDSAugmenter, augment_image, image_augmentations,",
        ")",
        "",
        "labels = sys.argv[1]",
        "image = np.zeros((9, 11, 3), dtype=np.uint8)",
        "for mode in ('none', 'flip_only', 'flip_jitter'):",
        "    import os",
        "    os.environ['FLAXDIFF_AUGMENT_MODE'] = mode",
        "    classes = (",
        "        ImageTFDSAugmenter(label_path=labels).create_transform(image_scale=32),",
        "        ImageGCSAugmenter().create_transform(image_scale=32),",
        "    )",
        "    assert all(issubclass(c, pygrain.RandomMapTransform) for c in classes)",
        "    out = augment_image(image_augmentations(), image, np.random.default_rng(0))",
        "    assert out.dtype == np.uint8 and out.shape == image.shape",
        "print('ok')",
    ])
    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT / "src"))
    result = subprocess.run(
        [sys.executable, "-c", script, str(labels_file)],
        capture_output=True, text=True, env=env, cwd=REPO_ROOT,
    )
    # the GCS augmenter prints its interpolation method; only the final
    # verdict line matters here
    assert result.stdout.strip().splitlines()[-1] == "ok"
    assert "torchvision" not in result.stderr


def test_created_transforms_receive_grains_record_rng(tmp_path, monkeypatch):
    """grain only hands a per-record rng to RandomMapTransform subclasses; the
    per-record seeding contract depends on that dispatch."""
    monkeypatch.setenv("FLAXDIFF_AUGMENT_MODE", "flip_jitter")
    labels_file = _write_labels(tmp_path)
    for cls in (
        ImageTFDSAugmenter(label_path=str(labels_file)).create_transform(image_scale=SCALE),
        ImageGCSAugmenter().create_transform(image_scale=SCALE),
    ):
        assert issubclass(cls, pygrain.RandomMapTransform)


# ---------------------------------------------------------------------------------
# FLAXDIFF_AUGMENT_MODE contract
# ---------------------------------------------------------------------------------

@pytest.mark.parametrize("kind", ["tfds", "gcs"])
def test_none_mode_returns_the_resized_image_bit_identical(kind, tmp_path, monkeypatch):
    monkeypatch.setenv("FLAXDIFF_AUGMENT_MODE", "none")
    labels_file = _write_labels(tmp_path)
    element = _element_for(kind)
    transform = _make_transform(kind, labels_file, monkeypatch)

    out = transform.random_map(element, _record_rng(0))

    assert out["image"].dtype == np.uint8
    np.testing.assert_array_equal(out["image"], _resized(kind, element))
    # the surrounding record wiring stays intact
    assert "input_ids" in out["text"] and "attention_mask" in out["text"]


@pytest.mark.parametrize("kind", ["tfds", "gcs"])
def test_flip_only_mode_returns_only_the_image_or_its_mirror(kind, tmp_path, monkeypatch):
    monkeypatch.setenv("FLAXDIFF_AUGMENT_MODE", "flip_only")
    labels_file = _write_labels(tmp_path)
    element = _element_for(kind)
    transform = _make_transform(kind, labels_file, monkeypatch)

    base = _resized(kind, element)
    mirror = base[:, ::-1, :]
    outcomes = set()
    for i in range(DRAWS):
        out = transform.random_map(element, _record_rng(i))["image"]
        assert out.dtype == np.uint8
        assert out.shape == base.shape
        assert np.array_equal(out, base) or np.array_equal(out, mirror)
        outcomes.add("same" if np.array_equal(out, base) else "mirror")

    assert outcomes == {"same", "mirror"}


@pytest.mark.parametrize("kind", ["tfds", "gcs"])
def test_flip_jitter_mode_keeps_shape_and_dtype_and_changes_statistics(kind, tmp_path, monkeypatch):
    monkeypatch.setenv("FLAXDIFF_AUGMENT_MODE", "flip_jitter")
    labels_file = _write_labels(tmp_path)
    element = _element_for(kind)
    transform = _make_transform(kind, labels_file, monkeypatch)

    base = _resized(kind, element)
    mirror = base[:, ::-1, :]
    outputs = [transform.random_map(element, _record_rng(i))["image"]
               for i in range(DRAWS)]

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


@pytest.mark.parametrize("kind", ["tfds", "gcs"])
def test_unset_mode_defaults_to_flip_jitter(kind, tmp_path, monkeypatch):
    monkeypatch.delenv("FLAXDIFF_AUGMENT_MODE", raising=False)
    labels_file = _write_labels(tmp_path)
    element = _element_for(kind)
    transform = _make_transform(kind, labels_file, monkeypatch)

    base = _resized(kind, element)
    mirror = base[:, ::-1, :]
    outputs = [transform.random_map(element, _record_rng(i))["image"]
               for i in range(DRAWS)]
    assert any(not np.array_equal(o, base) and not np.array_equal(o, mirror)
               for o in outputs)


# ---------------------------------------------------------------------------------
# Seeding: augmentation follows the record rng, nothing else
# ---------------------------------------------------------------------------------

@pytest.mark.parametrize("kind", ["tfds", "gcs"])
def test_augmentation_and_caption_repeat_from_the_same_record_rng(kind, tmp_path, monkeypatch):
    """The same record must get the same augmentation and the same caption
    whatever the worker or process count, and neither global RNG may move."""
    monkeypatch.setenv("FLAXDIFF_AUGMENT_MODE", "flip_jitter")
    labels_file = _write_labels(tmp_path)
    element = _element_for(kind)
    first = _make_transform(kind, labels_file, monkeypatch)
    other = _make_transform(kind, labels_file, monkeypatch)

    numpy_state = np.random.get_state()[1]
    python_state = random.getstate()
    one = first.random_map(element, _record_rng(7))
    # an unrelated record in between must not disturb the next one
    other.random_map(_element_for(kind, seed=1), _record_rng(999))
    two = other.random_map(element, _record_rng(7))

    np.testing.assert_array_equal(one["image"], two["image"])
    np.testing.assert_array_equal(one["text"]["input_ids"], two["text"]["input_ids"])
    assert np.array_equal(numpy_state, np.random.get_state()[1])
    assert random.getstate() == python_state


def test_the_caption_template_comes_from_the_record_rng(tmp_path):
    """A module-global random.choice picked the template, so a record's caption
    moved with the worker and process count while its image did not."""
    labelizer = images.labelizer_oxford_flowers102(str(_write_labels(tmp_path)))
    element = {"label": 1}

    captions = [labelizer(element, _record_rng(draw)) for draw in range(DRAWS)]

    assert len(set(captions)) > 1  # the rng really does pick the template
    assert set(captions) <= {t.format(LABELS[1]) for t in images.PROMPT_TEMPLATES}
    assert labelizer(element, _record_rng(3)) == captions[3]


# ---------------------------------------------------------------------------------
# Loader level: a record does not depend on how many workers produced it
# ---------------------------------------------------------------------------------

def _by_record(transform_cls, source, worker_count):
    """label -> (image, caption ids) for every record of one epoch.

    Keyed by label rather than compared batch for batch: grain gives each
    worker its own slice of the index stream, so batch composition follows
    worker_count while record content must not.
    """
    sampler = pygrain.IndexSampler(num_records=len(source), shuffle=True, seed=7,
                                   num_epochs=1, shard_options=pygrain.NoSharding())
    loader = pygrain.DataLoader(
        data_source=source,
        sampler=sampler,
        operations=[transform_cls(), pygrain.Batch(4, drop_remainder=True)],
        worker_count=worker_count,
    )
    records = {}
    for batch in loader:
        for position, label in enumerate(batch["label"]):
            records[int(label)] = (batch["image"][position].tobytes(),
                                   batch["text"]["input_ids"][position].tobytes())
    return records


def test_a_record_comes_out_the_same_with_and_without_worker_processes(tmp_path, monkeypatch):
    monkeypatch.setenv("FLAXDIFF_AUGMENT_MODE", "flip_jitter")
    monkeypatch.setattr(images, "AutoTextTokenizer", _StubTokenizer)
    # One label per record, so a batch says which records it carries.
    labels_file = tmp_path / "indexed_labels.txt"
    labels_file.write_text("\n".join(f"flower{i:02d}" for i in range(RECORDS)) + "\n")
    source = [{"image": _synthetic_image(i), "label": i} for i in range(RECORDS)]
    transform_cls = ImageTFDSAugmenter(
        label_path=str(labels_file)).create_transform(image_scale=SCALE)

    serial = _by_record(transform_cls, source, worker_count=0)
    parallel = _by_record(transform_cls, source, worker_count=2)

    assert sorted(serial) == list(range(RECORDS))
    assert serial == parallel
