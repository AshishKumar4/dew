"""Image transform tests: the augmentation field, mode by mode.

Ground truth, measured against torchvision 0.28 in a throwaway venv: the
previous torchvision v2 pipelines returned plain numpy arrays UNCHANGED
(`v2.RandomHorizontalFlip(p=1.0)(img) is img`), and both image sources feed
cv2-decoded uint8 HWC numpy arrays. Augmentation was therefore a silent
no-op before; it is live now, which is a training data distribution change,
not parity.

These tests run in the bare venv: torchvision is blocked in sys.modules to
prove nothing in the module's import or construction path reaches it.
"""

import dataclasses
import hashlib
import itertools
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

from dew.data import CC12M, OxfordFlowers, images  # noqa: E402
from dew.data.images import ImageTransform  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]

LABELS = ("pink rose", "yellow tulip", "white lily", "blue orchid")
SCALE = 48
DRAWS = 32
RECORDS = 16


def keep_captions(captions):
    """A caption reader that hands the words back, so a test can read what
    the dataset wrote before a run's encoder tokenizes it."""
    return {"caption": np.asarray(captions)}


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


def _element_for(kind, seed=0):
    image = _synthetic_image(seed)
    if kind == "tfds":
        return {"image": image, "label": 2}
    encoded = cv2.imencode(".jpg", image)[1].tobytes()
    return _pack_records({"jpg": encoded, "txt": b"a yellow tulip"})


def _spec(kind, labels_file, augmentation="flip_jitter"):
    """The TFDS flowers spec, or an arrayrecord one: `record` is what differs."""
    if kind == "tfds":
        return OxfordFlowers(image_size=SCALE, augmentation=augmentation, labels=str(labels_file))
    return CC12M(image_size=SCALE, augmentation=augmentation)


def _make_transform(kind, labels_file, monkeypatch, augmentation="flip_jitter"):
    return ImageTransform(_spec(kind, labels_file, augmentation))


def _resized(kind, element):
    """The deterministic prefix of random_map: decode, convert, resize.
    For arrayrecord, `element` is still the packed byte blob the transform unpacks."""
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
        "sys.modules['transformers'] = None",
        "",
        "import numpy as np",
        "import grain.python as pygrain",
        "from dew.data import CC12M, OxfordFlowers",
        "from dew.data.images import ImageTransform, augment_image, image_augmentations",
        "",
        "labels = sys.argv[1]",
        "image = np.zeros((9, 11, 3), dtype=np.uint8)",
        "for mode in ('none', 'flip_only', 'flip_jitter'):",
        "    for spec in (OxfordFlowers(labels=labels, augmentation=mode), CC12M(augmentation=mode)):",
        "        assert image_augmentations(spec.augmentation) is not None",
        "    assert issubclass(ImageTransform, pygrain.RandomMapTransform)",
        "    out = augment_image(image_augmentations(mode), image, np.random.default_rng(0))",
        "    assert out.dtype == np.uint8 and out.shape == image.shape",
        "print('ok')",
    ])
    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT / "src"), JAX_PLATFORMS="cpu")
    result = subprocess.run(
        [sys.executable, "-c", script, str(labels_file)],
        capture_output=True, text=True, env=env, cwd=REPO_ROOT,
    )
    assert result.stdout.strip().splitlines()[-1:] == ["ok"], result.stderr
    assert "torchvision" not in result.stderr


def test_the_transform_receives_grains_record_rng():
    """grain only hands a per-record rng to RandomMapTransform subclasses; the
    per-record seeding contract depends on that dispatch."""
    assert issubclass(ImageTransform, pygrain.RandomMapTransform)


# ---------------------------------------------------------------------------------
# The augmentation field
# ---------------------------------------------------------------------------------

@pytest.mark.parametrize("kind", ["tfds", "gcs"])
def test_none_mode_returns_the_resized_image_bit_identical(kind, tmp_path, monkeypatch):
    labels_file = _write_labels(tmp_path)
    element = _element_for(kind)
    transform = _make_transform(kind, labels_file, monkeypatch, augmentation="none")

    out = transform.random_map(element, _record_rng(0))

    assert out["image"].dtype == np.uint8
    np.testing.assert_array_equal(out["image"], _resized(kind, element))
    # the surrounding record wiring stays intact
    assert out["caption"] == "a yellow tulip" if kind == "gcs" else out["caption"]


@pytest.mark.parametrize("kind", ["tfds", "gcs"])
def test_flip_only_mode_returns_only_the_image_or_its_mirror(kind, tmp_path, monkeypatch):
    labels_file = _write_labels(tmp_path)
    element = _element_for(kind)
    transform = _make_transform(kind, labels_file, monkeypatch, augmentation="flip_only")

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
    labels_file = _write_labels(tmp_path)
    element = _element_for(kind)
    transform = _make_transform(kind, labels_file, monkeypatch, augmentation="flip_jitter")

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
def test_the_default_augmentation_is_flip_jitter(kind, tmp_path, monkeypatch):
    labels_file = _write_labels(tmp_path)
    element = _element_for(kind)
    spec = dataclasses.replace(_spec(kind, labels_file), augmentation=OxfordFlowers().augmentation)
    transform = ImageTransform(spec)

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
    assert one["caption"] == two["caption"]
    assert np.array_equal(numpy_state, np.random.get_state()[1])
    assert random.getstate() == python_state


def test_the_caption_template_comes_from_the_record_rng(tmp_path):
    """A module-global random.choice picked the template, so a record's caption
    moved with the worker and process count while its image did not."""
    spec = OxfordFlowers(labels=str(_write_labels(tmp_path)))
    element = {"image": _synthetic_image(), "label": 1}

    captions = [spec.record(element, _record_rng(draw))[1] for draw in range(DRAWS)]

    assert len(set(captions)) > 1  # the rng really does pick the template
    assert set(captions) <= {t.format(LABELS[1]) for t in images.PROMPT_TEMPLATES}
    assert spec.record(element, _record_rng(3))[1] == captions[3]
    assert spec.record(element, _record_rng(3))[2] == 1


@pytest.mark.parametrize("column", ["caption", "text"])
def test_a_record_caption_is_taken_as_it_is(column):
    """Hub datasets carry their own text, under either of two column names."""
    assert images.record_caption({column: "a yellow tulip"}) == "a yellow tulip"


def test_a_record_with_no_caption_column_says_what_it_has():
    with pytest.raises(KeyError, match="'caption' or a 'text' column"):
        images.record_caption({"image": None, "url": "x"})


def test_a_hub_record_captions_from_the_record_and_reads_no_label_file(monkeypatch):
    """The same image transform serves a hub dataset: what changes is where the
    caption comes from, and that a caption dataset has no class index."""
    transform = ImageTransform(images.HFImages(name="acme/pets", image_size=SCALE,
                                               augmentation="none"))

    element = {"image": _synthetic_image(0), "caption": "a yellow tulip"}
    out = transform.random_map(element, _record_rng(0))

    np.testing.assert_array_equal(
        out["image"], cv2.resize(element["image"], (SCALE, SCALE),
                                 interpolation=cv2.INTER_AREA))
    assert out["caption"] == "a yellow tulip"
    assert "label" not in out


# ---------------------------------------------------------------------------------
# Loader level: a record does not depend on how many workers produced it
# ---------------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class Flowers(OxfordFlowers):
    """The flowers spec over synthetic records, one label per record so a
    batch says which records it carries."""

    def source(self):
        return [{"image": _synthetic_image(i), "label": i} for i in range(RECORDS)]


def _by_record(spec, worker_count):
    """label -> (image, caption) for every record of one epoch.

    Keyed by label rather than compared batch for batch: grain gives each
    worker its own slice of the index stream, so batch composition follows
    worker_count while record content must not.
    """
    data = dataclasses.replace(spec, worker_count=worker_count).load(
        batch=4, tokenize=keep_captions)
    records = {}
    for batch in itertools.islice(data.train(), data.steps_per_epoch):
        for position, label in enumerate(batch["label"]):
            records[int(label)] = (batch["image"][position].tobytes(),
                                   str(batch["caption"][position]))
    return records


def test_a_record_comes_out_the_same_with_and_without_worker_processes(tmp_path):
    labels_file = tmp_path / "indexed_labels.txt"
    labels_file.write_text("\n".join(f"flower{i:02d}" for i in range(RECORDS)) + "\n")
    spec = Flowers(image_size=SCALE, labels=str(labels_file), val_batches=None, seed=7,
                   read_threads=1, read_buffer=1, worker_buffer=1)

    serial = _by_record(spec, worker_count=0)
    parallel = _by_record(spec, worker_count=2)

    assert sorted(serial) == list(range(RECORDS))
    assert serial == parallel
