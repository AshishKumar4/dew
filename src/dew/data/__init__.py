"""Datasets for training and evaluation: a store, and the steps that read it.

A store (`Files`, `Hub`, `TFDS`, `Memory`, several of them in turn
(`Concatenate`), or a Grain, Hugging Face or PyTorch dataset brought in) is a
`Dataset`, and every verb returns a new
one: `map`, `random_map`, `filter` and `flat_map` per record, `shuffle`,
`repeat`, `pack` and `batch` across records, `mix` and `holdout` across
datasets. The stage libraries (`images`, `video`, `audio`, `text`, `chat`,
`prompts`, `preferences`) hold the steps those verbs take.
`Trainer.fit` reads a dataset as the run's mesh shares it.

Each name below is imported from its own module when first read, so importing the package reads none of
the stage libraries a run does not use. `dew.data.storage` alone loads with it: it registers the
`dewcache::` protocol fsspec reads files through, before any store resolves.
"""

from typing import TYPE_CHECKING

from dew._exports import lookup as _lookup
from dew.data import storage as storage

if TYPE_CHECKING:  # written from _EXPORTS (tools/gen_exports.py)
    from dew.data import audio, chat, images, preferences, presets, prompts, text, video
    from dew.data.chat import Role
    from dew.data.dataset import Loading
    from dew.data.pipeline import Dataset, Rejected, concat, mix, phases, schedule
    from dew.data.sources import (
        TFDS,
        Concatenate,
        Files,
        Hub,
        Memory,
        Source,
        from_grain,
        from_hf,
        from_iter,
        from_torch,
    )
    from dew.data.text import ByteTokenizer, HFTokenizer

_EXPORTS = {
    "audio": "dew.data.audio",
    "chat": "dew.data.chat",
    "images": "dew.data.images",
    "preferences": "dew.data.preferences",
    "presets": "dew.data.presets",
    "prompts": "dew.data.prompts",
    "text": "dew.data.text",
    "video": "dew.data.video",
    "Role": "dew.data.chat",
    "Loading": "dew.data.dataset",
    "Dataset": "dew.data.pipeline",
    "Rejected": "dew.data.pipeline",
    "concat": "dew.data.pipeline",
    "mix": "dew.data.pipeline",
    "phases": "dew.data.pipeline",
    "schedule": "dew.data.pipeline",
    "TFDS": "dew.data.sources",
    "Concatenate": "dew.data.sources",
    "Files": "dew.data.sources",
    "Hub": "dew.data.sources",
    "Memory": "dew.data.sources",
    "Source": "dew.data.sources",
    "from_grain": "dew.data.sources",
    "from_hf": "dew.data.sources",
    "from_iter": "dew.data.sources",
    "from_torch": "dew.data.sources",
    "ByteTokenizer": "dew.data.text",
    "HFTokenizer": "dew.data.text",
}

__getattr__ = _lookup(__name__, _EXPORTS)

__all__ = [
    "TFDS",
    "ByteTokenizer",
    "Concatenate",
    "Dataset",
    "Files",
    "HFTokenizer",
    "Hub",
    "Loading",
    "Memory",
    "Rejected",
    "Role",
    "Source",
    "audio",
    "chat",
    "concat",
    "from_grain",
    "from_hf",
    "from_iter",
    "from_torch",
    "images",
    "mix",
    "phases",
    "preferences",
    "presets",
    "prompts",
    "schedule",
    "text",
    "video",
]
