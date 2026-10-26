"""Datasets for training and evaluation: a store, and the steps that read it.

A store (`Files`, `Hub`, `TFDS`, `Memory`, several of them in turn
(`Concatenate`), or a Grain, Hugging Face or PyTorch dataset brought in) is a
`Dataset`, and every verb returns a new
one: `map`, `random_map`, `filter` and `flat_map` per record, `shuffle`,
`repeat`, `pack` and `batch` across records, `mix` and `holdout` across
datasets. The stage libraries (`images`, `video`, `audio`, `text`, `chat`,
`prompts`, `preferences`) hold the steps those verbs take.
`Trainer.fit` reads a dataset as the run's mesh shares it.
"""

from . import audio, chat, images, preferences, presets, prompts, text, video
from .chat import Role
from .dataset import Loading
from .pipeline import Dataset, Rejected, concat, mix, phases, schedule
from .sources import TFDS, Concatenate, Files, Hub, Memory, Source, from_grain, from_hf, from_iter, from_torch
from .text import ByteTokenizer, HFTokenizer

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
