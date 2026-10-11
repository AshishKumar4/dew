"""Typed recipes and composable record pipelines for training and validation.

Recipes describe records and their transforms. ``load`` compiles a recipe
into a ``Dataset``; a custom order uses ``Pipeline`` and ``compile`` directly.
Storage is selected by the source URL, and ``read`` chooses memory, indexed
or streamed access for every source in the run.
"""

from .cache import Cache, Generation
from .chat import Role
from .compile import compile, iterate, publish, validation
from .dataset import Batch, Checkpointable, DataPartition, DataPhase, Dataset, Loading, Ramp, Reader, Stage
from .pipeline import Configured, ObjectiveData, Pipeline, Rejected
from .preferences import IDS_KEY, MASK_KEY
from .processors import AutoAudioProcessor
from .recipes import (
    Augment,
    Chat,
    HoldOut,
    Images,
    Preferences,
    Prompts,
    Recipe,
    Records,
    Split,
    Text,
    Video,
    load,
)
from .sinks import export
from .sources import TFDS, Files, Hub, Memory, Read, Shard, ShardManifest, Source
from .text import ByteTokenizer, HFTokenizer

__all__ = [
    "IDS_KEY",
    "MASK_KEY",
    "TFDS",
    "Augment",
    "AutoAudioProcessor",
    "Batch",
    "ByteTokenizer",
    "Cache",
    "Chat",
    "Checkpointable",
    "Configured",
    "DataPartition",
    "DataPhase",
    "Dataset",
    "Files",
    "Generation",
    "HFTokenizer",
    "HoldOut",
    "Hub",
    "Images",
    "Loading",
    "Memory",
    "ObjectiveData",
    "Pipeline",
    "Preferences",
    "Prompts",
    "Ramp",
    "Read",
    "Reader",
    "Recipe",
    "Records",
    "Rejected",
    "Role",
    "Shard",
    "ShardManifest",
    "Source",
    "Split",
    "Stage",
    "Text",
    "Video",
    "compile",
    "export",
    "iterate",
    "load",
    "publish",
    "validation",
]
