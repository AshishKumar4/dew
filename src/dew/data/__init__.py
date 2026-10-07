"""Data for a run: dataset specs, the `Dataset` value they load, and tokenizers.

A dataset spec is a frozen dataclass, with an alias in `dew.registry.datasets`,
and its `load(batch=)` returns a `Dataset` of batch iterators:

    data = TFDSImages(path="data/oxford_flowers102/2.1.1", image_size=128).load(batch=32)
    steps = epochs * data.steps_per_epoch

Each spec here reads one kind of store: prepared TFDS, Hugging Face,
ArrayRecord shards, url tables or video trees. Which corpus to read, with its
name, bucket path and caption wording, is up to the recipe. For example,
`recipes/diffusion/train.py` defines its own in `CORPORA`
(`oxford-flowers102`, `cc12m`, the LAION sets and others).

Importing this package defines every dataset without importing the heavy
dependencies. A spec imports cv2, tensorflow_datasets, HF `datasets`, the AV
readers or `transformers` only when it uses them, so a host that only needs
the token loaders never loads the image stack, and the reverse.
"""

from .chat import ChatMessages, Role
from .dataset import (
                      Batch,
                      Checkpointable,
                      Corpus,
                      DataPartition,
                      DataPhase,
                      Dataset,
                      DatasetSpec,
                      Loading,
                      Ramp,
                      Reader,
                      Stage,
)
from .images import ArrayRecordImages, HFImages, ImageDataset, TFDSImages
from .preferences import IDS_KEY, MASK_KEY, PreferencePairs
from .processors import AutoAudioProcessor
from .prompts import Prompts
from .providers import HubDataset, PreparedTFDS, load
from .sources.hf import HFOptions
from .sources.text import (
    HubText,
    TokenBytes,
    TokenCorpus,
    TokenDocumentSource,
    TokenRecords,
    TokenSource,
    TokenWindowSource,
)
from .sources.tfds import TFDSOptions
from .streaming import OnlineImages, OnlineVideos
from .text import ByteTokenizer, HFTokenizer
from .tokens import PackedTokens, TokenWindows
from .video import LocalVideos, VideoDataset

__all__ = ["IDS_KEY", "MASK_KEY", "ArrayRecordImages",
           "AutoAudioProcessor",
           "Batch", "ByteTokenizer", "ChatMessages", "Checkpointable",
           "Corpus", "DataPartition", "DataPhase", "Dataset",
           "DatasetSpec",
           "HFImages", "HFOptions", "HFTokenizer", "HubDataset", "HubText", "ImageDataset",
           "Loading", "LocalVideos", "OnlineImages", "OnlineVideos",
           "PackedTokens", "PreferencePairs", "PreparedTFDS", "Prompts", "Ramp", "Reader", "Role",
           "Stage",
           "TFDSImages", "TFDSOptions", "TokenBytes", "TokenCorpus",
           "TokenDocumentSource", "TokenRecords", "TokenSource", "TokenWindowSource",
           "TokenWindows", "VideoDataset", "load"]
