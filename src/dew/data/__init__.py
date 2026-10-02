"""Data for a run: dataset specs, the `Dataset` value they load, tokenizers.

A dataset is a frozen dataclass behind `@datasets(name)`, and `load(batch=)`
turns it into a `Dataset` of batch iterators:

    data = TFDSImages(path="data/oxford_flowers102/2.1.1", image_size=128).load(batch=32)
    steps = epochs * data.steps_per_epoch

The specs here read a kind of store: prepared TFDS, Hugging Face, ArrayRecord
shards, url tables, video trees. Which corpus they read, by name, bucket path
and caption wording, is a recipe's choice; `recipes/diffusion/train.py`
registers its own (`oxford_flowers102`, `cc12m`, the LAION sets and others).

Importing this package registers every dataset and costs none of the heavy
dependencies. cv2, tensorflow_datasets, HF `datasets`, the AV readers and
`transformers` are imported by a spec on use, so a host that only needs the
token loaders never pays for the image stack, and vice versa.
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
           "HFImages", "HFOptions", "HFTokenizer", "HubDataset", "ImageDataset",
           "Loading", "LocalVideos", "OnlineImages", "OnlineVideos",
           "PackedTokens", "PreferencePairs", "PreparedTFDS", "Prompts", "Ramp", "Reader", "Role",
           "Stage",
           "TFDSImages", "TFDSOptions", "TokenBytes", "TokenCorpus",
           "TokenDocumentSource", "TokenRecords", "TokenSource", "TokenWindowSource",
           "TokenWindows", "VideoDataset", "load"]
