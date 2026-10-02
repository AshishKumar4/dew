"""Data for a run: dataset specs, the `Dataset` value they load, tokenizers.

A dataset is a frozen dataclass behind `@datasets(name)`, and `load(batch=)`
turns it into a `Dataset` of batch iterators:

    data = OxfordFlowers(image_size=128).load(batch=32)
    steps = epochs * data.steps_per_epoch

The prepared web-scale image corpora (`CC3M`, `CC12M`, the LAION and
DiffusionDB sets) are importable from `dew.data.images` and registered under
their names in `dew.registry.datasets`; this namespace holds what a run
builds from.

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
                      mixture,
                      ramped,
)
from .images import ArrayRecordImages, HFImages, ImageDataset, OxfordFlowers
from .preferences import IDS_KEY, MASK_KEY, PreferencePairs
from .processors import AutoAudioProcessor
from .prompts import Prompts
from .providers import HubDataset, PreparedTFDS, load
from .sources.hf import HFOptions
from .sources.text import (
    TokenBytes,
    TokenDocumentSource,
    TokenRecords,
    TokenSource,
    TokenWindowSource,
    write_tokens,
)
from .sources.tfds import TFDSOptions
from .streaming import CombinedOnline, OnlineImages, OnlineVideos
from .text import ByteTokenizer, HFTokenizer, tokenizer_for
from .tokens import PackedTokens, TokenWindows
from .video import LocalVideos, VideoDataset, VoxCeleb2

__all__ = ["IDS_KEY", "MASK_KEY", "ArrayRecordImages",
           "AutoAudioProcessor",
           "Batch", "ByteTokenizer", "ChatMessages", "Checkpointable",
           "CombinedOnline", "Corpus", "DataPartition", "DataPhase", "Dataset",
           "DatasetSpec",
           "HFImages", "HFOptions", "HFTokenizer", "HubDataset", "ImageDataset",
           "Loading", "LocalVideos", "OnlineImages", "OnlineVideos", "OxfordFlowers",
           "PackedTokens", "PreferencePairs", "PreparedTFDS", "Prompts", "Ramp", "Reader", "Role",
           "Stage",
           "TFDSOptions", "TokenBytes",
           "TokenDocumentSource", "TokenRecords", "TokenSource", "TokenWindowSource",
           "TokenWindows", "VideoDataset", "VoxCeleb2", "load", "mixture",
           "ramped", "tokenizer_for", "write_tokens"]
