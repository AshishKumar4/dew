"""Data for a run: dataset specs, the `Dataset` value they load, tokenizers.

A dataset is a frozen dataclass behind `@datasets(name)`, and `load(batch=)`
turns it into a `Dataset` of batch iterators:

    data = datasets.OxfordFlowers(image_size=128).load(batch=32)
    steps = epochs * data.steps_per_epoch

Importing this package registers every dataset and costs none of the heavy
dependencies: cv2, tensorflow_datasets, HF `datasets`, the
AV readers and `transformers` are imported by a spec on use, so a host that
only needs the token loaders never pays for the image stack, and vice versa.
"""

from .chat import ChatMessages, Role
from .dataset import (
                      Batch,
                      Checkpointable,
                      Corpus,
                      Dataset,
                      DatasetSpec,
                      Loading,
                      Ramp,
                      Stage,
                      local_batch,
                      mixture,
                      ramped,
)
from .images import (
                      CC3M,
                      CC12M,
                      AestheticCoyo,
                      ArrayRecordImages,
                      Combined30M,
                      CombinedAesthetic,
                      CombinedMsml612,
                      DiffusionDB,
                      HFImages,
                      ImageDataset,
                      Laion2bAesthetic,
                      Laion12mCoco,
                      LaionaCoco,
                      LaionaCocoCoyo,
                      OxfordFlowers,
)
from .preferences import IDS_KEY, MASK_KEY, PreferencePairs
from .processors import AutoAudioProcessor, AutoTextTokenizer
from .prompts import Prompts
from .providers import HubDataset, PreparedTFDS, load
from .sources.hf import HFDatasetSource, HFOptions
from .sources.text import (
    TokenBytes,
    TokenColumn,
    TokenDocumentSource,
    TokenRecords,
    TokenSource,
    TokenWindowSource,
)
from .sources.tfds import TFDSOptions
from .streaming import CombinedOnline, OnlineImages
from .text import ByteTokenizer, HFTokenizer, tokenizer_for
from .tokens import PackedTokens, TokenWindows
from .video import LocalVideos, VideoDataset, VoxCeleb2

__all__ = ["CC3M", "CC12M", "IDS_KEY", "MASK_KEY", "AestheticCoyo", "ArrayRecordImages",
           "AutoAudioProcessor", "AutoTextTokenizer",
           "Batch", "ByteTokenizer", "ChatMessages", "Checkpointable", "Combined30M", "CombinedAesthetic",
           "CombinedMsml612", "CombinedOnline", "Corpus", "Dataset", "DatasetSpec", "DiffusionDB",
           "HFDatasetSource", "HFImages", "HFOptions", "HFTokenizer", "HubDataset", "ImageDataset",
           "Laion2bAesthetic", "Laion12mCoco",
           "LaionaCoco", "LaionaCocoCoyo", "Loading", "LocalVideos", "OnlineImages", "OxfordFlowers",
           "PackedTokens", "PreferencePairs", "PreparedTFDS", "Prompts", "Ramp", "Role", "Stage",
           "TFDSOptions", "TokenBytes", "TokenColumn",
           "TokenDocumentSource", "TokenRecords", "TokenSource", "TokenWindowSource",
           "TokenWindows", "VideoDataset", "VoxCeleb2", "load", "local_batch", "mixture",
           "ramped", "tokenizer_for"]
