"""Data for a run: dataset specs, the `Dataset` value they load, tokenizers.

A dataset is a frozen dataclass behind `@datasets(name)`, and `load(batch=)`
turns it into a `Dataset` of batch iterators:

    data = datasets.OxfordFlowers(image_size=128).load(batch=32)
    steps = epochs * data.steps_per_epoch

Importing this package registers every dataset and costs none of the heavy
dependencies: cv2, albumentations, tensorflow_datasets, HF `datasets`, the
AV readers and `transformers` are imported by a spec on use, so a host that
only needs the token loaders never pays for the image stack, and vice versa.
"""

from .chat import ChatMessages, Role
from .dataset import Batch, Checkpointable, Dataset, DatasetSpec, Loading, local_batch
from .images import (AestheticCoyo, CC3M, CC12M, Combined30M, CombinedAesthetic,
                     CombinedMsml612, DiffusionDB, HFImages, ImageDataset, Laion2bAesthetic,
                     Laion12mCoco, LaionaCoco, LaionaCocoCoyo, OxfordFlowers)
from .processors import AutoAudioProcessor, AutoTextTokenizer
from .preferences import IDS_KEY, MASK_KEY, PreferencePairs
from .prompts import Prompts
from .sources.hf import HFDatasetSource
from .sources.text import TokenDocumentSource, TokenFileSource
from .streaming import CombinedOnline, OnlineImages
from .text import ByteTokenizer, HFTokenizer
from .tokens import PackedTokens, TokenWindows
from .video import LocalVideos, VideoDataset, VoxCeleb2

__all__ = [
    "AestheticCoyo", "AutoAudioProcessor", "AutoTextTokenizer", "Batch", "ByteTokenizer",
    "Checkpointable", "Dataset", "DatasetSpec", "DiffusionDB", "HFDatasetSource", "HFImages", "HFTokenizer",
    "IDS_KEY", "ImageDataset", "Laion12mCoco", "Laion2bAesthetic", "LaionaCoco", "LaionaCocoCoyo",
    "Loading", "LocalVideos", "MASK_KEY", "OnlineImages", "OxfordFlowers",
    "PackedTokens", "PreferencePairs", "Prompts", "Role",
    "TokenDocumentSource", "TokenFileSource", "TokenWindows", "VideoDataset", "VoxCeleb2", "local_batch",
]
