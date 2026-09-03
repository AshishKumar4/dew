from __future__ import annotations

import grain.python as pygrain
from typing import Dict, Any, Optional, Tuple, Union, List, Callable, TYPE_CHECKING
from pathlib import Path
import numpy as np
import jax
import cv2
from dew.inputs.processors import AutoTextTokenizer
from .registry import datasetMap, onlineDatasetMap, mediaDatasetMap
from .sources.base import MediaDataset
# NOTE: .online_loader is imported lazily inside the two `*_online` factories.
# It needs HF `datasets`, which the grain paths must not require.
from functools import partial
from absl import flags

if TYPE_CHECKING:
    # Only for the load_data signature; importing dew.config at module scope
    # here would make the data stack depend on tyro.
    from dew.config import DataConfig

# grain's worker processes read absl flags; a script that never runs absl.app
# would crash on any worker_count > 0 with UnparsedFlagAccessError.
if not flags.FLAGS.is_parsed():
    flags.FLAGS.mark_as_parsed()


def generate_collate_fn(media_type="image"):
    """Generate a collate function based on media type.
    
    Args:
        media_type: Type of media ("image" or "video").
        
    Returns:
        A collate function for the specified media type. A malformed sample
        raises: a batch that silently became zeros trained as data.
    """
    auto_tokenize = AutoTextTokenizer(tensor_type="np")
    
    def image_collate(batch):
        captions = [sample["caption"] for sample in batch]
        results = auto_tokenize(captions)

        # Check if all images have the same shape
        image_shapes = [sample["image"].shape for sample in batch]
        if len(set(str(shape) for shape in image_shapes)) > 1:
            # Different shapes: resize everything to the largest. cv2 takes (width, height).
            target_h = max(shape[0] for shape in image_shapes)
            target_w = max(shape[1] for shape in image_shapes)
            images = np.stack([
                cv2.resize(sample["image"], (target_w, target_h))
                if sample["image"].shape[:2] != (target_h, target_w) else sample["image"]
                for sample in batch
            ], axis=0)
        else:
            # All same shape, can just stack
            images = np.stack([sample["image"] for sample in batch], axis=0)

        return {
            "image": images,
            "text": {
                "input_ids": results['input_ids'],
                "attention_mask": results['attention_mask'],
            }
        }

    def video_collate(batch):
        captions = [sample["caption"] for sample in batch]
        results = auto_tokenize(captions)

        # Check if all videos have the same shape
        video_shapes = [sample["video"].shape for sample in batch]
        if len(set(str(shape) for shape in video_shapes)) > 1:
            # Get max dimensions
            max_frames = max(shape[0] for shape in video_shapes)
            max_height = max(shape[1] for shape in video_shapes)
            max_width = max(shape[2] for shape in video_shapes)

            # Resize videos to the same shape
            videos = []
            for sample in batch:
                video = sample["video"]
                num_frames, height, width = video.shape[:3]

                if height != max_height or width != max_width:
                    # Resize each frame
                    video = np.array([
                        cv2.resize(frame, (max_width, max_height))
                        for frame in video
                    ])

                if num_frames < max_frames:
                    # Pad with duplicates of the last frame
                    padding = np.tile(video[-1:], (max_frames - num_frames, 1, 1, 1))
                    video = np.concatenate([video, padding], axis=0)

                videos.append(video)

            videos = np.stack(videos, axis=0)
        else:
            # All videos have the same shape, can just stack
            videos = np.stack([sample["video"] for sample in batch], axis=0)

        return {
            "video": videos,
            "text": {
                "input_ids": results['input_ids'],
                "attention_mask": results['attention_mask'],
            }
        }

    if media_type == "video":
        return video_collate
    else:  # Default to image
        return image_collate

class CaptionDeletionTransform(pygrain.MapTransform):
    def map(self, element):
        """Delete the caption from the element."""
        if "caption" in element:
            del element["caption"]
        return element

def get_dataset_grain(
    data_name="cc12m",
    batch_size=64,
    image_scale=256,
    count=None,
    num_epochs=None,
    method=None, #jax.image.ResizeMethod.LANCZOS3,
    worker_count=32,
    read_thread_count=64,
    read_buffer_size=50,
    worker_buffer_size=20,
    seed=0,
    dataset_source="/mnt/gcs_mount/arrayrecord2/cc12m/",
    val_batch_size=None,
    val_worker_count=8,
    val_count=None,
):
    """Legacy function for getting grain dataset loaders for images.
    
    Args:
        data_name: Name of the dataset in datasetMap.
        batch_size: Batch size for the dataset.
        image_scale: Size to scale images to.
        count: Optional count limit for the dataset.
        num_epochs: Number of epochs to iterate.
        method: Interpolation method for resizing.
        worker_count: Number of worker processes.
        read_thread_count: Number of read threads.
        read_buffer_size: Size of the read buffer.
        worker_buffer_size: Size of the worker buffer.
        seed: Random seed.
        dataset_source: Source path for the dataset.
        val_batch_size: Batch size for the validation loader. Defaults to the
            process-local training batch size.
        val_worker_count: Number of worker processes for the validation loader.
        val_count: Records held out from the head of the source, in canonical
            order, as the validation split; the train loader then covers the
            rest. None (default) validates over every record, training data
            included.
        
    Returns:
        Dictionary with train dataset function and metadata.
    """
    dataset = datasetMap[data_name]
    data_source = dataset["source"](dataset_source)
    augmenter = dataset["augmenter"](image_scale, method)
    local_batch_size = batch_size // jax.process_count()
    dataset_length = len(data_source) if count is None else count

    # A held-out slice off the head keeps the two loaders disjoint, so FID and
    # CLIP are not measured on records the model trained on.
    train_source = val_source = data_source
    train_length = val_length = dataset_length
    if val_count:
        if not 0 < val_count < dataset_length:
            raise ValueError(
                f"val_count must be within 1..{dataset_length - 1} records for "
                f"'{data_name}', got {val_count}"
            )
        val_source = _SourceSlice(data_source, 0, val_count)
        train_source = _SourceSlice(data_source, val_count, dataset_length)
        train_length, val_length = len(train_source), len(val_source)

    train_sampler = pygrain.IndexSampler(
        num_records=train_length,
        shuffle=True,
        seed=seed,
        num_epochs=num_epochs,
        shard_options=pygrain.ShardByJaxProcess(),
    )

    # Validation reads its records in canonical order: sharing the shuffled
    # train sampler made "validation" a random slice of training data.
    val_sampler = pygrain.IndexSampler(
        num_records=val_length,
        shuffle=False,
        seed=seed,
        num_epochs=num_epochs,
        shard_options=pygrain.ShardByJaxProcess(),
    )
    
    def get_trainset():
        transformations = [
            augmenter(),
        ]
        transformations.append(pygrain.Batch(local_batch_size, drop_remainder=True))

        loader = pygrain.DataLoader(
            data_source=train_source,
            sampler=train_sampler,
            operations=transformations,
            worker_count=worker_count,
            read_options=pygrain.ReadOptions(
                read_thread_count, read_buffer_size
            ),
            worker_buffer_size=worker_buffer_size,
        )
        return loader
    
    def get_valset():
        transformations = [
            augmenter(),
            pygrain.Batch(val_batch_size or local_batch_size, drop_remainder=True),
        ]

        loader = pygrain.DataLoader(
            data_source=val_source,
            sampler=val_sampler,
            operations=transformations,
            worker_count=val_worker_count,
            read_options=pygrain.ReadOptions(
                32, 128
            ),
            worker_buffer_size=32,
        )
        return loader

    return {
        "train": get_trainset,
        "train_len": train_length,
        "val": get_valset,
        "val_len": val_length,
        "local_batch_size": local_batch_size,
        "global_batch_size": batch_size,
    }


def get_dataset_online(
        data_name="combined_online",
        batch_size=64,
        image_scale=256,
        count=None,
        num_epochs=None,
        method=jax.image.ResizeMethod.LANCZOS3,
        worker_count=32,
        read_thread_count=64,
        read_buffer_size=50,
        worker_buffer_size=20,
        seed=0,
        dataset_source="/mnt/gcs_mount/arrayrecord2/cc12m/",
    ):
    """Legacy function for getting online streaming dataloader for images.
    
    Args:
        data_name: Name of the dataset in onlineDatasetMap.
        batch_size: Batch size for the dataset.
        image_scale: Size to scale images to.
        count: Optional count limit for the dataset.
        num_epochs: Number of epochs to iterate.
        method: Interpolation method for resizing.
        worker_count: Number of worker processes.
        read_thread_count: Number of read threads.
        read_buffer_size: Size of the read buffer.
        worker_buffer_size: Size of the worker buffer.
        seed: Random seed.
        dataset_source: Source path for the dataset.
        
    Returns:
        Dictionary with train dataset function and metadata.
    """
    if data_name not in onlineDatasetMap:
        raise ValueError(f"Dataset {data_name} not found in onlineDatasetMap")

    # Imported here, not at module scope: the streaming stack needs HF
    # `datasets`, which the grain paths above deliberately do without.
    from .online_loader import OnlineStreamingDataLoader

    local_batch_size = batch_size // jax.process_count()
    sources = onlineDatasetMap[data_name]["source"]
    dataloader = OnlineStreamingDataLoader(
            sources, 
            batch_size=local_batch_size,
            num_workers=worker_count,
            num_threads=read_thread_count,
            image_shape=(image_scale, image_scale),
            global_process_count=jax.process_count(),
            global_process_index=jax.process_index(),
            prefetch=worker_buffer_size,
            collate_fn=generate_collate_fn(),
            default_split="train",
        )
    
    def get_trainset():
        return dataloader
    
    return {
        "train": get_trainset,
        "train_len": len(dataloader) * jax.process_count(),
        "local_batch_size": local_batch_size,
        "global_batch_size": batch_size,
    }


# ---------------------------------------------------------------------------------
# New unified dataset loader for both images and videos
# ---------------------------------------------------------------------------------

class _SourceSlice:
    """Random-access view over `source[start:stop]`.

    Gives the train and validation loaders disjoint index ranges while each
    keeps a stock grain IndexSampler, so sharding and epoch handling stay
    grain's. Plain attributes only: grain pickles the source to its workers.
    """

    def __init__(self, source: Any, start: int, stop: int):
        self.source = source
        self.start = start
        self.length = stop - start

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int):
        if not 0 <= index < self.length:
            raise IndexError(index)
        return self.source[self.start + index]


_HF_PREFIX = "hf:"


def _hf_dataset_spec(data_name: str) -> Optional[Tuple[str, str]]:
    """`(dataset, split)` when `data_name` is an `hf:<dataset>[:<split>]` name.

    A hub dataset is named, not registered: one repo id per dataset and no
    path to resolve against, so the dataset string carries both halves. Repo
    ids are `namespace/name` and hold no colon, so the split parses off the
    tail unambiguously; it defaults to train.
    """
    if not data_name or not data_name.startswith(_HF_PREFIX):
        return None
    dataset, _, split = data_name[len(_HF_PREFIX):].partition(":")
    if not dataset:
        raise ValueError(
            f"'{data_name}' names no hub dataset; write it as hf:<dataset>:<split>")
    return dataset, split or "train"


def _hf_media_dataset(dataset: str, split: str) -> MediaDataset:
    """The image pipeline for a hub dataset: its Arrow table, the TFDS transform.

    Records go through the same augmenter as a TFDS image dataset. The caption
    comes from the record instead of a class-label file, which is the one
    thing a hub dataset does differently.
    """
    from .sources.hf import HFDatasetSource
    from .sources.images import ImageTFDSAugmenter, labelizer_record_caption

    return MediaDataset(
        source=HFDatasetSource(name=dataset, split=split),
        augmenter=ImageTFDSAugmenter(labelizer=labelizer_record_caption),
        media_type="image",
    )



def get_media_dataset_grain(
    data_name: str,
    batch_size: int = 64,
    media_scale: int = 256,
    sequence_length: int = 1,
    count: Optional[int] = None,
    num_epochs: Optional[int] = None,
    method: Any = cv2.INTER_AREA,
    worker_count: int = 32,
    read_thread_count: int = 64,
    read_buffer_size: int = 50,
    worker_buffer_size: int = 20,
    seed: int = 0,
    dataset_source: Optional[str] = None,
    media_type: Optional[str] = None,  # Will be auto-detected if None
    additional_transform_kwargs: Dict[str, Any] = None,
    val_count: Optional[int] = None,
    val_batch_size: Optional[int] = None,
):
    """Get a grain dataset loader for any media type (image or video).
    
    Args:
        data_name: Name of the dataset in mediaDatasetMap, or an
            `hf:<dataset>:<split>` reference to a Hugging Face hub dataset,
            which builds its own image pipeline.
        batch_size: Batch size for the dataset.
        media_scale: Size to scale media (image or video frames) to.
        sequence_length: Length of the sequence for video data.
        count: Optional count limit for the dataset.
        num_epochs: Number of epochs to iterate.
        method: Interpolation method for resizing.
        worker_count: Number of worker processes.
        read_thread_count: Number of read threads.
        read_buffer_size: Size of the read buffer.
        worker_buffer_size: Size of the worker buffer.
        seed: Random seed.
        dataset_source: Root path the dataset's source resolves its files
            against. Required - there is no sane default for a bucket mount
            or a local media tree. An `hf:` dataset needs none, it resolves
            by name.
        media_type: Type of media ("image" or "video"). Auto-detected if None.
        additional_transform_kwargs: Additional arguments for the transform.
        val_count: Records held out, in canonical source order, as a
            validation split; the train loader then covers the rest. None
            (default) means no validation loader.
        val_batch_size: Batch size of the validation loader. Defaults to the
            train loader's local batch size.

    Returns:
        Dictionary with train dataset function and metadata; with val_count set
        it also carries "val" and "val_len" for the held-out split.
    """
    hf_spec = _hf_dataset_spec(data_name)
    if hf_spec is not None:
        media_dataset = _hf_media_dataset(*hf_spec)
    else:
        if data_name not in mediaDatasetMap:
            raise ValueError(f"Dataset {data_name} not found in mediaDatasetMap")
        if not dataset_source:
            raise ValueError(
                f"get_media_dataset_grain('{data_name}') needs an explicit dataset_source: "
                "media sources resolve their files against it, and an unset one used to "
                "reach os.path.join(None, ...) inside the source itself."
            )
        media_dataset = mediaDatasetMap[data_name]

    # Auto-detect media_type if not provided
    if media_type is None:
        media_type = media_dataset.media_type

    # Get the data source and augmenter
    data_source = media_dataset.get_source(dataset_source)

    # Prepare transform kwargs. A frame size and a clip length are the video
    # transforms' arguments; the image transforms take neither, and handing
    # them one raised TypeError on every image dataset that came through here.
    if media_type == "image":
        transform_kwargs = {"image_scale": media_scale, "method": method}
    else:
        transform_kwargs = {
            "frame_size": media_scale,
            "method": method,
            "sequence_length": sequence_length,
        }
    if additional_transform_kwargs:
        transform_kwargs.update(additional_transform_kwargs)

    augmenter = media_dataset.get_augmenter(**transform_kwargs)

    # Calculate local batch size for distributed training
    local_batch_size = batch_size // jax.process_count()

    # Create a sampler for the dataset
    if hasattr(data_source, "__len__"):
        dataset_length = len(data_source) if count is None else count
    else:
        # Some data sources like video files list don't have __len__
        dataset_length = count if count is not None else 1000000  # Default large number

    train_source, val_source = data_source, None
    if val_count:
        if not hasattr(data_source, "__len__"):
            raise ValueError(
                "A validation split needs a data source of known length; "
                f"'{data_name}' does not report one."
            )
        if not 0 < val_count < dataset_length:
            raise ValueError(
                f"val_count must be within 1..{dataset_length - 1} records for "
                f"'{data_name}', got {val_count}"
            )
        val_source = _SourceSlice(data_source, 0, val_count)
        train_source = _SourceSlice(data_source, val_count, dataset_length)

    train_length = dataset_length if val_source is None else len(train_source)

    train_sampler = pygrain.IndexSampler(
        num_records=train_length,
        shuffle=True,
        seed=seed,
        num_epochs=num_epochs,
        shard_options=pygrain.ShardByJaxProcess(),
    )

    def get_trainset():
        """Get a training dataset iterator."""
        transformations = [
            augmenter(),
            pygrain.Batch(local_batch_size, drop_remainder=True),
        ]

        loader = pygrain.DataLoader(
            data_source=train_source,
            sampler=train_sampler,
            operations=transformations,
            worker_count=worker_count,
            read_options=pygrain.ReadOptions(
                read_thread_count, read_buffer_size
            ),
            worker_buffer_size=worker_buffer_size,
        )
        return loader

    dataset = {
        "train": get_trainset,
        "train_len": train_length,
        "local_batch_size": local_batch_size,
        "global_batch_size": batch_size,
        "media_type": media_type,
    }

    if val_source is None:
        return dataset

    # Its own unshuffled sampler: sharing the train sampler turned validation
    # into a random slice of the training stream.
    val_sampler = pygrain.IndexSampler(
        num_records=len(val_source),
        shuffle=False,
        seed=seed,
        num_epochs=num_epochs,
        shard_options=pygrain.ShardByJaxProcess(),
    )

    def get_valset():
        """Get a validation dataset iterator."""
        transformations = [
            augmenter(),
            pygrain.Batch(val_batch_size or local_batch_size, drop_remainder=True),
        ]

        loader = pygrain.DataLoader(
            data_source=val_source,
            sampler=val_sampler,
            operations=transformations,
            worker_count=worker_count,
            read_options=pygrain.ReadOptions(
                read_thread_count, read_buffer_size
            ),
            worker_buffer_size=worker_buffer_size,
        )
        return loader

    dataset["val"] = get_valset
    dataset["val_len"] = len(val_source)
    return dataset


def get_media_dataset_online(
    data_name: str = "combined_online",
    batch_size: int = 64,
    media_scale: int = 256,
    worker_count: int = 16,
    read_thread_count: int = 512,
    worker_buffer_size: int = 20,
    dataset_sources: List[str] = None,
    media_type: str = "image",  # Default to image for online datasets
    timeout: int = 15,
    retries: int = 3,
    min_media_scale: int = 128,
):
    """Get an online streaming dataset loader for any media type.
    
    Args:
        data_name: Name of the dataset in onlineDatasetMap, or "custom" for custom sources.
        batch_size: Batch size for the dataset.
        media_scale: Size to scale media (image or video frames) to.
        worker_count: Number of worker processes.
        read_thread_count: Number of read threads.
        worker_buffer_size: Size of the worker buffer.
        dataset_sources: Custom dataset sources if data_name is "custom".
        media_type: Type of media ("image" or "video"). 
        timeout: Timeout for dataset operations.
        retries: Number of retries for dataset operations.
        min_media_scale: Minimum scale for media items.
        
    Returns:
        Dictionary with train dataset function and metadata.
    """
    local_batch_size = batch_size // jax.process_count()
    
    # Get dataset sources
    if dataset_sources is None:
        if data_name not in onlineDatasetMap:
            raise ValueError(f"Dataset {data_name} not found in onlineDatasetMap")
        sources = onlineDatasetMap[data_name]["source"]
    else:
        sources = dataset_sources
    
    # Configure shape parameter based on media type
    shape_param = "image_shape" if media_type == "image" else "frame_size"
    shape_value = (media_scale, media_scale) if media_type == "image" else media_scale
    
    # Configure min scale parameter based on media type
    min_scale_param = "min_image_shape" if media_type == "image" else "min_frame_size"
    min_scale_value = (min_media_scale, min_media_scale) if media_type == "image" else min_media_scale
    
    # Prepare dataloader kwargs
    dataloader_kwargs = {
        "batch_size": local_batch_size,
        "num_workers": worker_count,
        "num_threads": read_thread_count,
        shape_param: shape_value,
        min_scale_param: min_scale_value,
        "global_process_count": jax.process_count(),
        "global_process_index": jax.process_index(),
        "prefetch": worker_buffer_size,
        "collate_fn": generate_collate_fn(media_type),
        "default_split": "train",
        "timeout": timeout,
        "retries": retries,
    }

    # Imported here, not at module scope: the streaming stack needs HF
    # `datasets`, which the grain paths above deliberately do without.
    from .online_loader import OnlineStreamingDataLoader

    dataloader = OnlineStreamingDataLoader(sources, **dataloader_kwargs)
    
    def get_trainset():
        """Get a training dataset iterator."""
        return dataloader
    
    return {
        "train": get_trainset,
        "train_len": len(dataloader) * jax.process_count(),
        "local_batch_size": local_batch_size,
        "global_batch_size": batch_size,
        "media_type": media_type,
    }


def get_token_dataset_grain(
    train_path: str,
    val_path: str,
    batch_size: int,
    seq_len: int,
    seed: int = 0,
    worker_count: int = 32,
    read_thread_count: int = 64,
    read_buffer_size: int = 50,
    worker_buffer_size: int = 20,
    num_epochs: Optional[int] = None,
    val_batch_size: Optional[int] = None,
):
    """Grain loaders over a tokenized corpus (see `dew.data.sources.text`).

    Train shuffles a seeded IndexSampler over the windows of train.bin; val
    reads val.bin in file order through its own unshuffled sampler, so the
    two splits are disjoint files and validation is reproducible. Both
    shard by JAX process like every other grain path.

    Returns:
        The standard loader dict: "train" fn, "train_len", "val" fn,
        "val_len", "local_batch_size", "global_batch_size".
    """
    from .sources.text import TokenFileSource

    train_source = TokenFileSource(train_path, seq_len)
    val_source = TokenFileSource(val_path, seq_len)
    local_batch_size = batch_size // jax.process_count()

    train_sampler = pygrain.IndexSampler(
        num_records=len(train_source),
        shuffle=True,
        seed=seed,
        num_epochs=num_epochs,
        shard_options=pygrain.ShardByJaxProcess(),
    )
    val_sampler = pygrain.IndexSampler(
        num_records=len(val_source),
        shuffle=False,
        seed=seed,
        num_epochs=num_epochs,
        shard_options=pygrain.ShardByJaxProcess(),
    )

    def get_trainset():
        loader = pygrain.DataLoader(
            data_source=train_source,
            sampler=train_sampler,
            operations=[pygrain.Batch(local_batch_size, drop_remainder=True)],
            worker_count=worker_count,
            read_options=pygrain.ReadOptions(
                read_thread_count, read_buffer_size
            ),
            worker_buffer_size=worker_buffer_size,
        )
        return loader

    def get_valset():
        loader = pygrain.DataLoader(
            data_source=val_source,
            sampler=val_sampler,
            operations=[pygrain.Batch(val_batch_size or local_batch_size, drop_remainder=True)],
            worker_count=worker_count,
            read_options=pygrain.ReadOptions(
                read_thread_count, read_buffer_size
            ),
            worker_buffer_size=worker_buffer_size,
        )
        return loader

    return {
        "train": get_trainset,
        "train_len": len(train_source),
        "val": get_valset,
        "val_len": len(val_source),
        "local_batch_size": local_batch_size,
        "global_batch_size": batch_size,
    }


def chunk_counts(lengths, chunk_len: int):
    """Chunks of at most `chunk_len` tokens each of `lengths` is cut into."""
    return -(-np.asarray(lengths, np.int64) // chunk_len)


class DocumentChunks(pygrain.MapDataset):
    """Documents cut into consecutive chunks of at most `chunk_len` tokens.

    Grain's packer refuses an element longer than the bin it packs into, so a
    document that outgrows the window is cut first; each chunk becomes its own
    segment in the packed row, which keeps attention inside the chunk and RoPE
    running from the chunk's own 0.

    The chunk table is built once from the document lengths, so a record costs
    one memmap slice rather than a walk over the documents before it.
    """

    def __init__(self, parent: pygrain.MapDataset, lengths, chunk_len: int):
        super().__init__(parent)
        self._chunk_len = chunk_len
        lengths = np.asarray(lengths, np.int64)
        counts = chunk_counts(lengths, chunk_len)
        self._document = np.repeat(np.arange(len(lengths), dtype=np.int64), counts)
        first_chunk = np.concatenate([[0], np.cumsum(counts)[:-1]])
        self._offset = (np.arange(len(self._document), dtype=np.int64)
                        - first_chunk[self._document]) * chunk_len

    def __len__(self) -> int:
        return len(self._document)

    def __getitem__(self, index):
        # grain's conventions: a slice is the sharding and windowing API
        # (ds[shard::count]), and an index past the end wraps, which is what
        # makes `repeat` a length change rather than a copy.
        if isinstance(index, slice):
            return self.slice(index)
        index = index % len(self)
        text = self._parent[int(self._document[index])]["text"]
        start = int(self._offset[index])
        return {"text": text[start:start + self._chunk_len]}


def get_packed_token_dataset_grain(
    train_path: str,
    val_path: str,
    batch_size: int,
    seq_len: int,
    seed: int = 0,
    worker_count: int = 32,
    worker_buffer_size: int = 20,
    num_epochs: Optional[int] = None,
    val_batch_size: Optional[int] = None,
    num_packing_bins: int = 8,
):
    """Grain loaders that pack whole documents into `seq_len + 1` windows.

    Documents come from `TokenDocumentSource`, which cuts the token stream at
    the eos ids the tokenize tool writes between files. Each document (in
    chunks, when it outgrows the window) is one element the packer adds to the
    first bin with room, and every emitted window carries `text_segment_ids`
    (which document each token is from, 0 for padding) and `text_positions`
    (the token's position inside its document), so the model can stop
    attention and the loss at document boundaries. This is grain's `Dataset`
    API rather than `DataLoader` for the reason grain gives for switching:
    packing.

    Train shuffles the documents from `seed`, reshuffled per epoch, and val
    reads them in file order, so the two splits stay disjoint files and
    validation is reproducible. Both shard by JAX process, by slicing the
    documents before packing: sharding after it would have every process pack
    the same documents. num_epochs None runs forever, as the fixed-window
    sampler does.

    Returns:
        The standard loader dict: "train" fn, "train_len", "val" fn,
        "val_len", "local_batch_size", "global_batch_size". The two lengths
        count window-sized chunks, which is the upper bound on the windows a
        pass over the split yields, and the count a run has before it packs
        anything: every emitted window holds at least one chunk, the bound is
        tight once documents reach the window, and which chunks share a
        window depends on the shuffle. A recipe divides this by the batch
        size for steps_per_epoch, so counting documents instead reports zero
        steps for a corpus of fewer documents than a batch.
    """
    from grain.experimental import FirstFitPackIterDataset

    from .sources.text import TokenDocumentSource

    local_batch_size = batch_size // jax.process_count()
    window = seq_len + 1

    # One source per split, reused by its loader: finding the boundaries reads
    # the whole file, and a run rebuilding it per epoch would read a
    # multi-gigabyte train.bin again for a table it already has.
    train_source = TokenDocumentSource(train_path)
    val_source = TokenDocumentSource(val_path)

    def build_loader(source, batch, shuffle):
        documents = DocumentChunks(
            pygrain.MapDataset.source(source), source.lengths, window)
        documents = documents[jax.process_index()::jax.process_count()]
        if shuffle:
            documents = documents.shuffle(seed)
        documents = documents.repeat(num_epochs)
        reads = documents.to_iter_dataset()
        if worker_count:
            # The workers read documents, and the packer stays behind them in
            # this process: grain runs a whole pipeline per worker, so packing
            # inside them would fill bins from one worker's slice of the
            # documents and make the windows depend on worker_count.
            reads = reads.mp_prefetch(pygrain.MultiprocessingOptions(
                num_workers=worker_count,
                per_worker_buffer_size=worker_buffer_size))
        packed = FirstFitPackIterDataset(
            reads,
            length_struct={"text": window},
            num_packing_bins=num_packing_bins,
            seed=seed,
            # Bins come out in packing order for val, so a validation pass is
            # the same batches every time.
            shuffle_bins=shuffle,
            padding_struct={"text": 0},
        )
        return packed.batch(batch, drop_remainder=True)

    def packed_windows(source) -> int:
        return int(chunk_counts(source.lengths, window).sum())

    return {
        "train": lambda: build_loader(train_source, local_batch_size, True),
        "train_len": packed_windows(train_source),
        "val": lambda: build_loader(
            val_source, val_batch_size or local_batch_size, False),
        "val_len": packed_windows(val_source),
        "local_batch_size": local_batch_size,
        "global_batch_size": batch_size,
    }


def _token_dataset_dir(name: Optional[str]) -> Optional[str]:
    """`name` if it names a tokenized-corpora directory, else None.

    A dataset entry in a registry can shadow a directory someone happens to
    name the same, so the registered factories keep precedence and this
    dispatch only fires on an explicit, existing directory.
    """
    if not name or name in datasetMap or name in mediaDatasetMap:
        return None
    root = Path(name)
    if root.is_dir() and (root / "train.bin").is_file():
        return name
    return None


def load_data(config: DataConfig) -> dict:
    """Dataset iterators for a run config: which factory, and with what knobs.

    'grain' and 'online' name the factory. 'auto' prefers grain when the
    dataset is registered for it, and only falls back to the online streamer
    for datasets registered solely there - the name alone once decided this
    ('online' in the dataset name), which chose a loader from spelling.

    A dataset that is a directory of tokenized text (train.bin [+ val.bin]
    from tools/tokenize_text.py) takes the token loader ahead of all of
    that, and needs DataConfig.sequence_length. With pack_sequences the
    windows are whole documents packed by grain instead of fixed strides.

    A dataset named `hf:<dataset>:<split>` is a Hugging Face hub dataset,
    which the media loader reads through grain's random access.
    """
    token_dir = _token_dataset_dir(config.dataset)
    if token_dir is not None:
        if not config.sequence_length:
            raise ValueError(
                f"dataset '{config.dataset}' is a tokenized text directory "
                "(train.bin present); it needs data.sequence_length set, in "
                "tokens per training window"
            )
        root = Path(token_dir)
        val_bin = root / "val.bin"
        train_bin = str(root / "train.bin")
        val_bin = str(val_bin if val_bin.is_file() else root / "train.bin")
        if config.pack_sequences:
            # The Dataset API reads its source directly, so the DataLoader's
            # read threads and buffers have nothing to configure here.
            return get_packed_token_dataset_grain(
                train_bin, val_bin,
                batch_size=config.batch_size,
                seq_len=config.sequence_length,
                seed=config.dataset_seed,
                worker_count=config.worker_count,
                worker_buffer_size=config.worker_buffer_size,
            )
        return get_token_dataset_grain(
            train_bin, val_bin,
            batch_size=config.batch_size,
            seq_len=config.sequence_length,
            seed=config.dataset_seed,
            worker_count=config.worker_count,
            read_thread_count=config.read_thread_count,
            read_buffer_size=config.read_buffer_size,
            worker_buffer_size=config.worker_buffer_size,
        )

    name = config.dataset
    if config.loader == 'grain':
        online = False
    elif config.loader == 'online':
        online = True
    else:
        online = name in onlineDatasetMap and name not in datasetMap

    read_thread_count = config.read_thread_count
    worker_buffer_size = config.worker_buffer_size
    # Enough records for one full validation pass, held out from training so
    # the loop's metrics are not read back off the training records.
    val_count = config.val_steps_per_epoch * config.batch_size
    if online:
        print("Using Online Dataset Generator")
        # Streaming reads are slower per shard than arrayrecord grain reads,
        # so more of them are kept in flight to hold the same throughput.
        read_thread_count *= 4
        worker_buffer_size *= 5
        return get_dataset_online(
            name,
            batch_size=config.batch_size, image_scale=config.image_size,
            worker_count=config.worker_count, read_thread_count=read_thread_count,
            read_buffer_size=config.read_buffer_size,
            worker_buffer_size=worker_buffer_size,
            seed=config.dataset_seed, dataset_source=config.dataset_path,
        )

    if name in datasetMap:
        return get_dataset_grain(
            name,
            batch_size=config.batch_size, image_scale=config.image_size,
            worker_count=config.worker_count, read_thread_count=read_thread_count,
            read_buffer_size=config.read_buffer_size,
            worker_buffer_size=worker_buffer_size,
            seed=config.dataset_seed, dataset_source=config.dataset_path,
            val_count=val_count,
        )
    return get_media_dataset_grain(
        name,
        batch_size=config.batch_size, media_scale=config.image_size,
        worker_count=config.worker_count, read_thread_count=read_thread_count,
        read_buffer_size=config.read_buffer_size,
        worker_buffer_size=worker_buffer_size,
        seed=config.dataset_seed, dataset_source=config.dataset_path,
        val_count=val_count,
    )
