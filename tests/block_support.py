"""The block-causal rule a layer reading blocks keeps (`block_causal`), written apart from Dew, and the tiny
decoder the block tests run it on (`model`, and `wrapped` under a multimodal wrapper). Shared by
test_block_visibility and test_prefix_blocks."""

import numpy as np
from model_support import TINY_DECODER

from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.multimodal import MultimodalTransformer
from dew.nn.vision import GemmaProjector, SiglipVision

VOCAB = 37


def model(layers: int = TINY_DECODER["num_layers"], **kinds):
    """A float32 decoder of `layers` layers of the kinds `kinds` names, in turn, each with its fields."""
    types = tuple(kinds) * (layers // len(kinds))
    return CausalTransformer(**{"vocab_size": VOCAB, **TINY_DECODER, "dtype": "float32", "num_layers": layers,
                                "layer_types": types, "kinds": kinds})


def wrapped(decoder):
    """`decoder` under a multimodal wrapper, with a tiny vision tower and projector beside it."""
    return MultimodalTransformer(
        decoder, SiglipVision(hidden_size=16, intermediate_size=32, num_layers=1, num_heads=2, image_size=8,
                              patch_size=4),
        GemmaProjector(text_width=TINY_DECODER["emb_features"], patches_per_side=2, tokens_per_side=1),
        family="gemma3", image_token_id=1)


def block_causal(blocks=None, images=None, *, segments=None, valid=None, window=None) -> np.ndarray:
    """The keys each query keeps, `[B, Q, K]`: a key at or before the query, or of an image of the query's
    (equal ids, -1 none), within a causal window of w, which keeps the keys fewer than w real tokens behind
    the query and every one ahead (Gemma's `AND(sliding_window, OR(causal, blockwise))`); or of a block of
    the query's, whatever the window (transformers' `block_sequence_ids`); inside the query's document (0
    padding, which sees nothing) and among the real keys `valid`."""
    rows, length = np.shape(next(groups for groups in (blocks, images) if groups is not None))

    def same(groups):
        if groups is None:
            return np.zeros((rows, length, length), bool)
        groups = np.asarray(groups)
        return (groups[:, :, None] == groups[:, None, :]) & (groups[:, :, None] >= 0)

    keep = np.tril(np.ones((length, length), bool))[None] | same(images)
    real = np.ones((rows, length), bool) if valid is None else np.asarray(valid, bool)
    if window is not None:
        places = np.maximum(np.cumsum(real, axis=1) - 1, 0)
        keep = keep & ((places[:, :, None] - places[:, None, :]) < window)
    keep = keep | same(blocks)
    if segments is not None:
        segments = np.asarray(segments)
        keep = keep & (segments[:, :, None] == segments[:, None, :]) & (segments[:, :, None] != 0)
    return keep & real[:, None, :]
