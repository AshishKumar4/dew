"""The draws a remote-code release is recorded over seeds at (tools/remote_code_reference.py --record) and
its test draws again (`indexed_recording`'s `inputs_of`): NumPy alone, as the release's own environment,
which serves its side of the recording, holds no JAX. Each release's sites are `SITES`; each site's inputs
a seed draws are `draw`'s."""

import numpy as np

ROWS, LENGTH = 2, 16
"""Each draw's rows of token ids and their length."""

DOCUMENTS = (3, 5, 8)
"""A packed row's documents (`packed`): two shorter than LFM3's six cached convolution inputs, so a
convolution reads across no document's start."""

CHUNKS = (5, 3, 1, 1, 1, 1, 1, 1, 1, 1)
"""The calls a cached decode (`decode`) feeds its rows' tokens in: a prefill, a second chunk after it, then
one token a call."""

PADDED = 4
"""The left padding of a padded draw's second row (`biased`), shorter than a convolution's history."""

RATE = 0.5
"""The learning rate of an update site's one SGD step (`update`)."""

IMAGE_GRID = (1, 4, 6)
"""A drawn image's patch grid `(t, h, w)` (`image`): six merge blocks of a 2 x 2 merge, rows and columns of
unequal counts, so a grid convolution's rows and columns are told apart."""

VIDEO_GRID = (2, 4, 4)
"""A drawn video's patch grid `(t, h, w)` (`video`): two temporal patches, each a frame of four merge
blocks of a 2 x 2 merge."""

IMAGE_AT, VIDEO_AT = (3,), (2, 9)
"""Where each drawn image's placeholder run starts, and each drawn video frame's: a frame's run apart from
the next, as the release's processor sets a frame's timestamp and its bounds between them."""

SITES = {"lfm3-tiny": ("logits", "packed", "decode", "gradients", "update"), "lfm3-biased-tiny": ("biased",),
         "lfm3-vl-tiny": ("logits", "image", "video"),
         "mimo-v2-tiny": ("logits",), "dream-tiny": ("logits",), "llada-tiny": ("logits",)}
"""Each release fixture's sites: its logits over plain rows; LFM3's over a packed row, through the decode
cache in chunks, its parameter gradients under a drawn cotangent, and its logits after one SGD step at
`RATE` on the rows' next-token cross-entropy (`update`), the tied embedding and head stepped once; with
its convolution's bias (`conv_bias`), over left-padded rows; and LFM3-VL's over a row whose placeholders a
drawn image fills (`image`), or a drawn video's frames (`video`), each packed as its processor packs it."""


def vocabulary(config: dict) -> int:
    """The token ids a fixture's `config` reads: its own `vocab_size`, or its text decoder's (a wrapper's)."""
    return int(config.get("text_config", config)["vocab_size"])


def draw(site: str, seed: int, config: dict) -> dict[str, np.ndarray]:
    """`site`'s inputs at `seed` over the fixture's `config`: `ROWS` rows of `LENGTH` token ids (one row
    of them packed for `packed`), with a cotangent on the logits of float32 normals for `gradients` and the
    second row's first `PADDED` slots padding for `biased` (`attention_mask` 0 there). An `image` or a
    `video` is one row, drawn below its placeholder ids, its placeholders at `IMAGE_AT` or `VIDEO_AT`
    beside a normal image's or video's patch vectors (`pixel_values`, `pixel_values_videos`), each a
    patch's channels, then its temporal patch's frames, then its pixels, in merge-block order, as
    the release's processor packs them, and its grid (`image_grid_thw`, `video_grid_thw`)."""
    rng = np.random.default_rng([sum(map(ord, site)), seed])
    if site in ("image", "video"):
        return _media(rng, site, config)
    rows, tokens = 1 if site == "packed" else ROWS, vocabulary(config)
    inputs = {"ids": rng.integers(0, tokens, (rows, LENGTH)).astype(np.int32)}
    if site == "gradients":
        inputs["cotangent"] = rng.normal(size=(rows, LENGTH, tokens)).astype(np.float32)
    if site == "biased":
        mask = np.ones((rows, LENGTH), np.int32)
        mask[1, :PADDED] = 0
        inputs["attention_mask"] = mask
    return inputs


def _media(rng: np.random.Generator, site: str, config: dict) -> dict[str, np.ndarray]:
    """An `image` or `video` site's row (`draw`)."""
    vision = config["vision_config"]
    merge = vision["spatial_merge_size"]
    grid, starts, token = ((IMAGE_GRID, IMAGE_AT, config["image_token_id"]) if site == "image"
                           else (VIDEO_GRID, VIDEO_AT, config["video_token_id"]))
    frames, height, width = grid
    ids = rng.integers(0, min(config["image_token_id"], config["video_token_id"]), (1, LENGTH))
    runs = 1 if site == "image" else frames
    for start in starts[:runs]:
        ids[0, start:start + height * width * frames // runs // merge ** 2] = token
    patch = vision["in_channels"] * vision["temporal_patch_size"] * vision["patch_size"] ** 2
    pixels = rng.normal(size=(frames * height * width, patch)).astype(np.float32)
    name = "" if site == "image" else "_videos"
    return {"ids": ids.astype(np.int32), f"pixel_values{name}": pixels,
            f"{site}_grid_thw": np.asarray([grid], np.int64)}


def positions(documents: tuple[int, ...] = DOCUMENTS) -> np.ndarray:
    """A packed row's positions `[1, sum(documents)]`, each document's from 0."""
    return np.concatenate([np.arange(length) for length in documents])[None].astype(np.int32)
