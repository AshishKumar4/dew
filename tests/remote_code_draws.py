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

IMAGE_AT, VIDEO_AT, FRAMES_AT = (3,), (2, 9), (2,)
"""Where each drawn image's placeholder run starts, and each drawn video frame's: a frame's run apart from
the next, as the release's processor sets a frame's timestamp and its bounds between them; or the frames'
one run, side by side, which a release that fills its placeholders by count reads alike (`frames`)."""

AUDIO_FRAMES = (45, 13)
"""A drawn audio row's clips, by their mel frames (`audio`): over MiMo-V2.6's tiny segments of 20 frames,
one cut into 20, 20 and 5, its last segment short, and one of 13, shorter than a segment."""

AUDIO_AT = (2, 9)
"""Where each drawn clip's placeholder run starts."""

SITES = {"lfm3-tiny": ("logits", "packed", "decode", "gradients", "update"), "lfm3-biased-tiny": ("biased",),
         "lfm3-vl-tiny": ("logits", "image", "video"),
         "mimo-v2-tiny": ("logits",),
         "mimo-v2-vl-tiny": ("logits", "image", "video", "frames", "audio", "mtp"),
         "dream-tiny": ("logits",), "llada-tiny": ("logits",)}
"""Each release fixture's sites: its logits over plain rows; LFM3's over a packed row, through the decode
cache in chunks, its parameter gradients under a drawn cotangent, and its logits after one SGD step at
`RATE` on the rows' next-token cross-entropy (`update`), the tied embedding and head stepped once; with
its convolution's bias (`conv_bias`), over left-padded rows; LFM3-VL's and MiMo-V2.6's over a row whose
placeholders a drawn image fills (`image`), or a drawn video's frames (`video`), each packed as its
processor packs it; and MiMo-V2.6's over a row whose video frames' placeholders sit side by side in one run
(`frames`), over one whose placeholders drawn audio clips fill (`audio`), and its prediction layer's over
plain rows (`mtp`)."""


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
    if site in ("image", "video", "frames"):
        return _media(rng, site, config)
    if site == "audio":
        return _audio(rng, config)
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
    """An `image`, `video` or `frames` site's row (`draw`): a video's frames each a run of their own, or
    side by side in one (`frames`, at `FRAMES_AT`)."""
    vision = config["vision_config"]
    merge = vision["spatial_merge_size"]
    grid, starts, token = ((IMAGE_GRID, IMAGE_AT, config["image_token_id"]) if site == "image"
                           else (VIDEO_GRID, FRAMES_AT if site == "frames" else VIDEO_AT,
                                 config["video_token_id"]))
    frames, height, width = grid
    ids = rng.integers(0, min(config["image_token_id"], config["video_token_id"]), (1, LENGTH))
    runs = frames if site == "video" else 1
    for start in starts[:runs]:
        ids[0, start:start + height * width * frames // runs // merge ** 2] = token
    # MiMo-V2.6's vision config names its channels as its ViT does (`in_chans`).
    channels = vision["in_channels"] if "in_channels" in vision else vision["in_chans"]
    patch = channels * vision["temporal_patch_size"] * vision["patch_size"] ** 2
    pixels = rng.normal(size=(frames * height * width, patch)).astype(np.float32)
    kind = "image" if site == "image" else "video"
    name = "" if site == "image" else "_videos"
    return {"ids": ids.astype(np.int32), f"pixel_values{name}": pixels,
            f"{kind}_grid_thw": np.asarray([grid], np.int64)}


def audio_tokens(frames: int, processing: dict) -> int:
    """The placeholders a clip of `frames` mel frames fills, as MiMo-V2.6's processing states it (its
    config's `processor_config`): each segment of `audio_segment_size` frames convolved, halved and pooled
    to its codes, a clip's codes in groups of `audio_group_size`."""
    size, kernel = processing["audio_segment_size"], processing["audio_kernel_size"]

    def codes(length: int) -> int:
        if length == 0:
            return 0
        convolved = (length + 3 - kernel + 2 - kernel) // processing["audio_stride_size"] + 1
        return -(-convolved // processing["audio_avg_pooler"])

    held = frames // size * codes(size) + codes(frames % size)
    return -(-held // processing["audio_group_size"])


def _audio(rng: np.random.Generator, config: dict) -> dict[str, np.ndarray]:
    """An `audio` site's row (`draw`), drawn below every placeholder id: its clips' mel frames, normals,
    padded to the longest with their valid-frame mask (`input_features`, `input_features_mask`), each
    clip's placeholders at `AUDIO_AT`."""
    processing = config["processor_config"]
    ids = rng.integers(0, min(config["image_token_id"], config["video_token_id"], config["audio_token_id"]),
                       (1, LENGTH))
    for start, frames in zip(AUDIO_AT, AUDIO_FRAMES, strict=True):
        ids[0, start:start + audio_tokens(frames, processing)] = config["audio_token_id"]
    features = rng.normal(size=(len(AUDIO_FRAMES), max(AUDIO_FRAMES), processing["audio_n_mels"]))
    mask = np.arange(max(AUDIO_FRAMES))[None, :] < np.asarray(AUDIO_FRAMES)[:, None]
    return {"ids": ids.astype(np.int32), "input_features": np.where(mask[..., None], features, 0).astype(
        np.float32), "input_features_mask": mask}


def positions(documents: tuple[int, ...] = DOCUMENTS) -> np.ndarray:
    """A packed row's positions `[1, sum(documents)]`, each document's from 0."""
    return np.concatenate([np.arange(length) for length in documents])[None].astype(np.int32)
