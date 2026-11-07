"""The sites a remote-code release is recorded at (tools/remote_code_reference.py --record) and its test
reads again: each site's draw (`Site.inputs`, which `indexed_recording`'s `inputs_of` draws again), Dew's
forward over it (`Site.dew`, tests/model_support.py's `remote_code_site`) and the release's own
(`Site.released`), declared together in one table (`SITE_FORMS`). Each release's sites are `SITES`. The
module imports NumPy alone, as the release's own environment, which serves its side of the recording,
holds no JAX: each side's forward imports its framework where it runs.
"""

import dataclasses
import functools
from collections.abc import Callable
from typing import TypedDict

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
         "mimo-v2-tiny": ("logits",), "mimo-v2-sliding-tiny": ("logits",),
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


class TextConfig(TypedDict):
    """A wrapper's text decoder's config, the field a draw reads."""

    vocab_size: int


class VisionConfig(TypedDict, total=False):
    """A wrapper's vision tower's config, the fields a media draw reads: its patches' geometry, its
    channels under the name its ViT gives them (MiMo-V2.6's `in_chans`)."""

    spatial_merge_size: int
    in_channels: int
    in_chans: int
    temporal_patch_size: int
    patch_size: int


class ProcessorConfig(TypedDict):
    """MiMo-V2.6's audio processing, as its config states it (`processor_config`)."""

    audio_segment_size: int
    audio_kernel_size: int
    audio_stride_size: int
    audio_avg_pooler: int
    audio_group_size: int
    audio_n_mels: int


class FixtureConfig(TypedDict, total=False):
    """A release fixture's config.json, the fields a site's draw reads: its vocabulary (its own, or a
    wrapper's text decoder's), and a wrapper's towers, placeholder ids and audio processing."""

    vocab_size: int
    text_config: TextConfig
    vision_config: VisionConfig
    processor_config: ProcessorConfig
    image_token_id: int
    video_token_id: int
    audio_token_id: int


def vocabulary(config: FixtureConfig) -> int:
    """The token ids a fixture's `config` reads: its own `vocab_size`, or its text decoder's (a wrapper's)."""
    text = config.get("text_config")
    return text["vocab_size"] if text is not None else config["vocab_size"]


def draw(site: str, seed: int, config: FixtureConfig) -> dict[str, np.ndarray]:
    """`site`'s inputs at `seed` over the fixture's `config` (`SITE_FORMS`' `inputs`), from a generator
    seeded by the site and the seed."""
    return SITE_FORMS[site].inputs(np.random.default_rng([sum(map(ord, site)), seed]), config)


def _tokens(rng: np.random.Generator, config: FixtureConfig, rows: int = ROWS) -> dict[str, np.ndarray]:
    """`rows` rows of `LENGTH` token ids."""
    return {"ids": rng.integers(0, vocabulary(config), (rows, LENGTH)).astype(np.int32)}


def _cotangent(rng: np.random.Generator, config: FixtureConfig) -> dict[str, np.ndarray]:
    """Rows with a cotangent on the logits of float32 normals."""
    inputs = _tokens(rng, config)
    inputs["cotangent"] = rng.normal(size=(ROWS, LENGTH, vocabulary(config))).astype(np.float32)
    return inputs


def _padded(rng: np.random.Generator, config: FixtureConfig) -> dict[str, np.ndarray]:
    """Rows whose second row's first `PADDED` slots are padding (`attention_mask` 0 there)."""
    inputs = _tokens(rng, config)
    mask = np.ones((ROWS, LENGTH), np.int32)
    mask[1, :PADDED] = 0
    inputs["attention_mask"] = mask
    return inputs


def _media(rng: np.random.Generator, config: FixtureConfig, *, site: str) -> dict[str, np.ndarray]:
    """An `image`, `video` or `frames` site's row: one row, drawn below its placeholder ids, its
    placeholders at `IMAGE_AT` or `VIDEO_AT` (a video's frames each a run of their own, or side by side in
    one, `frames`, at `FRAMES_AT`) beside a normal image's or video's patch vectors (`pixel_values`,
    `pixel_values_videos`), each a patch's channels, then its temporal patch's frames, then its pixels, in
    merge-block order, as the release's processor packs them, and its grid (`image_grid_thw`,
    `video_grid_thw`)."""
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


def audio_tokens(frames: int, processing: ProcessorConfig) -> int:
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


def _audio(rng: np.random.Generator, config: FixtureConfig) -> dict[str, np.ndarray]:
    """An `audio` site's row, drawn below every placeholder id: its clips' mel frames, normals,
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


# Dew's forwards (`Site.dew`): `model` and `variables` a loaded release's (`Pretrained`), in float64 where
# the test's truth asks it (tests/model_support.py's `remote_code_site`), `source` the release itself.

def _dew_logits(model, variables, source):
    """The logits over plain rows."""
    import jax

    return jax.jit(lambda inputs: {"logits": model.apply(variables, inputs["ids"])})


def _dew_packed(model, variables, source):
    """The logits over a packed row, each document's positions from 0."""
    import jax

    from dew.batch import coordinates

    segments = np.concatenate([np.full(length, index + 1) for index, length in enumerate(DOCUMENTS)])
    placed = coordinates(segments=segments[None].astype(np.int32), positions=positions(),
                         max_segments=len(DOCUMENTS) + 1)
    return jax.jit(lambda inputs: {"logits": model.apply(variables, inputs["ids"], coordinates=placed)})


def _dew_decode(model, variables, source):
    """The logits through the decode cache, the rows fed in `CHUNKS`."""
    import jax.numpy as jnp
    from model_support import decode_steps

    made, step = decode_steps(model)

    def decoded(inputs):
        cache, steps, start = made(variables, inputs["ids"].shape[0]), [], 0
        for chunk in CHUNKS:
            logits, cache = step(variables, cache, inputs["ids"][:, start:start + chunk])
            steps.append(logits)
            start += chunk
        return {"logits": jnp.concatenate(steps, axis=1)}
    return decoded


def _dew_gradients(model, variables, source):
    """The parameters' gradient of the logits' product with the cotangent, by each parameter's dotted
    path."""
    import jax
    import jax.numpy as jnp
    from model_support import flat_tree

    def gradients(inputs):
        def loss(params):
            logits = model.apply({**variables, "params": params}, inputs["ids"])
            return jnp.sum(logits * inputs["cotangent"].astype(logits.dtype))
        return dict(flat_tree(jax.grad(loss)(variables["params"])))
    return jax.jit(gradients)


def _dew_update(model, variables, source):
    """The logits after one SGD step at `RATE` on the rows' next-token cross-entropy."""
    import jax
    import optax

    def updated(inputs):
        def loss(params):
            logits = model.apply({**variables, "params": params}, inputs["ids"])
            return optax.softmax_cross_entropy_with_integer_labels(logits[:, :-1],
                                                                   inputs["ids"][:, 1:]).mean()
        params, optimizer = variables["params"], optax.sgd(RATE)
        steps, _ = optimizer.update(jax.grad(loss)(params), optimizer.init(params), params)
        return {"logits": model.apply({**variables, "params": optax.apply_updates(params, steps)},
                                      inputs["ids"])}
    return jax.jit(updated)


def _dew_biased(model, variables, source):
    """The logits over left-padded rows, each row's real slots alone (`row0`, `row1`)."""
    import jax

    def biased(inputs):
        logits = model.apply(variables, inputs["ids"], attention_mask=inputs["attention_mask"].astype(bool))
        return {"row0": logits[0], "row1": logits[1, PADDED:]}
    return jax.jit(biased)


def _dew_mtp(model, variables, source):
    """The language model's first prediction depth's logits over plain rows (`prediction_logits`); a
    wrapper's prediction depths are its language model's."""
    import jax
    from model_support import prediction_logits

    language = getattr(model, "language_model", model)
    held = {collection: tree["language_model"] for collection, tree in variables.items()
            if "language_model" in tree} if language is not model else variables
    return jax.jit(lambda inputs: {"logits": prediction_logits(language, held, inputs["ids"])})


_DRAWN = {"image": "pixel_values", "video": "pixel_values_videos", "frames": "pixel_values_videos",
          "audio": "input_features"}
"""The media field each media site draws, as the release's processor names it."""


def _dew_media(model, variables, source, *, site: str):
    """The logits over a row whose placeholders the drawn media fill, laid out as the source's processor
    reads its own processor's fields (`Processor.from_hf`): the processor's reading of a draw is the same
    for every draw of the site, whose placeholders, grids and clips are fixed (`IMAGE_AT`, `IMAGE_GRID`,
    `AUDIO_FRAMES`), so each patch vector, or each clip's frame, is marked by its index, one past it, where
    `from_hf` places it, which a traced draw's vectors then fill."""
    import jax
    import jax.numpy as jnp

    from dew.nn.inputs import ModelInputs

    template = draw(site, 0, dict(source.config))
    drawn = _DRAWN[site]
    held = "input_features" if site == "audio" else "pixel_values"
    shape = template[drawn].shape
    count = int(np.prod(shape[:-1]))
    marked = np.broadcast_to(np.arange(1, count + 1, dtype=np.float32).reshape(*shape[:-1], 1), shape)
    layout = source.processor.from_hf({"input_ids": template["ids"], **{
        name: marked if name == drawn else value for name, value in template.items() if name != "ids"}})
    places = np.asarray(layout.conditioning[held])[..., 0].astype(np.int64)

    def media(inputs):
        vectors = inputs[drawn].reshape(-1, shape[-1])
        placed = jnp.where((places > 0)[..., None], vectors[np.maximum(places - 1, 0)], 0)
        batch = ModelInputs(inputs["ids"], layout.token_fields,
                            {**layout.conditioning, held: placed.astype(inputs[drawn].dtype)})
        return {"logits": model.apply(variables, batch.tokens, **batch.kwargs())}
    return jax.jit(media)


# The release's forwards (`Site.released`), in its own environment: `model` its torch module, `inputs` a
# draw, `name` the fixture's.

def _inferred(forward):
    """`forward` without autograd's tape: a site that reads no gradient."""
    @functools.wraps(forward)
    def run(model, inputs, name):
        import torch

        with torch.no_grad():
            return forward(model, inputs, name)
    return run


def _ids(inputs):
    import torch

    return torch.from_numpy(inputs["ids"].astype(np.int64))


@_inferred
def _released_logits(model, inputs, name):
    return {"logits": model(input_ids=_ids(inputs)).logits.numpy()}


@_inferred
def _released_packed(model, inputs, name):
    import torch

    return {"logits": model(input_ids=_ids(inputs),
                            position_ids=torch.from_numpy(positions().astype(np.int64))).logits.numpy()}


@_inferred
def _released_decode(model, inputs, name):
    import torch

    ids, cache, steps, start = _ids(inputs), None, [], 0
    for chunk in CHUNKS:
        out = model(input_ids=ids[:, start:start + chunk], past_key_values=cache, use_cache=True)
        cache, start = out.past_key_values, start + chunk
        steps.append(out.logits)
    return {"logits": torch.cat(steps, dim=1).numpy()}


def _released_gradients(model, inputs, name):
    """The gradients by the release's own tensor names."""
    import torch

    model.zero_grad()
    with torch.enable_grad():
        logits = model(input_ids=_ids(inputs)).logits
        (logits * torch.from_numpy(inputs["cotangent"]).to(logits.dtype)).sum().backward()
    return {tensor: parameter.grad.detach().numpy() for tensor, parameter in model.named_parameters()}


def _released_update(model, inputs, name):
    """The logits after one SGD step, the model's weights restored after."""
    import torch

    ids = _ids(inputs)
    weights = {tensor: value.clone() for tensor, value in model.state_dict().items()}
    model.zero_grad()
    with torch.enable_grad():
        logits = model(input_ids=ids).logits
        torch.nn.functional.cross_entropy(logits[:, :-1].flatten(0, 1), ids[:, 1:].flatten()).backward()
    torch.optim.SGD(model.parameters(), lr=RATE).step()
    with torch.no_grad():
        stepped = model(input_ids=ids).logits.numpy()
        model.load_state_dict(weights)
    model.zero_grad()
    return {"logits": stepped}


@_inferred
def _released_biased(model, inputs, name):
    import torch

    logits = model(input_ids=_ids(inputs), attention_mask=torch.from_numpy(inputs["attention_mask"])).logits
    return {"row0": logits[0].numpy(), "row1": logits[1, PADDED:].numpy()}


SPELLED = {"mimo-v2-vl-tiny": {"pixel_values_videos": "video_pixel_values"}}
"""Each release's names for the media fields its forward spells otherwise than its processor's: MiMo-V2.6's
video patches (`MiMoV2ForCausalLM.forward`)."""


@_inferred
def _released_media(model, inputs, name):
    """The patch vectors in the model's dtype, as its processor's float32 ones are cast to it, under the
    release's own spellings (`SPELLED`)."""
    import torch

    dtype = next(model.parameters()).dtype
    spelled = SPELLED.get(name, {})
    media = {spelled.get(field, field): torch.from_numpy(value).to(
        dtype if field.startswith("pixel") else torch.int64)
        for field, value in inputs.items() if field != "ids"}
    return {"logits": model(input_ids=_ids(inputs), **media).logits.numpy()}


@_inferred
def _released_audio(model, inputs, name):
    """Each clip's valid mel frames through the release's own pipeline (`get_audio_feature`: its
    tokenizer's codes, then its encoder), in the model's dtype, filling the placeholders."""
    import torch

    dtype = next(model.parameters()).dtype
    clips = zip(inputs["input_features"], inputs["input_features_mask"], strict=True)
    mels = [torch.from_numpy(clip[valid]).to(dtype) for clip, valid in clips]
    embeds = model.audio_encoder.get_audio_feature(mels, model.speech_embeddings,
                                                   model.audio_tokenizer.encoder)
    return {"logits": model(input_ids=_ids(inputs), audio_embeds=embeds).logits.numpy()}


@_inferred
def _released_mtp(model, inputs, name):
    """The prediction layer's logits: each position's final state beside the next token's embedding."""
    ids = _ids(inputs)
    hidden = model.model(input_ids=ids).last_hidden_state
    embeds = model.model.embed_tokens(ids[:, 1:])
    return {"logits": model.lm_head(model.prediction_layer(embeds, hidden[:, :-1])).numpy()}


@dataclasses.dataclass(frozen=True)
class Site:
    """One recorded site, its three forms declared together: the inputs a seed's generator draws over a
    fixture's config (`inputs`), Dew's forward over a loaded release (`dew`: its model, variables and the
    release, to a draw's outputs by leaf) and the release's own (`released`: its torch module, a draw and
    the fixture's name, to the same leaves)."""

    inputs: Callable[[np.random.Generator, FixtureConfig], dict[str, np.ndarray]]
    dew: Callable
    released: Callable


SITE_FORMS: dict[str, Site] = {
    "logits": Site(_tokens, _dew_logits, _released_logits),
    "packed": Site(functools.partial(_tokens, rows=1), _dew_packed, _released_packed),
    "decode": Site(_tokens, _dew_decode, _released_decode),
    "gradients": Site(_cotangent, _dew_gradients, _released_gradients),
    "update": Site(_tokens, _dew_update, _released_update),
    "biased": Site(_padded, _dew_biased, _released_biased),
    "mtp": Site(_tokens, _dew_mtp, _released_mtp),
    **{site: Site(functools.partial(_media, site=site), functools.partial(_dew_media, site=site),
                  _released_media) for site in ("image", "video", "frames")},
    "audio": Site(_audio, functools.partial(_dew_media, site="audio"), _released_audio),
}
"""Every site, by name (`SITES` names each release's): its logits over plain rows; over a packed row, each
document's positions from 0 (`packed`); through the decode cache in `CHUNKS` (`decode`); the parameters'
gradient of the logits' product with a drawn cotangent (`gradients`); its logits after one SGD step at
`RATE` on the rows' next-token cross-entropy (`update`); over left-padded rows, each row's real slots alone
(`biased`); its prediction layer's over plain rows (`mtp`); and over a row whose placeholders drawn media
fill (`image`, `video`, `frames`, `audio`)."""
