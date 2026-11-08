"""The Stable Diffusion VAE as a release stores it: its torch tensors' names onto the encoder and decoder
`dew.nn.autoencoders.vae` computes, the SD1-era flax msgpack read back to those names, and where a
standalone release keeps it on its revisions (`located_vae`)."""

import json
import logging
import os
from pathlib import Path

import numpy as np

from dew.interop.weights import ParamTree, indexed_path, stored_shape, translate_parameters

from .common import check_class

_log = logging.getLogger(__name__)


_VAE_ATTENTION = {"to_q": "query", "to_k": "key", "to_v": "value"}


_FLAX_ATTENTION = {flax: torch for torch, flax in _VAE_ATTENTION.items()}


def _vae_path(torch_name: str, rank: int) -> tuple[str, ...]:
    """Map one diffusers AutoencoderKL tensor name to its path in this tree.

    Three names differ. diffusers numbers repeated children with a dot
    (`down_blocks.0`) where linen uses an underscore (`down_blocks_0`). It
    calls the mid-block attention's projections `to_q/to_k/to_v/to_out.0`
    where `FlaxAttentionBlock` calls them `query/key/value/proj_attn`. It
    stores every gain as `weight`, where linen has a norm's `scale` and a
    convolution's or a dense's `kernel`, which `rank` tells apart.
    """
    parts = torch_name.split(".")
    leaf = "bias" if parts[-1] == "bias" else ("scale" if rank == 1 else "kernel")
    names = parts[:-1]
    # to_out.0 is the projection, the module's last; there is no second entry.
    if names[-2:] == ["to_out", "0"]:
        names[-2:] = ["proj_attn"]
    return (*indexed_path(_VAE_ATTENTION.get(name, name) for name in names), leaf)


def translate_vae_weights(torch_tensors: ParamTree) -> ParamTree:
    """Convert diffusers AutoencoderKL tensors into this module's param tree.

    Convolution kernels transpose from torch's [out, in, kh, kw] to linen's
    [kh, kw, in, out], and dense kernels from [out, in] to [in, out]; norms
    and biases keep their layout. Every tensor maps, so an unknown name
    raises rather than loading half an autoencoder.
    """
    def stored(name: str) -> np.ndarray:
        tensor = torch_tensors[name]
        if isinstance(tensor, dict):
            raise ValueError(f"{name} is a subtree; a diffusers VAE table is flat")
        return np.asarray(tensor, dtype=np.float32)

    tensors = {name: stored(name) for name in torch_tensors}
    return translate_parameters(tensors, lambda name: _vae_path(name, tensors[name].ndim))


FLAX_REVISIONS = ("bf16", "flax")


FLAX_WEIGHTS = "diffusion_flax_model.msgpack"


TORCH_WEIGHTS = "diffusion_pytorch_model.safetensors"


Candidate = tuple[str, str | None, str | None]
"""A place a VAE may live on the Hub: its weight file, revision and subfolder."""


def _candidates(revision: str) -> list[Candidate]:
    """Every place a load looks, in the order it looks.

    A revision outside `FLAX_REVISIONS` is a pin: both layouts at that
    revision alone, the flax msgpack first, each with and without the `vae`
    subfolder. A layout name (`FLAX_REVISIONS`) looks first for the SD1-era
    flax msgpack at that revision and the `flax` branch, each with and
    without the subfolder, then for the torch safetensors on the repo's
    default branch.
    """
    if revision not in FLAX_REVISIONS:
        return [(weights, revision, subfolder) for weights in (FLAX_WEIGHTS, TORCH_WEIGHTS)
                for subfolder in ("vae", None)]
    return [
        (FLAX_WEIGHTS, *place)
        for place in [(revision, "vae"), ("flax", "vae"), (revision, None), (None, None)]
    ] + [(TORCH_WEIGHTS, *place) for place in [(None, "vae"), (None, None)]]


def _check_weights(modelname: str, candidates: list[Candidate], index: int) -> None:
    """Raise EntryNotFoundError when `candidates[index]` ships no weights, reading no weight bytes.

    Supplied params make the file's presence the only question, which a
    dry run answers online. With the Hub offline (HF_HUB_OFFLINE) the cache
    answers it: a cached file, or the Hub's cached record that the file is
    missing, settles it. When the cache knows neither, the candidate is
    taken only if no later candidate, in either layout, has a different
    cached config that could be the one an online load chose; otherwise
    which config applies cannot be told offline, and a FileNotFoundError
    says so rather than guessing.
    """
    from huggingface_hub import constants, hf_hub_download, try_to_load_from_cache
    from huggingface_hub.errors import EntryNotFoundError

    weights, revision, subfolder = candidates[index]
    if not constants.HF_HUB_OFFLINE:
        hf_hub_download(modelname, weights, revision=revision, subfolder=subfolder, dry_run=True)
        return

    def cached(name: str, place: Candidate):
        """The cached path (str), the Hub's cached miss (another object) or unknown (None)."""
        _, place_revision, place_subfolder = place
        return try_to_load_from_cache(
            modelname,
            name if place_subfolder is None else f"{place_subfolder}/{name}",
            revision=place_revision,
        )

    def recorded_missing(place: Candidate) -> bool:
        found = cached(place[0], place)
        return found is not None and not isinstance(found, str)

    candidate = candidates[index]
    if isinstance(cached(weights, candidate), str):
        return
    if recorded_missing(candidate):
        raise EntryNotFoundError(f"{modelname} has no {weights} at {candidate[1:]}, as the cache records")
    # A later candidate naming the same cached config (None is the default branch) is no rival.
    here = cached("config.json", candidate)
    rivals = [place[1:] for place in candidates[index + 1:]
              if isinstance(cached("config.json", place), str)
              and cached("config.json", place) != here and not recorded_missing(place)]
    if rivals:
        raise FileNotFoundError(
            f"offline, the cache cannot tell which VAE config of {modelname} applies: {weights} at "
            f"{candidate[1:]} is neither cached nor recorded missing, and {rivals} have cached configs too; "
            "load once with the Hub online, or put the weight file in the cache")


def _located(modelname: str, revision: str, params) -> tuple[Path, dict, str] | None:
    """The first candidate with a config and weights, as `located_vae` finds it: its directory in the
    Hub's cache, its config and its weight file's name (fetched where `params` are not supplied); None
    where no candidate has both. A candidate's miss is logged."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError, RevisionNotFoundError

    candidates = _candidates(revision)
    for index, (weights, candidate_revision, subfolder) in enumerate(candidates):
        try:
            config_path = hf_hub_download(modelname, "config.json",
                                          revision=candidate_revision, subfolder=subfolder)
            # A foreign autoencoder's config is refused before any of its weights is fetched.
            with open(config_path) as handle:
                config = json.load(handle)
            check_class(config, PUBLISHED, modelname, nameless=True)
            # Candidate eligibility must match a source load even when a
            # config exists in a revision that carries no matching weights.
            if params is not None:
                _check_weights(modelname, candidates, index)
            else:
                hf_hub_download(modelname, weights, revision=candidate_revision, subfolder=subfolder)
        except (EntryNotFoundError, RevisionNotFoundError) as missing:
            _log.debug("%s has no VAE at %s (%s)", modelname, (weights, candidate_revision, subfolder),
                       missing)
            continue
        return Path(config_path).parent, config, weights
    return None


PUBLISHED = ("AutoencoderKL", "FlaxAutoencoderKL")
"""The `_class_name`s an AutoencoderKL's config states, diffusers' and its Flax port's; an older config
names none (`check_class`'s `nameless`)."""


def located_vae(modelname: str, revision: str = "bf16", *, params=None) -> tuple[Path, dict, str]:
    """Where a standalone AutoencoderKL's config and weights are, local or on the Hub: its directory, its
    config and its weight file's name (`FLAX_WEIGHTS` or `TORCH_WEIGHTS`).

    Two weight layouts reach the same tree. The SD1-era repos ship flax
    `diffusion_flax_model.msgpack`, sometimes under a `vae` subfolder on a
    `flax`/`bf16` revision. Every 16-channel VAE (SD3.5, Flux) ships torch
    `diffusion_pytorch_model.safetensors` only. A local directory holding
    both reads the flax layout. Supplied `params` are authoritative: only the
    configuration and the repository's file metadata are read then, never
    weight bytes. A config naming another autoencoder's class is refused by
    that name before any weight is read (`check_class`).

    `revision` names the flax layout when it is one of `FLAX_REVISIONS`, and
    the torch path then reads the repo's default branch. Any other revision
    is a pin: both layouts are read at that revision alone, and a load raises
    FileNotFoundError when the repo has no weights at it.
    """
    if os.path.isdir(modelname):
        directory = Path(modelname)
        with open(directory / "config.json") as handle:
            config = json.load(handle)
        check_class(config, PUBLISHED, directory, nameless=True)
        return directory, config, FLAX_WEIGHTS if (directory / FLAX_WEIGHTS).exists() else TORCH_WEIGHTS
    located = _located(modelname, revision, params)
    if located is not None:
        return located
    pinned = revision not in FLAX_REVISIONS
    raise FileNotFoundError(
        f"no VAE weights in {modelname}"
        + (f" at revision {revision!r}" if pinned else "")
        + ": neither flax diffusion_flax_model.msgpack nor torch "
        "diffusion_pytorch_model.safetensors"
    )


def flax_vae(msgpack: Path) -> tuple[dict, dict[str, tuple[int, ...]]]:
    """diffusers' Flax VAE params in `msgpack`, the tree this module's is, and each leaf's diffusers
    tensor name with the shape the torch layout stores it in (`_vae_path`'s inverse, `stored_shape`),
    the layouts a bound tree writes back by (`bind_component`'s `shapes`)."""
    from flax.serialization import msgpack_restore
    from flax.traverse_util import flatten_dict

    restored = msgpack_restore(msgpack.read_bytes())
    if not isinstance(restored, dict):
        raise ValueError(f"{msgpack} does not hold a param tree")
    shapes = {}
    for path, leaf in flatten_dict(restored).items():
        if not isinstance(leaf, np.ndarray):
            raise ValueError(f"{msgpack}: {'/'.join(path)} holds no array")
        name, shape = _torch_name(path), stored_shape(path, leaf.shape)
        if _vae_path(name, len(shape)) != path:
            raise ValueError(f"{msgpack}: {'/'.join(path)} names no diffusers tensor ({name} reads into "
                             f"{'/'.join(_vae_path(name, len(shape)))})")
        shapes[name] = shape
    return restored, shapes


def _torch_name(path: tuple[str, ...]) -> str:
    """The diffusers tensor name a leaf at `path` of this tree reads from (`_vae_path`'s inverse)."""
    *names, leaf = path
    tokens: list[str] = []
    for name in names:
        if name == "proj_attn":
            tokens += ["to_out", "0"]
            continue
        name = _FLAX_ATTENTION.get(name, name)
        head, _, index = name.rpartition("_")
        tokens += [head, index] if head and index.isdigit() else [name]
    return ".".join([*tokens, "bias" if leaf == "bias" else "weight"])

