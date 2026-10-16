"""What several interop tests read off a reference: a checkpoint's stored
bytes, a source autoencoder's vector-Jacobian products and the audio features
a checkpoint's own extractor makes; and a Hub cache an offline process
resolves a repo from."""

import json
import shutil
import tarfile
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


def extract_fixture(archive: Path, directory: Path) -> Path:
    with tarfile.open(archive) as source:
        source.extractall(directory, filter="data")
    return directory


def hub_cache(cache: Path, repo_id: str, source: Path) -> str:
    """Put `source` in `cache` as `repo_id` at one commit, as a download leaves
    it: the snapshot, the branch's commit and the commit's tree listing
    (huggingface_hub's tree cache). A process with HF_HUB_CACHE=cache and
    HF_HUB_OFFLINE=1 then resolves `repo_id` to that commit, which this returns."""
    from huggingface_hub._tree_cache import TREE_CACHE_FORMAT_VERSION

    commit = "0123456789abcdef0123456789abcdef01234567"
    repo = cache / f"models--{repo_id.replace('/', '--')}"
    shutil.copytree(source, repo / "snapshots" / commit)
    (repo / "refs").mkdir()
    (repo / "refs" / "main").write_text(commit)
    (repo / "trees").mkdir()
    files = sorted(path for path in source.rglob("*") if path.is_file())
    (repo / "trees" / f"{commit}.json").write_text(json.dumps({
        "format_version": TREE_CACHE_FORMAT_VERSION,
        "files": {path.relative_to(source).as_posix(): {"size": path.stat().st_size,
                                                         "blob_id": f"{index:040x}"}
                  for index, path in enumerate(files)}}))
    return commit


def fixture_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        return dict(source)


def fetch(url: str, start: int, end: int) -> bytes:
    import requests
    response = requests.get(url, headers={"Range": f"bytes={start}-{end}"}, timeout=120)
    response.raise_for_status()
    return response.content


def unpacked(packed: np.ndarray, rows: int, columns: int) -> np.ndarray:
    """`FluxPipeline._unpack_latents`, read into NHWC: a position's channels
    run channel-major over its own two rows and columns."""
    batch, _, width = packed.shape
    channels = width // 4
    grouped = packed.reshape(batch, rows, columns, channels, 2, 2)
    return grouped.transpose(0, 1, 4, 2, 5, 3).reshape(batch, rows * 2, columns * 2, channels)


def packed(latent: np.ndarray) -> np.ndarray:
    """`FluxPipeline._pack_latents` from NHWC."""
    batch, height, width, channels = latent.shape
    grouped = latent.reshape(batch, height // 2, 2, width // 2, 2, channels)
    return grouped.transpose(0, 1, 3, 5, 2, 4).reshape(batch, height * width // 4, channels * 4)


def assert_same_stored_tensors(written: dict, original: dict) -> None:
    """`written` holds every tensor of `original` and no other, each with its
    dtype, shape and bytes; a scalar's bytes are read through a flat view."""
    assert set(written) == set(original)
    for name, value in original.items():
        assert (written[name].dtype, written[name].shape) == (value.dtype, value.shape), name
        np.testing.assert_array_equal(written[name].reshape(-1).view(np.uint8),
                                      value.reshape(-1).view(np.uint8), err_msg=name)


def assert_reloads(loaded, saved: Path, ids: np.ndarray) -> None:
    """The checkpoint `loaded` wrote to `saved` loads back to the model, tree and logits over `ids`
    it was saved from."""
    from dew.interop import Pretrained

    assert_restores(loaded.model, loaded.variables,
                    Pretrained.load(saved, dtype='float32', attention_impl='reference'), ids)


def assert_restores(model, variables, restored, ids: np.ndarray) -> None:
    """`restored`, a checkpoint read back, is `model` with `variables` bit for bit, so it computes
    the same logits over `ids`."""
    assert restored.model == model, "the checkpoint rebuilds a different model"
    assert jax.tree.structure(restored.variables) == jax.tree.structure(variables)
    for original, again in zip(jax.tree.leaves(variables), jax.tree.leaves(restored.variables), strict=True):
        np.testing.assert_array_equal(np.asarray(again), np.asarray(original))
    np.testing.assert_array_equal(restored.model.apply(restored.variables, ids), model.apply(variables, ids))


def assert_layout_tensors(layouts, variables, tensors, prefix: str) -> None:
    """Every source tensor has a layout that exports its dtype, shape and bytes."""
    assert {layout.name for layout in layouts} == {f"{prefix}{name}" for name in tensors}
    assert_same_stored_tensors(
        {layout.name.removeprefix(prefix): layout.export(variables) for layout in layouts}, tensors)


def scaled_gap(actual, expected) -> float:
    """Maximum float64 error, scaled by max(1, the reference's magnitude)."""
    expected = np.asarray(expected, np.float64)
    difference = np.abs(np.asarray(actual, np.float64) - expected).max()
    return float(difference / max(1.0, float(np.abs(expected).max())))


def assert_autoencoder_vjp(walk: str, call, params, operand, probes, layouts, reference, operand_gradient,
                           bound: float):
    """A source autoencoder's recorded vector-Jacobian product of one `walk`: `call(params, operand)`'s
    outputs dotted with `probes`, differentiated for the operand and the parameters. The operand's
    gradient is held to the source's `operand_gradient`, and each of `layouts`' parameters' gradient,
    written into its source tensor's layout as an export writes it, to the source's
    `{walk}.grad.<tensor>`, each within `bound` (`scaled_gap`). Returns the outputs."""
    def objective(params, operand):
        outputs = call(params, operand)
        return sum(jnp.sum(output * probe) for output, probe in
                   zip(jax.tree.leaves(outputs), probes, strict=True)), outputs

    (_, outputs), (grad_params, grad_operand) = jax.value_and_grad(objective, argnums=(0, 1), has_aux=True)(
        params, operand)
    operand_gap = scaled_gap(grad_operand, operand_gradient)
    gaps = {layout.name: scaled_gap(layout.export({"autoencoder": grad_params}),
                                    reference[f"{walk}.grad.{layout.name.removeprefix('vae/')}"])
            for layout in layouts}
    worst = max(gaps, key=lambda name: gaps[name])
    print(f"{walk}: input gradient gap {operand_gap:.3g}; worst parameter {worst} {gaps[worst]:.3g}")
    assert operand_gap < bound
    assert gaps[worst] < bound, worst
    return outputs



def gemma_features(model_type, config, waveforms):
    """The float32 `input_features` and boolean `input_features_mask` the
    checkpoint's own Transformers extractor makes of 16 kHz `waveforms`,
    with the arguments `Processor` passes it: padded to the longest, 30 s
    at most, frames a multiple of the encoder's 128."""
    from transformers import Gemma3nAudioFeatureExtractor, Gemma4AudioFeatureExtractor

    extractor = {"gemma3n_audio": Gemma3nAudioFeatureExtractor,
                 "gemma4_audio": Gemma4AudioFeatureExtractor}[model_type].from_dict(dict(config))
    features = extractor([np.asarray(waveform, np.float32) for waveform in waveforms],
                         padding="longest", max_length=480000, truncation=True,
                         pad_to_multiple_of=128, return_tensors="np", return_attention_mask=True)
    return {"input_features": np.asarray(features["input_features"], np.float32),
            "input_features_mask": np.asarray(features["input_features_mask"], np.bool_)}


NVFP4_QDQ = Path(__file__).parent / "fixtures" / "codecs" / "nvfp4_qdq"
"""Two compressed-tensors NVFP4 checkpoints of one tiny Qwen3 with input QDQ, one per local-scale
rounding, each with the author's reference forward (`reference.npz`)."""

NVFP4_QDQ_CASES = ("unrounded", "e4m3")


def nvfp4_qdq_case(kind: str) -> tuple[Path, dict[str, np.ndarray]]:
    """One NVFP4 QDQ checkpoint's directory and its reference arrays."""
    directory = NVFP4_QDQ / kind
    with np.load(directory / "reference.npz") as reference:
        return directory, dict(reference)

