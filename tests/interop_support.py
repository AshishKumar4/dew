"""What several interop tests read off a reference: a checkpoint's stored
bytes, a source autoencoder's vector-Jacobian products and the audio features
a checkpoint's own extractor makes; and a Hub cache an offline process
resolves a repo from."""

import functools
import json
import shutil
import tarfile
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from reference_error import widened


def allocated_bytes(monkeypatch, operation):
    """An operation's result and bytes allocated through NumPy.empty while it runs.
    The recorder delegates every allocation; it changes neither the buffers nor the operation."""
    total = 0
    allocate = np.empty

    def recording(*args, **kwargs):
        nonlocal total
        array = allocate(*args, **kwargs)
        total += array.nbytes
        return array

    with monkeypatch.context() as patch:
        patch.setattr(np, "empty", recording)
        value = operation()
    return value, total


TINY_DECODER_FIELDS = {"hidden_size": 64, "num_hidden_layers": 4, "num_attention_heads": 4,
                       "num_key_value_heads": 2, "head_dim": 16, "intermediate_size": 128, "vocab_size": 256,
                       "max_position_embeddings": 64, "tie_word_embeddings": False, "pad_token_id": 0,
                       "bos_token_id": 1, "eos_token_id": 2}
"""A tiny decoder's config fields transformers' classes take, for a tier-2 source made of
transformers' own class (`tiny_config`)."""


def tiny_config(model_type: str, **fields):
    """transformers' config of `model_type` at `TINY_DECODER_FIELDS`, `fields` over them; a field given
    as None is left out, as a released config that does not state it."""
    from transformers import AutoConfig

    stated = {key: value for key, value in {**TINY_DECODER_FIELDS, **fields}.items() if value is not None}
    return AutoConfig.for_model(model_type, **stated)


def drawn_ids(vocab_size: int, seed: int) -> dict[str, np.ndarray]:
    """A seed's token grid for a decoder's logits (dew.interop.verify's BATCH x LENGTH), check 1's draw
    for a recorded transformers reference (`write_logits_recording`, tools/hf_reference.py), the same on
    every run."""
    from dew.interop.verify import BATCH, LENGTH

    rng = np.random.default_rng([0x1D5, vocab_size, seed])
    return {"input_ids": rng.integers(0, vocab_size, (BATCH, LENGTH)).astype(np.int32)}


PROMPT_WORDS = ("a", "red", "cat", "on", "mat", "tiny", "photo", "of", "the", "blue", "sea", "under",
                "light")
"""The words a drawn prompt joins (`drawn_prompts`)."""


def drawn_prompts(salt: int, seed, count: int = 2) -> tuple[str, ...]:
    """`count` prompts of one to eight `PROMPT_WORDS` drawn from `seed` under a site's `salt`."""
    rng = np.random.default_rng([salt, int(seed)])
    return tuple(" ".join(rng.choice(PROMPT_WORDS, size=rng.integers(1, 9))) for _ in range(count))


def float64_logits(loaded):
    """The logits of `loaded` (a `Pretrained`) cloned to float64, its weights widened once, as a function
    of a draw (`drawn_ids`) run where x64 is enabled: a recorded decoder's float64 truth, which its
    recording's sketch holds."""
    model = loaded.model.clone(dtype=jnp.float64, attention_impl="reference")
    variables = widened(loaded.variables)
    return lambda inputs: {"logits": model.apply(variables, jnp.asarray(inputs["input_ids"]))}


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


def without_config_field(source: Path, directory: Path, *path: str) -> Path:
    """`source`'s checkpoint copied to `directory` with the config field at `path` (a key, nested
    keys after it) left out, as a config that leaves it to its default writes it."""
    shutil.copytree(source, directory)
    config = json.loads((directory / "config.json").read_text())
    section = config
    for key in path[:-1]:
        section = section[key]
    del section[path[-1]]
    (directory / "config.json").write_text(json.dumps(config))
    return directory


def fixture_with_meta(path: Path) -> tuple[dict[str, np.ndarray], dict]:
    """A recorded fixture's arrays and its JSON `meta` record, which it stores as bytes."""
    arrays = fixture_arrays(path)
    return arrays, json.loads(arrays.pop("meta").tobytes())


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


def assert_published_checkpoint(measured, name: str) -> None:
    """A published checkpoint's batches as `tools.wrapper_checkpoint_parity.measure_checkpoint` measures
    them: each batch's logits at every position as near its float64 truth as the reference's fp32 run
    (the gate's one-draw rule), and Dew's greedy continuation the reference's wherever every step's top
    reference logit clears the next by the gate's selection margin (`selection_margins.MARGIN`), so that
    float32 rounding cannot pick another token. A continuation with a step nearer a tie decides
    nothing, and is not compared."""
    from reference_error import assert_as_exact_as_the_reference
    from selection_margins import MARGIN, top_margin

    for row in measured:
        assert row.truth.dtype == np.float64
        label = f"published {name} {row.modality}"
        assert_as_exact_as_the_reference(row.dew, row.reference, row.truth, label)
        if np.all(top_margin(np.asarray(row.continuation_logits, np.float64), 1) > MARGIN):
            np.testing.assert_array_equal(row.dew_continuation, row.reference_continuation)


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


def source_transformer(source: Path, name: str, model_class, fields, path) -> tuple:
    """A Diffusers case's transformer read from its source (`source`/`name`/transformer): the model
    `fields(config)` builds, its parameters read through `path`, and their layouts."""
    from dew.interop.diffusion import component_tensors, read_transformer

    config = json.loads((source / name / "transformer" / "config.json").read_text())
    model = model_class(**fields(config, dtype="float32", attention_impl="xla"))
    params, layouts = read_transformer(component_tensors(source / name, "transformer"), path)
    return model, params, layouts


def assert_transformer_exports(source: Path, name: str, layouts, params) -> None:
    """Every tensor a Diffusers case's transformer publishes maps to a layout that exports it bit
    for bit (`assert_layout_tensors`)."""
    from dew.interop.diffusion import component_tensors

    assert_layout_tensors(layouts, {"params": params}, component_tensors(source / name, "transformer"),
                          "transformer/")


def channels_last(array) -> np.ndarray:
    """A channels-first source array `[B, C, ...]` channels last `[B, ..., C]`."""
    return np.moveaxis(np.asarray(array), 1, -1)


def posterior_of(moments) -> tuple[jax.Array, jax.Array]:
    """The posterior's mean and standard deviation, split from an encoder's stacked moments as
    Diffusers' `DiagonalGaussianDistribution` splits them."""
    from dew.nn.autoencoders.kl import posterior_moments

    mean, log_variance = posterior_moments(moments)
    return mean, jnp.exp(0.5 * log_variance)


def assert_coder_vjps(loaded, reference, operand: str, posterior, bound: float) -> None:
    """A source KL autoencoder's recorded encode and decode vector-Jacobian products
    (`assert_autoencoder_vjp`) for its `loaded` (autoencoder, parameters, layouts, config):
    `posterior(model, params, x)`'s mean and standard deviation against the source's `probe_mean`
    and `probe_std` from its `operand` (`image`, `video`), and the decode against its `probe` from
    its `latent`, each array channels last."""
    autoencoder, params, layouts, _ = loaded
    model = autoencoder.model

    def decode(params, latent):
        return model.apply({"params": params}, latent, method=model.decode)

    assert_autoencoder_vjp(
        "encode", functools.partial(posterior, model), params, channels_last(reference[operand]),
        [channels_last(reference[f"probe_{name}"]) for name in ("mean", "std")], layouts, reference,
        channels_last(reference[f"encode.grad_{operand}"]), bound)
    assert_autoencoder_vjp(
        "decode", decode, params, channels_last(reference["latent"]), [channels_last(reference["probe"])],
        layouts, reference, channels_last(reference["decode.grad_latent"]), bound)


def recorded_gradients(part, precision: str, inputs, names) -> np.ndarray:
    """A source's recorded gradients at `precision` (`fp32`, `fp64`), raveled in one order: its
    `inputs`' (`{precision}.grad_<input>`) then its parameters' (`{precision}.grad_param.<name>`),
    each read through `part`."""
    leaves = [part(f"{precision}.grad_{name}") for name in inputs]
    leaves += [part(f"{precision}.grad_param.{name}") for name in names]
    return np.concatenate([np.ravel(leaf) for leaf in leaves])


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

