"""What several interop tests read off a reference: a checkpoint's stored
bytes, a source autoencoder's vector-Jacobian products, the audio features
a checkpoint's own extractor makes and the pad row of its token lookup; a tiny
checkpoint's transformers reference run in the test over check 1's draws; and a
Hub cache an offline process resolves a repo from."""

import contextlib
import functools
import json
import shutil
import tarfile
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from reference_error import as_given, assert_as_exact, widened


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


def fixture_ids(directory: Path, mask=None, shape=None):
    """Check 1's draw over the tiny checkpoint at `directory`: a seed's ids uniform over its config's
    vocabulary in the shape of its `input_ids.npy` (or `shape`), 0 where `mask` is false (the pad its
    recorder wrote into padded rows), the same on every run."""
    shape = np.load(directory / "input_ids.npy").shape if shape is None else shape
    vocabulary = json.loads((directory / "config.json").read_text())["vocab_size"]

    def draw(seed):
        ids = np.random.default_rng(seed).integers(0, vocabulary, shape).astype(np.int32)
        return (ids if mask is None else np.where(mask, ids, 0).astype(np.int32),)

    return draw


def transformers_model(directory: Path, *, masked: bool = False, dtype: str = "float32", **options):
    """transformers' own class for the checkpoint at `directory`, as its config names it, the masked LM
    where `masked` and else the causal LM, in `dtype` (float32 unless a narrower run is the reference)
    on eager attention and eager experts, in eval mode, `options` given to `from_pretrained` too: the
    reference a tiny checkpoint's recorder ran (tools/hf_reference.py), run in the test."""
    import torch
    import transformers

    auto = transformers.AutoModelForMaskedLM if masked else transformers.AutoModelForCausalLM
    return auto.from_pretrained(directory, dtype=getattr(torch, dtype), attn_implementation="eager",
                                experts_implementation="eager", **options).eval()


def transformers_arms(model, run, *, source=None, twin=None):
    """Check 1's reference and truth over the transformers `model`: `run(model, *inputs)` under
    `torch.no_grad` on a draw's inputs as torch tensors (integers as int64), by `model` and by the float64
    twin (`tools.torch_precision.float64_twin`) of `source`, by default `model`, inside `float64`, each
    run `as_given`, its tensors returned as NumPy arrays, a bfloat16 one as the float32 it holds exactly.
    A narrower `model` (bfloat16) is measured from its stored weights' truth: `source`, the float32
    load. `twin` is the twin itself where the reference builds its own (a PEFT model's)."""
    import torch

    from tools.torch_precision import float64, float64_twin

    def tensor(value):
        value = np.asarray(value)
        return torch.from_numpy(value.astype(np.int64) if np.issubdtype(value.dtype, np.integer) else value)

    def arm(network, precision):
        @as_given
        def given(*inputs):
            with precision(), torch.no_grad():
                return jax.tree.map(lambda out: (out.float() if out.dtype == torch.bfloat16 else out).numpy(),
                                    run(network, *map(tensor, inputs)))
        return given

    if twin is None:
        twin = float64_twin(model if source is None else source)
    return arm(model, contextlib.nullcontext), arm(twin, float64)


def peft_twin(model):
    """A PEFT model's float64 twin (`tools.torch_precision.float64_twin`): the same base and adapter
    built inside `float64` by transformers' and PEFT's own builders, so every value their constructors
    compute is float64's (the rotary tables' among them), loaded with `model`'s weights, to run inside
    `float64` (`transformers_arms`' `twin`)."""
    import copy

    import torch
    from peft import get_peft_model
    from transformers import AutoModelForCausalLM

    from tools.torch_precision import float64_twin

    def build():
        config = copy.deepcopy(model.get_base_model().config)
        base = AutoModelForCausalLM.from_config(config, dtype=torch.float64)
        return get_peft_model(base, copy.deepcopy(model.peft_config["default"]))

    return float64_twin(model, build)


def next_token_losses(model, ids):
    """The torch causal LM `model`'s cross entropy at each next token of `ids` in its logits' precision,
    the per-token losses `LMObjective.token_scores` returns (a `run` of `transformers_arms`)."""
    import torch

    logits = model(ids, use_cache=False).logits
    return torch.nn.functional.cross_entropy(logits[:, :-1].transpose(1, 2), ids[:, 1:], reduction="none")


def extract_fixture(archive: Path, directory: Path) -> Path:
    with tarfile.open(archive) as source:
        source.extractall(directory, filter="data")
    return directory


def hub_cache(cache: Path, repo_id: str, source: Path,
              commit: str = "0123456789abcdef0123456789abcdef01234567") -> str:
    """Put `source` in `cache` as `repo_id` at `commit`, as a download leaves
    it: the snapshot, the branch's commit and the commit's tree listing
    (huggingface_hub's tree cache). A process with HF_HUB_CACHE=cache and
    HF_HUB_OFFLINE=1 then resolves `repo_id` to that commit, which this returns."""
    from huggingface_hub._tree_cache import TREE_CACHE_FORMAT_VERSION

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


def respelled(source: Path, directory: Path, edit) -> Path:
    """`source`'s checkpoint copied to `directory` under the config `edit` returns from its own."""
    shutil.copytree(source, directory)
    (directory / "config.json").write_text(json.dumps(edit(json.loads((source / "config.json").read_text()))))
    return directory


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


def written_tensors(directory) -> dict[str, np.ndarray]:
    """Every tensor a saved checkpoint holds, by its name in its file: its safetensors shards and a slot
    readout's `readout_head.pt` where it writes one."""
    from safetensors.numpy import load_file

    held = {}
    for shard in sorted(Path(directory).glob("*.safetensors")):
        held.update(load_file(str(shard)))
    if (Path(directory) / "readout_head.pt").is_file():
        import torch

        held.update({name: tensor.numpy() for name, tensor in
                     torch.load(Path(directory) / "readout_head.pt", weights_only=True).items()})
    return held


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


def clearing_selections(loaded, draw, label: str):
    """`draw`, for a decoder that selects discontinuously (a mixture's routing, Gemma 3n's gaussian
    top-k), redrawn until its selections clear `selection_margins.MARGIN`, so the float32 runs and the
    float64 run select alike (`selection_margins.clearing`); `draw` itself otherwise."""
    import jax.numpy as jnp
    import selection_margins

    if not selection_margins.selects(loaded.model_config):
        return draw
    captured = selection_margins.capture(loaded.model)

    def clears(inputs):
        selections = captured(loaded.variables, jnp.asarray(inputs[0]))
        return selection_margins.margin(loaded.model_config, selections, label) > selection_margins.MARGIN

    return selection_margins.clearing(draw, clears)


def torch_logits(source, precision=None):
    """A transformers causal LM's logits over int ids as float64 numpy, inside `precision` (a context
    manager factory, `tools.torch_precision.float64` for a float64 twin, whose code's float32 casts it
    widens), with no gradient."""
    import contextlib

    import torch

    def run(ids):
        with (precision or contextlib.nullcontext)(), torch.no_grad():
            return source(torch.from_numpy(np.asarray(ids)).long()).logits.double().numpy()
    return as_given(run)


def token_table(variables) -> tuple[tuple[str, ...], np.ndarray]:
    """The path of a decoder's token table among its params, and the table."""
    from flax.traverse_util import flatten_dict

    params = flatten_dict(variables["params"])
    (path,) = [path for path in params if path[-2:] == ("embed_tokens", "embedding")]
    return path, np.asarray(params[path])


def assert_pads_as(model, variables, lookup, wide, pad: int, label: str, *, seeds: int,
                   looked_up=lambda ids: ids) -> None:
    """A decoder's token lookup is its source's (ledger 1593): `model` pads the row `lookup`, the
    source's own token embedding module holding `model`'s table, pads (its `padding_idx`, None where
    the source passes none), and over drawn ids, a quarter of them `pad`, under a drawn cotangent on
    the looked-up rows, the table's gradient holds check 1 against that module's lookup
    (`nn.Embedding.forward`, before any scale a subclass applies) of the ids the source hands it
    (`looked_up` of the drawn ones), from its float64 twin `wide`: the pad row's lookups add nothing
    where it pads and train it where not."""
    import contextlib

    import torch
    from flax.traverse_util import flatten_dict

    from tools.torch_precision import float64

    padded = lookup.padding_idx is not None
    assert model.padding_idx == lookup.padding_idx in (pad, None)
    path, table = token_table(variables)
    vocab, width = table.shape

    def draw(seed):
        rng = np.random.default_rng([0x1593, seed])
        ids = rng.integers(0, vocab, (8, 32))
        ids[rng.random(ids.shape) < 0.25] = pad
        ids[0, 0] = pad
        return ids.astype(np.int32), rng.standard_normal((*ids.shape, width)).astype(np.float32)

    @jax.jit
    def gradient(variables, ids, cotangent):
        def loss(params):
            looked_up = model.apply({**variables, "params": params}, ids, method="token_embeddings")
            return jnp.sum(looked_up * cotangent)
        return flatten_dict(jax.grad(loss)(variables["params"]))[path]

    @as_given
    def ours(ids, cotangent):
        return {"token table": np.asarray(gradient(variables, ids, cotangent), np.float64)}

    def theirs(module, precision=contextlib.nullcontext):
        def run(ids, cotangent):
            with precision():
                module.zero_grad()
                rows = torch.nn.Embedding.forward(module, torch.from_numpy(looked_up(ids)).long())
                (rows * torch.from_numpy(cotangent).to(rows.dtype)).sum().backward()
                return {"token table": module.weight.grad.double().numpy()}
        return as_given(run)

    ids, cotangent = draw(0)
    assert np.any(ours(ids, cotangent)["token table"][pad]) != padded
    assert_as_exact(ours, theirs(lookup), draw, f"{label} token lookup gradient", seeds=seeds,
                    truth_fn=theirs(wide, float64))


def token_draw(salt: int, vocab: int, shape: tuple[int, ...]):
    """Check 1's draw of `shape` token ids below `vocab`, a stream of its own by `salt`."""
    def draw(seed):
        return (np.random.default_rng([salt, seed]).integers(0, vocab, shape).astype(np.int32),)
    return draw


def reference_read(case, directory, *, wide: bool):
    """transformers' read of an export (`decoder_export_reference.reference_forward`) in fp32 or, `wide`, in
    float64, run as given on a draw's ids."""
    from reference_error import as_given

    from tools.decoder_export_reference import reference_forward

    forward = reference_forward(case, directory, wide=wide)
    return as_given(lambda ids: forward(np.asarray(ids)))
