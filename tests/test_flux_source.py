"""Native Flux against the actual Diffusers 0.34.0 objects it reconstructs.

`tools/diffusers_flux_reference.py` builds tiny `FluxTransformer2DModel`
instances, saves each with its real config and safetensors, and runs each the
way its pipeline runs it - packed latents, the ids the pipeline lays out, the
timestep it has divided by the training count and the distilled guidance a
guidance-embedded checkpoint takes - recording the forward, the
vector-Jacobian products of the latent, the text tokens and the pooled
vector, and the gradient of every parameter, in float32 and in float64. Each
native tensor is held to the source's own float32 distance from float64
(tests/reference_error.py).

The native model owns its pipeline's 2x2 packing, so it works in latents.
These tests convert between the two layouts with the source's own permutation
written out here, rather than with the functions under test.
"""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from interop_support import extract_fixture, packed, unpacked
from reference_error import GateFailure, as_given, assert_as_exact, torch_with_gradients, with_gradients

from dew.diffusion.process import DenoisingCondition
from dew.interop.diffusion import component_tensors, flux_fields, flux_path, read_transformer
from dew.nn.backbones.flux import FluxTransformer

ROOT = Path(__file__).resolve().parents[1]
CASES = ("schnell", "dev", "rect", "deep", "mixed")
CASE_SEEDS = {"schnell": 256, "dev": 256, "rect": 256, "deep": 128, "mixed": 256}
"""Each transformer's declared seeds for check 1 (`assert_as_exact`), sized by `seeds_for` (2026-10-21,
on armada)."""


@pytest.fixture(scope="module")
def source(tmp_path_factory):
    return extract_fixture(ROOT / "tests/fixtures/flux_source.tar.xz", tmp_path_factory.mktemp("flux-source"))


def flux_walk(source, name: str, grid: tuple[int, int]):
    """A compiled native forward on a draw, with the fixture's weights, and the gradients of the packed
    latent, the context, the pooled embedding and every parameter under the draw's `probe`, its
    cotangent (`with_gradients`)."""
    config = json.loads((source / name / "transformer" / "config.json").read_text())
    model = FluxTransformer(**flux_fields(config, attention_impl="xla"))
    params, layouts = read_transformer(component_tensors(source / name, "transformer"), flux_path)

    @jax.jit
    def walk(inputs):
        guidance = inputs["guidance"] if config["guidance_embeds"] else None

        def forward(latent, context, pooled, params):
            return packed(model.apply({"params": params}, unpacked(latent, *grid), inputs["times"],
                                      DenoisingCondition(context, pooled, guidance=guidance)))

        return with_gradients(forward)(inputs["packed"], inputs["context"], inputs["pooled"], params,
                                       inputs["probe"])

    return walk, {entry.name: entry for entry in layouts}


@pytest.mark.parametrize("name", CASES)
def test_native_flux_matches_the_source_forward_and_every_gradient(name, source):
    """The forward, the three input gradients and every parameter's, each
    as near float64 as the unmodified source's float32 walk.

    The variants are the ones whose wiring differs - the schnell-style model
    with no guidance embedder, the distilled one with it, a rectangular packed
    grid whose row and column rotations differ, a different split between
    double and single blocks, and one walked at a different distilled
    guidance per row.
    """
    import torch
    from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel

    from tools.torch_precision import float64

    record = json.loads((source / "flux_transformer.json").read_text())
    case = record["cases"][name]
    grid = tuple(case["grid"])
    checkpoint = source / name / "transformer"
    config = json.loads((checkpoint / "config.json").read_text())
    walk, layout = flux_walk(source, name, grid)
    with np.load(source / "flux_transformer.npz") as arrays:
        original = {key: np.array(arrays[f"{name}.{key}"])
                    for key in ("packed", "context", "pooled", "times", "guidance", "probe")}
        prefix = f"{name}.grad_param."
        tensors = [key[len(prefix):] for key in arrays.files if key.startswith(prefix)]
    assert tensors

    batch, tokens = case["batch"], record["tokens"]
    rows, columns = grid
    shape = (batch, rows * columns, config["in_channels"])

    def draw(seed):
        """The fixture's inputs at seed 0, else the generator's normal draws at `seed`, with
        each embedded guidance value perturbed by a unit-normal draw."""
        if seed == 0:
            return ({key: np.asarray(value, np.float32) for key, value in original.items()},)
        generator = torch.Generator().manual_seed(24 + seed)
        guidance = (torch.tensor(case["guidance"][:batch], dtype=torch.float32)
                    if config["guidance_embeds"] else torch.zeros(0))
        inputs = {
            "packed": torch.randn(shape, generator=generator),
            "context": torch.randn((batch, tokens, config["joint_attention_dim"]), generator=generator),
            "pooled": torch.randn((batch, config["pooled_projection_dim"]), generator=generator),
            "times": torch.from_numpy(original["times"]),
            "probe": torch.randn(shape, generator=generator),
        }
        if guidance.numel():
            guidance = guidance + torch.randn(guidance.shape, generator=generator)
        return ({**{key: value.numpy() for key, value in inputs.items()}, "guidance": guidance.numpy()},)

    narrow_model = FluxTransformer2DModel.from_pretrained(
        checkpoint, torch_dtype=torch.float32, local_files_only=True).eval()
    with float64():
        wide_model = FluxTransformer2DModel.from_pretrained(
            checkpoint, torch_dtype=torch.float64, local_files_only=True).eval()
    positions = np.zeros((rows * columns, 3))
    positions[:, 1] = np.repeat(np.arange(rows), columns)
    positions[:, 2] = np.tile(np.arange(columns), rows)

    weights = {model: {key: value.detach().numpy() for key, value in model.named_parameters()}
               for model in (narrow_model, wide_model)}

    def reference_walk(model, drawn):
        """The source's forward in the draw's dtype, and the same gradients under the same cotangent
        as `flux_walk`'s (`torch_with_gradients`)."""
        dtype = torch.from_numpy(np.asarray(drawn["packed"])).dtype
        times, guidance = (torch.tensor(drawn[key], dtype=dtype) for key in ("times", "guidance"))

        def forward(latent, context, pooled, params):
            return torch.func.functional_call(model, params, (), {
                "hidden_states": latent, "encoder_hidden_states": context, "pooled_projections": pooled,
                "timestep": times / 1000, "guidance": guidance if guidance.numel() else None,
                "txt_ids": torch.zeros(tokens, 3, dtype=dtype),
                "img_ids": torch.tensor(positions, dtype=dtype), "return_dict": False})[0]

        return torch_with_gradients(forward)(drawn["packed"], drawn["context"], drawn["pooled"],
                                             weights[model], drawn["probe"])

    @as_given
    def truth(inputs):
        with float64():
            return reference_walk(wide_model, inputs)

    @as_given
    def native(inputs):
        *values, gradients = walk(inputs)
        return (*values, {tensor: layout["transformer/" + tensor].export({"params": gradients})
                          for tensor in tensors})

    assert_as_exact(native, as_given(lambda inputs: reference_walk(narrow_model, inputs)), draw,
                    f"Flux {name}", seeds=CASE_SEEDS[name], truth_fn=truth)


@pytest.fixture(scope="module")
def pipeline_record(source):
    return json.loads((source / "flux_transformer.json").read_text())["pipeline"]


WORDS = ("a", "red", "green", "blue", "cat", "bird", "dog", "tiny", "photo", "of", "tall", "tree")
"""The words a drawn prompt is made of, every one in the pipeline's tokenizers."""
PIPELINE_SEEDS = 8
"""The pipeline sites' declared seeds, sized by `seeds_for` paired as each check confirmed them."""


def prompts(indices: np.ndarray) -> list[dict[str, str]]:
    """A draw's prompts: each row's words in `WORDS` for its CLIP slot (`text`) and its T5 slot
    (`second`)."""
    def words(slot):
        return " ".join(WORDS[index] for index in slot)
    return [{"text": words(row[0]), "second": words(row[1])} for row in np.asarray(indices)]


def prompt_draw(seed: int) -> tuple[np.ndarray]:
    """Two rows' prompts as word indices, three words in each slot, the slots apart."""
    return (np.random.default_rng([185, seed]).integers(0, len(WORDS), (2, 2, 3)).astype(np.int32),)


@pytest.fixture(scope="module")
def pipelines(source):
    """The saved pipeline Diffusers loads in float32, and in float64 inside `float64`."""
    import torch
    from diffusers import FluxPipeline

    from tools.torch_precision import float64

    def read(dtype):
        return FluxPipeline.from_pretrained(source / "pipeline", torch_dtype=dtype, local_files_only=True)

    narrow = read(torch.float32)
    with float64():
        wide = read(torch.float64)
    for pipe in (narrow, wide):
        pipe.set_progress_bar_config(disable=True)
    return narrow, wide


def source_arms(pipe, run, *, wide: bool):
    """`run(pipe, *inputs)` as given under `torch.no_grad`, inside `float64` for the `wide` pipeline, its
    tensors returned as NumPy arrays."""
    import contextlib

    import torch

    from tools.torch_precision import float64

    @as_given
    def given(*inputs):
        with float64() if wide else contextlib.nullcontext(), torch.no_grad():
            return jax.tree.map(lambda out: out.numpy() if isinstance(out, torch.Tensor) else out,
                                run(pipe, *inputs))
    return given


def test_published_flux_prompt_encoding_matches_the_source_pipeline(source, pipeline_record, pipelines):
    """The conditioner composes what `encode_prompt` composes, on par with it over drawn prompts (check 1),
    its float64 run the truth.

    Flux reads its T5 tower's states as the sequence and its CLIP tower's pooled row unprojected, and its
    pipeline routes `prompt` to CLIP and `prompt_2` to T5. The two slots carry different words, so a native
    encoder that crossed them fails the same check.
    """
    import torch

    from dew.interop.pretrained import Pretrained

    loaded = Pretrained.load(str(source / "pipeline"), dtype="float32", attention_impl="xla")
    encoder = loaded.inputs.conditions["conditioning"].encoder
    params = loaded.variables["encoders"]["conditioning"]

    def encoded(rows):
        condition = encoder.encode(params, encoder.tokenize(rows))
        return {"context": condition.context, "pooled": condition.pooled}

    def encode_prompt(pipe, indices):
        rows = prompts(indices)
        embeds, pooled, _ = pipe.encode_prompt(
            prompt=[row["text"] for row in rows], prompt_2=[row["second"] for row in rows],
            device=torch.device("cpu"), num_images_per_prompt=1, max_sequence_length=512)
        return {"context": embeds, "pooled": pooled}

    narrow, wide = pipelines
    reference = source_arms(narrow, encode_prompt, wide=False)
    truth = source_arms(wide, encode_prompt, wide=True)
    assert_as_exact(as_given(lambda indices: encoded(prompts(indices))), reference, prompt_draw,
                    "Flux prompt encoding", seeds=PIPELINE_SEEDS, truth_fn=truth)
    # The distilled guidance rides in as a model input, at the value the pipeline defaults to, and guidance
    # is not two branches here.
    condition = encoder.encode(params, encoder.tokenize(prompts(prompt_draw(0)[0])))
    np.testing.assert_array_equal(np.asarray(condition.guidance),
                                  np.full(2, pipeline_record["default_guidance"]))
    crossed = as_given(lambda indices: encoded([{"text": row["second"], "second": row["text"]}
                                                for row in prompts(indices)]))
    with pytest.raises(GateFailure):
        assert_as_exact(crossed, reference, prompt_draw, "crossed Flux prompt encoding",
                        seeds=PIPELINE_SEEDS, truth_fn=truth)


@pytest.mark.parametrize("streamed", [False, True])
def test_published_flux_pipeline_walk_matches_the_source(source, pipeline_record, pipelines, streamed):
    """`Pretrained.load().text_to_image()` reproduces the source's own call, on par with it over drawn
    prompts and starting latents (check 1), its float64 run the truth.

    Nothing is passed in: the directory's declared pipeline carries the step count, the guidance the
    transformer embeds and the sigma seed its call lays out, and its own mu for this latent's packed token
    count. Loaded whole or streamed onto a mesh (tests/test_pipeline_streaming.py).
    """
    import torch

    from dew.interop.pretrained import Pretrained
    from dew.training import MeshSpec

    loaded = Pretrained.load(str(source / "pipeline"), dtype="float32", attention_impl="xla",
                             **({"mesh": MeshSpec()} if streamed else {}))
    task = loaded.text_to_image()
    assert task.steps == pipeline_record["default_steps"]
    assert task.guidance is None and pipeline_record["true_cfg"] == 1.0
    size = pipeline_record["size"]
    rows = size // 4

    def draw(seed):
        latents = np.random.default_rng([223, seed]).standard_normal((2, rows * rows, 64))
        return (*prompt_draw(seed), latents.astype(np.float32))

    @as_given
    def walked(indices, latents):
        prepared = task.prepare(prompts(indices), initial=unpacked(latents, rows, rows), key=0)
        walk = task(prepared, key=jax.random.PRNGKey(0)).host()
        return {"latents": packed(np.asarray(walk.latents)),
                "images": np.clip(np.asarray(walk.images) / 2 + 0.5, 0.0, 1.0)}

    def called(pipe, indices, latents):
        rows_given = prompts(indices)
        options = {"prompt": [row["text"] for row in rows_given],
                   "prompt_2": [row["second"] for row in rows_given], "height": size, "width": size,
                   "latents": torch.from_numpy(latents).to(pipe.dtype)}
        return {"latents": pipe(**options, output_type="latent").images,
                "images": pipe(**options, output_type="np").images}

    narrow, wide = pipelines
    assert_as_exact(walked, source_arms(narrow, called, wide=False), draw, "Flux pipeline walk",
                    seeds=PIPELINE_SEEDS, truth_fn=source_arms(wide, called, wide=True))


def test_a_trained_flux_step_exports_and_reloads(source, pipeline_record, tmp_path):
    """A real coupled-loss step over the published Flux source, then a resume
    and an export.

    The objective runs the whole source: the VAE encodes the pixels, the CLIP
    tower pools and the T5 tower encodes the prompt, and the transformer takes
    the gradient with the distilled guidance its checkpoint embeds. The
    released towers and the autoencoder are state, so every one of their
    leaves survives the step, and the export reloads to the same forward.
    """
    from diffusion_support import image_batch, source_objective, trained_step

    from dew.checkpoints import Checkpoints
    from dew.interop.pretrained import Pretrained

    loaded, objective = source_objective(source)
    batch = image_batch(loaded, pipeline_record["prompts"])
    _, state = trained_step(objective, batch, Checkpoints(str(tmp_path / "run")))

    export = tmp_path / "export"
    loaded.save(export, variables=state.variables)
    again = Pretrained.load(str(export), dtype="float32", attention_impl="xla")
    with np.load(source / "flux_transformer.npz") as arrays:
        grid = pipeline_record["size"] // 4
        latent = jnp.asarray(unpacked(np.asarray(arrays["pipeline.x_T"]), grid, grid))
        condition = DenoisingCondition(
            jnp.asarray(arrays["pipeline.context"]), jnp.asarray(arrays["pipeline.pooled"]),
            guidance=jnp.full((2,), pipeline_record["default_guidance"], jnp.float32))
    times = jnp.asarray([500.0, 100.0])
    trained = loaded.model.apply({"params": state.variables["params"]}, latent, times, condition)
    reloaded = again.model.apply({"params": again.variables["params"]}, latent, times, condition)
    np.testing.assert_array_equal(reloaded, trained)

    # Diffusers' own classes read the export: every component cleanly, the
    # pipeline whole, and its transformer recomputes the trained forward on
    # the packed latent with FluxPipeline's own position ids.
    from diffusion_support import assert_export_recomputes

    from tools import diffusers_consumer as consumer

    consumer.assert_components_load(export)
    forward = jax.jit(lambda drawn: loaded.model.apply({"params": state.variables["params"]}, drawn, times,
                                                       condition))

    def draw(seed):
        """A latent of the pipeline's shape, its values drawn."""
        return (np.random.default_rng([4441, seed]).standard_normal(latent.shape).astype(np.float32),)

    def call(model, tensor, drawn):
        # Imported after the consumer, which restores the transformers names
        # Diffusers' pipeline modules read.
        from diffusers import FluxPipeline

        positions = FluxPipeline._prepare_latent_image_ids(1, grid, grid, "cpu", model.dtype)
        output = model(hidden_states=tensor(packed(np.asarray(drawn))),
                       encoder_hidden_states=tensor(condition.context),
                       pooled_projections=tensor(condition.pooled), timestep=tensor(times) / 1000,
                       guidance=tensor(condition.guidance), img_ids=positions,
                       txt_ids=tensor(np.zeros((condition.context.shape[1], 3)))).sample
        return unpacked(output.numpy(), grid, grid)

    assert_export_recomputes(export, forward, call, draw, "trained Flux")


def test_each_records_guidance_reaches_the_model_and_survives_the_shared_seams(
        source, pipeline_record):
    """A row's own distilled guidance, through the ordinary seams.

    The value belongs to the row rather than to its caption: it reaches the
    model per row, a row's output does not depend on another row's value, and
    both caption dropout and the unconditional branch keep it, since dropping
    a caption changes what the model reads about the text and not the scale
    the checkpoint was distilled to walk at.
    """
    from dew.interop.pretrained import Pretrained
    from dew.objectives import Step
    from dew.objectives.diffusion import DiffusionObjective

    loaded = Pretrained.load(str(source / "pipeline"), dtype="float32", attention_impl="xla")
    encoder = loaded.inputs.conditions["conditioning"].encoder
    params = loaded.variables["encoders"]["conditioning"]
    rows = [dict(pipeline_record["prompts"][0], guidance=2.0),
            dict(pipeline_record["prompts"][1], guidance=6.0)]
    same = [dict(rows[0]), dict(rows[1], guidance=2.0)]
    tokens = encoder.tokenize(rows)
    np.testing.assert_array_equal(tokens["guidance"], [2.0, 6.0])
    given = encoder.encode(params, tokens)
    np.testing.assert_array_equal(np.asarray(given.guidance), [2.0, 6.0])
    # A value that is not a finite number is refused rather than embedded.
    with pytest.raises(ValueError):
        encoder.tokenize([{"text": "a red cat", "guidance": float("inf")}])

    grid = pipeline_record["size"] // 4
    with np.load(source / "flux_transformer.npz") as arrays:
        latent = jnp.asarray(unpacked(arrays["pipeline.x_T"], grid, grid))
    times = jnp.asarray([500.0, 500.0])
    variables = {"params": loaded.variables["params"]}
    mixed = loaded.model.apply(variables, latent, times, given)
    lowered = loaded.model.apply(variables, latent, times,
                                 encoder.encode(params, encoder.tokenize(same)))
    # The first row shares its value with the second run and matches exactly;
    # the second row's own value moves only its own output.
    np.testing.assert_array_equal(mixed[0], lowered[0])
    assert not np.allclose(mixed[1], lowered[1], atol=1e-6)

    # The unconditional branch of a guided call carries each row's scalar, so
    # it is the plain call with the empty caption at that row's own guidance.
    process, _ = loaded.grid(4)
    empty = {"text": "", "negative": True}
    denoise = process.denoiser(loaded.model, variables, {"conditioning": given},
                              {"conditioning": encoder.encode(
                                  params, encoder.tokenize([empty]))})
    _, raw = denoise.raw_both(latent, times)
    branch, _ = denoise.convert(latent, times, raw)
    per_row = encoder.encode(params, encoder.tokenize(
        [dict(empty, guidance=2.0), dict(empty, guidance=6.0)]))
    direct, _ = process.denoiser(loaded.model, variables, {"conditioning": per_row})(latent, times)
    # The guided call runs one model call over the doubled batch, so its
    # reductions are not the two-row call's bit for bit; the value that
    # matters is which guidance each row was walked at, and reading the
    # default instead of the row's own moves the output by far more.
    np.testing.assert_allclose(branch, direct, rtol=2e-5, atol=2e-5)
    default = encoder.encode(params, encoder.tokenize([dict(empty), dict(empty)]))
    other, _ = process.denoiser(loaded.model, variables, {"conditioning": default})(latent, times)
    assert not np.allclose(branch, other, rtol=2e-5, atol=2e-5)

    # Dropping every caption leaves the guidance in place, so the loss still
    # depends on it.
    objective = DiffusionObjective(loaded.model, loaded.process, loaded.inputs,
                                   autoencoder=loaded.autoencoder, variables=loaded.variables,
                                   unconditional_prob=1.0, ema_decay=None, steps=2)
    height, width = loaded.inputs.sample.shape[:2]
    pixels = np.tile(np.arange(height * width * 3, dtype=np.uint8).reshape(1, height, width, 3),
                     (2, 1, 1, 1))
    step = Step(jnp.asarray(0), jax.random.PRNGKey(11), None)

    def loss(records) -> float:
        batch = {"image": pixels, **loaded.inputs.tokenize(records)}
        value, _ = objective.loss(loaded.variables, batch, step)
        return float(value.total / value.mass)

    assert not np.isclose(loss(rows), loss(same), atol=1e-6)


def test_the_tied_t5_embedding_maps_under_either_name(source):
    """A published T5 ties its token embedding and may store either name.

    Both are the one native embedding, so both map to it and both are bound
    for export; a file carrying two copies that differ is refused, and one
    carrying neither is refused rather than initialized.
    """
    from dew.interop.diffusion import component_tensors
    from dew.interop.weights import record_layouts
    from dew.nn.text_encoders import _t5_path, translate_t5_weights

    tensors = dict(component_tensors(source / "pipeline", "text_encoder_2"))
    embedding = next(tensors[name] for name in
                     ("shared.weight", "encoder.embed_tokens.weight") if name in tensors)
    for stored in (["shared.weight"], ["encoder.embed_tokens.weight"],
                   ["shared.weight", "encoder.embed_tokens.weight"]):
        variant = {name: value for name, value in tensors.items()
                   if name not in ("shared.weight", "encoder.embed_tokens.weight")}
        variant.update(dict.fromkeys(stored, embedding))
        params, layouts = record_layouts("text_encoder_2", variant, _t5_path, ("encoders",))
        np.testing.assert_array_equal(params["embed_tokens"]["embedding"], embedding)
        bound = [entry.name for entry in layouts
                 if entry.paths[0][-2:] == ("embed_tokens", "embedding")]
        assert sorted(bound) == sorted(f"text_encoder_2/{name}" for name in stored)
        np.testing.assert_array_equal(
            translate_t5_weights(variant)["embed_tokens"]["embedding"], embedding)
    disagreeing = dict(tensors)
    disagreeing["shared.weight"] = np.asarray(embedding) + 1.0
    disagreeing["encoder.embed_tokens.weight"] = np.asarray(embedding)
    with pytest.raises(ValueError):
        translate_t5_weights(disagreeing)
    with pytest.raises(ValueError):
        translate_t5_weights({name: value for name, value in tensors.items()
                              if name not in ("shared.weight", "encoder.embed_tokens.weight")})
