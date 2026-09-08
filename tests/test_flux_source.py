"""Native Flux against the actual Diffusers 0.34.0 objects it reconstructs.

`tools/diffusers_flux_reference.py` builds tiny `FluxTransformer2DModel`
instances, saves each with its real config and safetensors, and runs each the
way its pipeline runs it - packed latents, the ids the pipeline lays out, the
timestep it has divided by the training count and the distilled guidance a
guidance-embedded checkpoint takes - recording the forward, the
vector-Jacobian products of the latent, the text tokens and the pooled
vector, and the gradient of every parameter.

The native model owns its pipeline's 2x2 packing, so it works in latents.
These tests convert between the two layouts with the source's own permutation
written out here, rather than with the functions under test.
"""

import json
import tarfile
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dew.interop.diffusion import component_tensors, flux_fields, translate_flux_weights
from dew.nn.backbones.flux import FluxTransformer
from dew.nn.backbones.unet_condition import DenoisingCondition

ROOT = Path(__file__).resolve().parents[1]
CASES = ("schnell", "dev", "rect", "deep", "mixed")


@pytest.fixture(scope="module")
def source(tmp_path_factory):
    directory = tmp_path_factory.mktemp("flux-source")
    with tarfile.open(ROOT / "tests/fixtures/flux_source.tar.xz") as archive:
        archive.extractall(directory, filter="data")
    return directory


def relative_gap(actual, expected) -> float:
    actual, expected = np.asarray(actual, np.float32), np.asarray(expected, np.float32)
    return float(np.abs(actual - expected).max() / max(1.0, float(np.abs(expected).max())))


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


def transformer_case(directory: Path, arrays, name: str, grid: tuple[int, int]):
    """The native model, its loaded parameters and the case's own inputs."""
    config = json.loads(str(arrays[f"{name}.config"]))
    model = FluxTransformer(**flux_fields(config, attention_impl="xla"))
    params, layouts = translate_flux_weights(component_tensors(directory / name, "transformer"))
    latent = jnp.asarray(unpacked(arrays[f"{name}.packed"], *grid))
    context = jnp.asarray(arrays[f"{name}.context"])
    pooled = jnp.asarray(arrays[f"{name}.pooled"])
    times = jnp.asarray(arrays[f"{name}.times"])
    recorded = arrays[f"{name}.guidance"]
    guidance = None if recorded.size == 0 else jnp.asarray(recorded)

    def forward(params, latent, context, pooled):
        return model.apply({"params": params}, latent, times,
                           DenoisingCondition(context, pooled, guidance=guidance))

    return forward, params, layouts, latent, context, pooled


@pytest.mark.parametrize("name", CASES)
def test_native_flux_matches_the_source_forward_and_every_gradient(name, source):
    """One published transformer's own tensors, read through the native model.

    The variants are the ones whose wiring differs: the schnell-style model
    with no guidance embedder, the distilled one with it, a rectangular packed
    grid whose row and column rotations differ, a stack with a different split
    between double and single blocks, and one walked at a different distilled
    guidance per row.

    Two records are compared. The unmodified source gives the forward and the
    latent, token and pooled gradients within 1e-5. The controlled record is
    the same source walk with Dew's own frequency table handed to its timestep
    embedding - a control, not unmodified-source parity - and every parameter
    gradient is held to 1e-4 there, because torch's CPU `exp` is not correctly
    rounded at six of those 128 frequencies and an ulp of a frequency is 1e-4
    of a 3500-radian angle. `test_the_frequency_table_is_the_only_component_
    that_differs` measures that difference and what it costs unmodified.
    """
    record = json.loads((source / "flux_transformer.json").read_text())
    grid = tuple(record["cases"][name]["grid"])
    with np.load(source / "flux_transformer.npz") as arrays:
        forward, params, layouts, latent, context, pooled = transformer_case(
            source, arrays, name, grid)
        probe = jnp.asarray(unpacked(arrays[f"{name}.probe"], *grid))
        output = jax.jit(forward)(params, latent, context, pooled)
        assert relative_gap(packed(np.asarray(output)), arrays[f"{name}.output"]) < 1e-5
        gradients = jax.jit(jax.grad(
            lambda p, l, c, q: jnp.sum(forward(p, l, c, q) * probe), argnums=(0, 1, 2, 3)))(
                params, latent, context, pooled)
        assert relative_gap(packed(np.asarray(gradients[1])), arrays[f"{name}.grad_packed"]) < 1e-5
        assert relative_gap(gradients[2], arrays[f"{name}.grad_context"]) < 1e-5
        assert relative_gap(gradients[3], arrays[f"{name}.grad_pooled"]) < 1e-5
        # The controlled record, where both sides read the same frequencies.
        assert relative_gap(packed(np.asarray(output)),
                            arrays[f"{name}.control_output"]) < 1e-5
        layout = {entry.name: entry for entry in layouts}
        prefix = f"{name}.control_grad_param."
        names = [key for key in arrays.files if key.startswith(prefix)]
        assert names, name
        for key in names:
            entry = layout["transformer/" + key[len(prefix):]]
            node = gradients[0]
            for step in entry.paths[0][1:]:
                node = node[step]
            value = np.asarray(node)
            if entry.transpose is not None:
                value = value.transpose(entry.transpose)
            assert value.shape == tuple(entry.shape), key
            assert relative_gap(value, arrays[key]) < 1e-4, key


def test_the_frequency_table_is_the_only_component_that_differs(source):
    """What the unmodified source's own `exp` costs, measured.

    Dew's frequency table is the correctly rounded float32 exponential of the
    same bit-identical exponent; torch's CPU `exp` differs from it at a few
    entries by one unit in the last place. Nothing else in the embedding
    differs, so with the source's own table handed back to it the whole model
    agrees, and the two weights whose gradients ARE those features are the
    only place the unmodified comparison exceeds 1e-4. That excess is reported
    here rather than accommodated: the bound above stays 1e-4 over every
    tensor of the controlled record.
    """
    features = ("time_text_embed.timestep_embedder.linear_1.weight",
                "time_text_embed.guidance_embedder.linear_1.weight")
    with np.load(source / "flux_transformer.npz") as arrays:
        native = arrays["dev.frequencies_native"]
        theirs = arrays["dev.frequencies_source"]
        differing = np.nonzero(native != theirs)[0]
        # The tables agree to one unit in the last place everywhere, and they
        # do differ, so the control is not comparing a table with itself.
        assert differing.size
        ulps = np.abs(native[differing] - theirs[differing]) / np.spacing(theirs[differing])
        np.testing.assert_array_equal(ulps, np.ones_like(ulps))
        # An ulp of a frequency times a guidance of 3500 is an ulp of the
        # angle, and that is what the two feature gradients carry.
        worst = {}
        for name in ("dev", "rect"):
            for key in features:
                unmodified = arrays[f"{name}.grad_param.{key}"]
                controlled = arrays[f"{name}.control_grad_param.{key}"]
                worst[f"{name}.{key}"] = float(
                    np.abs(unmodified - controlled).max()
                    / max(1.0, float(np.abs(unmodified).max())))
        assert max(worst.values()) < 3e-4, worst
        assert min(worst.values()) > 1e-5, worst


def test_every_declared_flux_tensor_is_mapped(source):
    """Every tensor the source stores lands in the native tree, and a name
    this translation does not know raises with that name rather than loading a
    checkpoint that means something else."""
    from dew.interop.diffusion import _flux_path

    tensors = component_tensors(source / "dev", "transformer")
    params, layouts = translate_flux_weights(tensors)
    assert len(layouts) == len(tensors)
    leaves = {"/".join(entry.paths[0]) for entry in layouts}
    assert len(leaves) == len(tensors)
    with pytest.raises(ValueError, match="unknown tensor name"):
        _flux_path("transformer_blocks.0.attn.to_out.1.weight")
    with pytest.raises(ValueError, match="unknown tensor name"):
        _flux_path("single_transformer_blocks.0.attn.to_add_out.weight")


def test_the_guidance_input_follows_the_checkpoints_own_embedder(source):
    """A distilled checkpoint reads a guidance value and a schnell-style one
    holds no embedder for it, so each refuses the other's call rather than
    ignoring an input the source would have used."""
    record = json.loads((source / "flux_transformer.json").read_text())
    with np.load(source / "flux_transformer.npz") as arrays:
        for name, wanted in (("dev", True), ("schnell", False)):
            grid = tuple(record["cases"][name]["grid"])
            config = json.loads(str(arrays[f"{name}.config"]))
            assert flux_fields(config)["guidance_embeds"] is wanted
            model = FluxTransformer(**flux_fields(config, attention_impl="xla"))
            params, _ = translate_flux_weights(
                component_tensors(source / name, "transformer"))
            latent = jnp.asarray(unpacked(arrays[f"{name}.packed"], *grid))
            condition = DenoisingCondition(
                jnp.asarray(arrays[f"{name}.context"]), jnp.asarray(arrays[f"{name}.pooled"]),
                guidance=None if wanted else jnp.full((latent.shape[0],), 3.5, jnp.float32))
            match = "needs it" if wanted else "no guidance embedder"
            with pytest.raises(ValueError, match=match):
                model.apply({"params": params}, latent, jnp.asarray(arrays[f"{name}.times"]),
                            condition)


def test_the_rotary_table_is_the_sources_own_interleaved_pairs(source):
    """The table Flux rotates with: one angle per adjacent channel pair, laid
    out per axis, over the ids its pipeline writes."""
    from dew.nn.backbones.flux import apply_rotary, flux_positions, rotary_table

    positions = flux_positions(3, 2, 4)
    # The text sits at the origin and each patch carries its row and column.
    assert positions.shape == (10, 3)
    np.testing.assert_array_equal(positions[:4], np.zeros((4, 3), np.float32))
    np.testing.assert_array_equal(positions[4:, 1], np.repeat(np.arange(3), 2))
    np.testing.assert_array_equal(positions[4:, 2], np.tile(np.arange(2), 3))
    cos, sin = rotary_table(positions, (4, 4, 4))
    assert cos.shape == sin.shape == (10, 12)
    # Adjacent channels share an angle, and the text row rotates by nothing.
    np.testing.assert_allclose(cos[:, 0::2], cos[:, 1::2], atol=0)
    np.testing.assert_array_equal(cos[0], np.ones(12, np.float32))
    np.testing.assert_array_equal(sin[0], np.zeros(12, np.float32))
    # A quarter turn on one pair takes (x0, x1) to (-x1, x0).
    quarter = np.zeros((1, 1, 1, 2), np.float32)
    rotated = apply_rotary(jnp.asarray([[[[1.0, 2.0]]]]), jnp.zeros((1, 1, 1, 2)) + 0.0,
                           jnp.ones((1, 1, 1, 2)) + quarter)
    np.testing.assert_allclose(np.asarray(rotated).ravel(), [-2.0, 1.0], atol=0)


@pytest.fixture(scope="module")
def pipeline_record(source):
    return json.loads((source / "flux_transformer.json").read_text())["pipeline"]


def test_published_flux_prompt_encoding_matches_the_source_pipeline(source, pipeline_record):
    """The conditioner composes what `encode_prompt` composes.

    Flux reads its T5 tower's states as the sequence and its CLIP tower's
    pooled row unprojected, and its pipeline routes `prompt` to CLIP and
    `prompt_2` to T5. The two slots carry different words, so a native
    encoder that crossed them or projected the pooled row would not land here.
    """
    from dew.interop.pretrained import load_pretrained

    with np.load(source / "flux_transformer.npz") as arrays:
        loaded = load_pretrained(str(source / "pipeline"), dtype="float32",
                                 attention_impl="xla")
        encoder = loaded.inputs.conditions["conditioning"].encoder
        params = loaded.variables["encoders"]["conditioning"]
        condition = encoder.encode(params, encoder.tokenize(pipeline_record["prompts"]))
        assert condition.context.shape == arrays["pipeline.context"].shape
        assert relative_gap(condition.context, arrays["pipeline.context"]) < 1e-5
        assert relative_gap(condition.pooled, arrays["pipeline.pooled"]) < 1e-6
        # The distilled guidance rides in as a model input, at the value the
        # pipeline defaults to, and guidance is not two branches here.
        assert condition.guidance is not None
        np.testing.assert_allclose(np.asarray(condition.guidance),
                                   np.full(2, pipeline_record["default_guidance"]), atol=0)
        crossed = encoder.encode(params, encoder.tokenize(
            [{"text": row["second"], "second": row["text"]} for row in pipeline_record["prompts"]]))
        assert relative_gap(crossed.context, arrays["pipeline.context"]) > 1e-3


def test_published_flux_pipeline_walk_matches_the_source(source, pipeline_record):
    """`load_pretrained().text_to_image()` reproduces the source's own call.

    Nothing is passed in: the directory's declared pipeline carries the step
    count, the guidance the transformer embeds and the sigma seed its call
    lays out, and its own mu for this latent's packed token count.
    """
    from dew.interop.pretrained import load_pretrained

    with np.load(source / "flux_transformer.npz") as arrays:
        loaded = load_pretrained(str(source / "pipeline"), dtype="float32",
                                 attention_impl="xla")
        task = loaded.text_to_image()
        assert task.steps == pipeline_record["default_steps"]
        assert task.guidance is None and pipeline_record["true_cfg"] == 1.0
        rows = pipeline_record["size"] // 4
        initial = unpacked(arrays["pipeline.x_T"], rows, rows)
        prepared = task.prepare(pipeline_record["prompts"], initial=initial, seed=0)
        walked = task(prepared, key=jax.random.PRNGKey(0)).host()
        assert relative_gap(packed(np.asarray(walked.latents)), arrays["pipeline.latents"]) < 2e-5
        images = np.clip(np.asarray(walked.images) / 2 + 0.5, 0.0, 1.0)
        assert relative_gap(images, arrays["pipeline.images"]) < 2e-5


def test_a_trained_flux_step_exports_and_reloads(source, pipeline_record, tmp_path):
    """A real coupled-loss step over the published Flux source, then a resume
    and an export.

    The objective runs the whole source: the VAE encodes the pixels, the CLIP
    tower pools and the T5 tower encodes the prompt, and the transformer takes
    the gradient with the distilled guidance its checkpoint embeds. The
    released towers and the autoencoder are state, so every one of their
    leaves survives the step, and the export reloads to the same forward.
    """
    import optax
    from dew.checkpoints import Checkpoints
    from dew.interop.pretrained import load_pretrained
    from dew.objectives import Step
    from dew.objectives.diffusion import DiffusionObjective
    from dew.training import Trainer

    loaded = load_pretrained(str(source / "pipeline"), dtype="float32", attention_impl="xla")
    height, width = loaded.inputs.sample.shape[:2]
    objective = DiffusionObjective(loaded.model, loaded.process, loaded.inputs,
                                   autoencoder=loaded.autoencoder, pretrained=loaded.variables,
                                   unconditional_prob=0.0, ema_decay=None, steps=2)
    rows = jax.device_count()
    pixels = np.tile(np.arange(height * width * 3, dtype=np.uint8).reshape(1, height, width, 3),
                     (rows, 1, 1, 1))
    batch = {"image": pixels, **loaded.inputs.tokenize(pipeline_record["prompts"] * (rows // 2))}
    checkpoints = Checkpoints(str(tmp_path / "run"))
    trainer = Trainer(objective, optax.sgd(1e-2), key=jax.random.PRNGKey(3),
                      checkpoints=checkpoints)
    initial = trainer.initial_state()
    fixed = Step(jnp.asarray(0), jax.random.PRNGKey(5), None)

    def value(params) -> float:
        loss, _ = objective.loss(params, batch, fixed)
        return float(loss.total / loss.mass)

    before = value(initial.params)
    state, _, _, _, accepted = trainer.compile(initial, batch)(initial, batch)
    assert bool(accepted)
    assert value(state.params) < before
    assert not np.allclose(state.params["params"]["proj_out"]["kernel"],
                           initial.params["params"]["proj_out"]["kernel"])
    for held in ("encoders", "autoencoder"):
        for got, want in zip(jax.tree.leaves(state.params[held]),
                             jax.tree.leaves(initial.params[held]), strict=True):
            np.testing.assert_array_equal(got, want)

    checkpoints.save(1, state, None, {})
    checkpoints.wait()
    restored, _, _ = trainer.place()
    for got, want in zip(jax.tree.leaves(restored), jax.tree.leaves(state), strict=True):
        np.testing.assert_array_equal(got, want)

    export = tmp_path / "export"
    loaded.save(export, variables=state.params)
    again = load_pretrained(str(export), dtype="float32", attention_impl="xla")
    with np.load(source / "flux_transformer.npz") as arrays:
        grid = pipeline_record["size"] // 4
        latent = jnp.asarray(unpacked(arrays["pipeline.x_T"], grid, grid))
        condition = DenoisingCondition(
            jnp.asarray(arrays["pipeline.context"]), jnp.asarray(arrays["pipeline.pooled"]),
            guidance=jnp.full((2,), pipeline_record["default_guidance"], jnp.float32))
    times = jnp.asarray([500.0, 100.0])
    trained = loaded.model.apply({"params": state.params["params"]}, latent, times, condition)
    reloaded = again.model.apply({"params": again.variables["params"]}, latent, times, condition)
    np.testing.assert_array_equal(reloaded, trained)


def test_each_records_guidance_reaches_the_model_and_survives_the_shared_seams(
        source, pipeline_record):
    """A row's own distilled guidance, through the ordinary seams.

    The value belongs to the row rather than to its caption: it reaches the
    model per row, a row's output does not depend on another row's value, and
    both caption dropout and the unconditional branch keep it, since dropping
    a caption changes what the model reads about the text and not the scale
    the checkpoint was distilled to walk at.
    """
    import optax
    from dew.interop.pretrained import load_pretrained
    from dew.objectives import Step
    from dew.objectives.diffusion import DiffusionObjective

    loaded = load_pretrained(str(source / "pipeline"), dtype="float32", attention_impl="xla")
    encoder = loaded.inputs.conditions["conditioning"].encoder
    params = loaded.variables["encoders"]["conditioning"]
    rows = [dict(pipeline_record["prompts"][0], guidance=2.0),
            dict(pipeline_record["prompts"][1], guidance=6.0)]
    same = [dict(rows[0]), dict(rows[1], guidance=2.0)]
    tokens = encoder.tokenize(rows)
    np.testing.assert_allclose(tokens["guidance"], [2.0, 6.0], atol=0)
    given = encoder.encode(params, tokens)
    np.testing.assert_allclose(np.asarray(given.guidance), [2.0, 6.0], atol=0)
    # A value that is not a finite number is refused rather than embedded.
    with pytest.raises(ValueError, match="finite number"):
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
    process, _ = loaded.task.grid(4)
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
                                   autoencoder=loaded.autoencoder, pretrained=loaded.variables,
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
