"""Native Z-Image against the actual Diffusers 0.40.0 objects it reconstructs.

`tools/diffusers_z_image_reference.py` builds tiny `ZImageTransformer2DModel`
instances with every parameter moved off its initialization, saves each with
its real config and safetensors, and calls each as `ZImagePipeline` calls it
- a list of latents, the time 1 - sigma, a list of prompt states of each
prompt's own length - recording the output and the gradients of the latents,
the prompt states and every parameter against a fixed cotangent, in float32
and in float64. It also saves one tiny `ZImagePipeline` over the published
configs and walks the unmodified call, guided at its default scale, the same
two ways. Each native tensor is held to the source's own float32 distance
from float64 (tests/reference_error.py).

The native transformer takes Dew's model time, sigma times a thousand, and
returns the flow, the negated source output; its prompt states arrive padded
to a budget with a mask. These tests convert with the arithmetic written out
here.
"""

import contextlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from interop_support import (
    assert_transformer_exports,
    drawn_prompts,
    extract_fixture,
    fixture_arrays,
    source_transformer,
)
from reference_error import as_given, assert_as_exact, torch_with_gradients, with_gradients

from dew.diffusion.process import DenoisingCondition
from dew.interop.diffusion import z_image_fields, z_image_path
from dew.nn.backbones.z_image import ZImageTransformer

ROOT = Path(__file__).resolve().parents[1]
CASES = ("padded", "exact", "deep")
CASE_SEEDS = {"padded": 512, "exact": 256, "deep": 512}
"""Check 1's declared seeds for each case, sized by `seeds_for` on armada."""
ROWS = 4
HEIGHT = 3
BUDGET = 64


@pytest.fixture(scope="module")
def source(tmp_path_factory):
    return extract_fixture(ROOT / "tests/fixtures/z_image_source.tar.xz",
                           tmp_path_factory.mktemp("z-image-source"))


@pytest.fixture(scope="module")
def record(source):
    return json.loads((source / "z_image.json").read_text())


@pytest.fixture(scope="module")
def arrays(source):
    return fixture_arrays(source / "z_image.npz")


def nhwc(latents):
    """`[B, C, 1, H, W]` to `[B, H, W, C]`."""
    return np.asarray(latents)[:, :, 0].transpose(0, 2, 3, 1)


def load(source, name):
    return source_transformer(source, name, ZImageTransformer, z_image_fields, z_image_path)


def z_image_walk(model, params, lengths):
    """A compiled native forward on a draw as the source's output, the negated flow, and the gradients
    of the latents, the prompts and every parameter under the draw's probe, its cotangent
    (`with_gradients`): the source's latents channels last, the model time from the source's
    1 - sigma, and each prompt padded to the budget and masked."""
    mask = jnp.asarray(np.arange(BUDGET)[None] < np.asarray(lengths)[:, None])

    @jax.jit
    def walk(latents, times, context, probe):
        time = 1000.0 * (1.0 - times)

        def forward(latents, context, params):
            return -model.apply({"params": params}, latents, time, DenoisingCondition(context, mask=mask))

        return with_gradients(forward)(latents, context, params, probe)

    return walk


@pytest.mark.parametrize("name", CASES)
def test_every_published_tensor_maps_and_exports_bit_identical(source, name):
    _, params, layouts = load(source, name)
    assert_transformer_exports(source, name, layouts, params)


@pytest.mark.parametrize("name", [pytest.param(name, marks=pytest.mark.whole_suite)
                                  if name in ("padded", "deep", "exact") else name for name in CASES])
def test_the_flow_and_its_gradients_match_the_source(source, arrays, record, name):
    """The native flow is the negated source output, its gradients the negated source gradients,
    every parameter's among them, whatever the rows' prompt lengths, each as near float64 as
    Diffusers' own float32 walk (`assert_as_exact`) over the fixture's draw and fresh ones; a
    prompt's padding takes no gradient."""
    import torch
    from diffusers import ZImageTransformer2DModel

    from tools.torch_precision import float64

    model, params, layouts = load(source, name)
    # Each draw holds the case's rows ROWS times over and its images HEIGHT times as tall, so a
    # parameter's gradient sums over more of them and one draw's error varies less: the first
    # layer's norm gradients did not resolve within 512 seeds at the fixture's size. Three times
    # the height keeps each case's padding: the padded case's 72 patches pad to 96, the exact
    # case's 96 fill three blocks of 32, and the deep case's 45 pad to 64.
    cased = len(record["cases"][name]["lengths"])
    lengths = tuple(record["cases"][name]["lengths"]) * ROWS
    walk = z_image_walk(model, params, lengths)
    tensors = {layout.name.removeprefix("transformer/"): layout for layout in layouts}
    assert {key.removeprefix(f"{name}.grad_param.") for key in arrays
            if key.startswith(f"{name}.grad_param.")} == set(tensors)
    captions = [f"caption.{row}" for row in range(len(lengths))]
    original = {key: np.concatenate([arrays[f"{name}.{key}"]] * ROWS)
                for key in ("latents", "times", "probe")}
    original |= {key: np.tile(original[key], (1, 1, 1, HEIGHT, 1)) for key in ("latents", "probe")}
    original |= {caption: np.array(arrays[f"{name}.caption.{row % cased}"])
                 for row, caption in enumerate(captions)}

    def draw(seed):
        """The fixture's inputs, its rows ROWS times over and its images tiled HEIGHT times down, at
        seed 0, else fresh normal draws of those shapes."""
        if seed == 0:
            return ({key: np.asarray(value, np.float32) for key, value in original.items()},)
        generator = torch.Generator().manual_seed(31 + seed)
        return ({**{key: torch.randn(original[key].shape, generator=generator).numpy()
                    for key in ("latents", "probe", *captions)}, "times": original["times"]},)

    checkpoint = source / name / "transformer"
    narrow_model = ZImageTransformer2DModel.from_pretrained(
        checkpoint, torch_dtype=torch.float32, local_files_only=True).eval()
    with float64():
        wide_model = ZImageTransformer2DModel.from_pretrained(
            checkpoint, torch_dtype=torch.float64, local_files_only=True).eval()

    weights = {model: {key: value.detach().numpy() for key, value in model.named_parameters()}
               for model in (narrow_model, wide_model)}

    def reference_walk(model, drawn):
        """The source's call in the draw's dtype, and the same gradients under the same cotangent as
        `z_image_walk`'s (`torch_with_gradients`), channels last."""
        times = torch.from_numpy(np.asarray(drawn["times"]))

        def forward(latents, prompts, params):
            return torch.stack(torch.func.functional_call(
                model, params, (list(latents.unbind(0)), times.to(latents.dtype), list(prompts)),
                {"return_dict": False})[0])

        output, grad_latents, grad_prompts, grad_params = torch_with_gradients(forward)(
            drawn["latents"], tuple(drawn[caption] for caption in captions), weights[model], drawn["probe"])
        return {"output": nhwc(output), "grad_latents": nhwc(grad_latents),
                **{f"grad_{caption}": gradient
                   for caption, gradient in zip(captions, grad_prompts, strict=True)},
                "grad_params": grad_params}

    def truth(inputs):
        with float64():
            return reference_walk(wide_model, inputs)

    @as_given
    def native(inputs):
        width = inputs[captions[0]].shape[-1]
        context = np.zeros((len(lengths), BUDGET, width), np.float32)
        for row, length in enumerate(lengths):
            context[row, :length] = inputs[f"caption.{row}"]
        output, grad_latents, grad_context, grad_params = walk(
            jnp.asarray(nhwc(inputs["latents"])), jnp.asarray(inputs["times"], jnp.float32),
            jnp.asarray(context), jnp.asarray(nhwc(inputs["probe"])))
        grad_context = np.asarray(grad_context)
        assert not any(grad_context[row, length:].any() for row, length in enumerate(lengths))
        return {"output": np.asarray(output), "grad_latents": np.asarray(grad_latents),
                **{f"grad_caption.{row}": grad_context[row, :length] for row, length in enumerate(lengths)},
                "grad_params": {tensor: np.asarray(layout.export({"params": grad_params}))
                                for tensor, layout in tensors.items()}}

    assert_as_exact(native, as_given(lambda inputs: reference_walk(narrow_model, inputs)), draw,
                    f"Z-Image {name}", seeds=CASE_SEEDS[name], truth_fn=truth)


@pytest.fixture(scope="module")
def pipeline(source):
    from dew.interop.pretrained import Pretrained

    return Pretrained.load(str(source / "pipeline"), dtype="float32", attention_impl="xla")


@pytest.fixture(scope="module")
def streamed(source):
    """The pipeline loaded onto a mesh, its weights streamed as recipes and
    placed a leaf at a time (tests/test_pipeline_streaming.py)."""
    from dew.interop.pretrained import Pretrained
    from dew.training import MeshSpec

    return Pretrained.load(str(source / "pipeline"), dtype="float32", attention_impl="xla", mesh=MeshSpec())


PIPELINE_SEEDS = {"prompts": 8, "walk": 8}
"""Check 1's declared seeds for the pipeline's prompt encoding and its walk, sized by `seeds_for` on
armada."""
def source_pipelines(source):
    """Diffusers' own `ZImagePipeline` the fixture saved, loaded in float32 and widened to float64,
    the scheduler's tables and float32 pins widened with it (`float64_scheduler`)."""
    import torch
    from diffusers import ZImagePipeline

    from tools.torch_precision import float64

    def loaded(dtype):
        return ZImagePipeline.from_pretrained(source / "pipeline", torch_dtype=dtype, local_files_only=True)

    narrow = loaded(torch.float32)
    with float64(), widened_schedulers():
        wide = loaded(torch.float64)
    for pipe in (narrow, wide):
        pipe.set_progress_bar_config(disable=True)
    return narrow, wide


@contextlib.contextmanager
def widened_schedulers():
    """The pipeline's and its FlowMatch scheduler's float32 pins widened (`float64_scheduler`): the
    scheduler forms its shifted sigmas, timesteps and steps in float32 otherwise."""
    from diffusers.pipelines.z_image import pipeline_z_image
    from diffusers.schedulers import scheduling_flow_match_euler_discrete

    from tools.diffusers_wan_reference import float64_scheduler

    with float64_scheduler(pipeline_z_image), float64_scheduler(scheduling_flow_match_euler_discrete):
        yield


def widened_call(pipe, run):
    """`run(pipe)` with the source's float32 pins widened, as its float64 truth runs."""
    from tools.torch_precision import float64

    with float64(), widened_schedulers():
        return run(pipe)


def test_prompt_encoding_matches_the_source_pipeline(pipeline, source, record):
    """The conditioner reads what `_encode_prompt` reads: the chat template with thinking on, the
    output of the encoder's second-to-last layer, and the real tokens, which the mask marks; on par
    with the source's own float32 encoding over drawn prompts and the empty one (check 1), the states
    past each prompt's tokens held to zero on both sides."""
    import torch

    encoder = pipeline.inputs.conditions["conditioning"].encoder
    params = pipeline.variables["encoders"]["conditioning"]
    narrow, wide = source_pipelines(source)

    def prompts_of(seed) -> tuple[str, ...]:
        return (*drawn_prompts(7, seed), "")

    def draw(seed):
        # The arms draw their prompts from the seed, a leaf the gate widens as it is.
        return (np.int32(seed),)

    def encoded(prompts):
        condition = encoder.encode(params, encoder.tokenize(list(prompts)))
        mask = np.asarray(condition.mask)
        return np.asarray(condition.context) * mask[..., None], mask

    def source_encoded(pipe, prompts):
        rows = np.zeros((len(prompts), encoder.tokens, record["pipeline"]["config"]["cap_feat_dim"]))
        with torch.no_grad():
            for row, prompt in enumerate(prompts):
                (states,) = pipe._encode_prompt(prompt=prompt, device=torch.device("cpu"))
                rows[row, :states.shape[0]] = states.numpy()
        return rows

    prompts = prompts_of(0)
    _, mask = encoded(prompts)
    lengths = [np.count_nonzero(np.abs(row).sum(-1)) for row in source_encoded(narrow, prompts)]
    np.testing.assert_array_equal(mask, np.arange(encoder.tokens)[None] < np.asarray(lengths)[:, None])
    assert_as_exact(as_given(lambda seed: encoded(prompts_of(seed))[0]),
                    as_given(lambda seed: source_encoded(narrow, prompts_of(seed))), draw, "Z-Image prompts",
                    seeds=PIPELINE_SEEDS["prompts"],
                    truth_fn=lambda seed: widened_call(
                        wide, lambda pipe: source_encoded(pipe, prompts_of(seed))))


@pytest.mark.parametrize("load", ["pipeline", "streamed"])
def test_pipeline_walk_matches_the_source(load, source, record, request):
    """`Pretrained.load().text_to_image()` reproduces the source's call: its 50 default steps guided
    at 5.0 its way, which is Dew's 6.0, and at the recorded step count the latent it ends on and the
    image it decodes, loaded whole or streamed onto a mesh, on par with the source's own float32 call
    over drawn starting latents (check 1)."""
    import torch

    pipeline = request.getfixturevalue(load)
    recorded = record["pipeline"]
    task = pipeline.text_to_image()
    assert task.steps == 50 and task.guidance is not None and task.guidance.scale == recorded["guidance"] + 1
    narrow, wide = source_pipelines(source)
    height, width, steps = recorded["height"], recorded["width"], recorded["steps"]
    shape = (len(recorded["prompts"]), recorded["config"]["in_channels"], height // 2, width // 2)

    def draw(seed):
        return (torch.randn(shape, generator=torch.Generator().manual_seed(53 + seed)).numpy(),)

    @as_given
    def native(x_T):
        prepared = task.prepare(recorded["prompts"], initial=x_T.transpose(0, 2, 3, 1), key=0, steps=steps)
        walked = task(prepared, key=jax.random.PRNGKey(0)).host()
        return {"latents": np.asarray(walked.latents),
                "images": np.clip(np.asarray(walked.images) / 2 + 0.5, 0.0, 1.0)}

    def source_walk(pipe, x_T):
        latents, images = [], []
        with torch.no_grad():
            for row, prompt in enumerate(recorded["prompts"]):
                call = {"prompt": prompt, "height": height, "width": width, "num_inference_steps": steps,
                        "guidance_scale": recorded["guidance"]}
                start = torch.from_numpy(np.asarray(x_T[row:row + 1]))
                latents.append(pipe(**call, latents=start.clone(), output_type="latent").images.numpy()[0])
                images.append(pipe(**call, latents=start.clone(), output_type="np").images[0])
        return {"latents": np.stack(latents).transpose(0, 2, 3, 1), "images": np.stack(images)}

    assert_as_exact(native, as_given(lambda x_T: source_walk(narrow, x_T)), draw, f"Z-Image {load} walk",
                    seeds=PIPELINE_SEEDS["walk"],
                    truth_fn=lambda x_T: widened_call(wide, lambda pipe: source_walk(pipe, x_T)))


@pytest.mark.parametrize(("latent", "real", "lengths"), [((6, 2), 8, (256, 2, 2)), ((4, 4), 32, (33, 8, 8))],
                         ids=["grid", "prompt"])
def test_positions_past_the_published_tables_rotate_at_their_own_places(latent, real, lengths):
    """Each axis's rotary table reaches every position a call takes: a latent grid of three rows past a
    two-row table, or a full 32-token prompt whose image sits at position 33 past a 33-row table, rotates
    at its own positions, so the model computes, bit for bit, what the same weights compute with longer
    tables, where a gather clamped to the table rotates the last rows as the row before them (audit
    NN-A4)."""
    height, width = latent
    short = ZImageTransformer(in_channels=4, dim=32, n_layers=1, n_refiner_layers=1, n_heads=2,
                              cap_feat_dim=16, axes_dims=(4, 6, 6), axes_lens=lengths)
    long = short.clone(axes_lens=(256, 64, 64))
    rng = np.random.default_rng(0)
    latents = jnp.asarray(rng.standard_normal((1, height, width, 4)), jnp.float32)
    condition = DenoisingCondition(jnp.asarray(rng.standard_normal((1, 32, 16)), jnp.float32),
                                   mask=jnp.asarray(np.arange(32)[None] < real))
    time = jnp.full((1,), 500.0)
    params = short.init(jax.random.key(0), latents, time, condition)
    params = jax.tree.map(lambda leaf: leaf + jnp.asarray(0.05 * rng.standard_normal(leaf.shape), leaf.dtype),
                          params)
    np.testing.assert_array_equal(short.apply(params, latents, time, condition),
                                  long.apply(params, latents, time, condition))
