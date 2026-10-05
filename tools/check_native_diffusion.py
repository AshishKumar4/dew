"""Native diffusion checkpoint, trajectory and update checks against saved oracles.

The runtime under test uses only Dew modules, Process and native solvers.
Diffusers/Transformers model implementations run in the reference tools,
and here on the trained export alone: Diffusers' own classes load every
component Dew wrote and its UNet recomputes the trained prediction
(`diffusers_reads`).
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import tempfile

import jax
import jax.numpy as jnp
import numpy as np

from dew.diffusion.schedules.source import SourceSchedule
from dew.inference import DenoisingInputs
from dew.inputs import Condition, Field, unit_range
from dew.inputs.diffusion import latent_image_conditions
from dew.interop import Pretrained
from dew.objectives.base import Step
from dew.objectives.diffusion import DiffusionObjective
from dew.sampling.guidance import CFG
from dew.sampling.sample import sample


def bundle(directory, streamed: bool = False):
    """The pipeline, loaded whole, or streamed onto a mesh a leaf at a time."""
    from dew.training import MeshSpec

    return Pretrained.load(str(directory), dtype="float32", attention_impl="xla",
                           **({"mesh": MeshSpec()} if streamed else {}))


def compare(errors, name, actual, expected, tolerance=5e-5):
    actual, expected = np.asarray(actual, np.float32), np.asarray(expected, np.float32)
    errors[name] = float(np.max(np.abs(actual - expected)))
    np.testing.assert_allclose(actual, expected, atol=tolerance, rtol=tolerance)


def conditions(source, extended=False):
    encoder = source.inputs.conditions["conditioning"].encoder
    params = source.variables["encoders"]["conditioning"]
    positive = {"text": "cat", **({"second": "fox"} if extended else {})}
    negative = {"text": "dog", "negative": True, **({"second": "owl"} if extended else {})}
    return ({"conditioning": encoder.encode(params, encoder.tokenize([positive]))},
            {"conditioning": encoder.encode(params, encoder.tokenize([negative]))})


def task_kind(source, meta):
    if "task" in meta:
        return meta["task"]
    name = source.config["model_index"]["_class_name"]
    return "inpaint" if "Inpaint" in name else "img2img" if "Img2Img" in name else "text"


def trajectory(source, reference, meta, key):
    """Compose the native image initialization, conditions and single solver walk."""
    kind = task_kind(source, meta)
    given, null = conditions(source, kind.startswith("xl-") or kind == "refiner")
    steps = meta.get("steps", 2)
    process, times = source.schedule.sampling(steps)
    noise = jnp.asarray(reference["noise"])
    initial = noise * process.sampler_schedule.prior_scale()
    pixels = unit_range(jnp.asarray(reference["pixels"][None])) if "pixels" in reference else None
    if kind in ("img2img", "xl-img2img", "refiner"):
        clean = source.autoencoder.encode(source.variables["autoencoder"], pixels,
                                           key if kind == "img2img" else None)
        if kind == "refiner":
            model_times = np.asarray(process.sampler_schedule.model_time(times[:-1]))
            cutoff = round(len(source.schedule.betas) * (1 - meta["denoising_start"]))
            times = times[int(np.flatnonzero(model_times < cutoff)[0]):]
            initial = clean
        else:
            start = steps - int(steps * meta.get("strength", 1.0))
            times = times[start:]
            alpha, sigma = process.sampler_schedule.rates(jnp.full((noise.shape[0],), times[0]))
            shape = (-1,) + (1,) * (noise.ndim - 1)
            initial = alpha.reshape(shape) * clean + sigma.reshape(shape) * noise
    if kind in ("inpaint", "xl-inpaint"):
        mask = (jnp.asarray(reference["mask"][None, ..., None]) >= 128).astype(jnp.float32)
        encode_key = jax.random.split(key)[1] if kind == "inpaint" else None
        spatial = latent_image_conditions(source.autoencoder, source.variables["autoencoder"], pixels, mask, encode_key)
        given, null = {**given, **spatial}, {**null, **spatial}
    denoise = process.denoiser(source.model, source.variables, given, null)
    latents = sample(denoise, initial, solver=source.schedule.solver, guidance=CFG(3.0), key=key,
                     times=times, final_denoise=False)
    images = jnp.clip(source.autoencoder.decode(source.variables["autoencoder"], latents), -1, 1)
    if source.finish is not None:
        images = source.finish(source.variables, images)
    return images, latents, given, null, process, times, initial


def public_start(task, source, reference, meta, key, given, null, process, times):
    """The trajectory's start through the public `prepare`, from what the
    reference pipeline was handed: its pixels (uint8), mask, noise, clean
    latents or already-noisy latents. Diffusers' own preparation
    (prepare_latents, prepare_mask_latents, the refiner's denoising_start)
    made the reference's start, so its images hold the image, image-latents,
    mask and encode-key branches here. Strength and denoising_start reach
    `prepare` as the selected `times`, which is the caller's conversion."""
    kind = task_kind(source, meta)
    noise = jnp.asarray(reference["noise"])
    pixels = reference["pixels"][None] if "pixels" in reference else None
    # The text conditions alone: `prepare` builds the spatial ones itself.
    shared = {"conditions": {"conditioning": given["conditioning"]},
              "unconditional": {"conditioning": null["conditioning"]},
              "key": key, "steps": meta.get("steps", 2), "times": times}
    if kind == "img2img":
        return task.prepare(**shared, image=pixels, noise=noise, encode_key=key)
    def clean():
        pixels_in_range = unit_range(jnp.asarray(pixels))
        return source.autoencoder.encode(source.variables["autoencoder"], pixels_in_range, None)

    if kind == "xl-img2img":
        prepared = task.prepare(**shared, image=pixels, noise=noise)
        bypassed = task.prepare(**shared, image_latents=clean(), noise=noise)
        np.testing.assert_allclose(bypassed.noise, prepared.noise, atol=1e-6, rtol=0)
        return prepared
    if kind == "refiner":
        return task.prepare(**shared, initial=clean())
    prior = noise * process.sampler_schedule.prior_scale()
    if kind in ("inpaint", "xl-inpaint"):
        # Full strength: the reference starts from noise alone, its image and
        # mask as conditions.
        encode_key = jax.random.split(key)[1] if kind == "inpaint" else None
        return task.prepare(**shared, image=pixels, mask=reference["mask"][None, ..., None], initial=prior,
                            encode_key=encode_key)
    return task.prepare(**shared, initial=prior)


def diffusers_reads(source, export, prediction, noise, given):
    """Diffusers' own classes load every weighted component of `export`
    cleanly (tools/diffusers_consumer.py; a component nothing here reads, the
    safety checker, is held to the `source` directory's file, which it
    writes back untrained), and its declared UNet, over the
    same noise, timestep and conditions in float32 and float64, holds Dew's
    trained `prediction` to tests/reference_error.py's rule, and returns
    the rule's ratio."""
    from reference_error import assert_as_exact_as_the_reference, distance
    from safetensors.numpy import load_file

    from tools import diffusers_consumer as consumer

    consumer.assert_components_load(export)
    for name in consumer.unread(export):
        held, written = (load_file(str(Path(root) / name / "model.safetensors")) for root in (source, export))
        assert held.keys() == written.keys(), name
        for key, value in held.items():
            np.testing.assert_array_equal(written[key], value, err_msg=f"{name}/{key}")
    condition = given["conditioning"]
    sample = np.concatenate([np.asarray(noise)] + [np.asarray(given[name]) for name in ("mask", "masked_image")
                                                   if name in given], -1)
    inputs = (export, sample, 10.0, np.asarray(condition.context),
              None if condition.pooled is None else np.asarray(condition.pooled),
              None if condition.time_ids is None else np.asarray(condition.time_ids))
    theirs, truth = consumer.unet_prediction(*inputs), consumer.unet_prediction(*inputs, wide=True)
    assert_as_exact_as_the_reference(np.asarray(prediction), theirs, truth, "trained UNet")
    return distance(np.asarray(prediction), truth) / distance(theirs, truth)


def train_and_reload(source, reference, key, given, noise):
    objective = DiffusionObjective(source.model, source.process, source.inputs, autoencoder=source.autoencoder,
                                   pretrained=source.variables, unconditional_prob=0.0, ema_decay=None, steps=2)
    pixels = reference["pixels"][None] if "pixels" in reference else np.rint(reference["images"] * 255).astype(np.uint8)
    batch = {"image": pixels, **source.inputs.tokenize(["cat"])}
    if source.inputs.mask is not None:
        batch[source.inputs.mask.key] = (reference["mask"][None, ..., None] >= 128).astype(np.float32)
    variables = objective.init(key)
    step = Step(jnp.asarray(0), key, None)
    def loss(params):
        value, _ = objective.loss({**variables, "params": params}, batch, step)
        return value.total / value.mass
    value, grads = jax.jit(jax.value_and_grad(loss))(variables["params"])
    updated = jax.tree.map(lambda p, g: p - 1e-3 * g, variables["params"], grads)
    after = loss(updated)
    assert float(after) < float(value), (value, after)
    trained = {**variables, "params": updated}
    with tempfile.TemporaryDirectory(prefix="dew-native-reload-") as saved:
        source.save(saved, variables=trained)
        restored = bundle(saved)
        before = source.model.apply(trained, noise, jnp.asarray([10]), **given)
        loaded = restored.model.apply(restored.variables, noise, jnp.asarray([10]), **given)
        np.testing.assert_array_equal(loaded, before)
        ratio = diffusers_reads(source.source, saved, before, noise, given)
        np.testing.assert_array_equal(restored.autoencoder.decode(restored.variables["autoencoder"], noise),
                                      source.autoencoder.decode(trained["autoencoder"], noise))
        if source.finish is not None:
            images = jnp.asarray(reference["images"] * 2 - 1)
            np.testing.assert_array_equal(restored.finish(restored.variables, images), source.finish(trained, images))
            checker = {**trained["encoders"]["safety"], "concept_embeds_weights": jnp.full((17,), -2.0)}
            blocked = {**trained, "encoders": {**trained["encoders"], "safety": checker}}
            source.save(saved, variables=blocked)
            restored = bundle(saved)
            np.testing.assert_array_equal(restored.finish(restored.variables, images), -1)
    return {"loss_before": float(value), "loss_after": float(after), "diffusers_ratio": ratio}


def check_pipeline(directory, streamed: bool = False):
    directory = Path(directory)
    source = bundle(directory, streamed)
    reference = np.load(directory / "reference.npz")
    meta = json.loads((directory / "reference.json").read_text()) if (directory / "reference.json").exists() else {}
    errors, key = {}, jax.random.PRNGKey(17)
    images, latents, given, null, process, times, initial = trajectory(source, reference, meta, key)
    if "hidden" in reference:
        compare(errors, "hidden", given["conditioning"].context, reference["hidden"])
    if "pooled" in reference:
        compare(errors, "pooled", given["conditioning"].pooled, reference["pooled"])
    if "encoded" in reference:
        pixels = unit_range(jnp.asarray(reference["pixels"][None]))
        compare(errors, "encode", source.autoencoder.encode(source.variables["autoencoder"], pixels, key), reference["encoded"])
        compare(errors, "decode", source.autoencoder.decode(source.variables["autoencoder"], jnp.asarray(reference["noise"])), reference["decoded"])
    if "final_latents" in reference:
        compare(errors, "latents", latents, reference["final_latents"])
    compare(errors, "trajectory_images", (images + 1) / 2, reference["images"], 1e-4)
    if source.finish is not None:
        raw = np.rint(reference["images"] * 255).astype(np.uint8)
        pixels = source.finish.transform(raw)
        compare(errors, "checker_pixels", pixels, reference["checker_pixels"])
        features = source.finish.model.apply({"params": source.variables["encoders"]["safety"]}, pixels,
                                             method=source.finish.model.features)
        compare(errors, "checker_features", features, reference["checker_embeddings"])
    # The same trajectory through the task facade, its start prepared by the
    # public `prepare` from the reference's own inputs.
    task = source.text_to_image()
    prepared = public_start(task, source, reference, meta, key, given, null, process, times)
    # Its real rows: on a mesh the rows pad to fill the devices.
    compare(errors, "prepared", np.asarray(prepared.noise)[:prepared.rows], initial, 1e-6)
    for name in ("mask", "masked_image"):
        if name in given:
            compare(errors, f"prepared_{name}", np.asarray(prepared.conditions[name])[:prepared.rows],
                    given[name], 1e-6)
    output = task(prepared, steps=meta.get("steps", 2), guidance=3.0, key=key)
    compare(errors, "task", output.host().images, images, 1e-5)
    compare(errors, "task_reference", (output.host().images + 1) / 2, reference["images"], 1e-4)
    training = train_and_reload(source, reference, key, given, jnp.asarray(reference["noise"]))
    errors["reload"] = 0.0
    print(json.dumps({"directory": str(directory), "errors": errors, "training": training}))


def check_grids(directory, grids, case=None):
    oracle = np.load(grids)
    source = bundle(directory)
    given, null = conditions(source)
    errors = {}
    cases = (case,) if case else ("pndm-prk", "pndm-plms", "lms", "lms-v", "lms-karras", "euler")
    for name in cases:
        schedule = SourceSchedule.from_config(json.loads(str(oracle[name + ".config"])))
        process, times = schedule.sampling(4)
        noise = jnp.asarray(oracle["noise"]) * process.sampler_schedule.prior_scale()
        task = replace(source.text_to_image(), grid=lambda count: (process, times), solver=schedule.solver)
        output = task(DenoisingInputs(noise, given, null, rows=1), steps=4, guidance=3.0, key=jax.random.PRNGKey(0))
        expected = jnp.clip(source.autoencoder.decode(source.variables["autoencoder"], jnp.asarray(oracle[name + ".latents"])), -1, 1)
        compare(errors, name, output.host().images, expected, 1e-4)
    print(json.dumps({"grids": errors}))


def handoff(directory, source, task, key):
    """SDXL's base walks the steps at or above the cutoff and stops
    (final_denoise=False, decode=False); the refiner continues from those
    latents (`initial=`) below it, as Diffusers' denoising_end and
    denoising_start split them. tools/diffusers_handoff_reference.py
    recorded Diffusers' own base and refiner pipelines doing so."""
    from reference_error import assert_as_exact_as_the_reference

    recorded = np.load(Path(__file__).resolve().parents[1] / "tests/fixtures/diffusers_handoff.npz")
    refiner = bundle(Path(directory).parent / "refiner")
    steps = 4

    def start(task, rows):
        encoder = task.inputs.conditions["conditioning"].encoder
        params = task.variables["encoders"]["conditioning"]
        given, null = ({"conditioning": encoder.encode(params, encoder.tokenize([row]))}
                       for row in ({"text": "cat"}, {"text": "dog", "negative": True}))
        process, times = task.prepared_process(steps)
        # Diffusers' discrete cutoff for denoising_end = denoising_start = 0.5.
        cutoff = round(len(source.schedule.betas) * 0.5)
        model_times = np.asarray(process.sampler_schedule.model_time(jnp.asarray(times[:-1])))
        split = int(np.flatnonzero(model_times < cutoff)[0])
        return given, null, process, (times[:split + 1] if rows == "prefix" else times[split:])

    # The base declares Flax classes; the reference runs their PyTorch
    # twin, whose UNet computes the exact GELU and normalizes attention
    # inputs at 1e-6 (tests/test_lora.py `torch_unet`).
    twin = replace(task.model, approximate_gelu=False, attention_norm_epsilon=1e-6)
    base_task = replace(task, model=twin, final_denoise=False)
    given, null, process, prefix = start(base_task, "prefix")
    initial = jnp.asarray(recorded["noise"]) * process.sampler_schedule.prior_scale()
    prepared = base_task.prepare(conditions=given, unconditional=null, key=key, steps=steps, initial=initial,
                                 times=prefix)
    latents = base_task(prepared, steps=steps, guidance=3.0, key=key, decode=False).host().latents
    refiner_task = replace(refiner.text_to_image(), final_denoise=False)
    given, null, _, tail = start(refiner_task, "tail")
    prepared = refiner_task.prepare(conditions=given, unconditional=null, key=key, steps=steps,
                                    initial=latents, times=tail)
    final = refiner_task(prepared, steps=steps, guidance=3.0, key=key, decode=False).host().latents
    images = refiner.autoencoder.decode(refiner.variables["autoencoder"], jnp.asarray(final))
    for name, ours in (("prefix", latents), ("final", final), ("images", images)):
        assert_as_exact_as_the_reference(np.asarray(ours), recorded[f"fp32.{name}"], recorded[f"fp64.{name}"],
                                         f"handoff {name}")


def mask_levels(directory, source, task, key):
    """Every byte level fills one latent cell's block of a mask, through the
    public `prepare` and through the inpainting pipeline's own mask
    processor and its nearest resize to the latent grid."""
    import torch
    from diffusers.image_processor import VaeImageProcessor
    from PIL import Image

    reference = np.load(Path(directory) / "reference.npz")
    height, width = reference["mask"].shape
    factor = source.autoencoder.downscale_factor
    grid = (height // factor, width // factor)
    rows = -(-256 // (grid[0] * grid[1]))
    levels = np.resize(np.arange(256, dtype=np.uint8), (rows, *grid))
    masks = np.repeat(np.repeat(levels, factor, axis=1), factor, axis=2)[..., None]
    prepared = task.prepare(["cat"] * rows, key=key, steps=2, image=reference["pixels"][None], mask=masks)
    processor = VaeImageProcessor(vae_scale_factor=factor, do_normalize=False, do_binarize=True,
                                  do_convert_grayscale=True)
    pictures = [Image.fromarray(mask[..., 0]) for mask in masks]
    theirs = processor.preprocess(pictures, height=height, width=width)
    theirs = torch.nn.functional.interpolate(theirs, size=grid)
    np.testing.assert_array_equal(np.asarray(prepared.conditions["mask"])[..., 0], theirs[:, 0].numpy())
    assert set(np.unique(theirs.numpy())) == {0.0, 1.0}


def regress(directory, case):
    source = bundle(directory)
    task = source.text_to_image()
    key = jax.random.PRNGKey(17)
    options = dict(key=key, steps=2, guidance=3.0)
    if case == "prepared":
        prepared = task.prepare(["cat"], key=key, steps=2)
        np.testing.assert_array_equal(task(prepared, **options).host().images, task(["cat"], **options).host().images)
    elif case == "geometry":
        old = source.inputs.conditions["conditioning"]
        encoder = replace(old.encoder, height=64, width=32)
        inputs = replace(source.inputs, sample=Field("image", (64, 32, 3)),
                         conditions={"conditioning": replace(old, encoder=encoder)})
        source = replace(source, inputs=inputs)
        expected = source.text_to_image()(["cat"], **options).host().images
        with tempfile.TemporaryDirectory(prefix="dew-native-geometry-") as saved:
            source.save(saved)
            np.testing.assert_array_equal(bundle(saved).text_to_image()(["cat"], **options).host().images, expected)
    elif case == "negative-policy":
        prepared = task.prepare(["cat"], key=key, steps=2)
        encoder = source.inputs.conditions["conditioning"].encoder
        negative = encoder.encode(source.variables["encoders"]["conditioning"],
                                   encoder.tokenize([{"text": "", "negative": True}]))
        expected = task(prepared.replace(unconditional={"conditioning": negative}), **options).host().images
        with tempfile.TemporaryDirectory(prefix="dew-native-policy-") as saved:
            source.save(saved)
            index = json.loads((Path(saved) / "model_index.json").read_text())
            index["force_zeros_for_empty_prompt"] = False
            (Path(saved) / "model_index.json").write_text(json.dumps(index))
            np.testing.assert_allclose(bundle(saved).text_to_image()(["cat"], **options).host().images, expected, atol=1e-5, rtol=1e-5)
        assert not np.allclose(task(["cat"], **options).host().images, expected, atol=1e-7)
    elif case == "handoff":
        handoff(directory, source, task, key)
    elif case == "mask-levels":
        mask_levels(directory, source, task, key)
    elif case == "mask-dropout":
        reference = np.load(Path(directory) / "reference.npz")
        objective = DiffusionObjective(source.model, source.process, source.inputs, autoencoder=source.autoencoder,
                                       pretrained=source.variables, unconditional_prob=1.0, ema_decay=None, steps=2)
        batch = {"image": reference["pixels"][None], **source.inputs.tokenize(["cat"]),
                 "mask": (reference["mask"][None, ..., None] >= 128).astype(np.float32)}
        other = {**batch, "mask": 1 - batch["mask"]}
        def loss(variables, data):
            value, _ = objective.loss({**source.variables, "params": variables}, data, Step(jnp.asarray(0), key, None))
            return value.total / value.mass
        gradient = jax.jit(jax.grad(loss))
        first, second = gradient(source.variables["params"], batch), gradient(source.variables["params"], other)
        difference = max(float(jnp.max(jnp.abs(a - b))) for a, b in zip(jax.tree.leaves(first), jax.tree.leaves(second)))
        assert difference > 1e-5
    else:
        raise ValueError(f"Unknown regression {case}")
    print(case, "passed")


def check_model(directory):
    from dew.interop.diffusion import component_tensors, translate_unet_weights, unet_fields
    from dew.diffusion.process import DenoisingCondition
    from dew.nn.backbones.unet_condition import UNet2DCondition
    directory = Path(directory)
    reference = np.load(directory / "reference.npz")
    model = UNet2DCondition(**unet_fields(json.loads((directory / "unet/config.json").read_text()), attention_impl="xla"))
    parameters, _ = translate_unet_weights(component_tensors(directory, "unet"), model)
    condition = DenoisingCondition(jnp.asarray(reference["context"]),
        jnp.asarray(reference["pooled"]) if "pooled" in reference else None,
        jnp.asarray(reference["time_ids"]) if "time_ids" in reference else None)
    def predict(value):
        return model.apply({"params": parameters}, value, jnp.asarray(reference["time"]), conditioning=condition)
    x = jnp.asarray(reference["input"])
    output = jax.jit(predict)(x)
    gradient = jax.jit(jax.grad(lambda value: jnp.sum(predict(value) * reference["probe"])))(x)
    errors = {}
    compare(errors, "prediction", output, reference["output"], 1e-4)
    compare(errors, "input_vjp", gradient, reference["vjp"], 1e-4)
    print(json.dumps({"model": directory.name, "errors": errors}))


if __name__ == "__main__":
    import sys

    # tools.* and the tests' reference_error, as the test suite imports them.
    ROOT = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(ROOT), str(ROOT / "tests")]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    parser.add_argument("--grids")
    parser.add_argument("--case")
    parser.add_argument("--model", action="store_true")
    parser.add_argument("--streamed", action="store_true", help="load the pipeline streamed onto a mesh")
    args = parser.parse_args()
    if args.model:
        check_model(args.directory)
    elif args.grids:
        check_grids(args.directory, args.grids, args.case)
    elif args.case:
        regress(args.directory, args.case)
    else:
        check_pipeline(args.directory, args.streamed)
