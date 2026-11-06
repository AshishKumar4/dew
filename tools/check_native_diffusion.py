"""Native diffusion checkpoint, trajectory and update checks against saved oracles.

The runtime under test uses only Dew modules, Process and native solvers.
Diffusers/Transformers model implementations run in the reference tools,
and here on the trained export alone: Diffusers' own classes load every
component Dew wrote and its UNet recomputes the trained prediction
(`diffusers_reads`).
"""
import argparse
import json
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from dew.diffusion.schedules.source import SourceSchedule
from dew.inference import DenoisingInputs
from dew.inputs import Field, unit_range
from dew.interop import Pretrained
from dew.objectives.base import Step
from dew.objectives.diffusion import DiffusionObjective
from dew.sampling.guidance import CFG
from dew.sampling.walks import sample


def bundle(directory, streamed: bool = False):
    """The pipeline, loaded whole, or streamed onto a mesh a leaf at a time."""
    from dew.training import MeshSpec

    return Pretrained.load(str(directory), dtype="float32", attention_impl="xla",
                           **({"mesh": MeshSpec()} if streamed else {}))


TRUTH = Path(__file__).resolve().parents[1] / "tests/fixtures/tiny_diffusers_f64.npz"
"""Each saved task's references rerun in float64 (`truth` in tools/diffusers_pipeline_reference.py
and tools/diffusers_extended_reference.py)."""


def held(errors, task, name, actual, reference, key):
    """`actual` as near float64 as the source's float32 `reference`, the truth
    the same saved pipeline computes in float64 (tests/reference_error.py)."""
    from one_draw import assert_as_exact_as_the_reference
    from reference_error import distance

    with np.load(TRUTH) as truth:
        wide = truth[f"{task}/{key}"]
    actual = np.asarray(actual, np.float64)
    errors[name] = {"dew": distance(actual, wide), "reference": distance(reference, wide)}
    assert_as_exact_as_the_reference(actual, reference, wide, f"{task} {name}")


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


def selection(source, meta):
    """The text conditions and the times the source's call walks: all of
    them, or those its strength or the refiner's denoising_start selects,
    which is the caller's conversion."""
    kind = task_kind(source, meta)
    given, null = conditions(source, kind.startswith("xl-") or kind == "refiner")
    steps = meta.get("steps", 2)
    process, times = source.schedule.sampling(steps)
    if kind == "refiner":
        model_times = np.asarray(process.sampler_schedule.model_time(times[:-1]))
        cutoff = round(len(source.schedule.betas) * (1 - meta["denoising_start"]))
        times = times[int(np.flatnonzero(model_times < cutoff)[0]):]
    elif kind in ("img2img", "xl-img2img"):
        times = times[steps - int(steps * meta.get("strength", 1.0)):]
    return given, null, process, times


def walked(prepared, given, null):
    """The conditions a walk from the public `prepare`'s start walks under: the text ones and the
    spatial ones `prepare` built, its real rows alone: on a mesh the rows pad to fill the devices."""
    spatial = {name: prepared.conditions[name][:prepared.rows] for name in ("mask", "masked_image")
               if name in prepared.conditions}
    return {**given, **spatial}, {**null, **spatial}


def trajectory(source, prepared, given, null, process, times, key):
    """The single solver walk from the public `prepare`'s start, the one
    path every caller starts from, and the conditions it walks under (`walked`)."""
    given, null = walked(prepared, given, null)
    denoise = process.denoiser(source.model, source.variables, given, null)
    latents = sample(denoise, prepared.noise[:prepared.rows], solver=source.solver, guidance=CFG(3.0),
                     key=key, times=times, final_denoise=False)
    images = jnp.clip(source.autoencoder.decode(source.variables["autoencoder"], latents), -1, 1)
    if source.finish is not None:
        images = source.finish(source.variables, images)
    return images, latents, given


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
        from one_draw import assert_as_exact_as_the_reference

        prepared = task.prepare(**shared, image=pixels, noise=noise)
        bypassed = task.prepare(**shared, image_latents=clean(), noise=noise)
        # Both starts from Dew's own encoding (compiled in `prepare`, eager here), held to the
        # start the source's float64 encoding makes under the same rates and noise.
        alpha, sigma = (np.asarray(rate, np.float64).reshape(-1, 1, 1, 1)
                        for rate in process.sampler_schedule.rates(jnp.full((1,), times[0])))
        with np.load(TRUTH) as truth:
            start = alpha * truth["xl-img2img/image_latents"] + sigma * np.asarray(noise, np.float64)
        rows = prepared.rows
        assert_as_exact_as_the_reference(np.asarray(bypassed.noise)[:rows], np.asarray(prepared.noise)[:rows],
                                         start, "bypassed image latents")
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


TRAINED_SEEDS = 8
"""Every task's declared seeds for check 1 of its trained UNet against Diffusers' (`diffusers_reads`),
sized by `seeds_for` paired with Dew's on armada (11-06): 8 at each of the eight tasks."""
TRAINED_TIME = 10.0
"""The timestep the trained UNet is read at."""


def diffusers_reads(task, source, export, model, variables, noise, given):
    """Diffusers' own classes load every weighted component of `export`
    cleanly (tools/diffusers_consumer.py; a component nothing here reads, the
    safety checker, is held to the `source` directory's file, which it
    writes back untrained), and its declared UNet holds Dew's trained one
    on par (check 1) over drawn noise, prompt states, pooled embedding and
    inpaint channels, of `noise`'s and `given`'s shapes, at `given`'s time
    ids, both measured from Diffusers' own UNet run in float64."""
    from reference_error import as_given, assert_as_exact
    from safetensors.numpy import load_file

    from dew.diffusion.process import DenoisingCondition
    from tools import diffusers_consumer as consumer

    consumer.assert_components_load(export)
    for name in consumer.unread(export):
        held, written = (load_file(str(Path(root) / name / "model.safetensors")) for root in (source, export))
        assert held.keys() == written.keys(), name
        for key, value in held.items():
            np.testing.assert_array_equal(written[key], value, err_msg=f"{name}/{key}")
    condition = given["conditioning"]
    spatial = [name for name in ("mask", "masked_image") if name in given]
    shapes = {"noise": np.shape(noise), "context": np.shape(condition.context),
              **({} if condition.pooled is None else {"pooled": np.shape(condition.pooled)}),
              **{name: np.shape(given[name]) for name in spatial}}

    def draw(seed):
        rng = np.random.default_rng([0x7A2, seed])
        return tuple(rng.standard_normal(shape).astype(np.float32) for shape in shapes.values())

    def named(inputs):
        return dict(zip(shapes, inputs, strict=True))

    def dew(*inputs):
        drawn = named(inputs)
        conditioning = DenoisingCondition(drawn["context"], drawn.get("pooled"), condition.time_ids)
        return model.apply(variables, drawn["noise"], jnp.full((1,), TRAINED_TIME),
                           conditioning=conditioning, **{name: drawn[name] for name in spatial})

    def diffusers(wide):
        predict = consumer.unet_predictor(export, wide=wide)

        def run(*inputs):
            drawn = named(inputs)
            sample = np.concatenate([drawn["noise"], *(drawn[name] for name in spatial)], -1)
            return predict(sample, TRAINED_TIME, drawn["context"], drawn.get("pooled"),
                           None if condition.time_ids is None else np.asarray(condition.time_ids))
        return as_given(run)

    assert_as_exact(dew, diffusers(False), draw, f"{task} trained UNet", seeds=TRAINED_SEEDS,
                    truth_fn=diffusers(True))


def train_and_reload(task, source, reference, key, given, noise):
    objective = DiffusionObjective(source.model, source.process, source.inputs, autoencoder=source.autoencoder,
                                   variables=source.variables, unconditional_prob=0.0, ema_decay=None, steps=2)
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
        diffusers_reads(task, source.source, saved, restored.model, restored.variables, noise, given)
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
    return {"loss_before": float(value), "loss_after": float(after)}


PARTS = ("walk", "update")
"""What a pipeline's check holds (`check_pipeline`): the suite runs each part in a process of its own, and a
call that names no `--part` runs both in one."""


def check_pipeline(directory, part: str, streamed: bool = False):
    """The saved pipeline against the source: its trajectory (`part` 'walk') or its update, reload and
    export ('update'). With `streamed`, the pipeline streamed onto a mesh a leaf at a time loads the
    same tree, placed on the mesh, its trajectory's images are held to the source as the whole load's
    are (`streamed_parity`), and it is the pipeline the update, reload and export run on, under the
    conditions its public `prepare` builds."""
    directory = Path(directory)
    reference = np.load(directory / "reference.npz")
    meta = json.loads((directory / "reference.json").read_text()) if (directory / "reference.json").exists() else {}
    errors, key = {}, jax.random.PRNGKey(17)
    if part == "walk":
        source = bundle(directory)
        given, null, process, times = selection(source, meta)
        qualified(errors, directory.name, source, reference, meta, key, given, null, process, times)
        if streamed:
            streamed_parity(errors, directory, source, reference, meta, key)
        print(json.dumps({"directory": str(directory), "errors": errors}))
        return
    source = bundle(directory, streamed=streamed)
    given, null, process, times = selection(source, meta)
    prepared = public_start(source.text_to_image(), source, reference, meta, key, given, null, process, times)
    given, _ = walked(prepared, given, null)
    noise = jnp.asarray(reference["noise"])
    training = train_and_reload(directory.name, source, reference, key, given, noise)
    print(json.dumps({"directory": str(directory), "reload": 0.0, "training": training}))


def qualified(errors, name, source, reference, meta, key, given, null, process, times, label=""):
    """`source`'s components and walk held to the source, as one load or the other loads them: the
    conditioning (hidden, pooled), the autoencoder's encode and decode, the final latents, the
    trajectory's images, the safety checker's pixels and features, and the public task call walking
    to the same images. `label` prefixes each measurement's name, so two loads report apart."""
    # The start the public `prepare` makes from what the reference pipeline was handed.
    task = source.text_to_image()
    prepared = public_start(task, source, reference, meta, key, given, null, process, times)
    images, latents, given = trajectory(source, prepared, given, null, process, times, key)
    if "hidden" in reference:
        held(errors, name, f"{label}hidden", given["conditioning"].context, reference["hidden"], "hidden")
    if "pooled" in reference:
        held(errors, name, f"{label}pooled", given["conditioning"].pooled, reference["pooled"], "pooled")
    if "encoded" in reference:
        pixels = unit_range(jnp.asarray(reference["pixels"][None]))
        encoded = source.autoencoder.encode(source.variables["autoencoder"], pixels, key)
        held(errors, name, f"{label}encode", encoded, reference["encoded"], "encoded")
        held(errors, name, f"{label}decode",
             source.autoencoder.decode(source.variables["autoencoder"], jnp.asarray(reference["noise"])),
             reference["decoded"], "decoded")
    if "final_latents" in reference:
        held(errors, name, f"{label}latents", latents, reference["final_latents"], "final_latents")
    shown = (np.asarray(images, np.float64) + 1) / 2
    held(errors, name, f"{label}trajectory_images", shown, reference["images"], "images")
    if source.finish is not None:
        raw = np.rint(reference["images"] * 255).astype(np.uint8)
        pixels = source.finish.transform(raw)
        held(errors, name, f"{label}checker_pixels", pixels, reference["checker_pixels"], "checker_pixels")
        # The tower on the source's own pixels, each stage held on the input the source gave it:
        # Dew's pixels carry their own float32 rounding, held above, which the tower amplifies.
        features = source.finish.model.apply({"params": source.variables["encoders"]["safety"]},
                                             jnp.asarray(reference["checker_pixels"]),
                                             method=source.finish.model.features)
        held(errors, name, f"{label}checker_features", features, reference["checker_embeddings"],
             "checker_embeddings")
    # The same walk through the task facade: the same images.
    output = task(prepared, steps=meta.get("steps", 2), guidance=3.0, key=key)
    np.testing.assert_array_equal(output.host().images, images)


def streamed_parity(errors, directory, whole, reference, meta, key):
    """The pipeline streamed onto a mesh a leaf at a time: every leaf the whole load holds, bit for
    bit, each placed on the mesh, and every component and its walk held to the source as the whole
    load's are (`qualified`): another program than the whole load's, which XLA need not round alike."""
    streamed = bundle(directory, streamed=True)
    assert jax.tree.structure(streamed.variables) == jax.tree.structure(whole.variables)
    for path, leaf in jax.tree_util.tree_leaves_with_path(streamed.variables):
        assert isinstance(leaf.sharding, jax.sharding.NamedSharding), jax.tree_util.keystr(path)
    for left, right in zip(jax.tree.leaves(streamed.variables), jax.tree.leaves(whole.variables),
                           strict=True):
        np.testing.assert_array_equal(np.asarray(left), np.asarray(right))
    given, null, process, times = selection(streamed, meta)
    qualified(errors, directory.name, streamed, reference, meta, key, given, null, process, times,
              label="streamed_")


def check_grids(directory, grids, case=None):
    from one_draw import assert_as_exact_as_the_reference
    from reference_error import distance

    oracle = np.load(grids)
    source = bundle(directory)
    given, null = conditions(source)
    errors = {}
    cases = (case,) if case else sorted(name.removesuffix(".config") for name in oracle.files
                                        if name.endswith(".config"))
    for name in cases:
        schedule = SourceSchedule.from_config(json.loads(str(oracle[name + ".config"])))
        process, times = schedule.sampling(4)
        noise = jnp.asarray(oracle["noise"]) * process.sampler_schedule.prior_scale()
        task = replace(source.text_to_image(), grid=lambda count: (process, times), solver=schedule.solver)
        output = task(DenoisingInputs(noise, given, null, rows=1), steps=4, guidance=3.0, key=jax.random.PRNGKey(0),
                      decode=False)
        # The walk's latents, as near float64 as the source scheduler's float32 walk over the same
        # predictions (tools/diffusers_scheduler_pipeline_reference.py --truth).
        latents = np.asarray(output.host().latents, np.float64)
        errors[name] = {"dew": distance(latents, oracle[name + ".latents.fp64"]),
                        "reference": distance(oracle[name + ".latents"], oracle[name + ".latents.fp64"])}
        # Every grid is walked in one process over the one loaded pipeline, and each case's verdict
        # is its own: a grid off the source is reported by name and the others still run.
        try:
            assert_as_exact_as_the_reference(latents, oracle[name + ".latents"],
                                             oracle[name + ".latents.fp64"], f"{name} grid")
        except AssertionError as failed:
            errors[name]["failed"] = str(failed)
    print(json.dumps({"grids": errors}))
    return not any("failed" in error for error in errors.values())


def handoff(directory, source, task, key):
    """SDXL's base walks the steps at or above the cutoff and stops
    (final_denoise=False, decode=False); the refiner continues from those
    latents (`initial=`) below it, as Diffusers' denoising_end and
    denoising_start split them. tools/diffusers_handoff_reference.py
    recorded Diffusers' own base and refiner pipelines doing so."""
    from one_draw import assert_as_exact_as_the_reference

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
            np.testing.assert_array_equal(bundle(saved).text_to_image()(["cat"], **options).host().images, expected)
        assert not np.array_equal(task(["cat"], **options).host().images, expected)
    elif case == "handoff":
        handoff(directory, source, task, key)
    elif case == "mask-levels":
        mask_levels(directory, source, task, key)
    elif case == "mask-dropout":
        reference = np.load(Path(directory) / "reference.npz")
        objective = DiffusionObjective(source.model, source.process, source.inputs, autoencoder=source.autoencoder,
                                       variables=source.variables, unconditional_prob=1.0, ema_decay=None, steps=2)
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


MODEL_TIME = 13.0
"""The geometry cases' UNet timestep, as tools/native_unet_reference.py made their weights for."""
MODEL_TIME_IDS = {"sdxl-inpaint": (32, 48, 0, 0, 32, 48), "refiner": (32, 48, 0, 0, 6)}
"""SDXL's micro-conditioning where a case's UNet takes it: the original size, the crop and the target
size, the refiner's aesthetic score in place of its target."""
MODEL_ROWS = 4
"""A draw's independent rows: the pooled embedding's gradient holds 8 values a row, too few in one row for
its rounding to resolve over draws."""
MODEL_SEEDS = {"sd1": 8, "sd2": 8, "sdxl-inpaint": 16, "refiner": 256}
"""Each geometry case's declared seeds for check 1 (`assert_as_exact`), sized by `seeds_for` paired with
Dew's on armada (11-06); the reference compared with itself needed 16, 8, 256 and 256."""


def model_site(directory):
    """A multilevel UNet geometry's check-1 arms: Dew's UNet and Diffusers' own Flax UNet on the same
    weights, each one's prediction and the gradients of its drawn inputs (the latents, the prompt states,
    a pooled embedding where the UNet takes one) under a drawn cotangent, and Diffusers' module built at
    float64 for the truth, its weights widened. Returns `(dew, reference, truth, draw)`."""
    from diffusers import FlaxUNet2DConditionModel
    from reference_error import with_gradients

    from dew.diffusion.process import DenoisingCondition
    from dew.interop.diffusion import component_tensors, translate_unet_weights, unet_fields
    from dew.nn.backbones.unet_condition import UNet2DCondition

    directory = Path(directory)
    config = json.loads((directory / "unet/config.json").read_text())
    model = UNet2DCondition(**unet_fields(config, attention_impl="xla"))
    parameters, _ = translate_unet_weights(component_tensors(directory, "unet"), model)
    source, weights = FlaxUNet2DConditionModel.from_pretrained(directory / "unet")
    wide, _ = FlaxUNet2DConditionModel.from_pretrained(directory / "unet", dtype=jnp.float64)
    ids = MODEL_TIME_IDS.get(directory.name)
    time_ids = None if ids is None else jnp.asarray([ids] * MODEL_ROWS, jnp.int32)
    shapes = [(MODEL_ROWS, 8, 12, config["in_channels"]), (MODEL_ROWS, 8, config["cross_attention_dim"])]
    if ids is not None:
        # The pooled embedding: what the added projection reads beside the micro-conditioning's embeddings.
        embedded = len(ids) * config["addition_time_embed_dim"]
        pooled = config["projection_class_embeddings_input_dim"] - embedded
        shapes.append((MODEL_ROWS, pooled))

    def draw(seed):
        """The latents (Dew's channels-last), the prompt states, any pooled embedding, and the cotangent."""
        rng = np.random.default_rng([0x0E3, seed])
        latents, context, *pooled = (rng.standard_normal(shape) for shape in shapes)
        cotangent = rng.standard_normal((MODEL_ROWS, 8, 12, 4))
        return tuple(value.astype(np.float32) for value in (latents, context * 0.1, *pooled, cotangent))

    def dew(latents, context, *pooled):
        condition = DenoisingCondition(context, *pooled, *(() if time_ids is None else (time_ids,)))
        time = jnp.full((MODEL_ROWS,), MODEL_TIME, latents.dtype)
        return model.apply({"params": parameters}, latents, time, conditioning=condition)

    def diffusers(module):
        def run(latents, context, *pooled):
            dtype = latents.dtype
            added = None if time_ids is None else {"text_embeds": pooled[0], "time_ids": time_ids}
            time = jnp.full((MODEL_ROWS,), MODEL_TIME, dtype)
            output = module.apply({"params": jax.tree.map(lambda leaf: leaf.astype(dtype), weights)},
                                  latents.transpose(0, 3, 1, 2), time, context, added_cond_kwargs=added).sample
            return output.transpose(0, 2, 3, 1)
        return run

    return with_gradients(dew), with_gradients(diffusers(source)), with_gradients(diffusers(wide)), draw


def check_model(directory):
    """A multilevel UNet geometry on par with Diffusers' own over drawn inputs (check 1, `model_site`)."""
    from reference_error import assert_as_exact

    directory = Path(directory)
    dew, reference, truth, draw = model_site(directory)
    assert_as_exact(dew, reference, draw, f"{directory.name} prediction and gradients",
                    seeds=MODEL_SEEDS[directory.name], truth_fn=truth)
    print(json.dumps({"model": directory.name, "held": True}))


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
    parser.add_argument("--part", choices=PARTS, help="the pipeline check's part; both where none is named")
    args = parser.parse_args()
    if args.model:
        check_model(args.directory)
    elif args.grids:
        sys.exit(0 if check_grids(args.directory, args.grids, args.case) else 1)
    elif args.case:
        regress(args.directory, args.case)
    else:
        for part in PARTS if args.part is None else (args.part,):
            check_pipeline(args.directory, part, args.streamed)
