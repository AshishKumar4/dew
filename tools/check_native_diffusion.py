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
from diffusers_scheduler_pipeline_reference import SCHEDULERS
from native_diffusion_worker import GROUPS, groups, leaves

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


RECORDINGS = Path(__file__).resolve().parents[1] / "tests/fixtures/native_diffusion"
"""Each task's recordings, `<task>-<group>.npz` (tools/venv_reference.py's `native-<task>-<group>` sites,
tools/native_diffusion_worker.py): its saved pipeline's own Diffusers pipeline called on drawn inputs, each
group of its leaves its own recording."""
WALK_SEEDS = {**{task: {"prompt": 8, "autoencoder": 8, "walk": 8}
                 for task in ("sd", "xl", "img2img", "inpaint")},
              **{task: {"prompt": 128, "walk": 8} for task in ("xl-img2img", "xl-inpaint", "refiner")},
              "safety": {"walk": 8, "checker": 64}}
"""Each task's declared seeds for check 1 of each group of its leaves (tools/native_diffusion_worker.py's
`groups`), each group's recording sized by `seeds_for` paired with Dew's computation of its leaves
(tools/venv_reference.py, armada 11-06). The PyTorch SDXL towers' pooled embedding is heavy-tailed across
draws (Dew's CLIP layer norms take flax's one-pass variance where torch's is exact, row 4561)."""
SAVED = {"prompt": "cat", "negative": "dog", "prompt_2": "fox", "negative_2": "owl"}
"""The prompts the saved references were called with, which the update and regression checks keep."""
LETTERS = 4
"""A drawn prompt's letters: the saved tokenizer reads each as a token, eight positions holding the start,
the letters and the end with room to pad."""
EXTENDED = ("xl-img2img", "xl-inpaint", "refiner")
"""The tasks whose conditioning reads a second prompt (SDXL's second tower): the PyTorch SDXL calls'. The Flax
SDXL call gives both towers the one prompt."""


def draw_for(task: str):
    """`task`'s draws: its prompts as letter codes, the noise channels last, the pixels an image call
    starts from and the autoencoder encodes, and the safety checker's raw images and tower input; a source
    grid's walk (`grid`) its noise alone."""
    def draw(seed):
        rng = np.random.default_rng([0xD1F, seed])
        if task == "grid":
            return {"noise": rng.standard_normal((1, 8, 8, 4)).astype(np.float32)}

        def letters():
            return rng.integers(ord("a"), ord("z") + 1, LETTERS).astype(np.uint8)

        inputs = {"prompt": letters(), "negative": letters()}
        if task in EXTENDED:
            inputs.update(prompt_2=letters(), negative_2=letters())
        inputs["noise"] = rng.standard_normal((1, 8, 8, 4)).astype(np.float32)
        if task == "safety":
            inputs["checker_raw"] = rng.integers(0, 256, (1, 32, 32, 3)).astype(np.uint8)
            inputs["checker_input"] = rng.standard_normal((1, 3, 8, 8)).astype(np.float32)
        else:
            inputs["pixels"] = rng.integers(0, 256, (32, 32, 3)).astype(np.uint8)
        return inputs
    return draw


def prompts(inputs) -> dict[str, str]:
    """A draw's prompts as text."""
    return {name: bytes(np.asarray(inputs[name], np.uint8)).decode() for name in SAVED if name in inputs}


def conditions(source, texts=SAVED, extended=False):
    """The conditions of `texts`' prompt and negative, SDXL's second tower's with them where `extended`."""
    encoder = source.inputs.conditions["conditioning"].encoder
    params = source.variables["encoders"]["conditioning"]
    positive = {"text": texts["prompt"], **({"second": texts["prompt_2"]} if extended else {})}
    negative = {"text": texts["negative"], "negative": True,
                **({"second": texts["negative_2"]} if extended else {})}
    return ({"conditioning": encoder.encode(params, encoder.tokenize([positive]))},
            {"conditioning": encoder.encode(params, encoder.tokenize([negative]))})


def task_kind(source, meta):
    if "task" in meta:
        return meta["task"]
    name = source.config["model_index"]["_class_name"]
    return "inpaint" if "Inpaint" in name else "img2img" if "Img2Img" in name else "text"


def selection(source, meta, texts=SAVED):
    """The conditions of `texts` and the times the source's call walks: all of
    them, or those its strength or the refiner's denoising_start selects,
    which is the caller's conversion."""
    kind = task_kind(source, meta)
    given, null = conditions(source, texts, kind.startswith("xl-") or kind == "refiner")
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
    path every caller starts from: its images, its images before any safety
    checker, its last latents, and the conditions it walks under (`walked`)."""
    given, null = walked(prepared, given, null)
    denoise = process.denoiser(source.model, source.variables, given, null)
    latents = sample(denoise, prepared.noise[:prepared.rows], solver=source.solver, guidance=CFG(3.0),
                     key=key, times=times, final_denoise=False)
    decoded = source.autoencoder.decode(source.variables["autoencoder"], latents)
    images = unchecked = jnp.clip(decoded, -1, 1)
    if source.finish is not None:
        images = source.finish(source.variables, images)
    return images, unchecked, latents, given


def public_start(task, source, start, meta, key, given, null, process, times, bypassed=False):
    """The trajectory's start through the public `prepare`, from what the
    reference pipeline was handed (`start`: the saved reference's or a draw's
    pixels (uint8), noise, and the saved mask): its pixels, clean latents or
    already-noisy latents. Diffusers' own preparation (prepare_latents,
    prepare_mask_latents, the refiner's denoising_start) made the reference's
    start, so its images hold the image, image-latents, mask and encode-key
    branches here. Strength and denoising_start reach `prepare` as the
    selected `times`, which is the caller's conversion. `bypassed` starts
    SDXL's image call from Dew's own encoding of the pixels, as clean image
    latents, in place of the pixels."""
    kind = task_kind(source, meta)
    noise = jnp.asarray(start["noise"])
    pixels = start["pixels"][None] if "pixels" in start else None
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
        return task.prepare(**shared, **({"image_latents": clean()} if bypassed else {"image": pixels}),
                            noise=noise)
    if kind == "refiner":
        return task.prepare(**shared, initial=clean())
    prior = noise * process.sampler_schedule.prior_scale()
    if kind in ("inpaint", "xl-inpaint"):
        # Full strength: the reference starts from noise alone, its image and
        # mask as conditions.
        encode_key = jax.random.split(key)[1] if kind == "inpaint" else None
        return task.prepare(**shared, image=pixels, mask=start["mask"][None, ..., None], initial=prior,
                            encode_key=encode_key)
    return task.prepare(**shared, initial=prior)


TRAINED_TIME = 10.0
"""The timestep the trained UNet is read at."""


def diffusers_reads(task, source, export, model, variables, noise, given):
    """Diffusers' own classes load every weighted component of `export`
    cleanly (tools/diffusers_consumer.py; a component nothing here reads, the
    safety checker, is held to the `source` directory's file, which it
    writes back untrained), and its declared UNet holds Dew's trained one
    on par (check 1) over drawn noise, prompt states, pooled embedding and
    inpaint channels, of `noise`'s and `given`'s shapes, at `given`'s time
    ids, both measured from Diffusers' own UNet run in float64 (`assert_export_recomputes`)."""
    from diffusion_support import assert_export_recomputes
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
        return run

    assert_export_recomputes(jax.jit(dew), diffusers, draw, f"{task} trained UNet")


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
    same tree, placed on the mesh, its walk is held to the source as the whole load's is
    (`streamed_parity`), and it is the pipeline the update, reload and export run on, under the
    conditions its public `prepare` builds."""
    directory = Path(directory)
    reference = dict(np.load(directory / "reference.npz"))
    meta = json.loads((directory / "reference.json").read_text()) if (directory / "reference.json").exists() else {}
    key = jax.random.PRNGKey(17)
    if part == "walk":
        source = bundle(directory)
        qualified(directory.name, source, reference, meta, key)
        if streamed:
            streamed_parity(directory, source, reference, meta, key)
        print(json.dumps({"directory": str(directory), "walk": "held"}))
        return
    source = bundle(directory, streamed=streamed)
    given, null, process, times = selection(source, meta)
    prepared = public_start(source.text_to_image(), source, reference, meta, key, given, null, process, times)
    given, _ = walked(prepared, given, null)
    noise = jnp.asarray(reference["noise"])
    training = train_and_reload(directory.name, source, reference, key, given, noise)
    print(json.dumps({"directory": str(directory), "reload": 0.0, "training": training}))


def walk_leaves(name, source, meta, key, start, wanted=None, bypassed=False):
    """Dew's leaves `wanted` (all of the task's `leaves`, by default) from `start`'s prompts, noise and
    pixels: the prompt's states, the autoencoder's encode of the pixels and decode of the noise, the walk's
    start, last latents and images before any safety checker (`public_start`, `trajectory`), and the
    checker's transform and tower on the draw's own inputs, each as the recording keeps it. Only what the
    wanted leaves read is computed: the walk where a walk leaf is wanted."""
    wanted = leaves(name) if wanted is None else wanted
    autoencoder = source.variables["autoencoder"]
    computed = {}
    if {"hidden", "pooled"} & set(wanted):
        given, _ = conditions(source, prompts(start), task_kind(source, meta).startswith("xl-")
                              or task_kind(source, meta) == "refiner")
        computed.update(hidden=given["conditioning"].context, pooled=given["conditioning"].pooled)
    if "encoded" in wanted:
        pixels = unit_range(jnp.asarray(start["pixels"][None]))
        computed["encoded"] = source.autoencoder.encode(autoencoder, pixels, key)
    if "decoded" in wanted:
        computed["decoded"] = source.autoencoder.decode(autoencoder, jnp.asarray(start["noise"]))
    if set(GROUPS["walk"]) & set(wanted):
        given, null, process, times = selection(source, meta, prompts(start))
        prepared = public_start(source.text_to_image(), source, start, meta, key, given, null, process, times,
                                bypassed)
        _, unchecked, latents, _ = trajectory(source, prepared, given, null, process, times, key)
        computed.update(start=prepared.noise[:prepared.rows], final_latents=latents,
                        images=(np.asarray(unchecked, np.float64) + 1) / 2)
    if set(GROUPS["checker"]) & set(wanted):
        computed["checker_pixels"] = source.finish.transform(start["checker_raw"])
        computed["checker_embeddings"] = source.finish.model.apply(
            {"params": source.variables["encoders"]["safety"]}, jnp.asarray(start["checker_input"]),
            method=source.finish.model.features)
    return {leaf: np.asarray(computed[leaf]) for leaf in wanted}


def qualified(name, source, reference, meta, key, label=""):
    """`source`'s walk on par with its saved pipeline's own Diffusers pipeline over drawn prompts, noise
    and pixels (check 1, `walk_leaves` against the task's recording), each group of its leaves on its own
    seeds (`groups`, WALK_SEEDS), SDXL's image call from Dew's own encoding as clean image latents on par at
    its start too, and the public task call walking to the same images on the saved reference's inputs.
    `label` prefixes each check's name, so two loads report apart."""
    from reference_error import as_given, assert_as_exact, indexed_recording, recording

    draw = draw_for(name)
    mask = {"mask": reference["mask"]} if "mask" in reference else {}

    def dew(wanted, bypassed=False):
        return as_given(lambda index, inputs: walk_leaves(name, source, meta, key, {**inputs, **mask}, wanted,
                                                          bypassed))

    for group, held in groups(name).items():
        seeds, stored = WALK_SEEDS[name][group], recording(RECORDINGS / f"{name}-{group}.npz")
        drawn, recorded, truth = indexed_recording(stored, f"native-{name}-{group}", seeds, inputs_of=draw)
        assert_as_exact(dew(held), recorded, drawn, f"{label}{name} {group}", seeds=seeds, truth_fn=truth)
    if task_kind(source, meta) == "xl-img2img":
        seeds, stored = WALK_SEEDS[name]["walk"], recording(RECORDINGS / f"{name}-walk.npz")
        drawn, recorded, truth = indexed_recording(stored, f"native-{name}-walk", seeds, inputs_of=draw,
                                                   leaves=("start",))
        assert_as_exact(dew(("start",), bypassed=True), recorded, drawn, f"{label}{name} bypassed start",
                        seeds=seeds, truth_fn=truth)
        # And on par with Dew's own start from the pixels, whose Diffusers' start may be the farther: the
        # recorded walk's seeds, its truth recorded at those alone (a paired sizing draws past them).
        assert_as_exact(dew(("start",), bypassed=True), dew(("start",)), drawn,
                        f"{label}{name} bypassed start against the pixels' start", seeds=seeds,
                        truth_fn=truth)
    # The same walk through the task facade: the same images.
    given, null, process, times = selection(source, meta)
    task = source.text_to_image()
    prepared = public_start(task, source, reference, meta, key, given, null, process, times)
    images, *_ = trajectory(source, prepared, given, null, process, times, key)
    output = task(prepared, steps=meta.get("steps", 2), guidance=3.0, key=key)
    np.testing.assert_array_equal(output.host().images, images)


def streamed_parity(directory, whole, reference, meta, key):
    """The pipeline streamed onto a mesh a leaf at a time: every leaf the whole load holds, bit for
    bit, each placed on the mesh, and its walk held to the source as the whole load's is (`qualified`):
    another program than the whole load's, which XLA need not round alike."""
    streamed = bundle(directory, streamed=True)
    assert jax.tree.structure(streamed.variables) == jax.tree.structure(whole.variables)
    for path, leaf in jax.tree_util.tree_leaves_with_path(streamed.variables):
        assert isinstance(leaf.sharding, jax.sharding.NamedSharding), jax.tree_util.keystr(path)
    for left, right in zip(jax.tree.leaves(streamed.variables), jax.tree.leaves(whole.variables),
                           strict=True):
        np.testing.assert_array_equal(np.asarray(left), np.asarray(right))
    qualified(directory.name, streamed, reference, meta, key, label="streamed ")


def pipeline_arms(site, root=None):
    """tools/venv_reference.py's `native-<site>` site, `grid-<name>` or `(task, group)`: the recording, the
    draw, Dew's computation of a draw's leaves (`grid_walker`'s, or `walk_leaves`'s of the group) and those
    leaves, the saved pipeline loaded from `root`, or extracted once."""
    from interop_support import extract_fixture

    if root is None:
        root = extract_fixture(Path(__file__).resolve().parents[1] / "tests/fixtures/tiny_diffusers.tar.xz",
                               Path(tempfile.mkdtemp(prefix="dew-native-arms-")))
    if isinstance(site, str):
        walked = grid_walker(Path(root) / "sd")
        return (RECORDINGS / f"{site}.npz", draw_for("grid"),
                lambda dtype, inputs: walked(site.removeprefix("grid-"), inputs), ("latents",))
    task, group = site
    directory = Path(root) / task
    reference = np.load(directory / "reference.npz")
    meta = json.loads((directory / "reference.json").read_text()) if (directory / "reference.json").exists() else {}
    mask = {"mask": reference["mask"]} if "mask" in reference else {}
    source, key, held = bundle(directory), jax.random.PRNGKey(17), groups(task)[group]

    def dew(dtype, inputs):
        return walk_leaves(task, source, meta, key, {**inputs, **mask}, held)

    return RECORDINGS / f"{task}-{group}.npz", draw_for(task), dew, held


GRID_SEEDS = dict.fromkeys(SCHEDULERS, 8)
"""Each source grid's declared seeds for check 1 of the SD pipeline's walk over it, sized by `seeds_for` paired
with Dew's walk as its recording ran it (tools/venv_reference.py's `native-grid-<name>`)."""


def grid_walker(directory):
    """The SD pipeline at `directory` walking a draw's noise over a source scheduler's own grid, "dog"
    unconditioned and "cat" at guidance 3, four steps, as tools/diffusers_scheduler_pipeline_reference.py's
    `walk` does, its last latents (`{"latents": ...}`): a function of the grid's name and the draw."""
    from diffusers_scheduler_pipeline_reference import STEPS, scheduler

    source = bundle(directory)
    given, null = conditions(source)

    def walked(name, inputs):
        schedule = SourceSchedule.from_config(dict(scheduler(name, directory).config))
        process, times = schedule.sampling(STEPS)
        noise = jnp.asarray(inputs["noise"]) * process.sampler_schedule.prior_scale()
        task = replace(source.text_to_image(), grid=lambda count: (process, times), solver=schedule.solver)
        output = task(DenoisingInputs(noise, given, null, rows=1), steps=STEPS, guidance=3.0,
                      key=jax.random.PRNGKey(0), decode=False)
        return {"latents": np.asarray(output.host().latents)}

    return walked


def check_grids(directory, case=None):
    """The SD pipeline's walk over each source scheduler's own grid on par with Diffusers' scheduler over the
    pipeline's own predictions (check 1, against `native-grid-<name>`'s recording). Every grid is walked in one
    process over the one loaded pipeline, and each case's verdict is its own: a grid off the source is
    reported by name and the others still run."""
    from reference_error import as_given, assert_as_exact, indexed_recording, recording

    walked, errors = grid_walker(directory), {}
    for name in (case,) if case else GRID_SEEDS:
        stored = recording(RECORDINGS / f"grid-{name}.npz")
        drawn, recorded, truth = indexed_recording(stored, f"native-grid-{name}", GRID_SEEDS[name],
                                                   inputs_of=draw_for("grid"))
        try:
            assert_as_exact(as_given(lambda index, inputs, name=name: walked(name, inputs)), recorded, drawn,
                            f"{name} grid", seeds=GRID_SEEDS[name], truth_fn=truth)
            errors[name] = {}
        except AssertionError as failed:
            errors[name] = {"failed": str(failed)}
    print(json.dumps({"grids": errors}))
    return not any("failed" in error for error in errors.values())


HANDOFF_SEEDS = 8
"""The SDXL handoff's declared seeds for check 1, sized by `seeds_for` paired with Dew's walk on armada."""


def handoff(directory, source, task, key):
    """SDXL's base walks the steps at or above the cutoff and stops
    (final_denoise=False, decode=False); the refiner continues from those
    latents (`initial=`) below it, as Diffusers' denoising_end and
    denoising_start split them: the base's latents, the refiner's and their
    decoded images on par with Diffusers' own base and refiner pipelines
    doing so (tools/diffusers_handoff_reference.py's `handed`, live) over
    drawn noise (check 1)."""
    from reference_error import as_given, assert_as_exact

    from tools.diffusers_handoff_reference import handed

    root = Path(directory).parent
    refiner = bundle(root / "refiner")
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
    base_given, base_null, process, prefix = start(base_task, "prefix")
    refiner_task = replace(refiner.text_to_image(), final_denoise=False)
    given, null, _, tail = start(refiner_task, "tail")

    def dew(noise):
        initial = jnp.asarray(noise) * process.sampler_schedule.prior_scale()
        prepared = base_task.prepare(conditions=base_given, unconditional=base_null, key=key, steps=steps,
                                     initial=initial, times=prefix)
        latents = base_task(prepared, steps=steps, guidance=3.0, key=key, decode=False).host().latents
        prepared = refiner_task.prepare(conditions=given, unconditional=null, key=key, steps=steps,
                                        initial=latents, times=tail)
        final = refiner_task(prepared, steps=steps, guidance=3.0, key=key, decode=False).host().latents
        images = refiner.autoencoder.decode(refiner.variables["autoencoder"], jnp.asarray(final))
        return {"prefix": np.asarray(latents), "final": np.asarray(final), "images": np.asarray(images)}

    def draw(seed):
        return (np.random.default_rng([0x4A0, seed]).standard_normal((1, 8, 8, 4)).astype(np.float32),)

    assert_as_exact(as_given(dew), as_given(lambda noise: handed(root, noise, wide=False)), draw, "handoff",
                    seeds=HANDOFF_SEEDS, truth_fn=as_given(lambda noise: handed(root, noise, wide=True)))


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
"""The geometry cases' UNet timestep: any fixed one, the same in every arm."""
MODEL_ROWS = 4
"""A draw's independent rows: the pooled embedding's gradient holds 8 values a row, too few in one row for
its rounding to resolve over draws."""
MODEL_SEEDS = {"sd1": 8, "sd2": 8, "sdxl-inpaint": 16, "refiner": 256}
"""Each geometry case's declared seeds for check 1 (`assert_as_exact`), sized by `seeds_for` paired with
Dew's on armada (11-06); the reference compared with itself needed 16, 8, 256 and 256."""


def model_site(directory):
    """A multilevel UNet geometry's check-1 arms: Dew's UNet and Diffusers' own Flax UNet on the same
    weights (tools/diffusers_consumer.py's `flax_unet`, at float64 for the truth), each one's prediction and
    the gradients of its drawn inputs (the latents, the prompt states, the pooled embedding where the
    geometry takes SDXL's added conditioning, tools/native_unet_reference.py's TIME_IDS) under a drawn
    cotangent. Returns `(dew, reference, truth, draw)`."""
    from native_unet_reference import POOLED, TIME_IDS
    from reference_error import with_gradients

    from dew.diffusion.process import DenoisingCondition
    from dew.interop.diffusion import component_tensors, translate_unet_weights, unet_fields
    from dew.nn.backbones.unet_condition import UNet2DCondition
    from tools.diffusers_consumer import flax_unet

    directory = Path(directory)
    config = json.loads((directory / "unet/config.json").read_text())
    model = UNet2DCondition(**unet_fields(config, attention_impl="xla"))
    parameters, _ = translate_unet_weights(component_tensors(directory, "unet"), model)
    ids = TIME_IDS.get(directory.name)
    time_ids = None if ids is None else jnp.asarray([ids] * MODEL_ROWS, jnp.int32)
    shapes = [(MODEL_ROWS, 8, 12, config["in_channels"]), (MODEL_ROWS, 8, config["cross_attention_dim"]),
              *(() if ids is None else ((MODEL_ROWS, POOLED),))]

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

    def diffusers(wide):
        arm = flax_unet(directory, wide=wide)

        def run(latents, context, *pooled):
            # The truth's time ids in float64, as its widened projections read them.
            added = () if time_ids is None else (pooled[0], time_ids.astype(latents.dtype))
            return arm(latents, jnp.full((MODEL_ROWS,), MODEL_TIME, latents.dtype), context, *added)
        return run

    return with_gradients(dew), with_gradients(diffusers(False)), with_gradients(diffusers(True)), draw


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
    parser.add_argument("--grids", action="store_true", help="the source schedulers' grids, on the SD pipeline")
    parser.add_argument("--case")
    parser.add_argument("--model", action="store_true")
    parser.add_argument("--streamed", action="store_true", help="load the pipeline streamed onto a mesh")
    parser.add_argument("--part", choices=PARTS, help="the pipeline check's part; both where none is named")
    args = parser.parse_args()
    if args.model:
        check_model(args.directory)
    elif args.grids:
        sys.exit(0 if check_grids(args.directory, args.case) else 1)
    elif args.case:
        regress(args.directory, args.case)
    else:
        for part in PARTS if args.part is None else (args.part,):
            check_pipeline(args.directory, part, args.streamed)
