"""Current-runtime checks for tools/diffusers_extended_reference.py fixtures."""
import argparse
import json
from pathlib import Path
import tempfile

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image

from dew.inference import TextToImage
from dew.interop.diffusers import load_diffusers_pipeline


def check(directory):
    directory = Path(directory)
    meta = json.loads((directory / "reference.json").read_text())
    ref = np.load(directory / "reference.npz")
    pipe = load_diffusers_pipeline(str(directory), from_pt=meta["from_pt"], local_files_only=True)
    errors = {}

    def compare(name, actual, expected, tolerance=5e-5):
        actual, expected = np.asarray(actual), np.asarray(expected)
        errors[name] = float(np.max(np.abs(actual.astype(np.float32) - expected.astype(np.float32))))
        np.testing.assert_allclose(actual, expected, atol=tolerance, rtol=tolerance)

    kwargs = dict(steps=meta.get("steps", 2), guidance=3.0, latents=jnp.asarray(ref["noise"]),
                  key=jax.random.PRNGKey(17), negative_prompts="dog")
    if meta["task"] == "safety":
        checker = pipe.safety_checker
        params = pipe.params["encoders"]["safety_checker"]
        compare("checker_features", checker.features(params, ref["checker_pixels"]), ref["checker_embeddings"])
        compare("checker_flags", checker(params, ref["checker_pixels"]), ref["checker_flags"], tolerance=0)
    else:
        kwargs.update(prompt_2="fox", negative_prompt_2="owl", strength=meta["strength"])
        given, null = pipe.conditions("cat", "dog", prompt_2="fox", negative_prompt_2="owl")
        compare("hidden", given["conditioning"]["encoder_hidden_states"], ref["hidden"])
        compare("pooled", given["conditioning"]["added_cond_kwargs"]["text_embeds"], ref["pooled"])
        compare("negative_hidden", null["conditioning"]["encoder_hidden_states"], ref["negative_hidden"])
        compare("negative_pooled", null["conditioning"]["added_cond_kwargs"]["text_embeds"], ref["negative_pooled"])
        processed, _ = pipe.prepare_image(Image.fromarray(ref["pixels"]))
        compare("preprocessing", processed.transpose(0, 3, 1, 2), ref["processed"], tolerance=0)
        if pipe.task == "inpainting":
            kwargs.update(image=Image.fromarray(ref["pixels"]), mask=Image.fromarray(ref["mask"]),
                          masked_image_latents=jnp.asarray(ref["masked_latents"]))
        else:
            kwargs["image"] = jnp.asarray(ref["image_latents"])
            if "denoising_start" in meta:
                kwargs["denoising_start"] = meta["denoising_start"]
        compare("trajectory_latents", pipe("cat", output_type="latent", **kwargs), ref["final_latents"])
    images = pipe("cat", **kwargs)
    # fp32 UNet/scheduler differences are amplified by the decoder; the
    # refiner observed 6.1e-5 image error, versus 2.4e-6 decoding equal latents.
    compare("images", (images + 1) / 2, ref["images"], tolerance=1e-4)
    with tempfile.TemporaryDirectory(prefix="dew-extended-reload-") as saved:
        pipe.save_pretrained(saved)
        restored = load_diffusers_pipeline(saved, local_files_only=True)
        compare("reload", restored("cat", **kwargs), images, tolerance=0)
        if meta["task"] == "safety":
            params = {**restored.params["encoders"]["safety_checker"], "concept_embeds_weights": jnp.full((17,), -2.0)}
            variables = {**restored.params, "encoders": {**restored.params["encoders"], "safety_checker": params}}
            blocked = restored.bind(variables)
            objective = blocked.objective(steps=2)
            counterpart = TextToImage.from_objective(objective, objective.init(jax.random.PRNGKey(17)))
            np.testing.assert_array_equal(counterpart("cat", **kwargs), -1)
            np.testing.assert_array_equal(blocked("cat", **kwargs), -1)
            blocked.save_pretrained(saved)
            np.testing.assert_array_equal(load_diffusers_pipeline(saved)("cat", **kwargs), -1)
    print(json.dumps({"task": meta["task"], "errors": errors, "backend": jax.default_backend()}))
    return errors


def regress(directory, case):
    pipe = load_diffusers_pipeline(directory, local_files_only=True)
    key = jax.random.PRNGKey(17)
    options = dict(key=key, steps=2, guidance=3.0)
    if case in ("pndm-prk", "pndm-plms", "lms", "lms-v", "lms-karras", "euler"):
        oracle = np.load(Path(__file__).resolve().parents[1] / "tests/fixtures/diffusers_pipeline_schedulers.npz")
        with tempfile.TemporaryDirectory(prefix="dew-source-grid-") as saved:
            pipe.save_pretrained(saved)
            config = json.loads(str(oracle[case + ".config"]))
            (Path(saved) / "scheduler/scheduler_config.json").write_text(json.dumps(config))
            index = json.loads((Path(saved) / "model_index.json").read_text())
            index["scheduler"] = ["diffusers", config["_class_name"]]
            (Path(saved) / "model_index.json").write_text(json.dumps(index))
            source = load_diffusers_pipeline(saved)
            actual = source("cat", negative_prompts="dog", latents=jnp.asarray(oracle["noise"]),
                            key=key, steps=4, guidance=3.0, output_type="latent")
            expected = oracle[case + ".latents"]
            np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=1e-4)
            print(case, "full-grid latent error", float(np.max(np.abs(actual - expected))))
        return
    if case == "prepared":
        prepared = pipe.prepare("cat", negative_prompts="dog", key=key)
        np.testing.assert_array_equal(pipe(prepared, **options), pipe("cat", negative_prompts="dog", **options))
    elif case == "geometry":
        pipe = load_diffusers_pipeline(directory, height=64, width=32)
        expected = pipe("cat", **options)
        with tempfile.TemporaryDirectory(prefix="dew-geometry-") as saved:
            pipe.save_pretrained(saved)
            restored = load_diffusers_pipeline(saved)
            np.testing.assert_array_equal(restored("cat", **options), expected)
    elif case == "negative-policy":
        expected = pipe("cat", negative_prompts="", **options)
        with tempfile.TemporaryDirectory(prefix="dew-negatives-") as saved:
            pipe.save_pretrained(saved)
            index = json.loads((Path(saved) / "model_index.json").read_text())
            index["force_zeros_for_empty_prompt"] = False
            (Path(saved) / "model_index.json").write_text(json.dumps(index))
            restored = load_diffusers_pipeline(saved)
            np.testing.assert_array_equal(restored("cat", **options), expected)
            assert not np.allclose(pipe("cat", **options), expected, atol=1e-7)
    elif case == "mask-dropout":
        from dew.objectives.base import Step
        objective = pipe.objective(unconditional_prob=1.0, ema_decay=None, steps=2)
        reference = np.load(Path(directory) / "reference.npz")
        batch = {"image": reference["pixels"][None], **pipe.inputs.tokenize(["cat"]),
                 "mask": (reference["mask"][None, ..., None] > 127).astype(np.float32)}
        other = {**batch, "mask": 1 - batch["mask"]}
        step = Step(jnp.asarray(0), key, None)
        def loss(params, data):
            value, _ = objective.loss({**pipe.params, "params": params}, data, step)
            return value.total / value.mass
        gradient = jax.jit(jax.grad(loss))
        first = gradient(pipe.params["params"], batch)
        second = gradient(pipe.params["params"], other)
        difference = max(float(jnp.max(jnp.abs(a - b))) for a, b in zip(jax.tree.leaves(first), jax.tree.leaves(second)))
        assert difference > 1e-5, difference
        sample = objective.evaluate(pipe.params, batch, step)
        preview = objective.preview(pipe.params, batch, step)
        np.testing.assert_array_equal(preview.images, sample.images)
        assert not np.allclose(objective.evaluate(pipe.params, other, step).images, sample.images)
        print("mask-conditioned gradient difference", difference)
    print(case, "passed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    parser.add_argument("--case")
    args = parser.parse_args()
    if args.case:
        regress(args.directory, args.case)
    else:
        check(args.directory)

