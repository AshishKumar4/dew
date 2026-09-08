"""Focused native-image repair oracles: Diffusers 0.34.0 / Transformers 4.49.0.

Run in the isolated reference environment on CPU:
  python tools/native_diffusion_edges_reference.py generate OUTPUT
  python tools/native_diffusion_edges_reference.py reload SOURCE EXPORTED
No model or source downloads. The reload command executes actual source pipelines.
"""
import argparse
import json
from pathlib import Path

import diffusers
import numpy as np
import torch
import transformers
from diffusers import DDIMScheduler, PNDMScheduler, EulerDiscreteScheduler, DPMSolverMultistepScheduler
from diffusers import DiffusionPipeline, UNet2DConditionModel


def check_versions():
    if (diffusers.__version__, transformers.__version__) != ("0.34.0", "4.49.0"):
        raise RuntimeError("Use the pinned isolated reference environment")
    torch.set_num_threads(2)


def generate(root):
    check_versions()
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    common = dict(num_train_timesteps=20, beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear")
    arrays = {}
    for name, cls, options in (
        ("ddim-linspace", DDIMScheduler, dict(timestep_spacing="linspace", clip_sample=False, set_alpha_to_one=False)),
        ("pndm-linspace", PNDMScheduler, dict(timestep_spacing="linspace", skip_prk_steps=True, set_alpha_to_one=False)),
        ("ddim-clip", DDIMScheduler, dict(clip_sample=True, set_alpha_to_one=False)),
        ("euler-zero-snr", EulerDiscreteScheduler, dict(rescale_betas_zero_snr=True, prediction_type="v_prediction")),
        ("dpm-karras", DPMSolverMultistepScheduler, dict(use_karras_sigmas=True)),
    ):
        scheduler = cls(**common, **options)
        scheduler.set_timesteps(4)
        scheduler.save_pretrained(root / name)
        config = json.loads((root / name / "scheduler_config.json").read_text())
        arrays[name + ".config"] = np.asarray(json.dumps(config))
        arrays[name + ".betas"] = scheduler.betas.numpy()
        arrays[name + ".times"] = scheduler.timesteps.numpy()
        if hasattr(scheduler, "sigmas"):
            arrays[name + ".sigmas"] = scheduler.sigmas.numpy()
        x = torch.tensor([[10.0, -8.0, 0.3]], dtype=torch.float32) * scheduler.init_noise_sigma
        trajectory = []
        for time in scheduler.timesteps:
            model_input = scheduler.scale_model_input(x, time)
            prediction = torch.sin(model_input) * 0.07 + time.to(torch.float32) * 0.001
            x = scheduler.step(prediction, time, x).prev_sample
            trajectory.append(x.numpy().copy())
        arrays[name + ".trajectory"] = np.stack(trajectory)
        arrays[name + ".final"] = x.numpy()
    np.savez_compressed(root / "schedulers.npz", **arrays)
    torch.manual_seed(173)
    model = UNet2DConditionModel(sample_size=8, in_channels=4, out_channels=4,
        down_block_types=("CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"),
        block_out_channels=(32, 32, 64), layers_per_block=1, cross_attention_dim=8,
        attention_head_dim=(4, 4, 8), norm_num_groups=8, norm_eps=0.001).eval()
    model.save_pretrained(root / "unet", safe_serialization=True)
    arrays = {}
    for name, width in (("norm", 12), ("odd", 10)):
        value = torch.randn((1, 4, 8, width), generator=torch.Generator().manual_seed(7), requires_grad=True)
        context = torch.randn((1, 8, 8), generator=torch.Generator().manual_seed(8))
        time = torch.tensor([13], dtype=torch.float32)
        output = model(value, time, encoder_hidden_states=context).sample
        probe = torch.randn(output.shape, generator=torch.Generator().manual_seed(9))
        gradient, = torch.autograd.grad((output * probe).sum(), value)
        arrays[name + ".input"] = value.detach().numpy().transpose(0, 2, 3, 1)
        arrays[name + ".context"] = context.numpy()
        arrays[name + ".time"] = time.numpy()
        arrays[name + ".output"] = output.detach().numpy().transpose(0, 2, 3, 1)
        arrays[name + ".probe"] = probe.numpy().transpose(0, 2, 3, 1)
        arrays[name + ".vjp"] = gradient.numpy().transpose(0, 2, 3, 1)
    np.savez_compressed(root / "unet_reference.npz", **arrays)
    print(json.dumps({"directory": str(root), "scheduler_cases": 5, "unet_cases": 2}))


def reload_source(source, exported):
    check_versions()
    index = json.loads((Path(source) / "model_index.json").read_text())
    if index["_class_name"].startswith("Flax"):
        import jax
        import jax.numpy as jnp
        factory = getattr(diffusers, index["_class_name"])
        options = {name: None for name in ("safety_checker", "feature_extractor")
                   if name in index and index[name][0] is None}
        before, original_params = factory.from_pretrained(source, local_files_only=True, **options)
        after, loaded_params = factory.from_pretrained(exported, local_files_only=True, **options)
        original_leaves, original_tree = jax.tree.flatten(original_params)
        loaded_leaves, loaded_tree = jax.tree.flatten(loaded_params)
        if original_tree != loaded_tree:
            raise AssertionError("Source Flax parameter structure changed")
        for left, right in zip(original_leaves, loaded_leaves):
            np.testing.assert_array_equal(left, right)
        key = jax.random.PRNGKey(41)
        noise = jax.random.normal(key, (1, 4, 8, 8), dtype=jnp.float32)
        kwargs = dict(prng_seed=key, num_inference_steps=2, guidance_scale=3.0, height=32, width=32, latents=noise)
        left = before(before.prepare_inputs(["cat"]), params=original_params,
                      neg_prompt_ids=before.prepare_inputs(["dog"]), **kwargs).images
        right = after(after.prepare_inputs(["cat"]), params=loaded_params,
                      neg_prompt_ids=after.prepare_inputs(["dog"]), **kwargs).images
        np.testing.assert_array_equal(left, right)
        print(json.dumps({"source": "Flax", "source_reload_tensors": len(original_leaves), "image_error": 0.0}))
        return
    before = DiffusionPipeline.from_pretrained(source, local_files_only=True, low_cpu_mem_usage=False)
    after = DiffusionPipeline.from_pretrained(exported, local_files_only=True, low_cpu_mem_usage=False)
    before.set_progress_bar_config(disable=True)
    after.set_progress_bar_config(disable=True)
    compared = 0
    for component in ("unet", "vae", "text_encoder", "text_encoder_2", "safety_checker"):
        first, second = getattr(before, component, None), getattr(after, component, None)
        if first is None:
            if second is not None:
                raise AssertionError(f"Unexpected component {component}")
            continue
        left, right = first.state_dict(), second.state_dict()
        if left.keys() != right.keys():
            raise AssertionError(f"Tensor keys differ for {component}")
        for name in left:
            torch.testing.assert_close(left[name], right[name], atol=0, rtol=0)
            compared += 1
    kwargs = dict(prompt="cat", negative_prompt="dog", num_inference_steps=2,
                  guidance_scale=3.0, height=32, width=32, output_type="np")
    left = before(generator=torch.Generator().manual_seed(41), **kwargs).images
    right = after(generator=torch.Generator().manual_seed(41), **kwargs).images
    np.testing.assert_array_equal(left, right)
    print(json.dumps({"source_reload_tensors": compared, "image_error": float(np.max(np.abs(left - right)))}))


def bundle(directory, destination):
    """Package the focused numeric oracles and tiny source UNet."""
    import tarfile
    root = Path(directory)
    with tarfile.open(destination, "w:xz") as archive:
        for path in (root / "schedulers.npz", root / "unet_reference.npz", root / "unet"):
            archive.add(path, arcname=path.name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    make = sub.add_parser("generate")
    make.add_argument("directory")
    reload = sub.add_parser("reload")
    reload.add_argument("source")
    reload.add_argument("exported")
    args = parser.parse_args()
    if args.command == "generate":
        generate(args.directory)
    else:
        reload_source(args.source, args.exported)
