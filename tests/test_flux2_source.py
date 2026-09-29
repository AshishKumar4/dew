"""Native FLUX.2 against the actual Diffusers 0.40.0 objects it reconstructs.

`tools/diffusers_flux2_reference.py` builds tiny `Flux2Transformer2DModel`
instances with every parameter moved off its initialization, saves each with
its real config and safetensors, and runs each as `Flux2Pipeline` runs it -
one token per latent position, the four-axis ids the pipeline lays out, the
timestep it has divided by the training count and the distilled guidance -
recording the forward and the gradients of the latent, the text states and
every parameter against a fixed cotangent.

The pipeline flattens its latent row-major, one token per position, so these
tests reshape between that and NHWC with numpy's own reshape.
"""

import json
import tarfile
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dew.diffusion.process import DenoisingCondition
from dew.interop.diffusion import component_tensors, flux2_fields, translate_flux2_weights
from dew.nn.backbones.flux2 import Flux2Transformer

ROOT = Path(__file__).resolve().parents[1]
CASES = ("dev", "rect", "unguided", "narrow")
FORWARD = 1e-5
GRADIENT = 1e-4


@pytest.fixture(scope="module")
def source(tmp_path_factory):
    directory = tmp_path_factory.mktemp("flux2-source")
    with tarfile.open(ROOT / "tests/fixtures/flux2_source.tar.xz") as archive:
        archive.extractall(directory, filter="data")
    return directory


@pytest.fixture(scope="module")
def record(source):
    return json.loads((source / "flux2_transformer.json").read_text())


@pytest.fixture(scope="module")
def arrays(source):
    with np.load(source / "flux2_transformer.npz") as loaded:
        return dict(loaded)


def relative_gap(actual, expected) -> float:
    actual, expected = np.asarray(actual, np.float64), np.asarray(expected, np.float64)
    return float(np.abs(actual - expected).max() / max(1.0, float(np.abs(expected).max())))


def load(source, name):
    config = json.loads((source / name / "transformer" / "config.json").read_text())
    model = Flux2Transformer(**flux2_fields(config, dtype="float32", attention_impl="xla"))
    params, layouts = translate_flux2_weights(component_tensors(source / name, "transformer"))
    return model, params, layouts


def walk(name, arrays, record):
    rows, columns = record["cases"][name]["grid"]
    latent = arrays[f"{name}.latent"].reshape(-1, rows, columns, arrays[f"{name}.latent"].shape[-1])
    guidance = arrays[f"{name}.guidance"]
    condition = DenoisingCondition(jnp.asarray(arrays[f"{name}.context"]),
                                   guidance=jnp.asarray(guidance) if guidance.size else None)
    return jnp.asarray(latent), jnp.asarray(arrays[f"{name}.times"]), condition, (rows, columns)


@pytest.mark.parametrize("name", CASES)
def test_every_published_tensor_maps_and_exports_bit_identical(source, name):
    model, params, layouts = load(source, name)
    tensors = component_tensors(source / name, "transformer")
    assert {layout.name for layout in layouts} == {f"transformer/{key}" for key in tensors}
    for layout in layouts:
        written = layout.export({"params": params})
        assert written.tobytes() == tensors[layout.name.removeprefix("transformer/")].tobytes(), layout.name


@pytest.mark.parametrize("name", CASES)
def test_the_forward_and_every_gradient_match_the_source(source, arrays, record, name):
    model, params, layouts = load(source, name)
    latent, times, condition, (rows, columns) = walk(name, arrays, record)
    probe = arrays[f"{name}.probe"].reshape(latent.shape[0], rows, columns, -1)

    def objective(params, latent, context):
        conditioned = DenoisingCondition(context, guidance=condition.guidance)
        return jnp.sum(model.apply({"params": params}, latent, times, conditioned) * probe)

    output = model.apply({"params": params}, latent, times, condition)
    forward = relative_gap(output.reshape(arrays[f"{name}.output"].shape), arrays[f"{name}.output"])
    grad_params, grad_latent, grad_context = jax.grad(objective, argnums=(0, 1, 2))(
        params, latent, condition.context)
    gaps = {"latent": relative_gap(grad_latent.reshape(arrays[f"{name}.grad_latent"].shape),
                                   arrays[f"{name}.grad_latent"]),
            "context": relative_gap(grad_context, arrays[f"{name}.grad_context"])}
    gaps.update({layout.name: relative_gap(layout.export({"params": grad_params}),
                                           arrays[f"{name}.grad_param.{layout.name.removeprefix('transformer/')}"])
                 for layout in layouts})
    worst = max(gaps, key=gaps.get)
    print(f"{name}: forward {forward:.3g}; worst gradient {worst} {gaps[worst]:.3g}")
    assert forward < FORWARD
    assert gaps[worst] < GRADIENT, worst


def test_a_guided_checkpoint_refuses_a_call_without_guidance(source, arrays, record):
    model, params, _ = load(source, "dev")
    latent, times, condition, _ = walk("dev", arrays, record)
    with pytest.raises(ValueError, match="embeds its guidance"):
        model.apply({"params": params}, latent, times, DenoisingCondition(condition.context))


@pytest.mark.parametrize("change", [{"patch_size": 2}, {"axes_dims_rope": [8, 8]}, {"axes_dims_rope": [4, 4, 4, 3]}])
def test_an_unsupported_config_is_refused(source, change):
    config = json.loads((source / "dev" / "transformer" / "config.json").read_text())
    flux2_fields(config)
    with pytest.raises(ValueError):
        flux2_fields({**config, **change})
