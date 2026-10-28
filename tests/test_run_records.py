"""A run config's record reads back as the config that wrote it.

`to_dict` writes every recorded field, `from_dict` builds the same value
back, a field the record lacks takes its declared default, and a field the
class does not declare is refused.
"""

import dataclasses
import json
from pathlib import Path

import pytest

from dew.config import ObjectiveConfig, RunConfig
from dew.config.diffusion import DiffusionRunConfig
from dew.config.jepa import JepaRunConfig
from dew.config.lm import LMRunConfig
from dew.decision.config import DecisionRunConfig
from dew.quantization import Quantization

ROOT = Path(__file__).resolve().parents[1]


def written(config) -> dict:
    """The record `save` writes, as the JSON file holds it."""
    return json.loads(json.dumps(config.to_dict()))


@pytest.mark.parametrize("config", ["run", "diffusion", "lm", "jepa", "decision"])
def test_a_run_config_reads_back_as_its_class_from_its_own_record(config):
    """`run.json` names the run's class, so a reader that knows only
    `RunConfig` rebuilds a run of any kind as the kind it is."""
    value = {"run": RunConfig(), "diffusion": DiffusionRunConfig(), "lm": LMRunConfig(),
             "jepa": JepaRunConfig(),
             "decision": DecisionRunConfig()}[config]
    assert RunConfig.read(json.loads(json.dumps(value.record()))) == value


def test_a_record_keeps_what_it_states_and_defaults_what_it_lacks():
    run = DiffusionRunConfig()
    quantized = dataclasses.replace(run, trainer=dataclasses.replace(
        run.trainer, quantization=Quantization(dtype="fp8", patterns=(".*mlp.*",), weight_only=True)))
    assert DiffusionRunConfig.from_dict(written(quantized)) == quantized
    stated = LMRunConfig(objective=ObjectiveConfig("lm", {"ema_decay": 0.99}))
    assert LMRunConfig.from_dict({"objective": {"name": "lm", "fields": {"ema_decay": 0.99}}}) == stated
    assert LMRunConfig.from_dict({}).objective == ObjectiveConfig("lm")


PUBLISHED = ROOT / "tests" / "fixtures" / "runs" / "hybrid-dit-176m" / "run.json"
LIVE_PIN = ROOT / "site" / "live" / "container" / "text-to-image"


def changed(written, held, at: str = "") -> list[str]:
    """Where `written` does not keep what `held` states: a key it lacks, a
    list of another length, or another value. What `held` does not state is
    no change."""
    if isinstance(held, dict) and isinstance(written, dict):
        return [place for key, value in held.items() for place in
                (changed(written[key], value, f"{at}/{key}") if key in written else [f"{at}/{key}"])]
    if isinstance(held, list) and isinstance(written, list) and len(written) == len(held):
        return [place for index, pair in enumerate(zip(written, held, strict=True))
                for place in changed(*pair, f"{at}/{index}")]
    return [] if written == held else [at]


@pytest.mark.parametrize("live", [False, pytest.param(True, marks=pytest.mark.network)])
def test_the_published_run_reads_back_as_it_was_written(live):
    """dewml/hybrid-dit-176m's run.json, as the revision the live image pins
    publishes it, is a current record: every field it states loads and writes
    back as it states it. A field added since is written beside them; a
    rename, a removal or a rewrite fails here, and the fix is to re-export the
    published run in the same change, since the site, the quick start and the
    live sampler all load it."""
    held = json.loads(PUBLISHED.read_text())
    if live:
        from huggingface_hub import hf_hub_download

        repo, revision = LIVE_PIN.read_text().strip().split("@")
        published = Path(hf_hub_download(repo, "run.json", revision=revision))
        assert json.loads(published.read_text()) == held
    assert changed(json.loads(json.dumps(RunConfig.read(held).record())), held) == []
    stated = held["fields"]["objective"]["fields"]
    built = RunConfig.read(held).objective.arguments
    assert (built["guidance"], built["steps"]) == (stated["guidance"], stated["steps"])


LAYOUT = ROOT / "tests" / "fixtures" / "runs" / "hybrid-dit-176m" / "layout.json"


def published_layout() -> tuple[dict, dict]:
    """The published run's checkpoint record and its variables as abstract
    arrays, nested as the checkpoint stores them (`layout.json`)."""
    import jax
    from flax.traverse_util import unflatten_dict

    from dew.inference.tasks import earlier_shapes

    fixture = json.loads(LAYOUT.read_text())
    variables = unflatten_dict({path: jax.ShapeDtypeStruct(tuple(shape), dtype)
                                for path, (shape, dtype) in fixture["variables"].items()}, sep="/")
    # The published run's record, as a run loader reads it (`SavedRun.pinned`).
    return earlier_shapes(fixture["artifact"]), variables


def test_the_published_runs_weights_are_the_ones_its_models_read():
    """dewml/hybrid-dit-176m's checkpoint layout (layout.json: each array's
    path, shape and dtype, and the model record beside them) is what the
    run's denoiser and autoencoder read: both trace over arrays of that
    layout. A parameter renamed, moved or reshaped in either fails here, and
    the fix is to republish the run in the same change, since the site, the
    quick start and the live sampler all load it."""
    import jax
    import jax.numpy as jnp

    from dew.inference.tasks import saved_model
    from dew.nn.autoencoders import AutoEncoder
    from dew.nn.dit import TextContext

    artifact, variables = published_layout()
    model = saved_model(artifact, None).build()
    autoencoder = AutoEncoder.bound(artifact["autoencoder"], params=variables["autoencoder"])
    sample = tuple(artifact["inputs"]["sample"]["shape"])
    latents = jax.ShapeDtypeStruct((1, *autoencoder.latent_shape(sample)), jnp.float32)
    # CLIP ViT-L/14's 77 tokens of 768 features, the published run's text condition.
    text = TextContext(jax.ShapeDtypeStruct((1, 77, 768), jnp.float32),
                       jax.ShapeDtypeStruct((1, 77), jnp.float32))
    denoiser = {name: tree for name, tree in variables.items() if name not in ("encoders", "autoencoder")}
    predicted = jax.eval_shape(lambda held, x, t, text: model.apply(held, x, t, textcontext=text),
                               denoiser, latents, jax.ShapeDtypeStruct((1,), jnp.float32), text)
    assert predicted.shape == latents.shape
    assert jax.eval_shape(autoencoder.decode, variables["autoencoder"], latents).shape == (1, *sample)


@pytest.mark.network
def test_the_live_run_holds_the_committed_layout_and_samples_an_image():
    """The revision the live image pins holds exactly the committed layout,
    and loads and samples an image, its text tower included, on one device
    of the lane's backend: the CPU unless the run sets JAX_PLATFORMS, as the
    live sampler runs it. Loaded over the lane's eight host devices, the
    weights would be held once per device, 10.9 GiB against 3.3."""
    import jax
    import numpy as np
    from flax.core import unfreeze

    import dew
    from dew.inference import TextToImage
    from dew.training import MeshSpec

    repo, revision = LIVE_PIN.read_text().strip().split("@")
    pipe = dew.pipeline(repo, revision=revision, mesh=MeshSpec().build(jax.devices()[:1]))
    assert isinstance(pipe, TextToImage)
    artifact, expected = published_layout()
    held = unfreeze(pipe.variables)
    assert jax.tree.structure(held) == jax.tree.structure(expected)
    def same(held, wanted) -> bool:
        return (held.shape, held.dtype) == (wanted.shape, wanted.dtype)

    assert jax.tree.all(jax.tree.map(same, held, expected))
    images = pipe(["a watercolor painting of a mountain lake at sunrise"], steps=1, key=0).host().images
    assert images.shape == (1, *artifact["inputs"]["sample"]["shape"])
    assert np.all(np.isfinite(np.asarray(images, np.float32)))
