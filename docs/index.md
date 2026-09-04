# dew

Dew is a framework for building and training modern architectures in JAX and Flax, at any scale from one CPU to a TPU pod. It trains image and video diffusion, flow matching, latent diffusion with a VAE, I-JEPA and V-JEPA encoders, autoregressive language models and masked diffusion language models, all through one trainer with data-parallel, fully sharded and expert-parallel layouts.

Dew is the successor to FlaxDiff, restructured once the trainer stopped being about diffusion alone. What to learn is an objective you plug in; how to learn it is the trainer's, and the same trainer serves every objective.

## Install

```bash
pip install "dew-ml @ git+https://github.com/AshishKumar4/dew"
```

There is no release on PyPI yet. The package will ship as `dew-ml` and imports as `dew`. Extras pull in the heavier dependencies only when you need them: `av` for video and audio readers, `metrics` for FID, `streaming` for the URL-streaming loader, `tfds` for TFDS datasets, `interop` for safetensors.

## A training run

```python
# runs elsewhere: downloads Oxford Flowers and the CLIP text tower, and trains for real
import jax, optax
import dew
from dew import Checkpoints, Condition, Field, InputSpec, MeshSpec, Trainer, models, presets
from dew.data import OxfordFlowers
from dew.inputs import CLIPText
from dew.objectives.diffusion import DiffusionObjective
from dew.sampling import Heun, CFG

data = OxfordFlowers(image_size=128).load(batch=16)
text = CLIPText.from_pretrained("openai/clip-vit-large-patch14")
inputs = InputSpec(Field("image", (128, 128, 3)), {"textcontext": Condition(text)})

model = models.SimpleDiT(emb_features=512, num_layers=8, num_heads=8, patch_size=8, dtype="bfloat16")
objective = DiffusionObjective(model, presets.EDM()(), inputs, sampler=Heun(), guidance=CFG(3.0), steps=40)

trainer = Trainer(objective, optax.adamw(2e-4), key=jax.random.key(4), mesh=MeshSpec(fsdp=1),
                  checkpoints=Checkpoints("runs/flowers-edm"))
state = trainer.fit(data, steps=100 * data.steps_per_epoch, eval_every=data.steps_per_epoch)
```

Every noun in that run is one of the API's: a registered model, a preset that builds a `Process`, an `InputSpec` of a sample and its conditions, an `Objective`, a `Dataset`, a `Trainer` given its capabilities. `import dew` fills the registries, so `models.SimpleDiT`, `presets.EDM` and `datasets["oxford_flowers102"]` resolve with nothing else imported, and an unknown name or field raises rather than being guessed. For a run described by flags instead of by code, see [Recipes](recipes.md).

## Where things are

- [The objectives seam](concepts/objectives.md): what the trainer owns, what the objective owns, and how to add one.
- [Distributed training](concepts/distributed.md): the mesh, the sharding declarations, prefetch, checkpoints, the pod.
- [The data pipeline](concepts/data.md): dataset specs, the `Dataset` value, determinism, resume.
- [Language models](concepts/language_models.md): the decoder, the data path, generation, the masked diffusion variant.
- [Mixture of experts](concepts/moe.md): the router, the expert axis, parity.
- [API](api.md): every public module and what it exports, written from the code.
- [The design](design/api.md): the twelve decisions the API follows and why.

## Tests

```bash
pip install -e .[test]
JAX_PLATFORMS=cpu pytest -m "mesh and not distributed" -n 3 --dist loadfile -q     # the mesh lane
JAX_PLATFORMS=cuda pytest -m "not mesh and not network" -n 4 --dist loadfile -q    # the GPU lane
JAX_PLATFORMS=cpu pytest -m distributed -q                                         # real process pools
```

`tests/conftest.py` asks XLA for eight host devices, so the sharding, checkpoint and resume tests run a real data-by-fsdp mesh without an accelerator; that is the `mesh` lane. Everything else runs on whatever accelerator `JAX_PLATFORMS` names, in parallel workers. The `distributed` lane spawns real `jax.distributed` pools and runs serially. Tests marked `network` download pretrained weights and are excluded by default. The code in this documentation runs as a test too (`tests/test_docs_run.py`), and `docs/api.md` is regenerated from the modules' exports.
