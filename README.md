<div align="center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/banner-dark.svg">
  <img src="docs/assets/banner-light.svg" alt="dew" width="360">
</picture>

<h1>Dew: building and training modern architectures at scale, in JAX and Flax</h1>

<a href="https://github.com/AshishKumar4/dew/actions/workflows/ci.yml"><img src="https://github.com/AshishKumar4/dew/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
<a href="pyproject.toml"><img src="https://img.shields.io/badge/python-3.11%2B-3776AB" alt="Python 3.11+"></a>
<a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-2aa7a1" alt="MIT"></a>

<p>
<a href="#diffusion"><b>Diffusion</b></a>
| <a href="#jepa"><b>JEPA</b></a>
| <a href="#language-models"><b>Language models</b></a>
| <a href="#objectives"><b>Objectives</b></a>
| <a href="#scaling"><b>Scaling</b></a>
| <a href="#roadmap"><b>Roadmap</b></a>
| <a href="#installation"><b>Install guide</b></a>
| <a href="docs/index.md"><b>Documentation</b></a>
</p>
</div>

## What is Dew?

Dew is a framework for building and training modern machine learning architectures in JAX and Flax, with full support for distributed training and scaling. It trains diffusion and flow matching models, JEPA encoders and autoregressive language models today, and it is built to reach parity with the large open model families: the same trainer, data pipeline and sharding serve every architecture, and a new one is a Flax module plus a class with a loss.

Dew separates what you train from how you train it. An [objective](#objectives) defines the parameters, the loss and the validation step. The [trainer](#scaling) defines the device mesh, the compiled step, gradient accumulation, EMA, checkpoints and logging, and treats every objective the same way. The same code runs on a CPU, one GPU, a TPU pod slice or many hosts: data parallel and fully sharded parameters are one code path, chosen by one number.

The [models](#models), the [diffusion maths](#schedules-and-prediction-transforms), the [samplers](#sampling) and the [metrics](#evaluation-and-export) are plain Flax and JAX, and each can be used on its own. The [data pipeline](#data) is built on Grain and gives the same batches on any number of workers and hosts. Everything that claims to be fast is [measured](#testing-and-benchmarks).

Dew is the successor to [FlaxDiff](https://github.com/AshishKumar4/FlaxDiff). It is a personal research project, not a product. Expect sharp edges, and please [report](https://github.com/AshishKumar4/dew/issues) the ones you find.

```python
# runs elsewhere: downloads Oxford Flowers and trains for real
import jax, optax
import dew
from dew import Checkpoints, Field, InputSpec, MeshSpec, Trainer, models, presets
from dew.data import OxfordFlowers
from dew.objectives.diffusion import DiffusionObjective

data = OxfordFlowers(image_size=64).load(batch=32)
model = models.SimpleDiT(emb_features=256, num_layers=6, num_heads=4, patch_size=4, dtype="bfloat16")
objective = DiffusionObjective(model, presets.EDM()(), InputSpec(Field("image", (64, 64, 3))))

trainer = Trainer(objective, optax.adamw(3e-4), key=jax.random.key(0),
                  mesh=MeshSpec(fsdp=1), checkpoints=Checkpoints("runs/flowers"))
state = trainer.fit(data, steps=50 * data.steps_per_epoch)
# state.params, state.ema and state.opt_state, sharded over every device; checkpoints under runs/flowers
```

<div align="center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/architecture-dark.svg">
  <img src="docs/assets/architecture-light.svg" alt="Dew modules by layer" width="100%">
</picture>
</div>

### Contents
* [Diffusion](#diffusion)
* [JEPA](#jepa)
* [Language models](#language-models)
* [Objectives](#objectives)
* [Scaling](#scaling)
* [Data](#data)
* [Configuration](#configuration)
* [Checkpoints and resume](#checkpoints-and-resume)
* [Logging and profiling](#logging-and-profiling)
* [Extending Dew](#extending-dew)
* [Recipes and examples](#recipes-and-examples)
* [Evaluation and export](#evaluation-and-export)
* [Testing and benchmarks](#testing-and-benchmarks)
* [Installation](#installation)
* [Roadmap](#roadmap)
* [Citing Dew](#citing-dew)
* [Reference documentation](#reference-documentation)

## Diffusion

### Schedules and prediction transforms

A diffusion model is defined by three choices: the noise schedule for training, the noise schedule for sampling, and the prediction transform, which says what the network outputs. A `Process` holds the three, and a preset is a dataclass that builds one; its fields are what a run records, so a run's process can always be rebuilt:

```python
from dew import presets

process = presets.EDM()()                          # EDM training, Karras sampling
process = presets.Flow(shift=3.0)()                # rectified flow, as in Stable Diffusion 3
process = presets.Cosine(min_snr_gamma=5.0)()
process = presets.build("edm", sigma_data=0.5)()   # by name, for a config; an unknown field raises
```

The parts are available on their own. `dew.diffusion.schedules` has the linear, cosine, exp, sqrt, Karras VE, EDM and flow matching schedules; a schedule is a value with `rates`, `sample_t`, `weight` and `model_time`, and holds no random state. `dew.diffusion.transforms` has the epsilon, x0, v, flow and Karras transforms. `dew.diffusion.discrete` has the masking schedule, process and unmasking solver a masked diffusion language model trains and samples with. Every schedule is tested against the invariants of its paper: monotone SNR, exact forward and inverse diffusion, variance preservation. See [docs/api.md](docs/api.md#diffusion) for the list.

### Conditioning

`InputSpec` describes the sample and its conditions. A condition pairs an encoder with the batch field it reads, under the keyword the model takes it as:

```python
# runs elsewhere: downloads the CLIP text tower
from dew import Condition, Field, InputSpec
from dew.inputs import CLIPText

text = CLIPText.from_pretrained("openai/clip-vit-large-patch14")
inputs = InputSpec(sample=Field("image", (128, 128, 3)),
                   conditions={"textcontext": Condition(text, field="text", unconditional="")})
```

The objective drops the conditions on 12% of each batch (`unconditional_prob`), so the model also learns the unconditional distribution that classifier-free guidance needs. An encoder tokenizes on the host and encodes on device as a pure function of its own parameters, which the trainer places like any other, so a frozen text tower is an argument of the compiled step and never a constant baked into it. `dew.inputs` ships `CLIPText`, `CharTable` (a table lookup that costs nothing, for tests and benchmarks) and an `Audio` encoder that says what it needs before it can load a tower; a new one implements `tokenize`, `encode` and `to_json` and registers with `@encoders(name)`. The CLIP encoder is dew's own port of the text tower, in [`dew/nn/text_encoders.py`](src/dew/nn/text_encoders.py), and it reads the checkpoint's safetensors itself, since transformers 5 removed the Flax classes it used to load. Pass `autoencoder=StableDiffusionVAE()` to the objective to train in the latent space of the Stable Diffusion VAE instead of pixels, or `SimpleAutoEncoder` to train your own.

### Sampling

Sampling is one function. `process.denoiser` turns the model and its parameters into a denoiser, a solver takes one step of the trajectory, and `sample` runs the steps in one `lax.scan`. Guidance wraps the denoiser; `CFG(scale, interval)` limits it to a part of the trajectory:

```python
import jax
from dew import CFG, sample
from dew.sampling import Heun

weights = {**state.params, **state.ema}                 # the EMA weights, with the frozen encoders beside them
encode = lambda prompts: {"textcontext": text.encode(weights["encoders"]["textcontext"], text.tokenize(prompts))}
denoise = process.denoiser(model, weights, encode(["a water lily", "a rose"]), unconditional=encode(["", ""]))
x_T = process.noise(jax.random.key(1), (2, *inputs.sample.shape))
images = sample(denoise, x_T, steps=8, solver=Heun(), guidance=CFG(4.0, interval=(0.1, 0.9)), key=jax.random.key(2))
assert images.shape == (2, *inputs.sample.shape)   # values in [-1, 1]
```

`TextToImage` does the encoding and the decoding for you, from a live model or from a run directory (`TextToImage.from_run("runs/flowers")`), and the solvers `ddpm`, `ddim`, `euler`, `euler_ancestral`, `heun`, `rk4` and `multistep_dpm` are registered in `samplers`. Each is tested to converge on an analytic denoiser, and the two that integrate in sigma refuse a variance-preserving schedule rather than integrating it wrong. There is no default seed anywhere in sampling: every call takes a `key`.

### Models

Every architecture is a Flax module registered under a name. The registry is a mapping, an attribute view and a strict builder over one table, so `models["simple_dit"]`, `models.SimpleDiT` and the class are the same object, and a field the model does not have raises instead of being dropped:

```python
import dew
from dew import models

unet = models.Unet(emb_features=256, feature_depths=[64, 128, 256, 512])
dit = models.SimpleDiT(emb_features=512, num_layers=12, num_heads=8, patch_size=2, scan_order="hilbert")
mmdit = models.build("simple_mmdit", emb_features=512, num_layers=12, num_heads=8, patch_size=2,
                     dtype="bfloat16", attention_impl="auto")
assert models["simple_dit"] is models.SimpleDiT
```

| Architecture | What it is |
|---|---|
| `unet`, `unet_3d` | Convolutional UNets for images and video; the 3D one inflates 2D checkpoints |
| `uvit`, `simple_udit` | U-shaped transformers |
| `simple_dit`, `simple_mmdit`, `hierarchical_mmdit` | DiT, the SD3-style dual-stream MMDiT, and a multi-resolution MMDiT, on one shared adaLN-Zero block |
| `hybrid_dit` | S5 state space blocks between attention blocks |
| `video_dit` | Factorized spatial and temporal attention |
| `jepa_encoder`, `jepa_video_encoder`, `jepa_predictor` | The ViTs the JEPA objective trains |
| `causal_transformer` | The decoder for language models, in the Hugging Face layout |

`scan_order` (`raster`, `hilbert` or `zigzag`) is the order in which patches enter a transformer. Every model takes `dtype` and `attention_impl`, and the parameter tree does not depend on either, so a checkpoint trained with cuDNN attention on a GPU loads on a TPU. `dew.nn` holds the pieces they are made of: attention (one module over the reference, XLA, cuDNN and TPU kernels), blocks, the DiT and ViT stacks, the S5 mixer, the scan orders, and the autoencoders. See [docs/benchmarks.md](docs/benchmarks.md) for what each architecture costs per step.

## JEPA

`JepaObjective` trains an I-JEPA or V-JEPA encoder. The predictor reads the encoder's embeddings of the visible patches and predicts the embeddings of masked target blocks; the targets come from a target encoder that is an EMA of the context encoder. The objective logs the representation standard deviation and off-diagonal covariance on every step, which is how a collapsing run shows itself.

```python
import jax, optax
from dew import Checkpoints, Field, Trainer, metrics, models
from dew.objectives.jepa import JepaObjective, multi_block_mask

encoder = models.JepaEncoder(patch_size=4, emb_features=32, num_layers=2, num_heads=2)
predictor = models.JepaPredictor(grid=(4, 4), emb_features=32, predictor_features=16, num_layers=1, num_heads=2)
objective = JepaObjective(encoder, predictor, mask=multi_block_mask((4, 4), num_targets=1, scale=(0.2, 0.3)), sample=Field("image", (16, 16, 3)))

trainer = Trainer(objective, optax.adamw(1e-3), key=jax.random.key(0), checkpoints=Checkpoints("runs/ijepa"))
state = trainer.fit(data, steps=steps, metrics=(metrics.linear_probe(5), metrics.knn_probe(5)))
# validation logs val/linear_probe and val/knn_probe, from classifiers fit on the frozen embeddings
```

`multi_block_mask` resolves the I-JEPA mask geometry (number of targets, scale and aspect ranges) for a patch grid. `jepa_video_encoder` and a `factorized=True` predictor do the same for video. See [docs/concepts/objectives.md](docs/concepts/objectives.md) for more.

## Language models

`CausalTransformer` is a decoder with the parts current open models use: RMSNorm, grouped-query attention, rotary positions, a gated MLP, q/k normalisation, and optional sliding-window layers, embedding scaling and logit softcapping. Its parameter tree follows the Hugging Face decoder layout. Qwen and Gemma checkpoint translators and parity tests are in progress; a checkpoint is not supported until those tests pass. `LMObjective` is next-token cross entropy in fp32; at validation it reports perplexity and generates text.

```python
# runs elsewhere: reads a tokenized corpus from disk and trains for real
import jax, optax
from dew import Checkpoints, Trainer, metrics, models
from dew.data import TokenWindows
from dew.objectives.lm import LMObjective, Samples
from dew.sampling import generate

data = TokenWindows(path="data/shakespeare", seq_len=256).load(batch=64)
model = models.CausalTransformer(vocab_size=256, emb_features=384, num_layers=6, num_heads=6,
                                 max_seq_len=512, dtype="bfloat16", attention_impl="auto")
objective = LMObjective(model, seq_len=256, samples=Samples(prompt=list(b"ROMEO:"), max_new_tokens=300))
trainer = Trainer(objective, optax.adamw(1e-3), key=jax.random.key(0), checkpoints=Checkpoints("runs/shakespeare"))
state = trainer.fit(data, steps=1200, metrics=(metrics.perplexity(),))
# val/perplexity 391 -> 4.6 over 1200 steps, on an RTX 4080

tokens = generate(model, state.ema, prompt, max_new_tokens=300, key=jax.random.key(0), temperature=0.8, top_k=40)
# prompt is int32 [1, 6] for b"ROMEO:"; tokens.shape == (1, 306)
```

`generate` prefills the KV cache on the prompt and decodes in one `lax.scan`: 0.9 ms per token for a 12-layer, 512-wide model on an RTX 4080. `tools/tokenize_text.py` writes the token files with a byte-level or any Hugging Face tokenizer, and `TokenFileSource` reads them by memory map, so a corpus is never held in Python. Each decoder block has a `mixer` slot, which is where linear attention and other token mixers go. See [docs/concepts/language_models.md](docs/concepts/language_models.md) for more.

## Objectives

An objective has three methods and three attributes. `init(key)` builds the parameter tree, which can hold several modules. `loss(params, batch, step)` returns a scalar and an `Aux` of metrics, plus any non-parameter collections to write back. `evaluate(params, batch, step)` returns a typed artifact, an `ImageGrid`, `TextSamples`, `TokenScores` or `Representations`, which a metric scores and a tracker renders. `inputs` says what a batch looks like, `ema` which part of the tree gets an exponential moving average, and `step` carries the step number, the step's key and the EMA parameters. Nothing in an objective logs, opens a file or touches a tracker.

<div align="center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/seam-dark.svg">
  <img src="docs/assets/seam-light.svg" alt="The four calls the trainer makes into an objective" width="100%">
</picture>
</div>

`LMObjective`, cut down from the class that ships, is the whole pattern:

```python
import jax.numpy as jnp, optax
from dew import Aux, EMASpec, Field, InputSpec, Objective, TokenScores

class LMObjective(Objective):
    def __init__(self, model, seq_len, *, ema_decay=0.999):
        self.model, self.seq_len = model, seq_len
        self.inputs = InputSpec(Field("text", (seq_len + 1,)))
        self.ema = EMASpec(decay=lambda step: ema_decay)

    def init(self, key):
        return self.model.init(key, jnp.zeros((1, self.seq_len), jnp.int32))

    def loss(self, params, batch, step):
        tokens = batch["text"]
        inputs, targets = tokens[:, :-1], tokens[:, 1:]
        logits = self.model.apply(params, inputs, train=True, rngs={"dropout": step.key}).astype(jnp.float32)
        ce = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
        return ce.mean(), Aux({"ce": ce.mean(), "token_accuracy": (jnp.argmax(logits, -1) == targets).mean()})

    def evaluate(self, params, batch, step):
        tokens = batch["text"]
        logits = self.model.apply(step.ema, tokens[:, :-1]).astype(jnp.float32)
        losses = optax.softmax_cross_entropy_with_integer_labels(logits, tokens[:, 1:])
        return TokenScores(losses=losses, weights=jnp.ones_like(losses))
```

The trainer compiles `loss` into one sharded step with the optimizer and the EMA update, and checkpoints the parameters, the EMA, the optimizer state, the key and the data position. The metrics an objective returns appear in the run as `train/<name>`, so this one logs `train/ce` and `train/token_accuracy`; `metrics.perplexity()` reads the `TokenScores` over a whole validation pass, weighted by target, so a batch of padding weighs nothing. The shipped class adds a `pad_id` mask, packed-document boundaries, the chunked fp32 head, and `TextSamples` when samples are configured. `DiffusionObjective` and `JepaObjective` are written the same way, and `EMASpec(select=...)` lets an objective average one subtree only, which is how JEPA keeps its target encoder. `Aux.variables` is how a mixture-of-experts objective moves its routing bias: the trainer writes those collections back into the tree. See [docs/concepts/objectives.md](docs/concepts/objectives.md) for more.

## Scaling

The trainer places the run on a mesh named `(data, expert, fsdp)`; `MeshSpec(fsdp=, expert=)` sets two of the sizes and the data axis fills the rest. The batch is split across all devices. With `fsdp=1` the parameters are replicated, which is data parallelism. With `fsdp=N` every parameter and optimizer moment above the layout's `min_shard` is split across `N` devices along the axis its module declared with `@logical_axes`, or its largest divisible axis where nothing is declared, and a layout that leaves more than `tolerance` of the model replicated stops the run rather than training it. One compiled step serves both, with its shardings declared to XLA and its buffers donated, so no code changes between a laptop and a pod.

<div align="center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/mesh-dark.svg">
  <img src="docs/assets/mesh-light.svg" alt="The (data, fsdp) mesh on two hosts" width="100%">
</picture>
</div>

```python
from dew import MeshSpec, Trainer
trainer = Trainer(objective, optimizer, key=key, mesh=MeshSpec(fsdp=1), accumulation=2)
# 8 devices: mesh (data=2, expert=1, fsdp=4); every large parameter split four ways, the EMA and Adam moments with it
```

On a TPU pod every host runs the same script. The recipes join the hosts into one JAX runtime from the cluster environment before the model is built, and stop with an error if they cannot. The data pipeline shards records by process, so each host reads its own part of the dataset; the dataset's `path` points at the GCS mount and a `gs://` checkpoint directory writes the shards to a bucket, which is the shared storage a resume across hosts needs. The `dew-tpu` command creates a slice, installs dew on every worker and starts a recipe on all of them; [docs/tpu.md](docs/tpu.md) is the walkthrough.

<div align="center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/training-loop-dark.svg">
  <img src="docs/assets/training-loop-light.svg" alt="The training loop" width="100%">
</picture>
</div>

Models compute in bf16 with fp32 parameters by default, and attention runs on the fused kernel for the current hardware (`attention_impl="auto"`: cuDNN flash attention on a GPU for the shapes cuDNN supports, XLA for the rest). Knobs the fused kernels cannot honor raise an error instead of being ignored. On an RTX 4080 a 142M parameter DiT trains 2.3x faster this way than in fp32 with reference attention, with a third of the activation memory. The compiled step for that model keeps the device busy for the whole step, with no host synchronisation, so the remaining costs are compile time (cached across runs), sampling and checkpointing.

| | Trainer argument | Recipe flag |
|---|---|---|
| FSDP degree | `mesh=MeshSpec(fsdp=4)` | `--trainer.mesh.fsdp 4` |
| Expert parallelism | `mesh=MeshSpec(expert=8)` | `--trainer.mesh.expert 8` |
| Smallest sharded parameter | `layout=Layout(min_shard=2**16)` | `--trainer.layout.min-shard 65536` |
| Gradient accumulation | `accumulation=2` | `--trainer.accumulation 2` |
| Gradient clipping | `optax.clip_by_global_norm` in the chain | `--optim.clip-grads 1.0` |
| Compute dtype | `models.build(..., dtype="bfloat16")` | `--model.dtype bfloat16` |
| Attention kernel | `models.build(..., attention_impl="auto")` | `--model.attention-impl auto` |
| fp16 loss scaling | `dynamic_scale=True` | `--trainer.dynamic-scale` |
| Process pool | `prepare_process(..., multi_host=True)` | `--trainer.multi-host True` |
| Compilation cache | `compilation_cache_dir="~/.cache/dew/xla"` | on by default |

See [docs/concepts/distributed.md](docs/concepts/distributed.md) for more.

## Data

The data pipeline is built on [Grain](https://github.com/google/grain). A dataset is a random-access source and a transform. Decoding, resizing and augmentation draw their randomness from the record's own generator, so a record gets the same augmentation and the same caption on any number of workers and hosts.

<div align="center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/pipeline-dark.svg">
  <img src="docs/assets/pipeline-light.svg" alt="From a data source to the mesh" width="100%">
</picture>
</div>

```python
# runs elsewhere: downloads Oxford Flowers
from dew.data import Loading, OxfordFlowers

data = OxfordFlowers(image_size=128, loading=Loading(workers=8)).load(batch=32)
batch = next(data.train())
# batch["image"].shape == (32, 128, 128, 3), uint8; batch["text"] holds the tokenized captions
# data.val() is the held-out records once, in a fixed order; data.steps_per_epoch is one pass over data.records
```

| Source | Records | Specs |
|---|---|---|
| TFDS datasets | images with labels or captions | `OxfordFlowers`, `HFImages` |
| ArrayRecord shards on a GCS mount | images with captions, at the scale of LAION and COYO | `ArrayRecordImages` and the named shard sets |
| Local video directories, VoxCeleb2 | video clips with audio | `LocalVideos`, `VoxCeleb2` |
| Tokenized text (`train.bin`, `val.bin`) | windows of token ids, or packed documents | `TokenWindows`, `PackedTokens` |
| URL streams (LAION-style tables) | images fetched while training | `OnlineImages`, `CombinedOnline` |

A dataset is a frozen spec in the `datasets` registry; its fields are its knobs, and tyro turns them into the recipes' flags. `load` returns a `Dataset`: `train` and `val` iterator factories, the record count and the global batch, with the per-process batch computed in one place that refuses a batch the processes cannot split. Validation is the held-out records once, in canonical order, disjoint from training, the same on any number of workers and hosts, and every process agrees how many batches it holds before a pass starts. A failed record is dropped and counted; nothing is ever replaced with zeros. Augmentation is a field of the spec, not an environment variable. `tools/benchmark_data.py` measures a loader on its own. See [docs/concepts/data.md](docs/concepts/data.md) for more.

## Configuration

A run is a `RunConfig` with four parts, and [tyro](https://github.com/brentyi/tyro) turns the tree into a command line: `--optim.learning-rate 1e-4` sets `config.optim.learning_rate`. The recipes subclass it to add their objective's knobs.

```python
from dew.config import RunConfig, ModelConfig, OptimConfig, TrainerConfig, Wandb
from dew.data import Loading, OxfordFlowers

config = RunConfig(
    model=ModelConfig("simple_dit", dict(patch_size=4, emb_features=512, num_layers=12, num_heads=8),
                      dtype="bfloat16", attention_impl="auto"),
    data=OxfordFlowers(image_size=128, loading=Loading(workers=16)),
    optim=OptimConfig(optimizer="adamw", learning_rate=2e-4, learning_rate_schedule="cosine",
                      learning_rate_warmup_steps=2000, weight_decay=0.01, clip_grads=1.0),
    trainer=TrainerConfig(name="flowers-dit", batch_size=64, epochs=500, checkpoint_every=2000,
                          wandb=Wandb(project="dew")),
)
config.save("runs/flowers-dit")        # run.json, next to the checkpoints; RunConfig.load rebuilds it, and raises on a field it does not know
```

| Part | Fields |
|---|---|
| `model` | `architecture`, `config` (the fields the registry builds the model from), `dtype`, `attention_impl` |
| `data` | a dataset spec, chosen as a subcommand (`data:oxford-flowers --data.image-size 128`); its fields are the spec's |
| `optim` | `optimizer` (adam, adamw, lamb, muon), `learning_rate`, `learning_rate_schedule`, `learning_rate_peak`, `learning_rate_end`, `learning_rate_warmup_steps`, `weight_decay`, `clip_grads` |
| `trainer` | `name`, `seed`, `batch_size`, `epochs`, `steps`, `checkpoint_dir`, `keep`, `checkpoint_every`, `eval_every`, `log_every`, `mesh` (`fsdp`, `expert`), `layout` (`min_shard`, `tolerance`), `accumulation`, `dynamic_scale`, `multi_host`, `xla_flags`, `profile` (`directory`, `steps`, `warmup`, unset traces nothing), `compilation_cache_dir`, `wandb` (`project`, `entity`, `offline`, unset runs without a tracker) |

The diffusion recipe adds `preset` and `sampler` (each a subcommand over its registry: `preset:edm --preset.sigma-data 0.5`), `guidance`, `sampling_steps`, `unconditional_prob`, `ema_decay`, `text_encoder`, `autoencoder` and `val_metrics`; the JEPA recipe adds `predictor`, `num_target_blocks`, `target_scale`, `target_aspect`, `momentum`, `momentum_steps`, `probe_classes` and `knn_k`; the language model recipe adds `tokenizer`, `ema_decay`, `sample_prompt`, `sample_tokens` and `pretrained`. The defaults carry no machine paths and no personal accounts. `--help` on any recipe prints the whole tree with its defaults. See [docs/recipes.md](docs/recipes.md) for more.

## Checkpoints and resume

A checkpoint holds six things: `step`, `params`, `opt_state`, `ema`, the run `key`, and `position`, every process's place in its data stream. The trainer writes one every `checkpoint_every` steps and once more when `fit` returns. Writes are asynchronous with Orbax; sharded arrays go from the devices to disk without passing through one host, and a `gs://` directory goes to the bucket as it is.

Retention is Orbax's job: the latest `keep` checkpoints stay.

```bash
python recipes/diffusion/train.py ... --trainer.checkpoint-dir runs/flowers-dit   # resumes from the directory's latest checkpoint
```

A run whose checkpoint directory already holds a checkpoint resumes from the latest one, and continues from the next unseen batch on every process; and a checkpoint written by two hosts refuses to resume on one, with the reason, rather than restoring the wrong position. nothing is deleted or overwritten. For inference:

```python
# runs elsewhere: needs a run directory a recipe wrote
from dew.sampling import TextToImage

pipe = TextToImage.from_run("runs/flowers-dit")            # run.json plus the latest checkpoint, EMA weights
images = pipe(["a water lily", "a sunflower"], steps=40, guidance=4.0, key=jax.random.key(0))
```

The run directory is the whole record: `run.json` is the resolved config the recipe trained with, and `from_run` rebuilds the model, the process and the encoders from it through the same function the recipe used, so what samples is what trained.

## Logging and profiling

A `Tracker` receives scalars and artifacts; `WandbTracker` sends them to Weights & Biases and renders each artifact by its type, and no tracker means the terminal. Every `log_every` steps: `train/loss`, `train/step_time_ms`, `train/samples_per_sec`, `train/mfu`, and every metric the objective returned as `train/<name>`. Every `eval_every` steps: each metric as `val/<name>`, and the objective's artifacts drawn (an image grid, a table of generated text). Publishing is a recipe's step, not the trainer's: `dew.io.publish(directory, name, tracker=...)` uploads a checkpoint and its `run.json` to the model registry after `fit`.

`train/mfu` is the step's FLOPs, counted off the compiled executable's optimized HLO, divided by the step time and by one device's dense peak; the table of peaks in `dew.telemetry.instrumentation` covers TPU v4 to v6e, A100, H100, H200 and the RTX 4080, and the metric is left out on hardware it does not know. `profile=Profile(directory, steps=N, warmup=2)` writes a profiler trace of `N` steps after the warmup into `directory`, for TensorBoard or Perfetto. The XLA compilation cache is on by default under `~/.cache/dew/xla`, so a restarted run compiles in seconds instead of minutes. A sustained non-finite loss stops the run.

## Extending Dew

**A model.** Write a Flax module that takes `(x, temb, **conditions)` for diffusion or `(tokens)` for language models, register it, and declare which of its dimensions mean what, so the layout can shard it:

```python
import flax.linen as nn
from dew import models
from dew.nn.sharding import logical_axes

@models("my_dit")
@logical_axes({("mixer_in",): ("embed", "mlp"), ("mixer_out",): ("mlp", "embed")})
class MyDiT(nn.Module):
    ...
# models.build("my_dit", ...) now works, recipes included; a declaration that disagrees with another module's for the same
# path is refused, so a renamed layer cannot silently change how it shards
```

**An objective.** A class with `init`, `loss` and `evaluate`, as in the [Objectives](#objectives) section, registered with `@objectives(name)`. The trainer needs nothing else.

**A dataset.** A frozen dataclass whose `load(batch=)` returns a `Dataset`, registered with `@datasets(name)`; it appears as a recipe subcommand with its fields as flags.

**A metric.** A function `(artifact, batch) -> value` behind `@metrics(name)`; it reads the artifact type it scores.

**A sampler.** A `Solver` with `init(x)` and `step(x, t, t_next, denoised, eps, state, key, process, denoise)`, registered with `@samplers(name)`. Guidance, conditioning and the loop are `sample`'s.

**A conditioning encoder.** Implement `tokenize`, `encode` and `to_json` on `ConditionEncoder` and register with `@encoders(name)`, so a run can rebuild it.

**A preset.** A frozen dataclass callable to a `Process`, registered with `@presets(name)`; its fields are what `run.json` stores.

## Recipes and examples

`recipes/diffusion/train.py`, `recipes/jepa/train.py` and `recipes/lm/train.py` are complete training programs over the config tree above:

```bash
python recipes/diffusion/train.py data:oxford-flowers --data.image-size 128 \
    --model.architecture simple_dit \
    --model.config '{"patch_size": 4, "emb_features": 512, "num_layers": 12, "num_heads": 8}' \
    --trainer.epochs 2000 --trainer.wandb-project my-project preset:edm

python recipes/jepa/train.py data:oxford-flowers --probe-classes 102 \
    --model.config '{"patch_size": 16, "emb_features": 384}'

python tools/tokenize_text.py --input shakespeare.txt --out data/shakespeare --tokenizer byte --val-fraction 0.02
python recipes/lm/train.py data:token-windows --data.path data/shakespeare --data.seq-len 256 \
    --model.config '{"emb_features": 384, "num_layers": 6, "num_heads": 6}' --sample-prompt "ROMEO:"
```

The scripts in [`examples/`](examples/) go from a dataset to a trained model, samples on disk and exported weights, and each one runs in the test suite at a tiny size:

* [`train_diffusion.py`](examples/train_diffusion.py): a text-to-image DiT on Oxford Flowers, four prompts sampled to `samples.png`, the EMA weights exported as safetensors.
* [`train_jepa.py`](examples/train_jepa.py): an I-JEPA encoder with linear and kNN probes, the encoder saved on its own.
* [`train_lm.py`](examples/train_lm.py): a byte-level language model on a tokenized corpus, a sample written after training.

The notebooks in [`tutorials/`](tutorials/) teach the library: [01](tutorials/01-diffusion-from-scratch.ipynb) builds diffusion from scratch without it, [02](tutorials/02-train-a-diffusion-model.ipynb) trains a DiT through the trainer, [03](tutorials/03-text-to-image-with-guidance.ipynb) adds CLIP conditioning, classifier-free guidance and the SD-VAE latent option, and [04](tutorials/04-samplers-and-schedules.ipynb) compares the samplers on the trained checkpoint. See [docs/recipes.md](docs/recipes.md) for the full config tree.

## Evaluation and export

Validation runs every `eval_every` steps: the objective produces artifacts (samples, embeddings, token scores) and the metrics score them.

```python
# runs elsewhere: the CLIP and Inception weights download
from dew import metrics

state = trainer.fit(data, steps=steps, eval_every=2000, metrics=(metrics.clip(), metrics.fid()))
# logs val/clip and val/fid; per-batch FID tracks a run over time and is not FID-50k
```

`dew.eval` has FID (with a vendored InceptionV3), CLIP score, PSNR, SSIM and perplexity. A trained model exports to the `model.safetensors` and `config.json` pair a Hugging Face style loader looks for. No leaf is renamed, transposed or cast, and the config is written as given, so anything that reads safetensors reads the tensors; loading an export as a transformers, vLLM or verl model is the per-family work in the [roadmap](#roadmap):

```python
from dew.interop import save_hf_layout

save_hf_layout(state.ema["params"], config={"architecture": "simple_dit", **fields}, directory="export/flowers")
# export/flowers/model.safetensors and export/flowers/config.json
```

`push_to_hub(directory, repo_id)` uploads that directory to a Hub repo, creating the repo if it does not exist yet. `pull_from_hub(repo_id)` downloads a repo snapshot and returns the local directory it landed in.

See [docs/api.md](docs/api.md#evaluation-and-interop) for more.

## Testing and benchmarks

The suite has three lanes. The mesh lane runs the FSDP, data parallel, checkpoint and resume tests on eight simulated CPU devices, so they run anywhere; the GPU lane runs everything else on the accelerator in parallel workers; the distributed lane spawns real `jax.distributed` process pools. CI runs the mesh and distributed lanes on every push to `main` and every pull request into it; the GPU lane runs on an RTX 4080 before each merge.

```bash
JAX_PLATFORMS=cpu pytest -m "mesh and not distributed" -n 3 --dist loadfile -q      # the mesh lane, about 5 minutes
JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false pytest -m "not mesh and not network" -n 4 --dist loadfile -q   # the GPU lane
JAX_PLATFORMS=cpu pytest -m distributed -q                                            # real process pools, serial
```

`tests/test_architectures.py` trains every registered architecture through `fit` on both an 8x1 data-parallel mesh and a 2x4 data-by-FSDP mesh, and fails if an architecture is added without a case. `tools/benchmark_step.py` measures the real training step per architecture:

| architecture | sample | batch | params | ms/step | samples/s | util |
|---|---|---|---|---|---|---|
| `simple_dit` | 64x64x3 | 16 | 19.8M | 7.9 | 2036 | 38% |
| `hierarchical_mmdit` | 64x64x3 | 16 | 55.5M | 32.7 | 489 | 23% |
| `video_dit` | 8x64x64x3 | 4 | 25.2M | 17.3 | 232 | 45% |
| `causal_transformer` | 512 tokens | 16 | 67.0M | 83.0 | 193 | 42% |

RTX 4080, bf16, excerpt; the full table with FLOPs, spread, memory and compile times is in [docs/benchmarks.md](docs/benchmarks.md). `tools/benchmark_data.py` measures the data pipeline alone.

## Installation

### Supported platforms

|            | Linux x86_64 | Cloud TPU VM |
|------------|--------------|--------------|
| CPU        | yes          | yes          |
| NVIDIA GPU | yes          | n/a          |
| Google TPU | n/a          | yes          |

CI runs the test suite on CPU on every push to `main` and every pull request into it, and the GPU lane runs on an RTX 4080 before each merge. TPU pods trained the models in the [gallery](docs/gallery.md); the current trainer's multi-host path is on the [roadmap](#roadmap) for revalidation.

### Instructions

Dew needs Python 3.11 or later. There is no release on PyPI yet; install from the repository. The base install comes with a CPU-only JAX; install the [JAX build](https://docs.jax.dev/en/latest/installation.html) for your accelerator as well.

| Platform   | Instructions                                                                 |
|------------|------------------------------------------------------------------------------|
| CPU        | `pip install "dew-ml @ git+https://github.com/AshishKumar4/dew"`               |
| NVIDIA GPU | `pip install "dew-ml @ git+https://github.com/AshishKumar4/dew" "jax[cuda12]"` |
| Google TPU | `pip install "dew-ml @ git+https://github.com/AshishKumar4/dew" "jax[tpu]"`    |

Optional extras: `[tfds]` for TFDS datasets, `[av]` for video and audio, `[streaming]` for URL streaming, `[metrics]` for FID, `[interop]` for safetensors, as in `"dew-ml[tfds,metrics] @ git+https://github.com/AshishKumar4/dew"`. The package imports as `dew`. The first release will ship as `dew-ml`; the bare `dew` name on PyPI is an unused placeholder, and a PEP 541 request for it is the plan.

To work on Dew itself, read [CONTRIBUTING.md](CONTRIBUTING.md) first: it states the design rules, the reference-parity requirement for every port, and what a merge needs.


```bash
git clone https://github.com/AshishKumar4/dew.git
cd dew && pip install -e ".[test]"
JAX_PLATFORMS=cpu pytest -m "not network" -q
```

## Roadmap

The goal is to train the way the large labs train and to run what they release, on the same trainer. In progress means a branch exists; the rest is ordered by dependency.

### Architecture parity

| Family | What it needs beyond today's decoder | Status |
|---|---|---|
| Qwen3 (dense) | checkpoint loading with logit parity against the reference implementation | in progress |
| Gemma 3 | sandwich norms, attention scale, per-layer-type RoPE; checkpoint loading | in progress |
| Qwen3.5, Qwen3.6, Qwen3.8 | the GatedDeltaNet linear-attention mixer, gated attention, mixture of experts, multi-token prediction | planned |
| Gemma 4 (E2B, E4B, 26B-A4B, 31B) | per-layer input embeddings, KV sharing across layers, mixture of experts | planned |
| DeepSeek V3, V3.2, V4 | multi-head latent attention, sparse attention, mixture of experts with shared experts and bias-based load balancing, multi-token prediction, FP8 weights | planned |
| GLM 4.5 and 5.x | mixture of experts, multi-token prediction | planned |
| Kimi K2 and Kimi Linear | mixture of experts, the MuonClip optimizer, the KDA linear-attention mixer | planned |
| gpt-oss | attention sinks, MXFP4 expert weights | planned |
| MiniMax, Nemotron-H | lightning attention and Mamba hybrids | planned |
| Llama 3 and 4, Mixtral | dense and mixture-of-experts baselines | planned |
| Diffusion language models | bidirectional attention and a mask token on the decoder, masked-diffusion and continuous-embedding objectives on the sqrt schedule, a rounding sampler; parity with the open-weight diffusion language models | planned |

Every family lands with the same proof: the reference implementation and Dew agree on the logits of a real checkpoint, and the export loads back in transformers. The research notes that shape this roadmap, each citing its sources, are in [docs/research/](docs/research/README.md).

### Systems

* Mixture of experts: an expert block for the decoder, ragged matmuls, and an `expert` mesh axis with expert parallelism
* Tensor, pipeline and context parallelism as further mesh axes on the same trainer, with the rules for choosing between them stated from arithmetic intensity
* FP8 training with fine-grained scaling, and MXFP4 weight loading
* The Muon and MuonClip optimizers; weight-decay and schedule utilities that match the published recipes
* Long context: RoPE scaling, sequence packing, two-stage context extension
* Multi-host validation on a TPU pod, emergency and multi-tier checkpointing, goodput measurement
* Scan over layers for compile time at depth

### Kernels and performance

* Pallas flash attention (Triton on GPUs, splash on TPUs) as a selectable kernel, adopted for `auto` only where it measures faster than cuDNN; in progress
* Fused norms, SwiGLU and ragged expert matmuls, measured against XLA's own fusion first
* A sweep of XLA GPU flags with the winners as the default; in progress
* Evaluation of tokamax and other published kernel libraries behind the same kernel path

### Multimodal and evaluation

* Audio-conditioned video models end to end, and vision-language inputs for the decoder
* FID-50k and the standard language model evaluations as offline jobs
* Streaming as a Grain pipeline with checkpointable position

### Tooling

* The `dew` name on PyPI, and a first release on it

## Citing Dew

```
@software{dew2026github,
  author = {Ashish Kumar Singh},
  title = {Dew: building and training modern architectures at scale, in JAX and Flax},
  url = {https://github.com/AshishKumar4/dew},
  version = {0.1.0},
  year = {2026},
}
```

## Acknowledgements

**This project is partially supported by [Google TPU Research Cloud](https://sites.research.google/trc/about/). I would like to thank the Google Cloud TPU team for providing me with the resources to train the bigger text-conditional models in multi-host distributed settings.**

Dew builds on [JAX](https://github.com/jax-ml/jax), [Flax](https://github.com/google/flax), [Optax](https://github.com/google-deepmind/optax), [Orbax](https://github.com/google/orbax), [Grain](https://github.com/google/grain), [tyro](https://github.com/brentyi/tyro), [albumentations](https://github.com/albumentations-team/albumentations) and [Weights & Biases](https://github.com/wandb/wandb). The VAE and parts of the attention code come from [diffusers](https://github.com/huggingface/diffusers), the InceptionV3 from [jax-fid](https://github.com/matthias-wright/jax-fid), and the Karras samplers follow [k-diffusion](https://github.com/crowsonkb/k-diffusion) and the [EDM](https://github.com/NVlabs/edm) reference code. The papers are listed in [docs/references.md](docs/references.md).

## Reference documentation

* [Concepts](docs/concepts/): [objectives](docs/concepts/objectives.md), [distributed training](docs/concepts/distributed.md), [the data pipeline](docs/concepts/data.md), [language models](docs/concepts/language_models.md), [mixture of experts](docs/concepts/moe.md)
* [API reference](docs/api.md), [recipes](docs/recipes.md), [benchmarks](docs/benchmarks.md), [TPUs](docs/tpu.md)
* [Diffusion explained](https://nbviewer.org/github/AshishKumar4/dew/blob/main/tutorials/01-diffusion-from-scratch.ipynb), a notebook that builds diffusion from scratch without the library
* [Gallery](docs/gallery.md) and [migrating from FlaxDiff](docs/from-flaxdiff.md)

Dew is released under the [MIT license](LICENSE).
