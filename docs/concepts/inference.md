# Inference

`dew.pipeline` loads an inference task. It returns `TextGeneration` for decoders, `BlockGeneration` for DiffusionGemma, or `TextToImage` for diffusion models. A task binds the native model, its variables and any tokenizer or condition encoders the source provides. Calls take inputs and a key or seed. Results keep their device placement.

```python
import dew

images = dew.pipeline("runs/flowers-dit")
result = images(["a water lily", "a sunflower"], seed=0)
pixels = result.host().images

text = dew.pipeline("runs/shakespeare")
print(text("ROMEO:", seed=0).text[0])

published = dew.pipeline("Qwen/Qwen3-0.6B")
print(published("The capital of France is", 32, seed=0).text[0])
```

`pipeline(source, *, mesh=None, layout=None, dtype=None, ema=True, step=None, revision=None)` accepts a run directory or a checkpoint directory or Hub repository. `run.json` must name its objective. Saved diffusion, LM, DPO, GRPO, PPO and block-diffusion runs have generation tasks. JEPA and masked-diffusion runs have no standalone generation task and raise at this entry point. Checkpoints in a published layout load through `dew.interop.load_pretrained`, including native latent-diffusion checkpoints described by `model_index.json`.

`step` selects a run's checkpoint; `revision` pins a Hub source. For runs, `dtype` casts floating weight leaves. For published checkpoints, it selects the loader's compute dtype.

## Weights

For ordinary LM, image-diffusion and block-diffusion objectives, `ema=True` requests the moving-average weights. A run or state with no EMA raises; use `ema=False` for live weights. DPO, GRPO and PPO use the EMA slot for a frozen loss reference. Their pipelines always publish the trained policy, never that reference. PPO also excludes the critic.

`objective.pipeline(state)` uses the same selection rule without reloading a checkpoint. It retains the arrays already placed by the trainer. `task.bind(variables)` creates a task over another variables snapshot. Mapping structure is captured; array buffers are shared. Do not mutate or donate them while a task uses them.

## Placement

Without `mesh`, the default `MeshSpec()` uses the current process pool's devices in data parallelism. `mesh=MeshSpec(...)` selects another layout. Weight placement uses `Layout.shardings` and its replication check, as the trainer does. Checkpoint restore gets explicit shardings for the current devices; it does not reuse the topology recorded by the writer.

Every process supplies its own rows. The cooperating processes must supply equal row counts and tokenized shapes and use matching execution controls, including the same number of continuations. Invalid inputs, conflicting controls and prepared-input errors are agreed before device collectives.

Results contain global row-sharded arrays, including any filler rows needed for device divisibility. `result.host()` returns the same record with NumPy arrays for this process's real rows. It does not gather the other processes' rows. Filler rows are added to the prompts, before the continuations exist, so a prompt's continuations stay together on the process that asked for them and the rows `host()` drops belong only to filler prompts. Token generation and image prior noise use global row keys. Canvas refinement and an explicitly sampled VAE posterior use batch-wide keys, so changing their placed batch shape can change the draws.

```python
from dew.training import Layout, MeshSpec

task = dew.pipeline("runs/flowers-dit", mesh=MeshSpec(fsdp=2), layout=Layout(min_shard=2**12))
result = task(prompts, steps=30, seed=0)
local_pixels = result.host().images
```

## Controls and text

A call takes exactly one of `seed` and `key`. `seed=n` means `jax.random.key(n)`.

`max_new_tokens` is the continuation budget. A checkpoint may supply a default in `generation_config.json`. If only `max_length` is declared, the continuation budget is that total length minus the padded prompt width. An explicit call budget takes precedence. Without either source limit, the caller must provide a budget. Active unsupported controls, such as repetition penalties or wall-clock stopping, raise when constructing a source-default task.

`n` is how many continuations a prompt gets, a positive integer. A checkpoint's `num_return_sequences` becomes the task's default, whatever sampling policy the caller passes; otherwise it is 1. An explicit call `n` takes precedence in both directions, so `n=1` overrides a source asking for more. Every result array then has `n` rows per prompt: prompt zero's continuations first, then prompt one's, in the order the prompts arrived, and `result.rows` counts this process's real prompts times `n`. `Generation.text` returns one string per row in the same order, and every row carries its own length, termination flag and likelihoods.

A prompt is prepared, tokenized and prefilled once for all its continuations, and its media reaches the processor once per call. The continuations then run one after another on the device, with each continuation's prompts batched as they are for a single continuation, so decode time grows with `n` while one continuation's cache working memory is reused by the next; the output arrays are what grows with `n`. Randomness follows the prompt's own key: continuation zero draws with it, so `n=1` and continuation zero of a larger request are the same draw, and continuation `j` folds `j` into that key, so asking for more continuations leaves the ones already drawn unchanged.

```python
import dew

task = dew.pipeline("runs/shakespeare")
result = task("ROMEO:", 32, n=4, seed=0)
for continuation in result.text:
    print(continuation)
```

An LM run records its `sampling` value and `sample_tokens` budget. Reloading the run preserves the preview policy. `Sampling` carries temperature, top-k, top-p, min-p, EOS ids and the output padding id. `TextToImage` carries solver, guidance and step defaults. Calls can override these values; `guidance=None` disables classifier-free guidance.

`Generation.text` and `CanvasGeneration.text` decode supported continuation tokens on first access and cache the strings. Results without a bound processor raise when text is requested. Pad-less tokenizers need no caller mutation. Dew tokenizes without padding, then pads numeric rows and masks at the same boundary used by `RunProcessor`; the tokenizer and its export remain unchanged.

Repeated calls of equal shapes and controls reuse compiled executables. Rebinding weights does not change the model's compilation identity.

## Workflows

### A trained diffusion model to images

Continue with an objective and dataset built as in the [diffusion guide](../guides/diffusion.md):

```python
import jax
import optax
from dew.training import Checkpoints, MeshSpec, Trainer

trainer = Trainer(objective, optax.adamw(1e-4), key=jax.random.key(0),
                  mesh=MeshSpec(fsdp=2), checkpoints=Checkpoints("runs/flowers-dit"))
state = trainer.fit(data, steps=20_000)
pipe = objective.pipeline(state)
images = pipe(["a water lily", "a sunflower"], steps=40, guidance=3.0, seed=1).host().images
```

Reloading also needs the model description. A recipe's `RunConfig.train` writes `run.json` alongside checkpoints. For a manually assembled run, save the matching configuration with `config.save(directory)`. Then `dew.pipeline(directory)` rebuilds the task. Checkpoint arrays alone do not describe the model.

### A trained language model to text

`LMObjective.pipeline` carries the policy and budget configured in its `Samples`. Bind a processor if the training objective only knew token IDs:

```python
from dew.data import ByteTokenizer
from dew.inference import RunProcessor

state = trainer.fit(data, steps=steps)
task = objective.pipeline(state, processor=RunProcessor(ByteTokenizer()))
print(task("ROMEO:", seed=1).text[0])
```

The LM recipe records the resolved model, tokenizer, sampling value and budget. `dew.pipeline("runs/shakespeare")` restores those settings. DPO, GRPO and PPO pipelines follow the trained-policy rule described above even when a frozen reference occupies `state.ema`.

### A published checkpoint on a mesh

```python
from dew.training import MeshSpec

task = dew.pipeline("Qwen/Qwen3-0.6B", mesh=MeshSpec(fsdp=8), dtype="bfloat16")
result = task(prompts, 64, seed=0)
for text in result.text:
    print(text)
```

Every process runs these lines with its own prompts. A real checkpoint needs enough host and device memory for its weights and cache. `Pretrained.text_generation`, `block_generation` and `text_to_image` construct the same task types from an already loaded bundle.

## Image initialization and latent handoff

`prepare` accepts normalized NHWC pixels or uint8 pixels with `image=`, and already encoded clean images with `image_latents=`. Shapes follow the task's `InputSpec` and autoencoder. `times=` selects an explicit starting point in the source grid. The initial state is `alpha(t) * image_latents + sigma(t) * noise`; `noise=` supplies unit Gaussian noise for this operation.

A pixel mask has shape `[B, H, W, 1]`. Its masked-image conditions reach both guidance branches. `encode_key=None` selects the VAE posterior mean; an explicit key samples its posterior. Condition encoders can accept native prompt records as well as strings. `unconditional=` supplies one row or one row per prompt in place of the configured unconditional datum.

`initial=` is an already-noisy latent state and is never noised again. Prepared inputs retain their process and concrete time grid. `task(prepared, seed=..., decode=False)` skips the VAE and image checker and returns unclipped `result.latents`; `result.images` is `None`. Use those latents as another task's `initial` with the appropriate partial grid for a base/refiner handoff. Normal calls return both decoded images and pre-decode latents.

Prepared inputs must belong to the task's mesh. A source-grid preparation also records its step count; changing that count requires preparing a new initial state.

## Serving

Serving stays outside Dew. Export with `Pretrained.save` and serve the checkpoint with vLLM or Ollama. `OllamaCompletion` and `OpenAICompletion` use their official clients. Their results retain backend metadata without inventing native raw-policy or behavior-policy likelihoods.
