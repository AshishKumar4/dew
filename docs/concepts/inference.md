# Inference

> An AI assistant maintains this document. It is presented as-is.

`dew.pipeline` loads an inference task. It returns `TextGeneration` for decoders, `BlockGeneration` for DiffusionGemma, `MaskedGeneration` for masked-diffusion language models such as LLaDA and Dream, and `TextToImage` for diffusion models. A task holds the native model, its variables, and any tokenizer or condition encoders the source provides. You call it with inputs and a key or seed. Results stay where the task placed them on the devices.

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

The signature is `pipeline(source, *, mesh=None, layout=None, dtype=None, param_dtype=None, ema=True, step=None, revision=None)`. `source` is a run directory, a checkpoint directory or a Hub repository. A run's `run.json` must name its objective. Saved diffusion, LM, DPO, GRPO, PPO, block-diffusion and masked-diffusion runs have generation tasks. Other kinds, such as JEPA runs, have no generation task and raise here. Checkpoints in a published layout load through `dew.interop.load_pretrained`, including native latent-diffusion checkpoints described by `model_index.json`.

`step` picks one of a run's checkpoints. `revision` pins a Hub source. `dtype` sets the compute dtype. `param_dtype` sets how parameters are stored: `None` keeps a run's stored dtypes and uses float32 master weights for a published source.

## Weights

For the plain LM, image-diffusion and block-diffusion objectives, `ema=True` asks for the moving-average weights. If the run or state has no EMA copy, this raises. Use `ema=False` for the live weights. DPO, GRPO and PPO use the EMA slot to hold a frozen reference for the loss. Their pipelines always publish the trained policy, never that reference. PPO also leaves out the critic.

`objective.pipeline(state)` picks weights by the same rule without reloading a checkpoint. It keeps the arrays the trainer has already placed. `task.bind(variables)` makes a task over another set of variables. It copies the mapping structure and shares the array buffers, so do not change or donate those arrays while a task uses them.

## Placement

Without `mesh`, the default `MeshSpec()` puts the current process pool's devices on data parallelism. Pass `mesh=MeshSpec(...)` for another layout. Weights are placed with `Layout.shardings` and its replication check, the same way the trainer places them. Checkpoint restore gets explicit shardings for the current devices. It does not reuse the topology the writer recorded.

Every process supplies its own rows. All cooperating processes must supply the same number of rows and the same tokenized shapes, and use the same execution controls, including the same number of continuations. The processes agree on invalid inputs, conflicting controls and errors in prepared inputs before any device collective runs.

Results hold global arrays sharded by row, including any filler rows added so the batch divides across devices. `result.host()` returns the same record with NumPy arrays for this process's real rows. It does not gather rows from other processes. Filler rows are added as prompts, before any continuation exists. So all continuations of a prompt stay on the process that asked for them, and the rows `host()` drops belong only to filler prompts. Token generation and image prior noise use keys per global row. Canvas refinement and an explicitly sampled VAE posterior use keys for the whole batch, so changing the placed batch shape can change their draws.

```python
from dew.training import Layout, MeshSpec

task = dew.pipeline("runs/flowers-dit", mesh=MeshSpec(fsdp=2), layout=Layout(min_shard=2**12))
result = task(prompts, steps=30, seed=0)
local_pixels = result.host().images
```

## Controls and text

A call takes exactly one of `seed` and `key`. `seed=n` means `jax.random.key(n)`.

`max_new_tokens` is the continuation budget. A checkpoint can set a default in `generation_config.json`. If it declares only `max_length`, the budget is that total length minus the padded prompt width. A budget in the call wins. If the source gives no limit, you must pass a budget. Building a task from source defaults raises if an unsupported control is active, such as a repetition penalty or wall-clock stopping.

`n` is the number of continuations per prompt, a positive integer. If the checkpoint sets `num_return_sequences`, that becomes the task's default, whatever sampling policy you pass. Otherwise the default is 1. An explicit `n` in the call wins in both directions, so `n=1` overrides a source that asks for more.

Every result array then has `n` rows per prompt, in prompt order: prompt zero's continuations first, then prompt one's. `result.rows` is this process's real prompts times `n`. `Generation.text` returns one string per row in the same order, and each row has its own length, termination flag and likelihoods.

Each prompt is prepared, tokenized and prefilled once for all its continuations, and its media goes through the processor once per call. The continuations then run one after another on the device, each with its prompts batched the same way as for a single continuation. Decode time therefore grows with `n`. The cache working memory of one continuation is reused by the next, so only the output arrays grow with `n`.

Randomness follows each prompt's own key. Continuation zero draws with that key, so `n=1` and continuation zero of a larger request are the same draw. Continuation `j` folds `j` into the key, so asking for more continuations does not change the ones already drawn.

```python
import dew

task = dew.pipeline("runs/shakespeare")
result = task("ROMEO:", 32, n=4, seed=0)
for continuation in result.text:
    print(continuation)
```

An LM run records its `sampling` value and `sample_tokens` budget, and reloading the run keeps that preview policy. `Sampling` holds temperature, top-k, top-p, min-p, EOS ids and the output padding id. `TextToImage` holds default solver, guidance and step count. A call can override any of these. `guidance=None` turns off classifier-free guidance.

`Generation.text` and `CanvasGeneration.text` decode the continuation tokens the first time you read them and cache the strings. A result with no processor bound raises when you ask for text. You do not have to change a tokenizer that has no pad token. Dew tokenizes without padding, then pads the numeric rows and masks at the same boundary `RunProcessor` uses. The tokenizer and its export stay unchanged.

Repeated calls with the same shapes and controls reuse the compiled executables. Binding new weights does not change what the model compiles to.

## Workflows

### A trained diffusion model to images

Start with an objective and dataset built as in the [diffusion guide](../guides/diffusion.md):

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

To reload the run later, Dew also needs the model description. A recipe's `RunConfig.train` writes `run.json` next to the checkpoints. For a run you assembled by hand, save the matching configuration with `config.save(directory)`. `dew.pipeline(directory)` then rebuilds the task. The checkpoint arrays on their own do not describe the model.

### A trained language model to text

`LMObjective.pipeline` uses the policy and budget set in its `Samples`. If the training objective only knew token IDs, bind a processor:

```python
from dew.data import ByteTokenizer
from dew.inference import RunProcessor

state = trainer.fit(data, steps=steps)
task = objective.pipeline(state, processor=RunProcessor(ByteTokenizer()))
print(task("ROMEO:", seed=1).text[0])
```

The LM recipe records the resolved model, tokenizer, sampling value and budget, and `dew.pipeline("runs/shakespeare")` restores them. DPO, GRPO and PPO pipelines publish the trained policy as described above, even when a frozen reference sits in `state.ema`.

### A published checkpoint on a mesh

```python
from dew.training import MeshSpec

task = dew.pipeline("Qwen/Qwen3-0.6B", mesh=MeshSpec(fsdp=8), dtype="bfloat16")
result = task(prompts, 64, seed=0)
for text in result.text:
    print(text)
```

Every process runs these lines with its own prompts. A real checkpoint needs enough host and device memory for its weights and cache. `Pretrained.text_generation`, `block_generation` and `text_to_image` build the same task types from a bundle you have already loaded.

## Image initialization and latent handoff

`prepare` accepts normalized NHWC pixels or uint8 pixels through `image=`, and images that are already encoded through `image_latents=`. Shapes follow the task's `InputSpec` and autoencoder. `times=` picks an explicit starting point on the source grid. The initial state is `alpha(t) * image_latents + sigma(t) * noise`. `noise=` supplies the unit Gaussian noise for this.

A pixel mask has shape `[B, H, W, 1]`. Its masked-image conditions go to both guidance branches. `encode_key=None` uses the VAE posterior mean. An explicit key samples from the posterior. Condition encoders accept native prompt records as well as strings. `unconditional=` supplies one row, or one row per prompt, in place of the configured unconditional input.

`initial=` is a latent state that is already noisy, and Dew never adds noise to it again. Prepared inputs keep their process and concrete time grid. `task(prepared, seed=..., decode=False)` skips the VAE and the image checker and returns unclipped `result.latents`, with `result.images` set to `None`. For a base and refiner handoff, pass those latents as another task's `initial` with the matching partial grid. Normal calls return both the decoded images and the latents before decoding.

Prepared inputs must belong to the task's mesh. A preparation on the source grid also records its step count. To change the count, prepare a new initial state.

## Serving

Dew does not include a server. Export with `Pretrained.save` and serve the checkpoint with vLLM or Ollama. `OllamaCompletion` and `OpenAICompletion` use those projects' official clients. Their results keep the backend's metadata and do not make up native raw-policy or behavior-policy likelihoods.
