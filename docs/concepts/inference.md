# Inference

Dew has one front door for inference, `dew.pipeline`, and three task types behind it: `TextGeneration` for decoders, `BlockGeneration` for DiffusionGemma, and `TextToImage` for diffusion models. A task binds a model, one variables tree and, when the source ships one, the host processor that turns text into model inputs. A call takes prompts and a seed and returns a typed record. The same code runs on one device, on a mesh of eight, and on a multi-process pool.

```python
import dew

task = dew.pipeline("runs/flowers-dit")          # a run directory -> TextToImage
images = task(["a water lily", "a sunflower"], seed=0).host().images

task = dew.pipeline("runs/shakespeare")          # an LM run directory -> TextGeneration
print(task("ROMEO:", seed=0).text[0])

task = dew.pipeline("Qwen/Qwen3-0.6B")           # a checkpoint -> TextGeneration
print(task("The capital of France is", 32, seed=0).text[0])
```

`pipeline(source, *, mesh=None, layout=None, dtype=None, ema=True, step=None, revision=None)` reads the kind of source from what is on disk. A directory holding `run.json` is a run: a diffusion run rebuilds its objective the way the recipe built it and becomes a `TextToImage`; a language-model run rebuilds the model `run.json` records, decodes through the run's tokenizer and becomes a `TextGeneration`. Anything else loads through `dew.interop.load_pretrained`: a decoder becomes a `TextGeneration` with the source's own sampling policy and budget, a DiffusionGemma a `BlockGeneration`. `ema` reads a run's averaged weights when the run kept them; `step` selects a run's checkpoint; `dtype` casts a run's floating weights or is the source loader's compute dtype.

## Placement

Weights are placed once, when the task is built. Without `mesh` they land on the default device. With `mesh=MeshSpec(...)` the front door builds that mesh and shards the weights under `layout` with the same rules the trainer uses for a train state (`Layout.shardings`); a run's checkpoint restores straight onto the mesh, without a host copy.

Every call splits its rows over the mesh's batch axes. Each process hands in its own rows, the way `SampledRollout` does, and every process must hand in the same number of rows at the same padded width; rows pad up to the device count with repeats that produce nothing. Results keep that placement: the arrays in a `Generation`, `CanvasGeneration` or `Images` are global arrays sharded by row, so a result can feed the next device computation without a gather. `result.host()` returns the same record over host arrays holding this process's real rows, in order. Each row's random draw folds in its global row index, so a pool draws exactly what a single process draws for the same prompts; the canvas sampler is the exception, since it draws one key per refinement for the whole batch.

```python
from dew.training import Layout, MeshSpec

task = dew.pipeline("runs/flowers-dit", mesh=MeshSpec(fsdp=2), layout=Layout(min_shard=2**12))
result = task(prompts, steps=30, seed=0)          # rows sharded over the mesh
grid = result.host().images                       # this process's rows, as NumPy
```

## Controls

`seed=n` is accepted wherever `key` is and means `jax.random.key(n)`; a call takes exactly one of them. `max_new_tokens` is a positional budget with a default the source declares: a Hub checkpoint's `generation_config.json`, an LM run's `sample_tokens`, an objective's `Samples`. A call of the same shapes and controls reuses the compiled executable, and so does the same task over other weights, so a training loop that rebinds a policy snapshot every step compiles once.

`Generation.text` decodes each row's valid continuation through the bound processor on first access; `task.decode(result)` is the same text. A task built without a processor takes token rows and returns token rows.

`Sampling` carries the text policy: temperature, top-k, nucleus top-p, relative min-p, the EOS ids and the padding id. `TextToImage` carries `steps`, `guidance` and `sampler` defaults; a call may override any of them, and `guidance=None` is the plain conditional prediction. `TextToImage.prepare(prompts, seed=...)` encodes prompts and draws their noise once, so several solvers can run on the same `DenoisingInputs`.

## Workflows

### A trained diffusion model to images

The trained objective is the pipeline. `objective.pipeline(state)` binds the state's published weights, the EMA copy merged over the live weights when the objective keeps one, on the mesh the state sits on. No reload and no copy; the task samples the way the objective's own evaluation does.

```python
import optax
from dew.training import Checkpoints, MeshSpec, Trainer

trainer = Trainer(objective, optax.adamw(1e-4), key=jax.random.key(0), mesh=MeshSpec(fsdp=2),
                  checkpoints=Checkpoints("runs/flowers-dit"))
state = trainer.fit(data, steps=20_000)
pipe = objective.pipeline(state)
images = pipe(["a water lily", "a sunflower"], steps=40, guidance=3.0, seed=1).host().images
```

Later, `dew.pipeline("runs/flowers-dit")` rebuilds the same task from the run directory; `TextToImage.from_run` and `from_pretrained` (a run published to the Hub) are the same path with a diffusion run named explicitly.

### A trained language model to text

`LMObjective.pipeline(state, processor=...)` binds the decoder over the state's published weights, with the sampling and budget of the objective's `Samples`. A run's tokenizer becomes a processor through `RunProcessor`, which left-pads prompt rows and decodes one string per row.

```python
from dew.data import ByteTokenizer
from dew.inference import RunProcessor

state = trainer.fit(data, steps=steps)
task = objective.pipeline(state, processor=RunProcessor(ByteTokenizer()))
print(task("ROMEO:", seed=1).text[0])
```

The LM recipe writes the resolved model into `run.json`, vocabulary and context included, so `dew.pipeline("runs/shakespeare")` rebuilds the model without the recipe and restores the checkpoint's weights, averaged when the run kept an EMA.

### A published checkpoint on a mesh

A Hub repository or a directory in its layout loads once and is placed on the mesh under the trainer's rules. The task decodes through the checkpoint's tokenizer, samples with its generation config and stops at its EOS ids. In a multi-process pool, every process runs the same lines with its own prompts.

```python
from dew.training import MeshSpec

task = dew.pipeline("Qwen/Qwen3-0.6B", mesh=MeshSpec(fsdp=8), dtype="bfloat16")
result = task(prompts, 64, seed=0)
for text in result.text:
    print(text)
```

`Pretrained.text_generation(sampling=...)` and `Pretrained.block_generation()` are the same tasks built by hand from a loaded bundle; `task.bind(variables)` gives the task over other weights, which is how a rollout draws from a training policy snapshot.

## Serving

Serving stays outside Dew. Export a checkpoint with `Pretrained.save` and serve it with vLLM or Ollama; `OllamaCompletion` and `OpenAICompletion` in `dew.inference` bind a model on those engines through their official clients, with an explicit `Sampling` translated into the backend's controls.
