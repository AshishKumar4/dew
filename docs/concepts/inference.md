# Inference

An inference task holds a Dew model, its variables, and any tokenizer or condition encoders supplied by the source. Call it with inputs and a key or seed to generate.

`dew.pipeline(source)` loads a task from a run directory, checkpoint directory or Hub repository. For decoders it returns `TextGeneration`; for DiffusionGemma, `BlockGeneration`. LLaDA, Dream and other masked-diffusion language models use `MaskedGeneration`. Diffusion models use `TextToImage`. To serve a `TextGeneration` task, `dew.inference.serving.Server` continuously batches requests over one resident KV cache.

## Example

The example trains a small byte-level decoder for a few seconds, then generates from it in three ways: from the training state, through a server, and from an exported checkpoint loaded with `dew.pipeline`.

```python
import dataclasses

import jax
import numpy as np
import optax

from dew import Dataset, Trainer
from dew.data import ByteTokenizer
from dew.inference import RunProcessor
from dew.nn.backbones import CausalTransformer
from dew.objectives.lm import LMObjective
from dew.sampling import Sampling

tokenizer = ByteTokenizer()
ids = np.array(tokenizer.encode("One day, Lily saw a big dog in the park. "
                                "The dog wanted to play. " * 40), dtype=np.int32)
rows = np.stack([ids[i:i + 65] for i in range(0, 16 * 64, 64)])
data = Dataset(train=lambda partition: iter([{"text": rows}] * 200), val=None,
               records=16, batch=16)
model = CausalTransformer(vocab_size=tokenizer.vocab_size,
                          emb_features=64, num_layers=2, num_heads=2, mlp_features=256,
                          max_seq_len=128)
objective = LMObjective(model, seq_len=64)
state = Trainer(objective, optax.adamw(3e-3), key=jax.random.key(0)).fit(
    data, steps=150, log_every=150)

task = objective.pipeline(state, processor=RunProcessor(tokenizer))
result = task(["One day", "The dog"], 20, key=0, n=2)
for text in result.text:
    print(repr(text))
print(result.lengths, result.terminated)
```

```text
Training CausalTransformer from step 0 to 150: 147,904 parameters, on 1 × cpu, batch 16, float32
step 150/150  loss 0.009061  ce 0.009061  perplexity 1.009  token_accuracy 99.6%  step_time_ms 59.83  samples_per_sec 267.4  accepted 100.0%
Trained 150 steps in 0:00:11: first step after 2.49 s, then 17.0 step/s
77.9% of the wall time in steps, final loss 0.009061
', Lily saw a big dog'
', Lily saw a big dog'
' play. One day, Lily'
' play. One day, Lily'
[20 20 20 20] [False False False False]
```

`objective.pipeline(state)` builds a `TextGeneration` from the trained weights without reloading a checkpoint. `RunProcessor(tokenizer)` lets it accept strings and decode results. The call asks for 20 new tokens and `n=2` continuations per prompt. The result therefore has four rows: prompt zero's continuations first, then prompt one's.

`result.text` contains the continuations without their prompts. `lengths` counts the generated tokens, and `terminated` reports whether a row stopped at EOS. The task samples at temperature 1.0, the `Sampling` default. This decoder has memorized its one sentence, so both continuations of each prompt agree.

## dew.pipeline

```text
pipeline(source, *, mesh=None, layout=None, dtype=None, param_dtype=None,
         ema=None, step=None, revision=None)
```

| Argument | Meaning |
|---|---|
| `source` | A run directory (it holds `run.json`), a checkpoint directory in a published layout, or a Hub repository. |
| `mesh`, `layout` | Where the weights are placed. Without `mesh`, `MeshSpec()` puts the current process pool's devices on data parallelism. |
| `dtype` | Compute dtype. |
| `param_dtype` | Parameter storage. `None` keeps a run's stored dtypes and uses float32 master weights for a published source; `"auto"` keeps the stored dtypes for both, which for a published source means its `config.json` `dtype`, or else its first floating tensor's. |
| `ema` | `None` (default): the run's moving-average weights when it stored them, else the live ones. `True` requires the average; `False` reads the live weights. |
| `step` | Which of a run's checkpoints to load; refused for a source checkpoint. |
| `revision` | Pins a Hub source; refused for a run directory. |

A run's `run.json` must name its objective. Saved diffusion, LM, DPO, GRPO, PPO, block-diffusion and masked-diffusion runs have generation tasks. Other kinds, such as JEPA runs, raise an error because they have no generation task. Loading a task also configures XLA's on-disk compilation cache so a restarted process can reuse compiled programs.

`dew.interop.Pretrained.load` reads checkpoints in a published layout, including latent-diffusion checkpoints described by `model_index.json`. For an original-format single-file checkpoint, use `Pretrained.load(repo_or_dir, single_file="name.safetensors")`.

The single-file tests compare Stable Diffusion 1.x, SDXL and FLUX.1 with diffusers' `from_single_file`. The released SD 1.5 file and tiny SD 1.x, SDXL and Flux files convert to the same tensors. For the released SDXL base, FLUX.1-dev and FLUX.1-schnell files, the checks cover the names and shapes expected by diffusers' models.

Using diffusers' key maps, Dew converts the file once into a cached diffusers pipeline layout. If `repo_or_dir` has a `model_index.json`, Dew uses its configs. Otherwise diffusers infers a repository from the file, and Dew uses that repository's configs pinned to the fetched commit. Components missing from the file, such as Flux text encoders and VAE, also load their weights from this location. If weights are missing there too, loading stops with an error naming the component.

Dew keeps the cache entry only after the pipeline loads. A newer diffusers version triggers conversion again. FLUX.1's config repositories are gated on the Hub, so accept the license and log in, or use a local directory with the configs. Conversion requires `pip install 'dewml[diffusers]'`. The loaded pipeline runs in JAX.

This exports the decoder from the example in the Hugging Face layout and loads it back:

```python
import jax.numpy as jnp

import dew
from dew.interop import PretrainedDecoder
from dew.training import MeshSpec

PretrainedDecoder.from_model(model, state.variables, tokenizer="byte").save("lily-decoder")
loaded = dew.pipeline("lily-decoder", mesh=MeshSpec(), dtype=jnp.float32)
loaded = dataclasses.replace(loaded, processor=RunProcessor(tokenizer),
                             sampling=Sampling(temperature=0.0))
print(loaded("One day", 20, key=0).text)
```

```text
(', Lily saw a big dog',)
```

The byte vocabulary has no Hugging Face tokenizer files, so the example attaches a processor to the loaded task. A checkpoint that includes a tokenizer loads with its processor. Its `generation_config.json` sets the sampling policy and budget.

A Hub name such as `"Qwen/Qwen3-0.6B"` loads the same way. You need enough host and device memory for the weights and cache. If you already loaded a bundle with `Pretrained.load`, use `PretrainedDecoder.text_generation`, `PretrainedBlockDecoder.block_generation` or `PretrainedPipeline.text_to_image` to build these task types.

Before `dew.pipeline(directory)` can rebuild a task, the run's configuration must be saved beside its checkpoints. A recipe's `RunConfig.train` writes `run.json` there. If you assemble the run yourself, save the matching configuration with `config.save(directory)`. The checkpoint arrays alone do not describe the model. The LM recipe records the resolved model, tokenizer, sampling value and `sample_tokens` budget. Reloading the run restores those settings.

## Weights

For plain LM, image-diffusion and block-diffusion objectives, `ema=True` selects moving-average weights. It raises an error if the run or state has no EMA copy. `ema=False` selects live weights.

`dew.pipeline`, the tasks' `from_run` and `from_pretrained`, `objective.pipeline` and `Pretrained.from_run` default to `ema=None`. This selects the EMA copy when present and live weights otherwise. DPO, GRPO and PPO use the EMA slot for a frozen reference, so their pipelines always use the trained policy and exclude the reference. PPO also excludes the critic.

`objective.pipeline(state)` selects weights by this rule and keeps the trainer's placed arrays. `LMObjective.policy(params, sampling)` returns a `TextGeneration` using the given parameter tree. GRPO rollouts sample with this task.

`task.bind(variables)` builds a task with another set of variables. It copies the mapping structure and shares the array buffers. While a task uses these arrays, do not change or donate them.

`TextGeneration.quantized(Quantization(...))` returns a task whose matching language-model weights are stored as int8 or fp8 values with scales. It uses the Qwix serving path and `dewml[quantization]` extra, as image tasks do. The processor, sampling policy and token budget stay unchanged. `KVCache.quantized` controls cache storage separately. `Server.from_task` supports both weight and cache quantization.

```python
import jax.numpy as jnp

import dew
from dew.training.quantization import Quantization

task = dew.pipeline("Qwen/Qwen3-0.6B", dtype=jnp.bfloat16)
served = task.quantized(Quantization(dtype="int8", weight_only=True))
print(served("The capital of France is", max_new_tokens=20, key=0).text[0])
```

This example downloads a Hub checkpoint. `dtype="fp8"` selects e4m3 weights; hardware support determines whether quantized operations run natively. The default example used to trace a decoder is one numeric token. For a multimodal model, pass `example=inputs`, a `ModelInputs` prepared by its processor with the media fields its weights need.

Qwix preserves the placement of resident JAX weights. For unplaced NumPy weights from a bundle, it quantizes one kernel at a time on the default device, then returns it to host storage. Temporary device storage is limited to one kernel. If that kernel will not fit, load with `mesh=` and `layout=` to place weights before quantization.

`TextToImage.quantized(Quantization(...))` uses Qwix post-training quantization to store denoiser kernels as int8 or fp8 values with scales. Install it with `pip install "dewml[quantization]"`. The encoders and autoencoder keep their weights.

`Quantization(dtype="int8")` quantizes weights and activations, so each matmul runs in int8. `weight_only=True` keeps activations in the compute dtype. This saves weight memory without the speed gain. `patterns` selects modules by path.

On the RTX 4080, the 176M text-to-image model stores 183 MiB of denoiser weights instead of 670 MiB. With bf16 computation and int8 weights and activations, its denoiser forward takes 21% less time. Its CLIP score stays within 0.002 of fp32 ([measurements](../performance.md#quantized-serving-of-the-176m-text-to-image-model-2026-09-28)). Speed depends on the device: on an A100, the quantized forward is slower. On a TPU v6e, int8 halves the fp32 forward's time but does not halve the bf16 time.

XLA:GPU computes grouped convolutions incorrectly or cannot compile them with quantized activations. Dew therefore rejects activation quantization for these convolutions on GPU. The example excludes the model's depthwise convolutions:

```python
import jax.numpy as jnp

from dew.sampling import TextToImage
from dew.training.quantization import Quantization

pipe = TextToImage.from_pretrained("dewml/hybrid-dit-176m", dtype=jnp.bfloat16,
                                   revision="32d59de89683d59824361144b87bdcaf3e742598")
served = pipe.quantized(Quantization(dtype="int8", patterns=("^(?!.*spatial_fusion).*",)))
images = served(["a red fox in a snowy forest"], steps=20, key=0).host().images
```

## Placement

Inference uses `Layout.shardings` and its replication check to place weights, as training does. Checkpoint restore uses explicit shardings for the current devices, without reusing the saved topology.

Every process supplies its own rows. Cooperating processes must agree on row counts, tokenized shapes and execution controls, including the number of continuations. They check for invalid inputs, conflicting controls and errors in prepared inputs before running any device collective.

Results contain global arrays sharded by row. They include filler rows added to make the batch divide across devices. `result.host()` returns the same record with NumPy arrays for this process's real rows. It does not gather other processes' rows.

Filler rows are added before generating continuations, so each prompt's continuations stay on the process that requested them. Token generation and image prior noise use keys per global row. Canvas refinement and an explicitly sampled VAE posterior use keys for the whole batch. Changing the placed batch shape can therefore change those draws.

### SSD-backed decoder banks

`SafetensorsBanks` and its `stream` let you run a decoder whose layer weights exceed device memory or host RAM. They read a local Hugging Face safetensors checkpoint through read-only memory maps and the ordinary decoder translator. The banked inference loop fetches each layer when it executes and keeps one host read-ahead slot for the next layer. It neither loads the full decoder stack nor captures it as a compiled constant.

```python
import jax
import jax.numpy as jnp

from dew.inference import SafetensorsBanks
from dew.interop import translate_config
from dew.nn.backbones import CausalTransformer
from dew.sampling.text import Sampling, generate

with SafetensorsBanks("path/to/gpt-oss-20b-BF16",
                       cache_bytes=0, param_dtype="auto") as source:
    fields = {**translate_config(source.config), "max_seq_len": 128, "scan_layers": True}
    model = CausalTransformer(**fields, dtype=jnp.bfloat16, attention_impl="xla")
    variables = source.stream(model)
    generated = jax.block_until_ready(generate(
        model, variables, [[1, 2, 3, 4]], max_new_tokens=8,
        key=0, sampling=Sampling(temperature=0.0)))
```

Set the path to a downloaded checkpoint snapshot's local directory. This loader accepts unquantized registered decoder families whose translation stays lazy. It rejects quantized codecs and family preparation that might materialize the model. It does not download a Hub repository, load a tokenizer or replace `dew.pipeline`'s resident loader. Tokenize inputs separately, or use these variables in a `TextGeneration` with the matching processor.

`cache_bytes` limits retained host layers. It is not a limit on the whole process. Complete layers are cached in read order while they fit and remain until the source closes. A sequential decoder revisits every layer for each token. Keeping this prefix avoids the cyclic eviction that a smaller LRU cache would cause.

Outside the cache, staging needs up to two host rows plus conversion scratch for one leaf. Embeddings, the head, the KV cache and the runtime need separate memory. Mapped pages are released after reading, but this budget does not control the kernel's shared filesystem cache.

Device storage must fit resident entries, two layer rows, activations and the KV cache. Expert tensors stream with the whole layer, including experts a token did not select. `read_ahead=False` disables the host read-ahead slot.

For a source's `place` or any other pinned-host placement, budget for allocator reserve as well as live weights, staging and runtime memory. XLA's pinned-host BFC allocator grows its regions in powers of two and retains freed chunks.

On an RTX 4080, the GPT-OSS-20B BF16 two-layer prefix held about 1.55 GB of reserve above 3.29 GB of live pinned weights. This used `place`, an eight-token context limit and `SafetensorsBanks(cache_bytes=0)`. The sharded array shares its per-device buffer, so this reserve is allocator capacity rather than another copy of the weights. You can reach a cgroup or container limit before the computed weight total. Do not assume this reserve is a fixed overhead for other models.

Streaming supports single-device inference only. It requires `scan_layers=True` and a layout without host placement. Dew rejects training, multi-device meshes and pipeline stages on this path.

Host callbacks need the CPU backend beside the accelerator. If you set platforms explicitly, use `JAX_PLATFORMS=cuda,cpu` (or `<accelerator>,cpu`) before initializing JAX. Keep the source open until all executions finish, and do not change its files in place.

The runtime-only `streaming` collection contains live callback handles, with no checkpoint weights. The streaming path saves nothing through `Checkpoints`, and its variables tree cannot be saved or exported as a resident checkpoint. The original safetensors directory remains the checkpoint.

On CPU and an RTX 4080, the tiny Mixtral checkpoint's singleton banks and four-layer scan match the all-resident model bitwise. This includes the runtime-fetched expert weights. The two-layer GPU scan uses a different RMSNorm fusion because the compiler cannot peel an effectful one-trip loop. Its logit gap stays within the elementwise spread between singleton and scanned fusions of the same resident weights. The measured maximum is 1.61e-6 at the highest matmul precision. Cached greedy tokens match.

The RTX 4080 check also used real GPT-OSS BF16 weights: the first two layers (both attention kinds), embeddings, final norm and head. At the highest matmul precision, all prefill and decode logits and greedy tokens matched the host-banked path bitwise. The check used three prompts and three decode steps.

The resident host-bank reference peaked at 7.23 GB RSS; SSD streaming peaked at 4.59 GB. A third host-resident layer exceeded the 8 GiB job budget. Each layer is 1.646 GB, plus pinned-placement and allocator overhead. This check covered the prefix only. It did not compare full-depth models.

`tools/benchmark_disk_banks.py` uses a fresh process for each cache budget. It measures peak RSS, live device allocations, allocator pool size, physical storage reads and decode throughput. Host read time includes I/O and layout conversion, which can overlap compute. Do not add it to elapsed time. An optional trace records host reads, transfers and GPU kernels. The serial probe measures a row read and its host-to-device copy separately.

The initial full-depth measurement used `unsloth/gpt-oss-20b-BF16` at revision `cc89b3e7fd423253264883a80a4fa5abc619649f`. Its weights total 41.83 GB, with 24 layers of 1.646 GB each. The run used an RTX 4080 with 16 GB VRAM, local NVMe under `/mnt/scratch` and an 8 GiB host-memory cap.

With no retained host cache and read-ahead enabled, two three-token decode intervals took 255.36 and 261.56 seconds, or 0.0116 tokens/s. Peak RSS was 4.98 GB, peak live device allocations 5.80 GB, and the peak allocator pool 8.59 GB. The two measured prefills took 64.15 and 101.45 seconds. These are shared-workstation measurements, without controlled bandwidth. Priority changed during this initial run. Subsequent large disk reads start under `ionice -c3 nice -n 19`.

Each point in the cache-size curve uses a fresh process with a four-token prompt and one measured greedy decode. Storage and computation are BF16, with the highest matmul precision. `--no-warmup` compiles both programs without an extra weight sweep because cache initialization already reads the model once. Read-ahead is enabled. At every point, peak live device storage was 5.80 GB and the allocator pool was 8.59 GB:

| Host-cache budget (GiB) | Retained layers | Decode seconds/token | Tokens/s | Peak RSS (GB) | Physical reads for measured prefill + decode (GB) |
|---|---|---|---|---|---|
| 0 | 0 | 33.59 | 0.0298 | 4.75 | 77.45 |
| 1.6 | 1 | 50.49 | 0.0198 | 6.53 | 75.79 |
| 3.2 | 2 | 64.87 | 0.0154 | 8.10 | 69.77 |

Retained layers occupy 1.646 GB each. Enlarging the cache reduced physical reads and raised RSS, without a measured throughput gain. Each measurement covers one interval on a shared workstation with a memory-limited filesystem cache. It measures capacity; it does not establish a cache speedup. All points drew the same first greedy token.

A separate zero-cache, read-ahead-disabled trace covered one prefill and one decode in 183.10 seconds. Its non-overlapping intervals were 173.10 seconds reading and converting host rows, 9.65 seconds in GPU host-to-device copies, and 0.044 seconds in device compute and layout kernels. That interval's decode took 80.21 seconds (0.0125 tokens/s), with 4.33 GB peak RSS. Host reads dominate this protocol; the trace does not establish a read-ahead speedup.

## Calls and results

A call takes `key`, either an integer seed or a JAX key; `key=n` means `jax.random.key(n)`.

| Control | Meaning |
|---|---|
| `max_new_tokens` | Continuation budget. A checkpoint can set a default in `generation_config.json`; if it declares only `max_length`, the budget is that total length minus the padded prompt width. A budget in the call wins. Without any, the call must pass one. |
| `n` | Continuations per prompt, a positive integer. The checkpoint's `num_return_sequences` becomes the default, whatever sampling policy you pass; otherwise 1. `n` in the call wins in both directions. |
| `sampling` | `Sampling`: temperature (default 1.0; 0 is argmax), top-k, top-p, min-p, typical-p, the repetition, presence and frequency penalties, `no_repeat_ngram_size`, `min_new_tokens`, `stop` strings, EOS IDs and the output padding ID. |

`task.sampling` holds the policy from the checkpoint's `generation_config.json`. To change one control while keeping the others, including the source's EOS IDs, replace it on that value:

```python
from dataclasses import replace

policy = replace(task.sampling, repetition_penalty=1.1, stop=("\n\n",))
result = task("The capital of France is", 64, sampling=policy, key=0)
```

A fresh `Sampling(...)` replaces the whole policy. EOS and padding IDs left as `None` still use the task's values because they belong to the model and tokenizer. `stop` strings require a processor and compile once against it. Controls run in Transformers' order, regardless of the order you supply them.

`logits=` overrides the full transform chain. To add your own transform to the policy's chain, use `logits=policy.transforms() + (my_transform,)`.

Building a task from source defaults raises an error if an unsupported control is active, such as DoLa or wall-clock stopping. An LM run records its `sampling` value and `sample_tokens` budget. Reloading the run restores this preview policy.

`TextToImage` stores a default solver, guidance and step count. You can override any of them in a call. `guidance=None` turns off classifier-free guidance.

Every result array has `n` rows per prompt, in prompt order. `result.rows` is this process's real prompt count times `n`. `Generation.text` returns one string per row in the same order. Each row has its own length, termination flag and likelihoods (`behavior_log_probs`, and `raw_log_probs` for the raw policy).

`Generation.text` and `CanvasGeneration.text` decode on the first read and cache the strings. A result without a processor raises an error when asked for text. You can use a tokenizer without a pad token. Dew tokenizes without padding, then pads numeric rows and masks at the `RunProcessor` boundary.

For a `TextToImage` result, `pil()` returns the real rows as a list of 8-bit PIL images. They use RGB, or grayscale for one channel, and the quantization from `dew.artifacts.uint8_pixels`. It rejects video batches and results from `decode=False`, which have no images.

Each prompt is prepared, tokenized and prefilled once for all its continuations. Its media goes through the processor once per call. Continuations then run one after another on the device, so decode time grows with `n`. They reuse cache working memory; only the output arrays grow.

Continuation zero uses the prompt's own key, so it matches the draw with `n=1`. Continuation `j` folds `j` into the key. Asking for more continuations therefore leaves the earlier draws unchanged.

Repeated calls with the same shapes and controls reuse compiled executables. Replacing a task's weights does not change the compiled model.

## Image initialization and latent handoff

`TextToImage.prepare` accepts normalized NHWC pixels or uint8 pixels through `image=`. For already-encoded images, use `image_latents=`. Shapes follow the task's `InputSpec` and autoencoder. `times=` selects an explicit starting point on the source grid. The initial state is `alpha(t) * image_latents + sigma(t) * noise`. Use `noise=` to supply the unit Gaussian noise.

A pixel mask has shape `[B, H, W, 1]`. Its masked-image conditions go to both guidance branches. `encode_key=None` uses the VAE posterior mean; an explicit key samples from the posterior. Condition encoders accept native prompt records as well as strings. `unconditional=` supplies one row, or one row per prompt, in place of the configured unconditional input.

`initial=` supplies an already-noisy latent state; Dew adds no further noise to it. Prepared inputs keep their process and concrete time grid. `task(prepared, key=..., decode=False)` skips the VAE and image checker. It returns unclipped `result.latents` and sets `result.images` to `None`.

To hand off from a base model to a refiner, pass those latents as the other task's `initial` with a matching partial grid. Normal calls return decoded images and the latents before decoding. Prepared inputs must belong to the task's mesh. Preparation on the source grid also records the step count. To change that count, prepare a new initial state.

Diffusers' `strength`, `denoising_end` and `denoising_start` select these grids. `strength=s` over `steps` starts at the point selected by Diffusers' `get_timesteps`. Set `denoising_end=f` on the base and `denoising_start=f` on the refiner to split the grid at the same point. This is where model time falls below `round(T * (1 - f))`, with `T` equal to the scheduler's `num_train_timesteps`. The base stops there, and the refiner continues from it. `tests/test_native_diffusion.py` compares both with Diffusers' pipelines:

<!-- not run: needs an SDXL base and refiner -->
```python
import dataclasses

import jax.numpy as jnp
import numpy as np

process, times = task.prepared_process(steps)
img2img = task.prepare(prompts, key=key, steps=steps, image=pixels,
                       times=times[steps - int(steps * strength):])

def split(task):
    process, times = task.prepared_process(steps)
    model_times = np.asarray(process.sampler_schedule.model_time(jnp.asarray(times[:-1])))
    return times, int(np.flatnonzero(model_times < round(T * (1 - f)))[0])

times, cut = split(base)
first = dataclasses.replace(base, final_denoise=False)
latents = first(first.prepare(prompts, key=key, steps=steps, times=times[:cut + 1]),
                key=key, decode=False).latents
times, cut = split(refiner)
images = refiner(refiner.prepare(prompts, key=key, steps=steps, initial=latents, times=times[cut:]),
                 key=key).images
```

## Serving

`Server.from_task(task, slots=, capacity=)` keeps `slots` cache rows resident, each with `capacity` positions. It admits queued requests into free rows while other rows continue decoding. In one compiled program, each step prefills newly admitted prompts and draws one token for each occupied row. A request releases its slot on EOS or at its token budget; the next request can then use it.

Each request uses its submitted key, folded as in a one-row `TextGeneration` call. It therefore draws the tokens it would draw alone.

```python
from dew.inference.serving import Server
from dew.nn.kv_cache import KVCache

greedy = dataclasses.replace(task, sampling=Sampling(temperature=0.0))
server = Server.from_task(greedy, slots=4, capacity=128,
                          kv_cache=KVCache(page_size=16, pages=32), prefix_cache=True)
prompts = ["One day, Lily saw a big dog in", "One day, Lily saw a big dog in the park."]
served = server(prompts[:1], 20, key=0) + server(prompts[1:], 20, key=0)
alone = greedy(prompts, 20, key=0)
print([generation.text[0] for generation in served])
print("same as TextGeneration:", [g.text[0] for g in served] == list(alone.text))
print("prompt tokens read from shared pages:", server.prefix_hits)
```

```text
[' the park. The dog w', ' The dog wanted to p']
same as TextGeneration: True
prompt tokens read from shared pages: 16
```

The second call's prompt begins with the first call's, so its first full 16-token page comes from the prefix cache instead of a prefill.

Calling the server with a batch submits every prompt, runs until they finish, and returns one `Generation` per prompt. `server.submit(prompt, max_new_tokens, key=...)` queues one request and returns a ticket that resolves to its `Generation`. `step()` runs one device call. `run()` continues until the queue and rows are empty.

The task's `n` must be one, and its strategy must be the row-wise sampler, with or without a grammar. `capacity` rounds up to whole 64-slot tiles and, for a paged cache, whole pages. It may not exceed the model's context.

| Option | Meaning |
|---|---|
| `slots` | Rows decoded at once. |
| `capacity` | Cache positions per row; a request whose prompt and budget do not fit is refused at `submit`. |
| `admission` | Prompts one step prefills; default the largest multiple of the mesh's row-group count up to eight, times `decode_steps`. |
| `kv_cache` | A `KVCache` layout replacing the model's own. |
| `chunk` | Prefill a long prompt in pieces of at most this many tokens, one piece a step. Needs a paged cache. |
| `prefix_cache` | Share the full prompt pages of an earlier request that began with the same tokens. Needs a paged cache. |
| `decode_steps` | Decode iterations per device call (default 1). |

`KVCache` (`dew.nn.kv_cache`) sets the cache layout. It is passed as `kv_cache=` to `from_task`, or set as a `CausalTransformer`'s `kv_cache` field for `TextGeneration`:

| Field | Meaning |
|---|---|
| `page_size` | Pages the cache. All rows share one pool, and a request is admitted once the pool holds its prompt and budget, so the pool is sized to memory rather than to `slots * capacity`. |
| `pages` | Size of the shared pool. `None` allocates one page per slot of every row, as much memory as the dense layout. Outside a server leave it unset: nothing hands out a smaller pool, and a row that would write past it fails its request. |
| `quantized` | `"int8"` stores keys and values in eight bits with one float32 scale per token and head, and rotates the keys by a Hadamard matrix first, which needs a power-of-two `head_dim`. `"float8_e4m3fn"` stores unrotated e4m3 at any `head_dim`. Either works dense or paged. |
| `groups` | Splits a paged pool into that many parts, one per equal group of rows. |

With the default dense cache, every served request draws the tokens it would draw alone. `prefix_cache` hashes each page with everything before it. `Server.reload(variables)` stops sharing pages written with the old weights.

To constrain output to a grammar, use `Sample(guided.json_schema(tokenizer, schema, eos_id))` or `Sample(guided.regex(tokenizer, pattern, eos_id))` as the task's strategy. This applies to both served and direct calls. The automaton comes from `outlines-core` (`pip install dewml[guided]`). A transform that forces a token forbidden by the grammar fails the request.

On TPU, a paged bfloat16 cache decodes with the Pallas kernel `jax.experimental.pallas.ops.tpu.paged_attention`. On supported CUDA devices, a full-precision BF16 pool with 16-token-aligned pages uses cuDNN's paged forward. The GPU path requires `'auto'` or `'cudnn'` and cuDNN-compatible heads. It accepts neither a logit softcap nor a reference-only precision request.

Both paths require one query per row and a mask containing exactly the filled slots. They support no window, sinks, image groups, pairwise mask or QK-Clip sow. Grouped pools, quantized storage and other masks use a gather followed by ordinary attention. The GPU VJP also uses this gathered path. Forward-mode attention uses the forward-mode context.

Dense storage is the default; choose paging for pooled storage and prefix sharing. The RTX 4080 measurements below use Qwen3-0.6B with BF16 storage and computation, 256-token prompts and 128 greedy output tokens with EOS ignored. Capacity is 384, with 16-token pages, admission 8 and one decode iteration per call. Warm output-token rates are medians of three runs:

| Slots | Old paged gather (tokens/s) | Native GPU paged forward (tokens/s) |
|---|---:|---:|
| 32 | 3,242 | 4,719 |
| 64 | 3,399 | 6,087 |
| 128 | 3,498 | 7,150 |

The table compares the old and new paged paths. It does not measure the default dense path. All 448 requests' 128 greedy tokens and both likelihood arrays were bit-identical before and after the kernel change. Reordered and shared page tests also preserve the old forward values and VJP exactly. They stay within the existing BF16 bound against float64 truth.

BF16 dense and paged prefill can round differently, changing greedy continuations when logits are nearly tied. Across the same 448 random-token prompts, 143 requests diverged. Replaying each dense prefix gave a maximum logit difference of 0.732 and a maximum dense top-two gap of 0.223.

Full-model float64 evaluations of these prefixes gave a paged/dense RMS rounding-error ratio of median 0.996 and maximum 1.896. This is within the existing two-times-reference bound. The maximum-error ratio was at most 1.597. By the triangle inequality, the two executions' logit difference is therefore bounded by three times dense's float64 maximum error. Every first-divergence gap and logit difference stayed below this per-row bound.

These checks found BF16 precision sensitivity, with no detected page-table or cache-position error. Because paging can change continuations, dense storage remains the default.

Verification records are in `/mnt/scratch/dew/runs/serving-attention`: `native-real.json`, `paged-old.json`, the generation archives, `paged-precision-envelope.json`, and the float64 truth chunks `fp64-chunk0.npz`, `fp64-chunk48.npz`, `fp64-chunk96.npz`. The throughput table predates the separate BF16 vocabulary-head output-rounding change; it holds that arithmetic fixed on both sides.

### Cache quality

These measurements use one A100-SXM4-40GB, bf16 weights and 51,100 wikitext-2 test tokens (100 sequences of 512). Each sequence prefills its first half, then decodes the second half one token per step through the cache. Every row is compared with the dense bf16 cache. The noise-floor row uses the same dense cache with `'xla'` attention instead of cuDNN. Greedy agreement measures 32 continuations of 256 tokens from 128-token prompts:

| Checkpoint | Cache | Perplexity | Δ perplexity | mean \|Δ log p\| | max \|Δ log p\| | Greedy identical |
|---|---|---|---|---|---|---|
| SmolLM2-135M | dense bf16 | 20.397 | | | | |
| | noise floor (xla kernel) | 20.394 | -0.003 | 0.017 | 0.59 | 20/32 |
| | paged | 20.397 | 0.000 | 0.000 | 0.00 | 32/32 |
| | int8 | 20.398 | +0.001 | 0.022 | 1.04 | 16/32 |
| | float8 | 20.442 | +0.046 | 0.058 | 1.62 | 11/32 |
| Qwen3-0.6B | dense bf16 | 27.824 | | | | |
| | noise floor (xla kernel) | 27.832 | +0.008 | 0.019 | 0.80 | 15/32 |
| | paged | 27.824 | 0.000 | 0.000 | 0.00 | 32/32 |
| | int8 | 27.832 | +0.008 | 0.052 | 4.01 | 12/32 |
| | float8 | 27.798 | -0.025 | 0.087 | 3.29 | 3/32 |

Prefer int8 for these checkpoints. It keeps perplexity within the kernel noise floor on both, while float8 does not. Paged and dense int8 give identical numbers, so the paged rows cover both. The float8 rows come from an earlier run of the same protocol, with batch 20 and no noise-floor row. They use the unrotated float8 code that ships. Without rotation, int8 added 0.665 to Qwen3-0.6B's perplexity. Rotating float8 keys added 1.03.

### Server throughput

Measured on the same A100 with a randomly initialized 12-layer, 1024-wide bf16 decoder (jax 0.11.1):

- Mixed traffic: 48 requests of 32-480 prompt tokens and 128 draws each, 16 slots of 1024. Dense ran at 6360 tokens/s on a 192 MiB cache. Paged ran at 6223 tokens/s on the same memory, and at 4767 tokens/s on a 72 MiB pool, where requests queue for pages. Over the same 192 MiB pool, 48 slots ran at 7540 tokens/s.
- int8 and float8 took 102 MiB and ran at 5123-5225 and 5034-5101 tokens/s. Over a pool of the same 205 MiB, 48 slots ran at 10716 tokens/s with int8 and 8296 with float8.
- Chunked prefill: six 1800-token prompts arrived among 16 decoding rows. `chunk=256` cut the longest step from 31.5 ms to 7.6 ms and raised throughput from 2797 to 2952 tokens/s.
- Prefix cache: 32 requests shared a 1024-token prefix. `prefix_cache` reused 16384 prompt tokens and raised throughput from 1927 to 2562 tokens/s.
- Guided decoding, with GPT-2's vocabulary and a three-field JSON schema: 16 of 16 guided rows parsed and validated, against 0 of 16 unguided. The automaton compiled in 0.74 s into 192 states by 166 token classes, and a guided step ran at 4278 against 6515 tokens/s.

### Serving on a mesh

If a task's weights are placed on a mesh, `Server.from_task` serves on that mesh. It runs one step program over all devices and places server state with `dew.nn.sharding.DEFAULT_RULES`, the same rules used for weights.

Slots split like batch rows (`activation_batch`: the data, expert and fsdp axes). Each device group keeps its own rows and dense cache, or its own part of a paged pool. A request goes to a group with room for it. With prefix caching, it goes to the group that already holds its prefix. The tensor axis splits heads, the MLP and vocabulary as in training, including the cache's heads. Slots, admission and a paged pool's page count must divide by the number of groups.

The server rejects a sequence axis above one: that axis splits sequence positions, but the decoder cache holds whole sequences. It also rejects a stage axis above one because the decoder cannot decode through the training pipeline. Meshes spanning several processes are unsupported because one process schedules the rows.

The server traces its step under the mesh. The rule table places activations, and a mixture layer runs global or exchange dispatch in a `shard_map`. JAX's checkify runs the sampler's device checks, but it cannot pass a check into a `shard_map` ([jax-ml/jax#40907](https://github.com/jax-ml/jax/issues/40907)). A server step therefore runs the model before drawing a token, which becomes input at the start of the next step.

The paged cache does not check writes on the device. Before execution, generation rejects a pool that cannot give every row its full capacity. `TextGeneration` traces without the mesh and passes each draw's checks into the next step's model call. It therefore rejects `Mixture(dispatch='exchange')`; `dispatch='global'` computes the same layer.

These measurements use jax 0.11.2 on one host with 4x RTX 3090 and PCIe 3.0. GPUs 0 and 1 are joined by NVLink; GPUs 2 and 3 share a PCIe bridge. The pairs are on different sockets.

The Qwen3-0.6B check used float32 and the highest matmul precision. It served eight prompts with 48 greedy draws each, twice, so the second wave could reuse the first wave's prefix pages. Layouts were one GPU, data=2, tensor=2, data=4, tensor=4, tensor=2 x data=2 and fsdp=2 x data=2. With dense, paged and chunked paged caches with prefix sharing, every layout drew the one-device `TextGeneration` tokens. Log-probabilities agreed within 4.0e-5. `TextGeneration` itself matched within 2.3e-5.

The one-device reference differs by 3.1e-5 from the same draws scored in fp64. Every layout therefore stayed within 1.3 times fp32's own decode rounding error. For comparison, `tools/layout_parity.py` uses a bound of four times the reference error for training steps.

With an int8 or float8 cache, reassociating an upstream sum can move a key or value near a rounding boundary by a full quantization step. Even on one GPU, serving eight rows together over the same int8 cache diverged from serving each row alone on three rows. Their fp64 logit margins were 0.02 to 0.11, with log-probabilities up to 0.10 apart before divergence. With float8, six rows diverged, with log-probabilities up to 0.21 apart. Differences across layouts were similar: 0.09 to 0.13 for int8 and 0.17 to 0.25 for float8.

A randomly initialized MoE decoder with four experts and top two served the same tokens on expert=4 and expert=2 x tensor=2. Numerical differences stayed within 1.7e-6 with all three caches. It also matched with all-to-all exchange dispatch on both layouts. On a randomly initialized two-layer decoder with one prediction depth, `TextGeneration`'s greedy and sampled speculative decoding and beam search matched the one-device draws on every layout above.

The Qwen3-1.7B throughput run used bfloat16, 256-token prompts and 128 greedy draws each. Output tokens per second at the best slot count were: one GPU 3061 (128 slots), tensor=2 on the NVLink pair 4765 (256), tensor=2 on the PCIe pair 3452 (256), data=2 on the PCIe pair 5332 (256), data=4 4859 (256), tensor=4 2719 (256), tensor=2 x data=2 6149 (512).

On these cards, host work limits decode on a tensor axis. Each step runs 57 all-reduces. XLA splits the step at those collectives into 59 command buffers. At 128 slots, this took 12.7 ms of host time and 14.5 ms of kernels per step; GPUs were busy 62% of the step.

`--xla_gpu_enable_command_buffer=+COLLECTIVES` records the all-reduces in one command buffer per step. It still re-recorded that buffer every step, taking 9.6 ms, because buffers move between steps. At 128 slots, defaults ran at 4480 tokens/s. The flag ran at 4488, the latency-hiding scheduler at 4493, no command buffers at 4394, `NCCL_PROTO=LL` at 4243, and cuBLAS in place of Triton GEMMs at 3929. Dew keeps the defaults.

`decode_steps=k` runs k decode iterations in one device call, reducing host launches to one per k tokens. Requests are admitted and draws reach the host at call boundaries. A request can therefore wait up to k iterations for a slot. The draws are the same for any k.

At 128 slots on the NVLink pair, k=4 reduced a decode iteration from 20.9 ms to 13.8 ms. GPU busy time rose from 67% to 97%. In the runs above, the 256-token prompts cost more than the 128 output draws. Overall throughput gained 1.7% at k=4 and 3.1% at k=8, while time to first token grew from 0.17 s to 0.64 s and 0.98 s. On one GPU, which was already busy throughout, k=4 lost 2.8%. The default is 1.

## Other runtimes

To serve with vLLM or Ollama, export with `Pretrained.save` and point the runtime at the directory. For a model trained in Dew, use `PretrainedDecoder.from_model` to build the export bundle. `OllamaCompletion` and `OpenAICompletion` (`dew.inference`) use those projects' official clients. Their results retain backend metadata without supplying Dew raw-policy or behavior-policy likelihoods.

llama.cpp's `convert_hf_to_gguf.py`, checked at v0.5.0, converts decoder exports to GGUF only when it recognizes the tokenizer. It reads Llama-style byte-fallback BPE directly: byte pieces `<0x00>` to `<0xFF>` in the vocabulary, with `▁` marking word starts.

For byte-level BPE, the converter recognizes only listed pre-tokenizer hashes from published models. A tokenizer trained from scratch, such as a custom tokenizer for a Dew run, therefore stops conversion with `NotImplementedError: BPE pre-tokenizer was not recognized`. For this tokenizer, use `ollama create` with `FROM <export directory>` in the Modelfile. Ollama's converter accepts it, and the result serves Dew's greedy continuation token for token.

Ollama records the pre-tokenizer as `default`. It splits runs of digits into groups of three, while a byte-level tokenizer keeps the run whole. Prompts with long numbers can therefore tokenize differently.

The converter's pinned environment, transformers 4.57.6, cannot read a `tokenizer_config.json` saved by transformers 5, which names its class `TokenizersBackend`. Run the converter with transformers 5 and `sentencepiece` installed.
