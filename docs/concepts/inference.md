# Inference

An inference task holds a Dew model, its variables, and any tokenizer or condition encoders that come with the source. You call it with inputs and a key or seed, and it generates. `dew.pipeline(source)` loads a task from a run directory, a checkpoint directory or a Hub repository. It returns `TextGeneration` for decoders, `BlockGeneration` for DiffusionGemma, `MaskedGeneration` for masked-diffusion language models such as LLaDA and Dream, and `TextToImage` for diffusion models. `dew.inference.serving.Server` serves a `TextGeneration` with continuous batching over one resident KV cache.

## Example

The example trains a small byte-level decoder for a few seconds, then generates from it in three ways: from the training state, through a server, and from an exported checkpoint loaded with `dew.pipeline`.

```python
import dataclasses

import jax
import numpy as np

from dew import Dataset, Trainer
from dew.config import OptimConfig
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
state = Trainer(objective, OptimConfig(learning_rate=3e-3), key=jax.random.key(0)).fit(
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

`objective.pipeline(state)` builds a `TextGeneration` over the trained weights without reloading a checkpoint, and `RunProcessor(tokenizer)` lets it take strings and decode its results. The call asks for 20 new tokens and `n=2` continuations per prompt, so the result has four rows: prompt zero's two continuations, then prompt one's. `result.text` has the continuations without their prompts, `lengths` counts the generated tokens, and `terminated` says whether a row stopped at EOS. The task samples at temperature 1.0, the `Sampling` default, but this decoder has memorized its one sentence, so both draws of a prompt agree.

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

A run's `run.json` must name its objective. Saved diffusion, LM, DPO, GRPO, PPO, block-diffusion and masked-diffusion runs have generation tasks. Other kinds, such as JEPA runs, have none, and loading them raises. Loading a task also points XLA at the on-disk compilation cache, so a restarted process reuses what it compiled before.

Checkpoints in a published layout load through `dew.interop.Pretrained.load`, including latent-diffusion pipelines in the diffusers layout, which a `model_index.json` describes. An original-format single-file checkpoint loads with `Pretrained.load(repo_or_dir, single_file="name.safetensors")`. diffusers' key maps convert the file once, into Dew's cache, as the diffusers pipeline it describes. The configs are `repo_or_dir`'s own when it has a `model_index.json`; otherwise they are the configs of the diffusers repo that diffusers infers from the file, pinned to the commit that was fetched. A component the file does not include, such as a Flux file's text encoders and VAE, gets its weights from the same place, and if there are none there, the load names the component and stops. The cache entry is kept only after the pipeline loads, and a newer diffusers converts the file again. FLUX.1's config repos are gated on the Hub, so accept the license and log in, or load from a local directory that holds the configs. The conversion needs `pip install 'dewml[diffusers]'`; the loaded pipeline runs in JAX.

Stable Diffusion 1.x, SDXL and FLUX.1 files are checked against diffusers' own `from_single_file`. The released SD 1.5 file and tiny SD 1.x, SDXL and Flux files convert to the same tensors as in diffusers, and the released SDXL base, FLUX.1-dev and FLUX.1-schnell files convert to the names and shapes that diffusers' models have.

A pipeline packed into one DDUF file loads with `Pretrained.load(repo_or_dir, dduf_file="name.dduf")`. huggingface_hub's own reader checks the archive and maps its entries, which are unpacked once into Dew's cache. The pipeline then loads as the directory the file packs, the same way diffusers' `from_pretrained(..., dduf_file=)` reads it. An entry whose name points outside that directory is refused. `tests/test_dduf.py` checks a packed Flux pipeline against diffusers' own DDUF load.

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

The byte vocabulary has no Hugging Face tokenizer files, so the loaded task has no processor and the example attaches one. A checkpoint that includes a tokenizer loads with its processor, and its `generation_config.json` sets the sampling policy and the token budget. A Hub name such as `"Qwen/Qwen3-0.6B"` loads the same way, as long as you have enough host and device memory for its weights and cache. If you already loaded a bundle with `Pretrained.load`, `PretrainedDecoder.text_generation`, `PretrainedBlockDecoder.block_generation` and `PretrainedPipeline.text_to_image` build the same task types from it.

`dew.pipeline(directory)` rebuilds the task from the configuration saved next to the checkpoints, because the checkpoint arrays alone do not describe the model. A recipe's `RunConfig.train` writes that configuration as `run.json`. For a run you assembled by hand, save the matching configuration yourself with `config.save(directory)`. The LM recipe records the resolved model, tokenizer, sampling value and `sample_tokens` budget, and reloading the run restores them.

## Weights

For the plain LM, image-diffusion and block-diffusion objectives, `ema=True` asks for the moving-average weights and raises if the run or state has no EMA copy, and `ema=False` reads the live weights. `dew.pipeline`, the tasks' `from_run` and `from_pretrained`, `objective.pipeline` and `Pretrained.from_run` all default to `ema=None`, which uses the EMA copy when the run or state has one and the live weights otherwise. DPO, GRPO and PPO keep a frozen reference in the EMA slot, so their pipelines always use the trained policy and never the reference. PPO's pipeline also leaves out the critic.

`objective.pipeline(state)` picks weights by the same rule and keeps the arrays where the trainer already placed them. `LMObjective.policy(params, sampling)` returns a `TextGeneration` over the given parameter tree, and GRPO rollouts sample with that task. `task.bind(variables)` makes a task over another set of variables. It copies the mapping structure but shares the array buffers, so do not change or donate those arrays while a task uses them.

`TextGeneration.quantized(Quantization(...))` returns a task that stores the language-model weights the `Quantization` matches as int8 or fp8 values with their scales. It goes through the same Qwix serving path as image tasks, needs the same `dewml[quantization]` extra, and keeps the processor, sampling policy and token budget. Weight quantization is separate from `KVCache.quantized`, which changes how the cache is stored, and `Server.from_task` can use both.

```python
import jax.numpy as jnp

import dew
from dew.training.quantization import Quantization

task = dew.pipeline("Qwen/Qwen3-0.6B", dtype=jnp.bfloat16)
served = task.quantized(Quantization(dtype="int8", weight_only=True))
print(served("The capital of France is", max_new_tokens=20, key=0).text[0])
```

This example downloads a Hub checkpoint. `dtype="fp8"` selects e4m3 weights, and whether the quantized operations run natively depends on the hardware. To trace a decoder, `quantized` uses one numeric token as its default example input. For a multimodal model, pass `example=inputs`, a `ModelInputs` that its processor prepared with the media fields its weights need.

Weights that are already JAX arrays on devices keep their placement through Qwix. Unplaced NumPy weights from a bundle are quantized one kernel at a time on the default device and returned to host storage, ready to be placed, so quantization needs temporary device storage for one kernel at most. If a single kernel will not fit, load with `mesh=` and `layout=` so the weights are placed before quantization.

`TextToImage.quantized(Quantization(...))` returns the task with its denoiser's kernels stored as int8 or fp8 values and their scales, using Qwix's post-training quantization (`pip install "dewml[quantization]"`). The encoders and the autoencoder keep their weights. `Quantization(dtype="int8")` quantizes weights and activations, so each matmul runs in int8. `weight_only=True` keeps activations in the compute dtype, which saves the weight memory but not the time. `patterns` chooses the modules by path.

On the RTX 4080, the 176M text-to-image model then holds 183 MiB of denoiser weights in place of 670 MiB. In bf16 with int8 weights and activations, its denoiser forward takes 21% less time, and its CLIP score stays within 0.002 of fp32 ([measurements](../performance.md#quantized-serving-of-the-176m-text-to-image-model-2026-09-28)). The time saved depends on the device: on an A100 the quantized forward is slower than the unquantized one, and on a TPU v6e int8 halves the fp32 forward's time but not the bf16 forward's. On a GPU, Dew refuses to quantize the activations of a grouped convolution, because XLA:GPU computes it wrongly or cannot compile it. So this example leaves the model's depthwise convolutions out:

```python
import jax.numpy as jnp

from dew.sampling import TextToImage
from dew.training.quantization import Quantization

pipe = TextToImage.from_pretrained("dewml/hybrid-dit-176m", dtype=jnp.bfloat16,
                                   revision="7187bb75a425dfb9fa0055951b8f0f7520185b87")
served = pipe.quantized(Quantization(dtype="int8", patterns=("^(?!.*spatial_fusion).*",)))
images = served(["a red fox in a snowy forest"], steps=20, key=0).host().images
```

## Placement

Inference places weights with `Layout.shardings` and its replication check, the same way the trainer does. Checkpoint restore gets explicit shardings for the current devices and does not reuse the topology that the writing process recorded.

Every process supplies its own rows. All cooperating processes must supply the same number of rows and the same tokenized shapes, and use the same execution controls, including the same number of continuations. Before any device collective runs, the processes check with each other for invalid inputs, conflicting controls and errors in prepared inputs, so an error on one process stops all of them.

Results hold global arrays sharded by row, including any filler rows that were added so the batch divides across the devices. `result.host()` returns the same record with NumPy arrays for this process's real rows; it does not gather rows from other processes. Filler rows are added as prompts, before any continuation exists, so all continuations of a prompt stay on the process that asked for them. Token generation and image prior noise use a key per global row. Canvas refinement and an explicitly sampled VAE posterior use keys for the whole batch, so changing the placed batch shape can change their draws.

### SSD-backed decoder banks

`SafetensorsBanks` and its `stream` run a decoder whose layer weights do not fit in device memory or in host RAM. The source reads a local Hugging Face safetensors checkpoint through read-only memory maps and the ordinary decoder translator. The banked inference loop then fetches each layer when it runs, with one host read-ahead slot for the next layer, so the full decoder stack is never loaded or captured as a compiled constant.

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

The path is the local directory of a downloaded checkpoint snapshot. The loader accepts unquantized registered decoder families whose translation stays lazy, and refuses a quantized codec or a family preparation step that might materialize the model. It does not download a Hub repository or load a tokenizer, and it does not replace `dew.pipeline`'s resident loader. Tokenize inputs yourself, or put these variables in a `TextGeneration` with the matching processor.

`cache_bytes` bounds the retained host layers, not the whole process. Complete layers are admitted in read order while they fit and kept until the source closes. A sequential decoder visits every layer for each token, so keeping this prefix avoids the cyclic eviction that a smaller LRU cache would go through. Beside that cache, staging needs up to two host rows plus one leaf's conversion scratch, and the embeddings, the head, the KV cache and the runtime need memory of their own. Mapped pages are released after they are read, but this budget does not control the kernel's shared filesystem cache. Device storage must fit the resident entries, two layer rows, activations and the KV cache. Expert tensors stream as part of a whole layer, not only the experts one token selected. `read_ahead=False` turns off the host read-ahead slot.

For a source's `place`, and any other pinned-host placement, budget for the allocator's reserve as well as the live weights, staging and the runtime. XLA's pinned-host BFC allocator grows its regions in powers of two and keeps freed chunks. On the RTX 4080, the GPT-OSS-20B BF16 two-layer prefix (with `place`, an eight-token context limit and `SafetensorsBanks(cache_bytes=0)`) held about 1.55 GB of reserve on top of 3.29 GB of live pinned weights. The sharded array that Dew assembles from the per-device buffer shares that buffer, so the reserve is allocator capacity and not a second copy of the weights. A cgroup or container limit can therefore be reached before the computed weight total. Do not take the observed reserve as a fixed overhead for other models.

Streaming is for single-device inference only. It requires `scan_layers=True` and a layout without host placement, and it refuses training, multi-device meshes and pipeline stages. Host callbacks need the CPU backend next to the accelerator, so if you set platforms explicitly, use `JAX_PLATFORMS=cuda,cpu` (or `<accelerator>,cpu`) before initializing JAX. Keep the source open until all executions have finished, and do not change its files in place. The runtime-only `streaming` collection holds live callback handles, not checkpoint weights. This path saves nothing through `Checkpoints`, and its variables tree is not a resident checkpoint that you can save or export; the original safetensors directory remains the checkpoint.

On CPU and an RTX 4080, the tiny Mixtral checkpoint's singleton banks and a real four-layer scan match the all-resident model bitwise, including the expert weights fetched at run time. The two-layer GPU scan fuses RMSNorm differently, and legitimately so, because a one-trip loop with effects cannot be peeled. Its logit gap stays within the elementwise spread between the singleton and scanned fusions of the same resident weights, with a measured maximum of 1.61e-6 at the highest matmul precision. The cached greedy tokens match.

I also compared the real GPT-OSS BF16 weights on the RTX 4080, using the first two layers (both attention kinds) with the real embeddings, final norm and head. For three prompts and three decode steps at the highest matmul precision, all prefill and decode logits and greedy tokens were bitwise equal to the existing host-banked path. The resident host-bank reference peaked at 7.23 GB RSS and SSD streaming at 4.59 GB. A third host-resident layer went over the 8 GiB job budget, since each layer is 1.646 GB and pinned placement and allocator overhead come on top of that. So this checks a prefix of the model; it is not a full-depth comparison against a reference.

`tools/benchmark_disk_banks.py` measures one cache budget per fresh process: peak RSS, live device allocations, allocator pool size, physical storage reads and decode throughput. Its host read time includes I/O and layout conversion and can overlap compute, so do not add it to the elapsed time. An optional trace records the host reads, transfers and GPU kernels, and the serial probe measures a row read and its host-to-device copy on their own.

The initial full-depth measurement used `unsloth/gpt-oss-20b-BF16` at revision `cc89b3e7fd423253264883a80a4fa5abc619649f`, which has 41.83 GB of weights in 24 layers of 1.646 GB each. It ran on an RTX 4080 with 16 GB VRAM, reading from local NVMe under `/mnt/scratch`, with an 8 GiB host-memory cap. With no retained host cache and read-ahead on, two three-token decode intervals took 255.36 and 261.56 seconds, or 0.0116 tokens/s, with 4.98 GB peak RSS, 5.80 GB peak live device allocations and an 8.59 GB peak allocator pool. The two measured prefills took 64.15 and 101.45 seconds. This was a shared workstation, so these are not controlled bandwidth numbers. The process priority was changed during this initial run, and later large disk reads start under `ionice -c3 nice -n 19`.

Each point of the cache-size curve below is a fresh process with a four-token prompt, one measured greedy decode, BF16 storage and compute, and the highest matmul precision. `--no-warmup` compiles both programs without an extra sweep over the weights, since cache initialization already reads the model once. Read-ahead is on. Peak live device storage stayed at 5.80 GB, and the allocator pool at 8.59 GB, for every point:

| Host-cache budget (GiB) | Retained layers | Decode seconds/token | Tokens/s | Peak RSS (GB) | Physical reads for measured prefill + decode (GB) |
|---|---|---|---|---|---|
| 0 | 0 | 33.59 | 0.0298 | 4.75 | 77.45 |
| 1.6 | 1 | 50.49 | 0.0198 | 6.53 | 75.79 |
| 3.2 | 2 | 64.87 | 0.0154 | 8.10 | 69.77 |

Retained layers take 1.646 GB each. These single-interval measurements do not show a throughput gain from a larger cache. Physical reads fall and RSS rises, but on a shared workstation with a memory-limited filesystem cache this measures capacity, not a controlled speedup from the cache. All points drew the same first greedy token.

A separate zero-cache, read-ahead-disabled trace covered one prefill and one decode in 183.10 seconds. Its non-overlapping intervals were 173.10 seconds reading and converting host rows, 9.65 seconds in GPU host-to-device copies, and 0.044 seconds in device compute and layout kernels. That interval's decode took 80.21 seconds (0.0125 tokens/s), with 4.33 GB peak RSS. Host reads dominate this protocol; the trace does not establish a read-ahead speedup.

## Calls and results

A call takes `key`, either an integer seed or a JAX key; `key=n` means `jax.random.key(n)`.

| Control | Meaning |
|---|---|
| `max_new_tokens` | Continuation budget. A checkpoint can set a default in `generation_config.json`; if it declares only `max_length`, the budget is that total length minus the padded prompt width. A budget in the call wins. Without any, the call must pass one. |
| `n` | Continuations per prompt, a positive integer. The checkpoint's `num_return_sequences` becomes the default, whatever sampling policy you pass; otherwise 1. `n` in the call wins in both directions. |
| `sampling` | `Sampling`: temperature (default 1.0; 0 is argmax), top-k, top-p, min-p, typical-p, the repetition, presence and frequency penalties, `no_repeat_ngram_size`, `min_new_tokens`, `stop` strings, EOS IDs and the output padding ID. |

`task.sampling` holds the policy that a loaded checkpoint's `generation_config.json` declares. To change one control, replace it on that value, and the others, including the source's EOS IDs, still apply:

```python
from dataclasses import replace

policy = replace(task.sampling, repetition_penalty=1.1, stop=("\n\n",))
result = task("The capital of France is", 64, sampling=policy, key=0)
```

A fresh `Sampling(...)` in a call replaces the whole policy, except that any EOS or padding ID it leaves as `None` comes from the task, because those IDs belong to the model and its tokenizer. `stop` strings are compiled once against the task's processor, so a task needs a processor to use them. The controls run in Transformers' order, whatever order you name them in. `logits=` still overrides everything; to add a transform of your own to the chain the policy compiles to, pass `logits=policy.transforms() + (my_transform,)`.

Building a task from a source's defaults raises if an unsupported control is active, such as DoLa or wall-clock stopping. An LM run records its `sampling` value and `sample_tokens` budget, and reloading the run keeps that preview policy. `TextToImage` has a default solver, guidance and step count. A call can override any of them, and `guidance=None` turns off classifier-free guidance.

Every result array has `n` rows per prompt, in prompt order, so `result.rows` is the number of this process's real prompts times `n`. `Generation.text` returns one string per row in the same order, and each row has its own length, termination flag and likelihoods (`behavior_log_probs`, and `raw_log_probs` for the raw policy). `Generation.text` and `CanvasGeneration.text` decode the first time they are read and cache the strings; a result with no processor raises when asked for text. A tokenizer without a pad token works as it is, because Dew tokenizes without padding and then pads the numeric rows and masks at the boundary that `RunProcessor` uses.

A `TextToImage` result's `pil()` returns the same real rows of an image batch as a list of 8-bit PIL images (RGB, or grayscale for one channel), converted to 8 bits the way `dew.artifacts.uint8_pixels` converts them. It refuses a video batch, and a result from `decode=False`, which has no images.

Each prompt is prepared, tokenized and prefilled once for all its continuations, and its media goes through the processor once per call. The continuations then run one after another on the device, so decode time grows with `n`. Each continuation reuses the cache working memory of the one before it, so only the output arrays grow. Continuation zero draws with the prompt's own key, so `n=1` and continuation zero of a larger request are the same draw. Continuation `j` folds `j` into the key, so asking for more continuations does not change the ones already drawn.

Repeated calls with the same shapes and controls reuse the compiled executables. Swapping in new weights with `bind` does not change what the model compiles to.

## Image initialization and latent handoff

`TextToImage.prepare` accepts normalized NHWC pixels or uint8 pixels through `image=`, and already-encoded images through `image_latents=`. Shapes follow the task's `InputSpec` and autoencoder. `times=` picks an explicit starting point on the source grid. The initial state is `alpha(t) * image_latents + sigma(t) * noise`, and `noise=` supplies the unit Gaussian noise.

A pixel mask has shape `[B, H, W, 1]`. Its masked-image conditions go to both guidance branches. `encode_key=None` uses the VAE posterior mean; an explicit key samples from the posterior. Condition encoders accept native prompt records as well as strings. `unconditional=` supplies one row, or one row per prompt, in place of the configured unconditional input.

`initial=` takes a latent state that is already noisy, and Dew does not add noise to it again. Prepared inputs keep their process and their concrete time grid. `task(prepared, key=..., decode=False)` skips the VAE and the image checker and returns unclipped `result.latents`, with `result.images` set to `None`. To hand off from a base model to a refiner, pass those latents to the refiner as `initial` with the matching partial grid. Normal calls return both the decoded images and the latents before decoding. Prepared inputs must belong to the task's mesh. A preparation on the source grid also records its step count, so to change the count, prepare a new initial state.

Diffusers' `strength`, `denoising_end` and `denoising_start` each choose a partial grid in this way. `strength=s` over `steps` starts where Diffusers' `get_timesteps` starts. `denoising_end=f` on a base and `denoising_start=f` on its refiner split the grid where the model time falls below `round(T * (1 - f))`, with `T` the scheduler's `num_train_timesteps`. The base runs to that point and stops, and the refiner starts from it. `tests/test_native_diffusion.py` checks both against Diffusers' own pipelines:

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

`Server.from_task(task, slots=, capacity=)` keeps `slots` cache rows of `capacity` positions resident, and it admits queued requests into free rows while the other rows keep decoding. Each step is one compiled program that prefills the prompts admitted in that step and draws one token for every occupied row. A request leaves its row at EOS or when its budget runs out, and the next request takes the slot. A served request draws with the key it was submitted with, folded the way a one-row `TextGeneration` call folds it, so it draws the tokens it would draw alone.

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

Calling the server with a batch submits every prompt, steps until they all finish and returns one `Generation` per prompt. `server.submit(prompt, max_new_tokens, key=...)` queues one request and returns a ticket that resolves to its `Generation`. `step()` runs one device call, and `run()` steps until the queue and the rows are empty. The task's `n` must be one, and its strategy must be the row-wise sampler, with or without a grammar. `capacity` is rounded up to whole 64-slot tiles (and to whole pages, for a paged cache), and it may not exceed the model's context.

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

With the default dense cache, every served request draws the tokens it would draw alone. `prefix_cache` hashes each page together with everything before it, and after `Server.reload(variables)` the server stops sharing the pages that the old weights wrote. To keep every draw inside a grammar, served or not, use `Sample(guided.json_schema(tokenizer, schema, eos_id))` or `Sample(guided.regex(tokenizer, pattern, eos_id))` as the task's strategy. The automaton comes from `outlines-core` (`pip install dewml[guided]`), and a transform that forces a token the grammar forbids fails the request.

On TPU, a paged bfloat16 cache decodes through the Pallas kernel `jax.experimental.pallas.ops.tpu.paged_attention`. On supported CUDA devices, an unquantized BF16 pool with 16-token-aligned pages uses cuDNN's own paged forward. The GPU path needs `'auto'` or `'cudnn'` attention, cuDNN-compatible heads, no logit softcap and no request for reference-only precision. Both paths need one query per row, with a mask that is exactly the filled slots and no window, sinks, image groups, pairwise mask or QK-Clip sow. Grouped pools, quantized storage and other masks keep the existing path, which gathers the pages and runs ordinary attention. The GPU VJP also uses that gathered path, and forward-mode attention uses the existing forward-mode context.

Dense storage remains the default, and you opt into paging for pooled storage and prefix sharing. On an RTX 4080, Qwen3-0.6B with BF16 storage and compute gave these warm output-token rates (medians of three runs), with 256-token prompts, 128 greedy output tokens with EOS ignored, capacity 384, 16-token pages, admission 8 and one decode iteration per call:

| Slots | Old paged gather (tokens/s) | Native GPU paged forward (tokens/s) |
|---|---:|---:|
| 32 | 3,242 | 4,719 |
| 64 | 3,399 | 6,087 |
| 128 | 3,498 | 7,150 |

These numbers compare the old and new paged paths; the default dense path is not in the table. All 448 requests' 128 greedy tokens and both likelihood arrays were bit-identical before and after the kernel change. Tests with reordered and shared pages also reproduce the old forward values and VJP exactly, within the existing BF16 bound against float64 truth.

BF16 dense and paged prefill can round differently, and that can change a greedy continuation where the top logits are nearly tied. Across the same 448 random-token prompts, 143 requests diverged. Replaying each dense prefix gave a maximum logit difference of 0.732 and a maximum dense top-two gap of 0.223. Full-model float64 evaluations of those prefixes put the ratio of paged to dense RMS rounding error at a median of 0.996 and a maximum of 1.896, within the existing bound of two times the reference, and the ratio of maximum errors was at most 1.597. By the triangle inequality, the two executions' logit difference is then at most three times dense's float64 maximum error, and every first-divergence gap and logit difference was below that per-row bound. So the divergence comes from BF16 precision, and no page-table or cache-position error was found. It is also why paging does not replace the dense default.

Verification records are in `/mnt/scratch/dew/runs/serving-attention`: `native-real.json`, `paged-old.json`, the generation archives, `paged-precision-envelope.json`, and the float64 truth chunks `fp64-chunk0.npz`, `fp64-chunk48.npz`, `fp64-chunk96.npz`. The throughput table predates the separate BF16 vocabulary-head output-rounding change; it holds that arithmetic fixed on both sides.

### Cache quality

I measured this on one A100-SXM4-40GB with bf16 weights over 51,100 wikitext-2 test tokens (100 sequences of 512). Each sequence prefills its first half and then decodes the second half one token per step through the cache, and every row is compared with the dense bf16 cache. The noise-floor row is the same dense cache under the `'xla'` attention kernel instead of cuDNN. Greedy agreement is over 32 continuations of 256 tokens from 128-token prompts:

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

On both checkpoints, int8 keeps perplexity within the kernel noise floor and float8 does not, so int8 is the quantized format to prefer. Paged and dense int8 give identical numbers, so the paged rows cover both. The float8 rows come from an earlier run of the same protocol (batch 20, no noise-floor row), over the unrotated float8 code that ships. Without the rotation, int8 cost Qwen3-0.6B +0.665 perplexity, and rotating the float8 keys cost it +1.03.

### Server throughput

Measured on the same A100 with a randomly initialized 12-layer, 1024-wide bf16 decoder (jax 0.11.1):

- Mixed traffic: 48 requests of 32-480 prompt tokens and 128 draws each, 16 slots of 1024. Dense ran at 6360 tokens/s on a 192 MiB cache. Paged ran at 6223 tokens/s on the same memory, and at 4767 tokens/s on a 72 MiB pool, where requests queue for pages. Over the same 192 MiB pool, 48 slots ran at 7540 tokens/s.
- int8 and float8 took 102 MiB and ran at 5123-5225 and 5034-5101 tokens/s. Over a pool of the same 205 MiB, 48 slots ran at 10716 tokens/s with int8 and 8296 with float8.
- Chunked prefill: six 1800-token prompts arrived among 16 decoding rows. `chunk=256` cut the longest step from 31.5 ms to 7.6 ms and raised throughput from 2797 to 2952 tokens/s.
- Prefix cache: 32 requests shared a 1024-token prefix. `prefix_cache` reused 16384 prompt tokens and raised throughput from 1927 to 2562 tokens/s.
- Guided decoding, with GPT-2's vocabulary and a three-field JSON schema: 16 of 16 guided rows parsed and validated, against 0 of 16 unguided. The automaton compiled in 0.74 s into 192 states by 166 token classes, and a guided step ran at 4278 against 6515 tokens/s.

### Serving on a mesh

A server runs on the mesh its task's weights are placed on. `Server.from_task` over such a task runs one step program over every device, and it places its own state with the same rule table that places the weights (`dew.nn.sharding.DEFAULT_RULES`). The slots are split the way a batch's rows are (`activation_batch`: the data, expert and fsdp axes), so each group of devices keeps its own rows and their dense cache, or its own part of a paged pool. A request is seated in a group that has room for it and, with a prefix cache, in the group that already holds its prefix. The tensor axis splits the heads, the MLP and the vocabulary as in training, and the cache's heads are split over it too. The number of slots, the admission and a paged pool's page count must all be multiples of the number of groups.

The server refuses a mesh whose sequence axis is larger than one, because the decoder refuses to decode under such an axis: the axis splits a sequence's positions, and the cache holds whole sequences. It refuses a stage axis larger than one because the decoder refuses to decode through the training pipeline, and a mesh over several processes because one process schedules all the rows.

The server traces its step under the mesh, so the rule table places the activations, and a mixture layer runs its dispatch, global or exchange, in a `shard_map`. The sampler's device checks go through JAX's checkify, which cannot pass a check into a `shard_map` ([jax-ml/jax#40907](https://github.com/jax-ml/jax/issues/40907)). So a server step runs the model before it draws, and the token a row draws is fed in at the start of the next step. For the same reason the paged cache checks no write on the device; generation refuses, before anything runs, a pool that cannot give every row its capacity. `TextGeneration` traces without the mesh and passes each draw's checks on to the next step's model call, so it refuses `Mixture(dispatch='exchange')`; `dispatch='global'` computes the same layer.

Measured on 4x RTX 3090 (PCIe 3.0, one host: GPUs 0 and 1 joined by NVLink, 2 and 3 under one PCIe bridge, the pairs across sockets), jax 0.11.2:

- Qwen3-0.6B at float32 and the highest matmul precision, eight prompts of 48 greedy draws each, served twice so the second wave reads the first wave's prefix pages. On one GPU, data=2, tensor=2, data=4, tensor=4, tensor=2 x data=2 and fsdp=2 x data=2, the dense cache, a paged cache and a chunked paged cache with prefix sharing all drew the tokens that `TextGeneration` draws on one device, with log-probabilities within 4.0e-5; `TextGeneration` itself matched within 2.3e-5. The one-device reference is 3.1e-5 from the same draws scored in fp64, so every layout stayed within 1.3 times fp32's own rounding of the decode, where `tools/layout_parity.py` allows a training step four times that rounding. An int8 or float8 cache is only as exact as its rounding, because a sum reassociated upstream moves a key or value that sits near a rounding boundary by a whole quantization step. On one GPU, serving the eight rows together and serving each row alone over the same int8 cache already diverged on three rows, at fp64 logit margins of 0.02 to 0.11, with log-probabilities up to 0.10 apart before they diverged (float8: six rows, 0.21). Every layout's spread was of that size, 0.09 to 0.13 for int8 and 0.17 to 0.25 for float8. A randomly initialized mixture-of-experts decoder (four experts, top two) served token for token, within 1.7e-6, on expert=4 and expert=2 x tensor=2 with all three caches, and also with the exchange dispatch, an all-to-all, on those two layouts. On a randomly initialized two-layer decoder with one prediction depth, `TextGeneration`'s greedy and sampled speculative decoding and its beam search drew what they draw on one device, on every layout above.
- Qwen3-1.7B in bfloat16, 256-token prompts, 128 greedy draws each, output tokens per second at the best slot count: one GPU 3061 (128 slots), tensor=2 on the NVLink pair 4765 (256), tensor=2 on the PCIe pair 3452 (256), data=2 on the PCIe pair 5332 (256), data=4 4859 (256), tensor=4 2719 (256), tensor=2 x data=2 6149 (512).

Decode on a tensor axis is bound by the host on these cards. Each step runs 57 all-reduces, and XLA launches the step as 59 command buffers split at them. At 128 slots that is 12.7 ms of host time per step for 14.5 ms of kernels, so the GPUs were busy 62% of the step. Recording the all-reduces into the command buffers (`--xla_gpu_enable_command_buffer=+COLLECTIVES`) left one command buffer per step, but XLA re-recorded it every step (9.6 ms), because the step's buffers move between steps. At 128 slots the defaults ran at 4480 tokens/s, that flag at 4488, the latency-hiding scheduler at 4493, no command buffers at 4394, `NCCL_PROTO=LL` at 4243 and cuBLAS in place of Triton GEMMs at 3929, so the defaults stay.

`decode_steps=k` runs k decode iterations in one device call, so the host launches the step once per k tokens. Requests are seated, and draws reach the host, only at call boundaries, so a request waits up to k iterations for its slot; the draws are the same for any k. At 128 slots on the NVLink pair, k=4 cut a decode iteration from 20.9 ms to 13.8 ms and raised the GPUs' busy share from 67% to 97%. The runs above, whose 256-token prompts cost more than their 128 draws, gained 1.7% at k=4 and 3.1% at k=8, while the time to first token grew from 0.17 s to 0.64 s and 0.98 s. On one GPU, which was already busy throughout, k=4 lost 2.8%. So the default stays 1.

## Other runtimes

To serve with vLLM or Ollama, export with `Pretrained.save` (for a model you trained, through `PretrainedDecoder.from_model`) and point the runtime at the directory. `OllamaCompletion` and `OpenAICompletion` (`dew.inference`) call those projects' official clients. Their results keep the backend's metadata, and they do not make up the raw-policy or behavior-policy likelihoods that Dew's own tasks compute.

Whether llama.cpp can make a GGUF file from an export depends on the tokenizer. llama.cpp's `convert_hf_to_gguf.py` (checked at v0.5.0) converts a decoder export whose tokenizer it recognizes. It reads a Llama-style byte-fallback BPE (byte pieces `<0x00>` to `<0xFF>` in the vocabulary, with `▁` marking word starts) directly. It recognizes a byte-level BPE only by its pre-tokenizer's hash, and it lists only published models' hashes, so any byte-level BPE trained from scratch, such as a custom tokenizer for a Dew run, stops the converter with `NotImplementedError: BPE pre-tokenizer was not recognized`. The converter's own pinned environment (transformers 4.57.6) also cannot read a `tokenizer_config.json` saved by transformers 5, which names its class `TokenizersBackend`, so run the converter with transformers 5 and `sentencepiece` installed.

On the CPU, llama.cpp's F32 logits are closer to exact than Dew's, torch's or a BLAS's. The difference is how each one sums a dot product: ggml's `vec_dot_f32` keeps 32 partial sums (four AVX2 registers of eight lanes), while XLA's CPU dot, OpenBLAS and torch each sum one chain of products per output. I measured one fp32 dot against float64, as the RMS error over the RMS result in units of fp32 rounding. XLA's error is 2.4 at K=64, 4.9 at K=256, 6.8 at K=1024 and 7.0 at K=4096, and the 32-lane sum's is 1.8, 2.1, 2.7 and 3.9, so 1.3 to 2.5 times smaller. On the tiny Llama in `tests/test_llama_cpp_export.py`, Dew's logits are 1.74 times as far from float64 as llama.cpp's, and 0.75 times as far when Dew's dots run in float64. No XLA:CPU flag in jaxlib 0.11 changes this (XNNPACK, oneDNN, single-threaded Eigen, strict dot math); [openxla/xla#50060](https://github.com/openxla/xla/issues/50060) asks for lane-blocked accumulation.

For a byte-level tokenizer trained from scratch, use Ollama. `ollama create` with `FROM <export directory>` in the Modelfile runs Ollama's own converter, which accepts the tokenizer, and the result serves Dew's greedy continuation token for token. Ollama records the pre-tokenizer as `default`, which splits a run of digits into groups of three where a byte-level tokenizer keeps the run whole, so a prompt with long numbers can tokenize differently there.
