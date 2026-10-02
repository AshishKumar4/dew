# Inference

An inference task holds a native model, its variables, and any tokenizer or condition encoders the source provides; calling it with inputs and a key or seed generates. `dew.pipeline(source)` loads a task from a run directory, a checkpoint directory or a Hub repository. It returns `TextGeneration` for decoders, `BlockGeneration` for DiffusionGemma, `MaskedGeneration` for masked-diffusion language models such as LLaDA and Dream, and `TextToImage` for diffusion models. `dew.inference.serving.Server` serves a `TextGeneration` with continuous batching over one resident KV cache.

## Example

The example trains a small byte-level decoder for a few seconds, then generates from it in three ways: from the training state, through a server, and from an exported checkpoint loaded with `dew.pipeline`.

```python
import dataclasses

import jax
import numpy as np
import optax

from dew import Dataset, Trainer, models
from dew.data import ByteTokenizer
from dew.inference import RunProcessor
from dew.objectives.lm import LMObjective
from dew.sampling import Sampling

tokenizer = ByteTokenizer()
ids = np.array(tokenizer.encode("One day, Lily saw a big dog in the park. "
                                "The dog wanted to play. " * 40), dtype=np.int32)
rows = np.stack([ids[i:i + 65] for i in range(0, 16 * 64, 64)])
data = Dataset(train=lambda partition: iter([{"text": rows}] * 200), val=None,
               records=16, batch=16)
model = models.build("causal_transformer", vocab_size=tokenizer.vocab_size,
                     emb_features=64, num_layers=2, num_heads=2, mlp_features=256,
                     max_seq_len=128)
objective = LMObjective(model, seq_len=64, ema_decay=None)
state = Trainer(objective, optax.adamw(3e-3), key=jax.random.key(0)).fit(
    data, steps=150, log_every=150)

task = objective.pipeline(state, ema=False, processor=RunProcessor(tokenizer))
result = task(["One day", "The dog"], 20, seed=0, n=2)
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

`objective.pipeline(state)` binds the trained weights to a `TextGeneration` without reloading a checkpoint. `RunProcessor(tokenizer)` lets it take strings and decode results. The call asks for 20 new tokens and `n=2` continuations per prompt, so the result has four rows: prompt zero's continuations first, then prompt one's. `result.text` holds the continuations without their prompts, `lengths` counts the generated tokens, and `terminated` says whether a row stopped at EOS. The task samples at temperature 1.0 (the `Sampling` default); this decoder has memorized its one sentence, so both draws of a prompt agree.

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

A run's `run.json` must name its objective. Saved diffusion, LM, DPO, GRPO, PPO, block-diffusion and masked-diffusion runs have generation tasks. Other kinds, such as JEPA runs, have none and raise. Loading a task also points XLA at the on-disk compilation cache, so a restarted process reuses what it compiled.

Checkpoints in a published layout load through `dew.interop.load_pretrained`, including native latent-diffusion checkpoints described by `model_index.json`. An original-format single-file checkpoint loads with `load_pretrained(repo_or_dir, single_file="name.safetensors")`. Stable Diffusion 1.x, SDXL and FLUX.1 files are checked against diffusers' own `from_single_file`: the released SD 1.5 file and tiny SD 1.x, SDXL and Flux files convert to the same tensors, and the released SDXL base, FLUX.1-dev and FLUX.1-schnell files convert to the names and shapes diffusers' models have. diffusers' key maps convert the file once into Dew's cache as the diffusers pipeline it describes. The configs are `repo_or_dir`'s own when it has a `model_index.json`, or else those of the diffusers repo diffusers infers from the file, pinned to the commit fetched. A component the file does not carry, such as a Flux file's text encoders and VAE, takes its weights from the same place; if there are none, the load names the component and stops. The cache entry is kept only after the pipeline loads, and a newer diffusers converts the file again. FLUX.1's config repos are gated on the Hub, so accept the license and log in, or load from a local directory that holds the configs. The conversion needs `pip install 'dewml[diffusers]'`; the loaded pipeline runs in JAX.

This exports the decoder from the example in the Hugging Face layout and loads it back:

```python
import dew
from dew.interop import save_pretrained_decoder
from dew.training import MeshSpec

save_pretrained_decoder(model, state.params, "lily-decoder", tokenizer="byte")
loaded = dew.pipeline("lily-decoder", mesh=MeshSpec(), dtype="float32")
loaded = dataclasses.replace(loaded, processor=RunProcessor(tokenizer),
                             sampling=Sampling(temperature=0.0))
print(loaded("One day", 20, seed=0).text)
```

```text
(', Lily saw a big dog',)
```

The byte vocabulary has no Hugging Face tokenizer files, so the loaded task has no processor, and the example attaches one. A checkpoint that ships a tokenizer loads with its processor, and its `generation_config.json` sets the sampling policy and budget. A Hub name such as `"Qwen/Qwen3-0.6B"` loads the same way and needs enough host and device memory for its weights and cache. `Pretrained.text_generation`, `block_generation` and `text_to_image` build the same task types from a bundle already loaded with `load_pretrained`.

A run assembled by hand needs its configuration saved next to the checkpoints before `dew.pipeline(directory)` can rebuild the task. A recipe's `RunConfig.train` writes `run.json` there; otherwise save the matching configuration with `config.save(directory)`. The checkpoint arrays alone do not describe the model. The LM recipe records the resolved model, tokenizer, sampling value and `sample_tokens` budget, and reloading the run restores them.

## Weights

For the plain LM, image-diffusion and block-diffusion objectives, `ema=True` asks for the moving-average weights and raises if the run or state has no EMA copy, and `ema=False` reads the live weights. `dew.pipeline` and the tasks' `from_run` and `from_pretrained` default to `ema=None`, which takes the EMA copy when the run stored one and the live weights otherwise; `objective.pipeline` defaults to `ema=True`. DPO, GRPO and PPO use the EMA slot for a frozen reference, so their pipelines always publish the trained policy, never the reference. PPO also leaves out the critic.

`objective.pipeline(state)` picks weights by the same rule and keeps the arrays the trainer has already placed. `LMObjective.policy(params, sampling)` returns a `TextGeneration` bound to the given tree, the task a GRPO rollout samples with. `task.bind(variables)` makes a task over another set of variables; it copies the mapping structure and shares the array buffers, so do not change or donate those arrays while a task uses them.

`TextGeneration.quantized(Quantization(...))` returns a task with matched language-model weights stored as int8 or fp8 values with their scales. It uses the same Qwix serving path and `dewml[quantization]` extra as image tasks, and keeps the processor, sampling policy and token budget. Weight quantization is separate from `KVCache.quantized`, which changes cache storage. Both can be used by `Server.from_task`.

```python
from dew.training.quantization import Quantization

task = dew.pipeline("Qwen/Qwen3-0.6B", dtype="bfloat16")
served = task.quantized(Quantization(dtype="int8", weight_only=True))
print(served("The capital of France is", max_new_tokens=20, seed=0).text[0])
```

This example downloads a Hub checkpoint. `dtype="fp8"` selects e4m3 weights; hardware support determines whether quantized operations run natively. The default example used to trace a decoder is one numeric token. For a multimodal model, pass `example=inputs`, a `ModelInputs` prepared by its processor with the media fields its weights need.

Resident JAX weights retain their placement through Qwix. Unplaced NumPy weights from a bundle quantize one kernel at a time on the default device and return to host storage, ready for placement. Temporary device storage is bounded by one kernel; if a single kernel will not fit, load with `mesh=` and `layout=` to place weights before quantization.

`TextToImage.quantized(Quantization(...))` returns the task with its denoiser's kernels stored as int8 or fp8 values and their scales, through Qwix's post-training quantization (`pip install "dewml[quantization]"`). The encoders and the autoencoder keep their weights. `Quantization(dtype="int8")` quantizes weights and activations, so each matmul runs in int8; `weight_only=True` keeps activations in the compute dtype, which saves the weight memory without the speed. `patterns` chooses the modules by path. On the RTX 4080 the 176M text-to-image model then holds 183 MiB of denoiser weights in place of 670 MiB, and in bf16 with int8 weights and activations its denoiser forward takes 21% less time, with its CLIP score within 0.002 of fp32 ([measurements](../performance.md#quantized-serving-of-the-176m-text-to-image-model-2026-09-28)). The time it saves depends on the device: on an A100 the quantized forward is slower than the unquantized one, and on a TPU v6e int8 halves the fp32 forward's time and not the bf16 one's. On a GPU, Dew refuses to quantize the activations of a grouped convolution, which XLA:GPU computes wrongly or cannot compile, so there the model's depthwise convolutions stay out:

```python
from dew.sampling import TextToImage
from dew.training.quantization import Quantization

pipe = TextToImage.from_pretrained("dewml/hybrid-dit-176m", dtype="bfloat16")
served = pipe.quantized(Quantization(dtype="int8", patterns=("^(?!.*spatial_fusion).*",)))
images = served(["a red fox in a snowy forest"], steps=20, seed=0).host().images
```

## Placement

Weights are placed with `Layout.shardings` and its replication check, as the trainer places them. Checkpoint restore gets explicit shardings for the current devices; it does not reuse the topology the writer recorded.

Every process supplies its own rows. All cooperating processes must supply the same number of rows and the same tokenized shapes, and use the same execution controls, including the same number of continuations. The processes agree on invalid inputs, conflicting controls and errors in prepared inputs before any device collective runs.

Results hold global arrays sharded by row, including any filler rows added so the batch divides across devices. `result.host()` returns the same record with NumPy arrays for this process's real rows; it does not gather rows from other processes. Filler rows are added as prompts, before any continuation exists, so all continuations of a prompt stay on the process that asked for them. Token generation and image prior noise use keys per global row. Canvas refinement and an explicitly sampled VAE posterior use keys for the whole batch, so changing the placed batch shape can change their draws.

### SSD-backed decoder banks

`SafetensorsBanks` and `stream_banked` run a decoder whose layer weights do not fit in device memory or host RAM. The source reads a local Hugging Face safetensors checkpoint through read-only memory maps and the ordinary decoder translator. The existing banked inference loop fetches a layer at execution, with one host read-ahead slot for the following layer. No full decoder stack is loaded or captured as a compiled constant.

```python
import jax

from dew import models
from dew.inference import SafetensorsBanks, stream_banked
from dew.interop.hf_decoders import translate_config
from dew.registry import with_precision
from dew.sampling.text import Sampling, generate

with SafetensorsBanks("path/to/gpt-oss-20b-BF16",
                       cache_bytes=0, param_dtype="auto") as source:
    record = translate_config(source.config)
    record["max_seq_len"] = 128
    model = models.build("causal_transformer", {
        **with_precision("causal_transformer", record,
                         dtype="bfloat16", attention_impl="xla"),
        "scan_layers": True,
    })
    variables = stream_banked(model, source)
    generated = jax.block_until_ready(generate(
        model, variables, [[1, 2, 3, 4]], max_new_tokens=8,
        seed=0, sampling=Sampling(temperature=0.0)))
```

Use a downloaded checkpoint snapshot's local directory for the path. This path accepts unquantized registered decoder families whose translation stays lazy; it refuses a quantized codec or family preparation that might materialize the model. It does not download a Hub repository, load a tokenizer or replace `dew.pipeline`'s resident loader. Tokenize inputs separately, or bind these variables to `TextGeneration` with the matching processor.

`cache_bytes` bounds retained host layers, not the entire process. Complete layers are admitted in read order while they fit and kept until the source closes. A sequential decoder revisits every layer each token, so retaining this prefix avoids the cyclic eviction of a smaller LRU cache. Staging needs up to two host rows plus one leaf's conversion scratch beside that cache. Embeddings, the head, the KV cache and the runtime are separate. Read mapped pages are released; the kernel's shared filesystem cache is not controlled by this budget. Device storage must fit resident entries, two layer rows, activations and the KV cache. Expert tensors stream as part of a whole layer, not just the experts selected for one token. `read_ahead=False` disables the host read-ahead slot.

Streaming is single-device inference only. It requires `scan_layers=True` and a layout without host placement; training, multi-device meshes and pipeline stages are refused. Host callbacks need the CPU backend beside the accelerator: when setting platforms explicitly, use `JAX_PLATFORMS=cuda,cpu` (or `<accelerator>,cpu`) before initializing JAX. Keep the source open until all executions have finished and do not change its files in place. The runtime-only `streaming` collection holds live callback handles, not checkpoint weights: this path saves nothing through `Checkpoints`, and its variables tree is not a resident checkpoint to save or export. The original safetensors directory remains the checkpoint.

On CPU and an RTX 4080, the tiny Mixtral checkpoint's singleton banks and a genuine four-layer scan match the all-resident model bitwise, including the runtime-fetched expert weights. The two-layer GPU scan has a different legitimate RMSNorm fusion because an effectful one-trip loop cannot be peeled; its logit gap stays within the elementwise spread of the same resident weights' singleton and scanned fusions, with a measured maximum of 1.61e-6 at the highest matmul precision. Cached greedy tokens match.

The real GPT-OSS BF16 weights were also compared on the RTX 4080 using the first two layers (both attention kinds), real embeddings, final norm and head: all prefill and decode logits and greedy tokens were bitwise equal to the existing host-banked path for three prompts and three decode steps, at the highest matmul precision. The resident host-bank reference peaked at 7.23 GB RSS and SSD streaming at 4.59 GB. A third host-resident layer exceeded the 8 GiB job budget; each layer is 1.646 GB, and pinned placement and allocator overhead are additional to that total. This was a prefix check, not a claimed full-depth reference comparison.

`tools/benchmark_disk_banks.py` measures one cache budget per fresh process, including peak RSS, live device allocations, allocator pool size, physical storage reads and decode throughput. Its host read time includes I/O and layout conversion and can overlap compute, so it must not be added to elapsed time. An optional trace records the host reads, transfers and GPU kernels; the serial probe separately measures a row read and its host-to-device copy.

The initial full-depth measurement used `unsloth/gpt-oss-20b-BF16` at revision `cc89b3e7fd423253264883a80a4fa5abc619649f`: 41.83 GB of weights, 24 layers of 1.646 GB each, on an RTX 4080 with 16 GB VRAM, from local NVMe under `/mnt/scratch`, with an 8 GiB host-memory cap. With no retained host cache and read-ahead enabled, two three-token decode intervals took 255.36 and 261.56 seconds: 0.0116 tokens/s, with 4.98 GB peak RSS, 5.80 GB peak live device allocations and an 8.59 GB peak allocator pool. This was a shared workstation, not a controlled bandwidth comparison; the two measured prefills took 64.15 and 101.45 seconds. Priority was changed during this initial run; subsequent large disk reads start under `ionice -c3 nice -n 19`.

The cache-size curve below uses a separate fresh process per point, a four-token prompt, one measured greedy decode, BF16 storage and compute, and the highest matmul precision. `--no-warmup` compiles both programs without an extra weight sweep; cache initialization already reads the model once. Read-ahead is enabled. Peak live device storage stayed at 5.80 GB and the allocator pool at 8.59 GB for every point:

| Host-cache budget (GiB) | Retained layers | Decode seconds/token | Tokens/s | Peak RSS (GB) | Physical reads for measured prefill + decode (GB) |
|---|---|---|---|---|---|
| 0 | 0 | 33.59 | 0.0298 | 4.75 | 77.45 |
| 1.6 | 1 | 50.49 | 0.0198 | 6.53 | 75.79 |
| 3.2 | 2 | 64.87 | 0.0154 | 8.10 | 69.77 |

Retained layers occupy 1.646 GB each. These single-interval measurements do not show a throughput gain from enlarging the cache: physical reads fall and RSS rises, but the shared workstation and its memory-limited filesystem cache make this a capacity measurement, not a controlled cache speedup. All points drew the same first greedy token.

A separate zero-cache, read-ahead-disabled trace covered one prefill and one decode in 183.10 seconds. Its non-overlapping intervals were 173.10 seconds reading and converting host rows, 9.65 seconds in GPU host-to-device copies, and 0.044 seconds in device compute and layout kernels. That interval's decode took 80.21 seconds (0.0125 tokens/s), with 4.33 GB peak RSS. Host reads dominate this protocol; the trace does not establish a read-ahead speedup.

## Calls and results

A call takes exactly one of `seed` and `key`; `seed=n` means `jax.random.key(n)`.

| Control | Meaning |
|---|---|
| `max_new_tokens` | Continuation budget. A checkpoint can set a default in `generation_config.json`; if it declares only `max_length`, the budget is that total length minus the padded prompt width. A budget in the call wins. Without any, the call must pass one. |
| `n` | Continuations per prompt, a positive integer. The checkpoint's `num_return_sequences` becomes the default, whatever sampling policy you pass; otherwise 1. `n` in the call wins in both directions. |
| `sampling` | `Sampling`: temperature (default 1.0; 0 is argmax), top-k, top-p, min-p, EOS IDs and the output padding ID. |

Building a task from source defaults raises if an unsupported control is active, such as a repetition penalty or wall-clock stopping. An LM run records its `sampling` value and `sample_tokens` budget, and reloading the run keeps that preview policy. `TextToImage` holds a default solver, guidance and step count; a call can override any of them, and `guidance=None` turns off classifier-free guidance.

Every result array has `n` rows per prompt, in prompt order. `result.rows` is this process's real prompts times `n`. `Generation.text` returns one string per row in the same order, and each row has its own length, termination flag and likelihoods (`behavior_log_probs`, and `raw_log_probs` for the raw policy). `Generation.text` and `CanvasGeneration.text` decode the first time they are read and cache the strings; a result with no processor raises when asked for text. A tokenizer without a pad token needs no change: Dew tokenizes without padding, then pads the numeric rows and masks at the boundary `RunProcessor` uses.

A `TextToImage` result's `pil()` returns the same real rows of an image batch as a list of 8-bit PIL images (RGB, or grayscale for one channel), quantized as `dew.artifacts.uint8_pixels` quantizes them. It refuses a video batch, and a result from `decode=False`, which has no images.

Each prompt is prepared, tokenized and prefilled once for all its continuations, and its media goes through the processor once per call. The continuations then run one after another on the device, so decode time grows with `n`; the cache working memory of one continuation is reused by the next, so only the output arrays grow. Continuation zero draws with the prompt's own key, so `n=1` and continuation zero of a larger request are the same draw. Continuation `j` folds `j` into the key, so asking for more continuations does not change the ones already drawn.

Repeated calls with the same shapes and controls reuse the compiled executables. Binding new weights does not change what the model compiles to.

## Image initialization and latent handoff

`TextToImage.prepare` accepts normalized NHWC pixels or uint8 pixels through `image=`, and already-encoded images through `image_latents=`. Shapes follow the task's `InputSpec` and autoencoder. `times=` picks an explicit starting point on the source grid. The initial state is `alpha(t) * image_latents + sigma(t) * noise`, and `noise=` supplies the unit Gaussian noise.

A pixel mask has shape `[B, H, W, 1]`. Its masked-image conditions go to both guidance branches. `encode_key=None` uses the VAE posterior mean; an explicit key samples from the posterior. Condition encoders accept native prompt records as well as strings. `unconditional=` supplies one row, or one row per prompt, in place of the configured unconditional input.

`initial=` is a latent state that is already noisy, and Dew never adds noise to it again. Prepared inputs keep their process and concrete time grid. `task(prepared, seed=..., decode=False)` skips the VAE and the image checker and returns unclipped `result.latents`, with `result.images` set to `None`. For a base and refiner handoff, pass those latents as another task's `initial` with the matching partial grid. Normal calls return both the decoded images and the latents before decoding. Prepared inputs must belong to the task's mesh. A preparation on the source grid also records its step count; to change the count, prepare a new initial state.

## Serving

`Server.from_task(task, slots=, capacity=)` keeps `slots` rows of `capacity` cache positions resident and admits queued requests into free rows while the other rows keep decoding. Each step prefills the prompts admitted that step and draws one token for every occupied row, in one compiled program; a row leaves on EOS or at its budget, and the next request takes its slot. A served request draws with the key it was submitted with, folded as a one-row `TextGeneration` call folds it, so it draws the tokens it would draw alone.

```python
from dew.inference.serving import Server
from dew.nn.kv_cache import KVCache

greedy = dataclasses.replace(task, sampling=Sampling(temperature=0.0))
server = Server.from_task(greedy, slots=4, capacity=128,
                          kv_cache=KVCache(page_size=16, pages=32), prefix_cache=True)
prompts = ["One day, Lily saw a big dog in", "One day, Lily saw a big dog in the park."]
served = server(prompts[:1], 20, seed=0) + server(prompts[1:], 20, seed=0)
alone = greedy(prompts, 20, seed=0)
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

Calling the server with a batch submits every prompt, steps until they finish and returns one `Generation` per prompt. `server.submit(prompt, max_new_tokens, seed=...)` queues one request and returns a ticket that resolves to its `Generation`; `step()` runs one device call and `run()` steps until the queue and the rows are empty. The task's `n` must be one and its strategy the row-wise sampler, with or without a grammar. `capacity` rounds up to whole 64-slot tiles, and whole pages of a paged cache, and may not exceed the model's context.

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

With the default dense cache every served request draws the tokens it would draw alone. `prefix_cache` hashes each page together with everything before it, and `Server.reload(variables)` stops sharing the pages the old weights wrote. `Sample(guided.json_schema(tokenizer, schema, eos_id))` or `Sample(guided.regex(tokenizer, pattern, eos_id))` as the task's strategy keeps every draw inside the grammar, served or not. The automaton comes from `outlines-core` (`pip install dewml[guided]`), and a transform that forces a token the grammar forbids fails the request.

On TPU, a paged bfloat16 cache decodes through the Pallas kernel `jax.experimental.pallas.ops.tpu.paged_attention`. It runs when the layer's `attention_impl` is `'auto'` or `'tpu'` and the decode mask is exactly the rows' filled slots, with no window, sinks, image groups, pairwise mask or QK-Clip sow. Every other paged decode gathers its pages and runs the ordinary attention kernels. GPUs always take the gather, because jax deprecated its Triton paged kernel (`ops.gpu.paged_attention`), which on an A100 ran at 5967 tokens/s against the gather's 5848.

### Cache quality

Measured on one A100-SXM4-40GB with bf16 weights over 51,100 wikitext-2 test tokens (100 sequences of 512). Each sequence prefills its first half and then decodes the second half one token per step through the cache. Every row is compared with the dense bf16 cache. The noise-floor row is the same dense cache under the `'xla'` attention kernel instead of cuDNN. Greedy agreement is over 32 continuations of 256 tokens from 128-token prompts:

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

int8 keeps perplexity within the kernel noise floor on both checkpoints and float8 does not, so int8 is the quantized format to prefer. Paged and dense int8 give identical numbers, so the paged rows cover both. The float8 rows come from an earlier run of the same protocol (batch 20, no noise-floor row) over the unrotated float8 code that ships. Without the rotation, int8 cost Qwen3-0.6B +0.665 perplexity; rotating float8 keys cost it +1.03.

### Server throughput

Measured on the same A100 with a randomly initialized 12-layer, 1024-wide bf16 decoder (jax 0.11.1):

- Mixed traffic: 48 requests of 32-480 prompt tokens and 128 draws each, 16 slots of 1024. Dense ran at 6360 tokens/s on a 192 MiB cache. Paged ran at 6223 tokens/s on the same memory, and at 4767 tokens/s on a 72 MiB pool, where requests queue for pages. Over the same 192 MiB pool, 48 slots ran at 7540 tokens/s.
- int8 and float8 took 102 MiB and ran at 5123-5225 and 5034-5101 tokens/s. Over a pool of the same 205 MiB, 48 slots ran at 10716 tokens/s with int8 and 8296 with float8.
- Chunked prefill: six 1800-token prompts arrived among 16 decoding rows. `chunk=256` cut the longest step from 31.5 ms to 7.6 ms and raised throughput from 2797 to 2952 tokens/s.
- Prefix cache: 32 requests shared a 1024-token prefix. `prefix_cache` reused 16384 prompt tokens and raised throughput from 1927 to 2562 tokens/s.
- Guided decoding, with GPT-2's vocabulary and a three-field JSON schema: 16 of 16 guided rows parsed and validated, against 0 of 16 unguided. The automaton compiled in 0.74 s into 192 states by 166 token classes, and a guided step ran at 4278 against 6515 tokens/s.

### Serving on a mesh

Weights placed on a mesh serve on it. `Server.from_task` over a task whose variables sit on a mesh runs one step program over every device and places its state through the rule table that places the weights (`dew.nn.sharding.DEFAULT_RULES`). The slots split as a batch's rows do (`activation_batch`: the data, expert and fsdp axes), so each group of devices keeps its own rows and their dense cache, or its own part of a paged pool. A request is seated in the group with room for it, and with a prefix cache in the group already holding its prefix. The tensor axis splits the heads, the MLP and the vocabulary as in training, and the cache keeps its heads on it. Slots, admission and a paged pool's page count have to divide by the number of groups.

A mesh with a sequence axis above one is refused, since the decoder refuses to decode under one: the axis splits a sequence's positions and the cache holds whole sequences. A stage axis above one is refused because the decoder refuses to decode through the training pipeline, and a mesh over several processes because one process schedules the rows.

The server traces its step under the mesh, so the rule table places the activations and a mixture layer runs its dispatch, global or exchange, in a `shard_map`. JAX's checkify, which carries the sampler's device checks, cannot carry a check into a `shard_map` ([jax-ml/jax#40907](https://github.com/jax-ml/jax/issues/40907)), so a step runs the model before it draws: the token a row draws is fed at the start of the next step, and the paged cache checks no write on the device (generation refuses a pool it cannot give every row its capacity before anything runs). `TextGeneration` traces without the mesh and carries each draw's checks into the next step's model call, so it refuses `Mixture(dispatch='exchange')`; `dispatch='global'` computes the same layer.

Measured on 4x RTX 3090 (PCIe 3.0, one host: GPUs 0 and 1 joined by NVLink, 2 and 3 under one PCIe bridge, the pairs across sockets), jax 0.11.2:

- Qwen3-0.6B at float32 and the highest matmul precision, eight prompts of 48 greedy draws each, served twice so the second wave reads the first wave's prefix pages. On one GPU, data=2, tensor=2, data=4, tensor=4, tensor=2 x data=2 and fsdp=2 x data=2, the dense cache, a paged cache and a chunked paged cache with prefix sharing drew the tokens `TextGeneration` draws on one device, with log-probabilities within 4.0e-5; `TextGeneration` itself matched within 2.3e-5. The one-device reference is 3.1e-5 from the same draws scored in fp64, so every layout stayed within 1.3 times fp32's own rounding of the decode, against the bound of four times it that `tools/layout_parity.py` holds a training step to. An int8 or float8 cache is only as exact as its rounding: a sum reassociated upstream moves a key or value that sits near a rounding boundary by a whole quantization step. On one GPU, the server's eight rows against each row alone over the same int8 cache already parted on three rows, at fp64 logit margins of 0.02 to 0.11, with log-probabilities up to 0.10 apart before they parted (float8: six rows, 0.21). Every layout's spread was of that size, 0.09 to 0.13 for int8 and 0.17 to 0.25 for float8. A randomly initialized mixture-of-experts decoder (four experts, top two) served token for token, within 1.7e-6, on expert=4 and expert=2 x tensor=2 with all three caches, and with the exchange dispatch, an all-to-all, on expert=4 and expert=2 x tensor=2. On a randomly initialized two-layer decoder with one prediction depth, `TextGeneration`'s greedy and sampled speculative decoding and its beam search drew what they draw on one device on every layout above.
- Qwen3-1.7B in bfloat16, 256-token prompts, 128 greedy draws each, output tokens per second at the best slot count: one GPU 3061 (128 slots), tensor=2 on the NVLink pair 4765 (256), tensor=2 on the PCIe pair 3452 (256), data=2 on the PCIe pair 5332 (256), data=4 4859 (256), tensor=4 2719 (256), tensor=2 x data=2 6149 (512).

Decode on a tensor axis is bound by the host on these cards. Each step runs 57 all-reduces, and XLA launches the step as 59 command buffers split at them: 12.7 ms of host time per step for 14.5 ms of kernels at 128 slots, so the GPUs were busy 62% of the step. Recording the all-reduces into the command buffers (`--xla_gpu_enable_command_buffer=+COLLECTIVES`) left one command buffer per step but re-recorded it every step (9.6 ms), because the step's buffers move between steps. At 128 slots the defaults ran at 4480 tokens/s; that flag ran at 4488, the latency-hiding scheduler at 4493, no command buffers at 4394, `NCCL_PROTO=LL` at 4243 and cuBLAS in place of Triton GEMMs at 3929, so the defaults stay.

`decode_steps=k` runs k decode iterations in one device call, so the host launches the step once per k tokens. Requests are seated and draws reach the host at call boundaries, so a request waits up to k iterations for its slot; the draws are the same for any k. At 128 slots on the NVLink pair, k=4 cut a decode iteration from 20.9 ms to 13.8 ms and the GPUs' busy share from 67% to 97%. The runs above, whose 256-token prompts cost more than their 128 draws, gained 1.7% at k=4 and 3.1% at k=8, and the time to first token grew from 0.17 s to 0.64 s and 0.98 s. On one GPU, busy throughout already, k=4 lost 2.8%. The default stays 1.

## Other runtimes

To serve with vLLM or Ollama instead, export with `save_pretrained_decoder` or `Pretrained.save` and point the runtime at the directory. `OllamaCompletion` and `OpenAICompletion` (`dew.inference`) call those projects' official clients. Their results keep the backend's metadata and do not make up native raw-policy or behavior-policy likelihoods.
