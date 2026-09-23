# Inference

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

`step` picks one of a run's checkpoints. `revision` pins a Hub source. `dtype` sets the compute dtype. `param_dtype` sets how parameters are stored: `None` keeps a run's stored dtypes and uses float32 master weights for a published source. `"auto"` keeps the stored dtypes for both: a source's `config.json` `dtype`, or its first floating tensor's.

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

`dew.inference.serving.Server` runs continuous batching over one resident KV cache: `Server.from_task(task, slots=, capacity=)` admits queued requests into free rows while the other rows keep decoding. With the default dense cache every served request draws the tokens it would draw alone. The cache layout is `dew.nn.kv_cache.KVCache`, passed as `kv_cache=` to `from_task` (or set as a `CausalTransformer`'s `kv_cache` field for `TextGeneration`):

- `KVCache(page_size=16, pages=N)` pages the cache. All rows share one pool of `N` pages, and a request is admitted once the pool can hold its prompt and budget, so the pool is sized to memory rather than to `slots * capacity`. Outside a server, leave `pages` unset: nothing hands out a smaller pool, and a row that would write past it fails its request.
- `KVCache(quantized="int8")` stores keys and values in eight bits with one float32 scale per token and head, and rotates the keys by a Hadamard matrix first, which needs a power-of-two `head_dim`. `quantized="float8_e4m3fn"` stores unrotated e4m3 at any `head_dim`. Either works dense or paged. Prefer int8: in the table below it keeps perplexity within the kernel noise floor on both checkpoints, while float8 does not.
- `chunk=256` prefills a long prompt in pieces, one piece per step, so rows that are already decoding are not held up behind it. It needs a paged cache.
- `prefix_cache=True` shares full prompt pages between requests that begin with the same tokens, hashing each page together with everything before it. It needs a paged cache. `Server.reload` stops sharing the pages the old weights wrote.
- `Sample(guided.json_schema(tokenizer, schema, eos_id))` or `Sample(guided.regex(tokenizer, pattern, eos_id))` as the task's strategy keeps every draw inside the grammar, served or not. The automaton comes from `outlines-core` (`pip install dew-ml[guided]`), and a transform that forces a token the grammar forbids fails the request.

On TPU, a paged bfloat16 cache decodes through the Pallas kernel `jax.experimental.pallas.ops.tpu.paged_attention`. It runs when the layer's `attention_impl` is `'auto'` or `'tpu'` and the decode mask is exactly the rows' filled slots: no window, sinks, image groups, pairwise mask or QK-Clip sow. Every other paged decode gathers its pages and runs the ordinary attention kernels. GPUs always take the gather, because jax deprecated its Triton paged kernel (`ops.gpu.paged_attention`), which on an A100 ran at 5967 tokens/s against the gather's 5848.

Quality, measured on one A100-SXM4-40GB with bf16 weights over 51,100 wikitext-2 test tokens (100 sequences of 512). Each sequence prefills its first half and then decodes the second half one token per step through the cache. Every row is compared with the dense bf16 cache. The noise-floor row is the same dense cache under the `'xla'` attention kernel instead of cuDNN. Greedy agreement is over 32 continuations of 256 tokens from 128-token prompts:

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

Paged and dense int8 give identical numbers, so the paged rows cover both. The float8 rows come from an earlier run of the same protocol (batch 20, no noise-floor row) over the unrotated float8 code that ships. Without the rotation, int8 cost Qwen3-0.6B +0.665 perplexity; rotating float8 keys cost it +1.03.

Throughput, measured on the same A100 with a randomly initialized 12-layer, 1024-wide bf16 decoder (jax 0.11.1):

- Mixed traffic: 48 requests of 32-480 prompt tokens and 128 draws each, 16 slots of 1024. Dense ran at 6360 tokens/s on a 192 MiB cache. Paged ran at 6223 tokens/s on the same memory, and at 4767 tokens/s on a 72 MiB pool, where requests queue for pages. Over the same 192 MiB pool, 48 slots ran at 7540 tokens/s.
- int8 and float8 took 102 MiB and ran at 5123-5225 and 5034-5101 tokens/s. Over a pool of the same 205 MiB, 48 slots ran at 10716 tokens/s with int8 and 8296 with float8.
- Chunked prefill: six 1800-token prompts arrived among 16 decoding rows. `chunk=256` cut the longest step from 31.5 ms to 7.6 ms and raised throughput from 2797 to 2952 tokens/s.
- Prefix cache: 32 requests shared a 1024-token prefix. `prefix_cache` reused 16384 prompt tokens and raised throughput from 1927 to 2562 tokens/s.
- Guided decoding, with GPT-2's vocabulary and a three-field JSON schema: 16 of 16 guided rows parsed and validated, against 0 of 16 unguided. The automaton compiled in 0.74 s into 192 states by 166 token classes, and a guided step ran at 4278 against 6515 tokens/s.

### Serving on a mesh

Weights placed on a mesh serve on it. `Server.from_task` over a task whose variables sit on a mesh runs one step program over every device and places its state through the rule table that places the weights (`dew.nn.sharding.DEFAULT_RULES`). The slots split as a batch's rows do (`activation_batch`: the data, expert and fsdp axes), so each group of devices keeps its own rows and their dense cache, or its own part of a paged pool (`pages`). A request is seated in the group with room for it, and with a prefix cache in the group already holding its prefix. The tensor axis splits the heads, the MLP and the vocabulary as in training, and the cache keeps its heads on it. Slots, admission and a paged pool's page count have to divide by the number of groups.

A mesh with a sequence axis above one is refused, since the decoder refuses to decode under one: the axis splits a sequence's positions and the cache holds whole sequences. A stage axis above one is refused because the decoder refuses to decode through the training pipeline, and a mesh over several processes because one process schedules the rows.

The server traces its step under the mesh, so the rule table places the activations and a mixture layer runs its dispatch, global or exchange, in a `shard_map`. Jax's checkify, which carries the sampler's device checks, cannot carry a check into a `shard_map` ([jax-ml/jax#40907](https://github.com/jax-ml/jax/issues/40907)), so a step runs the model before it draws: the token a row draws is fed at the start of the next step, and the paged cache checks no write on the device (generation refuses a pool it cannot give every row its capacity before anything runs). `TextGeneration` traces without the mesh and carries each draw's checks into the next step's model call, so it refuses `Mixture(dispatch='exchange')`; `dispatch='global'` computes the same layer.

Measured on 4x RTX 3090 (PCIe 3.0, one host: GPUs 0 and 1 joined by NVLink, 2 and 3 under one PCIe bridge, the pairs across sockets), jax 0.11.2:

- Qwen3-0.6B at float32 and the highest matmul precision, eight prompts of 48 greedy draws each, served twice so the second wave reads the first wave's prefix pages. On one GPU, data=2, tensor=2, data=4, tensor=4, tensor=2 x data=2 and fsdp=2 x data=2, the dense cache, a paged cache and a chunked paged cache with prefix sharing drew the tokens `TextGeneration` draws on one device, with log-probabilities within 4.0e-5; `TextGeneration` itself matched within 2.3e-5. The one-device reference is 3.1e-5 from the same draws scored in fp64, so every layout stayed within 1.3 times fp32's own rounding of the decode, against the bound of four times it that `tools/layout_parity.py` holds a training step to. An int8 or float8 cache is only as exact as its rounding: a sum reassociated upstream moves a key or value that sits near a rounding boundary by a whole quantization step. On one GPU, the server's eight rows against each row alone over the same int8 cache already parted on three rows, at fp64 logit margins of 0.02 to 0.11, with log-probabilities up to 0.10 apart before they parted (float8: six rows, 0.21). Every layout's spread was of that size, 0.09 to 0.13 for int8 and 0.17 to 0.25 for float8. A randomly initialized mixture-of-experts decoder (four experts, top two) served token for token, within 1.7e-6, on expert=4 and expert=2 x tensor=2 with all three caches, and with the exchange dispatch, an all-to-all, on expert=4 and expert=2 x tensor=2. On a randomly initialized two-layer decoder with one prediction depth, `TextGeneration`'s greedy and sampled speculative decoding and its beam search drew what they draw on one device on every layout above.
- Qwen3-1.7B in bfloat16, 256-token prompts, 128 greedy draws each, output tokens per second at the best slot count: one GPU 3061 (128 slots), tensor=2 on the NVLink pair 4765 (256), tensor=2 on the PCIe pair 3452 (256), data=2 on the PCIe pair 5332 (256), data=4 4859 (256), tensor=4 2719 (256), tensor=2 x data=2 6149 (512).

Decode on a tensor axis is bound by the host on these cards. Each step runs 57 all-reduces, and XLA launches the step as 59 command buffers split at them: 12.7 ms of host time per step for 14.5 ms of kernels at 128 slots, so the GPUs were busy 62% of the step. Recording the all-reduces into the command buffers (`--xla_gpu_enable_command_buffer=+COLLECTIVES`) left one command buffer per step but re-recorded it every step (9.6 ms), because the step's buffers move between steps. At 128 slots the defaults ran at 4480 tokens/s; that flag ran at 4488, the latency-hiding scheduler at 4493, no command buffers at 4394, `NCCL_PROTO=LL` at 4243 and cuBLAS in place of Triton GEMMs at 3929, so the defaults stay. `Server.from_task(decode_steps=k)` runs k decode iterations in one device call, so the host launches the step once per k tokens: at 128 slots on the NVLink pair, k=4 cut a decode iteration from 20.9 ms to 13.8 ms and the GPUs' busy share from 67% to 97%. The runs above, whose 256-token prompts cost more than their 128 draws, gained 1.7% at k=4 and 3.1% at k=8, and the time to first token grew from 0.17 s to 0.64 s and 0.98 s, since a request waits for a call's boundary. On one GPU, busy throughout already, k=4 lost 2.8%. The default stays 1.

To serve with vLLM or Ollama instead, export with `Pretrained.save`. `OllamaCompletion` and `OpenAICompletion` use those projects' official clients. Their results keep the backend's metadata and do not make up native raw-policy or behavior-policy likelihoods.
