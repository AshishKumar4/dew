# Core API reference

This page describes the interfaces used in the tutorials; it does not list every public module. Read [your first training run](../getting-started.md) for a complete example.

## Objective

Import `Objective`, `Aux`, `Step`, `Mean`, `mean_loss`, and `scalar_loss` from `dew.objectives`.

| Member | Contract |
|---|---|
| `init(key)` | Return a Flax variables mapping with a `params` collection. Pure; the trainer traces it for shapes and initialization. |
| `loss(variables, batch, step)` | Return additive statistics and `Aux`. Use `Mean(total, mass)` for a shared denominator; a scalar denotes a unit-mass term. |
| `reduce_loss(statistics)` | Return `(value, has_data)`. Override for an objective-owned composite Flax PyTree. |
| `apply_effects(variables, effects)` | Return nonparameter replacements from additive accepted-window observations. Required when the objective emits effects. |
| `evaluate(variables, batch, step)` | Return an artifact, a tuple of artifacts, or `None`. The base method returns `None`. |
| `ema` | Optional `EMASpec`; the base objective uses `None`. |
| `artifact` | Optional description of the objective's evaluation artifact type. |

`Step.step` counts accepted microbatches. Its `key` derives from consumed attempts, including rejected ones. `ema` holds selected averaged leaves overlaid onto the complete variables mapping, or `None`.

`Aux(metrics, variables=None, qk_stats=None, effects=None)` carries training measurements, sequential mutable replacements, QK maxima, and additive deferred effects. The trainer applies effects once on a supported optimizer commit. `scalar_loss(objective, variables, batch, step)` returns a scalar and the same Aux for direct JAX differentiation.

### Collections and EMA selection

A variables tree is a nested mapping. Its outer keys name collections such as `params` and `batch_stats`; leaves are arrays such as a dense kernel or a running mean. The optimizer updates the `params` collection. A mutable Linen call returns replacement state collections, which the objective supplies through `Aux.variables`. See the [stateful example](../concepts/objectives.md#update-non-parameter-state).

`EMASpec(decay, select=everything)` comes from `dew.objectives.base`. `decay` maps completed optimizer-update count to a scalar. `select` accepts a tuple of keys naming a leaf; `under("params", "context_encoder")` selects that subtree, and `everything` selects all leaves. EMA arithmetic uses at least fp32 and preserves explicit fp64, then rounds each result to the initialized EMA leaf dtype. Unit decay selects the frozen leaf exactly. Router bias updates also retain the initialized bias dtype; integer load comparisons avoid converting large counts to floats.

### Input descriptions

Import `Field`, `Condition`, and `InputSpec` from `dew.inputs` or `dew`.

```text
Field(key, shape)
Condition(encoder, field="text", unconditional="")
InputSpec(sample, conditions={})
```

`Field.shape` is the shape of one sample, excluding batch. A `Condition` connects a condition encoder to its tokenized batch field. `InputSpec.conditions` maps the model keyword, such as `textcontext`, to a `Condition`. Each condition must read a distinct batch field. `InputSpec.tokenize(captions)` returns tokenized fields for those conditions; an empty condition mapping returns no fields. Encoders define tokenization, parameters, and encoded output types.

### Mesh and layout

Import these from `dew.training`:

```text
MeshSpec(fsdp=1, expert=1, tensor=1, sequence=1, stage=1, microbatches=None)
Layout(rules=DEFAULT_RULES, min_shard=65536, tolerance=0.02, host=())
build_mesh(spec, devices=None)
```

`build_mesh` uses the supplied devices or JAX's visible devices; the specified factors must divide their count, and data parallelism fills the remaining factor. Explicit pipeline microbatches require `stage > 1` and a positive multiple of the stage count.

`Layout.rules` accepts an ordered logical-axis rule sequence or a mapping of overrides. Mapping entries update the default table. When dimensions compete for one mesh axis, rule order determines precedence; a non-divisible dimension cannot use that axis. Valid parameter mesh axes are `fsdp`, `expert`, and `tensor`. `min_shard` counts elements, not bytes. `tolerance` is the permitted fraction of shardable parameter elements left replicated. `host` names train-state fields, `opt_state` and `ema`, kept in pinned host memory between steps and fetched to the device inside each step. `shardings(mesh, tree)` returns a matching tree of placements; `check(params, shardings, mesh)` validates excessive replication. See [distributed training](../concepts/distributed.md).

## Trainer

Import `Trainer` from `dew` or `dew.training`.

```text
Trainer(objective, optimizer, *, key,
        mesh=MeshSpec(), layout=Layout(), accumulation=1,
        dynamic_scale=False, checkpoints=None, tracker=None,
        step=None, rollout=None, profile=None)
```

`objective` is an initialized objective object and `optimizer` an Optax gradient transformation. The required JAX `key` seeds initialization and the run. `mesh` and `layout` describe placement. Optional capability objects enable checkpoints, tracking, host-side rollouts, and profiling.

`accumulation` counts accepted microbatches per effective window. Shared means use a weighted gradient accumulator with at least fp32 precision, preserving float64 when enabled in JAX. Each finalized gradient enters Optax in its parameter's dtype; partially accumulated gradients keep the wider working dtype. Composite statistics retain independent normalizers and inputs for scalar-VJP replay. Parameters and EMA stay fixed within the window; sequential mutable replacements retain their original read snapshots. `dynamic_scale=True` persists scale/history and rejects nonfinite working or optimizer-input gradients without discarding the accepted prefix.

The custom `step(objective, optimizer)` factory owns accepted/update clocks, scaler, EMA and mutable writes. Its body returns `(state, loss, aux)`. The common compiled wrapper advances attempted `state.step`. A host `rollout` produces realized training arrays once per consumed attempt, before differentiation or replay.

### Train

```text
fit(data, *, steps, log_every=100, eval_every=None,
    checkpoint_every=None, metrics=(), preview=False) -> TrainState
```

- `data` is a `Dataset`.
- `steps` is the final target, including a restored step count.
- `log_every` controls training log ticks.
- `eval_every=None` disables evaluation. With an interval, evaluation also runs at the end.
- `checkpoint_every` controls periodic saves when a checkpointer exists. The normal completion path can still save a final checkpoint when a checkpointer is present.
- `metrics` reduce evaluation artifacts and require evaluation to be enabled.
- `preview=True` explicitly requests generated/display artifacts during evaluation; adding a scalar tracker alone does not request them.

Checkpointable data must supply the consumed iterator position. A failed scaled transaction advances attempted work and scaler history while preserving the earlier accepted prefix. Completely inactive windows close accepted slots without an optimizer call, weight decay, EMA, or deferred effects. Auxiliary-only windows can still be active.

`fit` owns the iterators returned by the dataset. It closes training read-ahead before final evaluation and closes validation passes on success or failure. A restored fit already at its target performs no data, evaluation, compilation, or save work. On exit it stops its trace and waits for pending checkpoint writes without closing the borrowed checkpointer or tracker. Cleanup failures attach to the primary error.

### Initialize, restore, and compile

`initial_state()` constructs an unplaced initial `TrainState`. `place()` returns `(state, shardings, position)`, restoring from the configured checkpointer when available. Eager and placed initialization can differ in low floating-point bits across backends; compare the actual path used by your run.

`compile(state, batch)` returns `compiled(state, batch) -> (state, loss, metrics, loss_finite, accepted)`. The scaler travels in `TrainState`. `loss_finite` and `accepted` are separate: a finite scalar can have a rejected nonfinite gradient. The callable does not donate state or batch arrays because retained replay records and asynchronous checkpoints may own those buffers.

## TrainState

Import `TrainState` from `dew.training`.

| Field | Meaning |
|---|---|
| `step` | Completed attempted batches, including rejected work |
| `microstep` | Accepted microbatches, including finite zero-support slots |
| `updates` | Committed supported optimizer updates |
| `params` | Complete Flax variables, including the inner `params` collection |
| `opt_state` | Underlying optimizer state, advanced only on commits |
| `ema` | Selected moving-average variables or `None` |
| `key` | Immutable root training key |
| `scale` | Dynamic loss scaler with finite-history state, or `None` |
| `window_size` | Persisted accumulation length, checked on restore |
| `accumulation` | Actual partial gradient/statistic/effect/replay state, or `None` |

`state.averaged` overlays EMA leaves onto `state.params` and raises when no EMA is configured. For fixed window length K, the accepted fill is `microstep % K`. A partial window is checkpointed without flushing.

## Dataset and Loading

Import `Dataset` and `Loading` from `dew.data`.

```text
Dataset(train, val, records, batch)
Loading(workers=32, threads=64, read_buffer=128, worker_buffer=20)
```

`train` opens a training iterator, and `val` opens one finite validation pass or is `None`. `records` is the known training-record count or `None`; `batch` is global. `steps_per_epoch` is integer division of records by batch, or `None`. `epoch_steps(epochs=1)` requires a finite record count.

Each factory call must return a fresh, exclusively owned iterator. Ordinary `close()` is finalization and must not race `next()` or checkpoint operations. A source that needs to interrupt blocking reads may additionally implement `request_stop()`: a thread-safe, nonblocking, idempotent signal, safe alongside both `next()` and `close()`. Tokenized wrappers forward these operations.

`DevicePrefetchIterator(iterator, mesh, depth=2, source_state=None)` in `dew.training.distributed` takes ownership on successful construction and starts its worker lazily on the first `next()`. Saved-position restoration also runs there, so its failures unwind through the already-owned iterator. Closing an unused iterator starts only its finalization, not a read or restoration. Use it in a `with` block, or call `close(timeout=5.0)`, even when a loop consumes only a fixed number of batches. Depth must be positive; the bound is `depth` queued device batches plus at most one in-flight batch, excluding the consumer and upstream buffers. Its `source_state` describes the last delivered batch, never speculative read-ahead. EOF and source failures follow preceding queued batches; early close discards unread data and speculative failures, but reports finalization failures.

The prefetch worker performs iteration, checkpoint operations, and final source close. Closing requests cancellation, discards queued batches, and joins the worker. A `TimeoutError` means the source read, placement, or finalization did not cooperate: the thread was **not** killed and its in-flight references may remain. Cancellation stays requested and close can be retried. After close, further iteration stops.

Grain lifecycle limits: the installed `DataLoaderIterator` exposes no public close, so Dew releases its owned references without reaching into private iterators or changing the sampling pipeline. Local-record probes release the source and child processes, but that is not a deterministic upstream shutdown contract. Grain also keeps a process-wide shared-memory deletion thread pool. Its `DatasetIterator.close()` is called on the iteration thread, but read-executor shutdown does not wait for already-running record reads. Arbitrary blocked upstream reads remain outside Dew's shutdown guarantee.

`Loading` controls Grain concurrency and buffers for built-in specifications. Use zero worker processes for small local examples; the defaults may be excessive for a tiny dataset. See [data preparation](../concepts/data.md) for field layouts and process partitioning.


## Checkpoints

Import `Checkpoints` from `dew`.

`Checkpoints(directory, *, keep=2, local_directory=None, local_every=None)` configures persistent storage and optional local emergency checkpoints. Local directory and cadence must be specified together. The object opens storage on use.

`save(step, state, position, metrics=None)` schedules a persistent save. `wait()` waits for pending writes. `latest` reports the newest eligible checkpoint. `restore(template=None, step=None)` returns restored state data and iterator position; a template controls structure and placement. `path(step)` identifies the persistent checkpoint location.

This object does not write `run.json`; run configuration saving is separate. Use the [complete resume example](../guides/checkpoints.md) before adapting these lower-level calls.

## Language modeling and generation

Import `LMObjective` from `dew.objectives.lm`.

```text
LMObjective(model, seq_len, *, ema_decay=0.999, pad_id=None, head_chunks=4,
            samples=None, pretrained=None, balance_rate=None, aux_loss_alpha=None,
            seq_aux=True, loss_role=None, mtp_weight=None, qk_stats=False,
            indexer=None)
IndexerTraining(phase, weight=1.0)
```

The model must implement Linen `hidden_states(tokens, train=..., positions=..., segment_ids=...)`, returning `(B, S, D)`, and `head_weight(params)`, returning the `(D, vocab)` vocabulary matrix. The objective also reads `final_logit_softcap` and `precision`. Prediction-depth training needs `mtp_hidden_states` and compatible prediction-depth configuration. Mutable router, QK and indexer collections are required when their options are enabled.

`pretrained` supplies the complete variables tree; `loss_role` requires aligned `text_roles`. `pad_id` masks matching targets. `head_chunks` controls vocabulary tiling; `samples` configures text previews. `ema_decay=None` trains without an averaged copy, and `1.0` retains a frozen one. Routing balance, auxiliary loss, prediction-depth weight, and QK statistics require matching model computation. These interfaces make `LMObjective` specific to compatible decoders.

`indexer` trains DeepSeek-V3.2's lightning indexer on a model whose `mla` mixer names `index_n_heads` and `index_head_dim`. `IndexerTraining("warmup")` needs a mixer without `index_topk`: the model runs dense attention, `init` keeps the indexer alone in `params` and the rest of the tree under `frozen` (a `pretrained` tree may omit the indexer, as a dense checkpoint does), and the loss is the KL of the indexer's softmax from the attention distribution, reported as `indexer_kl`. `IndexerTraining("sparse")` needs a mixer with `index_topk`: the whole tree trains, the cross entropy trains the main weights and the KL over the selected keys trains the indexer, whose inputs are detached; a warm-up checkpoint's split tree is accepted as `pretrained`. `weight` scales the KL term.

Import `generate`, `Sampling` and `Generation` from `dew.sampling`:

```text
generate(model, params, inputs, max_new_tokens, *, key=None, seed=None,
         sampling=Sampling()) -> Generation
Sampling(temperature=1.0, top_k=None, eos_id=None, pad_id=0, top_p=1.0, min_p=0.0)
```

`params` is the complete variables tree. `inputs` is a `ModelInputs` from `dew.nn.inputs`, or an integer `(B, P)` array normalized to all-valid text. `ModelInputs.token_fields["attention_mask"]` identifies real token slots; there is no separate generation length argument. Every row needs a real token. Only real tokens count against `model.max_seq_len`. Conditioning arrays are batch-aligned and used during prefill; decode keeps the model's cached logical positions. Exactly one of `key` and `seed` is given; `seed=n` is `jax.random.key(n)`.

The compiled decoder uses one padded input shape with per-row cache cursors and a fixed trip count. Finished rows preserve their cached state. On a mesh, rows split over the batch axes and the result keeps that sharding; each process hands in its own rows, at the same count and padded width on every process, and reads them back with `Generation.host()`. Keys fold in the global row index, so a pool draws what one process draws for the same rows. Invalid input on one rank raises on all ranks before device execution.

`Sampling.eos_id` accepts an integer or a tuple of ids; any of them terminates a row. The value normalizes the ids into an immutable tuple. Stochastic selection applies temperature, top-k, nucleus top-p, then relative min-p filtering. At least one token survives. `top_p=1` and `min_p=0` disable their filters. Zero temperature selects argmax without filtering.

`Generation.tokens` includes the original prompt and has shape `(B, P + max_new_tokens)` with `B` the placed rows. `lengths` counts response tokens including EOS. `terminated` marks EOS termination; false means the token budget. Slots after termination hold `Sampling.pad_id`. `behavior_log_probs` and `raw_log_probs` have shape `(B, max_new_tokens)`; the first describes the filtered distribution that drew each action and the second the unmodified policy. `rows` counts this process's real rows; `host()` returns the record over host arrays of those rows; `text` decodes them through the processor a task bound.

`LMObjective.per_token_log_probs(params, tokens, left_padding=...)` scores the raw policy. It left-aligns real tokens for the forward and restores the original next-token alignment. Unscored padding slots are zero. `SampledRollout` records these raw sampling-time values as `old_log_probs` and preserves actual draws as `behavior_log_probs`.

### Inference tasks

Import `pipeline`, `TextGeneration`, `BlockGeneration`, `TextToImage`, `Images`, `DenoisingInputs` and `RunProcessor` from `dew.inference`; `dew.pipeline` is the same front door. [Inference](../concepts/inference.md) describes placement and the three workflows.

```text
pipeline(source, *, mesh=None, layout=None, dtype=None, ema=True, step=None, revision=None)
    -> TextGeneration | BlockGeneration | TextToImage
Objective.pipeline(state, *, ema=True) -> the objective's task over state.averaged or state.params
LMObjective.pipeline(state, *, ema=True, processor=None) -> TextGeneration
TextGeneration(model, variables, processor=None, sampling=Sampling(), max_new_tokens=None)
task(request, max_new_tokens=None, *, key=None, seed=None, sampling=None, images=None) -> Generation
task.bind(variables) -> TextGeneration      task.decode(generation) -> tuple[str, ...]
BlockGeneration(model, variables, process, processor=None, eos_token_ids=(), pad_token_id=0,
                max_new_tokens=None)
task(request, max_new_tokens=None, *, key=None, seed=None, process=None, images=None) -> CanvasGeneration
Pretrained.text_generation(sampling=None) -> TextGeneration
Pretrained.block_generation() -> BlockGeneration
TextToImage(model, process, inputs, params, autoencoder=None, steps=50, guidance=None,
            sampler=DDIM(), grid=None, final_denoise=True, finish=None)
TextToImage.from_objective(objective, variables) -> TextToImage
TextToImage.from_run(directory, *, ema=True, step=None, mesh=None, layout=None, dtype=None)
TextToImage.from_pretrained(repo_id, *, ema=True, mesh=None, layout=None, dtype=None)
image_task.bind(variables) -> TextToImage
image_task.prepare(prompts, *, key=None, seed=None, steps=None) -> DenoisingInputs
image_task(prompts_or_prepared, *, steps=None, guidance=<default>, sampler=None, key=None, seed=None) -> Images
RunProcessor(tokenizer)   # a run's ByteTokenizer or HFTokenizer as a task processor
```

A task captures the variables mapping at construction and on `bind`. Replacing the caller's mapping does not change the existing task. Array buffers remain shared; do not mutate, donate or delete them while a task uses them. Text requests need a processor. Numeric token rows remain integers: mixed text and token rows, floats, booleans and strings are refused; a resident `jax.Array` or `ModelInputs` reaches the model without a host copy.

`max_new_tokens` defaults to the budget the source declares (a checkpoint's `max_new_tokens`, an LM run's `sample_tokens`, an objective's `Samples`); without one the call must pass it. Equal shapes and controls reuse the compiled executable, across calls and across `bind`.

Source-default text tasks preserve temperature, top-k, top-p, min-p, EOS and padding settings. Active unsupported controls such as repetition penalties or beam search raise when creating the default task. Loading weights for training or export does not select a sampling policy. Pass `source.text_generation(sampling=Sampling(...))` for an explicit policy.

`BlockGeneration` uses `BlockProcess.generate`; its `CanvasGeneration` carries lengths, termination and decoder-step counts, without autoregressive likelihoods, plus the same `rows`, `host()` and `text`. `TextToImage` carries the objective's or source's `steps`, `guidance` and `sampler` defaults; `prepare` encodes prompts and draws their noise once, placed for the task's mesh, and `Images.images` is `[rows, H, W, C]` in [-1, 1] with `host()` reading a process's rows back. `grid(steps)` answers the process and the explicit time grid a trajectory of that length walks, for a source whose sampler pairs its own sigma and model-time tables; the noise prior follows that process, so `prepare` takes the same `steps`. `final_denoise=False` ends a trajectory at the last grid point without the closing clean prediction. `sample(denoise, x_T, steps=None, *, solver, guidance=None, key, times=None, final_denoise=True)` in `dew.sampling` takes the same two controls; exactly one of `steps` and `times` is passed, and an explicit grid decides the trajectory's length. `finish(params, images)` runs on the decoded images under the same placement, for a source that ships a checker or an output transform. Rebinding preserves the compilation identity of every task.

### External engine clients

Dew does not run an HTTP server. Install `[inference-clients]` and inject the official client configured for your local or deployed engine. The adapter does not own the SDK client's lifetime.

```python
import ollama
import openai
from dew.inference import OllamaCompletion, OpenAICompletion

with ollama.Client(host="http://127.0.0.1:11434") as client:
    task = OllamaCompletion("my-exported-model", client)
    answer = task("Explain this result.", 64, seed=7,
                  options={"temperature": 0.8, "top_p": 0.9, "num_gpu": 0})
    print(answer.texts)

with openai.OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="local") as client:
    task = OpenAICompletion("my-exported-model", client)
    answer = task("Explain this result.", 64, top_p=0.9,
                  extra_body={"top_k": 40, "min_p": 0.05})
```

The convenience call returns `Completion(texts, finish_reasons, token_counts, usage, responses)`. Missing per-choice token counts and finish reasons remain `None`. Aggregate OpenAI usage is separate; it is never split among choices or replaced with zeros. `responses` retains the SDK models, including reported logprobs, token data and extensions. These are backend reports, not invented native raw/behavior policy likelihoods.

An explicit `sampling=Sampling(...)` sets the native policy controls supported by the selected backend. Ollama receives neutral repetition/presence/frequency penalties and explicit top-k/top-p/min-p values, so its hidden `repeat_penalty=1.1` default does not alter the request. Conflicting explicit options are refused. `OpenAICompletion(..., provider="vllm")` enables vLLM-specific Sampling translation, including `repetition_penalty=1.0` and optional EOS-token IDs. Without a Sampling value, provider defaults or the caller's SDK options apply. Native and backend tokenizers can still differ; this is not an RL interoperability guarantee.

`stream` returns native SDK response chunks. `chat(messages, max_new_tokens, stream=..., **parameters)` preserves SDK tools, tool-result messages, structured-output controls and media fields. Inject an `AsyncClient`/`AsyncOpenAI` and use `acall`, `astream` or `achat` for asynchronous execution. Ollama requests go through the SDK's public `generate` and `chat` methods, which own request conversion, HTTP behavior, error handling and line-stream framing; the adapter rejects request fields the installed SDK does not accept and negative token counts, and the SDK's own parsing rejects unparsable values. OpenAI completions use the SDK's public `with_raw_response` hook, so choice and usage fields are checked on the wire before parsing. OpenAI request parameters go to its completion/chat resources. vLLM-only parameters belong explicitly in `extra_body`. The task's model, prompt, token budget, requested choice count and an explicit `Sampling` policy cannot be overridden through provider extensions; the SDK writes `extra_body` over the named parameters, so a policy field there must equal the policy or the request is refused before any network call.

Live CPU verification imported a locally trained Dew model through `Pretrained.save`, including tokenizer assets, then exercised official SDK completion and streaming on Ollama and its OpenAI-compatible endpoint. Native save/reload was exact. This does not establish numerical parity between Dew and Ollama, or a live vLLM run.

## Diffusion and JEPA objectives

```text
DiffusionObjective(model, process, inputs, *, autoencoder=None,
                   unconditional_prob=0.12, ema_decay=0.999, sampler=DDIM(),
                   guidance=CFG(3.0), steps=200, pretrained=None)
JepaObjective(encoder, predictor, mask, sample, momentum=(0.996, 1.0),
              momentum_steps=100000, label_key="label")
```

Import `DiffusionObjective` from `dew.objectives.diffusion`. Its model accepts noisy arrays shaped `(B, *latent_shape)`, model noise levels shaped `(B,)`, and conditioning keywords from `InputSpec`. It returns a prediction with the sample's channel/spatial geometry. The `Process` determines the training target and prediction conversion. The objective passes `train=True` and a dropout RNG during training. An autoencoder changes sample geometry and must expose compatible encode/decode operations. `ema_decay=None` keeps no averaged copy, so previews and evaluation read the live variables. `steps`, `sampler`, and `guidance` configure preview sampling; they do not set the number of optimization steps.

Import `JepaObjective` from `dew.objectives.jepa`. The encoder receives normalized images/video and optional token indices plus `train` and RNG settings. It returns token features with the feature dimension last. The predictor consumes context features and context/target position indices and returns target features of the encoder width. Mask grid, patch geometry, and predictor dimensions must agree. `momentum` specifies the EMA schedule endpoints over `momentum_steps` optimizer updates; `label_key` identifies labels for representation evaluation. See the [JEPA example](../guides/representation-learning.md).

### Pretrained latent diffusion

The same `dew.interop.load_pretrained` entry reads a diffusion checkpoint directory with `model_index.json`, component configurations, safetensors and tokenizer files. The returned `Pretrained` holds a native `UNet2DCondition`, an `AutoencoderKL` behind the existing autoencoder seam, native CLIP conditioning, a `Process` and the native solver policy. Model and scheduler implementations from other libraries run only in the reference tools.

`source.text_to_image()` builds the native image task. `source.save(directory, variables=updated)` writes the updated component weights back to their published layouts, retaining tokenizer files, image geometry and any safety-head parameters. Flax-declared components retain their source class and receive Flax msgpack files alongside the safetensors used by Dew; export does not relabel them as PyTorch models.

Source UNet configuration selects normalization groups and epsilon, and the declared implementation selects GEGLU semantics: exact GELU for PyTorch sources, approximate GELU for Flax sources. Spatial upsampling targets the actual next skip shape, including odd intermediate dimensions.

The source scheduler policy retains rounded DDIM/PNDM model times and fixed transfer strides, DDIM clipping, zero-terminal-SNR beta rescaling, and DPM Karras grids. Unsupported active controls fail at load rather than being ignored, including dynamic thresholding and published PNDM velocity-domain history. Native PNDM itself integrates epsilon history.

```python
from dew.interop import load_pretrained
from dew.objectives.diffusion import DiffusionObjective

source = load_pretrained("./image-checkpoint", dtype="float32")
images = source.text_to_image()(["a flower"], steps=20, seed=0).host().images
objective = DiffusionObjective(
    source.model, source.process, source.inputs,
    autoencoder=source.autoencoder, pretrained=source.variables,
)
```

Training batches carry uint8 NHWC images and `source.inputs.tokenize(captions)`. A nine-channel inpainting source also specifies `inputs.mask`: binary NHWC masks with one channel, where white marks the region to repaint. Caption dropout preserves the mask and masked-image latents. This is neural conditioning, not a guarantee that decoded unmasked pixels equal the original image.

Prepared `DenoisingInputs` can supply encoded native conditions and initial latents. Explicit grids pass to `sample(times=..., final_denoise=False)` when the last latent is the result; without an explicit grid, the existing `steps` and final clean-prediction convention remain unchanged. Published tabulated PRK grids own their integer half-interval rule; ordinary native grids retain exact half-intervals.

## Configuration and registries

A registry maps names to known classes or factories. For example, `models.build(name, **fields)` validates model fields and reconstructs supported configuration records. Ordinary Python constructors provide a clearer typed interface when the class is known. Dynamic lookup cannot provide the same static type information as a specific constructor.

`RunConfig.save` writes the run configuration. It is separate from the state checkpoint. [Recipes](../recipes.md) describes the configuration entry points and their side effects.

For complete family restrictions, model-specific data, quantization, and deployment scope, use the [capability reference](support.md) and the relevant task guide.
