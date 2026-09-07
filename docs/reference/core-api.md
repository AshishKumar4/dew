# Core API reference

This page describes the interfaces used in the tutorials. Read [your first training run](../getting-started.md) for a complete example. The [module index](../api.md) locates other public modules; it is an index, not a complete signature reference.

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
generate(model, params, inputs, max_new_tokens, *, key,
         sampling=Sampling()) -> Generation
Sampling(temperature=1.0, top_k=None, eos_id=None, pad_id=0)
```

`params` is the complete variables tree. `inputs` is a `ModelInputs` from `dew.nn.inputs`, or an integer `(B, P)` array normalized to all-valid text. `ModelInputs.token_fields["attention_mask"]` identifies real token slots; there is no separate generation length argument. Every row needs a real token. Only real tokens count against `model.max_seq_len`. Conditioning arrays are batch-aligned and used during prefill; decode keeps the model's cached logical positions. Scalar `positions` supplied in token fields continue from the last valid position.

The compiled decoder uses one padded input shape with per-row cache cursors and a fixed trip count. Finished rows preserve their cached state. On a mesh, rows split over batch axes; each process receives its own rows. All participating processes validate inputs and agree on input shapes and sampling controls before device execution. Host input rejection propagates to peers; blocked device collectives cannot be recovered by this protocol. Changing padded shapes or static controls can still compile a new executable. Streaming and request scheduling are not part of this batch function.

`Sampling.eos_id` accepts an integer or a tuple of ids; any of them terminates a row. The value normalizes the ids into an immutable tuple.

`Generation.tokens` includes the original prompt and has shape `(B, P + max_new_tokens)`. `lengths` counts response tokens including EOS. `terminated` marks EOS termination; false means the token budget. Slots after termination hold `Sampling.pad_id`. `behavior_log_probs` and `raw_log_probs` have shape `(B, max_new_tokens)` and zero invalid tails. Only slots below `lengths` are likelihoods. Behavior probabilities include temperature/top-k; raw probabilities describe the unmodified model. Greedy behavior has probability one for its selected action.

`LMObjective.per_token_log_probs(params, tokens, left_padding=...)` scores the raw policy. It left-aligns real tokens for the forward and restores the original next-token alignment. Unscored padding slots are zero. `SampledRollout` records these raw sampling-time values as `old_log_probs` and preserves actual draws as `behavior_log_probs`. GRPO compares current and old raw-policy likelihoods; it does not silently substitute the behavior distribution. Reward text excludes EOS and the invalid tail. Models explicitly declaring `causal=False` are refused by the next-token objective.

### Engine and serving

Import `Engine`, `GenerationJob`, `WeightVersion`, `CapacityError` and `generation_family` from `dew.sampling`; import `serve` from `dew.serve`.

```text
Engine(source_or_model, variables=None, *, family=None, max_batch_size,
       max_pending_requests=64, max_request_bytes=256 MiB, prefix_cache_bytes=0,
       max_weight_versions=2, steps_per_dispatch=1)
engine.start() / with engine: ...        engine.close(cancel=False, timeout=None)
engine.submit(inputs, max_new_tokens, *, key, generation=None, version=None) -> GenerationJob
engine.publish(variables) -> WeightVersion      engine.version      engine.stats()
job.result(timeout=None)   job.stream()   job.cancel()   job.status   job.version
serve(engine, *, processor=None, host="127.0.0.1", port=0) -> Server
```

A `Pretrained` source supplies its model, variables and generation family; a bare native model uses `generation_family(model)`, which is `AutoregressiveFamily` for decoders and `CanvasFamily` for `DiffusionGemma`. The family owns the science: host validation, the deterministic prefill, one advance step, the per-request host record and the typed result. `Engine.submit` validates on the caller's thread and raises `ValueError`/`TypeError` for bad input, `CapacityError` when a bound would be exceeded, and `RuntimeError` after `close`. `generation` is a `Sampling` or a `BlockProcess`; `None` uses the family default, which a loaded source fills from its generation config.

`start` copies the initial variables; `publish` copies again and returns the new default version. Accepted requests keep the version they pinned, so the caller may donate or delete its arrays after `publish` returns. Unpinned older versions retire with their prefix entries; `publish` raises `CapacityError` while `max_weight_versions` snapshots are pinned, and `submit` rejects versions from another engine or already retired.

Requests of an `AutoregressiveFamily` with equal `Sampling` and equal state layout share one decode state; its slot count grows and shrinks in powers of two up to `max_batch_size`, which bounds the rows allocated at once. A `CanvasFamily` request keeps its own state because refinement draws noise for its whole batch; rows of different requests are never merged and keys never change. The prefix cache retains only deterministic prefill states, keyed by weight version, all token fields, all conditioning arrays and the geometry the family declares; retained bytes stay under `prefix_cache_bytes` with LRU eviction.

`result` blocks for the typed result; a timeout raises `TimeoutError` and leaves the request running. `stream` is a single-consumer iterator of `TokenEvent` (row, position, token, both log-probabilities, terminated) or `SpanEvent` (row, start, committed tokens, terminated, decoder steps). `cancel` returns false once the request is terminal; otherwise it ends with `CancelledError`, including a request whose last device step was in flight. A model error fails the request that caused it; a failure inside a shared decode step fails the requests of that state. `close` stops admission, finishes or cancels accepted work, then releases versions, prefixes and states. The engine schedules one process; pooled processes use `generate`.

`serve` answers `POST /generate` with JSON: `prompt` (text; needs `processor`) or `input_ids`, `max_new_tokens`, optional `seed`, `sampling` or `process` fields, and `stream`. A plain response carries the result fields, decoded `text` when a processor exists, and the version serial. A streaming response is newline-delimited JSON events followed by the result. Bad input answers 400, a full engine 429, a closed engine 503; a client that disconnects mid-stream cancels its request. `GET /health` reports engine statistics.

## Diffusion and JEPA objectives

```text
DiffusionObjective(model, process, inputs, *, autoencoder=None,
                   unconditional_prob=0.12, ema_decay=0.999, sampler=DDIM(),
                   guidance=CFG(3.0), steps=200)
JepaObjective(encoder, predictor, mask, sample, momentum=(0.996, 1.0),
              momentum_steps=100000, label_key="label")
```

Import `DiffusionObjective` from `dew.objectives.diffusion`. Its model accepts noisy arrays shaped `(B, *latent_shape)`, model noise levels shaped `(B,)`, and conditioning keywords from `InputSpec`. It returns a prediction with the sample's channel/spatial geometry. The `Process` determines the training target and prediction conversion. The objective passes `train=True` and a dropout RNG during training. An autoencoder changes sample geometry and must expose compatible encode/decode operations. `ema_decay=None` keeps no averaged copy, so previews and evaluation read the live variables. `steps`, `sampler`, and `guidance` configure preview sampling; they do not set the number of optimization steps.

Import `JepaObjective` from `dew.objectives.jepa`. The encoder receives normalized images/video and optional token indices plus `train` and RNG settings. It returns token features with the feature dimension last. The predictor consumes context features and context/target position indices and returns target features of the encoder width. Mask grid, patch geometry, and predictor dimensions must agree. `momentum` specifies the EMA schedule endpoints over `momentum_steps` optimizer updates; `label_key` identifies labels for representation evaluation. See the [JEPA example](../guides/representation-learning.md).

## Configuration and registries

A registry maps names to known classes or factories. For example, `models.build(name, **fields)` validates model fields and reconstructs supported configuration records. Ordinary Python constructors provide a clearer typed interface when the class is known. Dynamic lookup cannot provide the same static type information as a specific constructor.

`RunConfig.save` writes the run configuration. It is separate from the state checkpoint. [Recipes](../recipes.md) describes the configuration entry points and their side effects.

For complete family restrictions, model-specific data, quantization, and deployment scope, use the [capability reference](support.md) and the relevant task guide.
