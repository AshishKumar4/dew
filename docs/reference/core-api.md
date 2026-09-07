# Core API reference

This page describes the interfaces used in the tutorials. Read [your first training run](../getting-started.md) for a complete example. The [module index](../api.md) locates other public modules; it is an index, not a complete signature reference.

## Objective

Import `Objective`, `Aux`, and `Step` from `dew.objectives.base`.

| Member | Contract |
|---|---|
| `init(key)` | Return a Flax variables mapping with a `params` collection. Pure; the trainer traces it for shapes and initialization. |
| `loss(variables, batch, step)` | Return a scalar JAX loss and `Aux`. Differentiation targets `variables["params"]`. |
| `evaluate(variables, batch, step)` | Return an artifact, a tuple of artifacts, or `None`. The base method returns `None`. |
| `ema` | Optional `EMASpec`; the base objective uses `None`. |
| `artifact` | Optional description of the objective's evaluation artifact type. |

`Step` contains `step`, `key`, and `ema`. The EMA field holds a full variables mapping with selected averaged leaves overlaid, or `None` when no EMA is configured.

`Aux(metrics, variables=None, qk_stats=None)` carries training metrics, optional replacement non-parameter collections, and optional attention statistics for QK-Clip. Metrics should be scalar arrays suitable for logging. The optimizer exclusively owns parameter updates.

### Collections and EMA selection

A variables tree is a nested mapping. Its outer keys name collections such as `params` and `batch_stats`; leaves are arrays such as a dense kernel or a running mean. The optimizer updates the `params` collection. A mutable Linen call returns replacement state collections, which the objective supplies through `Aux.variables`. See the [stateful example](../concepts/objectives.md#update-non-parameter-state).

`EMASpec(decay, select=everything)` comes from `dew.objectives.base`. `decay` is an Optax-compatible callable from completed optimizer-update count to a scalar decay. `select` accepts a tuple of string keys describing a leaf path and returns a boolean. `under("params", "context_encoder")` selects that subtree; `everything` selects all leaves. Selection must retain at least one leaf. Unit decay preserves the selected initial reference; other decays update the weighted average.

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
Layout(rules=DEFAULT_RULES, min_shard=65536, tolerance=0.02)
build_mesh(spec, devices=None)
```

`build_mesh` uses the supplied devices or JAX's visible devices; the specified factors must divide their count, and data parallelism fills the remaining factor. Explicit pipeline microbatches require `stage > 1` and a positive multiple of the stage count.

`Layout.rules` accepts an ordered logical-axis rule sequence or a mapping of overrides. Mapping entries update the default table. When dimensions compete for one mesh axis, rule order determines precedence; a non-divisible dimension cannot use that axis. Valid parameter mesh axes are `fsdp`, `expert`, and `tensor`. `min_shard` counts elements, not bytes. `tolerance` is the permitted fraction of shardable parameter elements left replicated. `shardings(mesh, tree)` returns a matching tree of placements; `check(params, shardings, mesh)` validates excessive replication. See [distributed training](../concepts/distributed.md).

## Trainer

Import `Trainer` from `dew` or `dew.training`.

```text
Trainer(objective, optimizer, *, key,
        mesh=MeshSpec(), layout=Layout(), accumulation=1,
        dynamic_scale=False, checkpoints=None, tracker=None,
        step=None, rollout=None, profile=None)
```

`objective` is an initialized objective object and `optimizer` an Optax gradient transformation. The required JAX `key` seeds initialization and the run. `mesh` and `layout` describe placement. Optional capability objects enable checkpoints, tracking, host-side rollouts, and profiling.

`accumulation` wraps the optimizer with Optax MultiSteps. It currently averages microbatch gradients. This is not equivalent to a combined token mean when valid-target counts differ. `dynamic_scale=True` uses dynamic loss scaling; its overflow/resume behavior is under repair. These limitations are not solved by changing logging cadence.

The `step` factory replaces the default optimization step and owns its state-update contract. A custom `rollout` transforms a batch before the compiled step; it is distinct from replacing the step itself.

### Train

```text
fit(data, *, steps, log_every=100, eval_every=None,
    checkpoint_every=None, metrics=()) -> TrainState
```

- `data` is a `Dataset`.
- `steps` is the final target, including a restored step count.
- `log_every` controls training log ticks.
- `eval_every=None` disables evaluation. With an interval, evaluation also runs at the end.
- `checkpoint_every` controls periodic saves when a checkpointer exists. The normal completion path can still save a final checkpoint when a checkpointer is present.
- `metrics` reduce evaluation artifacts and require evaluation to be enabled.

For accurate continuation, checkpointable data must provide its iterator position. The overflow path currently has separate attempted-work and accepted-step counters; see [checkpoint limits](../guides/checkpoints.md#current-limits).

`fit` owns the iterators returned by the dataset, not the dataset itself. It closes training read-ahead before final evaluation and closes every validation pass, including failures and early exhaustion. A fit already at its target opens no training iterator or profiler; requested final evaluation still runs. On every exit it stops its own active trace and waits for pending checkpoint writes without closing the borrowed checkpointer or tracker. The original training/evaluation failure remains primary; cleanup failures are attached as exception notes.

### Initialize, restore, and compile

`initial_state()` constructs an unplaced initial `TrainState`. `place()` returns `(state, shardings, position)`, restoring from the configured checkpointer when available. Eager and placed initialization can differ in low floating-point bits across backends; compare the actual path used by your run.

`compile(state, batch, scale=None)` returns a callable invoked as `compiled(state, scale, batch)`, with `scale=None` for ordinary unscaled training. It returns `(state, scale, loss, metrics, finite)`. State buffers are donated, so callers must use the returned state and not reuse donated buffers. `finite` reports finite loss, not gradient acceptance.

## TrainState

Import `TrainState` from `dew.training`.

| Field | Meaning |
|---|---|
| `step` | Scalar counter stored with the state |
| `params` | Complete Flax variables, including the inner `params` collection |
| `opt_state` | Optimizer state, including accumulation state when enabled |
| `ema` | Selected moving-average variables or `None` |
| `key` | Run random key |

`state.averaged` overlays EMA leaves onto `state.params`. It raises when no EMA is configured. It does not silently return live variables. Dynamic loss-scaler state is not currently part of this checkpointed structure.

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
            seq_aux=True, loss_role=None, mtp_weight=None, qk_stats=False)
```

The model must implement Linen `hidden_states(tokens, train=..., positions=..., segment_ids=...)`, returning `(B, S, D)`, and `head_weight(params)`, returning the `(D, vocab)` vocabulary matrix. The objective also reads `final_logit_softcap` and `precision`. Prediction-depth training needs `mtp_hidden_states` and compatible prediction-depth configuration. Mutable router and QK collections are required when their options are enabled.

`pretrained` supplies the complete variables tree; `loss_role` requires aligned `text_roles`. `pad_id` masks matching targets. `head_chunks` controls vocabulary tiling; `samples` configures text previews. Routing balance, auxiliary loss, prediction-depth weight, and QK statistics require matching model computation. These interfaces make `LMObjective` specific to compatible decoders.

Import `generate`, `Sampling` and `Generation` from `dew.sampling`:

```text
generate(model, params, prompt, max_new_tokens, *, key,
         sampling=Sampling(), prompt_lengths=None) -> Generation
Sampling(temperature=1.0, top_k=None, eos_id=None, pad_id=0)
```

`params` is the complete variables tree. `prompt` is an integer `(B, P)` array. Optional `(B,)` `prompt_lengths` counts real tokens at each row's right edge. Each real prompt plus the token budget must fit the cache. The host groups equal lengths and removes padding before the compiled cached decoder; different lengths can produce different JIT shapes. The decode runs a fixed trip count with finished rows masked. On a mesh the group's rows split over the batch axes; in a pool each process passes and receives its own rows, and the group plan is agreed across processes so their collectives match. This API does not provide streaming or continuous request batching.

`Generation.tokens` includes the original prompt and has shape `(B, P + max_new_tokens)`. `lengths` counts response tokens including EOS. `terminated` marks EOS termination; false means the token budget. Slots after termination hold `Sampling.pad_id`. `behavior_log_probs` and `raw_log_probs` have shape `(B, max_new_tokens)` and zero invalid tails. Only slots below `lengths` are likelihoods. Behavior probabilities include temperature/top-k; raw probabilities describe the unmodified model. Greedy behavior has probability one for its selected action.

`LMObjective.per_token_log_probs(params, tokens, left_padding=...)` scores the raw policy. It left-aligns real tokens for the forward and restores the original next-token alignment. Unscored padding slots are zero. `SampledRollout` records these raw sampling-time values as `old_log_probs` and preserves actual draws as `behavior_log_probs`. GRPO compares current and old raw-policy likelihoods; it does not silently substitute the behavior distribution. Reward text excludes EOS and the invalid tail. Models explicitly declaring `causal=False` are refused by the next-token objective.

## Diffusion and JEPA objectives

```text
DiffusionObjective(model, process, inputs, *, autoencoder=None,
                   unconditional_prob=0.12, ema_decay=0.999, sampler=DDIM(),
                   guidance=CFG(3.0), steps=200)
JepaObjective(encoder, predictor, mask, sample, momentum=(0.996, 1.0),
              momentum_steps=100000, label_key="label")
```

Import `DiffusionObjective` from `dew.objectives.diffusion`. Its model accepts noisy arrays shaped `(B, *latent_shape)`, model noise levels shaped `(B,)`, and conditioning keywords from `InputSpec`. It returns a prediction with the sample's channel/spatial geometry. The `Process` determines the training target and prediction conversion. The objective passes `train=True` and a dropout RNG during training. An autoencoder changes sample geometry and must expose compatible encode/decode operations. `steps`, `sampler`, and `guidance` configure preview sampling; they do not set the number of optimization steps.

Import `JepaObjective` from `dew.objectives.jepa`. The encoder receives normalized images/video and optional token indices plus `train` and RNG settings. It returns token features with the feature dimension last. The predictor consumes context features and context/target position indices and returns target features of the encoder width. Mask grid, patch geometry, and predictor dimensions must agree. `momentum` specifies the EMA schedule endpoints over `momentum_steps` optimizer updates; `label_key` identifies labels for representation evaluation. See the [JEPA example](../guides/representation-learning.md).

## Configuration and registries

A registry maps names to known classes or factories. For example, `models.build(name, **fields)` validates model fields and reconstructs supported configuration records. Ordinary Python constructors provide a clearer typed interface when the class is known. Dynamic lookup cannot provide the same static type information as a specific constructor.

`RunConfig.save` writes the run configuration. It is separate from the state checkpoint. [Recipes](../recipes.md) describes the configuration entry points and their side effects.

For complete family restrictions, model-specific data, quantization, and deployment scope, use the [capability reference](support.md) and the relevant task guide.
