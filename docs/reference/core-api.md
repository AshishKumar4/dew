# Core API

This page describes the main interfaces and the contracts between them, grouped by task. Every public module also has a page generated from its docstrings, listed at the [end of this page](#all-modules) and in the sidebar. The [Quickstart](../getting-started.md) uses the core interfaces in one script.

## Objective

Import `Objective`, `Aux`, `Step` and `Ratio` from `dew.objectives`. `Ratio.mean()` reduces a ratio statistic, and `objective.scalar_loss(variables, batch, step)` evaluates and reduces a loss for direct differentiation. `dew.Supervised(model, loss, metrics=(), *, inputs)` trains any Flax model on a per-example `loss(outputs, batch)` without a subclass ([Custom objectives](../concepts/objectives.md#supervised)).

| Member | Contract |
|---|---|
| `init(key, variables=None)` | Return a Flax variables mapping with a `params` collection. Pure; the trainer traces it once for shapes and once for values. `variables` is a held tree the caller supplies, which is how the trainer passes it as data; with `None` the objective uses its own (`DiffusionObjective.held_variables`, for example). An objective that holds nothing ignores it. |
| `loss(variables, batch, step)` | Return additive statistics and `Aux`. Use `Ratio(total, mass)` for a shared denominator; a scalar denotes a unit-mass term. |
| `reduce_loss(statistics)` | Return `(value, has_data)`. Override for an objective-owned composite Flax PyTree. |
| `with_gradients(stats, gradients, params)` | Return `stats` whose derivative in `params` is `gradients`, for a loss that states its own gradient rule. |
| `apply_effects(variables, effects)` | Return nonparameter replacements from additive accepted-window observations. Required when the objective emits effects. |
| `evaluate(variables, batch, step)` | Return an artifact, a tuple of artifacts, or `None`. The base method returns `None`. |
| `preview(variables, batch, step, *, scored=None)` | Return display artifacts for a tracker, or `None`. The base method reuses `scored`, the first scoring artifacts. |
| `ema` | Optional `EMASpec`; the base objective uses `None`. |
| `optimizer(tx, *, accumulation)` | Return the optimizer `Trainer` steps `params` with, from the one it was handed; the base method returns `tx`. Per-network optimizers use `optax.multi_transform` and `optax.conditionally_mask`. |
| `averages(update)` | Return whether the update after `update` committed ones moves the EMA; the base method averages every update. |
| `artifact` | Optional description of the objective's evaluation artifact type. |

`Step.step` counts accepted microbatches. Its `key` derives from consumed attempts, including rejected ones. `ema` holds selected averaged leaves overlaid onto the complete variables mapping, or `None`.

`Aux(metrics, variables=None, qk_stats=None, effects=None)` holds training measurements, sequential mutable replacements, QK maxima and additive deferred effects. The trainer applies the effects once per supported optimizer commit. `objective.scalar_loss(variables, batch, step)` returns a scalar and the same Aux for direct JAX differentiation.

### Collections and EMA selection

A variables tree is a nested mapping. Its outer keys name collections such as `params` and `batch_stats`; leaves are arrays such as a dense kernel or a running mean. The optimizer updates the `params` collection. A mutable Linen call returns replacement state collections, which the objective supplies through `Aux.variables`. See the [BatchNorm example](../concepts/objectives.md#non-parameter-state).

`EMASpec(decay, select=everything)` comes from `dew.objectives.base`. `decay` maps completed optimizer-update count to a scalar. `select` accepts a tuple of keys naming a leaf; `under("params", "context_encoder")` selects that subtree, and `everything` selects all leaves. EMA arithmetic uses at least fp32 and preserves explicit fp64, then rounds each result to the initialized EMA leaf dtype. Unit decay selects the frozen leaf exactly. Router bias updates also retain the initialized bias dtype; integer load comparisons avoid converting large counts to floats.

### Input descriptions

Import `Field`, `Condition`, and `InputSpec` from `dew.inputs` or `dew`.

```text
Field(key, shape)
Condition(encoder, field="text", unconditional="")
InputSpec(sample, conditions={}, mask=None)
```

`Field.shape` is the shape of one sample, excluding batch. A `Condition` connects a condition encoder to its tokenized batch field. `InputSpec.conditions` maps the model keyword, such as `textcontext`, to a `Condition`. Each condition must read a distinct batch field. `mask` is an optional `Field` for a binary image mask, used for explicit masked-image latent conditioning. `InputSpec.tokenize(captions)` returns tokenized fields for the conditions whose encoder reads captions (`reads_captions`); an empty condition mapping returns no fields. An encoder of another modality, such as `HFAudio`, reads the field its dataset writes. Encoders define tokenization, parameters, and encoded output types.

### Mesh and layout

Import these from `dew.training`:

```text
MeshSpec(fsdp=1, expert=1, tensor=1, sequence=1, stage=1, microbatches=None, replicas=1)
Layout(rules=DEFAULT_RULES, min_shard=65536, tolerance=0.02, host=(), host_parameters=())
MeshSpec(...).build(devices=None)
```

`MeshSpec.build` uses the supplied devices, or JAX's visible devices. The specified factors must divide the device count, and data parallelism takes the remaining factor. Explicit pipeline microbatches require `stage > 1` and a positive multiple of the stage count. With `replicas` above 1, `build` makes a hybrid mesh whose data axis spans that many groups of granules (TPU slices, GPU hosts or NVLink domains, or processes whose devices all share one slice), and every other axis stays inside a group; see [training on several nodes](../guides/multi-node.md#mesh-layout-across-nodes). A sequence axis above 1 splits every attention call's positions, and each call chooses between the all-to-all and the gather exchange from its shape.

`Layout` has these fields:

- `rules` is an ordered sequence of logical-axis rules, or a mapping of overrides that update the default table. When dimensions compete for one mesh axis, rule order decides which one gets it, and a dimension the axis size does not divide cannot use that axis. The valid parameter mesh axes are `fsdp`, `expert` and `tensor`.
- `min_shard` is the size, in elements and not bytes, below which a parameter stays replicated, because sharding it would cost more in collectives than it saves in memory.
- `tolerance` is the fraction of shardable parameter elements a layout may leave replicated before `check` refuses it.
- `host` names train-state fields out of `params`, `opt_state` and `ema`. A named `opt_state` or `ema` stays in pinned host memory between steps, and the step fetches it to the device. Naming `params` instead puts the whole `TrainState` on the CPU, including optimizer state, EMA and accumulation. The optimizer transaction then runs on a CPU companion of the mesh, and on every process the runtime CPU device count must match the accelerator count before JAX initializes.
- `host_parameters` holds glob patterns over logical parameter paths (`params/layers_*`) that an inference placement keeps in pinned host memory. Only the `offloaded` placement reads them, and `check` refuses a layout that names them for any other placement.

`shardings(mesh, tree)` returns a matching tree of placements, and `check(params, shardings, mesh)` checks for excessive replication. See [distributed training](../concepts/distributed.md).

## Trainer

Import `Trainer` from `dew` or `dew.training`.

```text
Trainer(objective, optimizer, *, key,
        mesh=MeshSpec(), layout=Layout(), accumulation=1,
        dynamic_scale=False, checkpoints=None, tracker=None,
        step=None, rollout=None, profile=None)
```

`objective` is an initialized objective object, and `optimizer` is an Optax gradient transformation or a `dew.config.OptimConfig`. `fit` builds a config over the optimizer updates its `steps` make, `steps // accumulation`, so the config's schedule spans that run; `OptimConfig.weight_decay` says which parameters its decay spares. The required JAX `key` seeds initialization and the run. `mesh` and `layout` describe placement. The optional objects turn on checkpoints, tracking, host-side rollouts and profiling.

`accumulation` counts accepted microbatches per effective window. Shared means use a weighted gradient accumulator with at least fp32 precision, and keep float64 when it is enabled in JAX. A TPU has no float64 (XLA rewrites it into pairs of float32, which are not IEEE doubles), so a trainer whose parameters are stored in float64 on a TPU mesh is refused when it places the state. Each finalized gradient enters Optax in its parameter's dtype, and partially accumulated gradients keep the wider working dtype.

Composite statistics keep independent normalizers, and the inputs for scalar-VJP replay. Parameters and the EMA stay fixed within the window, and sequential mutable replacements keep the snapshots they originally read. `dynamic_scale=True` persists the scale and its history, and rejects nonfinite working or optimizer-input gradients without discarding the accepted prefix.

A custom `step(objective, optimizer)` factory is responsible for the accepted and update clocks, the scaler, the EMA and mutable writes. Its body returns `(state, loss, aux)`, and the common compiled wrapper advances the attempt counter `state.step`. A host `rollout` produces the actual training arrays once per consumed attempt, before differentiation or replay.

### Train

```text
fit(dataset, *, steps, log_every=100, eval_every=None,
    checkpoint_every=None, metrics=(), preview=False) -> TrainState
```

- `dataset` is a `Dataset`.
- `steps` is the final target, including a restored step count.
- `log_every` is the interval between training log lines.
- `eval_every=None` disables evaluation. With an interval, evaluation also runs at the end.
- `checkpoint_every` controls periodic saves when a checkpointer exists. The normal completion path can still save a final checkpoint when a checkpointer is present.
- `metrics` reduce evaluation artifacts and require evaluation to be enabled.
- `preview=True` explicitly requests generated/display artifacts during evaluation; adding a scalar tracker alone does not request them.

Checkpointable data must supply the consumed iterator position. A failed scaled transaction advances attempted work and scaler history while preserving the earlier accepted prefix. Completely inactive windows close accepted slots without an optimizer call, weight decay, EMA, or deferred effects. Auxiliary-only windows can still be active.

`fit` is responsible for closing the iterators the dataset returns. It closes training read-ahead before the final evaluation, and closes validation passes on success or failure. A restored fit that is already at its target does no data, evaluation, compilation or save work. On exit it stops its trace and waits for pending checkpoint writes, but it does not close the checkpointer or the tracker, which it only borrowed. Cleanup failures are attached to the primary error.

### Initialize, restore, and compile

`initial_state()` constructs an unplaced initial `TrainState`. `place()` returns `(state, shardings, position)`, restoring from the configured checkpointer when available. Eager and placed initialization can differ in low floating-point bits across backends; compare the actual path used by your run.

An `OptimConfig` is built by `fit` over the run's optimizer updates. If you need `initial_state`, `place` or `compile` before `fit`, pass `OptimConfig(...).build(steps)` to `Trainer` instead, where `steps` is the run's length in optimizer updates.

`compile(state, batch)` returns `compiled(state, batch) -> (state, loss, metrics, loss_finite, accepted)`. The scaler is part of `TrainState`. `loss_finite` and `accepted` are separate, because a finite scalar loss can come with a nonfinite gradient that is rejected. The callable consumes the state it is given (`donate_argnums=0`), and the returned state takes over its buffers, so write `new = compiled(old, batch)` and keep no reference to the old state. The batch is not donated, because the loader still owns it.

## TrainState

Import `TrainState` from `dew.training`.

| Field | Meaning |
|---|---|
| `step` | Completed attempted batches, including rejected work |
| `microstep` | Accepted microbatches, including finite zero-support slots |
| `updates` | Committed supported optimizer updates |
| `variables` | Complete Flax variables, including the inner `params` collection |
| `opt_state` | Underlying optimizer state, advanced only on commits |
| `ema` | Selected moving-average variables or `None` |
| `key` | Immutable root training key |
| `scale` | Dynamic loss scaler with finite-history state, or `None` |
| `window_size` | Persisted accumulation length, checked on restore |
| `accumulation` | Actual partial gradient/statistic/effect/replay state, or `None` |

`state.averaged` overlays EMA leaves onto `state.variables` and raises when no EMA is configured. For fixed window length K, the accepted fill is `microstep % K`. A partial window is checkpointed without flushing.

## Dataset and Loading

Import `Dataset`, `DataPartition` and `Loading` from `dew.data`.

```text
Dataset(train, val, records, batch, ramp=None)
Dataset.from_records(records, *, batch, seed=0, validation=None, loading=Loading())
Dataset.from_grain(train, *, batch, validation=None, records=None, loading=Loading())
Dataset.from_torch(dataset, *, batch, fields=None, seed=0, validation=None, loading=Loading())
DataPartition(index=0, count=1, readers=1, reader=0)
Loading(workers=0, threads=64, read_buffer=128, worker_buffer=2)
```

`from_records` reads records held in memory: a mapping of equal-length columns, a sequence of per-record mappings, or a source with `__len__` and `__getitem__`. Its training stream reshuffles from `seed` every epoch and saves a global record position. `validation` is one ordered pass over every record, with its last batch padded by repeated rows that `VALID_ROWS` marks. `from_grain` reads a Grain pipeline the caller built, in the caller's order. `from_torch` reads a map-style `torch.utils.data.Dataset` as `from_records` reads a source, naming a tuple sample's entries with `fields`; it refuses a `DataLoader` and an `IterableDataset`.

`train(partition)` opens a training iterator, and `val(partition)` opens one finite validation pass; `val` is `None` when there is no validation data. Each iterator reads the share of every global batch that its `DataPartition` names. The partition splits each batch into `count` disjoint shares; `index` is the share to read; `readers` is the number of processes that read that share alike; and `reader` is which of those processes this one is. `DataPartition.of(mesh)` is the share a process reads on a mesh, and `DataPartition()` is every row.

`records` is the known training-record count or `None`, and `batch` is the global batch size. `steps_per_epoch` is records divided by batch with integer division, or `None`. `epoch_steps(epochs=1)` requires a finite record count. `ramp` is set when the run grows its batch over its first records, and `batch` is then the batch size the ramp ends at.

Each factory call must return a fresh, exclusively owned iterator. Ordinary `close()` is finalization and must not race `next()` or checkpoint operations. A source that needs to interrupt blocking reads may additionally implement `request_stop()`: a thread-safe, nonblocking, idempotent signal, safe alongside both `next()` and `close()`. Tokenized wrappers forward these operations.

`DevicePrefetchIterator(iterator, mesh, depth=2, source_state=None)` in `dew.training.distributed` takes ownership of `iterator` once construction succeeds, and starts its worker on the first `next()`. Restoring a saved position also happens there, so a restoration failure unwinds through the iterator it already owns. Closing an unused iterator only finalizes it, without a read or a restoration. Use it in a `with` block, or call `close(timeout=5.0)`, even when a loop consumes only a fixed number of batches.

Depth must be positive. At most `depth` device batches are queued, plus at most one in flight, not counting the consumer and upstream buffers. `source_state` describes the last delivered batch, never speculative read-ahead. An EOF or a source failure is reported after the batches queued before it. An early close discards unread data and speculative failures, but reports finalization failures.

The prefetch worker runs the iteration, the checkpoint operations and the final close of the source. Closing requests cancellation, discards queued batches and joins the worker. A `TimeoutError` means the source read, placement or finalization did not stop when asked. The thread is not killed, and its in-flight references may remain. Cancellation stays requested, and you can retry the close. After close, further iteration stops.

Grain limits how cleanly Dew can shut a loader down. The installed `DataLoaderIterator` has no public close, so Dew releases the references it owns without reaching into private iterators or changing the sampling pipeline. In local-record probes the source and child processes are released, but Grain does not promise a deterministic shutdown. Grain also keeps a process-wide thread pool that deletes shared memory. Dew calls `DatasetIterator.close()` on the iteration thread, but shutting down the read executor does not wait for record reads already running. So a blocked upstream read is outside Dew's shutdown guarantee.

`Loading` sets Grain's concurrency and buffers for the built-in specifications. The default, which is also Grain's, starts no worker processes and reads with threads of the training process. Raise `workers` once you have measured that the input pipeline is the bottleneck. See [data preparation](../concepts/data.md) for field layouts and process partitioning.

Every dataset specification has `seed` and `loading` as keyword-only fields from `DatasetSpec`, and every one loads with `spec.load(batch=, tokenize=None)`. A specification that writes no captions raises `TypeError` when given a `tokenize` reader, because it cannot use one.

An image specification takes its validation data from `val_split`, one of the dataset's own splits, limited to `val_batches` batches. With `val_split` unset, it holds out the first `val_batches * batch` records of the training source instead.

`Dataset.from_grain(train, *, batch, validation=None, records=None, loading=Loading())` builds a run over Grain pipelines the caller assembled. A `MapDataset` is repeated and cut into the reader's share, and its position is saved as one global record count. A pipeline that is read as it comes is passed as a function of the `DataPartition` that builds the `IterDataset` of that share. That pipeline is batched where it is and reports Grain's own iterator state.

A token corpus is a `TokenSource`: `TokenBytes` over a `.bin` file or `TokenRecords` over ArrayRecord shards of token arrays. `TokenWindows` and `PackedTokens` read `path` as a directory of `train` and `val` files and take whichever store their suffix names, so the same corpus gives the same windows and the same packing plan in both. `TokenWindows(hub=HubText(name, split=, column=, tokenizer=, options=))` reads a Hugging Face text split instead, tokenized once into `dew_cache_dir()/tokens`, which is what `dew.data.load("hf/<name>", tokenizer=, seq_len=)` builds.


## Checkpoints

Import `Checkpoints` from `dew`.

`Checkpoints(directory, *, keep=2, local_directory=None, local_every=None)` configures persistent storage and optional local emergency checkpoints. Local directory and cadence must be specified together. The object opens storage on use.

`save(step, state, saved, metrics=None, *, share=None)` schedules a persistent save. `saved` is the iterator position as bytes, or `None`, and `share` is the `DataPartition` that the stream read; a position requires a share. `wait()` waits for pending writes, and `latest` reports the newest eligible checkpoint. `restore(template=None, step=None, *, share=None)` returns the restored state data and the iterator position that the reader of `share` resumes from, or `None` for the position when no share is given. A template controls structure and placement. `path(step)` gives the persistent checkpoint location.

This object does not write `run.json`; run configuration saving is separate. Use the [complete resume example](../guides/checkpoints.md) before adapting these lower-level calls.

## Language modeling and generation

Import `LMObjective` from `dew.objectives.lm`.

```text
LMObjective(model, seq_len, *, ema_decay=None, pad_id=None, head_chunks=4, head_tile=None,
            samples=None, variables=None, balance_rate=None, aux_loss_alpha=None,
            seq_aux=True, loss_role=None, mtp_weight=None, z_loss=0.0, router_z_loss=0.0,
            qk_stats=False, indexer=None, token_accuracy=True, processor=None)
IndexerTraining(phase, weight=1.0)
```

`model` may be a decoder bundle that `Pretrained.load` returned. The objective then reads the bundle's model, initial variables and processor, and `pipeline` decodes with that processor; passing `variables=` or `processor=` as well overrides that part. An adapted bundle (`bundle.adapt(LoRA(...), key=)`) trains only the adapter's factors, and its `save` and `export` merge them into the kernels. The model must implement Linen `hidden_states(tokens, train=..., positions=..., segment_ids=...)`, returning `(B, S, D)`, and `output_table()`, returning the `OutputTable` (the stored vocabulary matrix and its orientation, bias, softcap and precision) that its logits contract (`dew.nn.protocols`). Prediction-depth training needs `mtp_hidden_states` and compatible prediction-depth configuration. Mutable router, QK and indexer collections are required when their options are enabled.

`variables` is the tree training starts from, either whole or split by `dew.objectives.base.freeze` or an adapter into the part that trains and the part that stays frozen. `loss_role` requires aligned `text_roles`. `pad_id` masks matching targets. `head_chunks` controls vocabulary tiling, and `head_tile` the head's backward tile (`'whole'`, `'tiled'` or a tile shape; `None` picks one for the objective); `samples` configures text previews. `ema_decay` defaults to `None`, which trains without an averaged copy; a decay such as `0.999` keeps one that evaluation and previews read, and `1.0` retains a frozen one. Routing balance, auxiliary loss, prediction-depth weight, and QK statistics require matching model computation. These interfaces make `LMObjective` specific to compatible decoders. `z_loss` adds PaLM's auxiliary term, the coefficient times the squared log partition of every counted prediction; zero adds nothing. `router_z_loss` is the routers' own z-loss (ST-MoE), and zero adds nothing. `token_accuracy=False` drops the `token_accuracy` metric and the pass over every logit it costs.

`indexer` trains DeepSeek-V3.2's lightning indexer on a model whose `mla` mixer names `index_n_heads` and `index_head_dim`. `IndexerTraining("warmup")` needs a mixer without `index_topk`. The model then runs dense attention, `init` puts only the indexer in `params` and the rest of the tree under `frozen` (a `variables` tree may omit the indexer, as a dense checkpoint does), and the loss is the KL of the indexer's softmax from the attention distribution, reported as `indexer_kl`. `IndexerTraining("sparse")` needs a mixer with `index_topk`. The whole tree trains: the cross entropy trains the main weights, and the KL over the selected keys trains the indexer, whose inputs are detached. A warm-up checkpoint's split tree is accepted as `variables`. `weight` scales the KL term.

Import `generate`, `Sampling` and `Generation` from `dew.sampling`:

```text
generate(model, params, inputs, max_new_tokens, *, key=None,
         sampling=Sampling(), n=1, logits=None, stopping=None, strategy=None) -> Generation
Sampling(temperature=1.0, top_k=None, eos_id=None, pad_id=None, top_p=1.0, min_p=0.0,
         repetition_penalty=1.0, presence_penalty=0.0, frequency_penalty=0.0,
         no_repeat_ngram_size=0, min_new_tokens=0, typical_p=1.0, stop=())
```

`params` is the complete variables tree. `inputs` is a `ModelInputs` from `dew.nn.inputs`, or an integer `(B, P)` array normalized to all-valid text. `ModelInputs.token_fields["attention_mask"]` identifies real token slots; there is no separate generation length argument. Every row needs a real token. Only real tokens count against `model.max_seq_len`. Conditioning arrays are batch-aligned and used during prefill; decode keeps the model's cached logical positions. `key` is an integer seed or a JAX key; `key=n` is `jax.random.key(n)`.

The compiled decoder uses one padded input shape, with a cache cursor per row and a fixed trip count. Finished rows keep their cached state. On a mesh, rows are split over the batch axes and the result keeps that sharding. Each process passes in its own rows, with the same row count and padded width on every process, reads them back with `Generation.host()`, and passes the same `n`. Keys fold in the global row index, so a pool draws what one process draws for the same rows. Invalid input on one rank raises on all ranks before device execution.

`n` is the number of continuations drawn per prompt and must be a positive integer. A prompt's continuations share its prefill and then run one after another on the device, with the prompts batched as before for each continuation. So decode time grows with `n`, but each continuation reuses the cache working memory of the one before, and only the output storage grows with `n`.

A prompt's key is its global row key. Continuation zero draws with that key, so `n=1` and continuation zero of a larger request are the same draw. Continuation `j` folds `j` into the key, so raising `n` leaves the continuations already drawn unchanged. On a mesh, row padding pads the prompts before their continuations exist, so a prompt's `n` rows stay together on the process that asked for them and `host()` drops only the padded prompts' rows.

`Sampling.eos_id` accepts an integer or a tuple of IDs, and any of them terminates a row; the value is normalized to an immutable tuple. `Sampling` holds the common generation controls as one value, and each default changes nothing. The controls are:

- the repetition penalty (Transformers' `RepetitionPenalty`), over the prompt and the generated tokens;
- vLLM's presence and frequency penalties, over the generated tokens;
- `no_repeat_ngram_size`;
- `min_new_tokens`, which holds back EOS and so needs an `eos_id`;
- temperature, top-k, nucleus top-p, relative min-p and typical-p filtering.

`transforms()` compiles them in Transformers' `_get_logits_processor` order, which `dew.sampling.text.ordered_transforms` defines, with vLLM's penalties placed beside the repetition penalty where vLLM applies them. At least one token survives the filters. Zero temperature selects the argmax and runs no filter.

`stop` holds strings that end a row. They are compiled against a tokenizer's vocabulary, so `generate` refuses them and a `TextGeneration` compiles them through its processor. A `Sampling` value is a convenience over the components below, and it compiles to those transforms plus the EOS criterion. When a task runs the policy, an `eos_id` or `pad_id` left `None` takes the task's own value. `generate` on its own stops on no EOS unless one is named, and pads with 0.

`Generation.tokens` includes the original prompt and has shape `(B * n, P + max_new_tokens)`, where `B` is the number of placed prompt rows. The rows hold prompt zero's `n` continuations first, then the other prompts' in request order. The other fields are aligned with the rows of `tokens`:

- `lengths`, the count of response tokens including EOS;
- `terminated`, true when EOS ended the row and false when the token budget did (slots after termination hold `Sampling.pad_id`);
- `behavior_log_probs` and `raw_log_probs`, of shape `(B * n, max_new_tokens)`, the likelihoods of each action under the filtered distribution that drew it and under the unmodified policy.

`rows` counts this process's real prompts times `n`. `host()` returns the record over host arrays of those rows, and `text` decodes them through the task's processor, one string per row.

`LMObjective.per_token_log_probs(params, tokens, left_padding=...)` scores the raw policy. It left-aligns real tokens for the forward and restores the original next-token alignment. Unscored padding slots are zero. `SampledRollout` records the raw sampling-time values as `old_log_probs` and preserves actual draws as `behavior_log_probs`, in the packed layout `GRPOObjective.packed_log_probs` rescores. Reward text excludes EOS and padding. When a batch carries no `old_log_probs`, GRPO uses the behavior probabilities as the old policy and its behavior corrections follow verl's bypass mode. The next-token objective refuses models declaring `causal=False`.

### Decoding components

Decoding has three extension points: logits transforms, stopping criteria and strategies. Import the protocols from `dew.sampling` and the built-ins from `dew.sampling.decoding`.

```text
LogitsTransform: (StepState, logits[rows, vocab]) -> logits[rows, vocab]
Stopping:        (StepState, drawn_tokens[rows]) -> finished[rows]
Strategy:        (DecoderState, StepState, DecodeOps, transform, stopping, budget, n) -> Draws
StepState(tokens, valid, step, active, keys, prompt_width)
```

`StepState` is everything a transform or a criterion receives. `tokens` is the fixed-capacity buffer of the prompt followed by the slots for generated tokens, `[rows, prompt_width + max_new_tokens]`. `valid` marks the slots that hold a real token, so a row reads its own history whatever padding its prompt batch needed. `step` counts the tokens a row has committed, `active` marks the rows still generating, and `keys` holds one PRNG key per row. `state.history()` returns each row's real tokens left aligned with their count, `prompt_history()` returns the prompt region, and `total()` returns the real token count. A transform never sees model parameters or cache internals.

`logits` is the whole transform chain, in the order it runs. Left as `None`, it is the chain `sampling` compiles to. An explicit sequence replaces that chain entirely, and `()` runs no transform, so a caller who needs an order `Sampling` does not produce can write the order they want. A call's `logits` replaces the task's bound chain. An explicit `sampling=` on a call also clears a bound chain, because that chain was built for the policy the call just replaced.

`stopping` adds to the policy instead of replacing it. An explicit sequence runs beside the policy's EOS criterion, so naming a criterion cannot drop EOS termination. Criteria combine with OR and run after every committed token. The token that fired a criterion is emitted with its likelihoods, and later slots hold `pad_id` with zero likelihood.

Built-in transforms are pytrees, so a configuration that holds arrays is passed as data and does not become part of a compilation cache key. A plain function works too, and `jax.tree_util.Partial(fn, array)` passes array configuration to one. Everything runs inside the compiled loop, with no host callback. Across a pool, the resolved components are compared by their structure and by the contents of their configuration arrays, so two ranks that ban different tokens are refused instead of each running its own policy.

```python
import jax
import jax.numpy as jnp
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.sampling import Beam, Sampling, Speculative, decoding, generate

# An untrained decoder with one prediction depth, which Speculative drafts with.
model = CausalTransformer(vocab_size=64, emb_features=32, num_layers=1, num_heads=2,
                          mlp_features=64, max_seq_len=64, num_nextn_predict_layers=1)
variables = model.init(jax.random.key(0), jnp.zeros((1, 8), jnp.int32))
prompts = [[5, 6, 7], [8, 9, 10]]


def favor_short(state, logits):
    """Raise the end token's score once a row has drawn eight tokens."""
    return logits.at[:, 2].add(jnp.where(state.step >= 8, 3.0, 0.0))


# The common controls are one value, compiled in Transformers' order.
policy = Sampling(temperature=0.8, top_p=0.9, repetition_penalty=1.1, frequency_penalty=0.4,
                  no_repeat_ngram_size=3, eos_id=2, pad_id=0)
drawn = generate(model, variables, prompts, 32, key=0, sampling=policy,
                 stopping=(decoding.MaxNewTokens(24),))

# An explicit chain is complete: a transform of your own joins the policy's.
nudged = generate(model, variables, prompts, 32, key=0, sampling=policy,
                  logits=(favor_short, *policy.transforms()))

# The same request as a deterministic search over four beams, returning two.
searched = generate(model, variables, prompts, 32, key=0,
                    sampling=Sampling(eos_id=2, pad_id=0),
                    logits=(decoding.NoRepeatNGram(3),),
                    strategy=Beam(width=4, length_penalty=1.0), n=2)

# Drafted by the model's own prediction depths and verified by the model:
# tokens are distributed exactly as ordinary sampling under this call's
# policy; accepted drafts save target forwards, which untrained depths rarely give.
drafted = generate(model, variables, prompts, 32, key=0,
                   sampling=Sampling(temperature=0.8, top_p=0.9, eos_id=2),
                   strategy=Speculative(block=4))
print(drawn.tokens.shape, nudged.tokens.shape, searched.tokens.shape, drafted.tokens.shape)
```

The transforms port `transformers/generation/logits_process.py` from Transformers 5.16.1, with each row reading its own unpadded history instead of the batch's padded width.

| Transform | Reference | Notes |
| --- | --- | --- |
| `Temperature(value)` | `TemperatureLogitsWarper` | |
| `TopK(k)` | `TopKLogitsWarper` | |
| `TopP(p)` | `TopPLogitsWarper` | keeps the best token |
| `MinP(p)` | `MinPLogitsWarper` | |
| `Typical(mass)` | `TypicalLogitsWarper` | |
| `EpsilonCutoff(epsilon)` | `EpsilonLogitsWarper` | |
| `EtaCutoff(epsilon)` | `EtaLogitsWarper` | |
| `TopH(fraction=1.0)` | `TopHLogitsWarper` | over the reference's fixed head of 100 |
| `Greedy()` | greedy search | zero on the argmax, `-inf` elsewhere; what `temperature=0` compiles to |
| `Renormalize()` | `LogitNormalization` | shifts every score by one constant, so no later filter and neither likelihood can see it |
| `RemoveInvalidValues()` | `InfNanRemoveLogitsProcessor` | the only transform that repairs a broken distribution |
| `RepetitionPenalty(penalty)` | `RepetitionPenaltyLogitsProcessor` | over the row's valid prompt and drawn tokens |
| `PromptRepetitionPenalty(penalty)` | `EncoderRepetitionPenaltyLogitsProcessor` | the prompt is the encoder input; the reference inverts the argument |
| `FrequencyPenalty(penalty)` | vLLM `model_executor/layers/utils.py` | subtracts the penalty times each generated token's count |
| `PresencePenalty(penalty)` | vLLM `model_executor/layers/utils.py` | subtracts the penalty from every generated token |
| `NoRepeatNGram(size)` | `NoRepeatNGramLogitsProcessor` | |
| `PromptNoRepeatNGram(size)` | `EncoderNoRepeatNGramLogitsProcessor` | n-grams of the prompt |
| `sequence_bias(entries)` | `SequenceBiasLogitsProcessor` | `(token ids, bias)` pairs compiled into one table |
| `bad_words(ids, eos_id=None)` | `NoBadWordsLogitsProcessor` | a `-inf` table; single-token EOS sequences are dropped |
| `SuppressTokens(tokens)` | `SuppressTokensLogitsProcessor` | |
| `BeginSuppressTokens(tokens, offset=0)` | `SuppressTokensAtBeginLogitsProcessor` | `offset` is the generated index to suppress at: 0 for the first drawn token, 1 when a forced BOS takes that slot |
| `ForcedBOS(token)` | `ForcedBOSTokenLogitsProcessor` | |
| `ForcedEOS(eos, max_length=None)` | `ForcedEOSTokenLogitsProcessor` | `max_length` counts prompt and generated tokens; `None` is the request's own end, so a per-call budget moves it |
| `MinLength(length, eos)` | `MinLengthLogitsProcessor` | suppresses EOS below a total length |
| `MinNewTokens(count, eos)` | `MinNewTokensLengthLogitsProcessor` | suppresses EOS below a generated count |
| `ExponentialDecayLengthPenalty(start, factor, eos)` | `ExponentialDecayLengthPenalty` | `start` counts generated tokens |

| Criterion | Reference | Notes |
| --- | --- | --- |
| `EndOfSequence(eos)` | `EosTokenCriteria` | what `Sampling.eos_id` compiles to |
| `MaxNewTokens(count)` | | a budget below `max_new_tokens` |
| `MaxLength(length)` | `MaxLengthCriteria` | prompt and generated tokens together |
| `stop_strings(tokenizer, strings, vocab_size=None)` | `StopStringCriteria` | |

`stop_strings` reads the tokenizer once, on the host. It compiles where each token's text can fall inside each stop string, and how many of the string's trailing units the token's start can cover. The criterion then runs entirely on device and never decodes. A string counts only when it touches the token just drawn. So a string produced earlier does not stop the row later, while a string spelled across several tokens or overhanging either end does stop it.

A byte-level or byte-fallback vocabulary is read through its byte spelling and matched over UTF-8 bytes, so a stop string still ends the row when two tokens split one of its code points. Every other vocabulary is read through `convert_tokens_to_string` behind an ordinary prefix, because a decoder adds or removes a leading space depending on what came before. `tokenizer` is a Transformers tokenizer or a task `Processor` holding one, and `vocab_size` sizes the table for a model whose head is wider than the vocabulary.

An active row can be drawn from only when every score is finite or `-inf` and at least one is finite. A NaN or a `+inf` beside a finite score makes the draw arbitrary, and a row with nothing finite has no distribution at all, so in both cases `generate` raises instead of returning an index. A zero-temperature policy does not collapse such a row onto an argmax; it passes the row through unchanged. The model's own distribution must be defined too, because an undefined one cannot be reported truthfully. So generation raises there as well, rather than return a NaN likelihood, whatever a later `RemoveInvalidValues()` does to the scores. Add that transform to repair scores that the chain itself made invalid.

#### Strategies

```text
Sample(grammar=None)
Beam(width=1, length_penalty=1.0, early_stopping=False, stop_ids=1)
Speculative(block=4, confidence=0.0)
```

`Sample` draws every row independently, and a request without a strategy runs it. A `grammar` from `dew.sampling.guided` (`regex` or `json_schema`) keeps every draw inside the grammar.

`Beam` is deterministic beam search, with the bookkeeping of `_beam_search` in Transformers 5.16.1. Each step keeps the best `(1 + stop_ids) * width` continuations, so that `width` live beams always remain. A stopping criterion moves a beam into the completed set, with its score divided by its generated length raised to `length_penalty`, and `early_stopping` takes the reference's `False`, `True` and `"never"`. The prompt is prefilled once and copied into `width` cache rows, which each step reparents, so a branched beam decodes exactly like a separately selected prefix. `n` is how many completed hypotheses to return, and `n > width` is an error.

A selected path comes from a search, so its behaviour log probability is zero, while its raw log probabilities are still the model's own. Sampling with beams is refused, because the marginal probability of a selected beam is not the per-step candidate probability, so there is no correct behaviour likelihood to record.

`Speculative` drafts with the model's own prediction depths and verifies with the model, following algorithm 1 of [arXiv 2211.17192](https://arxiv.org/abs/2211.17192) as `_speculative_sampling` applies it. The first candidate of a block is an ordinary target draw, so it is always accepted, and each depth proposes the next candidate from the previous hidden state and the candidate's embedding. A proposal is accepted with probability `min(1, p(x) / q(x))`, where `p` is the target's post-transform distribution and `q` the draft's actual one. The first rejection draws from the normalized positive part of `p - q`, and a block with nothing rejected draws a bonus token from `p`. So the emitted tokens have exactly the distribution `Sample` gives them, though not draw for draw at one seed.

Every emitted action records the target's post-transform log probability as its behaviour value and the model's own as its raw value. The draft's `q`, the acceptance probability and the residual are never recorded. `confidence` stops the draft after the first candidate the draft is less sure of, as `ConfidenceCriteria` does. A candidate the draft never offered was not rejected, so the block then ends on an ordinary target draw. A model without prediction depths is refused, and there is no silent fallback.

The target cache is saved before a block and the accepted prefix is replayed into it, because a recurrent mixer keeps a running summary that no cursor can rewind. The prediction cache is rebuilt by the same rule. Depth `d`'s entry for token `t` reads depth `d - 1`'s hidden state at `t - 1` and `t`'s own prepared embedding, at `t`'s own coordinate, and the target is depth zero's predecessor. That is the input `mtp_hidden_states` trains the depths on, the one `MTPCandidateGenerator` corrects with, and the one `Qwen3_5MultiTokenPredictor.forward` takes. Each depth keeps its last state across blocks, so no entry is lost at a block boundary. The prompt seeds every depth from the embeddings the prefill already prepared, media replacements included, without running an encoder again.

A depth becomes usable after enough real tokens have preceded it. Rotary offsets and repeated image coordinates do not change that count. A newly available predecessor is retained even if its next depth cannot yet write a cache entry.

A model with a block drafter (DSpark, as DeepSeek-V4.1 and DeepSeek-V4-Flash-0731 ship it) drafts a block's later candidates in one pass instead of chaining depths. The drafter reads the context its target layers record. V4.1's averages each target layer's streams at its attention input, and V4-Flash-0731's at the layer's output. The prefill seeds it with the prompt's context, and each replayed block appends the context of its kept positions. As the pass reaches each position, the strategy draws that candidate from the drafter's logits for the position, Markov bias included, so `block - 1` may not exceed the drafter's own block size. A standalone drafter checkpoint, such as RadixArk/Kimi-K3-DSpark, which reads another model's hidden states and drafts through that model's embedding and head, is refused by design (`DrafterRefused`).

A continuing block emits at least two tokens when the remaining budget allows it. An EOS or the budget can truncate the block earlier. Thus `ceil(budget / 2)` iterations bound the loop. Each active block uses two target forwards. Once every row has finished, later iterations run no model call. The predicate reduces the whole batch, so every rank skips the same blocks.

#### Existing JAX decoding

[T5X decoding](https://t5x.readthedocs.io/en/latest/api_reference/t5x.decoding.html) provides JAX sampling and beam search with model callbacks and explicit cache state. Its [upstream tests](https://github.com/google-research/t5x/blob/0e2a6a810179baf684c0d3a74ffaacdbe9bd305c/t5x/decoding_test.py) cover uneven prefixes, cached starts, callbacks and probability scores. The comparison below uses that pinned revision.

| Capability | T5X contract | Fit for Dew |
| --- | --- | --- |
| Model callback | [`DecodingState` and `tokens_to_logits`](https://github.com/google-research/t5x/blob/0e2a6a810179baf684c0d3a74ffaacdbe9bd305c/t5x/decoding.py#L34-L127) separate the loop from model execution. | Adapt this separation. `Strategy` receives model operations; parameters stay outside the row carry. |
| Likelihoods | [`temperature_sample`](https://github.com/google-research/t5x/blob/0e2a6a810179baf684c0d3a74ffaacdbe9bd305c/t5x/decoding.py#L530-L584) returns one cumulative score per continuation. `rescale_log_probs=False` still scores logits after the logit callback. | Reference only. Rollouts need both original-model and actual-policy log probabilities for every emitted token. |
| Multiple continuations | [Expansion and final sorting](https://github.com/google-research/t5x/blob/0e2a6a810179baf684c0d3a74ffaacdbe9bd305c/t5x/decoding.py#L351-L402) duplicate the cache and order each prompt's samples by score. | Reference only. Dew preserves continuation identities and their row-owned keys. `Sample` shares prefill and reuses its working cache across continuations. |
| Cache reparenting | [`cache_map` and `cache_gather_beams`](https://github.com/google-research/t5x/blob/0e2a6a810179baf684c0d3a74ffaacdbe9bd305c/t5x/decoding.py#L727-L854) use named exclusions and an axis offset for scanned caches. | Adapt the operation, not the leaf rules. Dew's public cache has row axis zero, including GDN recurrence, MLA buffers and multimodal coordinates. |
| Beam ranking | [`brevity_penalty`](https://github.com/google-research/t5x/blob/0e2a6a810179baf684c0d3a74ffaacdbe9bd305c/t5x/decoding.py#L713-L725) uses `((5 + length) / 6) ** alpha`. | Reference only for source parity. Transformers uses generated length raised to `length_penalty`; the rankings can differ. |
| Ordered transforms | The [logit callback precedes T5X's temperature and top-k/top-p processing](https://github.com/google-research/t5x/blob/0e2a6a810179baf684c0d3a74ffaacdbe9bd305c/t5x/decoding.py#L424-L557). | A complete callback chain can adapt this interface by disabling that processing. It does not supply Dew's two likelihood tracks or speculative verification. |

Dew uses T5X as a reference and a test oracle, and adapts some of its state and cache operations. T5X is not a runtime dependency. Nothing in its JAX loops rules out ragged MoE, but Dew still has to handle row placement, inactive-row masks and collective agreement around model calls, and the T5X return contract does not cover them.

#### Source generation controls

A loaded source's `generation_config.json` is data. Dew assigns every control Transformers 5.16.1 writes there to one consumer: the native policy, a transform, a criterion, a strategy or the task. Otherwise the control is provenance only, or `PretrainedDecoder.text_generation()` refuses it and says why. The common controls become the task's `Sampling` value, `task.sampling`, so you change one with `dataclasses.replace(task.sampling, ...)`.

A source that also sets a control `Sampling` does not hold gets `task.logits`, the complete chain. The same compiler (`ordered_transforms`) builds that chain in `_get_logits_processor`'s order, with the policy's transforms in it. A source that runs beam search always gets a chain, which ends after the processors because the search picks its own continuations. An unset control, or one at the value where `generate()` adds no processor, criterion or search mode, has no effect. As upstream, beam-only and sampling-only controls are checked only when beam search or sampling is active.

Each source control has one rule for its consumer, neutral value, mode and refusal. Source value precedence remains `generation_config.json`, wrapper config, then text config. The source resolver selects the actual policy once. `Sampling` supplies convenience defaults at the request boundary; only resolved transforms, criteria, strategy and padding reach the compiled decoder. An explicit chain therefore has no unused sampling settings in its compilation or process-agreement identity.

| Control | Native mapping | Refused because |
| --- | --- | --- |
| `do_sample`, `temperature`, `top_k`, `top_p`, `min_p`, `typical_p`, `repetition_penalty`, `no_repeat_ngram_size`, `min_new_tokens`, `stop_strings`, `eos_token_id`, `pad_token_id` | `Sampling` (`stop_strings` as `stop`, compiled through the task's processor) | |
| `max_length`, `max_new_tokens` | the task's token budget | |
| `num_return_sequences` | the task's `n`, independent of any `sampling=` override | |
| `bos_token_id`, `decoder_start_token_id` | inapplicable to supplied-input causal decoding; the tokenizer prepares special tokens | |
| `max_cache_len` | capacity assertion only; it does not resize the cache or limit the request | a length above the model's `max_seq_len` |
| `encoder_repetition_penalty` | `PromptRepetitionPenalty` | |
| `encoder_no_repeat_ngram_size` | `PromptNoRepeatNGram` | |
| `sequence_bias` | `sequence_bias` | |
| `bad_words_ids` | `bad_words` | |
| `min_length` | `MinLength` | |
| `forced_bos_token_id` | `ForcedBOS` | |
| `forced_eos_token_id` | `ForcedEOS` at the request's own end, so a per-call budget moves it | |
| `suppress_tokens`, `begin_suppress_tokens` | `SuppressTokens`, `BeginSuppressTokens` | |
| `exponential_decay_length_penalty` | `ExponentialDecayLengthPenalty` | |
| `remove_invalid_values` | `RemoveInvalidValues` | |
| `renormalize_logits` | `Renormalize` | |
| `epsilon_cutoff`, `eta_cutoff`, `top_h` | `EpsilonCutoff`, `EtaCutoff`, `TopH` | |
| `use_cache` | native decoding always runs through its own cache | `use_cache=False` |
| `cache_implementation` | the fixed-capacity static cache | any other implementation |
| `cache_config` | | quantized and offloaded caches are not implemented |
| `prefill_chunk_size` | | the native prefill evaluates a prompt in one call |
| `continuous_batching_config` | | continuous batching is not implemented in the task's `generate` path; `dew.inference.Server` batches continuously as a separate object |
| `compile_config`, `disable_compile` | | the native decoder owns its compilation and always runs compiled |
| `low_memory` | | sequential beam evaluation is not implemented |
| `output_attentions`, `output_hidden_states`, `output_scores`, `output_logits` | | generation returns tokens, lengths, termination and both likelihood arrays, and none of these |
| `return_dict_in_generate` | inapplicable to the native result type; generation always returns a record | |
| `transformers_version`, `_from_model_config`, `_commit_hash`, `tokenizer_name` | provenance | |
| `num_beams`, `early_stopping`, `length_penalty` | `Beam(width, length_penalty, early_stopping)`, with `stop_ids` from the EOS ids | |
| `do_sample` with `num_beams` | | a selected beam's marginal probability is not the per-step candidate probability, so no honest behaviour likelihood exists |
| `num_beams` with `use_mtp` | | a request cannot run two strategies |
| `num_return_sequences` above `num_beams` | | a search returns at most its width |
| `num_beam_groups`, `diversity_penalty` | | diverse group beam search is not implemented |
| `constraints`, `force_words_ids` | | constrained beam search is not implemented |
| `max_time` | | a host clock cannot stop a coordinated device loop |
| `token_healing` | | retokenizing the prompt is prompt construction, not decoding |
| `guidance_scale` | | classifier-free guidance evaluates the model a second time per step |
| `penalty_alpha` | | contrastive search is a decoding strategy that is not implemented |
| `dola_layers` | | DoLa is a decoding strategy that is not implemented |
| `watermarking_config` | | no watermarking transform is implemented |
| `use_mtp`, `speculation_type`, `num_assistant_tokens` | `Speculative(block=num_assistant_tokens + 1)`, the drafted count plus the block's own target draw | a checkpoint without prediction-depth weights, or a proposer other than the model's own depths |
| `assistant_ensemble_weight` below one | | ensemble verification below one accepts a biased distribution |
| `assistant_confidence_threshold` | `Speculative(confidence=...)`, which stops the draft without changing the block size | |
| `num_assistant_tokens_schedule` | `"constant"` is the fixed block a device loop runs | any adaptive schedule |
| `assistant_early_exit` | | early-exit proposal is not implemented |
| `assistant_lookbehind`, `target_lookbehind` | | translating between two tokenizers' token spaces is not implemented |
| `prompt_lookup_num_tokens`, `max_matching_ngram_size` | | prompt lookup proposal is not implemented |
| `is_assistant` | | a source loads as a target model, not as another model's assistant |
| any other name | | the native decoder does not know the control, so it refuses instead of ignoring it |

### Inference tasks

Import `pipeline`, `TextGeneration`, `BlockGeneration`, `MaskedGeneration`, `TextToImage`, `Images`, `DenoisingInputs` and `RunProcessor` from `dew.inference`. `dew.pipeline` is the same function. [Inference](../concepts/inference.md) describes placement and the three workflows.

```text
pipeline(source, *, mesh=None, layout=None, dtype=None, param_dtype=None, ema=None, step=None,
         revision=None) -> TextGeneration | BlockGeneration | MaskedGeneration | TextToImage
Objective.pipeline(state, *, ema=None, processor=OMITTED) -> the objective's task (build_task) over
    state.averaged or state.variables; processor omitted keeps the objective's own, None clears it
Objective.build_task(variables, *, processor=OMITTED) -> TextGeneration (LM, GRPO, PPO's actor),
    MaskedGeneration (MDLM), BlockGeneration (block SFT), TextToImage (diffusion, which takes no processor)
TextGeneration(model, variables, processor=None, sampling=Sampling(), max_new_tokens=None,
               max_length=None, n=1, logits=None, stopping=(), strategy=None)
task(request, max_new_tokens=None, *, key=None, n=None, sampling=None,
     images=None, logits=None, stopping=None, strategy=None) -> Generation
task.bind(variables) -> TextGeneration      task.decode(generation) -> tuple[str, ...]
task.quantized(spec, example=((0,),)) -> TextGeneration
TextGeneration.from_run / BlockGeneration.from_run / MaskedGeneration.from_run
    (directory, *, ema=None, step=None, mesh=None, layout=None, dtype=None, param_dtype=None)
TextGeneration.from_pretrained / BlockGeneration.from_pretrained / MaskedGeneration.from_pretrained
    (repo_id, *, ema=None, step=None, mesh=None, layout=None, dtype=None, param_dtype=None)
BlockGeneration(model, variables, process, processor=None, eos_token_ids=(), pad_token_id=0,
                max_new_tokens=None, max_length=None, n=1)
task(request, max_new_tokens=None, *, key=None, n=None, process=None,
     images=None) -> CanvasGeneration
Pretrained.load(name_or_dir, *, dtype=jnp.bfloat16, param_dtype=jnp.float32, attention_impl="auto",
                max_seq_len=None, revision=None, gguf_file=None, single_file=None, dduf_file=None,
                mesh=None, layout=None, fallback=None) -> the kind it is called on, or the kind the source is
PretrainedDecoder.text_generation(*, sampling=None) -> TextGeneration
PretrainedMaskedDecoder.text_generation() -> MaskedGeneration
PretrainedBlockDecoder.block_generation() -> BlockGeneration
PretrainedPipeline.text_to_image() -> TextToImage
Pretrained.adapt(lora, *, key) -> the same kind, with the bound adapter as `adapter`
LoRA(rank, modules, alpha=None, rslora=False, dropout=0.0)
LoRA.apply(model, variables, *, key, layouts=None) -> Adapter
LoRA.load(model, variables, path, *, layouts=None) -> Adapter
Adapter.from_run(directory, *, step=None, ema=None) -> Adapter
Adapter.merge(variables) -> variables;  Adapter.save(variables, path)
Pretrained.save(directory, *, variables=None, max_shard_size="5GB")
Pretrained.push_to_hub(repo_id, *, variables=None, private=False, commit_message=..., max_shard_size="5GB")
TextToImage(model, process, inputs, params, autoencoder=None, steps=50, guidance=None,
            solver=DDIM(), grid=None, final_denoise=True, finish=None, blank=None)
TextToImage.from_objective(objective, variables) -> TextToImage
TextToImage.from_run(directory, *, ema=None, step=None, mesh=None, layout=None, dtype=None,
                     param_dtype=None)
TextToImage.from_pretrained(repo_id, *, ema=None, mesh=None, layout=None, dtype=None,
                            param_dtype=None)
LMObjective.policy(params, sampling=Sampling()) -> TextGeneration
image_task.bind(variables) -> TextToImage
image_task.quantized(spec) -> TextToImage
image_task.prepare(prompts, *, key=None, steps=None, unconditional=None,
                   image=None, image_latents=None, mask=None, noise=None, initial=None,
                   times=None, encode_key=None) -> DenoisingInputs
image_task(prompts_or_prepared, *, steps=None, guidance=<default>, solver=None, key=None, decode=True) -> Images
RunProcessor(tokenizer)   # a run's ByteTokenizer or HFTokenizer as a task processor
Server.from_task(task, *, slots, capacity, admission=None, kv_cache=None, chunk=None,
                 prefix_cache=False, decode_steps=1) -> Server
```

A task captures the variables mapping at construction and on `bind`. Replacing the caller's mapping does not change the existing task. Array buffers remain shared; do not mutate, donate or delete them while a task uses them. Text requests need a processor. Numeric token rows remain integers: mixed text and token rows, floats, booleans and strings are refused; a resident `jax.Array` or `ModelInputs` reaches the model without a host copy.

`max_new_tokens` takes precedence over a source default. If only `max_length` is declared, the budget is that total minus the padded prompt width. Otherwise an LM run records its `sample_tokens` and `sampling` value; an objective uses its `Samples`. With no limit the call must provide one. A call's `n` takes precedence over the task's bound count the same way, including `n=1` over a source that asks for more; omitting it uses the bound count. Equal shapes and controls reuse the compiled executable across calls and `bind`.

`LMObjective.policy(params)` uses those parameters directly. DPO, GRPO and PPO pipelines publish the trained policy and leave out the frozen reference their loss uses; PPO also removes the critic. For other generative objectives, `ema=True` requires the moving-average state and raises if it is absent; use `ema=False` for live weights. `objective.pipeline`, `dew.pipeline`, `Pretrained.from_run` and every task's `from_run` and `from_pretrained` default to `ema=None`, which uses the EMA copy merged over the live parameters when the state or run has one, and the live parameters otherwise. `ema=True` refuses a run without an EMA copy, and `ema=False` takes the live parameters.

`TextToImage.from_run` reads `run.json` and the latest checkpoint under one directory, and `from_pretrained` first pulls a published run directory from the Hub. The three text tasks are constructed the same way for the model kinds they generate for, and `dew.pipeline` picks the task class from the objective name in `run.json`. In `pipeline`, `dtype` selects the computation dtype and `param_dtype` the parameter storage. `None` keeps a run's stored dtypes and uses FP32 masters for a source, and `"auto"` keeps the stored dtypes of either.

A source-default text task keeps the temperature, top-k, top-p, min-p, EOS and padding settings as its `Sampling` value. It holds the source's complete chain as `logits`, its criteria as `stopping` and the strategy its config names, and takes its return count from `num_return_sequences`.

An explicit `sampling=` replaces the policy and the chain, and the controls behind them are then neither built nor checked, so a distribution control you just replaced cannot block the call. The criteria, the strategy and the return count still come from the source, and every control the task keeps is checked as usual. An unknown control name is always refused, because no consumer is defined for it. A control the native decoder does not implement raises when the default task is created, naming the control and the reason; the table above lists every one. Loading weights for training or export does not select a decoding policy.

`BlockGeneration` uses `BlockProcess.generate`. Its `CanvasGeneration` has lengths, termination and decoder-step counts but no autoregressive likelihoods, plus the same `rows`, `host()`, `text` and continuation rows. Its continuations refine the shared encoded prompt independently, each over the original prompt rows, so each row sees the same batch-wide canvas draw that a single continuation sees.

`TextToImage` keeps the objective's or source's `steps`, `guidance` and `solver` defaults. `prepare` encodes the prompts and draws their noise once, placed for the task's mesh. `Images.images` is `[rows, H, W, C]` in [-1, 1], and `host()` reads a process's rows back. For a source whose solver pairs its own sigma and model-time tables, `grid(steps)` returns the process and the explicit time grid that a trajectory of that length steps through. The noise prior follows that process, so `prepare` takes the same `steps`. `final_denoise=False` ends a trajectory at the last grid point without the final clean prediction. `sample(denoise, x_T, steps=None, *, solver, guidance=None, key, times=None, final_denoise=True)` in `dew.sampling` takes the same two controls; pass exactly one of `steps` and `times`, and an explicit grid decides the trajectory's length.

For a source that ships a checker or an output transform, `finish(params, images)` runs it on the decoded images under the same placement. `blank` is the task's unconditional branch, encoded once by whatever built the task (`DiffusionObjective.blank_conditions`); `None` encodes it on every call. Rebinding keeps every task's compilation identity.

`Server.from_task(task, slots=, capacity=)` serves a `TextGeneration` task with continuous batching over one resident KV cache. It holds `slots` rows of `capacity` cache slots each, admits queued requests into free rows, and runs one compiled step over every row per iteration. Every request runs the task's bound policy and draws with the key it was submitted with, so a served request draws the same tokens as the same request run alone. `submit` returns a ticket that resolves to the request's `Generation`, `step` runs one iteration, and `run` steps until the queue and the rows are empty. Calling the server with a batch of prompts submits them and runs them.

### External engine clients

Dew does not run an HTTP server. Install `[inference-clients]` and inject the official client configured for your local or deployed engine. The adapter does not own the SDK client's lifetime.

<!-- not run: needs running Ollama and OpenAI-compatible servers -->
```python
import ollama
import openai
from dew.inference import OllamaCompletion, OpenAICompletion

with ollama.Client(host="http://127.0.0.1:11434") as client:
    task = OllamaCompletion("my-exported-model", client)
    answer = task("Explain this result.", 64, key=7,
                  options={"temperature": 0.8, "top_p": 0.9, "num_gpu": 0})
    print(answer.texts)

with openai.OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="local") as client:
    task = OpenAICompletion("my-exported-model", client)
    answer = task("Explain this result.", 64, top_p=0.9,
                  extra_body={"top_k": 40, "min_p": 0.05})
```

The convenience call returns `Completion(texts, finish_reasons, token_counts, usage, responses)`. Missing per-choice token counts and finish reasons remain `None`. Aggregate OpenAI usage stays separate from the per-choice fields. `responses` retains the SDK models, including the logprobs, token data and extensions the backend reported.

An explicit `sampling=Sampling(...)` sets the native policy controls that the selected backend supports. Ollama receives neutral repetition, presence and frequency penalties and explicit top-k, top-p and min-p values, so its hidden `repeat_penalty=1.1` default does not change the request. Conflicting explicit options are refused.

`OpenAICompletion(..., provider="vllm")` or `provider="sglang"` turns on the translation of `Sampling` into the parameters both engines accept, including `repetition_penalty=1.0` and optional EOS token IDs. Without a provider, the client refuses `top_k`, `min_p` and `eos_id` rather than drop them. Without a `Sampling` value, the provider's defaults or the caller's SDK options apply. A backend's tokenizer can split the same prompt differently from the exported one, so prompt token counts agree more often than prompt IDs do.

`stream` returns native SDK response chunks. `chat(messages, max_new_tokens, stream=..., **parameters)` keeps SDK tools, tool-result messages, structured-output controls and media fields. For asynchronous execution, pass an `AsyncClient` or `AsyncOpenAI` and use `acall`, `astream` or `achat`.

Ollama requests go through the SDK's public `generate` and `chat` methods, which handle request conversion, HTTP behavior, errors and line-stream framing. The adapter rejects negative token counts and request fields the installed SDK does not accept, and the SDK's own parsing rejects unparsable values. OpenAI completions use the SDK's public `with_raw_response` hook, so the choice and usage fields are checked on the wire before parsing. OpenAI request parameters go to its completion and chat resources, and vLLM-only parameters go explicitly in `extra_body`.

Provider extensions cannot override the task's model, prompt, token budget, requested choice count or an explicit `Sampling` policy. The SDK writes `extra_body` over the named parameters, so a policy field in it must equal the policy, or the request is refused before any network call.

`Pretrained.from_run(run_dir, *, ema=None, step=None).save(destination)` writes a saved run in that same layout. `from_run` rebuilds the run's model from its record as the right kind of bundle (`PretrainedDecoder` for a decoder), and `save` passes it to its family's writer, which refuses by name a model with no published layout. `dew export <run> <dest>` does the same from the command line. `bundle.push_to_hub(repo_id, private=, commit_message=)` publishes what `save` writes. To publish in the form `from_pretrained` pulls back, upload the run directory itself with `huggingface_hub.HfApi().upload_folder`.

A decoder trained through the LM recipe, exported with `PretrainedDecoder.from_model(...).save` and converted by `ollama create` answers a greedy request with Dew's own greedy continuation, token for token, over the live daemon. A loaded source's `save` and a `from_model` bundle's leave the same files, so either export converts.

## Diffusion and JEPA objectives

```text
DiffusionObjective(model, process, inputs, *, autoencoder=None,
                   unconditional_prob=0.12, ema_decay=0.999, solver=DDIM(),
                   guidance=CFG(3.0), steps=200, variables=None)
JepaObjective(encoder, predictor, mask, sample, momentum=(0.996, 1.0),
              momentum_steps=100000, label_key="label", encoder_variables=None,
              predictor_variables=None)
```

Import `DiffusionObjective` from `dew.objectives.diffusion`. `model` may be a pipeline bundle, as in `DiffusionObjective(pipe)`. The objective then reads the bundle's denoiser, process, conditions, autoencoder, initial variables and sampling policy, and any keyword you pass overrides that part. `JepaObjective`'s `encoder_variables` and `predictor_variables` are the trees each module starts from, which can be a pretrained or adapted encoder's. `process` takes a preset such as `Flow()` or `EDM(regime="pixel")`, or a custom `Process`. A preset builds once and `objective.process` holds the resulting Gaussian process; masked-token presets are refused. `FlowGRPOObjective` accepts the same values. `TextToImage.from_objective` keeps the objective's built process, and a manually built image task takes a `Process`. Its model accepts noisy arrays shaped `(B, *latent_shape)`, model noise levels shaped `(B,)`, and conditioning keywords from `InputSpec`. It returns a prediction with the sample's channel/spatial geometry. The `Process` determines the training target and prediction conversion. The objective passes `train=True` and a dropout RNG during training. An autoencoder changes sample geometry and must expose compatible encode/decode operations. `ema_decay=None` keeps no averaged copy, so previews and evaluation read the live variables. `steps`, `solver`, and `guidance` configure preview sampling; they do not set the number of optimization steps.

Import `JepaObjective` from `dew.objectives.jepa`. The encoder receives normalized images/video and optional token indices plus `train` and RNG settings. It returns token features with the feature dimension last. The predictor consumes context features and context/target position indices and returns target features of the encoder width. Mask grid, patch geometry, and predictor dimensions must agree. `momentum` specifies the EMA schedule endpoints over `momentum_steps` optimizer updates; `label_key` identifies labels for representation evaluation. See the [JEPA example](../guides/representation-learning.md).

### Pretrained latent diffusion

`dew.interop.Pretrained.load` also reads a diffusion checkpoint directory with `model_index.json`, component configurations, safetensors and tokenizer files. Seven denoisers load:

- `UNet2DCondition`, reading one or two CLIP towers through cross attention;
- `SD3Transformer`, reading both CLIP towers jointly beside a T5 tower;
- `FluxTransformer`, reading a T5 sequence with a pooled CLIP vector;
- Qwen-Image 2.1's `QwenImageTransformer`, reading its Qwen3-VL encoder's prompt states;
- FLUX.2's `Flux2Transformer`, reading stacked hidden states of a Mistral-3 ([dev]) or Qwen3 ([klein]) encoder;
- Z-Image's `ZImageTransformer`, reading its Qwen3 encoder's second-to-last layer;
- Wan 2.1's `WanTransformer`, reading its UMT5 encoder's states over a video latent.

The last six run on flow-matching schedules. The denoiser component in the directory selects the family, and the class it declares selects the model. The returned `Pretrained` holds the native model, its autoencoder behind the existing autoencoder interface (an `AutoencoderKL`, or Qwen-Image 2.1's one-frame `QwenImageVAE`), native text conditioning, a `Process`, the native solver policy and the published pipeline's own call policy. Model and scheduler implementations from other libraries run only in the reference tools.

`source.text_to_image()` builds the native image task. `source.save(directory, variables=updated)` writes the updated component weights back to their published layouts, retaining tokenizer files, image geometry and any safety-head parameters. Flax-declared components retain their source class and receive Flax msgpack files alongside the safetensors used by Dew; export does not relabel them as PyTorch models.

The source UNet configuration selects the normalization groups and epsilon, and the declared implementation selects the GEGLU semantics: exact GELU for PyTorch sources and approximate GELU for Flax sources. Spatial upsampling targets the actual shape of the next skip connection, including odd intermediate dimensions.

A published MM-DiT reads its own modulation order, joint attention, optional query-key normalization and second self-attention, and its stored sin/cos position buffer. That buffer goes into a `buffers` collection, so an optimizer and an EMA see only `params` and export writes the stored array back unchanged. The buffer is cropped centred on the latent's patch grid.

Flux's transformer has a different stack. Its double-stream blocks keep the image and text residuals apart except inside attention. Then single-stream blocks run over the concatenation, with attention and feed-forward sharing one output projection, and queries and keys rotated by a three-axis interleaved-real rotary table over the text and image IDs. The pipeline's 2x2 latent packing is inside the model, so a caller works in latents. A guidance-embedded checkpoint reads the distilled guidance value as a model input and does not run two guided branches.

`DiffusionConditioner` builds the text conditioning for every family, in one of these compositions:

- one CLIP tower's last hidden states;
- two towers' penultimate states, concatenated, with the second tower's projected pooled vector and the size and crop time IDs;
- those states padded out to the T5 width, with the T5 states after them along the sequence and both projections as the pooled vector;
- the T5 states alone, with one tower's unprojected pooled vector, which is Flux's composition.

Each text slot is encoded by the tower that its source pipeline uses for it. An SD3 source that declares no third encoder gets the zero segment its own pipeline writes, sized to the CLIP tokenizer's window and not to the sequence length a call asks for. Flux has no such path, so a Flux composition without its T5 tower is refused.

Qwen-Image 2.1 has its own conditioner, `QwenImageConditioner`. It runs the Qwen3-VL encoder's language model over the pipeline's text-to-image chat template, reads the last layer's output before the final norm, and drops the system turn. Rows are padded on the right to a token budget (`tokens`, 512 by default), and a prompt past the budget is refused. `DenoisingCondition.mask` marks the real tokens, so the transformer excludes padded keys. The encoder's vision tower is not run for text-to-image, but its tensors are kept as stored so that an export writes the whole encoder back.

The transformer lays out its sequence image first, so each of its two attention calls ends a row's keys at a length (`key_value_seq_lengths` on `scaled_dot_product_attention`). cuDNN applies that length as its padding mask and skips the padded keys. The transformer keeps the source's block-causal attention, in which the text attends causally and the image attends to all the text and to itself, and it modulates the text from time zero. Each row starts its image's rotary frame position after its own text. The source pipeline starts it after the longest prompt in the call, so a padded row here matches the source run on that prompt alone.

The 2.1 VAE is an image-only Wan-style autoencoder with four channels (RGBA) and 64 latent channels, normalized per channel with the config's `latents_mean` and `latents_std`. Image editing is not supported, because it needs the Qwen3-VL vision tower with its deepstack mergers, which Dew does not implement.

FLUX.2's conditioner is `HiddenStatesConditioner`. Its pipelines format a prompt with the encoder's chat template ([dev]: a system turn and a user turn; [klein]: a user turn with thinking off) and pad every row on the right to 512 tokens. For each token they stack the encoder's hidden states after layers 10, 20 and 30 ([dev]) or 9, 18 and 27 ([klein]). The transformer reads every row whole, so the padding positions hold whatever states the encoder gives them under the padding mask. The condition's mask marks the real tokens, but FLUX.2's transformer does not read it. [dev] embeds its guidance scale, 4.0 by default, and [klein] guides two branches at 4.0 against the empty prompt unless its index marks it step-distilled.

FLUX.2's VAE (`AutoencoderKLFlux2`) folds each 2x2 block of its 32-channel latent into 128 channels and normalizes each channel by its batch norm's running statistics, so the transformer's latent is the autoencoder's own, 16 times smaller on a side. The pipelines pass the scheduler `linspace(1, 1/N, N)` and their own shift, `compute_empirical_mu` of the token count and the step count, which the grid reproduces (`empirical_mu`). A tiny [klein] pipeline built from the published configs runs four steps to within 1e-5 of the source's call, and a tiny Mistral-3 encoder under Mistral Small 3.1's chat template gives [dev]'s prompt states to 1e-5 (`tests/test_flux2_source.py`). [dev]'s own files are gated, so its published tokenizer is not among the fixtures.

Z-Image (S3-DiT) uses the same conditioner, with the chat template's thinking turned on. It reads the output of its Qwen3 encoder's second-to-last layer, and only the real tokens, which the condition's mask marks. Its transformer cuts the Flux VAE's latent into 2x2 patches, pads the image and the prompt each to a multiple of 32 tokens with learned pad tokens, and refines each separately. It then runs one stream over both, image first, with rotary positions that follow each row's padded prompt length.

The source is called with the time 1 - sigma and returns the negated flow, while `ZImageTransformer` takes Dew's model time and returns the flow. Its pipeline guides as `pos + 5.0 (pos - neg)` against the empty prompt, which is Dew's `CFG(6.0)`. Tiny transformers match the source to 1e-5 of the largest output and 1e-4 in every gradient, and a tiny pipeline built from the published configs runs four steps to within 1e-5 of the source's call (`tests/test_z_image_source.py`).

Wan 2.1 (text to video) conditions through `WanConditioner`. Its pipeline cleans each prompt with ftfy (the `wan` extra, `pip install 'dewml[wan]'`), unescapes HTML entities twice and collapses whitespace. It then pads the prompt to 512 UMT5 tokens, runs the encoder under the padding mask and zeroes every state past the prompt's own tokens; the transformer reads all 512. UMT5 is the T5 tower with a relative position bias table in every layer (`T5EncoderTransformer(per_layer_bias=True)`, which a `umt5` config sets).

The Wan VAE encodes and decodes in its source's chunks: the first frame alone, then four frames to one latent frame at a time, with each causal convolution's last frames carried from one chunk to the next (`chunked_moments`, `chunked_decode`). So a clip's length does not set the memory the VAE holds. A 480x832 decode peaks at 4.6 GiB on an RTX 4080 for 33, 49 or 81 frames, where the one-pass decode needs 10 GiB at 33 frames and 14 GiB of temporaries at 49.

`WanTransformer` cuts the Wan VAE's latent into 1x2x2 patches. It self-attends with a three-axis rotary table over (frame, row, column), with queries and keys normalized across all heads, and cross-attends to the text. The sample is a clip, `Field("video", (frames, height, width, 3))`, of 81 frames at 480x832 unless the index or `load_diffusion_source(size=(frames, height, width))` says otherwise. The frame count is 1 + 4k, which the VAE decodes whole. `text_to_image()` returns `[N, frames, height, width, 3]`, and the pipeline guides at 5.0 against the empty prompt, for 50 steps over its UniPC grid with flow shift 3.0. Tiny transformers match the source's float32 flow and gradients within twice its own RMS distance from float64, and a tiny pipeline run ends 2.8e-6 from the source's latent (`tests/test_wan_source.py`).

`SourceTask` holds the published pipeline's own call policy: the steps and guidance its `__call__` defaults to, and the grid it prepares, with the sigma origin and latent geometry that pipeline uses. That way `text_to_image()` runs what the source runs. A directory that declares no pipeline takes its family's reference pipeline. A declared pipeline Dew does not implement, or one that drives a different denoiser than the directory holds, is refused, so it never runs under another pipeline's defaults.

`SourceSchedule.from_config` reconstructs eighteen published scheduler classes: DDIM, PNDM, DDPM, LMS, Euler, Euler ancestral, Heun, KDPM2, KDPM2 ancestral, DPM-Solver multistep, singlestep and SDE, DEIS, UniPC, EDM DPM-Solver, LCM, TCD and flow-matching Euler. The returned process, solver and grid use the controls and defaults of the declared class, and the solver is resolved once when the file is read.

Supported policies include source timestep spacing and offsets; Karras, exponential and beta sigma grids; finite lambda clipping; zero-terminal-SNR rescaling; terminal sigma selection; solver orders and correctors; DDPM fixed posterior variances; and clipping or dynamic thresholding in the source conversion order. DDIM and PNDM keep fixed training strides. LCM and TCD select their grids from `original_inference_steps`. Two-evaluation solvers keep the source's stage sigma and model time.

EDM uses its own sigma range, data scale, rho, log-sigma model time and signed output preconditioning, and its training process draws EDM sigmas instead of using a VP beta table. A flow-matching file holds its static or resolution-dependent shift, its terminal stretch and both sigma origins its pipelines use. A resolution-dependent shift reads the calling pipeline's latent token count through the task's grid, which `text_to_image()` sets to the checkpoint's own geometry.

Unimplemented active controls fail explicitly: Lu-lambda and flow-sigma grids, UniPC external predictors, nonlinear sigma interpolation, continuous model times, learned variances, DDPM log-space wide variance, and PNDM velocity-domain history. TCD clipping/thresholding and epsilon-prediction UniPC thresholding are also refused because the published update does not apply them. Sigma-indexed source schedulers whose initial model time repeats are refused when preparing their grid: the source starts at the second match and cannot complete its published evaluation list. Repeated noninitial stage and corrector times remain supported.

The tiny oracles in `tools/diffusers_source_reference.py` run actual Diffusers scheduler objects. Ordinary stochastic trajectories share explicit Gaussian draws. DPM-Solver SDE runs the actual `torchsde` tree, and the native solver is compared on its recorded increments; separate tests check the native Brownian bridge law.

All native source trajectories and VJPs run in float32. Trajectories use a `1e-4` scaled-error bound. Every source VJP is checked with `tests/reference_error.py`: its RMS distance from the exact source evaluation must be at most twice the float32 source's. The exact evaluation builds every scheduler table and does the step arithmetic in float64, from the same initial float32 latent, cotangent and recorded random draws. Its gradient is kept in float64 and not rounded back to float32.

<!-- not run: needs a local diffusion checkpoint directory -->
```python
from dew.interop import PretrainedPipeline

source = PretrainedPipeline.load("./image-checkpoint", dtype=jnp.float32)
images = source.text_to_image()(["a flower"], steps=20, key=0).host().images
objective = DiffusionObjective(source)
```

Training batches hold uint8 NHWC images and `source.inputs.tokenize(captions)`. A nine-channel inpainting source also specifies `inputs.mask`, binary NHWC masks with one channel, where white marks the region to repaint. Caption dropout keeps the mask and the masked-image latents. The mask only conditions the network, so it does not guarantee that the decoded unmasked pixels equal the original image.

`Pretrained.load(..., mesh=, layout=)` and `load_diffusion_source(..., mesh=, layout=)` stream a pipeline's weights the way they stream a decoder's. Each leaf is read from the mapped checkpoint one device shard at a time and placed before the next, so the host never holds the translated pipeline. Convolution kernels and a UNet's per-head attention kernels are small, so they are read whole. The placed tree is identical to the host load's, bit for bit.

`load_diffusion_source(..., text=False)` reads and downloads every component except the text encoder. Load the family's conditioner separately to encode prompts, for example `WanConditioner.from_pretrained(checkpoint, mesh=MeshSpec())`. `TextToImage.prepare(conditions=..., unconditional=...)` accepts these encodings as `{"conditioning": condition}`, with one row per sample, in place of prompts. Each row gets the same noise as a prompted call, so both calls produce the same images. Without encoded `unconditional`, this call applies no guidance. A pipeline loaded with `text=False` refuses prompts, `DiffusionObjective(pipe)` and `save`, since they require the text encoder.

Prepared `DenoisingInputs` can supply encoded native conditions and initial latents. Explicit grids pass to `sample(times=..., final_denoise=False)` when the last latent is the result; without an explicit grid, the existing `steps` and final clean-prediction convention remain unchanged. Published tabulated PRK grids own their integer half-interval rule; ordinary native grids retain exact half-intervals.

## Configuration and registries

Code builds every model, preset, solver, dataset and metric from its class. A record names a class by its import path, `{"class": "dew.nn.backbones.dit:SimpleDiT", "fields": {...}}`, and a function as `{"function": "optax:adamw"}`; `RunConfig.from_dict` and `ModelConfig.build` import what a record names. Where a person writes a record or a flag, `dew.registry` maps short aliases (`simple_dit`) to those paths.

A run names its model as `ModelConfig(name, fields)` and its objective as `ObjectiveConfig(name, fields)`: the class's import path or alias and the fields or constructor arguments the run states. `ObjectiveConfig.build(**derived)` constructs the objective with the arguments its caller builds (the model, what the data decides), and a class that is not a dataclass is read through its `__init__` signature. `dew.config.sweep.override(config, {"model.num_layers": 12})` and `assigned(config, ["trainer.steps=2000"])` set fields by dotted path; `dew train` and sweeps go through them.

`RunConfig.save` writes the run configuration. It is separate from the state checkpoint. [Recipes](../recipes.md) describes the configuration entry points and their side effects.

`attention_impl` is a model field that names the attention kernel:

- `'reference'` is the einsum and softmax, and the only path that reads `dtype`, `precision` and `force_fp32_for_softmax`;
- `'xla'` and `'cudnn'` are the two implementations of `jax.nn.dot_product_attention`;
- `'tpu'` is the Pallas splash kernel;
- `'auto'` resolves on each trace, so a configuration logged as `'auto'` also runs on the next machine.

`'auto'` takes `'reference'` for a call that asks for arithmetic only that path performs: a matmul precision above DEFAULT, a softmax outside fp32, or a compute dtype other than the inputs'. `'auto'` and `'xla'` both take `'reference'` for float64 inputs (JAX's XLA attention otherwise narrows their softmax to float32) and for bf16 on a GPU older than sm80. Otherwise `'auto'` takes the first of these whose conditions hold:

- `'cudnn'`, on a GPU of sm80 or later with bf16 or fp16 inputs, a query head width that is a multiple of 8 and at most 128, no softcap, no sinks and no `--xla_gpu_deterministic_ops`;
- `'tpu'`, on a TPU backend with bf16 or fp32 inputs, query and key lengths that are multiples of 128 and at least 512, no additive bias, whole sequences at the kernel, and a mask splash can describe (the sequence-parallel all-to-all passes whole sequences to the kernel, and the key/value gather does not);
- `'xla'`.

Splash's mask is a block-sparse descriptor built along with the executable, so a causal or sliding-window sequence skips the blocks its mask empties and does not pay for the whole rectangle. The head width is unconstrained, because the kernel pads it, but the query and key lengths have to be multiples of 128 for its mask blocks to tile them.

An additive bias, a mask that is a value of the trace (a KV-cache decode mask, or the striped mask sequence parallelism builds), a mask past the published cell budget, or a length that is not a multiple of 128 sends an explicit `'tpu'` call to the older Pallas flash kernel and keeps `'auto'` on `'xla'`. Splash applies a logit softcap (Gemma 2) and attention sinks (GPT-OSS) itself, and a packed batch reaches it as segment IDs instead of a mask. The flash kernel has neither a softcap nor sinks, so an explicit `'tpu'` call that splash cannot describe refuses them. On a GPU, `'auto'` runs both on `'xla'`, and `'cudnn'` refuses them by name.

Below 512 keys `'auto'` stays on `'xla'`, where XLA's attention measured faster on a v6e (`SPLASH_MIN_LENGTH` in `dew/nn/attention.py`). The parameter tree does not change with the implementation, so checkpoints are interchangeable across hardware.

The [README model list](https://github.com/AshishKumar4/dew/blob/main/README.md#models) names which model configurations run the whole workflow; each task guide covers its own data and objective.
