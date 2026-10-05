# Core API

This page describes the main interfaces and their contracts, grouped by task. Every public module also has a page generated from its docstrings, listed at the [end of this page](#all-modules) and in the sidebar. The [Quickstart](../getting-started.md) shows the core interfaces in one script.

## Objective

Import `Objective`, `Aux`, `Step` and `Ratio` from `dew.objectives`. `Ratio.mean()` reduces a ratio statistic, and `objective.scalar_loss(variables, batch, step)` evaluates and reduces a loss for direct differentiation.

| Member | Contract |
|---|---|
| `init(key, variables=None)` | Return a Flax variables mapping with a `params` collection. Pure; the trainer traces it once for shapes and once for values. `variables` is a held tree the caller supplies, which is how the trainer passes it as data; with `None` the objective uses its own (`DiffusionObjective.held_variables`, for example). An objective that holds nothing ignores it. |
| `loss(variables, batch, step)` | Return additive statistics and `Aux`. Use `Ratio(total, mass)` for a shared denominator; a scalar denotes a unit-mass term. |
| `reduce_loss(statistics)` | Return `(value, has_data)`. Override for an objective-owned composite Flax PyTree. |
| `apply_effects(variables, effects)` | Return nonparameter replacements from additive accepted-window observations. Required when the objective emits effects. |
| `evaluate(variables, batch, step)` | Return an artifact, a tuple of artifacts, or `None`. The base method returns `None`. |
| `preview(variables, batch, step, *, scored=None)` | Return display artifacts for a tracker, or `None`. The base method reuses `scored`, the first scoring artifacts. |
| `ema` | Optional `EMASpec`; the base objective uses `None`. |
| `optimizer(tx, *, accumulation)` | Return the optimizer `Trainer` steps `params` with, from the one it was handed; the base method returns `tx`. Per-network optimizers use `optax.multi_transform` and `optax.conditionally_mask`. |
| `averages(update)` | Return whether the update after `update` committed ones moves the EMA; the base method averages every update. |
| `artifact` | Optional description of the objective's evaluation artifact type. |

`Step.step` counts accepted microbatches. Its `key` derives from consumed attempts, including rejected ones. `ema` holds selected averaged leaves overlaid onto the complete variables mapping, or `None`.

`Aux(metrics, variables=None, qk_stats=None, effects=None)` contains training measurements, sequential mutable replacements, QK maxima and additive deferred effects. The trainer applies effects once per supported optimizer commit. `objective.scalar_loss(variables, batch, step)` returns a scalar and the same Aux for direct JAX differentiation.

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

`MeshSpec.build` uses the supplied devices or JAX's visible devices. The specified factors must divide the device count; the remaining factor is data parallelism. Explicit pipeline microbatches require `stage > 1` and a positive multiple of the stage count.

With `replicas` above 1, the hybrid mesh's data axis spans that many groups of granules: TPU slices, GPU hosts, NVLink domains, or processes whose devices all share one slice. Every other axis stays within a group. See [training on several nodes](../guides/multi-node.md#mesh-layout-across-nodes). A sequence axis above 1 splits each attention call's positions; the call's shape determines whether it exchanges data with an all-to-all or a gather.

`Layout.rules` accepts an ordered sequence of logical-axis rules or a mapping of overrides to the default table. When dimensions compete for one mesh axis, rule order determines precedence, and a dimension not divisible by that axis's size cannot use it. The valid parameter mesh axes are `fsdp`, `expert` and `tensor`.

- `min_shard` counts elements, not bytes.
- `tolerance` sets the permitted fraction of shardable parameter elements left replicated.
- `host` selects train-state fields from `params`, `opt_state` and `ema`. Selected `opt_state` and `ema` fields stay in pinned host memory between steps and are fetched to the device for a step. Selecting `params` places the whole `TrainState` on the CPU, including optimizer, EMA and accumulation. The optimizer transaction runs on a CPU companion of the mesh, so every process's runtime CPU device count must match its accelerator count before JAX initializes.
- `host_parameters` holds globs over logical parameter paths (`params/layers_*`) to keep in pinned host memory during inference. Only the `offloaded` placement uses them; `check` refuses a layout that names them for another placement.

`shardings(mesh, tree)` returns a matching tree of placements, and `check(params, shardings, mesh)` checks for excessive replication. See [distributed training](../concepts/distributed.md).

## Trainer

Import `Trainer` from `dew` or `dew.training`.

```text
Trainer(objective, optimizer, *, key,
        mesh=MeshSpec(), layout=Layout(), accumulation=1,
        dynamic_scale=False, checkpoints=None, tracker=None,
        step=None, rollout=None, profile=None)
```

`objective` is an initialized objective instance, and `optimizer` is an Optax gradient transformation. The required JAX `key` seeds initialization and the run; `mesh` and `layout` describe placement. Optional objects enable checkpoints, tracking, host-side rollouts and profiling.

`accumulation` counts accepted microbatches per effective window. Shared means use a weighted gradient accumulator with at least fp32 precision, preserving float64 when enabled in JAX. Finalized gradients enter Optax in their parameters' dtypes; partial gradients keep the wider working dtype.

A TPU has no float64: XLA rewrites it into pairs of float32, which are not IEEE doubles. The trainer therefore refuses float64 parameter storage on a TPU mesh when placing the state.

Composite statistics keep independent normalizers and inputs for scalar-VJP replay. Parameters and EMA stay fixed within a window, while sequential mutable replacements keep snapshots of the state they originally read. `dynamic_scale=True` persists scale/history and rejects nonfinite working or optimizer-input gradients without discarding the accepted prefix.

The custom `step(objective, optimizer)` factory manages the accepted/update clocks, scaler, EMA and mutable writes. Its body returns `(state, loss, aux)`, and the common compiled wrapper advances attempted `state.step`. A host `rollout` produces realized training arrays once per consumed attempt, before differentiation or replay.

### Train

```text
fit(dataset, *, steps, log_every=100, eval_every=None,
    checkpoint_every=None, metrics=(), preview=False) -> TrainState
```

- `dataset` is a `Dataset`.
- `steps` is the final target, including a restored step count.
- `log_every` sets the training log interval.
- `eval_every=None` disables evaluation. With an interval, evaluation also runs at the end.
- `checkpoint_every` controls periodic saves when a checkpointer exists. The normal completion path can still save a final checkpoint when a checkpointer is present.
- `metrics` reduce evaluation artifacts and require evaluation to be enabled.
- `preview=True` explicitly requests generated/display artifacts during evaluation; adding a scalar tracker alone does not request them.

Checkpointable data must supply the consumed iterator position. A failed scaled transaction advances attempted work and scaler history while preserving the earlier accepted prefix. Completely inactive windows close accepted slots without an optimizer call, weight decay, EMA, or deferred effects. Auxiliary-only windows can still be active.

`fit` is responsible for the iterators returned by the dataset. It closes training read-ahead before final evaluation and closes validation passes on success or failure. If a restored run has already reached its target, `fit` does no data, evaluation, compilation or save work. On exit, it stops its trace and waits for pending checkpoint writes, leaving the borrowed checkpointer and tracker open. Cleanup failures attach to the primary error.

### Initialize, restore, and compile

`initial_state()` constructs an unplaced initial `TrainState`. `place()` returns `(state, shardings, position)`, restoring from the configured checkpointer when available. Eager and placed initialization can differ in low floating-point bits across backends; compare the actual path used by your run.

`compile(state, batch)` returns `compiled(state, batch) -> (state, loss, metrics, loss_finite, accepted)`, with the scaler stored in `TrainState`. `loss_finite` and `accepted` are separate because a finite scalar can have a nonfinite gradient that is rejected.

The callable consumes its input state (`donate_argnums=0`): the returned state takes over the buffers, so do not keep a reference to the old state after `new = compiled(old, batch)`. The batch stays with the loader and is not donated.

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
DataPartition(index=0, count=1, readers=1, reader=0)
Loading(workers=0, threads=64, read_buffer=128, worker_buffer=2)
```

`from_records` reads records held in memory: a mapping of equal-length columns, a sequence of per-record mappings, or a source with `__len__` and `__getitem__`. The training stream reshuffles from `seed` every epoch and saves a global record position; `validation` makes one ordered pass of whole batches. A source with fewer records than one batch is refused. `from_grain` reads a caller-built Grain pipeline in the caller's order.

`train(partition)` opens a training iterator, and `val(partition)` opens one finite validation pass. Use `val=None` when there is no validation data. Each iterator reads the share of every global batch specified by its `DataPartition`: `index` selects one of `count` disjoint shares, `readers` processes read that share alike, and `reader` identifies this process among them. `DataPartition.of(mesh)` gives a process's share on a mesh, while `DataPartition()` includes every row.

`records` is the known training-record count or `None`, and `batch` is global. `steps_per_epoch` divides records by batch with integer division, or is `None`; `epoch_steps(epochs=1)` requires a finite record count. Set `ramp` when the run grows its batch over the first records. In that case, `batch` is the batch size at the end of the ramp.

Each factory call must return a fresh, exclusively owned iterator. Ordinary `close()` is finalization and must not race `next()` or checkpoint operations. A source that needs to interrupt blocking reads may additionally implement `request_stop()`: a thread-safe, nonblocking, idempotent signal, safe alongside both `next()` and `close()`. Tokenized wrappers forward these operations.

`DevicePrefetchIterator(iterator, mesh, depth=2, source_state=None)` in `dew.training.distributed` takes ownership of the iterator when construction succeeds. Its worker starts on the first `next()` and also restores the saved position, so a restoration failure cleans up through the iterator it already owns. Closing an unused iterator only finalizes it, without reading or restoring. Use a `with` block or call `close(timeout=5.0)`, even if your loop consumes a fixed number of batches.

Depth must be positive. At most `depth` device batches are queued, with at most one more in flight; this excludes the consumer and upstream buffers. `source_state` describes the last delivered batch, never speculative read-ahead. Queued batches are delivered before EOF or a source failure. Early close discards unread data and speculative failures but reports finalization failures.

The prefetch worker handles iteration, checkpoint operations and final source close. Closing requests cancellation, discards queued batches and joins the worker. If a source read, placement or finalization does not cooperate, close raises `TimeoutError`. The thread is not killed and may still hold in-flight references; cancellation remains requested, and you can retry close. Further iteration stops after close.

Loader shutdown depends on Grain's interfaces. The installed `DataLoaderIterator` has no public close, so Dew releases its owned references without accessing private iterators or changing the sampling pipeline. Local-record probes release the source and child processes, but Grain does not guarantee deterministic shutdown. It also keeps a process-wide shared-memory deletion thread pool. Dew calls `DatasetIterator.close()` on the iteration thread, but shutting down the read executor does not wait for record reads already running. Dew cannot guarantee shutdown for arbitrary blocked upstream reads.

`Loading` controls Grain concurrency and buffers for built-in specifications. Like Grain's own default, it starts no worker processes and reads with threads in the training process. Increase `workers` once measurements show the input pipeline is the bottleneck. See [data preparation](../concepts/data.md) for field layouts and process partitioning.

Every dataset specification inherits the keyword-only fields `seed` and `loading` from `DatasetSpec` and loads through `spec.load(batch=, tokenize=None)`. A specification that writes no captions raises `TypeError` if given a `tokenize` reader it cannot use.

An image specification uses the dataset's `val_split` for validation, limited to `val_batches` batches. With `val_split` unset, it holds out the first `val_batches * batch` records of the training source instead.

`Dataset.from_grain(train, *, batch, validation=None, records=None, loading=Loading())` builds a run over caller-assembled Grain pipelines. It repeats a `MapDataset`, selects the reader's share and saves one global record count. For a pipeline read as it comes, pass a function of the `DataPartition` that builds that share's `IterDataset`. This pipeline is batched in place and reports Grain's own iterator state.

A token corpus is a `TokenSource`: `TokenBytes` over a `.bin` file or `TokenRecords` over ArrayRecord shards of token arrays. `TokenWindows` and `PackedTokens` read `path` as a directory of `train` and `val` files and take whichever store their suffix names, so the same corpus gives the same windows and the same packing plan in both.


## Checkpoints

Import `Checkpoints` from `dew`.

`Checkpoints(directory, *, keep=2, local_directory=None, local_every=None)` configures persistent storage and optional local emergency checkpoints. Local directory and cadence must be specified together. The object opens storage on use.

`save(step, state, saved, metrics=None, *, share=None)` schedules a persistent save. `saved` is the iterator position as bytes or `None`; with a position, you must also supply `share`, the `DataPartition` that stream read. `wait()` waits for pending writes, and `latest` reports the newest eligible checkpoint.

`restore(template=None, step=None, *, share=None)` returns restored state data and the iterator position for the reader of `share`, or `None` for the position if no share is given. A template controls structure and placement. `path(step)` identifies the persistent checkpoint location.

This object does not write `run.json`; run configuration saving is separate. Use the [complete resume example](../guides/checkpoints.md) before adapting these lower-level calls.

## Language modeling and generation

Import `LMObjective` from `dew.objectives.lm`.

```text
LMObjective(model, seq_len, *, ema_decay=None, pad_id=None, head_chunks=4, head_tile=None,
            samples=None, pretrained=None, balance_rate=None, aux_loss_alpha=None,
            seq_aux=True, loss_role=None, mtp_weight=None, z_loss=0.0, router_z_loss=0.0,
            qk_stats=False, indexer=None, trainable=None, token_accuracy=True,
            processor=None)
IndexerTraining(phase, weight=1.0)
```

`Pretrained.load` returns a bundle whose `lm_objective(seq_len, **options)` builds this objective with its model, initial variables and processor, which `pipeline` then decodes with. It refuses `pretrained=` because the bundle supplies those weights. `bundle.lora(...)` returns a bundle with a fresh adapter whose `lm_objective` trains the factors alone, and whose `save` and `export` merge them. The model must implement Linen `hidden_states(tokens, train=..., positions=..., segment_ids=...)`, returning `(B, S, D)`, and `head_weight(params)`, returning the `(D, vocab)` vocabulary matrix. The objective also reads `final_logit_softcap` and `precision`. Prediction-depth training needs `mtp_hidden_states` and compatible prediction-depth configuration. Mutable router, QK and indexer collections are required when their options are enabled.

`pretrained` supplies the complete variables tree; `loss_role` requires aligned `text_roles`. `pad_id` masks matching targets. `head_chunks` controls vocabulary tiling, and `head_tile` the head's backward tile (`'whole'`, `'tiled'` or a tile shape; `None` picks one for the objective); `samples` configures text previews. `ema_decay` defaults to `None`, which trains without an averaged copy; a decay such as `0.999` keeps one that evaluation and previews read, and `1.0` retains a frozen one. Routing balance, auxiliary loss, prediction-depth weight, and QK statistics require matching model computation. These interfaces make `LMObjective` specific to compatible decoders. `z_loss` adds PaLM's auxiliary term, the coefficient times the squared log partition of every counted prediction; zero adds nothing. `router_z_loss` is the routers' own z-loss (ST-MoE), and zero adds nothing. `token_accuracy=False` drops the `token_accuracy` metric and the pass over every logit it costs. `trainable` is a path filter over the parameter leaves the optimizer moves; the rest of the tree is kept under `frozen`. An adapter's filter, `dew.lora.LoRA.trainable`, goes here. `None` trains every leaf, and `trainable` cannot be combined with `indexer`.

`indexer` trains DeepSeek-V3.2's lightning indexer on a model whose `mla` mixer names `index_n_heads` and `index_head_dim`. `IndexerTraining("warmup")` needs a mixer without `index_topk`: the model runs dense attention, `init` keeps the indexer alone in `params` and the rest of the tree under `frozen` (a `pretrained` tree may omit the indexer, as a dense checkpoint does), and the loss is the KL of the indexer's softmax from the attention distribution, reported as `indexer_kl`. `IndexerTraining("sparse")` needs a mixer with `index_topk`: the whole tree trains, the cross entropy trains the main weights and the KL over the selected keys trains the indexer, whose inputs are detached; a warm-up checkpoint's split tree is accepted as `pretrained`. `weight` scales the KL term.

Import `generate`, `Sampling` and `Generation` from `dew.sampling`:

```text
generate(model, params, inputs, max_new_tokens, *, key=None,
         sampling=Sampling(), n=1, logits=None, stopping=None, strategy=None) -> Generation
Sampling(temperature=1.0, top_k=None, eos_id=None, pad_id=None, top_p=1.0, min_p=0.0,
         repetition_penalty=1.0, presence_penalty=0.0, frequency_penalty=0.0,
         no_repeat_ngram_size=0, min_new_tokens=0, typical_p=1.0, stop=())
```

`params` is the complete variables tree. `inputs` is a `ModelInputs` from `dew.nn.inputs`, or an integer `(B, P)` array normalized to all-valid text. `ModelInputs.token_fields["attention_mask"]` identifies real token slots; there is no separate generation length argument. Every row needs a real token. Only real tokens count against `model.max_seq_len`. Conditioning arrays are batch-aligned and used during prefill; decode keeps the model's cached logical positions. `key` is an integer seed or a JAX key; `key=n` is `jax.random.key(n)`.

The compiled decoder uses one padded input shape, per-row cache cursors and a fixed iteration count. Finished rows preserve their cached state. On a mesh, rows split over the batch axes and the result keeps that sharding. Each process supplies its own rows with the same count and padded width as every other process, then reads them back with `Generation.host()`. The processes must also pass the same `n`. Keys fold in the global row index, so the pool produces the same draws as one process given the same rows. Invalid input on one rank raises on all ranks before device execution.

`n` sets the number of continuations per prompt and must be a positive integer. They share the prompt's prefill, then run one after another on the device; for each continuation, the prompts are batched as before. Decode time grows with `n`, but each continuation reuses the previous one's cache working memory. Only output storage grows with `n`.

Continuation zero uses the prompt's global row key, so it gives the same draw as `n=1`. Continuation `j` folds `j` into that key, which means increasing `n` leaves existing continuations unchanged. On a mesh, prompts are padded before generating continuations. A prompt's `n` rows therefore stay on the requesting process, and `host()` drops only rows belonging to padded prompts.

`Sampling.eos_id` accepts an integer or a tuple of ids and normalizes them into an immutable tuple; any listed ID terminates a row. `Sampling` groups the common generation controls, each with a default that changes nothing: Transformers' `RepetitionPenalty` over the prompt and generated tokens, vLLM's presence and frequency penalties over generated tokens, `no_repeat_ngram_size`, `min_new_tokens` (which holds back EOS and needs an `eos_id`), temperature, top-k, nucleus top-p, relative min-p and typical-p filtering.

`transforms()` uses `dew.sampling.text.ordered_transforms` to compile these controls in Transformers' `_get_logits_processor` order, placing vLLM's penalties beside the repetition penalty as vLLM does. At least one token survives the filters. Zero temperature selects argmax and runs no filter. The resulting transforms and EOS criterion are the decoding components described below.

`stop` holds strings that terminate a row and must be compiled against a tokenizer's vocabulary. `generate` therefore refuses them, while `TextGeneration` compiles them through its processor. When a task runs a policy with `eos_id` or `pad_id` left `None`, it uses the task's own values. `generate` alone stops on no EOS unless one is named, and pads with 0.

`Generation.tokens` includes the original prompt and has shape `(B * n, P + max_new_tokens)`, where `B` counts placed prompt rows. Prompts stay in request order, starting with prompt zero's `n` continuations. Each row has its own `lengths` count of response tokens including EOS, and `terminated` flag: true for EOS termination, false for the token budget. Slots after termination hold `Sampling.pad_id`.

`behavior_log_probs` and `raw_log_probs` both have shape `(B * n, max_new_tokens)`. The first records the filtered distribution that drew each action, and the second records the unmodified policy. `rows` counts this process's real prompts times `n`; `host()` returns the record with host arrays for those rows. `text` decodes them with the task's processor, returning one string per row.

`LMObjective.per_token_log_probs(params, tokens, left_padding=...)` scores the raw policy. It left-aligns real tokens for the forward and restores the original next-token alignment. Unscored padding slots are zero. `SampledRollout` records the raw sampling-time values as `old_log_probs` and preserves actual draws as `behavior_log_probs`, in the packed layout `GRPOObjective.packed_log_probs` rescores. Reward text excludes EOS and padding. When a batch carries no `old_log_probs`, GRPO uses the behavior probabilities as the old policy and its behavior corrections follow verl's bypass mode. The next-token objective refuses models declaring `causal=False`.

### Decoding components

Decoding has three extension points: logits transforms, stopping criteria and strategies. Import the protocols from `dew.sampling` and the built-ins from `dew.sampling.decoding`.

```text
LogitsTransform: (StepState, logits[rows, vocab]) -> logits[rows, vocab]
Stopping:        (StepState, drawn_tokens[rows]) -> finished[rows]
Strategy:        (DecoderState, StepState, DecodeOps, transform, stopping, budget, n) -> Draws
StepState(tokens, valid, step, active, keys, prompt_width)
```

`StepState` supplies all the inputs to a transform or criterion. `tokens` is a fixed-capacity buffer `[rows, prompt_width + max_new_tokens]` with the prompt followed by generated-token slots. `valid` marks real tokens, so each row reads its own history regardless of the padding in its prompt batch. `step` counts committed tokens per row; `active` marks rows still generating; `keys` holds one PRNG key per row. `state.history()` returns each row's real tokens left aligned with their count, `prompt_history()` returns the prompt region, and `total()` returns the real token count. A transform has no access to model parameters or cache internals.

`logits` specifies the complete transform chain in execution order. `None` uses the chain compiled from `sampling`; an explicit sequence replaces that chain; `()` applies no transform. Use an explicit sequence if you need an order `Sampling` does not produce. A call's `logits` replaces the task's bound chain. Explicit `sampling=` also clears the bound chain because it replaces the policy that chain was built for.

An explicit `stopping` sequence runs alongside the policy's EOS criterion, so adding a criterion cannot remove EOS termination. Criteria combine with OR and run after every committed token. The token that triggers a criterion is emitted with its likelihoods; later slots hold `pad_id` with zero likelihood.

Built-in transforms are pytrees, so array configuration is passed as data and does not enter a compilation cache key. A plain function also works; use `jax.tree_util.Partial(fn, array)` for its array configuration. Everything runs inside the compiled loop, without a host callback. In a process pool, Dew compares the resolved components' structure and configuration arrays. Ranks that ban different tokens are refused, preventing them from running different policies.

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

`stop_strings` reads the tokenizer once on the host. It compiles where each token's piece can occur inside a stop string and how many trailing units of the string the token's start can cover. The criterion then runs entirely on device without decoding. A match must touch the token just drawn: an earlier string does not stop the row later, but a match across several tokens or overhanging either end does stop it.

Byte-level and byte-fallback vocabularies are matched by their byte spelling over UTF-8 bytes, so a code point split across two tokens still matches. Other vocabularies use `convert_tokens_to_string` after an ordinary prefix, since a decoder may add or remove a leading space depending on the preceding text. `tokenizer` is a Transformers tokenizer or a task `Processor` holding one; `vocab_size` sizes the table when a model's head is wider than its vocabulary.

An active row can be sampled only if every score is finite or `-inf` and at least one is finite. A NaN or `+inf` beside a finite score makes the draw arbitrary; a row with no finite score has no distribution. In either case, `generate` raises. Zero temperature leaves such rows unchanged instead of selecting an argmax, so it cannot bypass this check.

The model's own distribution must also be defined. Otherwise generation raises rather than reporting a NaN likelihood, even if a later `RemoveInvalidValues()` repairs the scores. Use that transform to repair scores made invalid by the transform chain itself.

#### Strategies

```text
Sample(grammar=None)
Beam(width=1, length_penalty=1.0, early_stopping=False, stop_ids=1)
Speculative(block=4, confidence=0.0)
```

`Sample` is the default strategy and draws each row independently. A `grammar` from `dew.sampling.guided` (`regex` or `json_schema`) constrains every draw to that grammar.

`Beam` implements deterministic beam search with Transformers 5.16.1's `_beam_search` bookkeeping. Each step keeps the best `(1 + stop_ids) * width` continuations so that `width` live beams remain. A stopping criterion moves a beam to the completed set, dividing its score by its generated length raised to `length_penalty`. `early_stopping` accepts the reference's `False`, `True` and `"never"` values.

The prompt is prefilled once and copied into `width` cache rows. Each step reparents these rows, so a branched beam decodes exactly like a separately selected prefix. `n` sets how many completed hypotheses to return; `n > width` is an error.

A selected path is a search result, so its behaviour log probability is zero while its raw log probabilities remain the model's own. Sampling with beams is refused because a selected beam's marginal probability differs from its per-step candidate probability. There is no correct behaviour likelihood to record.

`Speculative` drafts with the model's own prediction depths and verifies with the model, following algorithm 1 of [arXiv 2211.17192](https://arxiv.org/abs/2211.17192) as `_speculative_sampling` applies it. A block's first candidate is an ordinary target draw and is always accepted. Each prediction depth then proposes the next candidate from the previous hidden state and the candidate's embedding.

A proposal is accepted with probability `min(1, p(x) / q(x))`, where `p` is the target's post-transform distribution and `q` is the draft's actual distribution. At the first rejection, sampling uses the normalized positive part of `p - q`. A block with no rejection draws a bonus from `p`. The emitted tokens therefore have exactly the same distribution as `Sample`, although the draws need not match for a given seed.

For every emitted action, the behaviour log probability is the target's post-transform value, and the raw log probability is the model's own. The draft's `q`, acceptance probability and residual are never recorded. `confidence` stops drafting after the first candidate the draft is less sure of, as `ConfidenceCriteria` does. An unoffered candidate is not a rejection, so that block ends on an ordinary target draw. Models without prediction depths are refused; there is no silent fallback.

Before a block, the target cache is saved so the accepted prefix can be replayed into it: a recurrent mixer's running summary cannot be rewound with a cursor. Rebuilding the prediction cache follows the same invariant. Depth `d`'s entry for token `t` reads depth `d - 1`'s hidden state at `t - 1` and token `t`'s prepared embedding at its own coordinate; the target is depth zero's predecessor. This matches `mtp_hidden_states` training, `MTPCandidateGenerator` correction and the inputs to `Qwen3_5MultiTokenPredictor.forward`.

Each depth keeps its last state across blocks, so no entry is lost at a boundary. The prompt initializes every depth from the embeddings already prepared during prefill, including media replacements, without another encoder call.

A depth becomes usable after enough real tokens have preceded it. Rotary offsets and repeated image coordinates do not change that count. A newly available predecessor is retained even if its next depth cannot yet write a cache entry.

A block drafter, such as DeepSeek-V4.1's DSpark, proposes a block's later candidates in one pass rather than chaining prediction depths. It reads context recorded by the target layers: prefill initializes it with the prompt's context, and each replayed block appends context for its kept positions. As the pass reaches each position, the strategy draws from that position's drafter logits, including Markov bias. `block - 1` must not exceed the drafter's own block size. DSpark drafters for a V4 trunk (DeepSeek-V4-Flash-DSpark) are not built because the stages use V4.1's Single-Pass mHC blocks.

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

Dew uses T5X as a reference and test oracle and adapts some state and cache operations; it is not a runtime dependency. Nothing in T5X's JAX loops rules out ragged MoE, but Dew still has to handle row placement, inactive-row masks and collective agreement around model calls. The T5X return contract does not cover these requirements.

#### Source generation controls

A loaded source's `generation_config.json` is data. Every control written by Transformers 5.16.1 has a defined treatment: the native policy, a transform, a criterion, a strategy or the task implements it; it is kept as provenance; or `PretrainedDecoder.text_generation()` refuses it with a reason. Common controls become `task.sampling`, a `Sampling` value you can change with `dataclasses.replace(task.sampling, ...)`.

When a source sets a control outside `Sampling`, `task.logits` holds the complete chain. The same compiler, `ordered_transforms`, builds it in `_get_logits_processor` order, including the policy's transforms. Beam search always has a bound chain that ends after the processors, because the search selects its own continuations. Unset controls are inert, as are values for which `generate()` adds no processor, criterion or search mode. Like upstream, Dew checks beam-only and sampling-only controls only when the corresponding mode is active.

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
Objective.pipeline(state, *, ema=None) -> the objective's task over state.averaged or state.variables
LMObjective.pipeline(state, *, ema=None, processor=None) -> TextGeneration
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
                max_seq_len=None, revision=None, gguf_file=None, single_file=None, mesh=None,
                layout=None, fallback=None) -> the kind it is called on, or the kind the source is
PretrainedDecoder.text_generation(*, sampling=None) -> TextGeneration
PretrainedDecoder.lm_objective(seq_len, **options) -> LMObjective
PretrainedDecoder.lora(*, rank, modules, key, alpha=None, rslora=False, dropout=0.0) -> PretrainedDecoder
PretrainedMaskedDecoder.text_generation() -> MaskedGeneration
PretrainedBlockDecoder.block_generation() -> BlockGeneration
PretrainedPipeline.text_to_image() -> TextToImage
PretrainedPipeline.diffusion_objective(**options) -> DiffusionObjective
PretrainedPipeline.lora(*, rank, modules, key, alpha=None, rslora=False, dropout=0.0) -> PretrainedPipeline
PretrainedFallback.lm_objective(seq_len, **options) -> LMObjective
Pretrained.save(directory, *, variables=None, max_shard_size="5GB")
Pretrained.push_to_hub(repo_id, *, variables=None, private=False, commit_message=..., max_shard_size="5GB")
PPOObjective.pipeline(state, *, ema=None, processor=None) -> TextGeneration
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

`LMObjective.policy(params)` uses the supplied parameters directly. DPO, GRPO and PPO pipelines expose the trained policy rather than the frozen reference used in the loss; PPO also removes the critic. For other generative objectives, `ema=True` requires moving-average state and raises if it is absent. Use `ema=False` for live weights.

`objective.pipeline`, `dew.pipeline`, `Pretrained.from_run` and every task's `from_run` and `from_pretrained` default to `ema=None`. This overlays the EMA copy onto live parameters when the state or run has one, and uses live parameters otherwise. `ema=True` refuses a run without an EMA copy; `ema=False` uses live parameters.

`TextToImage.from_run` reads `run.json` and the latest checkpoint from one directory. `from_pretrained` first pulls a published run directory from the Hub. The three text tasks construct the same way for their respective model kinds, and `dew.pipeline` chooses the task class from the objective name in `run.json`.

For `pipeline`, `dtype` sets the computation dtype and `param_dtype` sets parameter storage. `None` keeps a run's stored dtypes and uses FP32 masters for a source; `"auto"` keeps the stored dtypes of either.

A source-default text task preserves temperature, top-k, top-p, min-p, EOS and padding in its `Sampling` value. It uses the source's complete chain as `logits`, its criteria as `stopping`, its configured strategy, and `num_return_sequences` as the return count.

Explicit `sampling=` replaces the policy and chain, so their original controls are neither built nor checked. A distribution control you replaced cannot block the call. Criteria, strategy and return count still come from the source, and the task checks every control it keeps. Unknown control names are always refused because they have no defined consumer. Creating a default task raises for unimplemented controls, naming each control and the reason listed in the table above. Loading weights for training or export does not select a decoding policy.

`BlockGeneration` uses `BlockProcess.generate`. Its `CanvasGeneration` records lengths, termination and decoder-step counts without autoregressive likelihoods, plus the same `rows`, `host()`, `text` and continuation rows. Each continuation independently refines the shared encoded prompt over the original prompt rows, so each row gets the same batch-wide canvas draw it would get with one continuation.

`TextToImage` keeps the objective's or source's `steps`, `guidance` and `solver` defaults. `prepare` encodes prompts and draws their noise once, placed on the task's mesh. `Images.images` is `[rows, H, W, C]` in [-1, 1], and `host()` reads back this process's rows.

For a source solver with paired sigma and model-time tables, `grid(steps)` returns the process and explicit time grid for that trajectory length. The noise prior follows that process, so `prepare` takes the same `steps`. `final_denoise=False` stops at the last grid point without a final clean prediction. `sample(denoise, x_T, steps=None, *, solver, guidance=None, key, times=None, final_denoise=True)` in `dew.sampling` accepts the same controls: pass exactly one of `steps` and `times`, with an explicit grid determining the trajectory's length.

If a source includes a checker or output transform, `finish(params, images)` applies it to decoded images under the same placement. `blank` stores the unconditional branch encoded when the task was built (`DiffusionObjective.blank_conditions`); `None` encodes it on every call. Rebinding preserves each task's compilation identity.

`Server.from_task(task, slots=, capacity=)` serves a `TextGeneration` task with continuous batching over one resident KV cache. It allocates `slots` rows with `capacity` cache slots each, admits queued requests into free rows, and runs one compiled step over all rows per iteration. Each request uses the task's bound policy and its submitted key, producing the same tokens as that request run alone. `submit` returns a ticket for the request's `Generation`; `step` runs one iteration; `run` steps until the queue and rows are empty. Calling the server with a batch of prompts submits them and runs them to completion.

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

Explicit `sampling=Sampling(...)` sets the native policy controls supported by the selected backend. Ollama receives neutral repetition/presence/frequency penalties and explicit top-k/top-p/min-p values, keeping its hidden `repeat_penalty=1.1` default from altering the request. Conflicting explicit options are refused.

`OpenAICompletion(..., provider="vllm")` or `provider="sglang"` enables Sampling translation for the controls both engines accept, including `repetition_penalty=1.0` and optional EOS-token IDs. Without either provider, the client refuses `top_k`, `min_p` and `eos_id` instead of dropping them. Without a Sampling value, provider defaults or the caller's SDK options apply. A backend tokenizer may segment a prompt differently from the exported tokenizer, so prompt token counts agree more often than prompt ids do.

`stream` returns native SDK response chunks, and `chat(messages, max_new_tokens, stream=..., **parameters)` preserves SDK tools, tool-result messages, structured-output controls and media fields. For asynchronous execution, inject an `AsyncClient`/`AsyncOpenAI` and use `acall`, `astream` or `achat`.

Ollama requests use the SDK's public `generate` and `chat` methods for request conversion, HTTP behavior, error handling and line-stream framing. The adapter rejects negative token counts and fields the installed SDK does not accept; SDK parsing rejects unparsable values. OpenAI completions use the public `with_raw_response` hook to check choice and usage fields on the wire before parsing. Request parameters go to the SDK's completion/chat resources, with vLLM-only parameters explicitly in `extra_body`.

Provider extensions cannot override the task's model, prompt, token budget, requested choice count or explicit `Sampling` policy. Since the SDK writes `extra_body` over named parameters, a policy field there must match the policy; otherwise the request is refused before any network call.

`Pretrained.from_run(run_dir, *, ema=None, step=None).save(destination)` exports a saved run in that layout. `from_run` rebuilds the model from its record as the appropriate kind (`PretrainedDecoder` for a decoder), and `save` uses the family's writer. A model with no published layout is refused by name. The command-line equivalent is `dew export <run> <dest>`.

`bundle.push_to_hub(repo_id, private=, commit_message=)` publishes the files written by `save`. To publish the run directory itself in the form `from_pretrained` reads, use `huggingface_hub.HfApi().upload_folder`.

A decoder trained through the LM recipe, exported with `PretrainedDecoder.from_model(...).save` and converted by `ollama create` answers a greedy request with Dew's own greedy continuation, token for token, over the live daemon. A loaded source's `save` and a `from_model` bundle's leave the same files, so either export converts.

## Diffusion and JEPA objectives

```text
DiffusionObjective(model, process, inputs, *, autoencoder=None,
                   unconditional_prob=0.12, ema_decay=0.999, solver=DDIM(),
                   guidance=CFG(3.0), steps=200, pretrained=None, trainable=None)
JepaObjective(encoder, predictor, mask, sample, momentum=(0.996, 1.0),
              momentum_steps=100000, label_key="label")
```

Import `DiffusionObjective` from `dew.objectives.diffusion`. `process` takes a preset such as `Flow()` or `EDM(regime="pixel")`, or a custom `Process`. A preset builds once and `objective.process` holds the resulting Gaussian process; masked-token presets are refused. `FlowGRPOObjective` accepts the same values. `TextToImage.from_objective` keeps the objective's built process, and a manually built image task takes a `Process`. Its model accepts noisy arrays shaped `(B, *latent_shape)`, model noise levels shaped `(B,)`, and conditioning keywords from `InputSpec`. It returns a prediction with the sample's channel/spatial geometry. The `Process` determines the training target and prediction conversion. The objective passes `train=True` and a dropout RNG during training. An autoencoder changes sample geometry and must expose compatible encode/decode operations. `ema_decay=None` keeps no averaged copy, so previews and evaluation read the live variables. `steps`, `solver`, and `guidance` configure preview sampling; they do not set the number of optimization steps.

Import `JepaObjective` from `dew.objectives.jepa`. The encoder receives normalized images/video and optional token indices plus `train` and RNG settings. It returns token features with the feature dimension last. The predictor consumes context features and context/target position indices and returns target features of the encoder width. Mask grid, patch geometry, and predictor dimensions must agree. `momentum` specifies the EMA schedule endpoints over `momentum_steps` optimizer updates; `label_key` identifies labels for representation evaluation. See the [JEPA example](../guides/representation-learning.md).

### Pretrained latent diffusion

`dew.interop.Pretrained.load` also reads diffusion checkpoint directories containing `model_index.json`, component configurations, safetensors and tokenizer files. It loads seven denoisers:

- `UNet2DCondition`, with cross attention to one or two CLIP towers.
- `SD3Transformer`, with joint attention to both towers alongside a T5 tower.
- `FluxTransformer`, with a T5 sequence and a pooled CLIP vector.
- Qwen-Image 2.1's `QwenImageTransformer`, with prompt states from its Qwen3-VL encoder.
- FLUX.2's `Flux2Transformer`, with stacked hidden states from a Mistral-3 ([dev]) or Qwen3 ([klein]) encoder.
- Z-Image's `ZImageTransformer`, with the Qwen3 encoder's second-to-last layer.
- Wan 2.1's `WanTransformer`, with UMT5 encoder states over a video latent.

The last six use flow-matching schedules. The directory's denoiser component determines the family, and its declared class determines the model. The returned `Pretrained` contains the native model, its autoencoder through the existing interface (an `AutoencoderKL`, or Qwen-Image 2.1's one-frame `QwenImageVAE`), native text conditioning, a `Process`, the native solver policy and the published pipeline's call policy. Other libraries' model and scheduler implementations run only in the reference tools.

`source.text_to_image()` builds the native image task. `source.save(directory, variables=updated)` writes the updated component weights back to their published layouts, retaining tokenizer files, image geometry and any safety-head parameters. Flax-declared components retain their source class and receive Flax msgpack files alongside the safetensors used by Dew; export does not relabel them as PyTorch models.

The source UNet configuration sets normalization groups and epsilon. Its declared implementation determines GEGLU semantics: exact GELU for PyTorch sources, approximate GELU for Flax sources. Spatial upsampling targets the next skip shape, including odd intermediate dimensions.

A published MM-DiT uses its own modulation order, joint attention, optional query-key normalization and second self-attention. The stored sin/cos position array goes into a `buffers` collection, leaving only `params` for the optimizer and EMA. Export writes that stored array back unchanged; the buffer is cropped centred on the latent's patch grid.

Flux's transformer first runs double-stream blocks, where image and text residuals join only inside attention. Single-stream blocks then process their concatenation, with attention and feed-forward sharing one output projection. Queries and keys use a three-axis interleaved-real rotary table over text and image ids. The model handles the pipeline's 2x2 latent packing, so callers work in latents. Guidance-embedded checkpoints take the distilled guidance value as a model input rather than evaluating two guided branches.

`DiffusionConditioner` assembles the text conditioning required by each family:

- One CLIP tower's last hidden states.
- Two towers' penultimate states concatenated, with the second tower's projected pooled vector and the size and crop time ids.
- Those states padded to the T5 width, followed by T5 states along the sequence, with both projections as the pooled vector.
- T5 states alone with one tower's unprojected pooled vector, as Flux uses.

Each text slot goes to the tower used by its source pipeline. An SD3 source without a third encoder gets the zero segment its pipeline writes, sized to the CLIP tokenizer's window rather than the sequence length requested by a call. Flux has no equivalent path, so a Flux composition without its T5 tower is refused.

Qwen-Image 2.1 uses `QwenImageConditioner`. It runs the Qwen3-VL encoder's language model with the pipeline's text-to-image chat template, reads the last layer's output before the final norm, and drops the system turn. Rows are right-padded to the token budget (`tokens`, 512 by default); a prompt exceeding it is refused. `DenoisingCondition.mask` marks real tokens so the transformer excludes padded keys.

The transformer's sequence is image-first. Both attention calls pass each row's key length as `key_value_seq_lengths` to `scaled_dot_product_attention`, allowing cuDNN to apply its padding mask and skip padded keys. The source's block-causal attention is preserved: text attends causally, while the image attends to all text and itself. Text is modulated from time zero. Each row's image rotary frame position starts after its own text. The source pipeline starts it after the longest prompt in the call, so a padded row here matches the source run on that prompt alone.

Text-to-image does not run the encoder's vision tower. Its tensors are preserved as stored so export can write the whole encoder back. The 2.1 VAE is an image-only Wan-style autoencoder with four channels (RGBA) and 64 latent channels, normalized per channel with the config's `latents_mean` and `latents_std`. Image editing is not supported because Dew does not implement the Qwen3-VL vision tower with its deepstack mergers.

FLUX.2 uses `HiddenStatesConditioner`. Its pipelines apply the encoder's chat template ([dev]: a system turn and a user turn; [klein]: a user turn with thinking off), then right-pad each row to 512 tokens. They stack hidden states per token after layers 10, 20 and 30 ([dev]) or 9, 18 and 27 ([klein]). The transformer reads each row whole, including the states the encoder produced for padding under its padding mask. The condition's mask marks real tokens, but FLUX.2's transformer does not read it.

[dev] embeds its guidance scale, 4.0 by default. [klein] guides two branches at 4.0 against the empty prompt unless its index marks it step-distilled. FLUX.2's VAE (`AutoencoderKLFlux2`) folds each 2x2 block of its 32-channel latent into 128 channels and normalizes each channel with its batch norm's running statistics. The transformer therefore reads the autoencoder's own latent, 16 times smaller on a side.

The pipelines give the scheduler `linspace(1, 1/N, N)` and a shift computed by `compute_empirical_mu` from token and step counts. The grid reproduces that shift with `empirical_mu`. A tiny [klein] pipeline built from the published configs matches the source's four-step call within 1e-5. A tiny Mistral-3 encoder using Mistral Small 3.1's chat template matches [dev]'s prompt states within 1e-5 (`tests/test_flux2_source.py`). [dev]'s own files are gated, so its published tokenizer is not among the fixtures.

Z-Image (S3-DiT) uses the same conditioner with thinking enabled in the chat template. It reads the Qwen3 encoder's second-to-last layer and includes only real tokens, marked by the condition's mask. The transformer divides the Flux VAE's latent into 2x2 patches, pads the image and prompt separately to multiples of 32 tokens with learned pad tokens, and refines them separately. It then runs one stream over both, image first, with rotary positions based on each row's padded prompt length.

The source takes time 1 - sigma and returns the negated flow; `ZImageTransformer` takes Dew's model time and returns the flow. The pipeline guides as `pos + 5.0 (pos - neg)`, equivalent to Dew's `CFG(6.0)`, against the empty prompt. Tiny transformers match the source to 1e-5 of the largest output and 1e-4 in every gradient. A tiny pipeline built from the published configs matches the source's four-step call within 1e-5 (`tests/test_z_image_source.py`).

Wan 2.1 (text to video) uses `WanConditioner`. Its pipeline cleans prompts with ftfy (the `wan` extra, `pip install 'dewml[wan]'`), unescapes HTML entities twice, collapses whitespace and pads to 512 UMT5 tokens. It runs the encoder under the padding mask and zeroes states past each prompt's real tokens; the transformer reads all 512. UMT5 is the T5 tower with a relative position bias table in every layer (`T5EncoderTransformer(per_layer_bias=True)`, set by a `umt5` config).

The Wan VAE uses the source's chunked encoding and decoding: the first frame alone, then four frames per latent frame, retaining each causal convolution's last frames between chunks (`chunked_moments`, `chunked_decode`). The chunk size held in memory therefore does not depend on clip length. A 480x832 decode peaks at 4.6 GiB on an RTX 4080 for 33, 49 or 81 frames. One-pass decoding needs 10 GiB at 33 frames and 14 GiB of temporaries at 49.

`WanTransformer` divides the latent into 1x2x2 patches. Self-attention uses a three-axis rotary table over (frame, row, column), with queries and keys normalized across all heads; cross attention reads the text. The sample is a clip, `Field("video", (frames, height, width, 3))`, defaulting to 81 frames at 480x832 unless the index or `load_diffusion_source(size=(frames, height, width))` overrides it. Frame count is 1 + 4k, and the VAE decodes it whole. `text_to_image()` returns `[N, frames, height, width, 3]`. The pipeline guides at 5.0 against the empty prompt for 50 steps on its UniPC grid with flow shift 3.0.

Tiny transformers match the source's float32 flow and gradients within twice its own RMS distance from float64. A tiny pipeline's final latent differs from the source's by 2.8e-6 (`tests/test_wan_source.py`).

`SourceTask` stores the published pipeline's call policy: its `__call__` step and guidance defaults, and its grid with the pipeline's sigma origin and latent geometry. This makes `text_to_image()` run the source policy. If a directory declares no pipeline, it uses the family's reference pipeline. Dew refuses a declared pipeline it does not implement or one whose denoiser differs from the directory's, without substituting another pipeline's defaults.

`SourceSchedule.from_config` reconstructs eighteen published scheduler classes: DDIM, PNDM, DDPM, LMS, Euler, Euler ancestral, Heun, KDPM2, KDPM2 ancestral, DPM-Solver multistep, singlestep and SDE, DEIS, UniPC, EDM DPM-Solver, LCM, TCD and flow-matching Euler. The returned process, solver and grid use the controls and defaults of the declared class, and the solver is resolved once when the file is read.

Supported policies include source timestep spacing and offsets; Karras, exponential and beta sigma grids; finite lambda clipping; zero-terminal-SNR rescaling; terminal sigma selection; solver orders and correctors; DDPM fixed posterior variances; and clipping or dynamic thresholding in the source conversion order. DDIM and PNDM retain fixed training strides. LCM and TCD select grids from `original_inference_steps`, and two-evaluation solvers retain the source stage sigma and model time.

EDM uses its own sigma range, data scale, rho, log-sigma model time and signed output preconditioning. Its training process draws EDM sigmas rather than using a VP beta table. A flow-matching file specifies the static or resolution-dependent shift, terminal stretch and both sigma origins used by its pipelines. A resolution-dependent shift reads the calling pipeline's latent token count through the task's grid; `text_to_image()` sets that grid to the checkpoint's own geometry.

Unimplemented active controls fail explicitly: Lu-lambda and flow-sigma grids, UniPC external predictors, nonlinear sigma interpolation, continuous model times, learned variances, DDPM log-space wide variance, and PNDM velocity-domain history. TCD clipping/thresholding and epsilon-prediction UniPC thresholding are also refused because the published update does not apply them. Sigma-indexed source schedulers whose initial model time repeats are refused when preparing their grid: the source starts at the second match and cannot complete its published evaluation list. Repeated noninitial stage and corrector times remain supported.

The small reference cases in `tools/diffusers_source_reference.py` run actual Diffusers scheduler objects. Ordinary stochastic trajectories share explicit Gaussian draws. DPM-Solver SDE runs the actual `torchsde` tree and compares native solver results using its recorded increments; separate tests exercise the native Brownian bridge law.

All native source trajectories and VJPs run in float32. Trajectories use a `1e-4` scaled-error bound. For every source VJP, `tests/reference_error.py` requires its RMS distance from exact source evaluation to be at most twice the float32 source's distance. Exact evaluation constructs scheduler tables and performs step arithmetic in float64, keeping the same initial float32 latent, cotangent and recorded random draws. Its gradient stays in float64 without rounding back to float32.

<!-- not run: needs a local diffusion checkpoint directory -->
```python
from dew.interop import PretrainedPipeline

source = PretrainedPipeline.load("./image-checkpoint", dtype=jnp.float32)
images = source.text_to_image()(["a flower"], steps=20, key=0).host().images
objective = source.diffusion_objective()
```

Training batches contain uint8 NHWC images and `source.inputs.tokenize(captions)`. A nine-channel inpainting source also specifies `inputs.mask`: binary, one-channel NHWC masks with white marking the region to repaint. Caption dropout preserves the mask and masked-image latents. These inputs condition the network but do not guarantee that decoded unmasked pixels equal the original image.

`Pretrained.load(..., mesh=, layout=)` and `load_diffusion_source(..., mesh=, layout=)` stream pipeline weights the same way as decoder weights. They read each leaf from the mapped checkpoint one device shard at a time, placing it before reading the next. The host therefore never holds the translated pipeline. Convolution kernels and a UNet's per-head attention kernels are small and are read whole. The placed tree matches the host-loaded tree bit for bit.

`load_diffusion_source(..., text=False)` reads and downloads every component except the text encoder. Load the family's conditioner separately to encode prompts, for example `WanConditioner.from_pretrained(checkpoint, mesh=MeshSpec())`. `TextToImage.prepare(conditions=..., unconditional=...)` accepts these encodings as `{"conditioning": condition}`, with one row per sample, in place of prompts. Each row gets the same noise as a prompted call, so both calls produce the same images. Without encoded `unconditional`, this call applies no guidance. A pipeline loaded with `text=False` refuses prompts, `diffusion_objective` and `save`, since they require the text encoder.

Prepared `DenoisingInputs` can supply encoded native conditions and initial latents. Explicit grids pass to `sample(times=..., final_denoise=False)` when the last latent is the result; without an explicit grid, the existing `steps` and final clean-prediction convention remain unchanged. Published tabulated PRK grids own their integer half-interval rule; ordinary native grids retain exact half-intervals.

## Configuration and registries

Code builds every model, preset, solver, dataset and metric from its class. `dew.registry` maps the names that configuration files, the command line and run records use to those classes and back; `RunConfig.from_dict` and `ModelConfig.build` read a record through it.

`RunConfig.save` writes the run configuration. It is separate from the state checkpoint. [Recipes](../recipes.md) describes the configuration entry points and their side effects.

The model field `attention_impl` selects the attention kernel. `'reference'` uses einsum and softmax and is the only path that reads `dtype`, `precision` and `force_fp32_for_softmax`. `'xla'` and `'cudnn'` select the two implementations of `jax.nn.dot_product_attention`, while `'tpu'` uses Pallas splash attention. `'auto'` resolves on each trace, allowing a configuration logged as `'auto'` to run on another machine.

`'auto'` uses `'reference'` for arithmetic only that path performs: a matmul precision above DEFAULT, a softmax outside fp32, or a compute dtype different from the inputs'. Both `'auto'` and `'xla'` use it for float64 inputs (JAX's XLA attention otherwise narrows their softmax to float32) or bf16 on a GPU older than sm80.

Otherwise, `'auto'` selects the first applicable kernel:

- `'cudnn'` on a GPU of sm80 or later, with bf16 or fp16 inputs and a query head width divisible by 8 and at most 128. The call must have no softcap, no sinks and no `--xla_gpu_deterministic_ops`.
- `'tpu'` on a TPU backend with bf16 or fp32 inputs, query and key lengths divisible by 128 and at least 512, no additive bias, and a mask splash can describe. The kernel must receive whole sequences: sequence-parallel all-to-all supplies them, while key/value gather does not.
- `'xla'` otherwise.

Splash builds a block-sparse mask descriptor when compiling the executable. Causal and sliding-window sequences skip empty blocks rather than computing the whole rectangle. Head width is unrestricted because the kernel pads it, but query and key lengths must be multiples of 128 for the mask's blocks to tile them.

The older Pallas flash kernel handles explicit `'tpu'` calls with an additive bias, a mask supplied as a trace value (a KV-cache decode mask or the striped mask from sequence parallelism), a mask beyond the published cell budget, or lengths not divisible by 128. `'auto'` uses `'xla'` for these calls. Splash supports logit softcaps (Gemma 2), attention sinks (GPT-OSS), and packed batches supplied as segment ids rather than a mask. The flash kernel supports neither softcaps nor sinks, so an explicit `'tpu'` call that splash cannot describe refuses them. On GPU, `'auto'` uses `'xla'` for both, while `'cudnn'` refuses them by name.

Below 512 keys, `'auto'` stays on `'xla'`, which measured faster on a v6e (`SPLASH_MIN_LENGTH` in `dew/nn/attention.py`). The parameter tree is independent of the implementation, so checkpoints are interchangeable across hardware.

The [README model list](https://github.com/AshishKumar4/dew/blob/main/README.md#models) names which model configurations run the whole workflow; each task guide covers its own data and objective.
