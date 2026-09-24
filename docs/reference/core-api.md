# Core API reference

This page describes the interfaces the tutorials use and the contracts between them. Every public module also has its own page, generated from its docstrings; the list is at the [end of this page](#all-modules). For a complete example, read [your first training run](../getting-started.md).

## Objective

Import `Objective`, `Aux`, `Step`, `Mean`, `mean_loss`, and `scalar_loss` from `dew.objectives`.

| Member | Contract |
|---|---|
| `init(key, variables=None)` | Return a Flax variables mapping with a `params` collection. Pure; the trainer traces it once for shapes and once for values. `variables` is a held tree the caller supplies, which is how the trainer passes it as data; with `None` the objective uses its own (`DiffusionObjective.held_variables`, for example). An objective that holds nothing ignores it. |
| `loss(variables, batch, step)` | Return additive statistics and `Aux`. Use `Mean(total, mass)` for a shared denominator; a scalar denotes a unit-mass term. |
| `reduce_loss(statistics)` | Return `(value, has_data)`. Override for an objective-owned composite Flax PyTree. |
| `apply_effects(variables, effects)` | Return nonparameter replacements from additive accepted-window observations. Required when the objective emits effects. |
| `evaluate(variables, batch, step)` | Return an artifact, a tuple of artifacts, or `None`. The base method returns `None`. |
| `preview(variables, batch, step, *, scored=None)` | Return display artifacts for a tracker, or `None`. The base method reuses `scored`, the first scoring artifacts. |
| `ema` | Optional `EMASpec`; the base objective uses `None`. |
| `artifact` | Optional description of the objective's evaluation artifact type. |

`Step.step` counts accepted microbatches. Its `key` derives from consumed attempts, including rejected ones. `ema` holds selected averaged leaves overlaid onto the complete variables mapping, or `None`.

`Aux(metrics, variables=None, qk_stats=None, effects=None)` carries training measurements, sequential mutable replacements, QK maxima and additive deferred effects. The trainer applies effects once on a supported optimizer commit. `scalar_loss(objective, variables, batch, step)` returns a scalar and the same Aux for direct JAX differentiation.

### Collections and EMA selection

A variables tree is a nested mapping. Its outer keys name collections such as `params` and `batch_stats`; leaves are arrays such as a dense kernel or a running mean. The optimizer updates the `params` collection. A mutable Linen call returns replacement state collections, which the objective supplies through `Aux.variables`. See the [stateful example](../concepts/objectives.md#update-non-parameter-state).

`EMASpec(decay, select=everything)` comes from `dew.objectives.base`. `decay` maps completed optimizer-update count to a scalar. `select` accepts a tuple of keys naming a leaf; `under("params", "context_encoder")` selects that subtree, and `everything` selects all leaves. EMA arithmetic uses at least fp32 and preserves explicit fp64, then rounds each result to the initialized EMA leaf dtype. Unit decay selects the frozen leaf exactly. Router bias updates also retain the initialized bias dtype; integer load comparisons avoid converting large counts to floats.

### Input descriptions

Import `Field`, `Condition`, and `InputSpec` from `dew.inputs` or `dew`.

```text
Field(key, shape)
Condition(encoder, field="text", unconditional="")
InputSpec(sample, conditions={}, mask=None)
```

`Field.shape` is the shape of one sample, excluding batch. A `Condition` connects a condition encoder to its tokenized batch field. `InputSpec.conditions` maps the model keyword, such as `textcontext`, to a `Condition`. Each condition must read a distinct batch field. `mask` is an optional `Field` for a binary image mask, used for explicit masked-image latent conditioning. `InputSpec.tokenize(captions)` returns tokenized fields for those conditions; an empty condition mapping returns no fields. Encoders define tokenization, parameters, and encoded output types.

### Mesh and layout

Import these from `dew.training`:

```text
MeshSpec(fsdp=1, expert=1, tensor=1, sequence=1, stage=1, microbatches=None, replicas=1)
Layout(rules=DEFAULT_RULES, min_shard=65536, tolerance=0.02, host=(), host_parameters=())
build_mesh(spec, devices=None)
```

`build_mesh` uses the supplied devices or JAX's visible devices; the specified factors must divide their count, and data parallelism fills the remaining factor. Explicit pipeline microbatches require `stage > 1` and a positive multiple of the stage count. `replicas` above 1 builds a hybrid mesh whose data axis spans that many groups of granules (TPU slices, GPU hosts or NVLink domains, or processes where every device shares one slice), with every other axis inside a group; see [training on several nodes](../guides/multi-node.md#lay-the-mesh-out-for-the-network). A sequence axis above 1 splits every attention call's positions, and each call picks the all-to-all or the gather exchange from its shape.

`Layout.rules` accepts an ordered logical-axis rule sequence or a mapping of overrides. Mapping entries update the default table. When dimensions compete for one mesh axis, rule order determines precedence; a non-divisible dimension cannot use that axis. Valid parameter mesh axes are `fsdp`, `expert`, and `tensor`. `min_shard` counts elements, not bytes. `tolerance` is the permitted fraction of shardable parameter elements left replicated. `host` names train-state fields out of `params`, `opt_state` and `ema`. The named `opt_state` and `ema` stay in pinned host memory between steps and the step fetches them to the device. Naming `params` instead makes the CPU own the whole `TrainState`, including optimizer, EMA and accumulation: the optimizer transaction runs on a CPU companion of the mesh, and the runtime CPU device count must match the accelerator count on every process before JAX initializes. `host_parameters` holds glob patterns over logical parameter paths (`params/layers_*`) that an inference placement keeps in pinned host memory; only the `offloaded` placement reads them, and `check` refuses a layout that names them for any other placement. `shardings(mesh, tree)` returns a matching tree of placements; `check(params, shardings, mesh)` validates excessive replication. See [distributed training](../concepts/distributed.md).

## Trainer

Import `Trainer` from `dew` or `dew.training`.

```text
Trainer(objective, optimizer, *, key,
        mesh=MeshSpec(), layout=Layout(), accumulation=1,
        dynamic_scale=False, checkpoints=None, tracker=None,
        step=None, rollout=None, profile=None)
```

`objective` is an initialized objective object and `optimizer` an Optax gradient transformation. The required JAX `key` seeds initialization and the run. `mesh` and `layout` describe placement. Optional capability objects enable checkpoints, tracking, host-side rollouts, and profiling.

`accumulation` counts accepted microbatches per effective window. Shared means use a weighted gradient accumulator with at least fp32 precision, preserving float64 when enabled in JAX. A TPU has no float64 (XLA rewrites it into pairs of float32, which are not IEEE doubles), so a trainer whose state holds float64 on a TPU mesh is refused when it places the state. Each finalized gradient enters Optax in its parameter's dtype; partially accumulated gradients keep the wider working dtype. Composite statistics retain independent normalizers and inputs for scalar-VJP replay. Parameters and EMA stay fixed within the window; sequential mutable replacements retain their original read snapshots. `dynamic_scale=True` persists scale/history and rejects nonfinite working or optimizer-input gradients without discarding the accepted prefix.

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

Import `Dataset`, `DataPartition` and `Loading` from `dew.data`.

```text
Dataset(train, val, records, batch, ramp=None)
DataPartition(index=0, count=1, readers=1)
Loading(workers=32, threads=64, read_buffer=128, worker_buffer=2)
```

`train(partition)` opens a training iterator, and `val(partition)` opens one finite validation pass, or `val` is `None`. Each reads the share of every global batch the `DataPartition` names: the `index`th of `count` disjoint shares, which `readers` processes read alike. `dew.training.data_partition(mesh)` is the share a process reads on a mesh, and `DataPartition()` is every row. `records` is the known training-record count or `None`; `batch` is global. `steps_per_epoch` is integer division of records by batch, or `None`. `epoch_steps(epochs=1)` requires a finite record count. `ramp` is set when the run grows its batch over its first records; `batch` is then the batch the ramp ends at.

Each factory call must return a fresh, exclusively owned iterator. Ordinary `close()` is finalization and must not race `next()` or checkpoint operations. A source that needs to interrupt blocking reads may additionally implement `request_stop()`: a thread-safe, nonblocking, idempotent signal, safe alongside both `next()` and `close()`. Tokenized wrappers forward these operations.

`DevicePrefetchIterator(iterator, mesh, depth=2, source_state=None)` in `dew.training.distributed` takes ownership on successful construction and starts its worker lazily on the first `next()`. Saved-position restoration also runs there, so its failures unwind through the already-owned iterator. Closing an unused iterator starts only its finalization, not a read or restoration. Use it in a `with` block, or call `close(timeout=5.0)`, even when a loop consumes only a fixed number of batches. Depth must be positive; the bound is `depth` queued device batches plus at most one in-flight batch, excluding the consumer and upstream buffers. Its `source_state` describes the last delivered batch, never speculative read-ahead. EOF and source failures follow preceding queued batches; early close discards unread data and speculative failures, but reports finalization failures.

The prefetch worker performs iteration, checkpoint operations, and final source close. Closing requests cancellation, discards queued batches, and joins the worker. A `TimeoutError` means the source read, placement, or finalization did not cooperate: the thread was not killed and its in-flight references may remain. Cancellation stays requested and close can be retried. After close, further iteration stops.

Grain limits how cleanly Dew can shut a loader down. The installed `DataLoaderIterator` exposes no public close, so Dew releases its owned references without reaching into private iterators or changing the sampling pipeline. Local-record probes release the source and child processes, but that is not a deterministic upstream shutdown contract. Grain also keeps a process-wide shared-memory deletion thread pool. Its `DatasetIterator.close()` is called on the iteration thread, but read-executor shutdown does not wait for already-running record reads. Arbitrary blocked upstream reads remain outside Dew's shutdown guarantee.

`Loading` controls Grain concurrency and buffers for built-in specifications. Use zero worker processes for small local examples; the defaults may be excessive for a tiny dataset. See [data preparation](../concepts/data.md) for field layouts and process partitioning.

Every dataset specification declares `seed` and `loading` as keyword-only fields of `DatasetSpec`, and `spec.load(batch=, tokenize=None)` is the call on all of them; a specification that writes no captions raises `TypeError` for a `tokenize` reader it cannot use.

An image specification takes validation from `val_split`, a split of the dataset's own bounded by `val_batches` batches, or, with `val_split` unset, from `val_batches * batch` records held out of the head of the training source.

`Dataset.from_grain(train, *, batch, validation=None, records=None, loading=Loading())` builds a run over Grain pipelines a caller assembled: a `MapDataset` is repeated, cut into the reader's share and saved as one global record count; a pipeline read as it comes arrives as a function of the `DataPartition` that builds the `IterDataset` of that share, which is batched where it is and reports Grain's own iterator state.

A token corpus is a `TokenSource`: `TokenBytes` over a `.bin` file, `TokenRecords` over ArrayRecord shards of token arrays, or `TokenColumn` over a parquet column of them. `TokenWindows` and `PackedTokens` read `path` as a directory of `train` and `val` files and take whichever store their suffix names, so the same corpus gives the same windows and the same packing plan in all three.


## Checkpoints

Import `Checkpoints` from `dew`.

`Checkpoints(directory, *, keep=2, local_directory=None, local_every=None)` configures persistent storage and optional local emergency checkpoints. Local directory and cadence must be specified together. The object opens storage on use.

`save(step, state, saved, metrics=None, *, share=None)` schedules a persistent save; `saved` is the iterator position as bytes, or `None`, and `share` the `DataPartition` its stream read, required with a position. `wait()` waits for pending writes. `latest` reports the newest eligible checkpoint. `restore(template=None, step=None, *, share=None)` returns restored state data and the iterator position the reader of `share` resumes from, or `None` without a share; a template controls structure and placement. `path(step)` identifies the persistent checkpoint location.

This object does not write `run.json`; run configuration saving is separate. Use the [complete resume example](../guides/checkpoints.md) before adapting these lower-level calls.

## Language modeling and generation

Import `LMObjective` from `dew.objectives.lm`.

```text
LMObjective(model, seq_len, *, ema_decay=0.999, pad_id=None, head_chunks=4,
            samples=None, pretrained=None, balance_rate=None, aux_loss_alpha=None,
            seq_aux=True, loss_role=None, mtp_weight=None, z_loss=0.0, qk_stats=False,
            indexer=None, trainable=None)
IndexerTraining(phase, weight=1.0)
```

The model must implement Linen `hidden_states(tokens, train=..., positions=..., segment_ids=...)`, returning `(B, S, D)`, and `head_weight(params)`, returning the `(D, vocab)` vocabulary matrix. The objective also reads `final_logit_softcap` and `precision`. Prediction-depth training needs `mtp_hidden_states` and compatible prediction-depth configuration. Mutable router, QK and indexer collections are required when their options are enabled.

`pretrained` supplies the complete variables tree; `loss_role` requires aligned `text_roles`. `pad_id` masks matching targets. `head_chunks` controls vocabulary tiling; `samples` configures text previews. `ema_decay=None` trains without an averaged copy, and `1.0` retains a frozen one. Routing balance, auxiliary loss, prediction-depth weight, and QK statistics require matching model computation. These interfaces make `LMObjective` specific to compatible decoders. `z_loss` adds PaLM's auxiliary term, the coefficient times the squared log partition of every counted prediction; zero adds nothing. `trainable` is a path filter over the parameter leaves the optimizer moves; the rest of the tree is kept under `frozen`. An adapter's filter, `dew.lora.LoRA.trainable`, goes here. `None` trains every leaf, and `trainable` cannot be combined with `indexer`.

`indexer` trains DeepSeek-V3.2's lightning indexer on a model whose `mla` mixer names `index_n_heads` and `index_head_dim`. `IndexerTraining("warmup")` needs a mixer without `index_topk`: the model runs dense attention, `init` keeps the indexer alone in `params` and the rest of the tree under `frozen` (a `pretrained` tree may omit the indexer, as a dense checkpoint does), and the loss is the KL of the indexer's softmax from the attention distribution, reported as `indexer_kl`. `IndexerTraining("sparse")` needs a mixer with `index_topk`: the whole tree trains, the cross entropy trains the main weights and the KL over the selected keys trains the indexer, whose inputs are detached; a warm-up checkpoint's split tree is accepted as `pretrained`. `weight` scales the KL term.

Import `generate`, `Sampling` and `Generation` from `dew.sampling`:

```text
generate(model, params, inputs, max_new_tokens, *, key=None, seed=None,
         sampling=Sampling(), n=1, logits=None, stopping=None, strategy=None) -> Generation
Sampling(temperature=1.0, top_k=None, eos_id=None, pad_id=0, top_p=1.0, min_p=0.0)
```

`params` is the complete variables tree. `inputs` is a `ModelInputs` from `dew.nn.inputs`, or an integer `(B, P)` array normalized to all-valid text. `ModelInputs.token_fields["attention_mask"]` identifies real token slots; there is no separate generation length argument. Every row needs a real token. Only real tokens count against `model.max_seq_len`. Conditioning arrays are batch-aligned and used during prefill; decode keeps the model's cached logical positions. Exactly one of `key` and `seed` is given; `seed=n` is `jax.random.key(n)`.

The compiled decoder uses one padded input shape with per-row cache cursors and a fixed trip count. Finished rows preserve their cached state. On a mesh, rows split over the batch axes and the result keeps that sharding; each process hands in its own rows, at the same count and padded width on every process, and reads them back with `Generation.host()`. Keys fold in the global row index, so a pool draws what one process draws for the same rows. The cooperating processes also pass the same `n`. Invalid input on one rank raises on all ranks before device execution.

`n` is the number of continuations drawn per prompt and must be a positive integer. The continuations of a prompt share its prefill and then run one after another on the device, with the prompts of each continuation batched as before: decode time grows with `n`, one continuation's cache working memory is reused by the next, and only the output storage grows with `n`. A prompt's key is its global row key: continuation zero draws with that key, so `n=1` and continuation zero of a larger request are the same draw, and continuation `j` folds `j` into it, so raising `n` leaves the continuations already drawn unchanged. Row padding on a mesh pads prompts before the continuations exist, so a prompt's `n` rows stay together on the process that asked for them and `host()` drops only padded prompts' rows.

`Sampling.eos_id` accepts an integer or a tuple of ids; any of them terminates a row. The value normalizes the ids into an immutable tuple. Stochastic selection applies temperature, top-k, nucleus top-p, then relative min-p filtering. At least one token survives. `top_p=1` and `min_p=0` disable their filters. Zero temperature selects argmax without filtering. A `Sampling` value is a convenience over the components below: it compiles to those four transforms, in that order, plus an EOS criterion, and `generate` appends them after whatever `logits` and `stopping` hold.

`Generation.tokens` includes the original prompt and has shape `(B * n, P + max_new_tokens)` with `B` the placed prompt rows, prompt zero's `n` continuations first and the prompts in request order. `lengths` counts response tokens including EOS. `terminated` marks EOS termination; false means the token budget. Slots after termination hold `Sampling.pad_id`. `behavior_log_probs` and `raw_log_probs` have shape `(B * n, max_new_tokens)`; the first describes the filtered distribution that drew each action and the second the unmodified policy. Each row carries its own length, termination and likelihoods. `rows` counts this process's real prompts times `n`; `host()` returns the record over host arrays of those rows; `text` decodes them through the processor a task bound, one string per row.

`LMObjective.per_token_log_probs(params, tokens, left_padding=...)` scores the raw policy. It left-aligns real tokens for the forward and restores the original next-token alignment. Unscored padding slots are zero. `SampledRollout` records the raw sampling-time values as `old_log_probs` and preserves actual draws as `behavior_log_probs`, in the packed layout `GRPOObjective.packed_log_probs` rescores. Reward text excludes EOS and padding. When a batch carries no `old_log_probs`, GRPO uses the behavior probabilities as the old policy and its behavior corrections follow verl's bypass mode. The next-token objective refuses models declaring `causal=False`.

### Decoding components

Decoding has three extension points: logits transforms, stopping criteria and strategies. Import the protocols from `dew.sampling` and the built-ins from `dew.sampling.decoding`.

```text
LogitsTransform: (StepState, logits[rows, vocab]) -> logits[rows, vocab]
Stopping:        (StepState, drawn_tokens[rows]) -> finished[rows]
Strategy:        (DecoderState, StepState, DecodeOps, transform, stopping, budget, n) -> Draws
StepState(tokens, valid, step, active, keys, prompt_width)
```

`StepState` is the whole input of a transform or a criterion. `tokens` is the fixed-capacity buffer of the prompt followed by the draw slots, `[rows, prompt_width + max_new_tokens]`, and `valid` marks the slots holding a real token, so a row reads its own history whatever padding its prompt batch needed. `step` counts the tokens a row has committed, `active` marks the rows still generating, and `keys` holds one PRNG key per row. `state.history()` returns each row's real tokens left aligned with their count, `prompt_history()` and `generated()` the two regions, and `total()` the real token count. A transform never sees model parameters or cache internals.

`logits` is the whole transform chain, in the order it runs. Left as `None` it is what `sampling` compiles to, an explicit sequence replaces that entirely, and `()` runs no transform, so a caller who needs an order `Sampling` does not produce writes the order they want. A call's value replaces a task's bound one, and an explicit `sampling=` on a call also clears a bound chain, because that chain was built around the policy the call just replaced. `stopping` composes instead: an explicit sequence runs beside the policy's EOS criterion rather than replacing it, so naming a criterion cannot drop termination. Criteria combine with OR and run after every committed token; the token that fired one is emitted with its likelihoods and later slots hold `pad_id` with zero likelihood.

Built-in transforms are pytrees, so a configuration holding arrays travels as data rather than entering a compilation cache key. A plain function works too, and `jax.tree_util.Partial(fn, array)` carries array configuration for one. Everything runs inside the compiled loop; there is no host callback. Across a pool the resolved components are compared by their structure and by the contents of their configuration arrays, so two ranks banning different tokens are refused instead of each running its own policy.

```python
import jax.numpy as jnp
from dew.sampling import Beam, Sampling, Speculative, decoding, generate


def favor_short(state, logits):
    """Raise the end token's score once a row has drawn eight tokens."""
    return logits.at[:, 2].add(jnp.where(state.step >= 8, 3.0, 0.0))


# The chain is complete, so the policy's own filters are written into it.
drawn = generate(
    model, variables, prompts, 32, seed=0,
    sampling=Sampling(eos_id=2, pad_id=0),
    logits=(decoding.RepetitionPenalty(1.1),
            decoding.NoRepeatNGram(3),
            decoding.FrequencyPenalty(0.4),
            favor_short,
            decoding.Temperature(0.8),
            decoding.TopP(0.9)),
    stopping=(decoding.MaxNewTokens(24),))

# The same request as a deterministic search over four beams, returning two.
searched = generate(model, variables, prompts, 32, seed=0,
                    sampling=Sampling(eos_id=2, pad_id=0),
                    logits=(decoding.NoRepeatNGram(3),),
                    strategy=Beam(width=4, length_penalty=1.0), n=2)

# Or drafted by the model's own prediction depths, with the same law as the
# first call and fewer target forwards.
drafted = generate(model, variables, prompts, 32, seed=0,
                   sampling=Sampling(temperature=0.8, top_p=0.9, eos_id=2),
                   strategy=Speculative(block=4))
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
| `TopH(fraction=1.0, candidates=100)` | `TopHLogitsWarper` | `candidates` is the reference's fixed head |
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

`stop_strings` reads the tokenizer once, on the host, and compiles where every token's piece can sit inside each stop string and how many of the string's trailing units its start can cover. The criterion then runs entirely on device and never decodes. A string counts only when it touches the token just drawn, so a string produced earlier does not stop the row later, and a string spelled across several tokens or overhanging either end does stop it. A byte-level or byte-fallback vocabulary is read through its byte spelling and matched over UTF-8 bytes, so a stop string whose code point two tokens split still ends the row; every other vocabulary is read through `convert_tokens_to_string` behind an ordinary prefix, because a decoder adds or removes a leading space depending on what came before. `tokenizer` is a Transformers tokenizer or a task `Processor` holding one, and `vocab_size` sizes the table for a model whose head is wider than the vocabulary.

An active row is drawable only when every score is finite or `-inf` and at least one is finite. A NaN or a `+inf` beside a finite score makes the draw arbitrary, and a row with nothing finite has no distribution at all, so `generate` raises instead of returning an index; a zero-temperature policy passes such a row through rather than collapsing it onto an argmax. The same holds for the model's own distribution: an undefined one cannot be reported truthfully, so it raises rather than returning a NaN likelihood, whatever a later `RemoveInvalidValues()` does to the scores. Add that transform to repair scores a chain itself made invalid.

#### Strategies

```text
Sample()
Beam(width=1, length_penalty=1.0, early_stopping=False, stop_ids=1)
Speculative(block=4, confidence=0.0)
```

`Sample` draws every row independently and is what a request without a strategy runs.

`Beam` is deterministic beam search, with `_beam_search`'s bookkeeping from Transformers 5.16.1: a step keeps the best `(1 + stop_ids) * width` continuations so `width` live beams always remain, a criterion moves one into the completed set with its score divided by its generated length raised to `length_penalty`, and `early_stopping` takes the reference's `False`, `True` and `"never"`. The prompt is prefilled once and copied into `width` cache rows, which each step reparents, so a branched beam decodes exactly like a separately selected prefix. `n` is how many completed hypotheses to return and `n > width` is an error. A selected path is a search result rather than a draw, so its behaviour log probability is zero while the raw ones stay the model's own. Sampling with beams is refused: the marginal probability of a selected beam is not the per-step candidate probability, so there is no correct behaviour likelihood to record.

`Speculative` drafts with the model's own prediction depths and verifies with the model, following algorithm 1 of [arXiv 2211.17192](https://arxiv.org/abs/2211.17192) as `_speculative_sampling` applies it. The first candidate of a block is an ordinary target draw, so it is always accepted, and each depth chains the next from the previous hidden state and the candidate's embedding. A proposal is accepted with probability `min(1, p(x) / q(x))` for the target's post-transform `p` and the draft's actual `q`; the first rejection draws from the normalized positive part of `p - q`, and a block with nothing rejected draws a bonus from `p`. The emitted tokens are therefore distributed exactly as `Sample` distributes them, though not draw for draw at one seed. Every emitted action records the target's post-transform log probability as its behaviour and the model's own as its raw value; the draft's `q`, the acceptance probability and the residual are never recorded. `confidence` stops the draft after the first candidate the draft is less sure of, as `ConfidenceCriteria` does; a candidate the draft never offered was not rejected, so the block then ends on an ordinary target draw. A model without prediction depths is refused rather than silently falling back.

The target cache is saved before a block and the accepted prefix is replayed into it, because a recurrent mixer keeps a running summary no cursor can rewind. The prediction cache is rebuilt on the same invariant: depth `d`'s entry for token `t` reads depth `d - 1`'s hidden state at `t - 1` and `t`'s own prepared embedding, at `t`'s own coordinate, with the target as depth zero's predecessor. That is what `mtp_hidden_states` trains the depths on, what `MTPCandidateGenerator` corrects with and what `Qwen3_5MultiTokenPredictor.forward` takes. Each depth carries its last state across blocks, so a boundary loses no entry, and the prompt seeds every depth from the embeddings the prefill already prepared, media replacements included, without running an encoder again.

A depth becomes usable after enough real tokens have preceded it. Rotary offsets and repeated image coordinates do not change that count. A newly available predecessor is retained even if its next depth cannot yet write a cache entry.

A model with a block drafter, DeepSeek-V4.1's DSpark, drafts a block's later candidates in one pass instead of chaining depths. The drafter reads the context its target layers record: the prefill seeds it with the prompt's, and each replayed block appends its kept positions'. The strategy draws each candidate from the drafter's logits for its position, Markov bias included, as the pass reaches it, so `block - 1` may not exceed the drafter's own block size. DSpark drafters for a V4 trunk (DeepSeek-V4-Flash-DSpark) are not built: the stages are V4.1's Single-Pass mHC blocks.

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

A loaded source's `generation_config.json` is data. Every control Transformers 5.16.1 writes there is classified: the native policy carries it, a transform, criterion or strategy carries it, the task owns it, it is provenance, or `Pretrained.text_generation()` refuses it and says why. The transforms a source binds are the complete chain, built in `_get_logits_processor`'s order, so the policy tail lands where the reference puts it; a source running beam search ends its chain after the processors, because the search picks its own continuations. An unset control, or one at the value where `generate()` adds no processor, criterion or search mode, is inert. Beam-only and sampling-only controls are judged only when beam search or sampling is active, as they are upstream.

Each source control has one rule for its consumer, neutral value, mode and refusal. Source value precedence remains `generation_config.json`, wrapper config, then text config. The source resolver selects the actual policy once. `Sampling` supplies convenience defaults at the request boundary; only resolved transforms, criteria, strategy and padding reach the compiled decoder. An explicit chain therefore has no unused sampling settings in its compilation or process-agreement identity.

| Control | Native mapping | Refused because |
| --- | --- | --- |
| `do_sample`, `temperature`, `top_k`, `top_p`, `min_p`, `eos_token_id`, `pad_token_id` | `Sampling` | |
| `max_length`, `max_new_tokens` | the task's token budget | |
| `num_return_sequences` | the task's `n`, independent of any `sampling=` override | |
| `bos_token_id`, `decoder_start_token_id` | inapplicable to supplied-input causal decoding; the tokenizer prepares special tokens | |
| `max_cache_len` | capacity assertion only; it does not resize the cache or limit the request | a length above the model's `max_seq_len` |
| `repetition_penalty` | `RepetitionPenalty` | |
| `encoder_repetition_penalty` | `PromptRepetitionPenalty` | |
| `no_repeat_ngram_size` | `NoRepeatNGram` | |
| `encoder_no_repeat_ngram_size` | `PromptNoRepeatNGram` | |
| `sequence_bias` | `sequence_bias` | |
| `bad_words_ids` | `bad_words` | |
| `min_length`, `min_new_tokens` | `MinLength`, `MinNewTokens` | |
| `forced_bos_token_id` | `ForcedBOS` | |
| `forced_eos_token_id` | `ForcedEOS` at the request's own end, so a per-call budget moves it | |
| `suppress_tokens`, `begin_suppress_tokens` | `SuppressTokens`, `BeginSuppressTokens` | |
| `exponential_decay_length_penalty` | `ExponentialDecayLengthPenalty` | |
| `remove_invalid_values` | `RemoveInvalidValues` | |
| `renormalize_logits` | `Renormalize` | |
| `typical_p`, `epsilon_cutoff`, `eta_cutoff`, `top_h` | `Typical`, `EpsilonCutoff`, `EtaCutoff`, `TopH` | |
| `stop_strings` | `stop_strings` | without the source's processor or the model's `vocab_size` there is no vocabulary to compile |
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
pipeline(source, *, mesh=None, layout=None, dtype=None, param_dtype=None, ema=True, step=None,
         revision=None) -> TextGeneration | BlockGeneration | MaskedGeneration | TextToImage
Objective.pipeline(state, *, ema=True) -> the objective's task over state.averaged or state.params
LMObjective.pipeline(state, *, ema=True, processor=None) -> TextGeneration
TextGeneration(model, variables, processor=None, sampling=Sampling(), max_new_tokens=None,
               max_length=None, n=1, logits=None, stopping=(), strategy=None)
task(request, max_new_tokens=None, *, key=None, seed=None, n=None, sampling=None,
     images=None, logits=None, stopping=None, strategy=None) -> Generation
task.bind(variables) -> TextGeneration      task.decode(generation) -> tuple[str, ...]
TextGeneration.from_run / BlockGeneration.from_run / MaskedGeneration.from_run
    (directory, *, ema=True, step=None, mesh=None, layout=None, dtype=None, param_dtype=None)
TextGeneration.from_pretrained / BlockGeneration.from_pretrained / MaskedGeneration.from_pretrained
    (repo_id, *, ema=True, step=None, mesh=None, layout=None, dtype=None, param_dtype=None)
BlockGeneration(model, variables, process, processor=None, eos_token_ids=(), pad_token_id=0,
                max_new_tokens=None, max_length=None, n=1)
task(request, max_new_tokens=None, *, key=None, seed=None, n=None, process=None,
     images=None) -> CanvasGeneration
Pretrained.text_generation(*, sampling=None) -> TextGeneration | MaskedGeneration
Pretrained.block_generation() -> BlockGeneration
Pretrained.text_to_image() -> TextToImage
PPOObjective.pipeline(state, *, ema=True, processor=None) -> TextGeneration
TextToImage(model, process, inputs, params, autoencoder=None, steps=50, guidance=None,
            sampler=DDIM(), grid=None, final_denoise=True, finish=None, blank=None)
TextToImage.from_objective(objective, variables) -> TextToImage
TextToImage.from_run(directory, *, ema=True, step=None, mesh=None, layout=None, dtype=None,
                     param_dtype=None)
TextToImage.from_pretrained(repo_id, *, ema=True, mesh=None, layout=None, dtype=None,
                            param_dtype=None)
LMObjective.policy(params, sampling=Sampling()) -> TextGeneration
image_task.bind(variables) -> TextToImage
image_task.prepare(prompts, *, key=None, seed=None, steps=None, unconditional=None,
                   image=None, image_latents=None, mask=None, noise=None, initial=None,
                   times=None, encode_key=None) -> DenoisingInputs
image_task(prompts_or_prepared, *, steps=None, guidance=<default>, sampler=None, key=None,
           seed=None, decode=True) -> Images
RunProcessor(tokenizer)   # a run's ByteTokenizer or HFTokenizer as a task processor
Server.from_task(task, *, slots, capacity, admission=None) -> Server
```

A task captures the variables mapping at construction and on `bind`. Replacing the caller's mapping does not change the existing task. Array buffers remain shared; do not mutate, donate or delete them while a task uses them. Text requests need a processor. Numeric token rows remain integers: mixed text and token rows, floats, booleans and strings are refused; a resident `jax.Array` or `ModelInputs` reaches the model without a host copy.

`max_new_tokens` takes precedence over a source default. If only `max_length` is declared, the budget is that total minus the padded prompt width. Otherwise an LM run records its `sample_tokens` and `sampling` value; an objective uses its `Samples`. With no limit the call must provide one. A call's `n` takes precedence over the task's bound count the same way, including `n=1` over a source that asks for more; omitting it uses the bound count. Equal shapes and controls reuse the compiled executable across calls and `bind`.

`LMObjective.policy(params)` binds those parameters directly. DPO, GRPO and PPO pipelines publish the trained policy, not their frozen loss reference; PPO also removes the critic. For other generative objectives, `ema=True` requires the moving-average state and raises if it is absent. Use `ema=False` for live weights. `TextToImage.from_run` reads `run.json` and the latest checkpoint under one directory, merging the EMA copy over the live parameters unless `ema=False`; `from_pretrained` pulls a published run directory from the Hub first. The three text tasks construct the same way from the kinds they generate for, and `dew.pipeline` picks the task class from the objective name in `run.json`. `pipeline`'s `dtype` selects the computation dtype and `param_dtype` the parameter storage: `None` keeps a run's stored dtypes and uses FP32 masters for a source, and `"auto"` keeps the stored dtypes of either.

Source-default text tasks preserve temperature, top-k, top-p, min-p, EOS and padding settings as their `Sampling` value, bind the source's complete chain as `logits`, its criteria as `stopping` and the strategy its config names, and take their return count from `num_return_sequences`. An explicit `sampling=` replaces the policy and the chain, and the controls behind them are then neither built nor judged, so a distribution control the caller just replaced cannot block the call; the criteria, the strategy and the return count still come from the source, and every control the task keeps is judged as always. An unknown control name is always refused, because no consumer is defined for it. A control the native decoder does not implement raises when the default task is created, naming the control and the reason; the table above lists every one. Loading weights for training or export does not select a decoding policy.

`BlockGeneration` uses `BlockProcess.generate`; its `CanvasGeneration` carries lengths, termination and decoder-step counts, without autoregressive likelihoods, plus the same `rows`, `host()`, `text` and continuation rows. Its continuations refine the shared encoded prompt independently, each over the original prompt rows, so the batch-wide canvas draw a row sees is the one a single continuation sees. `TextToImage` carries the objective's or source's `steps`, `guidance` and `sampler` defaults; `prepare` encodes prompts and draws their noise once, placed for the task's mesh, and `Images.images` is `[rows, H, W, C]` in [-1, 1] with `host()` reading a process's rows back. `grid(steps)` answers the process and the explicit time grid a trajectory of that length walks, for a source whose sampler pairs its own sigma and model-time tables; the noise prior follows that process, so `prepare` takes the same `steps`. `final_denoise=False` ends a trajectory at the last grid point without the closing clean prediction. `sample(denoise, x_T, steps=None, *, solver, guidance=None, key, times=None, final_denoise=True)` in `dew.sampling` takes the same two controls; exactly one of `steps` and `times` is passed, and an explicit grid decides the trajectory's length. `finish(params, images)` runs on the decoded images under the same placement, for a source that ships a checker or an output transform. `blank` is the task's unconditional branch, encoded once by whoever built the task (`DiffusionObjective.blank_conditions`); `None` encodes it on every call. Rebinding preserves the compilation identity of every task.

`Server.from_task(task, slots=, capacity=)` serves a `TextGeneration` task with continuous batching over one resident KV cache. It holds `slots` rows of `capacity` cache slots each, admits queued requests into free rows, and runs one compiled step per iteration over every row. Every request runs the task's bound policy, and a request draws with the key it was submitted with, so a served request draws the same tokens as the same request run alone. `submit` returns a ticket that resolves to the request's `Generation`, `step` runs one iteration, `run` steps until the queue and rows are empty, and calling the server with a batch of prompts does both.

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

The convenience call returns `Completion(texts, finish_reasons, token_counts, usage, responses)`. Missing per-choice token counts and finish reasons remain `None`. Aggregate OpenAI usage stays separate from the per-choice fields. `responses` retains the SDK models, including the logprobs, token data and extensions the backend reported.

An explicit `sampling=Sampling(...)` sets the native policy controls supported by the selected backend. Ollama receives neutral repetition/presence/frequency penalties and explicit top-k/top-p/min-p values, so its hidden `repeat_penalty=1.1` default does not alter the request. Conflicting explicit options are refused. `OpenAICompletion(..., provider="vllm")` or `provider="sglang"` enables the engine Sampling translation both engines accept, including `repetition_penalty=1.0` and optional EOS-token IDs; without either the client refuses `top_k`, `min_p` and `eos_id` rather than dropping them. Without a Sampling value, provider defaults or the caller's SDK options apply. A backend's tokenizer can segment the same prompt differently from the exported one, so prompt token counts agree more often than prompt ids do.

`stream` returns native SDK response chunks. `chat(messages, max_new_tokens, stream=..., **parameters)` preserves SDK tools, tool-result messages, structured-output controls and media fields. Inject an `AsyncClient`/`AsyncOpenAI` and use `acall`, `astream` or `achat` for asynchronous execution. Ollama requests go through the SDK's public `generate` and `chat` methods, which own request conversion, HTTP behavior, error handling and line-stream framing; the adapter rejects request fields the installed SDK does not accept and negative token counts, and the SDK's own parsing rejects unparsable values. OpenAI completions use the SDK's public `with_raw_response` hook, so choice and usage fields are checked on the wire before parsing. OpenAI request parameters go to its completion/chat resources. vLLM-only parameters belong explicitly in `extra_body`. The task's model, prompt, token budget, requested choice count and an explicit `Sampling` policy cannot be overridden through provider extensions; the SDK writes `extra_body` over the named parameters, so a policy field there must equal the policy or the request is refused before any network call.

`export_run(run_dir, destination, *, ema=True, step=None)` writes a saved run into that same layout: it loads the run the way `dew.pipeline` does and hands the rebuilt model to its family's writer, refusing a model with no published layout by name. `dew export <run> <dest>` is the command over it, and `push_to_hub` exports a run directory before uploading unless `raw=True`, which uploads the run itself for `from_pretrained` to pull back.

A decoder trained through the LM recipe, exported with `save_pretrained_decoder` and converted by `ollama create` answers a greedy request with Dew's own greedy continuation, token for token, over the live daemon. `Pretrained.save` and `save_pretrained_decoder` leave the same files, so either export converts.

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

The same `dew.interop.load_pretrained` entry reads a diffusion checkpoint directory with `model_index.json`, component configurations, safetensors and tokenizer files. Four denoisers load: a `UNet2DCondition` reading one or two CLIP towers through cross attention, an `SD3Transformer` reading both jointly beside a T5 tower, a `FluxTransformer` reading a T5 sequence with a pooled CLIP vector, and Qwen-Image 2.1's `QwenImageTransformer` reading its Qwen3-VL encoder's prompt states, the last three on flow-matching schedules. The denoiser component the directory holds selects the family, and the class it declares selects the model. The returned `Pretrained` holds the native model, its autoencoder behind the existing autoencoder seam (an `AutoencoderKL`, or Qwen-Image 2.1's one-frame `QwenImageVAE`), native text conditioning, a `Process`, the native solver policy and the published pipeline's own call policy. Model and scheduler implementations from other libraries run only in the reference tools.

`source.text_to_image()` builds the native image task. `source.save(directory, variables=updated)` writes the updated component weights back to their published layouts, retaining tokenizer files, image geometry and any safety-head parameters. Flax-declared components retain their source class and receive Flax msgpack files alongside the safetensors used by Dew; export does not relabel them as PyTorch models.

Source UNet configuration selects normalization groups and epsilon, and the declared implementation selects GEGLU semantics: exact GELU for PyTorch sources, approximate GELU for Flax sources. Spatial upsampling targets the actual next skip shape, including odd intermediate dimensions. A published MM-DiT reads its own modulation order, joint attention, optional query-key normalization and second self-attention, and the stored sin/cos position buffer, which lands in a `buffers` collection so an optimizer and an EMA see only `params` and export writes the stored array back unchanged; the buffer is cropped centred on the latent's patch grid. Flux's transformer carries the other stack: double-stream blocks whose image and text residuals join only inside attention, then single-stream blocks over the concatenation whose attention and feed-forward share one output projection, with queries and keys rotated by a three-axis interleaved-real rotary table over the text and image ids. Its pipeline's 2x2 latent packing is inside the model, so a caller works in latents, and a guidance-embedded checkpoint reads the distilled guidance value as a model input rather than as two guided branches.

`DiffusionConditioner` owns every family's text composition: one CLIP tower's last hidden states; two towers' penultimate states concatenated with the second tower's projected pooled vector and the size and crop time ids; those states padded out to the T5 width with the T5 states following them along the sequence and both projections as the pooled vector; or the T5 states alone with one tower's unprojected pooled vector, which is Flux's. Each text slot goes to the tower whose source pipeline reads it, and an SD3 source that declares no third encoder gets the zero segment its own pipeline writes, at the CLIP tokenizer's window rather than at the sequence a call asks for; Flux has no such path, so a Flux composition without its T5 tower is refused.

Qwen-Image 2.1 has its own conditioner, `QwenImageConditioner`. It runs the Qwen3-VL encoder's language model over the pipeline's text-to-image chat template, reads the last layer's output before the final norm, and drops the system turn. Rows are padded on the right to a token budget (`tokens`, 512 by default), and `DenoisingCondition.mask` marks the real tokens so the transformer excludes padded keys. The transformer lays its sequence out image first, so each of its two attention calls ends a row's keys at a length (`key_value_seq_lengths` on `scaled_dot_product_attention`), which cuDNN applies as its padding mask and skips the padded keys by. A prompt past the budget is refused. The encoder's vision tower is not run for text-to-image. Its tensors are kept as stored so an export writes the whole encoder back. The transformer keeps the source's block-causal attention (the text attends causally, the image attends to all the text and to itself) and modulates the text from time zero. Each row starts its image's rotary frame position after its own text. The source pipeline starts it after the longest prompt in the call, so a padded row here matches the source run on that prompt alone. The 2.1 VAE is an image-only Wan-style autoencoder with four channels (RGBA) and 64 latent channels, normalized per channel with the config's `latents_mean` and `latents_std`. Image editing is not supported: it needs the Qwen3-VL vision tower with its deepstack mergers, which Dew does not implement.

`SourceTask` carries the published pipeline's own call policy: the steps and guidance its `__call__` defaults to, and the grid it prepares with the sigma origin and latent geometry that pipeline uses. That way `text_to_image()` runs what the source runs. A directory that declares no pipeline takes its family's reference one; a declared pipeline Dew does not implement, or one that drives a different denoiser than the directory holds, is refused rather than run under another pipeline's defaults.

`SourceSchedule.from_config` reconstructs eighteen published scheduler classes: DDIM, PNDM, DDPM, LMS, Euler, Euler ancestral, Heun, KDPM2, KDPM2 ancestral, DPM-Solver multistep, singlestep and SDE, DEIS, UniPC, EDM DPM-Solver, LCM, TCD and flow-matching Euler. The returned process, solver and grid use the controls and defaults of the declared class, and the solver is resolved once when the file is read.

Supported policies include source timestep spacing and offsets; Karras, exponential and beta sigma grids; finite lambda clipping; zero-terminal-SNR rescaling; terminal sigma selection; solver orders and correctors; DDPM fixed posterior variances; and clipping or dynamic thresholding in the source conversion order. DDIM and PNDM retain fixed training strides. LCM and TCD select their grids from `original_inference_steps`. Two-evaluation solvers retain the source stage sigma and model time. EDM uses its own sigma range, data scale, rho, log-sigma model time and signed output preconditioning; its training process uses EDM sigma draws rather than a VP beta table. A flow-matching file carries its static or resolution-dependent shift, its terminal stretch and both sigma origins its pipelines use; the latent token count a resolution-dependent shift reads is the calling pipeline's, bound through the task's grid, which `text_to_image()` binds to the checkpoint's own geometry.

Unimplemented active controls fail explicitly: Lu-lambda and flow-sigma grids, UniPC external predictors, nonlinear sigma interpolation, continuous model times, learned variances, DDPM log-space wide variance, and PNDM velocity-domain history. TCD clipping/thresholding and epsilon-prediction UniPC thresholding are also refused because the published update does not apply them. Sigma-indexed source schedulers whose initial model time repeats are refused when preparing their grid: the source starts at the second match and cannot complete its published evaluation list. Repeated noninitial stage and corrector times remain supported.

The tiny oracles in `tools/diffusers_source_reference.py` run actual Diffusers scheduler objects. Ordinary stochastic walks share explicit Gaussian draws. DPM-Solver SDE runs the actual `torchsde` tree, and native solver parity uses its recorded increments; separate tests exercise the native Brownian bridge law. All native source-trajectory and VJP checks run in float32 at a fixed `1e-4` scaled-error bound, with VJPs compared directly to the actual float32 source. Saved float64 VJPs are diagnostic data, not a tolerance adjustment.

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

`attention_impl` is a model field and names the attention kernel: `'reference'` (also spelled `None`) is the einsum and softmax, the only path that reads `dtype`, `precision` and `force_fp32_for_softmax`; `'xla'` and `'cudnn'` are `jax.nn.dot_product_attention`'s two; `'tpu'` is the Pallas splash kernel; `'auto'` resolves per trace, so a configuration logged as `'auto'` runs on the next machine. `'auto'` takes `'cudnn'` where its kernel runs (a GPU backend, bf16 or fp16, a query head width that is a multiple of 8 and at most 128, no softcap, no sinks, and no `--xla_gpu_deterministic_ops`), then `'tpu'` where splash's runs (a TPU backend, bf16 or fp32, query and key lengths that are multiples of 128 and at least 512, no additive bias, no mesh splitting the sequence, and a mask splash can describe), and `'xla'` otherwise. Splash's mask is a block-sparse descriptor built while the executable is, so a causal or sliding-window sequence skips the blocks it empties instead of paying its rectangle; the head width is unconstrained, because the kernel pads it, while the query and key lengths have to be multiples of 128 for its mask blocking to tile them. An additive bias, a mask that is a value of the trace (a KV-cache decode mask, the striped mask sequence parallelism builds), a mask past the published cell budget, and a length that is not a multiple of 128 keep the older Pallas flash kernel under an explicit `'tpu'` and keep `'auto'` on `'xla'`. Splash applies a logit softcap (Gemma 2) and attention sinks (GPT-OSS) itself, and a packed batch reaches it as segment ids rather than a mask; the flash kernel has neither a softcap nor sinks, so an explicit `'tpu'` call that splash cannot describe refuses them, and on a GPU `'auto'` runs both on `'xla'` while `'cudnn'` refuses them by name. Below 512 keys `'auto'` stays on `'xla'`, where XLA's attention measured faster on a v6e (`SPLASH_MIN_LENGTH` in `dew/nn/attention.py`). The parameter tree never changes with the implementation, so checkpoints are interchangeable across hardware.

The [README model list](https://github.com/AshishKumar4/dew/blob/main/README.md#models) names which model configurations run the whole workflow; each task guide covers its own data and objective.
