# Post-training in Dew: SFT, DPO and online RL on one trainer

Design, rewritten 2026-09-03 against `api.md`. Nothing here is built. Every citation is a symbol in the tree this document sits in; a line number appears only where I read that line for this pass. The earlier version of this design was written against the trainer the API design replaced, so its mechanisms are restated here rather than renamed.

A modality is an objective, so post-training is more objectives, and the user learns one thing:

```
LMObjective(model, seq_len)                          # pretraining, as today
LMObjective(model, seq_len, loss_role=ASSISTANT)     # SFT: the same class, a loss mask, a chat data path
DPOObjective(model, seq_len, beta=0.1)
GRPOObjective(model, seq_len, beta=0.0)
FlowGRPOObjective(model, process, inputs, beta=0.0)
Trainer(objective, optimizer, ..., rollout=VLLMRollout(...))
```

They run on the `Trainer` as it is (`dew.training.trainer`), with the same state, sharding, EMA clock, checkpoints and tracker keys. Three mechanisms carry the design: the reference is the EMA tree at unit decay (§3), sampling is one more trainer capability (§4), and rewards are plain callables (§5). The surrogate math is already built and parity-tested in `dew.rl`, so the online objectives are assembly rather than derivation (§6).

## 1. Data

The three new paths are `DatasetSpec`s in the `datasets` registry, beside `TokenWindows` and `PackedTokens` (`dew/data/tokens.py:56, :132`). Each returns a `Dataset` from `load(batch=)`, so a recipe reaches them through the same `data:` subcommand every other run uses, and none of them changes the trainer.

### 1.1 Chat and SFT: a role per token

SFT needs one field beyond the pretraining contract: which role wrote each token, because the loss must count assistant tokens only.

`ChatMessages`, registered as `chat_messages`: a Grain source over a parquet file of verl-shaped rows (§1.4). One record is one conversation, a list of `{role, content}` messages, plus the columns a reward wants carried through. A transform renders the conversation and emits, per token, `text`, `text_roles` (int8: 0 pad, 1 system, 2 user, 3 assistant, 4 tool), `text_segment_ids` and `text_positions`.

The assistant span comes from prefix rendering, not from template markers:

- For message *k*, render `messages[:k]` with `add_generation_prompt=True` to ids `p_k`, and `messages[:k+1]` without it to `f_k`.
- Assert `f_k[:len(p_k)] == p_k`. A template whose rendering is not incremental in its prefix is refused by name rather than silently mis-masked.
- The assistant tokens are `f_k[len(p_k):]`, including the end-of-turn token. Training the stop token is the point.

TRL's `assistant_only_loss` needs a template carrying `{% generation %}` markers and swaps in a bundled template when the user's lacks them, which changes the rendered ids and not just the mask. verl tokenizes each message alone and concatenates, which is only equal to whole-conversation tokenization for templates that happen to be concatenation-safe. The prefix method is exact for any template and asserts it. TRL's mask is still the parity reference for the transform (§8).

Packing is unchanged machinery: `text_roles` joins as one more per-token feature in `FirstFitPackIterDataset` with its own `length_struct` and `padding_struct` entries (`dew/data/tokens.py:164, :194`), and `DocumentChunks` already cuts a document longer than the window (`:93`), which is what the packer requires. Segment ids still stop attention at a document boundary.

| key | shape | dtype | content |
| --- | --- | --- | --- |
| `text` | `[B, S+1]` | int32 | packed conversation ids |
| `text_segment_ids` | `[B, S+1]` | int32 | which document, 0 on pad |
| `text_positions` | `[B, S+1]` | int32 | position inside that document |
| `text_roles` | `[B, S+1]` | int8 | role per token, 0 on pad |

`loss_role` multiplies the objective's existing target weights by `(text_roles[:, 1:] == loss_role)`, composed with the pad and segment-boundary weights it already computes (`dew/objectives/lm/objective.py:198-205`). With no `loss_role`, every counted target counts, exactly as pretraining does.

### 1.2 Preference pairs

`PreferencePairs`, registered as `preference_pairs`: a parquet of `prompt` with `chosen` and `rejected`, each either text or a conversation. The transform renders the prompt with `add_generation_prompt=True`, appends each completion, left-pads both to a common length, and marks the completion.

| key | shape | dtype | content |
| --- | --- | --- | --- |
| `text` | `[2B, S]` | int32 | chosen at `[0::2]`, rejected at `[1::2]`, adjacent per pair |
| `text_segment_ids` | `[2B, S]` | int32 | 1 real, 0 pad |
| `text_positions` | `[2B, S]` | int32 | 0-based within the real tokens |
| `completion_mask` | `[2B, S]` | int8 | 1 on completion tokens, including the stop token |

Adjacent interleaving is verl-omni's layout and is chosen over TRL's concatenate-then-chunk so a pair shares a prompt without a reordering step.

For images, `DiffusionDPOObjective` reads a pair-of-images record, encodes both through the autoencoder, and applies one shared timestep and one shared noise per pair. The online variant needs no pair dataset: the rollout samples G images per prompt and the reward's best and worst become the pair.

### 1.3 Prompts for online RL

`Prompts`, registered as `prompts`: parquet of `prompt` plus whatever columns the reward reads. No masks, because the rollout produces them.

In: `prompt` `[b, P]` int32, left-padded, and `prompt_length` `[b]` int32. Out of the rollout, and what `loss` consumes:

| key | shape | dtype | content |
| --- | --- | --- | --- |
| `text` | `[b*G, P+N]` | int32 | prompt and continuation |
| `text_segment_ids` | `[b*G, P+N]` | int32 | 1 real, 0 pad |
| `text_positions` | `[b*G, P+N]` | int32 | 0-based within real tokens |
| `response_mask` | `[b*G, N]` | int8 | 1 on generated tokens through the first stop token |
| `old_log_probs` | `[b*G, N]` | float32 | from the params that sampled |
| `advantages` | `[b*G]` | float32 | group-relative |
| `reward` | `[b*G]` | float32 | raw scores, for telemetry |
| `truncated` | `[b*G]` | bool | hit N without a stop token |

Shapes are constants of the run, because `generate` runs for exactly `max_new_tokens` whatever it sees (`dew/sampling/text.py:57-65`). One shape per run is what keeps `Trainer.compile` tracing once (`dew/training/trainer.py:277`).

For diffusion the tables change only in what prompt and response mean: the prompt is the conditioning the `InputSpec` already describes, and the response is a latent trajectory (§7).

### 1.4 verl's parquet schema, mapped

| verl field | verl content | Dew |
| --- | --- | --- |
| `data_source` | dataset name, indexes the reward | `data_source`, passed to the reward (§5) |
| `prompt` | chat messages | rendered by the chat transform into `text` and `text_roles` |
| `reward_model` | `{"style": "rule", "ground_truth": str}` | `ground_truth`, passed to the reward |
| `extra_info` | bookkeeping | `extra_info`, passed to the reward |
| `ability` | task category | carried in `extra_info`; Dew dispatches on nothing |

verl's `compute_score(data_source, solution_str, ground_truth, extra_info)` is the same four things Dew's callable takes as fields of one record.

## 2. Why `loss_role` and not `mask=`

A loss mask is not a property of the model, and the data has to know roles anyway.

| option | cost | gives up |
| --- | --- | --- |
| `mask="assistant"` | a stringly-typed knob with a hidden coupling to the batch | nothing, but every caller learns which strings exist |
| `loss_role=ASSISTANT` | one typed field compared against a column the batch already carries | an arbitrary ad-hoc boolean mask |

Decision: `loss_role`. It is the same class, one field, one multiply. The objective refuses a batch with no `text_roles` when `loss_role` is set, naming the field.

## 3. The frozen reference

**Decision: the reference is the EMA tree at unit decay.** A preference or RL objective sets `ema = EMASpec(decay=optax.constant_schedule(1.0))` and reads `step.ema` in `loss`. `Step.ema` is the variables tree with the averaged leaves in place of the live ones (`dew/objectives/base.py:41-48`), so the reference forward runs through the same code as the policy forward with a different tree.

What falls out without new machinery:

- **Out of the optimizer.** Only the `params` collection is differentiated and only it reaches `tx.init`, so no masked optimizer and no zero-gradient tree.
- **Checkpointed and sharded.** `ema` is a field of `TrainState` (`dew/training/state.py:24-28`), so `Trainer.shardings` shards it like params and a resumed run restores its reference with everything else.
- **The clock is already right.** The EMA runs on completed optimizer updates, not micro-steps, and a rejected mixed-precision step is not an update (`dew/training/trainer.py:246-261`). A reference cannot drift under accumulation.
- **`select` scopes it.** `EMASpec.select` is a `PathFilter` (`dew/objectives/base.py:100-108`), so a reference over part of the tree, a frozen encoder beside a trained head, costs one filter.

One change is required, and it is small. `ema_update` is `decay * average + (1 - decay) * live` (`dew/training/trainer.py:70-73`). At decay 1.0 that is arithmetically the average, but `0.0 * NaN` is NaN, so a single non-finite parameter poisons a frozen reference on the step it appears, and the `finite` gate only exists when `dynamic_scale` is on. The fix is one select per leaf: return the average unchanged where `decay >= 1.0`. The test that ships with it forces a non-finite parameter and asserts the reference is bit-identical afterwards, with and without `dynamic_scale`.

The cost of the decision is that a run cannot hold a moving policy EMA and a frozen reference at the same time. No objective in scope wants both. The alternative, a masked subtree inside `params`, was rejected: one extra full copy in HBM, an optimizer wrapper, gradient and update trees for something that never moves, and a second checkpoint layout. The EMA slot is already allocated, already sharded and already checkpointed.

**In-step, not precomputed.** `loss` runs the reference forward per batch. TRL precomputes reference log-probs to free a second model's memory; under this mechanism there is no second model to free, so precomputing buys one forward per step, roughly a third of forward plus backward, at the price of a cache that silently goes stale when the data path changes. Revisited only if a measured run shows reference forwards dominating.

## 4. The rollout capability

Sampling is effectful, host-side and sometimes remote, so it is a capability the trainer is given, beside the checkpointer, the tracker and the profiler, all of which are `X | None = None` (`dew/training/trainer.py:105-126`):

```python
class Rollout(Protocol):
    def __call__(self, state: TrainState, batch: Batch, key: jax.Array) -> Batch: ...
```

`Trainer(..., rollout=None)` is exactly today's loop. When one is given, the trainer calls it between `batch = next(train)` and the compiled step (`dew/training/trainer.py:362, :373`) and reshards the result with `shard_batch(mesh, ...)`.

**Why not the `step=` seam.** `Trainer.step` replaces the compiled step's body and is documented as the one place for an update that is not one loss (`dew/training/trainer.py:117-126`). It runs inside `jit` and owns the counter, the EMA and the write-back. A rollout is the opposite kind of thing: it produces the batch the step then consumes, it may post to a vLLM server, and it cannot be traced. Putting it in the step body would either force generation inside `jit` or smuggle a host callback into a compiled function. Two seams, two kinds of work, and the design says which is which.

**Why not a method on `Objective`.** The objective's surface is pure and crosses `jit`: `loss` and `evaluate` are functions of variables, a batch and a `Step`. An objective that opens a socket is not that, and every objective would carry an identity method it never uses.

The rest falls out:

- **Fixed shapes**, so `compile` traces once and no shape polymorphism is needed.
- **Keys.** The rollout key is folded from `state.key` and `state.step`, the same stream the step key comes from, with one extra fold so a rollout and its step never share draws. Both are checkpointed, so a resumed run samples forward rather than replaying.
- **Accumulation.** One rollout per micro-batch. A group never straddles a micro-batch, so the group baseline is computed inside the batch it belongs to, which is the whole batch at the default `accumulation=1`.
- **Distributed.** The rollout runs per process on that process's slice, which is what `DevicePrefetchIterator` hands it, and `shard_batch` reassembles the global array. Groups are process-local by construction.
- **Telemetry through the existing channels.** Reward mean and generation length ride `Aux.metrics` out of `loss`, which the trainer logs under `train/`. Wall time cannot come from inside `jit`, so the trainer times the host-side call and folds `train/rollout_seconds` into the same log tick that carries throughput (`dew/training/trainer.py:395-404`). There is no second return channel.

## 5. Rewards

A reward is a callable from one finished record to a float: `reward(data_source, completion, ground_truth, extra_info) -> float`, the verl signature with the four fields the batch already carries (§1.4). It runs host-side inside the rollout, so no Python callable and no external process is ever reachable from a compiled step. A rule-based verifier, a hosted judge and a scoring model are all the same shape, and a run composes several by summing weighted scores in its own function.

## 6. The language objectives

They are assembly over `dew.rl`, whose estimators and surrogates are built, ported from Tunix and verl, and pinned against their fixtures (`dew/rl/advantage.py`, `dew/rl/surrogate.py`, `tests/fixtures/rl/*.npz`).

- **SFT** is `LMObjective` with `loss_role` (§2).
- **DPO** takes per-sequence log-probabilities as the negated per-token cross entropies the chunked head already returns (`dew/objectives/lm/chunked.py:83-89`), summed under `completion_mask`, for the policy and for `step.ema`. The loss is `-logsigmoid(beta * ((pi_c - ref_c) - (pi_r - ref_r)))` over the pair rows.
- **GRPO** reads `old_log_probs`, `advantages` and `response_mask` from the rolled-out batch and is one composition: `clipped_surrogate(token_log_ratio(...), advantages, response_mask)` plus `beta * token_mean(k3_kl(...), response_mask)` (`dew/rl/surrogate.py:110, :74, :156, :60`). The advantages come from `group_advantage` or `rloo_advantage` inside the rollout (`dew/rl/advantage.py:99, :120`), where the rewards are.

Nothing in that list derives new math, which is the point of having landed `dew.rl` first.

## 7. Diffusion RL

Two additions, both small because the sampling seam was rebuilt for this shape:

**A trajectory beside the sample.** `sample(denoise, x_T, steps, *, solver, guidance, key)` returns the final sample from one scan (`dew/sampling/sample.py:8-14`). RL needs the path: the recorded `(x_t, x_next, t, t_next)` per step. `dew.sampling.trajectory` shares that scan body and returns the stacked path, so a normal sample never pays to materialise it and a rollout never re-derives the walk. Delete `trajectory` and only RL loses its data source.

**An SDE solver, as an ordinary solver.** The `Solver` protocol hands `step` the key, the process, the model callable and both of the model's predictions (`dew/sampling/solvers.py:26-32`). The earlier design needed a change to the sampler seam for this; the current protocol already passes everything an SDE transition needs, so an `SDE` solver is a new file and no existing file changes. Its `State` carries the log-probability of each transition, which is what the loss recomputes under the policy.

`FlowGRPOObjective` then recomputes the model at each recorded `(x_t, t)`, rebuilds the transition mean, takes its log-probability, and applies the same `clipped_surrogate` with a per-step mask of ones. The reward is an image scorer, host-side, in the rollout.

## 8. Parity plan

| check | reference | recorded |
| --- | --- | --- |
| chat rendering and the assistant mask | TRL's mask for the same template and conversation | ids equal, mask equal |
| DPO loss and gradient | TRL on fixed tensors | largest difference |
| GRPO surrogate and KL | the committed `dew.rl` fixtures | already pinned |
| group and RLOO advantages | verl on the same rewards | already pinned |
| SDE transition and its log-probability | verl-omni's flow-GRPO step | largest difference |
| a rolled-out run's shapes | the tables in §1.3 | one trace of `compile` per run |

Each new test records the largest observed difference and tightens its tolerance to it. Each objective gets one mutation per branch that must fail parity.

## 9. What fits

Per parameter, fp32 params plus Adam's two moments plus a frozen reference is 16 bytes, before activations. For Qwen3-0.6B (`tests/fixtures/hf/qwen3-0.6b/config.json`: hidden 1024, 28 layers, 16 heads, 8 KV heads, ffn 3072, vocab 151936), that is about 9.5 GiB of state, which fits a 16 GiB card with a small batch and bf16 activations, and leaves the rollout's KV cache as the next thing to size. For a 7B the same arithmetic is about 112 GiB, so the first configuration that fits is FSDP across two 80 GiB devices, and the reference is not what makes it not fit: the fp32 optimizer state is.

## 10. Crossing over to verl and vLLM

Research scale uses Dew's own `generate` and its samplers. Beyond that, the crossover is at the capability boundary and nowhere else: a `Rollout` that posts prompts to a vLLM server and returns token ids is one implementation of the protocol in §4, and the weights it serves come from `save_pretrained_decoder`, which writes the HF layout verl and vLLM already consume (`dew/interop/hf_decoders`). Nothing about either reaches the trainer, the objective or the state.

## 11. Waves

| # | wave | acceptance |
| --- | --- | --- |
| 1 | chat data path and `loss_role` | mask parity against TRL; a packed SFT batch carries four aligned per-token fields; loss counts assistant targets only |
| 2 | unit-decay reference | a non-finite parameter leaves the reference bit-identical, with and without `dynamic_scale`; a resumed run restores it |
| 3 | the rollout capability | identity default leaves the loop unchanged; one `compile` per run with a rollout; `train/rollout_seconds` in the log tick; a resumed run does not replay |
| 4 | DPO | loss and gradient parity against TRL; the pair layout asserted |
| 5 | GRPO on `dew.rl` | the composition matches verl end to end on one fixed rollout; a mutation of each term fails |
| 6 | trajectory, SDE solver, FlowGRPO | transition parity against verl-omni; no existing sampler file changed |
