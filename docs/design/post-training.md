# Post-training in Dew: SFT, DPO and online RL on one trainer

> An AI assistant maintains this document. It is presented as-is.

Built through wave 6. The current Flow-GRPO usage is in [the post-training guide](../concepts/post_training.md#flow-grpo). The earlier sections record the language post-training implementation as it was when those waves landed.

Correction (2026-09-22): I checked this record against current source. The sketch below still matches the constructors (`src/dew/objectives/lm/objective.py:431-450`, `src/dew/objectives/rl/preference.py:39`, `src/dew/objectives/rl/grpo.py:58`, `src/dew/training/trainer.py:188-203`). Where a section no longer matches, a dated note at the end of that section says what changed.

A modality is an objective, so post-training adds objectives, and the user learns one pattern:

```
LMObjective(model, seq_len)                                  # pretraining, as today
LMObjective(model, seq_len, loss_role=Role.ASSISTANT)        # SFT: the same class, a loss mask, a chat data path
DPOObjective(model, seq_len, beta=0.1)
GRPOObjective(model, seq_len, beta=0.0)
Trainer(objective, optimizer, ..., rollout=SampledRollout(...))
```

They run on the `Trainer` as it is (`dew.training.trainer`), with the same state, sharding, EMA clock, checkpoints and tracker keys. The design rests on three mechanisms. The reference is the EMA tree at unit decay (§3), sampling is one more trainer capability (§4), and rewards are plain callables (§5). `dew.rl` already has the surrogate math, built and parity-tested, so the online objectives only assemble it (§6). Staged runs chain these links in `recipes/chain.py` (§12).

## 1. Data

The three paths are `DatasetSpec`s in the `datasets` registry, beside `TokenWindows` and `PackedTokens`: `ChatMessages` (`dew/data/chat.py`), `PreferencePairs` (`dew/data/preferences.py`) and `Prompts` (`dew/data/prompts.py`). Each returns a `Dataset` from `load(batch=)`, so a recipe reaches them through the same `data:` subcommand every other run uses, and none of them changes the trainer.

### 1.1 Chat and SFT: a role per token

SFT needs one field beyond the pretraining contract: which role wrote each token. The loss must count assistant tokens only.

`ChatMessages`, registered as `chat_messages`, is a Grain source over a parquet file whose `prompt` column holds verl-shaped conversations, plus a tokenizer whose chat template renders them. `ConversationSource` reads the conversations. `RenderConversation` renders each one and emits, per token, `text` and `text_roles` (int8: 0 pad, 1 system, 2 user, 3 assistant, 4 tool). The packer carries `text_roles` beside `text_segment_ids` and `text_positions`. A `_lengths` pass renders every row up front, so a bad row fails the run with its index before packing starts.

The assistant span comes from prefix rendering, not from template markers:

- For message *k*, render `messages[:k]` with `add_generation_prompt=True` to ids `p_k`, and `messages[:k+1]` without it to `f_k`.
- Assert `f_k[:len(p_k)] == p_k`. Dew refuses a template whose rendering is not incremental in its prefix, and names it, instead of masking it wrong.
- The assistant tokens are `f_k[len(p_k):]`, including the end-of-turn token, because the model has to learn to emit the stop token.

TRL's `assistant_only_loss` needs a template carrying `{% generation %}` markers. When the user's template lacks them, TRL swaps in a bundled template, which changes the rendered ids as well as the mask. verl tokenizes each message alone and concatenates the results, which equals whole-conversation tokenization only for templates that happen to be concatenation-safe. The prefix method is exact for any template and asserts it. TRL's mask is still the parity reference for the transform (§8).

Packing reuses the existing machinery. `text_roles` joins as one more per-token feature in `FirstFitPackIterDataset` with its own `length_struct` and `padding_struct` entries. `DocumentChunks` cuts every per-token field of a document longer than the window, so ids and roles stay aligned. A conversation that starts with the assistant is refused, because the template gives no way to separate its opening header from its completion. Segment ids still stop attention at a document boundary.

| key | shape | dtype | content |
| --- | --- | --- | --- |
| `text` | `[B, S+1]` | int32 | packed conversation ids |
| `text_segment_ids` | `[B, S+1]` | int32 | which document, 0 on pad |
| `text_positions` | `[B, S+1]` | int32 | position inside that document |
| `text_roles` | `[B, S+1]` | int8 | role per token, 0 on pad |

`loss_role` multiplies the objective's existing target weights by `(text_roles[:, 1:] == loss_role)`, together with the pad and segment-boundary weights it already computes. With no `loss_role`, every counted target counts, as in pretraining.

Correction (2026-09-22): packing now goes through `PackedWindows`, one plan over the whole corpus (`src/dew/data/tokens.py:250`); `FirstFitPackIterDataset` is gone from `src/`. A window also carries `text_roles_segment_ids` and `text_roles_positions`, identical to the text ones (`src/dew/data/chat.py:9-12`). `ChatMessages` reads a parquet file, a `.jsonl` file or a Hub dataset id; the conversation column defaults to `messages` and falls back to `prompt` (`src/dew/data/chat.py:672-704`). `Role` has a sixth value, `DEVELOPER = 5` (`src/dew/data/chat.py:65-80`).

### 1.2 Preference pairs

`PreferencePairs`, registered as `preference_pairs`, reads a parquet file or JSON rows of `chosen` and `rejected` token-id lists, with `chosen_mask` and `rejected_mask` marking the completion tokens. Absent masks default to all-completion. Every element is one `[2, seq_len]` pair, chosen at index 0, padded with `pad_id` and 0, so a grain batch is `[B, 2, seq_len]` and shuffling never separates a pair. A row longer than `seq_len` fails with its lengths. A pair cannot be chunked without cutting a completion, and a silent cut would train the wrong preference.

| key | shape | dtype | content |
| --- | --- | --- | --- |
| `input_ids` | `[B, 2, S]` | int32 | pairs, chosen at index 0 |
| `completion_mask` | `[B, 2, S]` | int32 | 1 on completion tokens, including the stop token |

The objective reads the pair index out explicitly. A reshape would interleave two pairs into the halves and compare across them.

### 1.3 Prompts for online RL

`Prompts`, registered as `prompts`, reads a parquet file or JSON rows of `prompt` (messages, a string, or token ids) plus the reward columns `data_source`, `ground_truth` and `extra_info`, which default to empty when absent. Each row encodes to a left-padded `prompt` of `max_prompt_len` ids plus `prompt_length`. The reward columns become fixed-width UTF-8 byte arrays so every leaf survives the device transfer. Prompts longer than the window keep their tail; blank prompts fail. The source makes no masks, because the rollout produces them.

In: `prompt` `[B, P]` int32, left-padded, `prompt_length` `[B]` int32, and the three reward columns as UTF-8 bytes. Out of the rollout, and what `loss` consumes:

| key | shape | dtype | content |
| --- | --- | --- | --- |
| `input_ids` | `[N, P+T]` | int32 | prompt and continuation, groups contiguous per prompt |
| `response_mask` | `[N, T]` | float32 | 1 through the first stop token |
| `old_log_probs` | `[N, T]` | float32 | raw policy likelihoods recorded during generation |
| `behavior_log_probs` | `[N, T]` | float32 | actual temperature/top-k sampling likelihoods |
| `response_length` | `[N]` | int32 | generated actions, including EOS |
| `terminated` | `[N]` | bool | EOS termination instead of the token budget |
| `advantages` | `[N, T]` | float32 | group or RLOO advantages broadcast over the width |
| `rewards` | `[N]` | float32 | raw scores, for telemetry |
| `prompt_length` | `[N]` | int32 | real tokens before padding |

The rollout returns fixed-width arrays even when EOS ends a completion early. Response masks and lengths identify valid actions. `Prompts.prompt_length` stays data metadata; the adapter turns it into `ModelInputs.token_fields["attention_mask"]` before generation. One compiled padded prefill and decode scan serves different row lengths through native cache cursors and validity. The rollout reads each process's owned rows from the globally sharded batch and returns those rows. All ranks validate before collective execution, agree on input shapes and sampling controls, and run the same fixed scan. Different prompt lengths or EOS outcomes do not change the collective schedule.

Correction (2026-09-22): `SampledRollout` returns `input_ids`, `response_mask`, `old_log_probs`, `behavior_log_probs`, `advantages`, `rewards` and `prompt_length`. It returns no `response_length` or `terminated` key (`src/dew/objectives/rl/rollout.py:139-147`).

Correction (2026-09-22, packed layout): the table above is superseded. `SampledRollout` builds its batch with `sessions.pack`, one call per completion: every column is `[N, P+T]` and aligned with `input_ids`, with `text_segment_ids`, `text_positions`, `versions`, `session_index` and `call_index` beside the mask, likelihoods and advantages. No `rewards` or `prompt_length` column remains; GRPO reads only this layout.

### 1.4 verl's parquet schema, mapped

| verl field | verl content | Dew |
| --- | --- | --- |
| `data_source` | dataset name, indexes the reward | `data_source`, passed to the reward (§5) |
| `prompt` | chat messages | rendered by the chat transform into `text` and `text_roles`, or encoded by the prompt source |
| `reward_model` | `{"style": "rule", "ground_truth": str}` | `ground_truth`, passed to the reward |
| `extra_info` | bookkeeping | `extra_info`, passed to the reward |
| `ability` | task category | carried in `extra_info`; Dew dispatches on nothing |

verl's `compute_score(data_source, solution_str, ground_truth, extra_info)` takes the same four things that Dew's callable takes as fields of one record.

## 2. Why `loss_role` and not `mask=`

A loss mask is not a property of the model, and the data has to know roles anyway.

| option | cost | gives up |
| --- | --- | --- |
| `mask="assistant"` | a stringly-typed knob with a hidden coupling to the batch | nothing, but every caller learns which strings exist |
| `loss_role=ASSISTANT` | one typed field compared against a column the batch already carries | an arbitrary ad-hoc boolean mask |

Decision: `loss_role`. It keeps the same class and adds one field and one multiply. When `loss_role` is set, the objective refuses a batch with no `text_roles` and names the field.

## 3. The frozen reference

Decision: the reference is the EMA tree at unit decay. A preference or RL objective sets `ema = EMASpec(decay=optax.constant_schedule(1.0))` and reads `step.ema` in `loss`. `Step.ema` is the variables tree with the averaged leaves in place of the live ones (`dew/objectives/base.py`), so the reference forward runs through the same code as the policy forward with a different tree.

Several properties come with this choice at no extra cost:

- The reference stays out of the optimizer. Only the `params` collection is differentiated and only it reaches `tx.init`, so there is no masked optimizer and no zero-gradient tree.
- It is checkpointed and sharded. `ema` is a field of `TrainState`, so `Trainer.shardings` shards it like params and a resumed run restores its reference with everything else.
- Its clock is already right. The EMA runs on completed optimizer updates, not micro-steps, and a rejected mixed-precision step is not an update. A reference cannot drift under accumulation.
- `select` scopes it. `EMASpec.select` is a `PathFilter`, so a reference over part of the tree, such as a frozen encoder beside a trained head, costs one filter.

The design needed one small change. `ema_update` is `decay * average + (1 - decay) * live`. At decay 1.0 that is arithmetically the average, but `0.0 * NaN` is NaN, so a single non-finite parameter poisons a frozen reference on the step it appears. The `finite` gate only exists when `dynamic_scale` is on. The fix is one select per leaf: return the average unchanged where `decay >= 1.0` (`dew/training/trainer.py:134`). Its test forces a non-finite parameter and asserts the reference is bit-identical afterwards, with and without `dynamic_scale`.

The cost of the decision is that a run cannot hold a moving policy EMA and a frozen reference at the same time. No objective in scope wants both. We rejected the alternative, a masked subtree inside `params`, because it costs one extra full copy in HBM, an optimizer wrapper, gradient and update trees for something that never moves, and a second checkpoint layout. The EMA slot is already allocated, sharded and checkpointed.

The reference runs in the step; nothing is precomputed. `loss` runs the reference forward per batch. TRL precomputes reference log-probs to free a second model's memory. Here there is no second model to free, so precomputing would save one forward per step, roughly a third of forward plus backward, at the price of a cache that silently goes stale when the data path changes. We revisit this only if a measured run shows reference forwards dominating.

Correction (2026-09-22): `ema_update` now lives in `src/dew/training/transaction.py:41-55`; the unit-decay select is at line 54. The preference and GRPO objectives set the unit decay through `ema_decay=1.0`, which `LMObjective` turns into `EMASpec(decay=optax.constant_schedule(1.0))` (`src/dew/objectives/rl/preference.py:50`, `src/dew/objectives/lm/objective.py:545-546`).

## 4. The rollout capability

Sampling is effectful, host-side and sometimes remote, so it is a capability the trainer is given, beside the checkpointer, the tracker and the profiler, all of which are `X | None = None`:

```python
class Rollout(Protocol):
    def __call__(self, state: TrainState, batch: Batch, key: jax.Array) -> Batch: ...
```

`Trainer(..., rollout=None)` is the loop without a rollout. When one is given, the trainer calls it between `batch = next(train)` and the compiled step (`dew/training/trainer.py:477`) and reshards the result with `shard_batch(mesh, ...)`.

Why not the `step=` seam? `Trainer.step` replaces the compiled step's body and is documented as the one place for an update that is not one loss. It runs inside `jit` and owns the counter, the EMA and the write-back. A rollout is the opposite kind of work. It produces the batch the step then consumes, it may post to a vLLM server, and it cannot be traced. Putting it in the step body would either force generation inside `jit` or smuggle a host callback into a compiled function. The two seams hold two kinds of work, and the design keeps them apart.

Why not a method on `Objective`? The objective's surface is pure and crosses `jit`: `loss` and `evaluate` are functions of variables, a batch and a `Step`. An objective that opens a socket breaks that, and every objective would carry an identity method it never uses.

The rest follows from the capability design:

- Shapes are fixed, so `compile` traces once and needs no shape polymorphism.
- The rollout key is folded from `state.key` and `state.step`, the same stream the step key comes from, with one extra fold so a rollout and its step never share draws. Both are checkpointed, so a resumed run samples forward instead of replaying.
- Accumulation runs one rollout per micro-batch. A group never straddles a micro-batch, so the group baseline is computed inside the batch it belongs to, which is the whole batch at the default `accumulation=1`.
- The rollout runs per process on that process's slice, which is what `DevicePrefetchIterator` hands it, and `shard_batch` reassembles the global array. Groups are process-local by construction.
- Telemetry goes through the existing channels. Reward means ride `Aux.metrics` out of `loss`, which the trainer logs under `train/`. Wall time cannot come from inside `jit`, so the trainer times the host-side call and folds `train/rollout_seconds` into the same log tick that carries throughput (`dew/training/trainer.py:526-531`). There is no second return channel.

Correction (2026-09-22): the call site is now `src/dew/training/trainer.py:751-753`, through `_rollout` at lines 966-976, which folds the key at line 975 and reshards at line 976. `train/rollout_seconds` is logged at lines 1051-1052. The `Rollout` protocol is at lines 112-121.

## 5. Rewards

A reward is a callable from one finished record to a float: `reward(data_source, completion, ground_truth, extra_info) -> float`, the verl signature with the four fields the batch already carries (§1.4). It runs host-side inside the rollout, so no Python callable and no external process is ever reachable from a compiled step. A rule-based verifier, a hosted judge and a scoring model are all the same shape, and a run composes several by summing weighted scores in its own function.

## 6. The language objectives

The language objectives assemble pieces from `dew.rl`, whose estimators and surrogates are built, ported from Tunix and verl, and pinned against their fixtures (`dew/rl/advantage.py`, `dew/rl/surrogate.py`, `tests/fixtures/rl/*.npz`).

- SFT is `LMObjective` with `loss_role` (§2).
- DPO (`dew/objectives/rl/preference.py`) takes per-sequence log-probabilities as the negated per-token cross entropies the chunked head already returns, summed under the shifted `completion_mask`, for the policy and for `step.ema`. The loss is `preference_logsigmoid` over the pair halves, `-logsigmoid(beta * ((pi_c - ref_c) - (pi_r - ref_r)))`, averaged (`dew/rl/surrogate.py:173`). Validation scores the chosen responses' perplexity.
- GRPO (`dew/objectives/rl/grpo.py`) reads `old_log_probs`, `advantages` and `response_mask` from the rolled-out batch and is one composition: `clipped_surrogate(token_log_ratio(...), advantages, response_mask)` plus `beta * token_mean(k3_kl(...), response_mask)` (`dew/rl/surrogate.py:109, :73, :155, :60`). The current log-probabilities are sliced out of the concatenation one before the prompt width; `beta=0.0` leaves the reference unread. The advantages come from `group_advantage` or `rloo_advantage` inside the rollout (`dew/rl/advantage.py:100, :121`), where the rewards are. Validation scores the prompts' own perplexity off `prompt_length`, since a validation pass never samples.

Both compute per-token log-probabilities through `LMObjective.per_token_log_probs`, the negated chunked cross entropies, so the policy and the reference share one head path. None of these objectives derives new math, which is why `dew.rl` landed first.

Correction (2026-09-22): the objectives now call the `_terms` forms and reduce by the mask mass themselves: GRPO uses `token_log_ratio`, `clipped_surrogate_terms` and `k3_kl` (`src/dew/objectives/rl/grpo.py:142-162`), and DPO uses `preference_logsigmoid_terms` (`src/dew/objectives/rl/preference.py:100`). GRPO also takes `epsilon_low`, `epsilon_high`, a `dual_clip` that defaults to 3.0, and an optional `behavior_importance_cap` (`grpo.py:58-60`). Current lines in `src/dew/rl/surrogate.py`: `token_mean` 59, `token_log_ratio` 72, `clipped_surrogate` 154, `k3_kl` 163, `preference_logsigmoid` 199. In `src/dew/rl/advantage.py`: `group_advantage` 99, `rloo_advantage` 120.

## 7. Diffusion RL

Built in `dew.sampling.flow` and `dew.objectives.rl.flow`. FlowSDE implements the Solver contract; FlowTrajectory records states, times, joint Gaussian log densities, and stochastic support. FlowRollout collects complete reward groups through the existing host capability. FlowGRPOObjective returns additive Mean statistics for clipped per-coordinate policy ratios and conditional Gaussian transition KL. It excludes deterministic intervals. The policy loss has no dual clip. The reference is frozen only when beta is positive; evaluation uses live policy parameters.

## 8. Parity plan

| check | reference | recorded |
| --- | --- | --- |
| chat rendering and the assistant mask | TRL 1.12's mask for the same template and conversation | ids exact, mask exact, difference 0 |
| DPO loss and gradient | TRL 1.12 on fixed tensors | loss 5.96e-08, gradients exact |
| GRPO loss and gradient | verl 0.9's PPO path on one fixed rollout | loss 7.45e-08, gradients exact |
| group and RLOO advantages | verl on the same rewards | already pinned |
| SDE transition and its log-probability | author code at `879042cf`, with conditional KL checked independently through Gaussian distributions | `tests/test_flow_grpo.py`; fixture generated by `tools/parity_flow_grpo.py` |
| a rolled-out run's shapes | the tables in §1.3 | one trace of `compile` per run |

Each new test records the largest observed difference and tightens its tolerance to it. DPO carries a swapped-halves test guarding the pair order; GRPO carries one failing mutation per term: an unclipped ratio, a k1 penalty, and a flat mean.

## 9. What fits

Per parameter, fp32 params plus Adam's two moments plus a frozen reference is 16 bytes, before activations. For Qwen3-0.6B (`tests/fixtures/hf/qwen3-0.6b/config.json`: hidden 1024, 28 layers, 16 heads, 8 KV heads, ffn 3072, vocab 151936), that is about 9.5 GiB of state. It fits a 16 GiB card with a small batch and bf16 activations, and the rollout's KV cache is the next thing to size. For a 7B the same arithmetic is about 112 GiB, so the first configuration that fits is FSDP across two 80 GiB devices. The fp32 optimizer state is what stops it fitting on one device; the reference is not.

Correction (2026-09-22): the two totals above are in GB, not GiB. The fixture config has tied embeddings and gives about 596M parameters, so 16 bytes each is 9.5 GB (8.9 GiB). A 7B model at 16 bytes is 112 GB (104 GiB). The conclusions do not change.

## 10. Crossing over to verl and vLLM

Research scale uses Dew's own `generate` and its samplers. Beyond that, the crossover sits at the capability boundary. A `Rollout` that posts prompts to a vLLM server and returns token ids is one implementation of the protocol in §4, and the weights it serves come from `save_pretrained_decoder`, which writes the HF layout verl and vLLM already consume (`dew/interop/hf_decoders`). Neither reaches the trainer, the objective or the state.

## 11. Waves

| # | wave | acceptance | status |
| --- | --- | --- | --- |
| 1 | chat data path and `loss_role` | mask parity against TRL; a packed SFT batch carries four aligned per-token fields; loss counts assistant targets only | built |
| 2 | unit-decay reference | a non-finite parameter leaves the reference bit-identical, with and without `dynamic_scale`; a resumed run restores it | built |
| 3 | the rollout capability | identity default leaves the loop unchanged; one `compile` per run with a rollout; `train/rollout_seconds` in the log tick; a resumed run does not replay | built |
| 4 | DPO | loss and gradient parity against TRL; the pair layout asserted | built |
| 5 | GRPO on `dew.rl` | the composition matches verl end to end on one fixed rollout; a mutation of each term fails | built |
| 6 | trajectory, SDE solver, FlowGRPO | density and gradient parity; Gaussian KL oracle; real DiT update; two-process ownership and one-process parity | built |

## 12. Staged runs

`recipes/chain.py` links the waves into one run. A `Recipe` is an ordered tuple of `Stage`s over one built decoder, each stage naming its data, its loss (`sft`, `dpo` or `grpo`), its step count and, for GRPO, its reward and sampling sizes. Every stage trains in its own checkpoint directory. Every stage after the first initializes from the previous stage's final parameters through the objective's `pretrained` mechanism, which also freezes the next stage's reference. The optimizer restarts each stage. A stage refuses mismatched data, and a GRPO stage refuses to run without a reward.

`recipes/lm/train.py` stays the pretraining recipe. Its `--objective` flag stays `lm | masked_diffusion`, because its data path (a token directory), its `--pretrained` (a Hugging Face decoder, not a dew run) and its sampling setup assume pretraining from the start. Post-training stages need pair and prompt data, dew-checkpoint init and reward callables, and that command line has none of them, so they live in the chain instead of behind its flag.

Correction (2026-09-22): a `Stage` no longer names its loss. The type of its data decides it: `ChatMessages` trains SFT, `PreferencePairs` DPO and `Prompts` GRPO (`recipes/chain.py:37-68`). The LM recipe's `--objective` accepts `lm`, `masked_diffusion` and `block_diffusion` (`src/dew/objectives/lm/config.py:33-35`, `:77-79`).
