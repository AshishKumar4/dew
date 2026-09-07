# Post-training in Dew: SFT, DPO and online RL on one trainer

Built through wave 6. The current Flow-GRPO usage is in [the post-training guide](../concepts/post_training.md#flow-grpo). The earlier sections record the language post-training implementation at the time those waves landed.

A modality is an objective, so post-training is more objectives, and the user learns one thing:

```
LMObjective(model, seq_len)                                  # pretraining, as today
LMObjective(model, seq_len, loss_role=Role.ASSISTANT)        # SFT: the same class, a loss mask, a chat data path
DPOObjective(model, seq_len, beta=0.1)
GRPOObjective(model, seq_len, beta=0.0)
Trainer(objective, optimizer, ..., rollout=SampledRollout(...))
```

They run on the `Trainer` as it is (`dew.training.trainer`), with the same state, sharding, EMA clock, checkpoints and tracker keys. Three mechanisms carry the design: the reference is the EMA tree at unit decay (§3), sampling is one more trainer capability (§4), and rewards are plain callables (§5). The surrogate math is built and parity-tested in `dew.rl`, so the online objectives are assembly rather than derivation (§6). Staged runs chain these links in `recipes/chain.py` (§12).

## 1. Data

The three paths are `DatasetSpec`s in the `datasets` registry, beside `TokenWindows` and `PackedTokens`: `ChatMessages` (`dew/data/chat.py`), `PreferencePairs` (`dew/data/preferences.py`) and `Prompts` (`dew/data/prompts.py`). Each returns a `Dataset` from `load(batch=)`, so a recipe reaches them through the same `data:` subcommand every other run uses, and none of them changes the trainer.

### 1.1 Chat and SFT: a role per token

SFT needs one field beyond the pretraining contract: which role wrote each token, because the loss must count assistant tokens only.

`ChatMessages`, registered as `chat_messages`: a Grain source over a parquet file whose `prompt` column holds verl-shaped conversations, plus a tokenizer whose chat template renders them. `ConversationSource` reads the conversations; `RenderConversation` renders each and emits, per token, `text`, `text_roles` (int8: 0 pad, 1 system, 2 user, 3 assistant, 4 tool), which the packer carries beside `text_segment_ids` and `text_positions`. A `_lengths` pass renders every row up front, so a bad row fails the run with its index before packing starts.

The assistant span comes from prefix rendering, not from template markers:

- For message *k*, render `messages[:k]` with `add_generation_prompt=True` to ids `p_k`, and `messages[:k+1]` without it to `f_k`.
- Assert `f_k[:len(p_k)] == p_k`. A template whose rendering is not incremental in its prefix is refused by name rather than silently mis-masked.
- The assistant tokens are `f_k[len(p_k):]`, including the end-of-turn token. Training the stop token is the point.

TRL's `assistant_only_loss` needs a template carrying `{% generation %}` markers and swaps in a bundled template when the user's lacks them, which changes the rendered ids and not just the mask. verl tokenizes each message alone and concatenates, which is only equal to whole-conversation tokenization for templates that happen to be concatenation-safe. The prefix method is exact for any template and asserts it. TRL's mask is still the parity reference for the transform (§8).

Packing is unchanged machinery: `text_roles` joins as one more per-token feature in `FirstFitPackIterDataset` with its own `length_struct` and `padding_struct` entries, `DocumentChunks` cuts every per-token field of a document longer than the window so ids and roles stay aligned, and a conversation that starts with the assistant is refused, since its opening header cannot be separated from its completion through the template. Segment ids still stop attention at a document boundary.

| key | shape | dtype | content |
| --- | --- | --- | --- |
| `text` | `[B, S+1]` | int32 | packed conversation ids |
| `text_segment_ids` | `[B, S+1]` | int32 | which document, 0 on pad |
| `text_positions` | `[B, S+1]` | int32 | position inside that document |
| `text_roles` | `[B, S+1]` | int8 | role per token, 0 on pad |

`loss_role` multiplies the objective's existing target weights by `(text_roles[:, 1:] == loss_role)`, composed with the pad and segment-boundary weights it already computes. With no `loss_role`, every counted target counts, exactly as pretraining does.

### 1.2 Preference pairs

`PreferencePairs`, registered as `preference_pairs`: a parquet file or JSON rows of `chosen` and `rejected` token-id lists with `chosen_mask` and `rejected_mask` marking the completion tokens; absent masks default to all-completion. Every element is one `[2, seq_len]` pair, chosen at index 0, padded with `pad_id` and 0, so a grain batch is `[B, 2, seq_len]` and shuffling never separates a pair. A row longer than `seq_len` fails with its lengths: a pair cannot be chunked without cutting a completion, and silent cuts train the wrong preference.

| key | shape | dtype | content |
| --- | --- | --- | --- |
| `input_ids` | `[B, 2, S]` | int32 | pairs, chosen at index 0 |
| `completion_mask` | `[B, 2, S]` | int32 | 1 on completion tokens, including the stop token |

The objective reads the pair index out explicitly: a reshape would interleave two pairs into the halves and compare across them.

### 1.3 Prompts for online RL

`Prompts`, registered as `prompts`: a parquet file or JSON rows of `prompt` (messages, a string, or token ids) plus the reward columns `data_source`, `ground_truth` and `extra_info`, defaulting to empty when absent. Each row encodes to a left-padded `prompt` of `max_prompt_len` ids plus `prompt_length`, with the reward columns as fixed-width UTF-8 byte arrays so every leaf survives the device transfer. Prompts longer than the window keep their tail; blank prompts fail. No masks, because the rollout produces them.

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

The rollout returns fixed-width arrays even when EOS ends a completion early. The response mask and lengths describe valid actions. Generation groups equal prompt lengths before its cached decoder; those groups can create different JIT shapes. The returned training rectangle keeps `Trainer.compile` independent of completion lengths. The rollout receives the globally sharded batch, samples the rows this process's devices hold, and returns those rows; the group plan and decode trip count are agreed across processes, so ranks holding different lengths or stopping at different steps issue the same collectives over sharded parameters.

### 1.4 verl's parquet schema, mapped

| verl field | verl content | Dew |
| --- | --- | --- |
| `data_source` | dataset name, indexes the reward | `data_source`, passed to the reward (§5) |
| `prompt` | chat messages | rendered by the chat transform into `text` and `text_roles`, or encoded by the prompt source |
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

**Decision: the reference is the EMA tree at unit decay.** A preference or RL objective sets `ema = EMASpec(decay=optax.constant_schedule(1.0))` and reads `step.ema` in `loss`. `Step.ema` is the variables tree with the averaged leaves in place of the live ones (`dew/objectives/base.py`), so the reference forward runs through the same code as the policy forward with a different tree.

What falls out without new machinery:

- **Out of the optimizer.** Only the `params` collection is differentiated and only it reaches `tx.init`, so no masked optimizer and no zero-gradient tree.
- **Checkpointed and sharded.** `ema` is a field of `TrainState`, so `Trainer.shardings` shards it like params and a resumed run restores its reference with everything else.
- **The clock is already right.** The EMA runs on completed optimizer updates, not micro-steps, and a rejected mixed-precision step is not an update. A reference cannot drift under accumulation.
- **`select` scopes it.** `EMASpec.select` is a `PathFilter`, so a reference over part of the tree, a frozen encoder beside a trained head, costs one filter.

One change was required, and it is small. `ema_update` is `decay * average + (1 - decay) * live`. At decay 1.0 that is arithmetically the average, but `0.0 * NaN` is NaN, so a single non-finite parameter poisons a frozen reference on the step it appears, and the `finite` gate only exists when `dynamic_scale` is on. The fix is one select per leaf: return the average unchanged where `decay >= 1.0` (`dew/training/trainer.py:134`). The test that ships with it forces a non-finite parameter and asserts the reference is bit-identical afterwards, with and without `dynamic_scale`.

The cost of the decision is that a run cannot hold a moving policy EMA and a frozen reference at the same time. No objective in scope wants both. The alternative, a masked subtree inside `params`, was rejected: one extra full copy in HBM, an optimizer wrapper, gradient and update trees for something that never moves, and a second checkpoint layout. The EMA slot is already allocated, already sharded and already checkpointed.

**In-step, not precomputed.** `loss` runs the reference forward per batch. TRL precomputes reference log-probs to free a second model's memory; under this mechanism there is no second model to free, so precomputing buys one forward per step, roughly a third of forward plus backward, at the price of a cache that silently goes stale when the data path changes. Revisited only if a measured run shows reference forwards dominating.

## 4. The rollout capability

Sampling is effectful, host-side and sometimes remote, so it is a capability the trainer is given, beside the checkpointer, the tracker and the profiler, all of which are `X | None = None`:

```python
class Rollout(Protocol):
    def __call__(self, state: TrainState, batch: Batch, key: jax.Array) -> Batch: ...
```

`Trainer(..., rollout=None)` is exactly today's loop. When one is given, the trainer calls it between `batch = next(train)` and the compiled step (`dew/training/trainer.py:477`) and reshards the result with `shard_batch(mesh, ...)`.

**Why not the `step=` seam.** `Trainer.step` replaces the compiled step's body and is documented as the one place for an update that is not one loss. It runs inside `jit` and owns the counter, the EMA and the write-back. A rollout is the opposite kind of thing: it produces the batch the step then consumes, it may post to a vLLM server, and it cannot be traced. Putting it in the step body would either force generation inside `jit` or smuggle a host callback into a compiled function. Two seams, two kinds of work, and the design says which is which.

**Why not a method on `Objective`.** The objective's surface is pure and crosses `jit`: `loss` and `evaluate` are functions of variables, a batch and a `Step`. An objective that opens a socket is not that, and every objective would carry an identity method it never uses.

The rest falls out:

- **Fixed shapes**, so `compile` traces once and no shape polymorphism is needed.
- **Keys.** The rollout key is folded from `state.key` and `state.step`, the same stream the step key comes from, with one extra fold so a rollout and its step never share draws. Both are checkpointed, so a resumed run samples forward rather than replaying.
- **Accumulation.** One rollout per micro-batch. A group never straddles a micro-batch, so the group baseline is computed inside the batch it belongs to, which is the whole batch at the default `accumulation=1`.
- **Distributed.** The rollout runs per process on that process's slice, which is what `DevicePrefetchIterator` hands it, and `shard_batch` reassembles the global array. Groups are process-local by construction.
- **Telemetry through the existing channels.** Reward means ride `Aux.metrics` out of `loss`, which the trainer logs under `train/`. Wall time cannot come from inside `jit`, so the trainer times the host-side call and folds `train/rollout_seconds` into the same log tick that carries throughput (`dew/training/trainer.py:526-531`). There is no second return channel.

## 5. Rewards

A reward is a callable from one finished record to a float: `reward(data_source, completion, ground_truth, extra_info) -> float`, the verl signature with the four fields the batch already carries (§1.4). It runs host-side inside the rollout, so no Python callable and no external process is ever reachable from a compiled step. A rule-based verifier, a hosted judge and a scoring model are all the same shape, and a run composes several by summing weighted scores in its own function.

## 6. The language objectives

They are assembly over `dew.rl`, whose estimators and surrogates are built, ported from Tunix and verl, and pinned against their fixtures (`dew/rl/advantage.py`, `dew/rl/surrogate.py`, `tests/fixtures/rl/*.npz`).

- **SFT** is `LMObjective` with `loss_role` (§2).
- **DPO** (`dew/objectives/rl/preference.py`) takes per-sequence log-probabilities as the negated per-token cross entropies the chunked head already returns, summed under the shifted `completion_mask`, for the policy and for `step.ema`. The loss is `preference_logsigmoid` over the pair halves: `-logsigmoid(beta * ((pi_c - ref_c) - (pi_r - ref_r)))`, meaned (`dew/rl/surrogate.py:173`). Validation scores the chosen responses' perplexity.
- **GRPO** (`dew/objectives/rl/grpo.py`) reads `old_log_probs`, `advantages` and `response_mask` from the rolled-out batch and is one composition: `clipped_surrogate(token_log_ratio(...), advantages, response_mask)` plus `beta * token_mean(k3_kl(...), response_mask)` (`dew/rl/surrogate.py:109, :73, :155, :60`). The current log-probabilities are sliced out of the concatenation one before the prompt width; `beta=0.0` leaves the reference unread. The advantages come from `group_advantage` or `rloo_advantage` inside the rollout (`dew/rl/advantage.py:100, :121`), where the rewards are. Validation scores the prompts' own perplexity off `prompt_length`, since a validation pass never samples.

Both read and write per-token log-probabilities through `LMObjective.per_token_log_probs`, the negated chunked cross entropies, so the policy and the reference share one head path. Nothing in that list derives new math, which is the point of having landed `dew.rl` first.

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

Per parameter, fp32 params plus Adam's two moments plus a frozen reference is 16 bytes, before activations. For Qwen3-0.6B (`tests/fixtures/hf/qwen3-0.6b/config.json`: hidden 1024, 28 layers, 16 heads, 8 KV heads, ffn 3072, vocab 151936), that is about 9.5 GiB of state, which fits a 16 GiB card with a small batch and bf16 activations, and leaves the rollout's KV cache as the next thing to size. For a 7B the same arithmetic is about 112 GiB, so the first configuration that fits is FSDP across two 80 GiB devices, and the reference is not what makes it not fit: the fp32 optimizer state is.

## 10. Crossing over to verl and vLLM

Research scale uses Dew's own `generate` and its samplers. Beyond that, the crossover is at the capability boundary and nowhere else: a `Rollout` that posts prompts to a vLLM server and returns token ids is one implementation of the protocol in §4, and the weights it serves come from `save_pretrained_decoder`, which writes the HF layout verl and vLLM already consume (`dew/interop/hf_decoders`). Nothing about either reaches the trainer, the objective or the state.

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

`recipes/chain.py` links the waves into one run: a `Recipe` is an ordered tuple of `Stage`s over one built decoder, each stage naming its data, its loss (`sft`, `dpo` or `grpo`), its step count and, for GRPO, its reward and sampling sizes. Every stage trains in its own checkpoint directory; every stage after the first initializes from the previous stage's final parameters through the objective's `pretrained` mechanism, which is also what freezes the next stage's reference. The optimizer restarts each stage. A stage refuses mismatched data, and a GRPO stage refuses to run without a reward.

`recipes/lm/train.py` stays the pretraining recipe: its `--objective` flag stays `lm | masked_diffusion`, because its data path (a token directory), its `--pretrained` (a Hugging Face decoder, not a dew run) and its sampling setup assume pretraining from the start. Post-training stages need pair and prompt data, dew-checkpoint init and reward callables, none of which survive that command line, so they live in the chain instead of behind its flag.
