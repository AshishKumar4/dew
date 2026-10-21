# Post-training in Dew: SFT, DPO and online RL on one trainer

This is the binding design for post-training. It uses the [training](training.md) and [data](data.md) contracts and the shared [numerical gate](testing.md#numerical-accuracy). Implementation and qualification work lives in the [roadmap](../roadmap.md#rl-and-serving). Usage examples are in the [post-training guide](../concepts/post_training.md).

A modality is an objective, so post-training adds objectives, and the user learns one pattern:

```python
LMObjective(model, seq_len)
LMObjective(model, seq_len, loss_role=Role.ASSISTANT)
DPOObjective(model, seq_len, beta=0.1)
GRPOObjective(model, seq_len, beta=0.0)
Trainer(objective, optimizer, ..., rollout=SampledRollout(...))
```

They run on the same Trainer, state, sharding, EMA clock, checkpoints and tracker keys. The reference is the EMA tree at unit decay (§3), sampling is a host-side trainer capability (§4), and rewards are plain callables (§5). The online objectives assemble the shared estimators and surrogates in `dew.rl`; staged runs compose them in `recipes/chain.py` (§12).

## 1. Data

Post-training uses the same `Dataset` chain as every other objective. `Files`, `Hub` or `Memory` reads records; chat, preference and prompt transforms prepare them; the shared packer and executor handle placement and resume. Run configurations use `ChatData`, `PreferenceData` and `PromptData` to build these same chains.

### 1.1 Chat and SFT: a role per token

SFT needs one field beyond the pretraining contract: which role wrote each token. The loss counts assistant targets only when `loss_role=Role.ASSISTANT`.

`RenderChat` applies the tokenizer's chat template to the messages and optional tool schemas. It writes int32 `text` ids and aligned int8 `text_roles`: 0 padding, 1 system, 2 user, 3 assistant, 4 tool and 5 developer. Text parts join in order; nontext parts require a processor.

The assistant span comes from prefix rendering:

- For message *k*, render `messages[:k]` with `add_generation_prompt=True` to ids `p_k`, and `messages[:k+1]` without it to `f_k`.
- Both must be prefixes of the whole-conversation rendering, and `f_k[:len(p_k)]` must equal `p_k`. A template that rewrites the prefix fails at rendering.
- The assistant targets are `f_k[len(p_k):]`, including the end-of-turn id. Its opening header carries no assistant loss.

Other roles retain their spans in the same whole-conversation rendering. A turn that renders to nothing or rewrites an earlier turn fails with its message index. The template is never silently replaced to obtain a mask.

Packing carries ids and roles through one document-coordinate plan. A chunking or truncation step cuts every aligned field together. The prepared batch is:

| Key | Shape | Dtype | Content |
|---|---|---|---|
| `text` | `[B, S+1]` | int32 | Packed conversation ids |
| `text_coordinates` | `Coordinates`, `[B, S+1]` leaves | int32/bool | Document membership, padding and positions |
| `text_roles` | `[B, S+1]` | int8 | Role per id, 0 on padding |

`loss_role` multiplies the objective's target weights by `text_roles[:, 1:] == loss_role`, together with the existing pad and document-boundary weights. With no `loss_role`, every valid target counts. A role-masked objective requires the roles column.

### 1.2 Preference pairs

`PreferencePair` reads `chosen` and `rejected` id lists with optional `chosen_mask` and `rejected_mask`. A missing mask marks the whole sequence as completion. The transform pads each pair together; choosing `seq_len=S+1` prepares the row width expected by `DPOObjective(..., seq_len=S)`.

| Key | Shape | Dtype | Content |
|---|---|---|---|
| `input_ids` | `[B, 2, S+1]` | int32 | Chosen at pair index 0, rejected at 1 |
| `completion_mask` | `[B, 2, S+1]` | int32 | Completion marks aligned with the ids |

The objective reads the pair index explicitly and shifts the completion mask with the next-token targets. Shuffling never separates a pair. A row that exceeds the declared width fails; chunking a pair would cut the completion whose preference was scored.

### 1.3 Prompts and packed rollouts

`RenderPrompt` accepts a message conversation, string or id sequence. It left-pads ids to `max_prompt_len` and writes `prompt_length` plus `data_source`, `ground_truth` and `extra_info` as fixed-width UTF-8 byte values stored in int32 arrays. Missing reward fields become empty strings. Over-length prompts are dropped by default; `long="truncate"` explicitly keeps the tail.

`SampledRollout` generates a group of completions per prompt, scores each completion and builds one-call sessions through `sessions.pack`. The learner reads fixed-width packed chains; a prompt and completion must fit the objective's `seq_len+1`. The primary aligned columns are:

| Key | Shape | Content |
|---|---|---|
| `input_ids` | `[N, W]` | Prompt and completion chains |
| `text_segment_ids`, `text_positions` | `[N, W]` | Chain identity and position, with segment 0 on padding |
| `response_mask` | `[N, W]` | Trainable sampled actions, including EOS |
| `old_log_probs` | `[N, W]` | Raw proximal-policy likelihoods |
| `behavior_log_probs` | `[N, W]` | Likelihoods under the actual sampler |
| `advantages` | `[N, W]` | The session's group or RLOO advantage on its actions |
| `versions`, `session_index`, `call_index` | `[N, W]` | Policy version and ownership of each action |
| `session_weights` | `[N, W]` | Weights for averaging each session's token mean equally |

Here `W=seq_len+1`. Optional routing records and filtered sampling-support columns travel beside these arrays when an engine supplies them. Every likelihood refers to the same action as its mask and advantage. Prompt lengths remain input metadata; the loss uses packed coordinates. Rewards and completion lengths are reported as rollout metrics.

EOS, or the ID completing one of the policy's stop strings, marks a completed session. Exhausting the generation budget marks it truncated; the explicit truncation policy decides its training mass and reward treatment. The text passed to the reward excludes an EOS ID the draw ended on and keeps a stop string it ended on, while the response mask includes either ID. Raw policy likelihoods and sampling likelihoods remain distinct.

The rollout reads each process's owned prompts and validates them before collective generation. A fixed padded prefill and decode program handles different prompt lengths and stopping outcomes without changing the collective schedule. The trainer places the returned learner batch on the training mesh.

### 1.4 verl's parquet schema

| verl field | Content | Dew boundary |
|---|---|---|
| `data_source` | Dataset or reward-rule name | Prompt reward context |
| `prompt` | Chat messages | Shared chat or prompt rendering |
| `reward_model.ground_truth` | Reference answer | `ground_truth` |
| `extra_info` | Additional task context | Preserved reward context |

The adapter owns this mapping. A record that supplies the reference answer both at the top level and under `reward_model` is ambiguous and fails at that boundary.

verl's `compute_score(data_source, solution_str, ground_truth, extra_info)` takes the completion and the three fields that Dew's reward takes as one `PromptRecord`; `dew.objectives.rl.verl.verl_reward` and `compute_score` convert between the two at the verl boundary.

## 2. The loss role

A loss mask belongs to the objective's targets. `loss_role` names one typed role already present in the batch rather than introducing a second mask configuration. The same `LMObjective` serves pretraining and SFT, and both use the shared scoring and head path.

## 3. The frozen reference

DPO keeps the starting policy in the existing EMA slot at unit decay; GRPO and FlowGRPO keep it when their positive KL weight reads it. The objective declares this through `EMASpec.constant`, and the loss reads the resulting variables through `step.ema`.

- Only live parameters reach the optimizer. The frozen tree has no optimizer state or gradient updates.
- The reference is checkpointed and placed with the training state.
- The EMA clock advances on completed optimizer updates, not micro-steps or rejected updates.
- At unit decay, `training/transaction.py::ema_update` returns the old average unchanged. It must remain bit-identical even if a live parameter is non-finite; multiplying that parameter by zero would not preserve it.

The reference forward uses the same model and scoring code as the policy with a different variable tree. The EMA slot holds this frozen reference instead of a moving policy average. Reference likelihoods are computed from the current batch, so there is no second cache of scores to invalidate when the data path changes.

## 4. The rollout capability

Sampling is effectful, host-side and sometimes remote, so the trainer receives it as a capability beside its other effects:

```python
class Rollout(Protocol):
    def __call__(self, state: TrainState, batch: Batch, key: jax.Array) -> Batch: ...
```

`Trainer(..., rollout=None)` runs without a rollout. With one, it reads a batch, calls the rollout, places its output and invokes the compiled step. The compiled `step=` interface owns optimizer work; external generation belongs outside it. The objective's loss and evaluation stay pure functions of variables, batch and step.

The rollout key folds the checkpointed run key with the step and a distinct rollout tag. Sampling and the update do not share draws. Each prompt's reward group stays together. Shapes are fixed for each compiled signature; EOS outcomes change validity, not array shapes.

The trainer measures host rollout time as `train/rollout_seconds`. The rollout's metric mapping reports under `rollout/<name>`; objective statistics travel through `Aux.metrics`. A resumed run restores the key, training state and data position before collecting the next rollout.

## 5. Rewards

A reward is a callable from a completion and its prompt's record to a float: `reward(completion, record) -> float`, where `record` is the `PromptRecord` of the three fields the batch already carries (§1.4), the ground truth normalized where the source reads it. A reward reads the fields it needs; verl's four-argument signature lives at the verl boundary (`verl_reward`). It runs host-side inside the rollout, so no Python callable and no external process is ever reachable from a compiled step. A rule-based verifier, a hosted judge and a scoring model are all the same shape, and a run composes several by summing weighted scores in its own function.

The prompt's tokenizer decodes completions before scoring. The [roadmap](../roadmap.md#rl-and-serving) records per-member reward routing.

## 6. The language objectives

The objectives assemble the estimators and surrogates in `dew.rl.advantage` and `dew.rl.surrogate`.

- SFT is `LMObjective` with `loss_role`.
- DPO computes policy and frozen-reference completion log-probabilities over the chosen and rejected halves. `preference_logsigmoid_terms` forms `-logsigmoid(beta * ((pi_c - ref_c) - (pi_r - ref_r)))`; the objective averages over pairs and validates on chosen-response perplexity.
- GRPO scores each packed chain through `packed_log_probs`. Its default policy loss combines `clipped_surrogate_terms(token_log_ratio(...), advantages, response_mask)` with `beta * k3_kl(...)`. It returns additive `Ratio` statistics under the response mask and reads no reference when `beta=0`.

Policy and reference use the same `TokenScorer` and shared LM-head path. GRPO exposes asymmetric clipping, a dual-clip parameter, policy-loss selection and token-mean or session-mean aggregation. Its behavior correction compares detached proximal and sampling likelihoods: a TIS cap or IcePop band, with optional sequence or geometric rejection bands. Engine routing and sampling-support records are replayed when present, at the engine's sampling temperature.

Group and RLOO advantages are formed from each prompt's own completion group. GRPO validation scores the prompts' perplexity; evaluation does not collect new completions.

## 7. Diffusion RL

`FlowSDE` implements the solver contract. `FlowTrajectory` records states, times, joint Gaussian log densities and stochastic support. `FlowRollout` collects complete reward groups through the host rollout capability.

`FlowGRPOObjective` returns additive `Ratio` statistics for clipped per-coordinate policy ratios and conditional Gaussian transition KL. Deterministic intervals carry no policy mass. The policy loss has no dual clip. The reference is frozen only when beta is positive; evaluation uses live policy parameters.

## 8. Reference parity

All numerical outputs and gradients use `tests/reference_error.py` and the [shared accuracy contract](testing.md#numerical-accuracy). The bound comes from each reference element's measured error against float64 truth over reduction orders, with fresh-order confirmation. No tolerance is fitted or tightened to a single observed difference.

| Behavior | Independent comparison |
|---|---|
| Chat rendering and assistant spans | The same tokenizer/template and conversations, with exact discrete ids and roles |
| DPO loss and gradients | TRL on the same policy/reference scores and masks |
| GRPO surrogates, corrections and gradients | verl on the same packed rollout and likelihoods |
| Group and RLOO advantages | The reference estimator on the same grouped rewards |
| Diffusion transitions and gradients | The algorithm's reference implementation and an independent conditional-Gaussian KL calculation |

Reference versions, inputs and reproduction commands live with their fixture generators. Comparison includes every output and gradient the flow touches. A failure is investigated in Dew, the reference arm or the bound's derivation; it is never removed by fitting a tolerance, filtering positions or swapping the oracle.

## 9. Memory

With fp32 parameters, Adam's two moments and a frozen fp32 reference, persistent state costs 16 bytes per parameter before activations and the generation cache. A roughly 596M-parameter decoder needs about 9.5 GB (8.9 GiB); a 7B model needs 112 GB (104 GiB). The planner must also account for activations, temporary buffers and rollout KV state. FSDP shards the persistent trees; a reduced-precision checkpoint does not remove the optimizer's fp32 state.

## 10. External generation engines

A remote `Rollout` posts generation requests and returns the same session layout the native collector produces. Export uses the engine's expected weight layout through the existing interop writer. Weight publication and generation do not change the trainer, objective or checkpoint schema. Each engine's adapter owns its EOS, truncation, likelihood and policy-version semantics.

## 11. End-to-end qualification

The [testing spec](testing.md) applies to every objective. A lifecycle checks training, checkpoint, resume, evaluation, export and reload on the relevant model. SFT keeps roles aligned through packing; the unit-decay reference survives rejected and non-finite updates; resumed collection continues its recorded random stream.

DPO includes a swapped-halves mutation. GRPO includes a failing mutation for each surrogate, clipping and KL term. Diffusion RL checks a real model update and ownership and numerical parity across processes. All continuous comparisons use section 8's shared gate.

## 12. Staged runs

`Recipe` is an ordered tuple of `Stage` values over one decoder. Each stage names its loss, and its data is a chain ending in that loss's preset: `"pretrain"` over `presets.lm`, `"sft"` over `presets.sft`, `"dpo"` over `presets.preferences`, `"grpo"` over `presets.prompts`, and `"ppo"` over rows of task ids with `Episodes`. `Recipe.run` checks each stage's data against its loss through the chain's facts, the fields its preset writes, and refuses a mismatch by name. A `Teacher` distills a pretraining or SFT stage.

Each stage has its own checkpoint directory and a restarted optimizer. `Objective.model_variables` defines the handoff: PPO passes its policy without the critic; distillation passes the student without the teacher. The next stage initializes from those variables and freezes its own reference where required. `Recipe.run` builds all stages' objectives and rollouts before the first stage trains, so invalid data or settings fail before work begins.

A GRPO stage requires a reward and uses its prompts' tokenizer for decoding and EOS. PPO receives its environment, verifier and critic through `Episodes`. Full-control runs build a `Trainer` directly. The user guide is [Stage chains](../concepts/post_training.md#stage-chains).

## 13. External integrations

External generation engines, agent tools and training frameworks enter through existing rollout and session interfaces. Each integration uses the maintained system's public API.

| Boundary | Contract |
|---|---|
| Generation and weight publication | `RolloutServer` owns requests and policy versions. Native, vLLM and SGLang adapters produce the same rollout records; `SafetensorsReload` and `NCCLPush` publish through that interface. |
| Asynchronous collection | `RolloutScheduler` owns look-ahead, lag, synchronization and admission. Session sources interleave turns through the engine batch. Restore accounts for admitted and in-flight work explicitly. |
| Agent environments | `Environment` and `EnvironmentFactory` enter through `EpisodeRollout` and `EnvironmentSource`. Agent commands use the gateway's recording interface. Process and sandbox lifecycles stay behind `Runner`. |
| verl interchange | `to_verl` and `from_verl` map sessions to `AgentLoopOutput`. Record interchange alone establishes no live transport, worker orchestration or learner integration. |
| Diffusion RL interchange | An adapter first qualifies a shared loss and its gradients on fixed inputs, then maps rollout records through the same session boundary. |

Every engine claim names the exercised flow. Qualification includes policy-version publication, EOS and truncation semantics, cancellation of a long decode, session restore and a real tool-using rollout. Record conversion, simulated engines and small-model runs establish only the behavior they exercise. Remaining work belongs in the [roadmap](../roadmap.md#rl-and-serving).
