# Post-training in Dew: SFT, DPO and online RL on one trainer

Built through wave 6. The earlier sections record the language post-training implementation as it was when those waves landed. It uses the [training](training.md) and [data](data.md) contracts and the shared [numerical gate](testing.md#numerical-accuracy). Usage examples are in the [post-training guide](../concepts/post_training.md).

A modality is an objective, so post-training adds objectives, and the user learns one pattern:

```python
LMObjective(model, seq_len)
LMObjective(model, seq_len, loss_role=Role.ASSISTANT)
DPOObjective(model, seq_len, beta=0.1)
GRPOObjective(model, seq_len, beta=0.0)
Trainer(objective, optimizer, ..., rollout=RolloutScheduler.sampled(...))
```

They run on the same Trainer, state, sharding, EMA clock, checkpoints and tracker keys. The reference is the EMA tree at unit decay (§3), sampling is a host-side trainer capability (§4), and rewards are plain callables (§5). The online objectives assemble the shared estimators and surrogates in `dew.rl`; staged runs compose them in `recipes/chain.py` (§12).

## 1. Data

Post-training uses the same `Dataset` chain as every other objective. `Files`, `Hub` or `Memory` reads records; chat, preference and prompt transforms prepare them; the shared packer and executor handle placement and resume. Run configurations use `ChatData`, `PreferenceData` and `PromptData` to build these same chains.

### 1.1 Chat and SFT: a role per token

SFT needs one field beyond the pretraining contract: which role wrote each token. The loss counts assistant targets only when `loss_role=Role.ASSISTANT`. `RenderChat` writes that field, `text_roles`, beside the ids; the batch's fields, shapes and dtypes are [data's post-training batches](data.md#post-training-batches).

The assistant span comes from prefix rendering:

- For message *k*, render `messages[:k]` with `add_generation_prompt=True` to ids `p_k`, and `messages[:k+1]` without it to `f_k`.
- Assert `f_k[:len(p_k)] == p_k`. Dew refuses a template whose rendering is not incremental in its prefix, and names it, instead of masking it wrong.
- The assistant targets are `f_k[len(p_k):]`, including the end-of-turn id. Its opening header carries no assistant loss.

Other roles retain their spans in the same whole-conversation rendering.

`loss_role` multiplies the objective's target weights by `text_roles[:, 1:] == loss_role`, together with the existing pad and document-boundary weights. With no `loss_role`, every valid target counts. When `loss_role` is set, the objective refuses a batch with no `text_roles` and names the field.

### 1.2 Preference pairs

A pair's chosen and rejected ids and completion marks arrive as [data's preference batch](data.md#post-training-batches). The objective reads the pair index explicitly and shifts the completion mask with the next-token targets. Shuffling never separates a pair. A row longer than `seq_len` fails with its lengths. Chunking a pair would cut the completion whose preference was scored.

### 1.3 Prompts and packed rollouts

`RenderPrompt` writes [data's prompt batch](data.md#post-training-batches). Prompts longer than the window keep their tail; `long="drop"` drops them instead, each dropped prompt counted as a rejection. Blank prompts fail. The source makes no masks, because the rollout produces them.

`RolloutScheduler.sampled` generates a group of completions per prompt (`PolicyCollector`), scores each completion and packs the one-call sessions through `sessions.pack`. The learner reads fixed-width packed chains; a prompt and completion must fit the objective's `seq_len+1`. The primary aligned columns are:

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
| `ability` | task category | carried in `extra_info`; Dew dispatches on nothing |

The adapter owns this mapping.

verl's `compute_score(data_source, solution_str, ground_truth, extra_info)` takes the completion and the three fields that Dew's reward takes as one `PromptRecord`; `dew.objectives.rl.verl.verl_reward` and `compute_score` convert between the two at the verl boundary.

## 2. The loss role

A loss mask belongs to the objective's targets. `loss_role` names one typed role already present in the batch rather than introducing a second mask configuration. The same `LMObjective` serves pretraining and SFT, and both use the shared scoring and head path.

## 3. The frozen reference

DPO keeps the starting policy in the existing EMA slot at unit decay; GRPO and FlowGRPO keep it when their positive KL weight reads it. The preference and GRPO objectives set the unit decay through `ema_decay=1.0`, which `LMObjective` turns into `EMASpec(decay=optax.constant_schedule(1.0))` (`src/dew/objectives/rl/preference.py:50`, `src/dew/objectives/lm/objective.py:545-546`).

- Only live parameters reach the optimizer. The frozen tree has no optimizer state or gradient updates.
- The reference is checkpointed and placed with the training state.
- The EMA clock advances on completed optimizer updates, not micro-steps or rejected updates.
- The fix is one select per leaf: return the average unchanged where `decay >= 1.0` (`training/transaction.py::ema_update`). It must remain bit-identical even if a live parameter is non-finite; multiplying that parameter by zero would not preserve it. Its test forces a non-finite parameter and asserts the reference is bit-identical afterwards, with and without `dynamic_scale`.

The reference forward uses the same model and scoring code as the policy with a different variable tree. The EMA slot holds this frozen reference instead of a moving policy average. Reference likelihoods are computed from the current batch, so there is no second cache of scores to invalidate when the data path changes. We revisit this only if a measured run shows reference forwards dominating.

## 4. The rollout capability

Sampling is effectful, host-side and sometimes remote, so the trainer receives it as a capability beside its other effects:

```python
class Rollout(Protocol):
    def __call__(self, state: TrainState, batch: Batch, key: jax.Array) -> Batch: ...
```

`Trainer(..., rollout=None)` runs without a rollout. With one, it reads a batch, calls the rollout, places its output and invokes the compiled step. The compiled `step=` interface owns optimizer work; external generation belongs outside it. The objective's loss and evaluation stay pure functions of variables, batch and step.

The rollout key folds the checkpointed run key with the step and a distinct rollout tag. Sampling and the update do not share draws. Each prompt's reward group stays together. Shapes are fixed, so `compile` traces once and needs no shape polymorphism.

The trainer measures host rollout time as `train/rollout_seconds`. Reward means ride `Aux.metrics` out of `loss`, which the trainer logs under `train/`. There is no second return channel. A resumed run restores the key, training state and data position before collecting the next rollout.

## 5. Rewards

A reward is a callable from a completion and its prompt's record to a float: `reward(completion, record) -> float`, where `record` is the `PromptRecord` of the three fields the batch already carries (§1.4), the ground truth normalized where the source reads it. A reward reads the fields it needs; verl's four-argument signature lives at the verl boundary (`verl_reward`). It runs host-side inside the rollout, so no Python callable and no external process is ever reachable from a compiled step. A rule-based verifier, a hosted judge and a scoring model are all the same shape, and a run composes several by summing weighted scores in its own function.

The [roadmap](../roadmap.md#rl-and-serving) records per-member reward routing.

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

All numerical outputs and gradients use `tests/reference_error.py` and the [shared accuracy contract](testing.md#numerical-accuracy). No tolerance is fitted or tightened to a single observed difference.

| Behavior | Independent comparison |
|---|---|
| chat rendering and the assistant mask | TRL 1.12's mask for the same template and conversation |
| DPO loss and gradients | TRL on the same policy/reference scores and masks |
| GRPO loss and gradient | verl 0.9's PPO path on one fixed rollout |
| group and RLOO advantages | verl on the same rewards |
| Diffusion transitions and gradients | The algorithm's reference implementation and an independent conditional-Gaussian KL calculation |

Comparison includes every output and gradient the flow touches. A failure is investigated in Dew, the reference arm or the bound's derivation; it is never removed by fitting a tolerance, filtering positions or swapping the oracle.

## 9. Memory

With fp32 parameters, Adam's two moments and a frozen fp32 reference, persistent state costs 16 bytes per parameter before activations and the generation cache. A roughly 596M-parameter decoder needs about 9.5 GB (8.9 GiB); a 7B model needs 112 GB (104 GiB). FSDP shards the persistent trees; a reduced-precision checkpoint does not remove the optimizer's fp32 state.

## 10. External generation engines

A remote `Rollout` posts generation requests and returns the same session layout the native collector produces. A `Rollout` that posts prompts to a vLLM server and returns token ids is one implementation of the protocol in §4, and the weights it serves come from `Pretrained.save` (over `Pretrained.export`), which writes the HF layout verl and vLLM already consume (`dew/interop/pretrained`, `dew/interop/hf_decoders`). Weight publication and generation do not change the trainer, objective or checkpoint schema.

## 11. End-to-end qualification

The [testing spec](testing.md) applies to every objective. A lifecycle checks training, checkpoint, resume, evaluation, export and reload on the relevant model. SFT keeps roles aligned through packing; the unit-decay reference survives rejected and non-finite updates; resumed collection continues its recorded random stream.

DPO carries a swapped-halves test guarding the pair order; GRPO carries one failing mutation per term: an unclipped ratio, a k1 penalty, and a flat mean. DPO qualifies by loss and gradient parity against TRL; the pair layout asserted. Diffusion RL qualifies by density and gradient parity; Gaussian KL oracle; real DiT update; two-process ownership and one-process parity. All continuous comparisons use section 8's shared gate.

## 12. Staged runs

`Recipe` is an ordered tuple of `Stage` values over one decoder. Each stage names its loss, and its data is a chain ending in that loss's preset: `"pretrain"` over `presets.lm`, `"sft"` over `presets.sft`, `"dpo"` over `presets.preferences`, `"grpo"` over `presets.prompts`, and `"ppo"` over rows of task ids with `Episodes`. `Recipe.run` checks each stage's data against its loss through the chain's facts, the fields its preset writes, and refuses a mismatch by name. A `Teacher` distills a pretraining or SFT stage.

Each stage has its own checkpoint directory and a restarted optimizer. `Objective.model_variables` defines the handoff: PPO passes its policy without the critic; distillation passes the student without the teacher. The next stage initializes from those variables and freezes its own reference where required. `Recipe.run` builds all stages' objectives and rollouts before the first stage trains, so invalid data or settings fail before work begins.

A GRPO stage requires a reward and uses its prompts' tokenizer for decoding and EOS. PPO receives its environment, verifier and critic through `Episodes`. Full-control runs build a `Trainer` directly. The user guide is [Stage chains](../concepts/post_training.md#stage-chains).

## 13. External integrations

External generation engines, agent tools and training frameworks enter through existing rollout and session interfaces. Each integration uses the maintained system's public API. Dew builds no parallel sandbox, server or orchestration platform: integrations come first.

| Boundary | Contract |
|---|---|
| Generation and weight publication | `RolloutServer` owns requests and policy versions. Native, vLLM and SGLang adapters produce the same rollout records; `SafetensorsReload` and `NCCLPush` publish through that interface. |
| Rollout collection | `RolloutScheduler` owns grouping, admission, packing and reporting for every way sessions are drawn: `Synchronous` collects at each call from the trainer's variables, `Asynchronous` owns look-ahead, lag and synchronization. Session sources interleave turns through the engine batch. Deferred: durable scheduler state, since reopening the stream cancels in-flight work; a multi-host run. |
| Agent environments | `Environment` and `EnvironmentFactory` enter through `EpisodeCollector` and `EnvironmentSource`. External CLI harnesses run through Harbor trials and the `rllm-model-gateway` recorder (`HarborSource`, `Gateway`: `src/dew/interop/harbor.py:105-216,395-647`). Local isolation: `SubprocessEnvironment`, `ProcessRunner`, `ContainerRunner` and `SandboxFleet` (`src/dew/rl/sandbox.py`). Deferred: remote sandbox providers through the `Environment` protocol, adapting the provider's API without building one. |
| verl interchange | `to_verl` and `from_verl` map sessions to `AgentLoopOutput`. Deferred: live verl transport, Ray workers and HybridFlow resharding; a Dew learner inside a verl deployment. |
| Diffusion RL interchange | First match one shared algorithm's loss (FlowGRPO) against verl-omni on fixed tensors, then decide on a rollout-row interchange like `to_verl`'s. |

Evidence so far: simulated engines (`tests/test_rollout_servers.py`) and single-host vLLM runs (`docs/concepts/post_training.md`, Rollout servers).
