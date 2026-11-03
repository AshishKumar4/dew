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

They run on the same Trainer, state, sharding, EMA clock, checkpoints and tracker keys. The reference is the EMA tree at unit decay (§3), sampling is a host-side trainer capability (§4), and rewards are plain callables (§5). The online objectives assemble the shared estimators and surrogates in `dew.rl`; staged runs compose them in `recipes/chain.py` (§12). Control and world-model objectives run on the same trainer with an environment in place of a generation engine (§14).

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

`FlowSDE` implements the solver contract. `FlowTrajectory` records states, times, joint Gaussian log densities and stochastic support.

`GroupedRollout` is the rollout the flow methods share. It samples each source row `groups` times with a policy of the trainer's state over `steps` time points, scores every sample once on process zero with the run's `reward` (a callable from the images and their batch to one float each) and broadcasts the scores, then cuts the generation into the training rows this process owns. The batch it returns reaches the compiled step through the host rollout capability. A method supplies the weights it samples with, its generation and its training rows. `FlowRollout` collects complete reward groups of SDE trajectories sampled with the live weights, and trains the first `train_steps` transitions of each.

`FlowGRPOObjective` returns additive `Ratio` statistics for clipped per-coordinate policy ratios and conditional Gaussian transition KL. Deterministic intervals carry no policy mass. The policy loss has no dual clip. The reference is frozen only when beta is positive; evaluation uses live policy parameters.

`DiffusionNFTObjective` (arXiv 2509.16117) trains a rectified-flow velocity model on the forward process. It computes no likelihood, so its sampler is any solver over the old policy and it uses no classifier-free guidance. A sample the old policy drew has an optimality probability r, its advantage clipped to `adv_clip_max` and mapped onto [0, 1]. The loss trains the sample at r on the implicit positive policy, (1 - beta) v_old + beta v_theta, and at 1 - r on the implicit negative one, (1 + beta) v_old - beta v_theta, in the released trainer's form, each policy's clean-sample estimate against the sample over the detached mean absolute error. A positive `kl` adds the squared difference of the live velocity from the base model's, which the objective reads from the starting tree and holds frozen under `TEACHER`. It returns additive `Ratio` statistics.

`NFTRollout` is its `GroupedRollout`. It draws `groups` clean samples of each source row with the old policy, the objective's solver over `steps` time points and no guidance, and trains each at the first `train_steps` times of that grid. A sample's advantage is its reward less its group's mean, over the population standard deviation of every reward in the batch plus 1e-4. The old policy is the EMA slot, which follows the trained weights at `old_decay` after each update, so it holds a policy to sample from and not an average to publish. Evaluation, previews and the published task use the live weights, as FlowGRPO's do.

`FlowRolloutConfig` builds the rollout of either objective in a run configuration, scoring each sample with an image metric, and `diffusion_nft` is registered beside `flow_grpo`.

## 8. Reference parity

All numerical outputs and gradients use `tests/reference_error.py` and the [shared accuracy contract](testing.md#numerical-accuracy). No tolerance is fitted or tightened to a single observed difference.

| Behavior | Independent comparison |
|---|---|
| chat rendering and the assistant mask | TRL 1.12's mask for the same template and conversation |
| DPO loss and gradients | TRL on the same policy/reference scores and masks |
| GRPO loss and gradient | verl 0.9's PPO path on one fixed rollout |
| group and RLOO advantages | verl on the same rewards |
| Diffusion transitions and gradients | The algorithm's reference implementation and an independent conditional-Gaussian KL calculation |
| DiffusionNFT loss, advantages and old-policy decay | NVlabs/DiffusionNFT's released trainer at cbb14f84 (`scripts/train_nft_sd3.py`, `flow_grpo/stat_tracking.py`), transcribed in float64 |

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

## 14. Environments, rollouts and world models

Control and world-model objectives run on the same trainer as the language and diffusion ones, with an environment in place of a generation engine. An objective is a loss over batches of steps. Its `Rollout` (§4) acts in the environment and returns those batches, and the trainer keeps the optimizer, the EMA slot, the checkpoints and the tracker. The trainer still reads a batch, which the rollout replaces whole, so a run passes an idle loader: `idle()` (`dew.objectives.rl.config`) is a batch of no records, one value for the step, which any mesh places whole, the same each time it opens so that a resume reads on, with no record count to make an epoch of. The environment run kind passes it to its trainer, and a run composed in Python passes it to `Trainer.fit` the same way. A rollout reports the mean return of the episodes that ended as `rollout/episode_return`, and `ControlPPOObjective` and `DreamerV3Objective` have no validation loss, since each is scored by its episodes' return. A rollout whose next batches depend on what it carries between calls holds that state in the run's checkpoints beside the training state (the `Rollout` protocol's `saved`, `saved_shapes` and `restored`, `dew.training.rollout`): `ControlRollout` its environments mid episode, its collection and its minibatches' order; `DreamerRollout` the environment steps it took and, with `checkpointed_replay`, its replay and environments too (§14.3). A resumed run carries on from it and samples the batches the uninterrupted run did. A step written without a rollout's state (by an earlier Dew) resumes with the environments begun again and, for `DreamerRollout`, owes only the steps of the updates after it. A cooldown branch runs on the rollout's state as the run left it and puts it back (`branch`); a `Simulator` puts its own environments back by its `branch()`, and one with none is refused the branch. A run of several processes with a checkpointer is refused one as its fit starts, since each process holds its own environments; a `Simulator`'s environments live outside JAX and begin again on a resume.

The pieces compose through three types. The `Env` is what a policy acts in, a simulator or a world model alike. The `Transition` and `Carry` are what `env_rollout` and `simulator_rollout` both record. The `Rollout` is what the trainer calls. A run builds them in Python, `Trainer(objective, optimizer, rollout=rollout)`, or from its record as the run kind `env` (`EnvRunConfig`, `dew.objectives.rl.config`). Its `env` names the environment by its class and arguments (`EnvConfig`: a JAX `Env`, a host `Simulator` or a function that returns one, `flight`, `gate_race`, `playground` or `skydreamer`), batched (`count`) and cut (`episode_length`, `action_repeat`) where it says. Its `model` and `objective` are the agent, each named by its import path (a `WorldModel`, its `dynamics` a nested `TSSM` record, or an `ActorCritic`; a `DreamerV3Objective` or a `ControlPPOObjective`), whose action and observation widths the run derives from the environment, and its `rollout` is the rollout's class and settings (`RolloutConfig`: `dreamer` or `control`). A `teacher` names a saved PPO run of the kind: it labels each step with its greedy action where the objective imitates labels (`imitates="label"`, DAgger, `dew.rl.aggregation`), and where it `demonstrates` it flies the prefill with its policy's draws (`DreamerRollout(demonstrator=)`). `pretrained` names the saved run a stage starts from, its model and weights at a pinned step, and `episodes` the splits each validation scores by flying the agent's greedy policy, or with `drawn` its draws as it acts (`act`), each by its name (`EpisodesConfig`, `dew.training.Episodes`): `steps` from fresh episodes of the run's own JAX `Env` or of an environment the split names, a `Simulator` among them (the run's own simulator holds its rollout's environments mid episode, which a flight would move), logged as `<split>/return` and `<split>/steps` and, for each view in `reached`, its mean at each first episode's last step. The run reads no data, so the trainer is handed `idle()`, which the rollout replaces. The environment is what the run reads, so a resume may change it, as a curriculum does; every other field is what it trains. `tools/fpv_race.py` trains its teacher and its world-model stages this way, `tools/skydreamer_race.py` SkyDreamer's recipe (§14.2), and `tools/rl_acceptance.py` its PPO and DreamerV3 runs, DreamerV3's scored by its draws, as its paper reports it, and by its greedy policy beside them. An `Env` here is a simulator or a world model, and not §13's `Environment`, a language agent's sandbox or harness. Dreamer 4's world model learns from video, so it is a run over data, the run kind `world_model` (§14.4).

### 14.1 Environments and replay

An `Env` (`dew.rl.env`) is two jittable functions over a state that is a pytree, `reset(key)`, which starts an episode, and `step(state, action)`, which advances it, each answering the new state and a `TimeStep`. A `TimeStep` holds the observation (an array, or a dict of named views), the reward, and two flags a loss reads apart. `terminated` is where the episode ended in a terminal state, whose value is zero, and `truncated` where a time limit cut it, whose value is the state's own. An action is `action_size` numbers in [-1, 1]. `Batched(env, count)` is `count` copies under `jax.vmap`, each reset from its own key, its leading axis shardable as a batch axis is. `TimeLimit` cuts an episode after `episode_length` steps, each repeating its action `action_repeat` times and summing their rewards, as Brax's `EpisodeWrapper` does. `dew.rl.playground` makes MuJoCo Playground's environments Envs (the `playground` extra).

`env_rollout(env, policy, params, carry, steps)` is the one loop over an `Env`, a `lax.scan` of a policy and a step. A policy is `policy(params, observation, key, memory) -> (action, extras, memory)`, its `memory` a recurrent policy's state. The loop answers the carry and each step's `Transition`, leading with the step axis. An episode that ends restarts in place, its transition keeping the observation it ended at. A learned environment, whose imagined episodes never restart, passes `reset=False`. `Driven` steps a batched environment as a world-model learner's driver does (DreamerV3's, `embodied/core/driver.py`). Its observation is the whole step, a dict of the `observation`, the `reward` that came with it and whether it is its episode's `first`, `last` or `terminal`, and the step after an episode's last restarts it.

A simulator written outside JAX, in CUDA, Warp, PyTorch or C++, cannot step inside a jitted program. `dew.rl.simulator` is the boundary it plugs into. A `Simulator` is a batch of environments stepped from the host, with `reset(seed)`, `step(action)` and `restart(mask)`, whose arrays cross by DLPack, without a copy where both sides hold them on one device. `simulator_rollout` is `env_rollout`'s counterpart, the policy compiled through Dew's compile point (under the options of the `models` it runs) and its executable fetched once, so no step builds a dispatch key, and the simulator stepped from the host, recording the same `Transition` and `Carry`, and `DrivenSimulator` is `Driven`'s. A rollout therefore runs over either kind of environment, and `rollout_start` starts either.

`ReplayBuffer` (`dew.rl.replay`) keeps the last `capacity` steps of each environment on the device, a ring per environment over a pytree of step fields. `add` writes steps in order, the oldest overwritten, and `sample(key, batch, length)` draws windows of `length` consecutive steps uniformly among those it holds whole, as DreamerV3's uniform replay does. Both are jittable.

`dew.rl.evaluation` scores a policy by the episodes it flies. `flights(policy, env, steps)` is the one flight: from fresh episodes of a JAX `Env` in one program, or of a `Simulator` or `DrivenSimulator` from the host, it answers the steps that followed each action, their observations the environment's own. `first_episodes` marks each environment's first episode in them, and `first_episode_rows` reads each whole: its return, steps, whether it was terminated, its last reward and the views asked for at its last step. `episodes` is the scorer an `Episodes` split runs, their means.

### 14.2 Drone racing

`dew.rl.flight` races first-person drones through gates, the largest environment built on these pieces. `dew.rl.quadrotor` holds a five-inch racing quadrotor's physics in JAX for one drone, which `vmap` batches. It has rotor thrust and drag, first-order motor lag, a battery's sag, a flight controller in Betaflight's form over the body rates, rotor, body and vertical drag, ground effect, and disturbances and noise. Each episode draws every parameter, its spread scaled by a `ramp` from the nominal drone (0) to the full spread (1).

`FlightEnv` is one drone's race as an `Env`. An action is a throttle and three body rates. Each episode flies a course of its own, drawn from four families (published layouts, loops, chained gate primitives and figure-eights) at the race's `difficulty`, and starts anywhere on it. The reward is the progress made toward the next gate less costs on the body rates and on the action's rate of change, plus a reward for the camera pointing at the next gate (Swift's perception term), a bonus for each gate passed through its opening and a penalty for a crash, which ends the episode. Without the perception term a teacher of the privileged views flies with the next gate out of its camera's view, a flight no camera policy can learn from. The ground, the gates' frames and the obstacles are solid along each physics step. The observation is a dict of the privileged views a run asks for: the state, the next gates' corners, the drone's airframe and drawn physics, the nearest obstacles, and `teacher`, the observation of a privileged PPO teacher. A curriculum raises `difficulty` and `ramp` between collections.

`GateRace` is a batch of drones raced by `FlightEnv` on the device and seen through a `Renderer`, offered as a `Simulator`. Its observations add the camera's `image`, which a world-model agent sees alone. `Renderer` is a protocol with one method, `render(scene, cameras, opened)`, that answers each drone's image from what its course holds (`Scene`), blurred by the drone's motion between the shutter's open and close. `dew.rl.hyperlux`'s `HyperLuxRenderer` draws it with HyperLux's ray tracer (the `hyperlux` extra, an NVIDIA GPU with OptiX) over the assets `tools/fpv_assets.py` writes, as a pinhole, or as an equidistant fisheye resampled from a larger pinhole of the same field of view (`fisheye`, `source`), and the race develops the image by the drone's drawn exposure, white balance, gamma and noise. `Seen` is the batch beneath `GateRace`: any JAX race of one drone that shows its views and its `Scene`, stepped on the device and seen through a renderer.

`dew.rl.skydreamer` is SkyDreamer's race (Verraest et al., arXiv 2510.14783) on the same quadrotor and renderer. An action is a collective thrust and three body rates (`commanded`, shared with `FlightEnv`), sent after the command's latency (`delayed`). The reward is SkyDreamer's: progress toward the pre-gate before each gate, a cost on the body rates, and a reward for each crossing of a gate's pre-gate, plane and post-gate by how near its centre, the episode ending at a crossing outside the gate, on the ground, or, where the race strikes them (`Sky.frames`), at another gate's frame. Each episode starts at rest before a gate (Table III) on one of the race's `tracks` (`TRACKS`: a circle, and SkyDreamer's inverted loop and ladder, reconstructed with their unseen gates). The observation adds to the image the body rates, the rotors' speeds and the flight plan (Eq 12), and the privileged `state` a world model decodes. `ramp` scales every spread, the drone's, the image's, the rotor noise and the latency's, from none. `SkyRace` is one drone's race as an `Env`, and `SkyDreamerRace` its batch over a renderer (`Seen`).


### 14.3 Control and DreamerV3

`ControlPPOObjective` is Brax's PPO for continuous control, as distinct from §12's `PPOObjective` over language episodes. Its `ActorCritic` is a tanh-squashed normal policy and a separate value network, and its loss is the clipped surrogate over advantages computed from the live critic, a value loss and an entropy bonus. A step that ends its episode at a time limit bootstraps from the observation it reached, and one that ends in a terminal state takes no value after it. `ControlRollout` collects `unrolls` sequences of `unroll_length` steps from each environment of an `Env` or a `Simulator` with the live policy, then serves them as `epochs` shuffled passes of `minibatches` minibatches, one an update, and collects again once they are spent. Its environments stay mid-episode on the device between collections. Where observations are dicts, `reads` names the views the actor and critic read, so a privileged teacher reads the views of a `FlightEnv` that a camera policy never sees. `episode_return` scores a policy by the return of its first episodes.

`DreamerV3Objective` is DreamerV3 as its code computes it, over a `WorldModel`, an RSSM of a block GRU's deterministic state and categorical latents with a decoder, a reward head and a continue head, for vector or image observations and with optional heads that predict privileged entries. A batch is windows of replayed steps in `Driven`'s convention. The world model trains on the decoder, reward and continue losses and on the dynamics and representation KLs with one free nat. The actor and critic train in imagination. `LatentEnv` is the world model as an `Env`. Its state is the latent, its step the GRU and a draw from the prior, its observation the latent's feature and its reward the reward head's mean. Imagining is therefore the same `env_rollout` that acts in a simulator, run with `reset=False` for 15 steps from every replayed posterior state. The critic trains toward the imagination's lambda-returns, toward its own slow EMA copy (the EMA slot over the critic alone) and toward the replay's lambda-returns. The actor trains by REINFORCE on the returns less the values, scaled by the 5th-to-95th percentile range of the returns floored at 1, with an entropy bonus.

`DreamerRollout` is the loop around it. The agent, `DreamerV3Objective.act`, acts in an `Env` (as `Driven` drives it) or a `Simulator` (as `DrivenSimulator` does) and writes each step to a `ReplayBuffer`. Each update trains on `batch` windows of `length` replayed steps while the agent takes the environment steps that make each environment replay `train_ratio` steps, after `prefill` steps of uniform random actions. A `replay` of demonstrations (`demonstration_replay`) starts it in place of the prefill. A `demonstrator`, a policy of the environment's observation, flies the prefill in place of the uniform actions, `length` steps at a time, the agent's memory passed through; a `teacher` writes its own action at every step the rollout takes beside it, its label (`dew.rl.aggregation`), which an objective that `imitates` it learns (DAgger). With `checkpointed_replay` its checkpoints hold its replay and environments mid episode, so a resumed run ends where the uninterrupted one does, at the replay's size a checkpoint; without it they hold the steps it took, and a resumed run makes its replay again (§14). With the `policy`, `value` and `replay_value` scales at 0 and `imitation` positive, a run pretrains the world model from demonstrations and clones the pilot's actions before the agent acts. `episodes_return` scores an agent by its episodes' return, and `greedy` acts on the policy's mean, as a judge flies it. A `WorldModel` reads, beside its image, the vector entries its `proprio` names, symlogged through an MLP (SkyDreamer's body rates, rotor speeds and flight plan). With `slow_targets` the imagined returns bootstrap from the slow critic and the advantage is taken over it (DreamerV3's `slowtar`), and `smoothness` weighs the squared change of the policy's mean between imagined steps (SkyDreamer's Eq 11). A `decoder` scale of 0 rebuilds no observation, so the auxiliary heads alone decode privileged entries: SkyDreamer's informed world model.

`WorldModel(dynamics=TSSM(...))` replaces the GRU with TransDreamer's transformer state-space model (`dew.objectives.rl.tssm`). The posterior reads an observation's tokens alone, and the transformer reads each step's previous latent and action, attending within its episode, and answers the state the prior predicts from. Training runs a window in parallel. Acting and imagination run the same attention a step at a time over a cache of keys and values, so the policy and the imagined rollouts read the function the model trained as.

### 14.4 A world model of video

Dreamer 4's world model (`dew.objectives.world_model`, arXiv 2509.24527) learns from video and actions and has no policy. `CausalTokenizer` turns each frame, one at a time, into a few latents through a tanh, and `TokenizerObjective` trains it by masked-autoencoder reconstruction, each frame's patches masked at a rate drawn from U(0, 0.9). `Dynamics` is a transformer over a window's frames, each frame's latents regrouped into `spatial` tokens beside an action token and a shortcut token (its signal level's and step size's), attending block-causally, so a frame reads itself and the past. `WorldModelObjective` trains it over the frozen tokenizer's latents by shortcut forcing, and with `shortcut=False` by diffusion forcing, the baseline the paper improves on. `generate` rolls future latent frames out from a context in a few steps, and the tokenizer decodes them. The objectives read a clip's `video` in [-1, 1]: uint8 pixels, as the data workers write clips, are scaled to it (`unit_frames`).

A run over clips trains both stages as the run kind `world_model` (`WorldModelRunConfig`, `dew.config.world_model`): the tokenizer where its `objective` is a `TokenizerObjective`, the dynamics where it is a `WorldModelObjective`. Both read their batches from the run's data, as a JEPA run does, so this is a run over data, and not §14's environment run, which reads none and trains an agent that acts in an environment. The tokenizer's weights are no value a record can hold, so a dynamics stage names the saved tokenizer run that holds them (`tokenizer`: a `TokenizerConfig`, the run's directory and a step, `latest` or `best` too), as an environment run names its teacher: the run pins the step as it starts, so its record names the step it read, builds the tokenizer from that run's model for the frames of its own data, and refuses weights that do not fit them. The run derives what its data and the tokenizer state, and refuses a record that states it: the tokenizer's frame from the data's clips, the dynamics' latents and bottleneck from the tokenizer and its action width from the data's `actions`. The data states its clips' shape through its preset (`presets.clips`) or, for clips stored by `dew.data.sinks.export`, the `fields` of the export's metadata. `tools/rl_acceptance.py dreamer4` trains its tokenizer and its two dynamics models this way. Dew builds the world model at a small scale. The paper's agent, trained inside it, is specified in `docs/research/dreamer4.md` and not built.

Evidence so far: `tests/test_env_rollout.py`, `tests/test_dreamer.py`, `tests/test_aggregation.py`, `tests/test_env_runs.py`, `tests/test_tssm.py`, `tests/test_world_model.py`, `tests/test_quadrotor.py` and `tests/test_flight.py` hold each piece to its source's arithmetic transcribed apart from Dew's code and run in float64 (`assert_computes_the_oracle`, §8), or to a step-by-step loop and exact bookkeeping; `tests/test_world_model_runs.py` holds each stage of the world-model run kind to the same stage composed in Python, bit for bit. `tools/rl_acceptance.py` runs PPO, DreamerV3 and Dreamer 4 against the papers' reported scores, and `tutorials/12-world-models.ipynb` acts and imagines in one loop. Deferred: a run at a paper's scale, and HyperLux's renderer, which no test draws an image through.
