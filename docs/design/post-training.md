# Post-training in Dew: SFT, DPO and online RL on one trainer

Design, 2026-09-02. Not built. Every file and line cited against the main worktree unless the path starts with `.worktrees/`, which names the wave branch carrying that code.

The owner's rule decides the whole shape: a modality is an Objective, so post-training is more objectives, and the user learns one thing. The trainer does not change.

```
LMObjective(model)                                # pretraining, as today
LMObjective(model, loss_role=ASSISTANT)          # SFT: the same class, a loss mask, a chat data path
DPOObjective(model, pretrained=variables, beta=0.1)
GRPOObjective(model, pretrained=variables, reward=math_verifier, group=8)
FlowGRPOObjective(denoiser, ..., sampler=SDERolloutSampler(...), reward=ocr_scorer, group=8)
```

All six run on the unchanged `ObjectiveTrainer` (`src/dew/training/objective_trainer.py:60`) with the same checkpoints, sharding, EMA clock and wandb keys. Three mechanisms carry the design: a frozen reference in the state tree (§3), one optional rollout hook on the Objective (§4), and rewards as plain callables (§5). Where the six-constructor surface above cannot be honest, this document says so and fixes it; there are two such places (§2 for `mask=` becoming `loss_role=`, §9 for `reference=` becoming `pretrained=`).

## 1. Data paths

Three paths, all Grain, all producing the batch contract the objectives consume. verl's parquet schema maps onto them (§1.4).

### 1.1 Chat / SFT: records to packed rows with a role per token

The pretraining path reads fixed strides from `train.bin` (`src/dew/data/sources/text.py:23`), or whole documents packed with `text_segment_ids` and `text_positions` (`TokenDocumentSource` and `get_packed_token_dataset_grain`, `.worktrees/grain-packing/src/dew/data/dataloaders.py:721`). SFT needs one more per-token field: which role wrote each token, because the loss must count assistant tokens only.

**Source.** `ChatParquetSource` in a new `src/dew/data/sources/chat.py`: a Grain source over a parquet file of verl-shaped rows (§1.4). A record is one conversation, a list of `{role, content}` messages, plus the non-tensor columns carried through (`data_source`, `reward_model`, `extra_info`).

**Template and mask.** A Grain transform `ChatTokens(tokenizer, chat_template)` renders the conversation and emits, per token: `text` (ids), `text_roles` (int8: 0 pad, 1 system, 2 user, 3 assistant, 4 tool), `text_segment_ids`, `text_positions`. The assistant span is recovered by prefix rendering, not by `{% generation %}` markers:

- For each message k, render `messages[:k]` with `add_generation_prompt=True` to ids `p_k`, and `messages[:k+1]` without it to ids `f_k`.
- Assert `f_k[:len(p_k)] == p_k`; a template whose rendering is not incremental in its prefix is refused by name, not silently mis-masked.
- The assistant tokens are `f_k[len(p_k):]`, which includes the end-of-turn token. Training the stop token is the point; TRL warns when a template leaves it out of the mask (`trl/trainer/sft_trainer.py:1255-1263`, `/tmp/design/trl`).

TRL's mechanism was rejected as the primary path: `assistant_only_loss` requires a chat template carrying `{% generation %}` markers and swaps in a bundled fallback template when the user's lacks them (`trl/trainer/sft_trainer.py:1248-1253`, `/tmp/design/trl`), which changes the rendered ids, not just the mask. verl's mechanism (tokenize each message alone, strip the duplicated system prompt, concatenate; `verl/utils/dataset/multiturn_sft_dataset.py:299-319`) was rejected because per-message tokenization is only equal to whole-conversation tokenization for templates that happen to be concatenation-safe; the prefix method is exact for any template and asserts it. The mask TRL produces is the parity reference for the transform (§7).

**Packing.** `text_roles` joins the packing as one more per-token feature next to `text`, with `length_struct={"text": window, "text_roles": window}` and `padding_struct={"text": 0, "text_roles": 0}` in `FirstFitPackIterDataset` (`.worktrees/grain-packing/src/dew/data/dataloaders.py:782-791`). Segment ids still stop attention at document boundaries; the packed forward pass already accepts `segment_ids` and `positions` (`.worktrees/grain-packing/src/dew/nn/backbones/causal_transformer.py:138-197`).

**Batch contract for SFT** (a superset of the pretraining contract; the packed branch's keys are `TEXT_KEY`, `text_segment_ids`, `text_positions` at `.worktrees/grain-packing/src/dew/objectives/lm/objective.py:132-139`):

| key | shape | dtype | content |
| --- | --- | --- | --- |
| `text` | `[B, S+1]` | int32 | padded, packed conversation ids |
| `text_segment_ids` | `[B, S+1]` | int32 | document id per token, 0 on pad |
| `text_positions` | `[B, S+1]` | int32 | position inside each document |
| `text_roles` | `[B, S+1]` | int8 | role id per token, 0 on pad |

`loss_role` multiplies the existing weights by `(roles == loss_role)` on the target side (`weights * (text_roles[:, 1:] == ASSISTANT)`), composed with the segment-boundary weights already computed at `.worktrees/grain-packing/src/dew/objectives/lm/objective.py:112-121`. With no mask, every non-padding token counts, exactly as pretraining does today.

### 1.2 Preference pairs: chosen and rejected

`PreferenceParquetSource` over a parquet of TRL's preference shape: rows with `prompt` and `chosen`/`rejected`, each either plain text or a conversation (`trl/data_utils.py:213-224` lists the accepted key-sets, `/tmp/design/trl`). A transform builds two sequences per row, left-padded to a common length, plus a completion mask:

- prompt tokens: `tokenizer.apply_chat_template(prompt_messages, add_generation_prompt=True)`
- chosen/rejected: the same prompt rendering with each completion appended
- `completion_mask`: 0 over the prompt, 1 over the completion, 0 on pad; the completion includes its end-of-turn token.

**Batch contract for `DPOObjective`:**

| key | shape | dtype | content |
| --- | --- | --- | --- |
| `text` | `[2B, S]` | int32 | chosen rows `[0::2]`, rejected rows `[1::2]`, adjacent per pair |
| `text_segment_ids` | `[2B, S]` | int32 | 1 real, 0 pad (left-pad) |
| `text_positions` | `[2B, S]` | int32 | 0..len-1 within the real tokens, right-aligned |
| `completion_mask` | `[2B, S]` | int8 | 1 on completion tokens |

TRL concatenates chosen and rejected into one batch of `2B` and splits with `chunk(2, dim=0)` (`trl/trainer/dpo_trainer.py:1383`, `/tmp/design/trl`); verl-omni's online DPO lays them out adjacent and slices `[0::2]` / `[1::2]` (`verl_omni/trainer/diffusion/diffusion_algos.py:729-744`, `/tmp/design/verl-omni`). Adjacent interleaving is chosen (verl-omni's) so one row pair shares a prompt without a reordering step. Log-probs are the negative per-token cross entropy, which `chunked_cross_entropy` already returns per token (`.worktrees/lm-head/src/dew/objectives/lm/chunked.py:83-128`), summed over the completion (§6.2).

For images, `DiffusionDPOObjective` reads a pair-of-images record (`prompt`, `chosen` image path or array, `rejected` image path or array), encodes both through the AutoEncoder, and applies one shared timestep and one shared noise per pair (verl-omni draws one noise and timestep per pair and repeats them across chosen and rejected, `verl_omni/pipelines/utils.py:246-284` per the scout's citation). The online variant needs no pair dataset at all: the rollout samples G images per prompt, the reward scores them, and the top and bottom of each group become the pair (§6.5).

### 1.3 Prompts for online RL

`PromptParquetSource`: parquet of `prompt` (string or conversation) plus whatever non-tensor columns the reward wants. No masks: the rollout produces them.

**Batch contract for `GRPOObjective` (the input side of the seam):**

| key | shape | dtype | content |
| --- | --- | --- | --- |
| `prompt` | `[b, P]` | int32 | template-rendered, left-padded to P |
| `prompt_length` | `[b]` | int32 | real tokens per row |

**Batch contract out of the rollout (what `loss` consumes):**

| key | shape | dtype | content |
| --- | --- | --- | --- |
| `text` | `[b*G, P+N]` | int32 | prompt + generated continuation |
| `text_segment_ids` | `[b*G, P+N]` | int32 | 1 real, 0 pad |
| `text_positions` | `[b*G, P+N]` | int32 | 0-based within real tokens |
| `response_mask` | `[b*G, N]` | int8 | 1 on generated tokens before the first EOS, inclusive |
| `old_log_probs` | `[b*G, N]` | float32 | teacher-forced, from the same params that sampled |
| `advantages` | `[b*G]` | float32 | group-relative |
| `reward` | `[b*G]` | float32 | raw scores, for telemetry |
| `truncated` | `[b*G]` | bool | the continuation hit N without an EOS |

`response_mask` matches verl's (`attention_mask[:, -response_length:]`, `verl/trainer/ppo/ray_trainer.py:120-135`) and TRL's `completion_mask` built from the first EOS (`trl/trainer/grpo_trainer.py:2505-2512`, `/tmp/design/trl`). Fixed shapes throughout: `generate` runs for exactly `N` new tokens whatever it sees (`.sampling/text.py:28-48`), so every key above has one shape per run and the compiled step lowers once.

For diffusion the same tables change only in what "prompt" and "response" mean: `prompt` is conditioning (caption text through the CLIP encoder the diffusion recipe already builds, `recipes/diffusion/train.py:107-121`), and the response is a latent trajectory recorded by the SDE sampler (§6.6).

### 1.4 verl's parquet schema, mapped

verl's RL data files are parquet with five fields, written by a `make_map_fn` (`verl/docs/preparation/prepare_data.rst:72-96` and `verl/examples/data_preprocess/gsm8k.py:68-85`, `/tmp/design/verl`):

| verl field | verl content | Dew field |
| --- | --- | --- |
| `data_source` | dataset name, indexes the reward function | `data_source`, passed to the reward callable (§5) |
| `prompt` | chat messages, template applied at load (`verl/utils/dataset/rl_dataset.py:259-262`) | rendered by `ChatTokens` into `text`/`text_roles` |
| `reward_model` | `{"style": "rule", "ground_truth": str}` | `ground_truth`, passed to the reward callable |
| `extra_info` | bookkeeping (`index`, `split`, ...) | `extra_info`, passed to the reward callable |
| `ability` | task category | carried in `extra_info`; Dew does no dispatch on it |

verl's reward signature is `compute_score(data_source, solution_str, ground_truth, extra_info)` (`verl/utils/reward_score/__init__.py:19-28`, `/tmp/design/verl`); Dew's callable takes the same four things as fields of one record (§5). verl-data on GitHub is an empty placeholder; the schema above is read from the code, as briefed.

TRL dataset shapes map the same way: `{messages}` and `{prompt, completion}` are the SFT paths, `{prompt, chosen, rejected}` the preference path, `{prompt}` plus extra columns the RL path (`trl/docs/source/dataset_formats.md:409-423`, `/tmp/design/trl`).

## 2. Objectives: the six constructors, honestly

The one dishonest constructor in the owner's list is `LMObjective(model, mask="assistant")`: a loss mask is not a property of the model, and the data must know roles anyway. Two smaller options exist, and the smaller one wins:

| option | interface cost | gives up |
| --- | --- | --- |
| `mask="assistant"` string, objective parses roles | a stringly-typed knob with a hidden coupling to the batch carrying `text_roles` | nothing, but every caller must learn which strings exist |
| `loss_role="assistant"`, an int8 role id compared against the batch | same one field, typed, no string parsing; the batch contract carries roles for every path | cannot mask on arbitrary boolean masks supplied ad hoc |

Decision: `LMObjective(model, seq_len, ..., loss_role=ASSISTANT)`. It is the same class, one dataclass field, one `weights *` line. The objective refuses a batch that carries no `text_roles` when `loss_role` is set, naming the field, the way the trainer refuses a missing `input_shapes` (`src/dew/training/objective_trainer.py:111-120`).

`DPOObjective`, `GRPOObjective`, `DiffusionDPOObjective` and `FlowGRPOObjective` are new files under `src/dew/objectives/`, one each: `lm/dpo.py`, `lm/grpo.py`, `diffusion/dpo.py`, `diffusion/flow_grpo.py`. Nothing about them needs a registry entry beyond `Objective.tag`, which names their checkpoint artifacts (`src/dew/objectives/base.py:47`).

## 3. The frozen reference

**Decision: the reference is the EMA tree at unit decay.** A preference or RL objective sets `EMASpec(decay=lambda step: 1.0, path=())`; `loss` reads `ema_params` as the reference. This is the smallest mechanism that works, and it is the JEPA pattern taken to its limit: the target encoder already lives in `ema_params` under `stop_gradient` (`.worktrees/../src/dew/objectives/jepa/objective.py:7-9, 115-148`; `EMASpec.path` at `src/dew/objectives/base.py:21-29`).

What falls out for free:

- **Out of the optimizer.** Only `params` reaches `tx.init` (`src/dew/training/trainer.py:240-246`), so no masked optimizer is needed at all. The EMA update `1.0 * ema + 0.0 * param` returns exactly `ema` for finite params, so the tree never moves.
- **Checkpointing.** `ema_params` is saved, restored and sharded exactly like params (`SimpleTrainer.save` writes the whole state, `src/dew/training/trainer.py:329-351`; sharding is derived from the abstract state at `:218-227`). A resumed DPO run restores its reference with everything else, bit for bit.
- **Sharding.** It is a full tree in `TrainState`, so FSDP shards it like params.

What it costs, stated: a run cannot hold a moving EMA of the policy and a frozen reference at the same time. No objective in scope wants both (RL and DPO runs do not use a policy EMA; the frontier survey records no post-training recipe that does). The alternative considered, a masked subtree inside `params` (`optax.masked` over a `reference` key, generalizing JEPA's two-key tree at `.worktrees/../src/dew/objectives/jepa/objective.py:99-109`), was rejected: one extra full parameter copy in HBM, an optimizer wrapper, zero-valued gradient and update trees for the reference, and a new checkpoint layout. The EMA slot is already allocated and already checkpointed; using it costs nothing that is not already being paid.

- **No per-step work.** `_define_train_step` omits the EMA path entirely when the objective's decay schedule is identically 1.0 (`decay(0) == 1.0` and `decay(2**31 - 1) == 1.0`, a compile-time branch), so a reference costs one copy in HBM and nothing per step. The skip is also safer than the lerp: `1.0 * ema + 0.0 * param` returns `ema` for finite params, but a NaN reaching the params would poison the frozen tree through `0 * NaN`, and the skip cannot. The test that ships with it: params and gradients identical with and without the skip, and the reference untouched after a step whose params were forced non-finite.
- **The EMA clock composes unchanged** for every other decay: `apply_ema` still runs per completed update under `MultiSteps` (`src/dew/training/objective_trainer.py:269-287`).

**Precompute versus in-step.** In-step evaluation is the one path: `loss` runs the reference forward through the same code as the policy, per batch, no cache. TRL's `precompute_ref_log_probs` exists to free the reference model's memory during training (`trl/trainer/dpo_config.py:69-72`, `/tmp/design/trl`); under the EMA mechanism there is no separate model to free, so the option buys step time (one forward per step, roughly 1/3 of the policy's forward+backward cost) at the price of a stale-prone dataset cache that silently produces wrong losses if the data path changes. Not built; revisited only if a measured run shows reference forwards dominating the step. The memory arithmetic that would force this tradeoff elsewhere is in §8.

## 4. The rollout seam

One optional method on `Objective`, mirroring `loss`:

```python
class Objective:
    def rollout(self, params, ema_params, batch, rng, step):
        """Expand a batch of prompts by sampling. The default is the identity:
        an objective that trains on what the loader produced never samples."""
        return batch
```

The trainer calls it in `train_loop`, right after `batch = next(train_ds)` and before the compiled step, with a key split from the same `RandomMarkovState` stream the step key comes from, and reshards the result:

```python
batch = next(train_ds)
if objective rolls out:                      # the base default makes this false
    rng_state, rollout_key = rng_state.get_random_key()
    batch = shard_batch(self.batch_sharding,
                        objective.rollout(train_state.params, train_state.ema_params,
                                          batch, rollout_key, train_state.step))
train_state, loss, aux, rng_state, is_finite = compiled_step(
    train_state, rng_state, batch)
```

The mechanics fall out of what already exists:

- **Identity is free.** SFT and DPO never override `rollout`, so the trainer's branch is false for them and the loop is exactly today's (one `hasattr`-style check against the base method, like the check that `_compiled_step` runs once per function identity, `src/dew/training/trainer.py:405-422`).
- **Fixed shapes.** The rollout's output shapes are constants of the run (`b`, `G`, `P`, `N`), so `_compiled_train_step` still compiles once. jax's shape-polymorphism is not needed.
- **Grad accumulation.** One rollout per micro-batch: each micro-batch of `b` prompts expands to `b*G` rows, and a group never straddles a micro-batch. `MultiSteps` accumulates the micro-gradients and `loss` token-means within its own batch, exactly as SFT does under accumulation today (`src/dew/training/objective_trainer.py:106-109`). The group baseline is computed inside each micro-batch, which is the whole batch whenever `grad_accum_steps == 1`, the default.
- **A pure function of the batch.** The returned batch carries rewards, advantages, old log-probs and the response mask as ordinary named arrays (§1.3, §6.6), so `loss` stays a function of the batch alone and nothing about RL reaches the trainer. The deletion test holds: delete `rollout` and every RL objective loses its data source; nothing else changes.
- **Telemetry through the existing channels.** Reward mean and mean generation length ride the aux dict `loss` already returns, as `train/rollout_reward` and `train/rollout_gen_len` read from the batch's own columns (`src/dew/training/trainer.py:504-510` folds aux into `train/*`). Rollout seconds cannot come from the jitted aux, because there is no wall clock inside jit; the trainer times the host-side rollout call and merges `train/rollout_seconds` into the log tick the way `_throughput_metrics` already derives `train/step_time_ms` from host-side elapsed time (`src/dew/training/trainer.py:554-565`). No second return channel.
- **EMA and checkpoints.** The EMA clock is untouched (§3); the trainer's `save`, resume and `dataset_state` are untouched; the rollout's RNG comes from the checkpointed stream, so a resumed run samples different, correctly-streamed continuations rather than replaying.
- **Distributed.** The rollout runs per process on its slice of the global batch (the `DevicePrefetchIterator` hands each process its shard, `src/dew/training/distributed.py:78-126`), and `shard_batch` assembles the global array from process-local data, which is its documented job (`:72-76`). Rewards that need the whole group's samples per prompt are therefore computed per process, over that process's groups. This requires `data.batch_size` to be divisible by `jax.process_count() * G`, which the recipe states and asserts.

**Rejected alternatives:**

- **Rollouts inside `loss` under `stop_gradient`.** Generation inside the jitted step means the KV cache, sampling and reward all trace into the same executable as the gradient; a Python-loop reward cannot trace at all, and EOS-dependent masks become compile-time-constant shapes. It also couples the rollout to the donate-argument discipline of the step. verl, TRL and Tunix all keep generation outside the gradient step (verl's `generate_sequences` before `_compute_old_log_prob`, `verl/trainer/ppo/ray_trainer.py:1490-1544`; TRL's `_generate_and_score_completions` before `_compute_loss`, `trl/trainer/grpo_trainer.py:2343+`; Tunix's `RLEngine.generate` separate from `update_actor`, `tunix/rl/rl_cluster.py:793-880` and `:758-762`).
- **A separate RL trainer class.** Splits the trainer seam in two: a second mesh builder, checkpoint writer and EMA clock, none of which differ. The brief's acceptance bar is that the trainer does not change; a second trainer is the largest possible interface for the smallest possible difference. Tunix's learner/engine cluster split (three layers, `rl_cluster.py` facade over `PeftTrainer` and `BaseRollout`) exists to host vLLM and multi-mesh rollouts; Dew's rollout runs in the trainer process on the same devices, so the whole layer would be dead weight.
- **A data source that generates.** Hides sampling inside the loader: the loader then needs model params, the sampler config and the reward to produce a batch, which inverts the data layer's contract (`data sources produce records`; `CONTRIBUTING.md` seam list). It also cannot give `loss` fresh `old_log_probs` computed under the sampling params without reaching back into the trainer state.

## 5. Rewards

Rewards are plain callables from outputs to scores. One signature per modality:

```python
# text
def reward(record: dict, completion: str, completion_ids: np.ndarray) -> float
# images
def reward(record: dict, image: np.ndarray) -> float   # [H, W, 3] uint8
```

`record` is the parquet row verbatim, so a verifier reads `record["data_source"]` and `record["reward_model"]["ground_truth"]` exactly as verl's `compute_score(data_source, solution_str, ground_truth, extra_info)` does (`verl/utils/reward_score/__init__.py:19-28`, dispatched from `NaiveRewardManager.__call__` at `verl/workers/reward_manager/naive.py:118-136`, `/tmp/design/verl`). The shape is deliberately one argument-flatter than verl's: verl unpacks the four fields because they ride in a `DataProto`; Dew hands the record over and the callable reads what it needs.

**Where they run.** On the host, synchronously, inside the rollout, right after the device arrays are converted to numpy. This is verl's own default behaviour (`NaiveRewardManager` loops and decodes per sample on CPU), and the frontier's cheap verifiers (string match, Levenshtein, JPEG complexity) are host code anyway. A reward model is the same callable whose body calls a jitted device forward (a CLIP scorer is `dew.eval.images` plus a scorer head; the cached CLIP machinery already exists at `src/dew/eval/images.py:6-33`). Async reward workers (verl-omni's pool, `docs/algo/async_reward.md`) and HTTP scorer services are not built; they are vLLM-scale concerns, and the callable interface leaves room for them without a seam change.

Two rewards ship with the objectives, as examples that double as tests: a GSM8K-style `math_verifier` (string match against `ground_truth` after `####`) and an OCR scorer built on an OCR model, both named in the owner's snippet. JPEG compressibility (`verl_omni/utils/reward_score/jpeg_compressibility.py:52-60`) is the model-free image example. Nothing else; rewards are user code.

## 6. The four new objectives

Shared pure functions live in a small `src/dew/rl/` package, `advantage.py` and `surrogate.py` (§13 step 1): `group_advantage(rewards, group, *, std, ddof, eps)`, the clipped surrogate on fixed tensors, and the k3 KL estimator. These are ports, with parity tests, not shared abstractions invented here (§7); text and diffusion do not share one loss because their references disagree (§6.7).

### 6.1 SFT

`LMObjective` with `loss_role=ASSISTANT`, over the chat path (§1.1). The loss is the existing masked cross entropy with one more weight factor; the token-mean over the global batch falls out of the GSPMD-partitioned sum the same way it does for pretraining. TRL reduces the same way: sum over unmasked tokens divided by the global `num_items_in_batch` (`trl/trainer/sft_trainer.py:159-162, 223-229`, `/tmp/design/trl`).

### 6.2 `DPOObjective(model, pretrained, beta=0.1)`

Per pair, with `loss` over the `[2B, S]` batch (§1.2):

- policy per-token log-probs: `-chunked_cross_entropy(...)` under `params`
- reference per-token log-probs: the same under `ema_params`, no gradient
- `logps = sum(per_token_logps * completion_mask)` per row (TRL sums over the completion, `trl/trainer/dpo_trainer.py:1368-1371`, `/tmp/design/trl`)
- `delta = (chosen_logps - ref_chosen_logps) - (rejected_logps - ref_rejected_logps)`
- loss `= -mean(log_sigmoid(beta * delta))` (`trl/trainer/dpo_trainer.py:1455-1456`)

This is Rafailov et al. 2023 equation 7 (`arXiv:2305.18290v3`, §4: $\mathcal{L}_{\text{DPO}} = -\mathbb{E}[\log\sigma(\beta\log\frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta\log\frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)})]$), with TRL's current `sigmoid` type as the code reference. Label smoothing, `ipo`, `ld_alpha`, f-divergence variants and `rpo_alpha` are not built; TRL's own list is at `trl/trainer/dpo_config.py:80-87` and the owner's surface has none of them. Telemetry: `train/reward_chosen`, `train/reward_rejected`, `train/reward_margin`, `train/dpo_accuracy` (TRL logs the same four, `dpo_trainer.py:1654-1670`).

The chunked head is what makes the 0.6B numbers fit §8: log-probs are per-token cross entropies, which the chunked path already computes without materializing `[tokens, vocab]` (`.worktrees/lm-head/src/dew/objectives/lm/chunked.py:83-128`).

### 6.3 `GRPOObjective(model, pretrained, reward, group=8)`

`rollout` (§4):

1. `generate(model, params, prompt, N, rng=rng, temperature=...)` per prompt, G times (one call with `b*G` rows; the KV cache is allocated at `max_seq_len` per row, `.sampling/text.py:30-33` and `src/dew/nn/attention.py:50-79`).
2. Teacher-forced `old_log_probs` from the same params, through the same chunked head the loss uses. The rollout samples in decode mode and the loss scores a full-sequence forward, and the two are not bitwise equal (cache dtype, kernel reductions); scoring old log-probs in the training code path is the GLM determinism rule from the frontier note (train/rollout log-prob agreement; `docs/research/frontier-training.md:203`), and it makes the ratio honest rather than assumed-1.
3. Host: `response_mask` from the first EOS, `truncated` flags, rewards via the callable, `group_advantage` with per-group mean and std, `ddof=1`, `eps=1e-6`.

`loss`: the clipped surrogate on `log_probs - old_log_probs`, token-mean over `response_mask`, plus `beta * k3` KL against `ema_params` when `beta != 0`. Constants match the reference per test (§7): verl's advantage (group mean, group std, eps 1e-6, `verl/trainer/ppo/core_algos.py:267-331`), verl's vanilla loss (symmetric clip plus dual-clip `clip_ratio_c=3.0`, `negative_approx_kl` clamped to ±20, `token-mean` aggregation, `core_algos.py:1285-1376`), verl's k3 (`kl_penalty_forward`, `core_algos.py:2239-2245`). One update per rollout batch (`mu = 1`): `old_log_probs` comes from the batch, so raising `mu` later is a trainer knob, not a contract change. GRPO's own paper is sequence-mean-then-token-mean (`arXiv:2402.03300v3` §4 equation 3); verl's `token-mean` default is what Dew ports and states.

`mask_truncated_completions` (TRL zeroes the mask of completions that never hit EOS, `trl/trainer/grpo_trainer.py:2505-2512`) is a flag, default off, because it changes the estimator and belongs to the run config, not the code path.

### 6.4 The PPO variant, named

PPO is `GRPOObjective` with GAE advantages instead of group-relative ones, which requires a value head. Both are out until a run needs them (§10); the batch contract already carries per-token rewards and advantages, so a value head is a model change plus an advantage function, not a new seam.

### 6.5 `DiffusionDPOObjective(denoiser, ..., pretrained, beta)`

Per pair, one shared `t` and one shared `noise`:

- `x_t` for chosen and rejected from the existing `forward_diffusion` (`src/dew/diffusion/transforms.py:27-32`)
- `target = noise - x0` (the flow velocity, `FlowMatchPredictionTransform.get_target`, `transforms.py:105-106`)
- `model_err = mean((v_theta(x_t) - target)^2)` per sample, `ref_err` the same under `ema_params`
- `inside = -0.5 * beta * ((model_err_w - ref_err_w) - (model_err_l - ref_err_l))`, loss `= -mean(log_sigmoid(inside))`

This is verl-omni's `DPOLoss.compute_loss` line for line (`verl_omni/trainer/diffusion/diffusion_algos.py:734-747`, `/tmp/design/verl-omni`), which is Wallace et al. 2023 equation 14 with the length normalization folded into the per-sample mean over latent dims (`arXiv:2311.12908v1`, §4: $-\log\sigma(-\beta T\omega(\lambda_t)(\|\epsilon^w-\epsilon_\theta\|_2^2-\|\epsilon^w-\epsilon_{\text{ref}}\|_2^2-(\|\epsilon^l-\epsilon_\theta\|_2^2-\|\epsilon^l-\epsilon_{\text{ref}}\|_2^2))$). The paper's `T` counts an expectation over timesteps; one shared `t` per pair per step is the verl-omni sampling of it. `precompute_ref_log_probs`-style caching does not apply: the reference forward needs the policy's own noise draw, so it is in-step by construction.

### 6.6 `FlowGRPOObjective(denoiser, ..., sampler, reward, group=8)`

The SDE rollout is a `DiffusionSampler` subclass, `SDERolloutSampler`, in `src/dew/sampling/sde.py`. Dew's sampler seam can produce everything FlowGRPO needs; one thing is missing and two are additions:

- `take_next_step` receives `reconstructed_samples` and `pred_noise`, i.e. the model output after the prediction transform, not the raw velocity (`src/dew/sampling/common.py:149-163`). For the flow preset the transform is the identity on the output (`.worktrees/../src/dew/diffusion/transforms.py:95-106`), so the raw velocity is available as `eps - x0` or by passing it through; **the gap**: `take_next_step` never sees it, and the SDE mean is defined on it. Fix: thread `model_output` into `take_next_step` alongside `reconstructed_samples` and `pred_noise`. This touches every sampler's signature (the files in `src/dew/sampling/`), mechanically, because the base class signature changes once.
- The step itself, from the FlowGRPO paper (equation 8; `arXiv:2505.05470`): with flow convention `x_t = (1-t)x_0 + t*x_1` and Dew's rates `(alpha, sigma) = (1-t, t)` (`src/dew/diffusion/schedules/flow.py:45-47`), the SDE transition is

  `std_dev_t = sqrt(sigma / (1 - sigma)) * noise_level` (guarding `sigma == 1` with the schedule's second sigma, as the reference does)

  `mean = x_t * (1 + std_dev_t^2 / (2*sigma) * dt) + v * (1 + std_dev_t^2 * (1 - sigma) / (2*sigma)) * dt`, with `dt = sigma_next - sigma`

  `x_next = mean + std_dev_t * sqrt(-dt) * eps`

  This is verbatim `FlowMatchSDEDiscreteScheduler.sample_previous_step`'s `sde_type="sde"` branch (`verl_omni/pipelines/schedulers/flow_match_sde.py:214-243`, `/tmp/design/verl-omni`), with sigma playing t. Dew's `FlowMatchingScheduler.get_rates` returns `(1-t, t)` directly (`flow.py:45-47`), so `sigma` is the noise rate and no convention change is needed.
- The per-step log-prob is the Gaussian density of `x_next` under `(mean, std_dev_t*sqrt(-dt))`, **mean-pooled over all non-batch dimensions** (the reference reduces with `.mean(dim=tuple(range(1, ndim)))`, `flow_match_sde.py:312-313`; verl-omni's own docs note diffusion log-probs are mean-pooled, `docs/algo/rollout_correction.md`). The normalizer constants are included, as the reference's default does.
- `SDERolloutSampler.take_next_step` computes the step from the shared distribution and returns it; a pure `step_distribution(x_t, v, t, t_next)` function beside it is what the rollout records and the loss recomputes under gradient. This is the split verl-omni enforces with `prev_sample=...` (replaying a trajectory to re-evaluate log-probs, `flow_match_sde.py:179-181`): one function defines the transition, both sides call it.

**The rollout** (the objective's `rollout`):

1. G trajectories per prompt from `x_1 ~ N(0, I)` (the sampler's `_get_initial_samples`, `common.py:373-381`), through the SDE step above, recording `(x_t, x_next, t, t_next, old_log_prob)` for the steps inside a randomly placed window of `sde_window_size` steps (verl-omni samples the window start per row, `verl_omni/pipelines/request_batch.py:32-53`; outside the window the noise level is 0 and the step is the plain Euler ODE). This is FlowGRPO's denoising reduction (paper §3.3): inference uses all steps, training scores a window.
2. Decode through the VAE (`post_process`, `common.py:112-118`) to uint8 images, reward on the host (§5).
3. `group_advantage(rewards, G)` with verl-omni's constants (eps 1e-4, `norm_adv_by_std` on; `global_std` is a flag: verl-omni's default normalizes by the batch-wide std, `diffusion_algos.py:192-267`).
4. Flatten the window steps into rows: the batch is `[b*G*W, ...]` with `advantages` broadcast per trajectory (verl-omni expands the scalar reward across the window, `ray_diffusion_trainer.py:1371-1374`).

**The loss**, per row: recompute `v_theta(x_t)` under `params`, rebuild `mean_theta` and `log_prob_theta` from the recorded `(x_t, x_next, t, t_next)`, then the clipped surrogate with `adv_clip_max=5.0`, `clip_ratio=1e-4` (verl-omni's defaults, `verl_omni/workers/config/diffusion/actor.py:39-47`), flat mean over rows; plus `kl_loss_coef * ||mean_theta - mean_ref||^2 / (2*std_dev_t^2)` against `ema_params` when enabled (the `KLLoss` formula, `diffusion_algos.py:1013-1037`). Note the reference's KL divides by `2*std_dev_t^2` without the `dt` the paper's equation 5 carries ($\frac{\Delta t}{2}(\ldots)^2$); the port matches the code, which is the parity target, and the difference is recorded here.

**Batch contract** (what the rollout hands `loss`):

| key | shape | dtype | content |
| --- | --- | --- | --- |
| `latents` | `[b*G, W+1, ...]` | float32 | the trajectory's window states |
| `timesteps` | `[b*G, W]` | float32 | sigma values of the window steps |
| `old_log_probs` | `[b*G, W]` | float32 | recorded at sampling time |
| `advantages` | `[b*G]` | float32 | group-relative, broadcast per row |
| `reward` | `[b*G]` | float32 | raw scores |
| conditioning | per condition | | the CLIP context the sampler already builds |

(verl-omni's names are `trajectory_latents`, `trajectory_timesteps`, `trajectory_log_probs`, `diffusion_rollout_output.py:36-81`; Dew keeps its own flat `text`/`latents` vocabulary.)

### 6.7 Where the maths coincides, and where it cannot

| quantity | text | flow | shared? |
| --- | --- | --- | --- |
| group advantage | per prompt over G completions | per prompt over G trajectories | yes: `group_advantage(rewards, G, std=..., ddof=..., eps=...)`; only the constants differ (verl 1e-6, verl-omni 1e-4, TRL 1e-4), so they are arguments |
| importance ratio | per token | per step, mean-pooled over latent dims | no: token masks versus pooled log-densities |
| clipping | symmetric + dual-clip 3.0, `token-mean` | symmetric 1e-4, advantage-clamped, flat mean | no: same shape, different constants and reductions; two functions |
| KL to reference | k3 on token log-probs | Gaussian mean-matching $\|\mu_\theta-\mu_{\text{ref}}\|^2/2\sigma^2$ | no: the reference policies are different objects |
| reference | `ema_params`, teacher-forced | `ema_params`, same forward code as the sampler | yes: one mechanism (§3) |

The shared module holds the one function that is genuinely shared; everything else is per-modality code next to its objective, because a shared "policy loss" abstraction would have to be parameterized by all four rows of the table, which is a bigger interface than the two implementations.

## 7. Parity

Every new loss is a port, so every new loss ships with a parity test against the reference at fp32 on fixed tensors, with the tolerance and the largest observed difference written in the test (`CONTRIBUTING.md`, reference parity). The fixture generators are committed under `tools/`.

| port | reference | compared quantity | tolerance |
| --- | --- | --- | --- |
| SFT masked loss | TRL `SFTTrainer` on the same messages | (a) the assistant mask, position for position, against `return_assistant_tokens_mask` on a template that has `{% generation %}` markers; (b) the masked token-mean loss on the same batch | masks exactly equal; loss max abs diff ≤ 1e-6 |
| DPO loss | TRL `DPOTrainer._compute_loss` (sigmoid, no smoothing) | loss and `chosen/rejected/margin` rewards on the same pairs with the same reference log-probs (the reference forward run once in the fixture) | max abs diff ≤ 1e-6 |
| GRPO advantage | verl `compute_grpo_outcome_advantage` | advantages on fixed rewards, `response_mask` and `uid` groups, both std settings | max abs diff ≤ 1e-7 (the eps is 1e-6; differences come only from summation order) |
| GRPO loss | verl `compute_policy_loss_vanilla` + `agg_loss(token-mean)` + k3 `kl_penalty_forward` | the loss, `pg_clipfrac`, `ppo_kl` on fixed `(log_prob, old_log_prob, advantages, response_mask)` | max abs diff ≤ 1e-6 |
| group advantage (JAX) | Tunix `compute_advantages` | advantages on fixed rewards, ddof=1 | max abs diff ≤ 1e-7 |
| diffusion DPO | verl-omni `DPOLoss.compute_loss` | loss and implicit rewards on fixed `(noise, latent, noise_pred, ref_noise_pred)` | max abs diff ≤ 1e-6 |
| FlowGRPO advantage | verl-omni `compute_flow_grpo_outcome_advantage` | advantages on fixed rewards and uids, both `global_std` settings | max abs diff ≤ 1e-6 |
| FlowGRPO SDE step and log-prob | verl-omni `FlowMatchSDEDiscreteScheduler.sample_previous_step` | `x_next`, `log_prob`, `mean`, `std_dev_t` on fixed `(x_t, v, sigma, sigma_prev, eps)`, including the sigma=1 guard | max abs diff ≤ 1e-6 (fp32, same formulas) |
| FlowGRPO loss | verl-omni `FlowGRPOLoss.compute_loss` + `KLLoss` | the loss and `pg_clipfrac` on fixed tensors | max abs diff ≤ 1e-6 |

Tolerances are starting values; each test records the largest observed difference and tightens to it. The fixtures are small fixed tensors plus, for the SFT mask test, one real tokenizer/template pair (the qwen3-tiny fixture from `.worktrees/hf-decoders/tests/fixtures/hf/` serves). Generators under `tools/`: one per reference, named `parity_<name>.py`, importing nothing from Dew except the fixture path convention, writing `.npz` files under `tests/fixtures/parity/`.

A second, separate test class holds the end-to-end guarantees, each with the mutation that would break it (a dropped term, a flipped comparison): the reference never moves under training (gradients are exactly zero through the reference branch); `rollout`'s identity default leaves the batch untouched; a group of constant rewards yields exactly zero advantage; the loss of a batch where chosen equals rejected is `log(2)`; the SDE step at `noise_level=0` equals the Euler step; a DPO run and an SFT run on the same data produce different checkpoints.

## 8. Memory arithmetic

Numbers for the two briefed models. Qwen3-0.6B from its config (`.worktrees/hf-decoders/tests/fixtures/hf/qwen3-0.6b/config.json`: hidden 1024, 28 layers, 16 heads, 8 KV heads, head_dim 128, vocab 151936, tied embeddings) gives 0.596B parameters, 2.22 GiB in fp32. The 7B stand-in is Llama-2-7B (4096/32/32, untied 32000-vocab embeddings, 512 MiB/token KV): 6.74B parameters, 25.1 GiB fp32. Dew's trainer state is params + Adam first and second moments + `ema_params`, all fp32, plus a gradient tree and (under `MultiSteps`) an accumulator copy.

| | Qwen3-0.6B, 16 GiB card | Qwen3-0.6B, 80 GiB card | Llama-2-7B, 80 GiB card |
| --- | --- | --- | --- |
| state (params + m + v + ema=reference) | 8.9 GiB | 8.9 GiB | 100.4 GiB |
| + grads (transient) | 11.1 GiB | 11.1 GiB | 125.5 GiB |
| + MultiSteps accumulator (grad_accum > 1) | 13.3 GiB | 13.3 GiB | 150.6 GiB |
| policy activations, bf16, 2B rows x 1024 tokens | ~1-2 GiB | ~1-2 GiB | ~4-8 GiB |
| fits full fine-tune | yes, tightly; grad accumulation off or small | yes, comfortably | **no**, not even without the accumulator |
| reference at in-step | 0 (the ema tree is the reference and is already in the state) | 0 | 0 |
| reference as precomputed log-probs | saves nothing in state; saves ~1 forward/step | same | saves ~1 forward/step, which does not fix the 100 GiB state |

Readings:

- For 0.6B on 16 GiB, full-parameter DPO/GRPO fits with the reference in-step and small batches. This is why the precompute option is not needed (§3): there is no memory to save.
- For 7B, the blocker is the fp32 state, not the reference. FSDP over 2 devices (100.4 GiB state / 2, plus activations and grads) is the first configuration that fits an 80 GiB device; over 4 it is comfortable. This is the existing trainer's `fsdp_size` knob, unchanged. LoRA, which is how verl-omni runs its Qwen-Image recipes (LoRA rank 64, `docs/algo/diffusion_dpo.md:150-151`), is out of scope for Dew until a run needs it, and it composes with the same reference mechanism (a reference tree that differs from the policy by the adapters).
- The rollout's KV cache is a separate budget, and it is the sharper one. The cache is allocated at `max_seq_len` per row in the model's dtype (`src/dew/nn/attention.py:73-79`), so `b*G` rows of 2048 tokens at 0.6B cost `64 * 2048 * 112 KiB = 14 GiB` in bf16. The rollout model must be built with `max_seq_len = P + N` (the LM recipe already sizes the cache from the sampling budget, `recipes/lm/train.py:76-86`), and `b*G` is the knob that trades it. The same numbers for 7B: 64 rows of 2048 is 64 GiB; the rollout batches shrink accordingly.

## 9. `pretrained=` and the one change to the surface

`LMObjective` on wave/hf-decoders already takes `pretrained=<variables>` and returns them from `init_params` (`.worktrees/hf-decoders/src/dew/objectives/lm/objective.py`, diff `@@ -80,6 +87`). The four new objectives take the same argument, and `init_params` returns it. The owner's `reference=pretrained` is therefore realized as: **the reference is the policy's initial weights**, held frozen in the EMA slot.

Honesty note, stated rather than hidden: the argument does not name a second model distinct from the starting point, because the mechanism holds one extra tree, and under §3 that tree is the initial weights. This covers TRL's default (reference = SFT model, `trl/trainer/dpo_trainer.py:1390-1403`), verl's default (a frozen copy of the actor's initial checkpoint), verl-omni's LoRA reference (adapters disabled = the base weights the run started from, `lora_adapter_mixin.py:172-178`), and Tunix's `ref_model` (a plain frozen `nnx.Module`, `tunix/sft/dpo/dpo_trainer.py:216-229`). It does not cover a reference that differs from the starting policy (iterative DPO round 2 referencing round 1 while starting from a further-trained policy). When a run needs that, the mechanism extends by seeding `ema_params` from a second tree in `generate_states` (one `TrainState.create(ema_params=reference_tree)` instead of `ema_params=params`), which is a contained change to one call site; it is not built now.

## 10. Rollout engine: Dew's own, and the crossover

Research scale uses Dew's `generate` and the `DiffusionSampler`. The crossover to verl/vLLM is stated honestly:

- `generate` decodes one token per `lax.scan` step over the full batch with a fixed-size cache (`.sampling/text.py:28-48`); it has no continuous batching, no paged KV and no prefix sharing. Its cost per decode step is the weights plus the cache read, so throughput falls as the batch shrinks, and a long prompt pays its cache every step.
- vLLM buys continuous batching, paged attention and prefix caching; verl buys a colocated engine, sleep/wake memory management and resharding. Those matter when the model needs tensor parallelism (Dew's mesh has no TP axis and the frontier note records that it should not grow one, `docs/research/frontier-training.md:208`), or when rollout and training are pipelined across separate device pools, which is Tunix's `rl_cluster` layer (`tunix/rl/rl_cluster.py:257-393`).
- The honest trigger is a measured one, not a parameter count: when the rollout's wall time dominates the gradient step on the hardware a run actually uses, measured by `train/step_time_ms` and a rollout timer in the aux metrics, the run has crossed. At Dew's scale (0.6B on one 16 GiB card, 7B on a few 80 GiB cards) the rollout of a research batch is minutes, and `generate`'s scan loop is a small fraction of it.
- The experiment that decides it, named: the first 0.6B GRPO run reports `train/rollout_seconds` beside `train/step_time_ms` (§4). A ratio above about 3 justifies a decoupled sampler process, which is a new seam between the trainer and a serving engine, not a change to `Objective.rollout`; below it, the synchronous host rollout stands.

The interop path already exists and is the whole story: `save_pretrained_decoder` writes an HF layout Dew-trained weights (`.worktrees/hf-decoders/src/dew/interop/hf_decoders.py:446-481`), verl consumes HF checkpoints, and the parquet schemas in §1.4 are the data interchange both ways. Nothing in Dew wraps, embeds or imports verl, TRL or verl-omni; they are read as parity references and used as parallel projects at a different layer. Weights come back through `load_pretrained_decoder`.

## 11. Tunix: what to take, and why it is not a dependency

Take, as Apache-2.0 reference code (the whole of `tunix/`, `LICENSE:1-4`):

- The GRPO advantage estimator (`tunix/rl/algo_core.py:640-657`): the only pure-JAX, `numpy`-on-rewards implementation among the three references, so it is the cross-check that Dew's JAX port agrees with a second implementation, not just with PyTorch ports. Its constants (ddof=1, eps 1e-6) match verl's.
- `compute_kl_divergence` with the three estimators (`tunix/rl/common.py:140-190`), `selective_log_softmax` (`common.py:192-209`) and the loss-aggregation family (`common.py:848+`), as the reference for the same pieces in JAX.
- The separation pattern: rollout behind a narrow `BaseRollout` interface (`tunix/rl/rollout/base_rollout.py:69-107`), gradient step in a generic trainer, learner above both. Dew's `Objective.rollout` is this shape with the cluster deleted, because Dew's rollout runs in-process on the training devices.

Not a dependency, for the reasons already on record (`docs/research/google-jax-stack.md:246-268`): the model interface is NNX end to end (`nnx.Module` constructor args, `nnx.value_and_grad` with `nnx.DiffState` at `tunix/sft/peft_trainer.py:347-351, 527-530`; `nnx.split`/`merge` at `tunix/rl/algo_core.py:414`), so a Linen `CausalTransformer` cannot be passed to it, and PyPI is a 0.0.0 placeholder. Tunix's own SFT path has no chat-mask logic to lift anyway; the only role-mask code in the repo lives in its agentic parser (`tunix/rl/agentic/parser/chat_template_parser/parser.py:167-168`).

## 12. Recipes

One recipe file per objective is wrong: it multiplies entry points and re-describes the trainer knobs six times. The existing per-modality recipes grow an objective choice, and the config tree stays flat because tyro renders nested dataclasses as dotted flags (`recipes/lm/train.py` already subclasses `RunConfig` into `LmRunConfig`, `src/dew/config/__init__.py:151-167`):

```python
@dataclass(frozen=True)
class LmRunConfig(RunConfig):
    objective: Literal["lm", "sft", "dpo", "grpo"] = "lm"
    pretrained: Optional[str] = None        # already on wave/hf-decoders
    dpo: DpoConfig = field(default_factory=DpoConfig)      # --dpo.beta
    grpo: GrpoConfig = field(default_factory=GrpoConfig)   # --grpo.group --grpo.max-new-tokens ...
```

Stages chain through artifacts, not orchestration, and the stage contract is two flags that both accept both artifact shapes: `--pretrained` accepts an HF-layout directory (`model.safetensors` plus `config.json`, what `load_pretrained_decoder` reads) or a Dew checkpoint step directory (what `SimpleTrainer.load` reads, `src/dew/training/trainer.py:297-327`), dispatched on what the directory contains, refusing with an error that names both expected shapes, the way `_token_dataset_dir` already dispatches on a directory's contents (`src/dew/data/dataloaders.py:805-817`); `--trainer.load-from-checkpoint` names a Dew checkpoint directory, and an SFT run hands off to GRPO through either. Pretrain ends in a checkpoint or an HF export; DPO and GRPO start from the SFT run's artifact the same way. Provenance is the wandb run config (`run_config` already records the full tree, `recipes/lm/train.py:180-188`) plus the pretrained path. A stage orchestrator is explicitly out of scope; it belongs to the layer above recipes.

## 13. Build order

Each step is one reviewed branch, smallest-first, and nothing is built that a step after it does not need.

| # | branch | contents | why this order |
| --- | --- | --- | --- |
| 1 | `dew/rl/advantage.py`, `dew/rl/surrogate.py` + fixtures | `group_advantage`, the clipped surrogate, k3 KL, and the Tunix/verl parity fixtures and tests, with no objective touching them yet; the first commits of the `dew/rl/` package the RL brief lays out, one module per function, never a `rl.py` monolith | pure functions, the riskiest numerics, zero integration surface; every later step depends on these being right |
| 2 | chat data path | `ChatParquetSource`, `ChatTokens` (prefix-diff mask), packing with `text_roles`, `loss_role` in `LMObjective` | SFT is the cheapest objective and the first consumer of both the data path and the reference-free seam changes; it lands on the grain-packing branch once that merges |
| 3 | preference data path + `DPOObjective` | `PreferenceParquetSource`, the adjacent-pair batch, the objective with the EMA reference | exercises the frozen-reference mechanism on a pure-gradient objective, no rollout |
| 4 | the rollout seam | `Objective.rollout`, the trainer's three-line call, `generate`'s left-pad mask | the trainer's only change; landed with GRPO's needs visible but only an identity default and a test |
| 5 | `GRPOObjective` | prompts data path, the rollout, teacher-forced old log-probs, the objective | the first online objective; everything it needs exists after 1-4 |
| 6 | diffusion DPO | `DiffusionDPOObjective` + pair data path | the flow reference mechanism on the mature diffusion side, before the SDE work |
| 7 | SDE sampler | `SDERolloutSampler`, `step_distribution`, threading `model_output` through `take_next_step` | the one change to existing sampler files, in its own branch with its own parity test |
| 8 | `FlowGRPOObjective` | the rollout, the recorded-window batch, the objective | last: everything it consumes landed in 1, 4, 7 |

Not built until a run needs it, with the trigger named: PPO value heads and GAE (trigger: a task where group-relative baselines demonstrably fail), a reward-model trainer (trigger: a preference dataset that needs one; Dew then trains it with `LMObjective` on pairs, which is SFT), multi-turn agents and tool use (trigger: an environment that produces the transcripts; then the chat path already parses them), LoRA (trigger: the 7B state arithmetic of §8 biting a real run), async reward workers (trigger: reward wall time dominating steps, §10's measured crossover), and `mu > 1` (trigger: a run that shows one-pass updates underusing a rollout).

## Appendix A: the concepts outline

Carried into `docs/concepts/post_training.md` when the code lands, not before: `CONTRIBUTING.md` says the docs describe what the code does today, and this file is the design, not the feature. Headings, and the code block the page opens with, are the owner's six constructors (the block at the top of this file).

- **One trainer, more objectives**: the six constructors; the seams unchanged.
- **Supervised fine-tuning is pretraining with a mask**: the chat data path; roles; why the mask includes the stop token.
- **Preferences without a reward model**: DPO; the reference as the starting point; what beta means.
- **The frozen reference**: where it lives, why it never moves, what a run gives up.
- **Online RL from verifiable rewards**: GRPO; groups; why the rollout runs outside the step; rewards as callables.
- **Diffusion and flow post-training**: the same three mechanisms; the SDE sampler; what the ODE gives up to gain exploration.
- **When to leave Dew for verl**: artifact-level interop both ways; the measured crossover.

## Appendix B: reference commits

All read as clones under `/tmp/design/` on 2026-09-02: verl `896a9bb`, TRL `8397289`, Tunix `b9f5e65`, verl-omni `9ab544d`. Papers: DPO `arXiv:2305.18290v3`, DeepSeekMath/GRPO `arXiv:2402.03300v3`, DeepSeek-R1 `arXiv:2501.12948v1`, Diffusion-DPO `arXiv:2311.12908v1`, FlowGRPO `arXiv:2505.05470v4`.
