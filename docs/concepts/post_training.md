# Post-training

Post-training changes how a model behaves after pretraining. In supervised fine-tuning (SFT), you supply example answers. In direct preference optimization (DPO), you supply a preferred and a rejected answer to the same prompt. In group-relative policy optimization (GRPO), the language model generates answers and your reward function scores them. Proximal policy optimization (PPO) also learns a critic that estimates future rewards. Flow-GRPO scores samples from a rectified-flow model.

Dew runs all of these objectives on the same `Trainer`. Their batches carry different kinds of supervision. Read [language models](language_models.md) for next-token prediction and [objectives](objectives.md) for how models, objectives and the trainer fit together. The first full example below trains a tiny DPO model without downloading a tokenizer, a dataset or a pretrained checkpoint.

## SFT: learn from assistant answers

An SFT conversation is a list of messages, each with a role. `ChatMessages` reads conversations from a Parquet file, a `.jsonl` file or a Hub dataset id. Each row holds one list of messages in the column named by `column`, which defaults to `messages`. A row that has a `prompt` column instead, as in verl's layout, is read from there, and tool schemas go in a `tools` column where a row has any. This shows one row in the verl layout; it is not a script:

```text
prompt = [
    {"role": "user", "content": "What is two plus two?"},
    {"role": "assistant", "content": "Four."}
]
```

Give `ChatMessages` your tokenizer (a Hub name or local path) in `tokenizer`, your conversations in `path` and the prediction length in `seq_len`. For a Hub dataset, `split` picks the split and `options` takes the same `HFOptions` the `hf` provider passes to `datasets.load_dataset`. The tokenizer must have a chat template, which is the rule for turning message boundaries, role headers and content into tokens. Each model is trained on its own format. If you join the message strings yourself, you can change both the input and which tokens count toward the loss.

Dew renders the conversation one prefix at a time to give each token a role. For an assistant turn, it leaves the generation header out of the assistant span. It checks that each tokenized prefix matches the start of the longer render, and raises if the template changes earlier tokens. This check does not guarantee correct assistant masks for every chat template or string delimiter. Look at the rendered tokens and roles for some typical conversations before a real run. Conversations must start with a system or user message. Dew rejects one that starts with an assistant message.

`ChatMessages.load(batch=B)` packs conversations into these arrays:

| Field | Shape | Meaning |
| --- | --- | --- |
| `text` | `[B, L + 1]` | Token IDs, including the extra next-token target. |
| `text_roles` | `[B, L + 1]` | `Role` value at each token. |
| `text_segment_ids` | `[B, L + 1]` | Document identity; separates packed conversations. |
| `text_positions` | `[B, L + 1]` | Position within each packed document. |

Here `L` is `ChatMessages.seq_len`. Use `LMObjective(model, L, loss_role=Role.ASSISTANT)`, with `Role` imported from `dew.data.chat`. The objective shifts IDs and roles together: input position `i` predicts token `i + 1`, and the role of the target decides whether that prediction counts. It also skips padding and the transitions between packed documents. With `loss_role` set, a batch without `text_roles` raises. Without `loss_role`, the loss is not limited to assistant targets.

`val_path` can name a separate file of conversations. Keep evaluation conversations out of the training file. See [evaluation](../guides/evaluation.md) for what token metrics measure.

## DPO: learn from preference pairs

DPO compares how much the policy prefers one answer over the other with how much a fixed reference policy does. The policy is the model you update. The reference is a snapshot of its starting parameters. For each prompt, the chosen and rejected sequences hold the prompt followed by their own completion. Completion masks mark answer tokens with 1 and prompt tokens with 0.

`PreferencePairs` takes either a Parquet `path` or a tuple of JSON strings in `records`, not both. Each row has `chosen`, `rejected`, `chosen_mask` and `rejected_mask`. Each mask has the same length as its ID list. If you leave out a mask, Dew treats every token as completion. It does not guess a boundary from the text, so always give masks for prompt-and-answer data.

A loaded batch holds `input_ids` and `completion_mask`, both shaped `[B, 2, S]`. Index 0 of the middle axis is the chosen sequence and index 1 the rejected one. Dew right-pads shorter rows to `S` and gives padding a mask weight of zero. Rows that are too long raise an error. `PreferencePairs.seq_len` is the full row width `S`, so use `DPOObjective(model, seq_len=S - 1)`.

The objective sums the next-token log-probabilities over each completion and applies the log-sigmoid preference loss. `beta` must be positive. It scales the comparison between policy and reference. Validation measures the perplexity of the chosen answers under the policy. That number alone does not measure how often the preferred answer wins or how good the responses are.

### Run a complete offline DPO example

Run this block in a fresh Python process after [installing Dew](../installation.md). All inputs are in memory. The made-up vocabulary has eight tokens, just enough to show how pairs are built and optimized. IDs 1 and 2 (or 1 and 6) form the prompt, 3 is the preferred answer, 4 is the rejected answer and 5 ends the answer. The end token counts toward the completion loss. Repeating the two pairs gives a batch of eight, which also divides across eight local devices.

```python
import json

import jax
import numpy as np
import optax

from dew import Trainer, models
from dew.data import Loading, PreferencePairs
from dew.objectives.rl import DPOObjective

rows = [
    {
        "chosen": [1, 2, 3, 5],
        "rejected": [1, 2, 4, 5],
        "chosen_mask": [0, 0, 1, 1],
        "rejected_mask": [0, 0, 1, 1],
    },
    {
        "chosen": [1, 6, 3, 5],
        "rejected": [1, 6, 4, 5],
        "chosen_mask": [0, 0, 1, 1],
        "rejected_mask": [0, 0, 1, 1],
    },
]
row_width = 4
spec = PreferencePairs(
    records=tuple(json.dumps(row) for row in rows * 4),
    seq_len=row_width,
    pad_id=0,
    loading=Loading(workers=0, threads=1, read_buffer=2),
)
data = spec.load(batch=8)
model = models.build(
    "causal_transformer",
    vocab_size=8,
    emb_features=16,
    num_layers=1,
    num_heads=2,
    mlp_features=32,
    max_seq_len=row_width,
)
objective = DPOObjective(model, seq_len=row_width - 1, beta=0.1)
trainer = Trainer(objective, optax.adam(1e-3), key=jax.random.key(0))
initial = trainer.initial_state()
# Copy the snapshot to host memory before training donates device buffers.
reference = jax.tree.map(lambda x: np.array(x, copy=True), initial.ema)
state = trainer.fit(data, steps=2, log_every=1)
assert int(state.step) == 2
for before, after in zip(
    jax.tree.leaves(reference), jax.tree.leaves(state.ema), strict=True
):
    np.testing.assert_allclose(before, np.asarray(after), rtol=0, atol=1e-6)
print("Completed", int(state.updates), "DPO updates; reference stayed fixed.")
```

You should see two training updates and the final confirmation. `fit` builds its state inside a compiled function, so its copy of the reference can differ from the eager `initial_state()` by float rounding (1.8e-7 at most on an L4). The tolerance of 1e-6 allows that rounding and still fails if an optimizer step moves the reference, because one Adam step at this learning rate moves weights by about 1e-3. This run writes no checkpoints or tracker records. For a real dataset, build both sequences with the same tokenizer and chat format, check that they share the prompt, and take the masks from known token boundaries. Do not search the string for an assistant marker and assume the character offset you find is a token boundary.

### Account for reference memory

Dew stores the DPO reference in `TrainState.ema`. EMA stands for *exponential moving average*, but DPO fixes its decay at 1, so this tree never moves. The objective refuses an `ema_decay` override. `rewards/chosen` and `rewards/rejected` are beta times each side's sequence log-ratio of policy over reference. A policy that has not moved therefore reports zero rewards and no wins. You do not create a second model object or optimize the reference. You do still keep a separate parameter tree and run forward passes through the reference. Budget memory for the policy parameters, reference parameters, optimizer state, gradients, activations and batches. Storing the reference in the EMA field does not make it free.

SFT uses the language-model objective's moving EMA by default. Pass `ema_decay=None` to train without an averaged copy. GRPO keeps a frozen reference only when `beta > 0`. The `pretrained` argument takes a full Flax variables mapping, including its outer `params` collection. When a DPO or GRPO stage starts, those starting weights become the frozen reference.

## GRPO: generate answers and score them

GRPO needs a stream of prompts and a reward function before it can build a training batch. `Prompts` accepts Parquet or JSON records with `prompt`, `data_source`, `ground_truth` and `extra_info`. The prompt can be token IDs, a string or a list of role and content messages. Strings and messages need a tokenizer; token-ID lists do not load one. Missing reward fields become empty strings, and reward metadata that is not a string is passed along as JSON text.

The prompt loader produces left-padded `prompt` IDs of shape `[B, P]` and `prompt_length` of shape `[B]`, where `P` is `max_prompt_len`. If a prompt is too long, it keeps the end. The metadata columns travel as fixed-width UTF-8 byte arrays, and `SampledRollout` turns them back into strings for this callable interface:

```text
reward(data_source: str, completion: str,
       ground_truth: str, extra_info: str) -> float
```

Pick a reward whose score you can check independently of training. For example, for an exact-answer task, compare the decoded completion with the ground-truth answer under a normalization rule you write down. `SampledRollout.decode` defaults to token IDs separated by spaces, not natural-language text. If your reward reads text, pass the model tokenizer's decode function.

Construct `SampledRollout` with the objective, the reward callable, `groups=G` and `max_new_tokens=R`, and pass it to the trainer as `rollout`. `G` must be at least 2. The trainer calls the rollout on the host before the compiled update. Each prompt gets `G` sampled completions. Group-relative rewards give their advantages; `estimator="mean"` only centres them (Dr.GRPO) and `estimator="rloo"` compares each with the rest of its group, the same names `pack` takes. An advantage says how a completion's reward compares with the other rewards in its group.

The rollout turns each completion into a one-call `Session` and builds the batch with `pack`, the same layout every GRPO batch in Dew has ([Engine-sourced rollouts and packed rows](#engine-sourced-rollouts-and-packed-rows)). With `N = B * G`, every column is `[N, P + R]` and aligned with `input_ids`: each chain is a prompt without its padding, then the sampled response, and chains may share a row.

| Field | Meaning |
| --- | --- |
| `input_ids`, `text_segment_ids`, `text_positions` | The chains, which chain each id belongs to, and its position in that chain. |
| `response_mask` | 1 on sampled ids, EOS included. |
| `old_log_probs` | Raw model-policy likelihoods recorded at each sampled id. |
| `behavior_log_probs` | Actual temperature/top-k sampling likelihoods. |
| `advantages` | The completion's advantage repeated across its chain. |
| `versions`, `session_index`, `call_index` | The policy version, the completion (`row * G + group`) and its call, on sampled ids. |

Use `GRPOObjective(model, seq_len=P + R - 1)`, and give the decoder enough context for `P + R` tokens. Every completion is scored, whether it stopped on EOS or on the budget. GRPO combines a clipped policy-ratio loss with a k3 KL penalty against the frozen reference when `beta > 0`. The clipping parameters are `epsilon_low`, `epsilon_high` and `dual_clip`.

Pass `sampling=Sampling(eos_id=..., temperature=..., top_k=...)` from `dew.sampling`. You can give one EOS id or a tuple of ids. The response mask includes EOS. The text passed to the reward leaves out EOS and padding. The rollout turns `Prompts.prompt_length` into the standard `ModelInputs` attention mask.

Generation prefills the padded batch once and packs the real tokens into each row's cache, so prompts of different valid lengths reuse the same compiled shape. After a row hits EOS, later steps leave its cache unchanged. Rescoring reads each chain on its own through its segment ids and positions. GRPO validation scores prompt perplexity over real next-token transitions. It does not generate answers for a separate reward evaluation.

`old_log_probs` holds the raw policy likelihoods from the cached forward pass at sampling time. `behavior_log_probs` holds the likelihoods after temperature and top-k. For greedy sampling, the chosen action has a behavior log-probability of zero. GRPO's PPO ratio compares the current raw policy with the old raw policy. A correction from behavior policy to proximal policy is a separate algorithm choice, and Dew does not apply one unless you ask.

To ask for one, set `GRPOObjective(..., behavior_importance=2.0)`. This applies detached, token-level truncated importance weights from the recorded raw old policy to the recorded behavior policy. The default, `None`, keeps the uncorrected loss. On supported actions the weight is `min(exp(clip(old_raw - behavior, -20, 20)), cap)`. It multiplies the policy-surrogate terms before the usual reduction over token count. PPO clipping still compares the current policy with the raw old policy, and the KL term does not change. Missing or misaligned behavior likelihoods are refused. This follows verl's token TIS implementation at revision `d040717b21af2e23e8e789a3e354cff2394ae2de`, and `tools/parity_behavior.py` checks it against the installed reference. Truncation, token-level weighting and filtered action support mean this is not an unbiased estimator of the raw policy over full trajectories.

## Tool episodes

`dew.objectives.rl.EpisodeRollout` collects multi-turn episodes through an environment you supply and an inference policy that can be bound to weights. It uses the ordinary `Trainer(rollout=...)` hook and `GRPOObjective`. It does not pick an executor for you. Pass `dew.inference.TextGeneration(model, variables)` as its policy. Collection calls `policy.bind(snapshot)` once. Each call after that receives the exact context token IDs, a response budget, a key and `Sampling`, and returns a `Generation` with raw and behavior log-probabilities. Canvas generation is not an autoregressive policy and cannot supply these likelihoods.

The input dataset yields integer `task_id` rows. Your environment factory receives an `EpisodeId` with the task, the attempted step, the sample index and a random seed, and returns an `Environment` to use as a context manager. Its `reset()` returns an `Observation` with the first context. `step(action)` takes a finished model turn and returns either the exact next context or a terminal result. The environment is responsible for decoding, validating tool calls, chat formatting, execution, timeouts and cleaning up resources. Dew itself never executes generated code by default.

`SubprocessEnvironment(command, limits)` is one environment factory you can choose. It starts the argv `command` in its own session and temporary directory, and talks to it in JSON lines. `reset` carries the episode identity. `step` carries the action's context, tokens, termination flag and policy step. Replies carry `context`, `status` and `detail`. `SandboxLimits` sets RLIMIT_CPU and RLIMIT_AS in the worker, plus a wall-clock deadline per session and a message size cap in the parent. On exit, Dew kills the worker's process group, and the worker gets SIGKILL if the parent dies. The worker runs with the caller's OS permissions and has no filesystem or network isolation, so untrusted code needs an outer sandbox. `tests/test_sandbox.py` runs a real worker through round trips, a hang, an exceeded memory limit, a crash, malformed output, parent death and a full `EpisodeRollout` cohort.

`Observation.status` tells apart running, completed, truncated, cancelled and error outcomes. Model EOS ends an action, and the environment decides when the episode ends. A response that hits its token limit without EOS is recorded as truncated and is not sent to a tool. A context longer than `max_prompt_tokens` also truncates the episode, without cutting off the input already recorded. `max_turns` limits the number of model calls.

The verifier takes an `Episode` and returns a finite scalar reward. It sees the termination status, the result detail and every `Transition(action, observation)`, so it can score finished and truncated outcomes differently. Group-relative advantages are computed over episodes of the same task and shared across all their action tokens. The objective rewards the final outcome only; it does not assign credit to individual tool calls. Exceptions, cancellation and verifier failures abort the group before any update. The optional `record` callback receives completed or failed host records. `EpisodeFailure` and `EpisodeCancelled` carry the partial episode.

Verification runs while the environment context is still open, so a verifier you write can inspect temporary files or a live sandbox. Resources are released after verification, whether it succeeded or failed. If release fails, the episode is thrown out instead of being scored.

Episodes train through the same packer as engine-sourced rollouts ([Engine-sourced rollouts and packed rows](#engine-sourced-rollouts-and-packed-rows)). `session_of(episode, group=...)` turns each episode into a `Session` whose calls are its actions. When the environment's next context starts with the previous context plus the sampled actions, the two calls merge into one chain; any other context starts a new chain. Chains share rows `max_prompt_tokens + max_new_tokens` IDs wide, so set `GRPOObjective.seq_len` to that width minus one. The batch keeps B*G*K rows for B tasks, G samples and K turns, which always fits and keeps shapes fixed. Only sampled actions, EOS included, carry loss mass. Observations are never targets. Raw likelihoods land in `old_log_probs` and behavior likelihoods in `behavior_log_probs`, copied from the actual inference result without retokenizing or rescoring a transcript. Truncated episodes are masked rather than trained on, and they are left out of the group baseline.

Collection binds one snapshot of the variables until it returns. Every rank supplies the same number of task rows and agrees on budgets, sampling and clocks before it opens any environment. All local episode slots are sampled together in each round. Finished slots keep their global row positions with inert prompts whose outputs are thrown away, so episodes with different turn counts do not change the order of collectives or the random draws. Ranks agree on a tool, reset or verifier failure before the next generation, and every rank releases the environments it opened. The committed update clock stays in `policy_step`, and the attempted-work clock in the episode identities. Continuing from a checkpoint restores the Trainer and the input iterator.

Projection accepts records from one collection binding only. Each episode and action keeps an internal origin identity, so mixing records from policies bound separately fails even when their attempt and update clocks match. This identity is not a version handle you set and not a hash of the weights. Project batches you collected separately on their own. The identity does not affect random draws or whether replayed public records compare equal.

Both asyncio and concurrent-futures cancellations raise `EpisodeCancelled` with `CANCELLED` status. The original cancellation object is kept as its `__cause__`, and the environment exits under that original exception before the wrapper is raised. Cancelled observations that the environment returns itself have no source exception.

`tests/test_tool_episodes.py` trains a small policy that samples a call to a square tool, receives the computed result and samples a final answer. It checks categorical gradients on actions only, raw and behavior probabilities, resource cleanup, failures and cancellation, and that Trainer updates match with and without a checkpoint restore. These tests run offline and check the lifecycle and the numbers. They do not show that a remote sandbox works.

`tests/test_episode_pool.py` runs two real CPU processes with different episode turn counts. Their actions, likelihoods, projected arrays and updated parameters exactly match a single-process run on the same two global devices. It also runs a tool failure on one rank and a mismatched cohort configuration. Both end on all ranks together, without leaking any open environments.

The lifecycle matches the responsibilities in [verl BaseTool](https://github.com/volcengine/verl/blob/main/verl/tools/base_tool.py): create, execute, calculate reward and release. verl's [multi-turn guide](https://verl.readthedocs.io/en/latest/sglang_multiturn/multiturn.html) describes assistant-only masks and warns about differences caused by retokenization. Dew keeps each actual model call instead of rebuilding sampled tokens from message deltas. A remote sandbox adapter would also have to handle creation, timeout policy and termination, as the [E2B Python SDK](https://github.com/e2b-dev/E2B/blob/main/packages/python-sdk/e2b/sandbox_sync/main.py) does. Dew bundles neither an E2B client nor a verl runtime adapter. `SubprocessEnvironment` covers only the local case with resource limits. For single-shot verification, `SandboxFleet` with `ContainerRunner` runs each program in a network-less container; see [asynchronous RLVR](#asynchronous-rlvr).

`dew.objectives.rl.verl.to_verl(episodes)` exports JSON-compatible `AgentLoopOutput` rows, and `from_verl(rows)` turns them back into episodes. Each row holds one model call with its exact prompt, action tokens, action mask and sampled behavior likelihoods, so compacted contexts stay intact. `extra_fields.dew` carries turn boundaries, raw-policy likelihoods, sampling controls, observations, rewards and the private collection origin. Native verl rows without those fields cannot recover that information, and Dew refuses them. The rows fit verl's per-call tensor mapping. Joining them into one long episode would change the conditioning wherever contexts were compacted.

Live, imported and journal-restored actions all go through the same validation: a nonempty context, nonnegative integer token ids (booleans excluded), aligned finite likelihoods, and a termination flag that agrees with the configured EOS. Tokens after EOS are refused. Checking the upper bound against the vocabulary is left to the caller, which knows the model.

`tools/parity_verl_episodes.py` checks eight real calls against verl revision `d040717b21af2e23e8e789a3e354cff2394ae2de`, using its installed `AgentLoopOutput` model and `as_dict()` tensor conversion. The committed fixture keeps the exact ids, masks, behavior log probabilities and where the final reward is placed. Dew does not import torch or verl at runtime. This is a format for exchanging trajectories. It is not a verl distributed trainer or a remote inference adapter.

Set `EpisodeRollout(..., journal=EpisodeJournal("run/episodes"))` to save sampled pending actions and completed turns in a SQLite WAL, one per rank. The environment must implement `get_state() -> bytes` and `set_state(state: bytes) -> None`. Those snapshots must hold the tool state and any workspace state that later calls or the verifier need. The subprocess adapter exposes these operations as JSON requests, with base64 state strings and a `{"restored": true}` acknowledgment. Environments without snapshots still work when you do not use a journal.

Recovery restores completed turns with their exact contexts, raw and behavior likelihoods, rewards and environment snapshots. Pending draws are saved before the tool runs and reused after a crash. A completed turn is committed before the next tool call or training update. An external effect that happens between a pending record and its completed commit can run twice, so the environment must use the episode identity and action to make such effects idempotent. The journal does not promise that arbitrary external effects happen exactly once, and it cannot resume in the middle of a tool call.

Use one journal directory per run. Recovery checks task identities, sampling controls, topology, training clocks, the key and a digest of the local policy shards. Computing that digest reads the local weights once per collection. The journal refuses concurrent writers, and refuses a different policy at the same clocks. SQLite commits in FULL synchronous mode keep turn boundaries intact. `tests/test_episode_recovery.py` kills a real two-device training process while a subprocess tool call is pending, restores it, and gets identical sampled actions and updated parameters. Its audit log confirms that each completed square call ran once.

## Asynchronous RLVR

Reinforcement learning with verifiable rewards (RLVR) scores each completion by checking it: running the program it wrote against test cases, or comparing its final answer with a reference. `RolloutScheduler` runs GRPO rollouts from a `RolloutSource` while the trainer updates, and `SandboxFleet` runs the checks. `examples/train_rlvr.py` puts them together.

### Rollout servers

`dew.inference.RolloutServer` is the interface: `submit(prompt_ids, max_new_tokens, seed=...)` returns a future `Draw` with the sampled ids, their behavior log-probabilities, the raw-policy log-probabilities when the backend has them, whether EOS ended the draw, and the policy `version` the request was submitted under. `load(variables, version)` pushes new weights. Generation continues across a push; a draw's version is the oldest weights that may have produced any of its tokens.

- `NativeRolloutServer(Server.from_task(task, slots=..., capacity=...))` drives Dew's continuous-batching `Server` on a background thread. `load` copies the trainer's tree onto the served device in place (`Server.reload`) between two steps, casting floating leaves to the served precision, so you can train in float32 and serve in bfloat16. Draws carry both raw and behavior likelihoods.
- `OpenAIRolloutServer(OpenAICompletion(name, client, provider="vllm" or "sglang"), sampling, weights)` posts token-id prompts to a vLLM or SGLang completions endpoint. Each request carries the `Sampling` controls, a seed, `logprobs=0` and the field that makes the engine return the sampled ids: `return_tokens_as_token_ids` for vLLM, which renders them into the logprob tokens, and `return_token_ids` for SGLang, which lists them on the choice. The draw holds the ids the engine sampled instead of text that would need retokenizing. The engines report different distributions. vLLM reports raw model log-probabilities unless it runs with `--logprobs-mode processed_logprobs`, so for vLLM the server refuses a transforming `Sampling` (temperature other than one, top-k, top-p or min-p) unless you pass `processed_logprobs=True`. SGLang's `/v1/completions` reports `log_softmax(logits / temperature)` before its top-k, top-p and min-p filters and has no field for the filtered likelihood, so for SGLang the server accepts any temperature and refuses every filter. SGLang's native `/generate` does report the filtered likelihood under `return_sampling_mask`, for a finite top-k; that is the route a filtered policy would need. Leave `SGLANG_RETURN_ORIGINAL_LOGPROB` unset, because it switches SGLang to raw log-probabilities. SGLang honors the seed only under `--enable-deterministic-inference`. Remote draws carry no raw likelihood.
- `SafetensorsReload(pretrained, directory, base_url, engine)` is the weight push for either engine. It writes the policy with `Pretrained.save` in bfloat16, stages the files beside `directory` and moves each in with `os.replace`, then asks the engine to reload. Launch the engine on the same directory, initially written by `SafetensorsReload.write`. For `engine="vllm"` (checked against v0.30.0) it calls `POST /pause?mode=wait`, which lets in-flight requests finish and schedules no new ones, then `POST /collective_rpc {"method": "reload_weights"}`, `POST /reset_prefix_cache` and `POST /resume`. Those are vLLM development endpoints, enabled by `VLLM_SERVER_DEV_MODE=1`; expose them only on a trusted network. vLLM reports a reset that did not happen as HTTP 200 with `{"success": false}`, so the push fails unless the reset answers `{"success": true}`. A push that fails after the pause leaves vLLM paused, so no draw comes from weights the push may have half loaded; the next successful push resumes it. For `engine="sglang"` it makes one call, `POST /update_weights_from_disk {"model_path": directory, "flush_cache": true, "abort_all_requests": false}`. SGLang starts the load once every in-flight request has finished, holds new requests until it returns, and flushes the radix cache before answering. In-flight draws therefore finish on the old weights, and the push takes as long as the longest of them. A failed SGLang load answers 400 with `{"success": false}`, and the push fails on either, and SGLang then re-reads the same directory as its rollback. In both cases a failed push leaves the server's `version` where it was.

`OpenAICompletion` accepts token-id rows as prompts, and its `Completion` records per-choice `tokens` and `log_probs` when the engine reports them.

### Scheduling rollouts

`RolloutScheduler(objective, source, weights, width=W, rows=N, tasks=..., groups=G, max_lag=1, ahead=1, sync_every=1)` is the trainer's `Rollout` for any `RolloutSource`. Train on `scheduler.tasks(dataset)` with `Trainer(..., rollout=scheduler)`. `tasks` turns a batch into `Task`s: `task_ids` reads integer `task_id` rows, `prompt_tasks` reads prompt rows. The wrapped stream registers each batch as the trainer's prefetch reads it. When the trainer hands the scheduler batch `i`, the scheduler submits batches `i + 1` through `i + ahead` under the version `weights` serves at that moment; batch `i` was submitted `ahead` calls earlier and has been running since. Nothing is read ahead of the trainer's prefetch, so the checkpointed data position stays the trainer's. A resumed run re-reads and resubmits whatever was in flight, and reopening the stream cancels the rollouts the old stream left running.

Each task becomes one group of `G` rollouts. The scheduler relabels every admitted rollout with its own group id, sample index and attempt, so a resubmitted sample rejoins its group. Admission goes by status:

- `COMPLETED` and `AGENT_ERROR` rollouts are admitted with their verifier reward.
- `TRUNCATED` rollouts complete their group, and `pack` masks them: they carry no loss and enter no baseline.
- `INFRA_ERROR` and `CANCELLED` rollouts are never scored. The sample is submitted again under the served weights, up to `max_attempts` failures per sample, after which its group is abandoned.
- A rollout whose oldest call is more than `max_lag` updates behind is discarded and submitted again. One still running whose submission is already past the bound is cancelled without being waited on.

A source that raises instead of returning a status is broken: the exception propagates after the batch's work is cancelled. Two settings cut the long tail without changing the batch shape. `oversample=K` runs `G + K` samples per group and admits the first `G` to finish. `admit=M` admits the first `M` groups of a batch to complete and cancels the rest. Both select by completion time, which favors short rollouts. Rollouts running when a push lands keep running; their later calls carry the new version, and the rollout's staleness is its oldest call's.

The scheduler pushes weights through `weights.load(params, updates)` whenever the served version falls `sync_every` updates behind, so a first submission is at most `ahead + sync_every - 1` updates stale. Construction refuses a `max_lag` below that. A push that does not move the served version is tried once more, then raises. Admitted groups are packed by `pack(rollouts, W, rows=N)` into fixed `[N, W]` rows with per-rollout advantages. `old_log_probs` is always rescored under the trainer's current weights, which serve as the proximal policy ([AReaL](https://arxiv.org/abs/2505.24298)); `behavior_log_probs` keeps what the engine reported, and `GRPOObjective(behavior_importance_cap=...)` weights each token by proximal over behavior. A schedule that allows any lag requires that cap. `log` receives a `SchedulerRecord` per call: the oldest admitted version and lag, admitted groups, mean reward and mean reward components, admitted rollouts by status, resubmissions by cause, cancellations, abandoned groups and seconds waited. One trainer process owns the scheduler; multi-process trainers are refused.

Two sources run on a `RolloutServer`. `PromptSource(server, reward, decode=..., max_new_tokens=R)` draws one completion per sample and scores its decoded text on a thread pool as the draw finishes. A draw that ends on its budget is `TRUNCATED`; a failed draw or reward is `INFRA_ERROR`. `EnvironmentSource(server, environment, verifier, max_prompt_tokens=P, max_new_tokens=R, max_turns=T, workers=N)` runs the in-process `Environment` protocol of [tool episodes](#tool-episodes), one session per sample on a worker thread. Each turn submits the observation's context to the server, so turns of different sessions share the engine's batch instead of a lock-step cohort, and each call keeps the version it was submitted under. An environment that raises or reports `ERROR`, a failed draw and a verifier crash are infrastructure failures; context, token and turn limits truncate. The verifier returns a float or a `Score(reward, components, detail)`. `cancel` stops a session at its next turn, including mid-draw, and releases its environment.

### Sandbox fleet and verifiable rewards

A `Program` is files written into a fresh temporary directory, an argv run there without a shell, and its stdin. `SandboxFleet(runner, limits=SandboxLimits(...), workers=N)` runs programs on `N` threads, each in its own process, and returns an `Outcome` per program: its `Verdict` (`COMPLETED`, `FAILED`, `TIMEOUT`, `CRASHED` or `OUTPUT_LIMIT`), exit code, captured output and seconds. A failed program returns an `Outcome` like any other.

- `ProcessRunner` starts the program through the `SubprocessEnvironment` launcher: RLIMIT_CPU, RLIMIT_AS, no core dumps, its own session, SIGKILL on parent death and a minimal environment. The process group is killed at the wall deadline, on excess output, and after exit. It keeps your user, filesystem and network, so a program can read and reach whatever you can.
- `ContainerRunner(image, runtime="docker")` runs each program in a fresh container with no network, a read-only root, the job directory mounted read-only at `/work`, all capabilities dropped, `no-new-privileges`, an unprivileged user, and memory, CPU-share, process-count and CPU-time limits. The container is killed by name at the deadline. Use this runner for untrusted code.

`CodeReward(fleet)` extracts the last fenced `python` block from a completion and runs it once per test case. The reward's ground truth is a JSON list of `{"stdin", "stdout"}` cases. The reward is the fraction of cases whose stdout matches line by line, or all-or-nothing with `all_or_nothing=True`. `MathReward()` reads the last `\boxed{}` answer and compares it with the reference as an exact rational, so `0.5`, `1/2` and `\frac{1}{2}` agree. Both have the `Reward` signature and work with `SampledRollout` and `PromptSource`. `PromptSource` scores each completion on a thread pool as its draw finishes, so verification overlaps generation and the update.

`tests/test_fleet.py` runs real programs through each verdict, including a forked child that must die at the timeout and, when Docker and `python:3.12-slim` are present, a container that must have no network and a read-only mount. `tests/test_rollout_scheduler.py` checks admission, retries, truncation masks, stragglers, the staleness bound, resume and a `Trainer` run through multi-turn environments on the native server. `tests/test_rollout_sources.py` checks both sources' statuses and cancellation. `tests/test_rollout_servers.py` checks the vLLM and SGLang requests and responses against each engine's wire format, the version of a draw in flight across a push, a safetensors reload read back by `load_pretrained`, and reloads refused on both engines, by status or by a 200 that says `success: false`.

## Engine-sourced rollouts and packed rows

`dew.objectives.rl.sessions` holds the records any recording gateway can supply. A `Call` has the `prompt_ids` the engine read, the `sampled_ids` it drew (EOS included on a natural stop), one engine-reported `behavior_log_probs` entry per sampled id, a `finish_reason` and the policy `version` served when the request was submitted. A `Session` is one harness session: `task`, `group`, `sample`, `attempt`, its calls in submission order, a `Status` (`COMPLETED`, `TRUNCATED`, `AGENT_ERROR`, `INFRA_ERROR`, `CANCELLED`), the verifier's `reward`, its `components` and a `detail` string. A `SessionSource` turns a `Task(id, data)` into futures of sessions.

`pack(sessions, width, rows=None, estimator="group")` builds one training batch. It merges call k + 1 into call k's chain only when call k + 1's prompt ids start with every id of the chain so far, sampled ids included. It never merges on a prompt-only prefix, which would splice sampled ids into a context the model did not see. Chains are packed first-fit into `[rows, width]` with `text_segment_ids` and `text_positions`, so attention and rotary positions stay inside each chain. Every column is aligned with `input_ids`: `response_mask`, `behavior_log_probs`, `versions`, `session_index`, `call_index`, `advantages` and `session_weights`. Only `COMPLETED` and `AGENT_ERROR` sessions are trainable; the others take no rows and no part in the baseline. `advantages` are group-relative over `(task, group)`. `sessions.session_metrics(sessions, batch, source=..., latencies=..., version=...)` reports calls per chain, the masked share of sampled ids by status, reward by source and component, the latency tail and version lag; the mismatch between trainer and engine is the loss's own metric.

`GRPOObjective` trains a packed batch directly. `packed_log_probs(params, batch)` scores each id against its own chain's prefix; a scheduler that rescores the proximal policy stores the result as `old_log_probs`, and otherwise the behavior likelihoods stand in. Without `old_log_probs` the corrections follow verl's bypass mode: the band and the sequence masks compare the detached current policy with behavior, the band only masks, since the ratio already is current over behavior (verl's `token_k1` rejection with threshold `1/high_1/low`, which `tools/parity_agentic.py` checks); out-of-band tokens also leave the token-mean denominator, and a TIS cap is refused because it would count that correction twice. The objective's options, each checked against verl `12ebe0c` by `tests/test_agentic_losses.py` (fixture from `tools/parity_agentic.py`):

| Option | Effect | Reference |
|---|---|---|
| `policy_loss="ppo"` | Dual-clipped token surrogate | `compute_policy_loss_vanilla` |
| `policy_loss="gspo"` | Clipped sequence ratio, pooled per chain, no dual clip | `compute_policy_loss_gspo` |
| `policy_loss="cispo"` | `-sg(clip(r)) * A * log pi` | `compute_policy_loss_cispo` |
| `aggregation="session-mean"` | Mean over sessions of each session's token mean | `seq-mean-token-mean`; Agent Lightning `per_rollout_mean` |
| `behavior_importance=c` | Token TIS weight `min(pi_old / mu, c)` | `compute_rollout_correction_weights` |
| `behavior_importance=(lo, hi)` | IcePop: token weight zero outside the band | same, `"lo_hi"` threshold |
| `sequence_mask=(lo, hi)` | Reject a chain whose summed k1 leaves `[log lo, log hi]` | `compute_rollout_rejection_mask`, `seq_sum_k1` |
| `geometric_mask=(lo, hi)` | Same with the mean k1 | `seq_mean_k1` |

Whenever behavior likelihoods are present the loss reports `mismatch/kl`, `mismatch/k3_kl` and `mismatch/ess`, and each mask reports the share of trainable tokens it removed. `tests/test_packed_grpo.py` checks that packed GRPO equals GRPO over the unmerged chains, one call per row, gradients included.

Before training a new model family or harness setting, run `tools/audit_template.py template --tokenizer <name>` to see whether its chat template keeps histories append-only, and `tools/audit_template.py sessions traces.jsonl` to measure calls per chain on recorded sessions. The audit takes the sampled turn to be what the template writes for a final assistant turn, so each family's own tool-call and reasoning syntax is checked. With Qwen3 (`Qwen/Qwen3-0.6B`), only a reasoning turn followed by a tool-role observation merges: user-role observations drop the reasoning from history, a turn without reasoning is written with an empty `<think></think>` block that history drops, and, in the reasoning setting that otherwise merges, arguments sent as a JSON string still merge while compact tool-call JSON is re-serialized with spaces and splits.

## PPO with a critic

`PPOObjective(model, seq_len, critic=..., value_coefficient=.5, value_clip=.2, beta=...)` trains the policy and critic as two subtrees of one variables tree, with the ordinary optimizer. `ValueHead(backbone)` adds a scalar Dense head on a decoder's hidden states. A custom critic can instead return one scalar per token position from packed token ids with their `segment_ids` and `positions`. The critic is initialized separately from the policy. The unit-decay reference covers only the policy subtree.

`PPORollout(objective, episodes, gamma=1., lam=.95)` wraps an `EpisodeRollout` and adds critic evaluation and GAE. The collector's policy is `objective.policy(variables)`, which picks out the policy subtree when it binds the Trainer snapshot. GAE runs over the sampled actions of a whole episode in draw order, across tool turns, chains and rows. The verifier's reward goes on the last action. Advantages are whitened over the global set of actions before training. A cohort with exactly one trainable action token is refused, since its whitening is undefined; a cohort with none, such as one where every episode truncated, returns zero targets and makes no update. Completed episodes have zero tail bootstrap, following the pinned verl convention; truncated ones are masked.

The prepared batch keeps `old_log_probs`, `behavior_log_probs` and `response_mask`, and adds `old_values`, `advantages` and `returns`, each shaped like the packed `input_ids`. Both losses reduce over the same action-token mass, so PPO keeps token-mean aggregation and refuses sequence masks. `value_coefficient` scales the half-squared critic loss, and `value_clip` clips predictions around the recorded values. Policy clipping, KL strength and the optional behavior-importance cap are the same controls as in GRPO. You can also keep these targets in a dataset and run several PPO updates on them without sampling again.

This example imports repository test fixtures, so run it from a checkout with `PYTHONPATH=src:tests`. It runs on CPU with `JAX_PLATFORMS=cpu`. It samples square-tool calls, runs the in-memory square environment and learns from the final-answer reward. It needs no checkpoint or network service. The fixtures define a small bigram policy and trainable token features, not a useful pretrained language model.

```python
import itertools
import jax
import numpy as np
import optax
from dew import Trainer
from dew.data import Dataset
from dew.objectives.rl import EpisodeRollout, PPOObjective, PPORollout, ValueHead
from test_tool_episodes import ToolPolicy, Harness, verify, SAMPLING
from test_ppo import TokenFeatures

key = jax.random.key(19)
objective = PPOObjective(ToolPolicy(), seq_len=10,
                         critic=ValueHead(TokenFeatures()), beta=.03)
episodes = EpisodeRollout(objective.policy(objective.init(key)), Harness(), verify,
                          max_prompt_tokens=8, max_new_tokens=3, max_turns=3,
                          groups=4, sampling=SAMPLING)
trainer = Trainer(objective, optax.sgd(.05), key=key,
                  rollout=PPORollout(objective, episodes, gamma=.97, lam=.9))
data = Dataset(train=lambda: itertools.repeat({
    "task_id": np.arange(jax.device_count(), dtype=np.int32)}),
    val=None, records=None, batch=jax.device_count())
state = trainer.fit(data, steps=2, log_every=1)
assert int(state.updates) == 2
```

`tools/parity_ppo.py` records installed verl's GAE, clipped policy and value losses and autograd gradients at revision `d040717b21af2e23e8e789a3e354cff2394ae2de`. `tests/test_ppo.py` checks those tensors, the complete Objective loss and parameter gradients, and a two-update run that changes the policy and critic weights, lowers the critic error and keeps the policy reference fixed. Removing the critic, KL or policy-clipping term makes the composite reference comparison fail.

## Flow-GRPO

Flow-GRPO trains a generative model on its own samples using a reward function. It needs a rectified-flow Process with velocity prediction. The objective reuses DiffusionObjective's InputSpec, conditioning encoders and optional autoencoder. The reward receives decoded samples in [-1, 1] and the repeated source batch, and returns one finite scalar per sample.

This offline example trains a small DiT. Brightness is only a demonstration reward. The image field sets the batch size; the rollout does not train on those zero-valued pixels.

```python
import itertools
import jax
import numpy as np
import optax
from dew import Field, InputSpec, Trainer, models, presets
from dew.data import Dataset
from dew.objectives.rl import FlowGRPOObjective, FlowRollout

inputs = InputSpec(Field("image", (4, 4, 1)))
model = models.SimpleDiT(output_channels=1, patch_size=2, emb_features=8,
                         num_layers=1, num_heads=2, mlp_ratio=2)
objective = FlowGRPOObjective(model, presets.Flow()(), inputs,
                              guidance=None, beta=0.01, steps=5)

def brightness(images, batch):
    return images.mean(axis=(1, 2, 3))

rollout = FlowRollout(objective, brightness, groups=4, steps=5, train_steps=2)
batch = {"image": np.zeros((2, 4, 4, 1), dtype=np.uint8)}
data = Dataset(train=lambda: itertools.repeat(batch), val=None, records=2, batch=2)
trainer = Trainer(objective, optax.adam(1e-3), key=jax.random.key(0), rollout=rollout)
state = trainer.fit(data, steps=2, log_every=1)
print(int(state.updates))
```

This prints two committed updates. A conditioned run can supply the fields from `inputs.tokenize(prompts)`, with no image field, for rollout, evaluation and preview. To turn on validation, set `eval_every` and add the metric consumers you need. Metrics such as FID still need real comparison images in the batch, which prompt-only generation does not provide. Each source row forms one contiguous group. The rollout uses the population standard deviation of the group with epsilon 1e-4. Rows with zero advantage stay in the fixed-shape batch with their transition mask cleared.

Callback scores stay float64 through collection, JSON and byte transfer between ranks, and the group statistics. The normalized training advantages are then cast to float32. The host rollout's `rewards` column keeps float64 values. The objective's `reward` metric is a float32 diagnostic, and with JAX x64 off, moving the reward column to the device also narrows it. Those diagnostics can round away differences that still affect learning. This differs on purpose from the [released SD3 trainer's float32 score conversion](https://github.com/yifan123/flow_grpo/blob/879042cf5707f8b90daa98d147d7deac2317c5da/scripts/train_sd3.py#L695-L698). A callback that returns values already rounded to float32 cannot get the lost precision back.

`steps` counts time points, including both endpoints. Here five points give four stochastic transitions, and `train_steps=2` picks the first two for the update. `train_steps=None` uses all transitions. The objective's own `steps` and `sampler` set up evaluation. The public `sample_trajectory` function returns a FlowTrajectory with states, times, joint log densities and marks for stochastic support. Deterministic intervals have no defined Gaussian density and never enter the policy loss.

`FlowGRPOObjective` uses per-coordinate log-density ratios and averages the loss over the selected stochastic transitions. A positive `beta` freezes the starting policy in the EMA slot. A `beta` of zero keeps no reference. Evaluation and previews use the live policy. The KL metric is `transition_kl`, the per-coordinate conditional Gaussian KL from [section 4 of the paper](https://arxiv.org/html/2505.05470v5#S4), with the elapsed time included in the transition variance. The [released SD3 training script](https://github.com/yifan123/flow_grpo/blob/879042cf5707f8b90daa98d147d7deac2317c5da/scripts/train_sd3.py#L897-L899) uses a time-reweighted regularizer, so its `beta` values do not carry over directly to this conditional KL.

On several hosts, every rank takes part in generation. Rewards run on rank zero. The rollout returns rows owned by each process, and the trainer reassembles them. Its host conversion gathers the full trajectory on every rank before picking the owned rows, so host memory grows with the global rollout size. The two-process CPU check shows correct row ownership and a matching single update. It does not measure throughput on a cluster.

## Move between stages

The Python-only `recipes.chain.Recipe` takes a shared decoder, optimizer, key, output directory, batch size and a tuple of `Stage` values. The type of a stage's dataset picks its objective: `ChatMessages` picks SFT, `PreferencePairs` picks DPO and `Prompts` picks GRPO. The stage name only labels its directory. A name such as `"sft"` does not infer masks or convert data.

In new stage directories, each stage starts a fresh optimizer and step counter from the previous stage's final policy variables. DPO and GRPO freeze that starting policy as their reference. If a stage directory already exists, the stage can restore its checkpoint instead. For a fresh chain, use distinct stage names and a new run directory. The returned list keeps every stage's final state, so a long chain can hold a lot of memory.

The chain exposes `beta`, `reward`, `groups`, `max_new_tokens` and `sample`. It does not expose the rollout's `decode`, `eos_id` or `temperature`, so its default reward input is token-ID text. For rewards on decoded text or for stop tokens, build `SampledRollout` and `Trainer` yourself. The [LM command-line recipe](../recipes.md) accepts the `lm`, `masked_diffusion` and `block_diffusion` objectives over token files. It is not a command-line SFT, DPO and GRPO chain.

## Limits and evidence

Multi-turn text episodes run through `EpisodeRollout` and environments you supply, including on coordinated JAX process pools and with the local `SubprocessEnvironment`. `EpisodeJournal` restores completed turn boundaries for environments that can snapshot their state. Dew does not bundle remote sandbox adapters for episodes; `SandboxFleet` runs verification programs in local processes or containers.

The episode tests cover continuing from a checkpoint both after an optimizer update and in the middle of a two-microbatch accumulation window. They also check that an environment error or a cancellation leaves the previous checkpoint intact. Recovering a turn also needs a journal and complete environment snapshots. Read [checkpoints](../guides/checkpoints.md) and [evaluation](../guides/evaluation.md) before you rely on these paths in a long run.

These are the recorded comparisons on fixed tensors:

| Check | Reference | Largest recorded difference |
| --- | --- | --- |
| Chat IDs and assistant mask | TRL 1.12 | 0, both exact |
| DPO loss | TRL 1.12 | 5.96e-08 |
| DPO gradients | TRL 1.12 autograd | Exact |
| GRPO loss | verl 0.9 | 7.45e-08 |
| GRPO gradients | PyTorch autograd | Exact |
| PPO composite loss | verl d040717 | 3.26e-08 |
| PPO parameter gradients | verl/PyTorch autograd | 1.50e-08 |
| PPO episode GAE advantages | verl d040717 | 2.39e-07 |
| PPO episode returns | verl d040717 | 5.97e-08 |

They are narrow numerical checks. They do not measure learned behavior, and they are not evidence for multi-host post-training. They do not cover all tokenizers or model families, and they do not show parity with a full TRL or verl training run. See [references](../references.md) for the methods and upstream projects.
