# Post-training

Post-training changes a model's behavior after pretraining. In supervised fine-tuning (SFT), you supply example answers. In direct preference optimization (DPO), you supply a preferred and a rejected answer to the same prompt. In group-relative policy optimization (GRPO), the language model generates answers and your reward function scores them. Flow-GRPO scores samples from a rectified-flow model.

Dew provides objectives and data specifications for the three language-model methods, plus a Flow-GRPO objective and rollout. They use the same `Trainer`, but they do not consume interchangeable batches. Start with [language models](language_models.md) for next-token prediction and [objectives](objectives.md) for the model/objective/trainer relationship. The first example below trains a tiny DPO model without a tokenizer download, a dataset download, or a pretrained checkpoint.

## SFT: learn from assistant answers

An SFT conversation contains messages with explicit roles. `ChatMessages` reads a Parquet file whose `prompt` column contains a list of messages per row. This is the input layout, not a script:

```text
prompt = [
    {"role": "user", "content": "What is two plus two?"},
    {"role": "assistant", "content": "Four."}
]
```

Supply `ChatMessages` with your tokenizer directory in `tokenizer`, your Parquet file in `path`, and the prediction length in `seq_len`. The tokenizer must have a chat template: a rule for rendering message boundaries, role headers, and content as tokens. A model's training format matters here; joining the message strings yourself can change both the input and which tokens count toward the loss.

Dew renders successive conversation prefixes to assign a role to each token. For an assistant turn, it excludes the generation header from the assistant span. It checks that each tokenized prefix agrees with the longer render and raises when a template changes earlier tokens. This is a checked prefix-rendering method, not a claim that arbitrary chat templates or string delimiters yield correct assistant masks. Inspect the rendered tokens and roles for representative conversations before a real run. Start with a system or user message; an assistant first message is rejected.

`ChatMessages.load(batch=B)` packs conversations into these arrays:

| Field | Shape | Meaning |
| --- | --- | --- |
| `text` | `[B, L + 1]` | Token IDs, including the extra next-token target. |
| `text_roles` | `[B, L + 1]` | `Role` value at each token. |
| `text_segment_ids` | `[B, L + 1]` | Document identity; separates packed conversations. |
| `text_positions` | `[B, L + 1]` | Position within each packed document. |

Here `L` is `ChatMessages.seq_len`. Use `LMObjective(model, L, loss_role=Role.ASSISTANT)`, importing `Role` from `dew.data.chat`. The objective shifts IDs and roles together: input position `i` predicts token `i + 1`, and the target's role determines whether that prediction counts. It also excludes padding and transitions between packed documents. With `loss_role` set, a batch without `text_roles` raises. Without `loss_role`, the objective does not restrict the loss to assistant targets.

`val_path` can name a separate conversation file. Keep evaluation conversations out of the training file; see [evaluation](../guides/evaluation.md) for what token metrics measure.

## DPO: learn from preference pairs

DPO compares the policy's relative preference for two answers with a fixed reference policy. The **policy** is the model you update. The **reference** is its starting parameter snapshot. For each prompt, the chosen and rejected sequences include the prompt followed by their respective completions. Completion masks mark the answer tokens with 1 and prompt tokens with 0.

`PreferencePairs` accepts a Parquet `path` or a tuple of JSON strings in `records`, but not both. Each row has `chosen`, `rejected`, `chosen_mask`, and `rejected_mask`. The masks have the same lengths as their ID lists. If you omit a mask, Dew treats every token as completion; it does not infer a boundary from the text. Always provide masks for prompt-and-answer data.

A loaded batch contains `input_ids` and `completion_mask`, both shaped `[B, 2, S]`. Index 0 of the middle axis is chosen, index 1 is rejected. Dew right-pads shorter rows to `S` and gives padding zero mask weight. Overlong rows raise an error. `PreferencePairs.seq_len` is the full row width `S`; use `DPOObjective(model, seq_len=S - 1)`.

The objective sums next-token log-probabilities over each completion and applies the preference log-sigmoid loss. `beta` must be positive and controls the scale of the policy/reference comparison. Validation measures the chosen answers' perplexity under the policy. That metric alone does not measure preference win rate or response quality.

### Run a complete offline DPO example

Run this block in a fresh Python process after [installing Dew](../installation.md). All inputs are in memory. The invented eight-token vocabulary demonstrates pair construction and optimization. IDs 1 and 2 (or 1 and 6) form the prompt, 3 is the preferred answer, 4 is the rejected answer, and 5 ends the answer. The end token counts toward completion loss. Repeating the two pairs supplies a batch of eight, which also divides across eight local devices.

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
    np.testing.assert_array_equal(before, np.asarray(after))
print("Completed", int(state.updates), "DPO updates; reference stayed fixed.")
```

You should see two training updates and the final confirmation. This run writes no checkpoints or tracker records. For a real dataset, construct both sequences with the same tokenizer and chat format, verify their common prompt, and derive masks from known token boundaries. Do not search arbitrary strings for an assistant marker and assume the resulting character offset is a token boundary.

### Account for reference memory

Dew stores the DPO reference in `TrainState.ema`. EMA means *exponential moving average*, but DPO fixes its decay to 1, so this tree never moves. The objective refuses an `ema_decay` override. `rewards/chosen` and `rewards/rejected` are beta times each side's policy-minus-reference sequence log-ratio, so an unmoved policy reports zero rewards and no wins. You do not create a second model object or optimize the reference, but you still retain a separate parameter tree and run reference forward passes. Budget memory for policy parameters, reference parameters, optimizer state, gradients, activations, and batches. Reusing the EMA field does not make the reference free.

SFT uses the language-model objective's moving EMA by default; pass `ema_decay=None` to train without any averaged copy. GRPO allocates the frozen reference only when `beta > 0`. The `pretrained` argument takes a full Flax variables mapping, including its outer `params` collection. At a new DPO or GRPO stage, that initialization becomes the frozen reference.

## GRPO: generate answers and score them

GRPO needs a prompt stream and a reward function before it can produce a training batch. `Prompts` accepts Parquet or JSON records with `prompt`, `data_source`, `ground_truth`, and `extra_info`. The prompt may be token IDs, a string, or role/content messages. Strings and messages require a tokenizer; token-ID lists do not load one. Missing reward fields become empty strings, and non-string reward metadata travels as JSON text.

The prompt loader produces left-padded `prompt` IDs of shape `[B, P]` and `prompt_length` of shape `[B]`, where `P` is `max_prompt_len`. It retains the tail of an overlong prompt. The metadata columns become fixed-width UTF-8 byte arrays for transport, then `SampledRollout` turns them back into strings for this callable interface:

```text
reward(data_source: str, completion: str,
       ground_truth: str, extra_info: str) -> float
```

Choose a reward whose score you can check independently of training. For example, an exact-answer task can compare a decoded completion with the ground-truth answer under an explicit normalization rule. `SampledRollout.decode` defaults to space-separated token IDs, not natural-language decoding. Supply the model tokenizer's decoding function when your reward reads text.

Construct `SampledRollout` with the objective, reward callable, `groups=G`, and `max_new_tokens=R`, then pass that object as the trainer's `rollout` argument. `G` must be at least 2. The trainer calls the rollout on the host before the compiled update. Each prompt gets `G` sampled completions; group-relative or leave-one-out (`sample="rloo"`) rewards determine their advantages. An advantage expresses how a completion's reward compares with its group's rewards.

With `N = B * G`, the objective consumes:

| Field | Shape | Meaning |
| --- | --- | --- |
| `input_ids` | `[N, P + R]` | Prompt followed by sampled response, with each prompt's group contiguous. |
| `old_log_probs` | `[N, R]` | Raw model-policy likelihoods recorded at each sampled action. |
| `behavior_log_probs` | `[N, R]` | Actual temperature/top-k sampling likelihoods. |
| `response_length` | `[N]` | Valid response actions, including EOS. |
| `terminated` | `[N]` | True for EOS termination, false for the token budget. |
| `advantages` | `[N, R]` | Each completion's advantage repeated across response positions. |
| `response_mask` | `[N, R]` | Response positions that count toward the loss. |
| `rewards` | `[N]` | Scalar reward for each completion. |

Use `GRPOObjective(model, seq_len=P + R - 1)`, and give the decoder enough context for `P + R` tokens. GRPO combines a clipped policy-ratio loss with a k3 KL penalty against the frozen reference when `beta > 0`. The clipping parameters are `epsilon_low`, `epsilon_high`, and `dual_clip`.

Pass `sampling=Sampling(eos_id=..., temperature=..., top_k=...)` from `dew.sampling`. One EOS id or a tuple of ids is supported. The response mask includes EOS. `response_length` counts valid actions; `terminated` marks EOS termination. Reward text excludes EOS and padding. The rollout converts `Prompts.prompt_length` into the canonical `ModelInputs` attention mask. Generation prefills the padded batch once and compacts real tokens into each row's cache; different valid lengths reuse the same compiled shape. Subsequent steps after EOS leave that row's cache unchanged. Full rescoring left-aligns real context within a fixed shape. GRPO validation scores prompt perplexity over real next-token transitions; it does not generate an independent reward evaluation.

`old_log_probs` records raw policy likelihoods from the cached forward at sampling time. `behavior_log_probs` records the distribution after temperature/top-k; greedy behavior has log-probability zero for its selected action. GRPO retains its current/old raw-policy PPO ratio. A behavior-to-proximal importance correction is a separate algorithm choice and is not applied implicitly.

## Tool episodes

`dew.objectives.rl.EpisodeRollout` collects multi-turn episodes through a user-supplied environment and a bindable inference policy. It uses the ordinary `Trainer(rollout=...)` capability and `GRPOObjective`; it supplies no executor or sandbox. Pass `dew.inference.TextGeneration(model, variables)` as its policy. Collection calls `policy.bind(snapshot)` once. Each subsequent call receives exact context token IDs, a response budget, a key and `Sampling`, and returns `Generation` with raw and behavior log-probabilities. Canvas generation is not an autoregressive policy and cannot supply these likelihoods.

The input dataset emits integer `task_id` rows. Your environment factory receives an `EpisodeId` containing the task, attempted step, sample index and random seed, then yields a context-managed `Environment`. Its `reset()` returns an `Observation` with the initial context; `step(action)` handles a completed model turn and returns either the exact next context or a terminal result. The environment owns decoding, tool-call validation, chat formatting, execution, timeouts and resource cleanup. Nothing in Dew executes generated code by default.

`Observation.status` distinguishes running, completed, truncated, cancelled and error outcomes. Model EOS completes an action, while the environment decides when the episode ends. A response that reaches its token limit without EOS is recorded as truncated and is not sent to a tool. A context exceeding `max_prompt_tokens` also truncates the episode without cutting away its recorded input. `max_turns` bounds the number of model calls.

The verifier takes an `Episode` and returns a finite scalar reward. It sees the termination status, result detail and every `Transition(action, observation)`, so it can score terminal and truncated outcomes differently. Group-relative advantages are computed over episodes of the same task and shared across their action tokens. This is a terminal-reward objective, not inferred per-tool credit. Exceptions, cancellation and verifier failures abort the group before an update. The optional `record` callback receives completed or failed host records; `EpisodeFailure` and `EpisodeCancelled` carry the partial episode.

Verification runs while the environment context is still open, so a harness-owned verifier can inspect temporary files or a live sandbox. Resource release follows verification on both success and failure. A release failure invalidates the episode rather than admitting an otherwise scored trajectory.

Every model call becomes a separate GRPO row. For B tasks, G samples and K allowed turns, the batch has B*G*K rows, each with `max_prompt_tokens + max_new_tokens` IDs. Set `GRPOObjective.seq_len` to that width minus one. The prompt contains the exact context, including observations, and the response contains only sampled actions, including EOS. Unused turn slots and response padding have zero loss mask. Observations are never response targets. Raw and behavior likelihoods are copied from the actual inference result without retokenizing or rescoring a transcript.

Collection binds one variables snapshot until it returns. Every rank supplies the same number of task rows and agrees budgets, sampling and clocks before opening environments. All local episode slots are sampled together on each round. Finished slots keep their global row positions with inert prompts whose outputs are discarded, so differing turn counts do not change collective order or random draws. A tool/reset/verifier failure is agreed before the next generation; every rank releases its opened environments. The committed update clock remains in `policy_step` and the attempted-work clock in episode identities. Checkpoint continuation restores the Trainer and input iterator. Recovery inside an external tool call, durable journals and exactly-once effects remain separate harness responsibilities.

Projection accepts records from one collection binding. Each episode and action retains an internal origin identity, so splicing records from separately bound policies fails even when their attempt and update clocks match. This identity is neither a user-configured version handle nor a weight-content hash; separately collected batches must be projected separately. It does not affect numerical randomness or equality of replayed public records.

Raised asyncio and concurrent-futures cancellations both produce `EpisodeCancelled` with `CANCELLED` status. The original cancellation object remains its `__cause__`, and the environment exits under that original exception before the wrapper is raised. Explicit cancelled observations have no source exception.

`tests/test_tool_episodes.py` exercises a trainable small policy that samples a square-tool call, receives its computed observation and samples a final answer. It checks action-only categorical gradients, raw/behavior probabilities, resource cleanup, failures and cancellation, and uninterrupted versus checkpoint-restored Trainer updates. These are offline lifecycle and numerical proofs, not evidence of remote sandbox operation.

`tests/test_episode_pool.py` runs two real CPU processes with unequal episode turn counts. Their actions, likelihoods, projected arrays and updated parameters match a single-process run on the same two global devices exactly. It also exercises a rank-local tool failure and inconsistent cohort configuration; both terminate collectively without leaking opened environments.

The lifecycle fits the responsibilities documented by [verl BaseTool](https://github.com/volcengine/verl/blob/main/verl/tools/base_tool.py): create, execute, calculate reward and release. Its [multi-turn guide](https://verl.readthedocs.io/en/latest/sglang_multiturn/multiturn.html) describes assistant-only masks and warns about retokenization differences. Dew retains each actual model call instead of reconstructing sampled tokens from message deltas. A remote sandbox adapter would additionally own creation, timeout policy and termination, as in the [E2B Python SDK](https://github.com/e2b-dev/E2B/blob/main/packages/python-sdk/e2b/sandbox_sync/main.py). Neither an E2B client nor a verl runtime adapter is bundled here.

## Flow-GRPO

Flow-GRPO optimizes generated samples with a reward function. It requires a rectified-flow Process with velocity prediction. The objective reuses DiffusionObjective's InputSpec, conditioning encoders, and optional autoencoder. The reward receives decoded samples in [-1, 1] and the repeated source batch, and returns one finite scalar per sample.

This offline example trains a small DiT. Brightness is a demonstration reward. The image field supplies the batch size; the rollout does not train on those zero-valued pixels.

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

This prints two committed updates. A conditioned run can supply the fields from `inputs.tokenize(prompts)` without an image field for rollout, evaluation, and preview. Enable validation with `eval_every` and appropriate metric consumers. Metrics such as FID still require real comparison images in the batch; prompt-only generation does not supply those images. Each source row forms one contiguous group. The rollout uses population group standard deviation with epsilon 1e-4. Zero-advantage rows remain in the fixed-shape batch with their transition mask cleared.

Callback scores stay float64 through collection, JSON/byte transfer between ranks, and group statistics. Normalized training advantages are then cast to float32. The host rollout's `rewards` column retains float64 values. The objective's `reward` metric is a float32 diagnostic; with JAX x64 disabled, device transfer also narrows the reward column. Those diagnostics can round away distinctions that still drive learning. This input-precision contract deliberately differs from the [released SD3 trainer's float32 score conversion](https://github.com/yifan123/flow_grpo/blob/879042cf5707f8b90daa98d147d7deac2317c5da/scripts/train_sd3.py#L695-L698). A callback that returns already-rounded float32 values cannot recover their earlier precision.

`steps` counts time points including both endpoints. Here, five points produce four stochastic transitions, and `train_steps=2` selects the first two for the update. `train_steps=None` uses all transitions. The objective's separate `steps` and `sampler` configure evaluation. The public `sample_trajectory` function returns a FlowTrajectory with states, times, joint log densities, and stochastic-support marks. Deterministic intervals have undefined Gaussian density and never contribute to the policy loss.

`FlowGRPOObjective` uses per-coordinate log-density ratios and averages the loss over selected stochastic transitions. Positive `beta` freezes the initial policy in the EMA slot; zero `beta` allocates no reference. Evaluation and previews use the live policy. The KL metric is `transition_kl`, the per-coordinate conditional Gaussian KL from [the paper's section 4](https://arxiv.org/html/2505.05470v5#S4), with elapsed time included in the transition variance. The [released SD3 training script](https://github.com/yifan123/flow_grpo/blob/879042cf5707f8b90daa98d147d7deac2317c5da/scripts/train_sd3.py#L897-L899) uses a time-reweighted regularizer, so its `beta` values are not directly interchangeable with this conditional-KL contract.

On multiple hosts, every rank joins generation. Rewards run on rank zero. The rollout returns process-owned rows for the trainer to reassemble. Its current host conversion gathers the full trajectory on each rank before selecting owned rows, so host memory grows with the global rollout size. The two-process CPU check proves row ownership and one-update parity; it does not establish cluster throughput.


## Move between stages

The Python-only `recipes.chain.Recipe` accepts a shared decoder, optimizer, key, output directory, batch size, and a tuple of `Stage` values. A stage's dataset **type** selects its objective: `ChatMessages` selects SFT, `PreferencePairs` selects DPO, and `Prompts` selects GRPO. The stage name labels its directory; a name such as `"sft"` does not infer masks or convert data.

In new stage directories, each stage starts a fresh optimizer and step counter from the previous stage's final policy variables. DPO and GRPO freeze that starting policy as their reference. An existing stage directory can instead trigger checkpoint restoration. Use distinct stage names and a new run directory when you intend a fresh chain. The returned list retains every stage's final state, so long chains can retain substantial memory.

The chain exposes `beta`, `reward`, `groups`, `max_new_tokens`, and `sample`, but does not expose the rollout's `decode`, `eos_id`, or `temperature`. Its default reward input is therefore token-ID text. For tokenizer-decoded rewards or stop-token handling, construct `SampledRollout` and `Trainer` yourself. The [LM command-line recipe](../recipes.md) accepts `lm`, `masked_diffusion` and `block_diffusion` objectives over token files; it is not a command-line SFT/DPO/GRPO chain.

## Limits and evidence

Multi-turn text episodes run through `EpisodeRollout` and user-supplied environments, including coordinated JAX process pools. Built-in sandbox execution and mid-tool recovery are not yet provided.

The episode tests cover checkpoint continuation both after an optimizer update and inside a two-microbatch accumulation window. They also check that an environment error or cancellation leaves the previous checkpoint intact. External environment recovery is not included in that guarantee. See [checkpoints](../guides/checkpoints.md) and [evaluation](../guides/evaluation.md) before relying on those paths in a long run.

The recorded fixed-tensor comparisons are:

| Check | Reference | Largest recorded difference |
| --- | --- | --- |
| Chat IDs and assistant mask | TRL 1.12 | 0, both exact |
| DPO loss | TRL 1.12 | 5.96e-08 |
| DPO gradients | TRL 1.12 autograd | Exact |
| GRPO loss | verl 0.9 | 7.45e-08 |
| GRPO gradients | PyTorch autograd | Exact |

These are narrow numerical comparisons, not benchmarks of learned behavior or evidence of multi-host post-training. They do not establish tokenizer coverage, model-family coverage, or parity with a full TRL/verl training run. See [references](../references.md) for the methods and upstream projects.
