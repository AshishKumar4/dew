# Dew at scale: the plan

Design note, 2026-09-02. Author: the MasterPlan agent. Companion document:
`docs/design/post-training.md`, which is the detailed spec for SFT, preference
and RL objectives. This document is the frame that spec sits in.

## What this is, and how to read it

This is the plan for taking Dew from a framework that trains research-sized
models on one card to one that trains real models at real scale, with
post-training and reinforcement learning, on one design. It covers the package
map and the seams, what gets lifted from Google's JAX code and what gets
written, the telemetry design, the scale waves with their acceptance runs, the
RL framework, the Recipe layer, the sequencing, and the risks.

It is a plan, not code. Nothing here is implemented by this document.

Citation convention:

| Form | Means |
| --- | --- |
| `src/dew/...:12-34` | Dew at `main`, commit `25e9573` |
| `wave/<branch>:src/...:12` | a branch under review, at the head named in section 4.1 |
| `maxtext configs/base.yml:673` | the MaxText clone at `/tmp/plan/maxtext`, commit `c114e25`, 2026-09-02, Apache 2.0, paths relative to `src/maxtext/` |
| `tokamax ...`, `grain ...`, `optax ...`, `orbax ...`, `kauldron ...`, `gemma ...` | clones under `/tmp/plan/`, commits in appendix A |
| `tunix ...`, `trl ...` | clones under `/tmp/design/`, commits in appendix A |
| `docs/research/x.md:12` | a research note in this repository |

Line numbers into the clones are exact for those commits and will drift. The
commit is given so a reader can check.

Three rules were applied throughout, from `CONTRIBUTING.md:7-11`. A capability
has one implementation. A reimplementation needs a reason a reader can check. A
new abstraction survives only if inlining it makes the code worse, and this
document states that test for every one it proposes.

## 0. The ten decisions this plan makes

| # | Decision | Where it is argued |
| --- | --- | --- |
| 1 | RL lives in `dew.rl` inside this repository, not in a separate `dew-rl` distribution, and a test asserts the import arrow only ever points from `dew.rl` into `dew`, never back | 5.1 |
| 2 | The only new `Objective` method is `rollout`, host side, batch in and batch out; RL never reaches the trainer | 5.4, agreed with post-training.md |
| 3 | The frozen reference model is the existing EMA tree with decay identically 1.0, and the trainer skips the EMA update in that case at trace time | 5.2, agreed with post-training.md |
| 4 | One `PolicyGradientObjective` with five swappable slots makes PPO, GRPO and its variants, FlowGRPO, DiffusionNFT and on-policy distillation instances; DPO is not forced into that shape and says so | 5.3 |
| 5 | A world model is an `Env`, so imagination training is the same rollout code as reality | 5.6 |
| 6 | Telemetry is one `Tracker` protocol with five methods, one frozen metric-key vocabulary, and probes on their own compiled step at their own cadence | 3 |
| 7 | MaxText's MoE block is patterned, not lifted: it is 3731 lines of NNX bound to MaxText's config object (`maxtext layers/moe.py`, `:419` `class RoutedMoE(nnx.Module)`) | 2, 4.7 |
| 8 | The multi-host wave targets `v5e-16`, two hosts and sixteen chips, because that is the smallest slice with a real ICI and a real second host, and `dew-tpu` already creates it (`docs/tpu.md:17-23,58-70`) | 4.6 |
| 9 | `Recipe` is a list of stages plus a manifest chain, run as separate processes, and nothing else; it earns its place only by the manifest | 6 |
| 10 | The first environment integration is MuJoCo Playground on MJX, not `brax.envs`, because Brax's own README says only `brax/training` is maintained | 5.5 |

## 1. Architecture as it will be

### 1.1 The map

Everything below `src/dew` today, plus what this plan adds, marked `new`.

```
dew/
  nn/            modules: attention, blocks, dit, vit, ssm, backbones/, autoencoders/
                 new: moe.py (router + experts), mixers/ (gated delta rule, MLA)
  objectives/    base.py, diffusion/, jepa/, lm/
                 new: rl/ (policy_gradient, preference, distillation, actor_critic)
  training/      trainer.py, objective_trainer.py, distributed.py, optim.py, runtime.py
  sampling/      diffusion samplers, text.py
  data/          sources/, dataloaders.py, registry.py, online_loader.py
  eval/          metrics that score artifacts
  interop/       safetensors_io.py, hf_decoders.py (on wave/hf-decoders)
  telemetry/     instrumentation.py, devices.py
                 new: tracker.py, keys.py, goodput.py, probes.py
  config/        the typed run config
  cli/           dew-tpu; imports no jax
  diffusion/     schedules, transforms
  inputs/        conditioning encoders and input configs
  checkpoints/   checkpoint path and serialization helpers
  registry.py    architecture registry and precision policy
  new rl/        advantage.py, surrogate.py, rollout.py, buffer.py, env.py, reward.py
  new recipes/   Stage, Recipe, manifest, runner
recipes/         lm/train.py, diffusion/train.py, jepa/train.py: the run entrypoints
```

Two placements need a word.

`dew.rl` holds primitives, `dew.objectives.rl` holds the objectives that use
them. The split is the same one the repository already draws between
`dew.diffusion` (schedules and transforms) and `dew.objectives.diffusion` (the
loss that uses them). It keeps the primitives testable without a trainer.

`dew.recipes` (the package) is not `recipes/` (the directory of entrypoints).
The package composes runs; the directory holds the programs a run invokes.
Section 6 keeps them from becoming two systems.

### 1.2 The seams, one sentence each

| Seam | Contract |
| --- | --- |
| `dew.nn` | A module maps arrays to arrays, carries logical axis names on its parameters, and knows nothing about losses, data, meshes or training (`CONTRIBUTING.md:9`) |
| `dew.objectives` | An objective owns its parameter tree, its loss and aux metrics, its EMA policy, its validation artifacts, and from now on the rollout that produces its own batch (`src/dew/objectives/base.py:44-67`) |
| `dew.training` | The trainer owns the mesh, the compiled step, EMA bookkeeping, checkpoints, the loops, and the emission of telemetry (`src/dew/training/objective_trainer.py:227-298`) |
| `dew.sampling` | A sampler is a pure function of model, params, conditioning and rng, jitted on its own; it never sees trainer state (`src/dew/sampling/text.py:57-86`) |
| `dew.data` | A source produces records, a transform is a Grain transform, and the pipeline hands back an iterator plus a state that can be checkpointed (`src/dew/data/dataloaders.py`, `src/dew/training/distributed.py:78-126`) |
| `dew.eval` | A metric scores artifacts and reduces across batches; it holds no model and no training state (`src/dew/eval/common.py:7-14`) |
| `dew.interop` | Translates between a Dew parameter tree and a foreign layout in both directions, with a parity test (`wave/hf-decoders:src/dew/interop/hf_decoders.py:146,319,413,446`) |
| `dew.telemetry` | Measures (FLOPs, goodput, time, probes) and hands numbers to one `Tracker`; it never decides what a number means |
| `dew.config` | One frozen dataclass tree per run, with `to_dict`/`from_dict` round trip (`src/dew/config/__init__.py:151-167`) |
| `dew.cli` | Controls processes and machines, imports no array library (`src/dew/cli/__init__.py`) |
| `dew.rl` | Pure functions and small pytrees over rollout batches: advantages, surrogates, buffers, environments, rewards; it imports `dew.nn` and `dew.sampling` and nothing from `dew.training` |
| `dew.recipes` | Chains stages, resolves one stage's output into the next stage's input, and records provenance |

### 1.3 What crosses no seam

These are the lines a change may not cross. Each one is checkable.

| Never | Why, and the check |
| --- | --- |
| The mesh reaches a model or an objective | Sharding is expressed as logical axis names on parameters and resolved by a rules table in the trainer (`wave/sharding-rules:src/dew/training/distributed.py:33-47`). A grep for `Mesh` under `src/dew/nn` and `src/dew/objectives` returns nothing |
| The trainer knows what a loss term means | The objective returns `(scalar, aux)` and the trainer prefixes `train/` and logs (`src/dew/training/trainer.py:505-510`). A grep for a domain word (`kl`, `reward`, `advantage`) in `src/dew/training` returns nothing |
| RL reaches the trainer | `rollout` returns a batch; rewards, advantages and log probabilities are ordinary named arrays in it |
| A tokenizer reaches `dew.nn` | Vocabulary size is an integer field |
| wandb reaches anything but `dew.telemetry` | Section 3. Today `self.wandb` is used in the trainer and passed into `Objective.log_validation_artifacts` (`src/dew/objectives/base.py:66`); both become `Tracker` |
| Data reaches `dew.nn` | Models take arrays |
| `dew` imports `dew.rl` | Section 5.1, enforced by a test |

### 1.4 Every new abstraction, and its deletion test

An abstraction that fails its own test is deleted, not documented.

| New thing | Deletion test: what breaks if it is inlined or removed |
| --- | --- |
| `Objective.rollout` | Every RL objective loses its data source and has to reach into the trainer loop. Nothing else in Dew changes, because the default implementation returns the batch unchanged |
| `Tracker` | The trainer holds a wandb handle again and a second backend means editing the trainer. If Dew never gains a second backend and never wants a null tracker in tests, delete it |
| `dew.telemetry.keys` (the vocabulary table) | Metric names live in string literals at their call sites, and the cadence declarations in `src/dew/training/trainer.py:145-153` drift from them. If the table is ever only read by the wandb tracker, inline it there |
| `ProbeSpec` and the probe step | Activation statistics either run every step and cost throughput, or they do not exist. If a probe is ever cheap enough to leave on, move it into the aux dict and delete the second step |
| `dew.rl.advantage`, `dew.rl.surrogate` | Each algorithm carries its own copy of the same masked reductions, and a fix to one does not reach the others. If only one algorithm ever ships, inline it into that objective |
| `PolicyGradientObjective` | PPO, GRPO, FlowGRPO and NFT become four objectives that share nothing but comments. If two of the five slots are never swapped, collapse it |
| `Env` | PPO and Dreamer each write their own rollout scan, and imagination stops being the same code path as reality. If a world model never implements it, delete the protocol and keep the MJX adapter |
| `ReplayBuffer` | Off-policy and model-based algorithms cannot exist. Every on-policy algorithm ignores it, so if Dew stays on-policy, delete it |
| `Recipe` and `Stage` | A three-stage run becomes three commands and a copied path. The manifest chain is the only thing lost, so if nobody reads manifests, delete the layer |
| `dew.nn.moe` | No sparse model, no expert axis. There is no way to inline this one |
| Logical axis rules (landing on `wave/sharding-rules`) | Sharding returns to shape inference, and tensor, sequence and expert axes become unreachable (`docs/research/google-jax-stack.md:148,674`) |

## 2. Lift or write

The rule from `CONTRIBUTING.md:7`: compose before writing, and a
reimplementation needs a reason a reader can check. This section applies it
candidate by candidate. The verdicts are:

| Verdict | Meaning |
| --- | --- |
| lift | the code is copied into Dew with its licence header and adapted at the edges |
| pattern | the design is followed and the code is written against Dew's seams, because the source is coupled to something Dew does not have |
| depend | the library is called, not copied |
| fresh | written from the paper or the arithmetic, because no source fits |

Every source named here is Apache 2.0. The licence column records what a lift
has to carry into Dew.

### 2.1 The table

| Candidate | Verdict | Source, file:line | Licence | Adaptation for Dew's seams | Proof it is correct |
| --- | --- | --- | --- | --- | --- |
| MoE block and router | pattern | `maxtext layers/moe.py:419` (`class RoutedMoE(nnx.Module)`), routing at `:751` `get_topk`, `:881-908` `deepseek_routing`, DeepSeek weight scaling `:835-841`, aux-loss-free bias update `:238-261` | Apache 2.0 | The file is 3731 lines, is NNX, and reads `self.config` for about forty fields. Dew is Linen with explicit module fields (`docs/research/google-jax-stack.md:656`). Port the routing math and the expert layout into `dew/nn/moe.py` as two modules, `Router` and `ExpertMLP`, with logical axis names `exp`, `mlp`, `embed` | Router parity at fp32 on CPU against `MixtralSparseMoeBlock` and `DeepseekV3MoE` from transformers, with the largest observed difference written in the test (`docs/research/model-families.md:1322`); a mutation test that removes the renormalisation and fails |
| Aux-loss-free load balancing | lift | `maxtext layers/moe.py:238-261`, which cites arXiv 2408.15664 | Apache 2.0 | 24 lines, no MaxText types in the body, returns a bias update from top-k indices. Copy with header, call it from `Router`, keep the bias in fp32 | A test that a deliberately imbalanced router drives the bias in the direction that rebalances it, and that the bias never enters the gating value, only the selection |
| Ragged dot path | depend | `tokamax _src/ops/ragged_dot/api.py:77-110`, same signature as `jax.lax.ragged_dot`; MaxText's own dispatch at `maxtext layers/moe.py:1516-1542,1560-1566,1581` | Apache 2.0 | Add `tokamax` as an optional extra and route `dew.nn.moe`'s grouped matmul through it, with `jax.lax.ragged_dot` as the value of `implementation='xla'`. Same shape as `dew.nn.attention`'s existing `implementation` argument (`src/dew/nn/attention.py:119-205`) | Numerical equality of the grouped matmul against a dense masked reference at fp32, and a benchmark row per implementation on the same shapes (the fast tokamax paths do not lower on this workstation's Ada card, `docs/research/google-jax-stack.md:304-316`) |
| Expert parallel axis | pattern | `maxtext configs/base.yml:549` (twelve mesh axes including `expert`), `:704` `ici_expert_parallelism`, `:690` `dcn_expert_parallelism`, sharding guidance in the MoE config block `:296-299` | Apache 2.0 | One more axis in `build_mesh` and one more rule entry (`exp` to `expert`) in the table that `wave/sharding-rules` introduces (`wave/sharding-rules:src/dew/training/distributed.py:33-47`). No new machinery | Loss equality to 1e-6 over 50 steps between `expert=1` and `expert=4` on the simulated eight-device mesh, same seed |
| Flash and splash attention | depend | `tokamax _src/ops/attention/api.py:81`; MaxText vendors the splash kernel and exposes `use_tokamax_splash` | Apache 2.0 | Add `implementation='tokamax'` to `dew.nn.attention.scaled_dot_product_attention`, which already has this seam (`src/dew/nn/attention.py:119-205`). Replaces the older `jax.experimental.pallas.ops.tpu.flash_attention` import on the TPU path | Existing attention parity tests run against the new implementation value; a benchmark row on v5e-16 |
| Fused linear cross entropy | depend | `tokamax _src/ops/linear_softmax_cross_entropy_loss/api.py:46` | Apache 2.0 | An alternative to the chunked head that `wave/lm-head-perf` implements. Keep one path: whichever is faster on the two targets, with the number | Loss and gradient equality to fp32 tolerance against the chunked head on the same weights, which `wave/lm-head-perf` already tests (`tests/test_chunked_cross_entropy.py`) |
| Vocabulary tiling | pattern | `maxtext utils/vocabulary_tiling.py:59-278` (`vocab_tiling_linen_loss`, a custom vjp over head chunks), config at `maxtext configs/base.yml:722-729` | Apache 2.0 | Already landing as `wave/lm-head-perf`, which chunks the head and keeps the tiles for the backward pass (`wave/lm-head-perf` commits `5e77167`, `3d184ef`). MaxText's version is the reference for the `ag_once` idea: gather the head once for the backward | Landed with `tests/test_chunked_cross_entropy.py` and a step-time number in `docs/performance.md` |
| Logical axis rules and sharding tolerance | pattern, landing | `maxtext configs/base.yml:549,550-658,673` | Apache 2.0 | Landed on `wave/sharding-rules` as a 13-entry table (`:33-47`), a spec canonicaliser (`:104-124`) and a startup assertion (`:159-208`). MaxText's table is 109 lines because it has twelve axes; Dew's grows one entry per axis it gains | Landed with `tests/test_parallelism.py` (231 new lines) |
| Named remat policy | pattern | `maxtext layers/decoders.py:344-408` (`get_remat_policy`), names at `configs/base.yml:373-403` | Apache 2.0 | MaxText's policy names are tied to its tensor names. Dew's version is a `Literal` field on `TrainerConfig` mapping to `jax.checkpoint_policies` built from Dew's own sown names, applied in `ObjectiveTrainer._define_train_step` | Peak HBM and step time for each policy on one config, in `docs/performance.md`, plus loss equality to fp32 tolerance across policies |
| Goodput accounting | pattern | `maxtext common/goodput.py:31-36` (the event taxonomy), `:140-168` (the recorder), config at `configs/base.yml:1088` | Apache 2.0 | MaxText's recorder is a wrapper over `ml_goodput_measurement`, which uploads to Cloud Logging and GCM (`common/goodput.py:60,167`). Dew needs the taxonomy, not the uploader: `dew/telemetry/goodput.py` accumulates the same five event kinds and emits `perf/goodput` and `perf/badput/<cause>` through the Tracker | A test that a simulated event sequence with a known 10 second stall yields the arithmetically correct goodput, and that an unclosed event is attributed rather than dropped |
| Replica-parallel writes | depend, landing | `orbax .../experimental/v1/_src/context/options.py:342-347,405` | Apache 2.0 | Landed on `wave/adopt-small` (`57b5f82`, `src/dew/training/trainer.py`), which is correct because Dew's default mesh has replicas whenever `fsdp_size < device_count` (`src/dew/training/distributed.py:23-39`) | Landed with `tests/test_parallelism.py` |
| Emergency and multi-tier checkpointing | depend | `orbax .../experimental/emergency/checkpoint_manager.py:436` ("composes a local and a persistent checkpoint managers"), options at `:267-301` including `replica_axis_index`; MaxText's switches at `maxtext configs/base.yml:500,527,533` | Apache 2.0 | Only reachable with more than one slice or a local disk worth writing to. Wave 4.8 wires it behind one `TrainerConfig` field and keeps the existing `CheckpointManager` as the single-slice path | The acceptance run in 4.8: kill a worker, resume, and report steps lost and seconds to recover |
| Separate checkpointables and save decision policy | depend | `orbax .../experimental/v1/_src/training/checkpointer.py`, policies split into save-decision and preservation (`docs/research/google-jax-stack.md:336-341`) | Apache 2.0 | Removes the variable-length data-iterator leaf that Dew threads through its own pytree (`src/dew/training/trainer.py:286-289`) and lets Grain's own handler own iterator state | A restore test that resumes mid-epoch and consumes exactly the records the interrupted run had not seen |
| Sequence packing | depend, landing | `grain .../transformations/packing.py:341` (`FirstFitPackIterDataset`), `:446` (best fit), segment ids and positions at `packing_packed_batch.py:116-117` | Apache 2.0 | Landed on `wave/grain-packing` with a document source, packed positions and a segment mask on the decoder, and the boundary targets dropped (`wave/grain-packing` commits `92e6f78`, `5530149`, `9adfacb`) | Landed with `tests/test_text_data.py` and `tests/test_lm_objective.py`, including a test that the packed loss equals the loss of the documents alone |
| Muon | depend, landing | `optax contrib/_muon.py:379` (`scale_by_muon`), Newton-Schulz at `:280-361`, the RMS and dimension-numbers plumbing at `:151-192` | Apache 2.0 | Landed on `wave/adopt-small` as one entry in `OPTIMIZER_MAP` (`wave/adopt-small:src/dew/training/optim.py:13-18`). What is still missing is the parameter-group split the labs describe: AdamW for embeddings, heads and norms, Muon for matrices, and a `weight_dimension_numbers` spec for parameters whose matrix axes are not `(0, 1)` (`docs/research/frontier-training.md:183`) | Wave 4.9: a fixed-token run comparing Muon and AdamW at equal tokens on the same data, and a unit test that the dimension-numbers spec names every matrix in a Dew decoder |
| Muon dimension-numbers spec | pattern | `maxtext utils/muon_utils.py:100-188` (`transform_logic`, `get_muon_weight_dimension_numbers`) | Apache 2.0 | MaxText walks its own parameter paths. Dew's version walks a Dew tree and returns the same shape of answer | A test asserting the spec covers every 2D and higher parameter of `CausalTransformer` and `SimpleDiT`, and fails when a new module adds an uncovered one |
| QK-Clip | pattern | `maxtext utils/qk_clip_utils.py:31-138` (`calculate_max_logit_metric`, `apply_qk_clip`), config at `configs/base.yml:479-481` | Apache 2.0 | A post-step transform on the state that reads the maximum attention logit out of the intermediates. Dew's probe step (section 3.5) is the mechanism that can see those logits. Not scheduled: the frontier moved to norm-on-queries instead (`docs/research/frontier-training.md:186`), so this is only for an MLA mixer that cannot use QK-norm | Not scheduled, so no test is promised |
| FP8 | depend | Qwix `QtProvider` with `bwd_stochastic_rounding` (`docs/research/google-jax-stack.md:489-499`); Transformer Engine's gate is compute capability 8.9 and above for per-tensor scaling (`:503-514`) | Apache 2.0 both | Qwix applies to a Linen module by rule, with no branch inside `dew.nn` (`:480-487`). One path: Qwix for training precision on both backends; Transformer Engine stays a GPU kernel choice behind `dew.nn.attention`'s existing `implementation` argument (`src/dew/nn/attention.py:119-205`) | Wave 4.10: 2000 steps at 1B, loss within a stated fraction of the bf16 run at the same seed, plus a gemm-level equality test at the stated tolerance |
| Model definitions as parity references: Gemma 4, DeepSeek V4, Qwen3.5, Llama 4 | depend | transformers 5.16.1 in the pinned venv is the reference implementation for all four (`docs/research/model-families.md:55` Gemma 4, `:59` DeepSeek V4, `:50-51` Qwen3.5, `:70` Llama 4); MaxText ships its own JAX copies of the same four (`maxtext models/`: `gemma4`, `deepseek4`, `qwen3_5`, `llama4`, `docs/research/google-jax-stack.md:142`) | Apache 2.0 | Dew loads the real checkpoint through `dew.interop.hf_decoders` and compares logits against transformers. MaxText's files are read for their logical axis annotations, not copied. Reachability differs per family and the plan says so: Qwen3.5-0.8B is 1.7 GB and ungated, Gemma 4 E2B is 10.2 GB and ungated, Llama 4 Scout is gated with 109B as the smallest, DeepSeek V4 has no fixture that fits (`docs/research/model-families.md:1323-1331,1266,1342`) | Per family, the parity test already specified in `docs/research/model-families.md:1146-1245`: full logits at fp32 for Qwen3.5 and Gemma 4 E2B, block-level for the DeepSeek attention and router, and nothing promised for Llama 4 until the gate and the size allow one |
| Kauldron evaluators | pattern | `kauldron kauldron/evals/evaluators.py:63-143`, named evaluators as a field on the trainer (`kauldron/train/trainer_lib.py:232`) | Apache 2.0 | Kauldron needs Python 3.12 and is a Copybara export (`docs/research/google-jax-stack.md:191`). Dew has one validation loop and a list of metrics (`src/dew/eval/common.py:7-14`). The idea to take is several named evaluation suites with their own data and cadence, which is what post-training needs to score SFT and RL differently | Wave 4.13: two named evaluators on one run, with different cadences, both reaching the tracker under `val/<name>/<metric>` |
| Weight provenance hook | pattern | `kauldron kauldron/checkpoints/partial_loader.py:44` (`InitTransform`), wired as a trainer field at `kauldron/train/trainer_lib.py:219` | Apache 2.0 | Dew's version already exists in the two shapes that matter: `--pretrained` for a foreign layout and `--trainer.load-from-checkpoint` for a Dew one (`wave/hf-decoders:recipes/lm/train.py:58,105-138`). The plan keeps those two and does not add a third | Section 6's manifest records which one a stage used |
| Generation protocol | pattern | `gemma gemma/gm/nn/_transformer_like.py:79-158` (`TransformerLike`), `Output` and `init_cache` at `:42-54` | Apache 2.0 | Dew's decoder already has `init_cache` and a decode path (`src/dew/sampling/text.py:28-48`). Naming the protocol lets Dew's models be driven by gemma's samplers and gives the rollout seam a published shape to test against | A test that a Dew `CausalTransformer` satisfies the protocol structurally and produces the same tokens under gemma's greedy sampler as under `dew.sampling.text.generate` |
| GRPO and its variants | pattern | `tunix rl/algo_core.py:365-478` (`grpo_loss_fn`, including the GSPO sequence-ratio stop-gradient trick at `:471-477`), advantages at `:640-700`, GAE at `:33-113`, masked reductions at `:117-165`; config surface at `tunix rl/grpo/grpo_learner.py:47-127` | Apache 2.0 | Tunix's model interface is NNX (`docs/research/google-jax-stack.md:252`), and its learner owns the loop. What ports is the array math, which is framework-free: `masked_whiten`, `masked_mean`, the clipped ratio, the group-mean advantage, the RLOO and Dr.GRPO variants. Those become `dew/rl/advantage.py` and `dew/rl/surrogate.py` | Per function, equality against the paper's equation on a hand-computed small case, and a mutation test per surrogate (drop the clip, flip the sign of the advantage) |
| Reward orchestration | pattern | `tunix rl/reward_manager.py:45-114` | Apache 2.0 | Tunix's manager logs, aggregates and weights several reward functions. Dew's version is a list of callables and weights inside the objective, because the aggregation is three lines and the logging goes through the aux dict | A test that two rewards with known values produce the weighted sum, and that each appears in the aux dict under its own name |
| DPO | pattern | `trl trl/trainer/dpo_trainer.py`, `dpo_config.py` | Apache 2.0 | TRL is PyTorch. What transfers is the loss shape and the defaults, which post-training.md specifies | Its own document; the parity target is the loss value on a fixed batch against the TRL formula computed by hand |
| Distillation | pattern | `tunix distillation/distillation_trainer.py` and `strategies/` | Apache 2.0 | On-policy distillation is a surrogate (reverse KL on the student's own samples) plus a rollout, so it is an instance of section 5.3 rather than a trainer | A test that the KL surrogate is zero when student and teacher parameters are equal, and positive otherwise |
| Environments | depend | MuJoCo Playground on MJX; Brax's README states "Only `brax/training` is actively being maintained as of 0.13.0" and directs environment users to MuJoCo Playground and MJX (`https://github.com/google/brax/blob/main/README.md`) | Apache 2.0 | An adapter in `dew/rl/env.py` that presents a Playground environment as Dew's `Env` protocol. Brax stays as a reference for the PPO update rule only | Wave 4.12: a PPO run on a named Playground task reaching a stated return |
| Dreamer v3 | fresh | arXiv 2301.04104; the reference implementation is not JAX-Flax and its own code owns its loop | n/a for a fresh write | The RSSM, symlog and two-hot pieces are written from the paper against Dew's seams; the image decoder reuses `dew.nn.autoencoders`, the EMA target critic reuses `EMASpec` with a subtree path (`src/dew/objectives/base.py:21-31`), and imagination reuses `Env` | Wave 4.12: a stated return on a named DMC task at 100k environment steps, compared against the paper's figure for the same task |
| Dreamer v4 | fresh, unspecified | arXiv 2509.24527: shortcut forcing, a tokenizer plus an efficient transformer dynamics model, real-time inference on one GPU | n/a | Dew has the substrate: flow matching schedules and samplers in `dew.diffusion`, a DiT in `dew.nn.dit`, block-causal attention as a mask (`maxtext configs/base.yml:411` lists `block_diffusion` as an attention type). The shortcut forcing objective is not specified in this plan; risk 10.7 names the reading pass and the experiment that specifies it | Open-loop prediction error against a diffusion-forcing baseline on the same small video dataset |
| Kernels beyond the above | skip | DeepSeek's FlashMLA and DeepEP, MiniMax's MSA kernels, FlashKDA | MIT and others | All are CUDA behind PyTorch bindings with no JAX path (`docs/research/frontier-training.md:207`) | Not applicable |
| MaxText, Kauldron, Tunix as dependencies | skip | `docs/research/google-jax-stack.md:157,191,266` | Apache 2.0 | MaxText is a program, Kauldron requires Python 3.12 and is a peer of `ObjectiveTrainer`, Tunix is NNX with a placeholder release | Not applicable |

### 2.2 The rule this table follows

A lift is only allowed when the copied body has no types from its source
framework in it. Two candidates pass that bar: the load-balance bias update
(`maxtext layers/moe.py:238-261`, plain jax) and the masked reductions in
`tunix rl/algo_core.py:117-165`. Everything else in the table is either a
dependency or a pattern, because the source code is bound to NNX modules, to a
config object with forty fields, or to a loop Dew already owns. That is the
honest answer to "port code from MaxText": the valuable part of MaxText is its
configuration surface and its arithmetic, and both transfer as design.

## 3. Telemetry

One design, three parts: a seam to write through, a vocabulary to write, and
measurement sources that produce numbers.

### 3.1 The Tracker seam

```python
class Tracker(Protocol):
    def log(self, metrics: Mapping[str, float], step: int) -> None: ...
    def media(self, key: str, value: Any, step: int) -> None: ...
    def summary(self, key: str) -> Any: ...
    def set_summary(self, key: str, value: Any) -> None: ...
    def artifact(self, path: str, name: str, kind: str, aliases: Sequence[str]) -> None: ...
```

Five methods, because five things are done with wandb today: metric logging
(`src/dew/training/trainer.py:505-510`), images and tables from validation
(`src/dew/objectives/base.py:66`), reading the resume step out of the run summary
(`src/dew/training/trainer.py:129`), writing run-level bests, and pushing the
checkpoint artifact to the registry (`src/dew/training/objective_trainer.py:557`).

Implementations: `WandbTracker` (the default, exactly today's behaviour) and
`NullTracker` (tests and offline runs). A third backend is a new file and no
trainer edit, which is the only reason the protocol exists.

Cadence declarations move out of the trainer. `wandb.define_metric` calls
(`src/dew/training/trainer.py:145-153`) are the wandb spelling of a general
fact: some metrics are per step and some are per epoch. That fact becomes the
vocabulary table below, and `WandbTracker` turns it into `define_metric` calls
at construction.

### 3.2 The metric vocabulary

Metric keys are frozen by `CONTRIBUTING.md:11`, so this table is the migration
plan and the contract at once. Existing keys keep their names exactly.

| Key | Cadence | Source | Status |
| --- | --- | --- | --- |
| `train/step`, `train/epoch` | step, epoch | the loop | exists (`trainer.py:145-146`) |
| `train/loss` | step | objective scalar | exists (`:507`) |
| `train/<aux>` | step | the objective's aux dict, verbatim | exists (`:508`) |
| `train/step_time_ms`, `train/samples_per_sec` | step | host clock over the log interval | exists (`:559-560`) |
| `train/mfu` | step | HLO FLOPs over step time | exists (`:562-564`), numerator changes in 3.4 |
| `train/epoch_time`, `train/avg_time_per_step`, `train/avg_loss`, `train/best_loss` | epoch | the loop | exists (`:642-646`) |
| `val/<metric>` | epoch | `EvaluationMetric` reduction | exists (`:399-400`) |
| `val/<suite>/<metric>` | its own | named evaluators (2.1, Kauldron pattern) | new |
| `train/grad_norm/<group>` | step | global norm per optimizer parameter group | new |
| `train/param_norm/<group>`, `train/update_norm/<group>` | step | same reduction over params and updates | new |
| `train/rollout_reward`, `train/rollout_gen_len` | step | the rollout batch's own columns, through the aux dict | new, named identically in post-training.md |
| `train/rollout_seconds` | step | host clock around the host-side rollout call, merged at the log tick | new, named identically in post-training.md |
| `probe/<name>` | probe cadence | the probe step (3.5) | new |
| `perf/goodput`, `perf/badput/<cause>` | its own timer | the goodput accumulator | new |
| `perf/hbm_peak_gb`, `perf/compile_seconds` | run and on change | JAX device stats, compile timing | new |
| `data/records_per_sec`, `data/tokens_per_sec`, `data/queue_wait_ms` | step | the prefetch iterator and Grain | new |

The namespace rule, so that a new metric has one obvious home: `train/` is
about the model, `val/` about its quality, `probe/` about its internals,
`perf/` about the machine, `data/` about the pipeline. A metric that would fit
two namespaces is two metrics.

JEPA's collapse telemetry is the worked example of why objective-provided
metrics need no new path: `representation_health` returns
`{"repr_std", "repr_cov_offdiag"}` from `loss`
(`src/dew/objectives/jepa/objective.py:38-58,148`) and arrives as
`train/repr_std` with no trainer knowledge. Every objective-owned probe uses
that road: reward statistics, entropy, KL to the reference, router load,
expert utilisation.

### 3.3 Cadence

Four clocks, each with one owner.

| Clock | Owner | Today |
| --- | --- | --- |
| Every step | the compiled step returns aux; the host logs at the log interval | `trainer.py:493-511`, one device sync per interval |
| Every epoch | the loop | `trainer.py:641-646` |
| Probe cadence | a separate compiled step, run every `ProbeSpec.every` steps | new |
| Wall-clock cadence | the goodput accumulator and the profiler window | profiler exists (`trainer.py:456-467`) |

The rule that keeps this from becoming a pile of flags: nothing is added to the
per-step path unless its cost is measured and under a stated fraction of step
time. Section 4.3's acceptance run states that fraction for gradient norms.

### 3.4 Measurement sources

| Number | How it is obtained | Change from today |
| --- | --- | --- |
| Step FLOPs | parse the optimized HLO and sum dots, convolutions and known custom calls (`wave/adopt-small:src/dew/telemetry/instrumentation.py:338` `hlo_flops`) | `compiled_flops`, which reads `cost_analysis()`, is deleted. It undercounts convolutions by 22.5x and varies between identical recompiles (`docs/research/benchmark-parity.md:93-102`) |
| Peak FLOPs per device | the table at `src/dew/telemetry/instrumentation.py:10-24` | unchanged, with the caveat from `docs/research/benchmark-parity.md:123-125` written next to it: the spec number is not a sustained ceiling, and small shapes reach 49 to 93 TFLOP/s on this card |
| MFU | FLOPs per step over step time over peak, one SPMD partition (`src/dew/telemetry/instrumentation.py:53-66`) | unchanged arithmetic, honest numerator |
| Goodput | an accumulator over the five event kinds patterned from `maxtext common/goodput.py:31-36`, with badput attributed to its cause | new, fresh code, no GCP dependency |
| Profiler window | `jax.profiler.start_trace` once per run after a warmup (`trainer.py:456-467,530-538`) | becomes a `ProfileSpec` with start step and length, and the window can be requested again on demand; kernel time, not wall time, is what the trace is read for (`docs/research/google-jax-stack.md:296`) |
| Data throughput | Grain's own read options and a wait timer inside `DevicePrefetchIterator` (`src/dew/training/distributed.py:78-126`) | new: `data/queue_wait_ms` is the number that says whether the pipeline is the bottleneck, which no current metric answers |
| Activation and gradient statistics | 3.5 | new |

### 3.5 Probing

The requirement is activation statistics, gradient norms per parameter group,
and objective-specific collapse telemetry, through one path and without a flag
per statistic.

Three mechanisms, chosen by cost.

1. Anything the objective already computes goes in the aux dict. This covers
   collapse telemetry, KL, entropy, reward statistics, router load.
2. Gradient and parameter norms per group are reductions over trees the
   trainer already holds. They are computed in the compiled step when
   `ProbeSpec.norms` is on. Grouping reuses the optimizer's own label tree, so
   `train/grad_norm/matrices` and `train/grad_norm/embeddings` mean exactly what
   Muon's parameter split means.
3. Activations inside a model are captured with Linen's
   `apply(..., capture_intermediates=...)`, which needs no edit to any module
   in `dew.nn`. Because capture keeps every captured output alive, it runs on a
   second compiled function, the probe step, at `ProbeSpec.every` steps. The
   probe step is forward-only, so it costs one extra forward pass per probe and
   nothing per training step.

`ProbeSpec` is one dataclass with three fields: `every`, `norms`, `capture`
(a filter over module paths). The deletion test is in 1.4.

### 3.6 What each existing piece becomes

| Today | Becomes |
| --- | --- |
| `self.wandb` in `SimpleTrainer` | `self.tracker`, a `Tracker`; `WandbTracker` is constructed in the recipe, not in the trainer |
| `wandb.define_metric` calls (`trainer.py:145-153`) | rows in `dew/telemetry/keys.py`, read by `WandbTracker` |
| `_throughput_metrics` (`trainer.py:554-565`) | `dew/telemetry/throughput.py`, returning keys from the table |
| `Objective.log_validation_artifacts(wandb, ...)` (`objectives/base.py:66`) | `log_validation_artifacts(tracker, ...)`. This is a change to a frozen `Objective` method, so it is a migration: all three objectives change in the same commit, `CONTRIBUTING.md:11` is updated to name `tracker`, and a test constructs each objective and calls the method with a `NullTracker` |
| `compiled_flops` (`instrumentation.py:27-38`) | deleted, replaced by `hlo_flops` from `wave/adopt-small` |
| `PEAK_FLOPS_PER_DEVICE` (`instrumentation.py:10-24`) | stays, moves next to `dew/telemetry/devices.py` |
| the profiler block in `train_loop` (`trainer.py:456-467`) | `dew/telemetry/profile.py` holding `ProfileSpec` and the window state machine; the loop asks it whether to start and stop |
| `_check_finite` (`trainer.py:540-552`) | unchanged in behaviour, and its non-finite streak becomes `perf/badput/nonfinite` so a diverged run shows up in goodput |
| registry artifact push (`objective_trainer.py:557`) | `tracker.artifact(...)` |

## 4. The scale roadmap

Each wave is one branch, one reviewer and one acceptance run. An acceptance run
is a training run with a named model and a named dataset, not only a unit test.
Unit tests are still required by `CONTRIBUTING.md:31-41`; they are not
sufficient here.

### 4.1 In flight today

Seven branches are open and under review. They are the base every wave below
assumes, and their merge order matters because they touch the same files.

| Branch | Head | Contributes | Touches that collide |
| --- | --- | --- | --- |
| `wave/sharding-rules` | `b6fef2d` | logical axis rules, spec canonicalisation, sharding tolerance | `config/__init__.py`, `training/distributed.py`, `training/trainer.py`, `tests/test_parallelism.py` |
| `wave/adopt-small` | `00c2e56` | Muon in the optimizer map, HLO FLOP counter, replica-parallel writes, 100-step benchmarks | `config/__init__.py`, `training/trainer.py`, `telemetry/instrumentation.py`, `tests/test_parallelism.py` |
| `wave/grain-packing` | `d0c9b96` | document source, Grain packing, packed positions and segment mask, boundary targets dropped | `objectives/lm/objective.py`, `tests/test_lm_objective.py` |
| `wave/lm-head-perf` | `814fa07` | head split off the trunk, chunked cross entropy with kept tiles | `objectives/lm/objective.py`, `nn/backbones/causal_transformer.py`, `tests/test_lm_objective.py` |
| `wave/hf-decoders` | `13a174e` | HF config and weight translation, `load_pretrained_decoder`, `--pretrained`, Qwen3, Gemma 3, Llama parity fixtures | `nn/backbones/causal_transformer.py`, `recipes/lm/train.py` |
| `wave/hf-interop` | `fbcc56f` | HF datasets as a Grain source, export directory push and pull | `data/registry.py`, `interop/` |
| `wave/tpu-cli` | merged | `dew-tpu` | none |

Merge order: `sharding-rules`, then `adopt-small`, then `grain-packing`, then
`lm-head-perf`, then `hf-decoders`, then `hf-interop`. The two sharding-adjacent
branches go first because every later wave reads the rules table; the two LM
branches go next and in that order because the chunked head is easier to rebase
onto packing than the reverse.

### 4.2 Wave map

| Wave | Name | Branch | Level |
| --- | --- | --- | --- |
| 4.3 | Telemetry: Tracker, vocabulary, goodput, probes | `wave/telemetry` | task, expert review |
| 4.4 | Recipe layer | `wave/recipes` | task |
| 4.5 | Sequence and tensor axes | `wave/parallel-axes` | expert |
| 4.6 | Multi-host on v5e-16 | `wave/multi-host` | task, expert review of the parity claim |
| 4.7 | MoE and expert parallelism | `wave/moe` | expert |
| 4.8 | Fault tolerance | `wave/fault-tolerance` | task, expert review |
| 4.9 | Optimizer at scale: Muon parameter split, WSD | `wave/optim-scale` | task |
| 4.10 | Quantized training (FP8) | `wave/fp8` | expert |
| 4.11 | Checkpoint format for the largest models | `wave/checkpoint-v1` | expert |
| 4.12 | RL core | `wave/rl-core` | expert |
| 4.13 | Post-training stages | `wave/post-training` | task, expert for the loss ports |
| 4.14 | Diffusion RL: FlowGRPO and DiffusionNFT | `wave/diffusion-rl` | expert |
| 4.15 | Environments and world models | `wave/world-models` | expert |

### 4.3 Telemetry

Contents: section 3 in full. Depends on `wave/adopt-small` for `hlo_flops`.

Acceptance run: 2000 steps of the 85M decoder (12 layers, 768 wide, vocabulary
50,304, the large `benchmark_step.py` preset, `docs/research/benchmark-parity.md:26`)
on FineWeb-Edu `sample-10BT` tokenized with `tools/tokenize_text.py`, on the
RTX 4080, wandb offline. The run must show `perf/goodput` above 0.95,
`train/mfu` from the HLO counter, `data/queue_wait_ms`, `train/grad_norm/*` for
two parameter groups, and `probe/` statistics from a probe step every 200 steps.
Then the same 2000 steps with `NullTracker` and probes off: step time within 1%
of the first run, and the measured cost of `ProbeSpec.norms` written into
`docs/performance.md`. If gradient norms cost more than 2% of step time, they
move behind the probe cadence instead.

One prerequisite, because every wave below trains on a real corpus and today's
path stops short of one. `tools/tokenize_text.py` reads `.txt` files and, with
`--pack` on `wave/grain-packing`, writes an end-of-sequence id after every
input file, so a document is a file (`wave/grain-packing:tools/tokenize_text.py`
docstring). A corpus like FineWeb-Edu arrives as dataset records, not as
millions of files. The addition is one argument on that tool: read records from
an `HFDatasetSource` (`wave/hf-interop:src/dew/data/sources/hf.py:50`) and emit
one document per record. Roughly thirty lines, it belongs to whichever wave
lands first, and its test is that the token count and the document count match
the dataset's own.

### 4.4 Recipe layer

Contents: section 6. Depends on 4.3 only for the run identifier in the
manifest.

Acceptance run: the three-stage recipe from section 7 executed end to end on
the 4080: 500 steps of pretraining on the byte-tokenized Shakespeare corpus
(`README.md:419`), 200 steps of SFT, 100 steps of DPO, each stage a separate
process, each writing a manifest that names its parent. Then `dew recipe verify`
reports drift after one config field is edited, and `dew recipe export` prints
three commands that reproduce the chain.

### 4.5 Sequence and tensor axes

Contents: `build_mesh` grows `sequence` and `tensor` axes; the rules table
grows the entries that map `sequence` and `heads`, `mlp`, `vocab` onto them;
`BATCH_SPEC` grows the sequence dimension (`src/dew/training/distributed.py:16`).
Nothing else. The whole point of the rules table is that an axis is a
configuration change (`docs/research/google-jax-stack.md:71`).

Acceptance run: on the simulated eight-device CPU mesh, the same 50 steps of the
85M decoder at the same seed under four mesh configurations, `fsdp=8`,
`fsdp=4,tensor=2`, `fsdp=4,sequence=2`, `fsdp=2,tensor=2,sequence=2`, with
losses equal to 1e-6 and the `sharding_tolerance` assertion passing in each. On
the 4080, one long-context run that does not fit without the sequence axis, at
the sequence length where it starts to fit. The wave also produces the table of
step times per configuration on v5e-16 as part of 4.6, which is the evidence
that decides whether the tensor axis is worth using at Dew's scale. A result of
"tensor parallelism does not pay below N chips" is an acceptable outcome and is
what `docs/research/frontier-training.md:208` predicts.

### 4.6 Multi-host on v5e-16

Slice: `v5e-16`, spelled `v5litepod-16` on the wire, two workers, sixteen chips,
16 GB per chip. Reason: it is the smallest slice with a second host and a real
ICI, `dew-tpu create` already builds it, and `dew-tpu run` already launches the
same command on both workers (`docs/tpu.md:17-23,58-70,147-154`).

Model and data: continue-train `Qwen/Qwen3-0.6B` loaded through
`load_pretrained_decoder` (`wave/hf-decoders:src/dew/interop/hf_decoders.py:413`)
on FineWeb-Edu `sample-10BT`, packed, sequence length 4096, `fsdp=16`.

Parity with single-host: the same 200 steps at the same global batch and the
same seed on one `v5e-8` host with `fsdp=8`, and on the two-host slice with
`fsdp=16`. Losses agree to 1e-3 per step and the mean absolute difference over
the 200 steps is reported. Bitwise equality is not claimed and is not expected:
the collective order changes. A second check that is exact: a checkpoint saved
on sixteen chips restores on eight and the parameters compare equal.

Also in this wave: the process-count assumptions stay in the two files that
have them today, `dew.data.dataloaders` and `dew.training.distributed`
(`docs/research/google-jax-stack.md:634-638`), so a later move to a
single-controller runtime is a change in two files.

### 4.7 MoE and expert parallelism

Contents: `dew/nn/moe.py` with `Router` and `ExpertMLP`, the aux-loss-free bias,
fp32 router logits, the `expert` mesh axis and its rule, the grouped matmul
behind an `implementation` argument, and a `moe` architecture entry in the
registry.

Parity, in the order it can be proved:

1. `Router` against `MixtralSparseMoeBlock` and `DeepseekV3MoE` at fp32 on CPU
   with copied weights, largest observed difference in the test
   (`docs/research/model-families.md:1322`).
2. Gemma 4's routing, which is softmax in fp32, top-8 of 128, renormalised, and
   then multiplied by a learned per-expert scale, with the dense branch summed
   in (`docs/research/model-families.md:55`). Proved at the block level against
   transformers at fp32 on a small config, not on the real checkpoint.
3. The real `google/gemma-4-26B-A4B-it` is 25.2B parameters. Full fp32 logits
   parity does not fit this workstation and is not promised. What is promised is
   a two-layer hidden-state comparison on CPU if host RAM allows, and risk 10.4
   names the measurement that decides.

Acceptance run: a small MoE decoder, 8 experts, top-2, trained for 2000 steps on
FineWeb-Edu on v5e-16 with `expert=4, fsdp=4`, showing router load balanced
within a stated band, `train/expert_utilisation` in the aux dict, and a loss
curve below the dense model of the same active parameter count on the same
tokens.

### 4.8 Fault tolerance

Contents: Orbax emergency and multi-tier checkpointing behind one
`TrainerConfig` field, a local checkpoint period, single-replica restore
broadcast, and the goodput accumulator from 4.3 wired to the restart path.

Acceptance run: a 2000-step run on v5e-16 in which worker 1 is killed with
`dew-tpu reset` at step 900. The run resumes from the local checkpoint, loses
fewer steps than the local checkpoint period, and reports `perf/goodput` and
`perf/badput/preemption` for the whole run. Both numbers go in
`docs/tpu.md`.

### 4.9 Optimizer at scale

Contents: the parameter-group split Muon needs (AdamW for embeddings, heads and
norms, Muon for matrices), the `weight_dimension_numbers` spec patterned from
`maxtext utils/muon_utils.py:188`, weight decay on norm scales
(`docs/research/frontier-training.md:184`), and a `wsd` schedule beside
`cosine` in `OptimConfig` (`src/dew/config/__init__.py:78-93`).

Acceptance run: two runs of the 0.4B decoder on FineWeb-Edu at an equal token
budget, one AdamW and one Muon with the split, on v5e-16. The comparison is the
loss at equal tokens, and the numbers go in `docs/performance.md` with the
commands. A negative result is a result.

### 4.10 Quantized training

Contents: Qwix `QtProvider` applied by rule at trainer construction, fp32
master weights kept, first and last layers excluded
(`docs/research/frontier-training.md:199`), and Transformer Engine left as a
GPU kernel option behind the existing attention `implementation` argument.

Acceptance run: 2000 steps at 1B on v5e-16, fp8 against bf16 at the same seed.
Accept if the loss curves agree within a stated fraction and step time
improves; report both. A gemm-level equality test at the stated tolerance ships
with it.

### 4.11 Checkpoint format for the largest models

Contents: the Orbax v1 API, model state and optimizer state and data iterator
as separate checkpointables, Grain's own handler for the iterator, OCDBT with a
2 GB target data file size (`maxtext configs/base.yml:83`), single-replica
restore, and a resharding tool under `tools/`.

Acceptance run: save and restore a 26B-A4B-shaped parameter tree at fp32 across
sixteen chips, restore it onto eight, compare parameters equal, and report save
seconds and bytes written against today's path. Plus the migration test
`CONTRIBUTING.md:11` requires: a checkpoint written by today's layout loads
under the new one.

### 4.12 RL core

Contents: section 5. Depends on 4.3 (rollout metrics), 4.4 (stage chaining) and
`wave/hf-decoders` (a real policy to start from).

Acceptance run: GRPO on GSM8K with `Qwen/Qwen3-0.6B` as the policy, 200 update
steps, group size 8, reward exact-match on the final answer. Accept if
`train/rollout_reward` rises from the base model's measured baseline by a stated
margin and the sampled completions are logged. The run also reports
`train/rollout_seconds` over `train/step_time_ms`, which is the experiment for
risk 10.1.

### 4.13 Post-training stages

Contents: SFT, DPO and GRPO as Recipe stages, in whatever order
`docs/design/post-training.md` decides. This plan owns only the chaining
(section 7).

Acceptance run: the three-stage chain on `Qwen/Qwen3-0.6B`, each stage scored by
its own named evaluator, with the eval numbers per stage in one table and the
manifest chain intact.

### 4.14 Diffusion RL

Contents: FlowGRPO (arXiv 2505.05470) as a `PolicyGradientObjective` instance
with an SDE rollout and a per-step Gaussian log probability, and DiffusionNFT
(arXiv 2509.16117) as an instance with no log probability, no reference and a
reward-weighted flow-matching surrogate.

Acceptance run: the flowers DiT (`oxford_flowers102`, already in the data
registry at `src/dew/data/registry.py:120`) fine-tuned by both methods against
one reward model, 500 steps each, on the 4080. Report the reward curve for both
and the steps each needed to reach the same reward. The papers claim NFT is up
to 25 times more sample-efficient than FlowGRPO; the acceptance is our own
number on our own model, whatever it says.

### 4.15 Environments and world models

Contents: the `Env` protocol, the MuJoCo Playground adapter, `env_rollout`,
`ReplayBuffer`, PPO as a `PolicyGradientObjective` instance with a critic head,
then Dreamer v3 as an actor-critic objective over a learned `Env`.

Acceptance runs, in order:

1. PPO on a named Playground task from the DM Control suite, reaching a stated
   return within a stated wall-clock time on the 4080.
2. Dreamer v3 on the same task at 100k environment steps, compared against the
   figure in arXiv 2301.04104 for that task, with the gap reported.

Dreamer v4 is not in this wave. Risk 10.7 names what has to happen first.

## 5. Reinforcement learning

### 5.1 Package decision: `dew.rl` in this repository

The choice is between `dew.rl` as a subpackage and `dew-rl` as a separate
distribution.

Coupling argument for the subpackage. Every RL objective needs
`dew.objectives.base.Objective`, `dew.sampling`, `dew.nn` and the trainer's EMA
mechanism. A separate distribution would depend on `dew-ml` for all four, so the
dependency does not disappear, it only gains a release boundary. Meanwhile the
core needs exactly one thing from RL, the `rollout` method, and that method is
in the core anyway because it is useful without RL: self-distillation and
curriculum filtering are non-RL objectives that want to modify their own batch.

Cost of the subpackage: `dew-ml` gains optional dependencies for environments.
That is handled the way the repository already handles five other optional
groups (`pyproject.toml:42-53`): an `rl` extra holding `mujoco_mjx` and
`mujoco_playground`, imported lazily inside the adapter, exactly as
`dew.data` already does for AV readers.

The invariant that makes the decision reversible: nothing under `src/dew`
outside `src/dew/rl` and `src/dew/objectives/rl` may import `dew.rl`. A test
walks the module tree and asserts it. If `dew.rl` later grows its own process
model, a distributed sampler and its own deployment story, the split is then a
directory move and a `pyproject.toml` entry, because the arrow has never
pointed the other way.

### 5.2 The primitives

| Primitive | Dew form | When it is not a separate thing |
| --- | --- | --- |
| Policy | the objective's parameter tree and the model's `apply`; there is no `Policy` class | always |
| Generator or environment | `Env` protocol in `dew/rl/env.py`: `reset(rng) -> TimeStep`, `step(state, action) -> TimeStep`, both pure and jittable, `TimeStep` a `flax.struct.dataclass` of observation, reward, done and state | text and diffusion rollouts have no environment; the generator there is `dew.sampling` |
| Rollout | `Objective.rollout(params, ema_params, batch, rng, step) -> batch`, host side, fixed shapes; implementations in `dew/rl/rollout.py` for text, environment and SDE sampling | offline algorithms (SFT, DPO) use the default, which returns the batch unchanged |
| Rollout buffer | `ReplayBuffer` in `dew/rl/buffer.py`: a fixed-capacity device pytree with `insert` and `sample`, including sequence sampling for model-based use | every on-policy algorithm omits it |
| Reward | a host-side callable per sample, `(record, completion, completion_ids) -> float` for text and `(record, image) -> float` for images, in a list on the objective with weights; verifiable examples in `dew/rl/reward.py` | when the environment supplies the reward it is already a batch column |
| Advantage estimator | functions in `dew/rl/advantage.py`: `group_mean`, `group_mean_unnormalised`, `rloo`, `gae`, `lambda_returns` | DPO and DiffusionNFT pass the reward through unchanged |
| Update rule | functions in `dew/rl/surrogate.py`: `clipped_ratio`, `sequence_ratio`, `cispo`, `preference_logsigmoid`, `weighted_flow_matching`, `reverse_kl`, `reinforce_entropy` | never; this is the one primitive every algorithm uses |
| Reference model | the existing EMA tree with `EMASpec(decay=lambda step: 1.0)` (`src/dew/objectives/base.py:21-31`), read in `loss` as `ema_params` | DiffusionNFT and Dreamer have no reference and set no spec |
| World model | an `Objective` with its own loss that also implements `Env` | model-free algorithms have none |

The batch key vocabulary is frozen the way metric keys are, and is identical to
the one in `docs/design/post-training.md`: `prompt_ids`, `completion_ids`,
`response_mask`, `old_log_probs`, `reward`, `advantages`, and `values` when a
critic exists. Control adds `obs`, `actions`, `dones`. Diffusion adds
`latents`, `timesteps`, `noise` and `step_log_probs`.

Two device-side slots complete the design, and they are what makes the
algorithms instances rather than subclasses:

```
policy_forward(params, batch) -> out      # per-token log probs, or a velocity prediction
surrogate(out, batch) -> (loss, aux)      # the update rule
```

### 5.3 The algorithms as instances

`PolicyGradientObjective(model, generate, rewards, advantage, policy_forward, surrogate, reference=..., critic=None, group=1)`.
Identity means the slot is filled by a function that returns its input.

| Algorithm | Generator | Reward | Advantage | policy_forward | Surrogate | Reference | Critic | Buffer |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PPO | `env_rollout` | environment | `gae` | token log probs | `clipped_ratio` | optional KL | value head | no |
| GRPO | text sampler, G per prompt | verifiable | `group_mean` | token log probs | `clipped_ratio` | EMA at 1.0 | none | no |
| Dr.GRPO | same | same | `group_mean_unnormalised` | same | same | same | none | no |
| RLOO | same | same | `rloo` | same | same | same | none | no |
| GSPO | same | same | `group_mean` | same | `sequence_ratio` | same | none | no |
| CISPO | same | same | `group_mean` | same | `cispo` | same | none | no |
| DAPO | same, with group filtering in the rollout | same | `group_mean` | same | `clipped_ratio` with asymmetric epsilon | same | none | no |
| FlowGRPO | SDE sampling of the flow model | reward model | `group_mean` | per-step Gaussian log prob | `clipped_ratio` | EMA at 1.0 | none | no |
| DiffusionNFT | any sampler, no log probs | reward model | identity | velocity prediction on noised samples | `weighted_flow_matching` | none | none | no |
| On-policy distillation | student samples | identity (no reward) | identity | token log probs | `reverse_kl` to the teacher | teacher as the reference tree | none | no |
| Dreamer v3 actor | `env_rollout` against the world model | learned reward head | `lambda_returns` | action log probs in latent space | `reinforce_entropy` | none | critic head with an EMA target | yes, for the world model |
| DPO family | identity | identity (the preference label is in the data) | not used | chosen and rejected log probs | `preference_logsigmoid` | EMA at 1.0 | none | no |

The last row is the honest one. DPO is not a policy-gradient instance: it has no
rollout, no reward and no advantage, so forcing it through
`PolicyGradientObjective` would mean three identity slots and a surrogate that
ignores the advantage the fourth slot produced. It is its own small objective in
`dew/objectives/rl/preference.py` and it shares exactly three things with the
others: the reference mechanism, the log-probability helper and
`surrogate.preference_logsigmoid`. Eleven of the twelve rows are instances of
one object; one is not, and saying so is cheaper than a false unification.

### 5.4 How it composes with `ObjectiveTrainer`

The trainer changes in exactly two places, both agreed with
`docs/design/post-training.md`.

1. `train_loop` calls `batch = objective.rollout(params, ema_params, batch, key, step)`
   after `next(train_ds)` and before the compiled step, times the call on the
   host, re-shards the returned batch with `shard_batch`
   (`src/dew/training/distributed.py:72-75`), and merges
   `train/rollout_seconds` into the log tick the way `_throughput_metrics`
   already merges `train/step_time_ms` (`src/dew/training/trainer.py:493-511,554-565`).
2. `_define_train_step` omits the EMA apply entirely when the spec's decay is
   identically 1.0, decided at trace time. A reference then costs one parameter
   copy in HBM and no per-step work, and a non-finite parameter can no longer
   reach the frozen tree through a multiply by zero.

Everything else is unchanged. Gradient accumulation, the EMA clock, checkpoints
and sharding all keep working because a rollout happens once per micro-batch and
produces a fixed-shape batch (`src/dew/training/objective_trainer.py:278-287`).

The critic is a parameter subtree in the objective's own tree, and its loss is
another term in the same scalar. Three learning rates for three parameter groups,
which Dreamer needs, is `optax.multi_transform` with a label tree, so it is one
`GradientTransformation`, one optimizer state and one compiled step. The cost of
that choice, stated plainly: the three losses are summed, so their relative
weight is a constant, not a separate step schedule. Dreamer v3's own
configuration differs between the three only in learning rate, which
`multi_transform` expresses exactly.

### 5.5 Environments

`Env` is two pure functions and a pytree, which is the shape every JAX-native
simulator already has. First integration: MuJoCo Playground on MJX. The reason
is in Brax's own README: "Only `brax/training` is actively being maintained as
of 0.13.0", with environment users directed to MuJoCo Playground and physics
users to MJX
(`https://github.com/google/brax/blob/main/README.md`). So Brax contributes its
PPO implementation as a reference for the update rule, and MJX through Playground
contributes the environments.

The adapter is small: Playground environments already return an observation, a
reward and a done flag from a jittable step, so the adapter renames fields into
`TimeStep` and nothing more. Vectorisation is `jax.vmap` over the environment
state, and the batch axis is sharded like any other batch
(`src/dew/training/distributed.py:16`).

### 5.6 World models

The design claim, and the reason `Env` earns its place: a world model is an
`Env` whose state is a latent and whose `step` is a learned function. Then
imagination training is the same `env_rollout` call as environment training,
with a different `Env`. Dreamer's two loops become one primitive used twice.

What Dreamer v3 needs from what Dew already has:

| Dreamer v3 piece | Dew today | Gap |
| --- | --- | --- |
| Latent dynamics | `dew/nn/ssm.py` is a sequence model, not an RSSM | the RSSM (recurrent state, categorical posterior and prior heads) is fresh code against `dew.nn` primitives |
| Image encoder and decoder | `dew/nn/autoencoders/` | reuse the decoder architecture; the loss is the world model's |
| EMA target critic | `EMASpec` with a subtree path (`src/dew/objectives/base.py:21-31`), the exact mechanism JEPA uses for its target encoder (`src/dew/objectives/jepa/objective.py:94-97`) | none |
| Symlog and two-hot returns | absent | fresh, small, testable against the paper's equations |
| Imagination rollout | `env_rollout` from 5.5 | none |
| Replay | `ReplayBuffer` | none |

What Dreamer v4 (arXiv 2509.24527) needs: a video tokenizer, which is
`dew.nn.autoencoders`; a transformer dynamics model with block-causal attention,
which is `dew.nn.dit` plus a mask that Dew's attention already accepts
(`docs/research/model-families.md:1286-1293`); and the shortcut forcing
objective, which this plan does not specify. Risk 10.7 names the reading pass
and the experiment. The rest of Dreamer v4, the actor-critic in imagination, is
the same code as v3.

The diffusion side contributes more than the tokenizer: the flow-matching
schedules and samplers in `dew.diffusion` are the substrate a shortcut-forcing
world model needs, and 4.14's diffusion RL work builds the reward and advantage
plumbing that a world-model agent then reuses. That overlap is the argument for
one repository rather than two.

### 5.7 What Dew's RL does not do

| Not doing | Why |
| --- | --- |
| A distributed rollout service | The rollout is a host-side call in the training process. Risk 10.1 names the measurement that would justify a decoupled sampler, and it would be a new seam, not a change to `rollout` |
| A vLLM or SGLang integration | Same reason. Dew's own sampler is the generator until measured otherwise |
| A learner and actor process split | Same |
| Multi-turn agent and tool-use scaffolding | Tunix ships this (`docs/research/google-jax-stack.md:250`). It is a data and reward concern, and it comes after single-turn works |
| An off-policy replay algorithm zoo (SAC, TD3) | `ReplayBuffer` makes them possible; nothing in the roadmap needs them |

## 6. The Recipe layer

### 6.1 What it is

```python
@dataclass(frozen=True)
class Stage:
    name: str
    recipe: str                          # "lm", "diffusion", "jepa"
    overrides: Mapping[str, str]         # exactly the flags the recipe's tyro CLI takes
    init_from: str | None = None         # a previous stage's name

@dataclass(frozen=True)
class Recipe:
    name: str
    stages: tuple[Stage, ...]
```

A `Recipe` is a Python object in a file under `recipes/`, not a new
configuration language. It composes what already exists: each stage names one of
the existing entrypoints (`recipes/lm/train.py:225-226` parses `LmRunConfig`
with tyro) and its flags. `init_from` is the only thing the layer resolves: the
named stage's output checkpoint directory becomes the next stage's
`--pretrained` or `--trainer.load-from-checkpoint`. Both accept a Dew checkpoint
directory or an exported HF-layout directory, dispatched on the directory's
contents, which is agreed with post-training.md.

Stages run as separate processes. The reason is not isolation for its own sake:
a stage that changes the model shape needs a fresh JAX process, and a crash in
stage three must not leave stage two's donated buffers in an unknown state.

### 6.2 Provenance

Each stage writes `recipe.json` into its run directory:

| Field | Value |
| --- | --- |
| `recipe`, `stage` | names from the `Recipe` |
| `command` | the exact argv |
| `config_sha256` | over `RunConfig.to_dict()` (`src/dew/config/__init__.py:151-167`) |
| `git` | commit and whether the tree was dirty |
| `dataset` | for token directories, the hash of `meta.json` (`recipes/lm/train.py:60-67`); for HF sources, name, split and revision; for GCS sources, the prefix and shard count |
| `parent` | the parent stage's manifest path and its hash |
| `run_id` | the tracker's run identifier |
| `outputs` | checkpoint directory, exported directory if any |

`dew recipe run` executes the chain, `dew recipe export` prints the commands,
`dew recipe verify` recomputes the hashes and reports what drifted. `--resume`
skips a stage whose manifest matches its recomputed inputs.

Those three verbs are a second console script beside `dew-tpu`
(`pyproject.toml:39-40`), named `dew`, and it obeys the same rule as the first:
it imports no array library, because resolving a chain and hashing a config
needs neither (`src/dew/cli/__init__.py`). Each stage's own process imports jax.

### 6.3 How it relates to what exists

| Existing thing | What happens to it |
| --- | --- |
| `recipes/lm/train.py`, `diffusion/train.py`, `jepa/train.py` | unchanged as programs. They grow `--objective` per post-training.md, which is their own change, not the Recipe layer's |
| `RunConfig` and its subclasses (`src/dew/config/__init__.py:151-167`) | unchanged. A stage's `overrides` are flags for the same tyro parser, so there is no second schema |
| `dew-tpu run` (`docs/tpu.md:147-154`) | unchanged. It runs a command on a slice; a Recipe produces commands. `dew-tpu run` of `dew recipe run` is the multi-host path, and neither layer learns about the other |
| wandb config records | unchanged. The manifest is the on-disk half of the same record, and it exists because a checkpoint directory has to be self-describing without network access |

### 6.4 What it does not do

No scheduler, no retries, no DAG (stages are a list), no cluster submission, no
artifact store, no caching beyond the manifest comparison, and no second
configuration language. The deletion test is in 1.4: without the manifest chain
this layer is three shell commands, and if nobody reads manifests it should be
deleted.

## 7. Post-training as the first consumer

`docs/design/post-training.md` is the spec for the objectives. This section is
only the chaining, and the two documents agree on the three shared items:
`rollout`, the reference as an EMA at unit decay, and stages that hand off
through `--pretrained` or `--trainer.load-from-checkpoint`.

The chain, in the order post-training.md decides, expressed as one `Recipe`:

| Stage | Recipe and objective | Init from | Data | What it proves |
| --- | --- | --- | --- | --- |
| `pretrain` | `lm`, objective `lm` | nothing, or `--pretrained Qwen/Qwen3-0.6B` | FineWeb-Edu `sample-10BT`, packed | the base model trains and the packed pipeline holds |
| `sft` | `lm`, objective `sft` | `pretrain` | an instruction corpus named by post-training.md | loss masking on prompts, and the chat template round trip |
| `dpo` | `lm`, objective `dpo` | `sft` | a binarised preference set | the reference mechanism at unit decay |
| `grpo` | `lm`, objective `grpo` | `sft` or `dpo` | GSM8K prompts, exact-match reward | the rollout seam end to end |
| `flow_grpo` | `diffusion`, objective `flow_grpo` | a trained flowers DiT | flowers prompts, one reward model | that the same seams serve a diffusion policy |

Each stage is one row in a `Recipe` file and one manifest on disk. The RL stages
are the first consumers of `dew.rl`, and the fact that a diffusion stage is the
fifth row using the same primitives is the check that section 5's design is a
framework rather than an LM-shaped special case.

Where the two documents divide, so that neither repeats the other. This plan
owns the package layout, the trainer's two changes, the telemetry keys and the
Recipe layer. `docs/design/post-training.md` owns the six objective
constructors, the chat and preference data paths, the reward signatures, the
memory arithmetic and the build order. Two names for the same code are
reconciled here: post-training.md's step 1, "`rl.py` plus fixtures" with
`group_advantage`, the clipped surrogate and the k3 KL estimator
(`docs/design/post-training.md:390`), is the first commit of `dew/rl/` in
section 1.1's layout, landing as `advantage.py` and `surrogate.py`. A single
`rl.py` is the smaller interface for those three functions and stops being
smaller at the sixth, which is why the package is the shape and its first
commit is that file's contents.

The eight branches of post-training.md's build order
(`docs/design/post-training.md:384-398`) decompose waves 4.12, 4.13 and 4.14 of
this plan; that table is the finer-grained sequence and this one is the frame.
The reward contract in section 5.2 is that document's, taken as written. What
the per-sample shape gives up is one batched device call: a CLIP-style reward
model pays a forward per sample instead of one per group. The trigger for an
adapter, not a new contract, is a measured rollout in which reward forwards
dominate; `dew/rl/reward.py` then gains one `batched(fn)` wrapper.

## 8. Tutorials and documentation

One per capability, on real data, added by the wave that adds the capability. A
wave is not done until its tutorial runs top to bottom.

| Wave | Document or notebook | Data |
| --- | --- | --- |
| 4.3 telemetry | `docs/concepts/telemetry.md`, with the metric table and the probe example | the 2000-step FineWeb-Edu run |
| 4.4 recipes | `docs/recipes.md` grows the chaining section; a notebook that runs the three-stage chain | Shakespeare, then the small instruction set |
| 4.5 axes | `docs/concepts/distributed.md` grows the axis table and the arithmetic from `docs/research/google-jax-stack.md:563-583` | the simulated mesh |
| 4.6 multi-host | fills the empty `tutorials/multi-host data-parallel training.ipynb` | FineWeb-Edu on v5e-16 |
| 4.7 MoE | `docs/concepts/moe.md` and a notebook training the 8-expert model | FineWeb-Edu |
| 4.8 fault tolerance | `docs/tpu.md` grows the kill-and-resume walkthrough with its two numbers | the v5e-16 run |
| 4.9 optimizer | `docs/performance.md` grows the Muon against AdamW table with both commands | the 0.4B equal-token runs |
| 4.10 fp8 | `docs/performance.md` grows the fp8 row and its command | the 1B run |
| 4.11 checkpoint format | `docs/concepts/checkpoints.md`, new: the checkpointables, the resharding tool and the migration | the 26B-A4B-shaped tree |
| 4.12 RL core | `docs/concepts/rl.md` (the primitives and the instance table) and a notebook running GRPO | GSM8K |
| 4.13 post-training | `docs/design/post-training.md` becomes `docs/concepts/post-training.md` when it ships, plus the chain notebook | the three-stage chain |
| 4.14 diffusion RL | a notebook fine-tuning the flowers DiT with both methods | `oxford_flowers102` |
| 4.15 world models | `docs/concepts/world-models.md` and a notebook: PPO on Playground, then Dreamer on the same task | the Playground task |

The notebook series being written now (the `tutorials/` numbered notebooks) is
the base this list extends; new notebooks continue that numbering rather than
starting a second series.

## 9. Sequencing

### 9.1 Dependencies

| Wave | Depends on | Branch | Acceptance run | Reviewer | Agent level |
| --- | --- | --- | --- | --- | --- |
| 4.1 landing | none | the seven open branches | their own tests plus one smoke run each | the branch's reviewer | done |
| 4.3 telemetry | adopt-small | `wave/telemetry` | 2000 steps, 85M decoder, FineWeb-Edu, 4080 | owner | task, expert review |
| 4.4 recipes | 4.3 (run id only) | `wave/recipes` | three-stage chain, Shakespeare, 4080 | owner | task |
| 4.5 axes | sharding-rules | `wave/parallel-axes` | four mesh configs, loss equal to 1e-6 | expert | expert |
| 4.6 multi-host | 4.5 | `wave/multi-host` | Qwen3-0.6B on FineWeb-Edu, v5e-16, 200-step parity | expert | task, expert review |
| 4.7 MoE | 4.5, 4.6 | `wave/moe` | 8-expert decoder, 2000 steps, v5e-16, EP=4 | expert | expert |
| 4.8 fault tolerance | 4.6 | `wave/fault-tolerance` | kill worker 1 at step 900, resume | expert | task, expert review |
| 4.9 optimizer | adopt-small | `wave/optim-scale` | Muon against AdamW at equal tokens, 0.4B | owner | task |
| 4.10 fp8 | 4.6 | `wave/fp8` | 1B, 2000 steps, fp8 against bf16 | expert | expert |
| 4.11 checkpoint format | 4.7 | `wave/checkpoint-v1` | 26B-A4B-shaped tree, 16 chips to 8 | expert | expert |
| 4.12 RL core | 4.3, 4.4, hf-decoders | `wave/rl-core` | GRPO on GSM8K, Qwen3-0.6B, 200 steps | expert | expert |
| 4.13 post-training | 4.12 | `wave/post-training` | three-stage chain with per-stage evals | owner | task, expert for the loss ports |
| 4.14 diffusion RL | 4.12 | `wave/diffusion-rl` | flowers DiT, FlowGRPO and NFT, 500 steps each | expert | expert |
| 4.15 world models | 4.12 | `wave/world-models` | PPO on Playground, then Dreamer v3 | expert | expert |

### 9.2 What can run in parallel

Three groups, no shared files inside a group.

| Group | Waves | Shared files to watch |
| --- | --- | --- |
| A: scale | 4.5, then 4.6, then 4.7 and 4.8 and 4.10 and 4.11 in that dependency order | `training/distributed.py`, `training/trainer.py`, `config/__init__.py` |
| B: framework | 4.3, then 4.4 | `training/trainer.py`, `telemetry/`, `objectives/base.py` |
| C: learning | 4.9 alone; 4.12, then 4.13 and 4.14 and 4.15 | `objectives/`, `rl/`, `recipes/` |

Group A and group B both touch `training/trainer.py`. That is the one
serialisation point in the plan: 4.3 lands before 4.6, or 4.6 rebases onto it.
Group C's RL waves touch the trainer in exactly the two places listed in 5.4,
so they can proceed alongside group A once 4.3 has landed.

### 9.3 Agent level

Expert-level work, where a wrong answer is expensive and the design is not
already written down: 4.5 (sharding arithmetic), 4.7 (MoE and kernels), 4.10
(quantized numerics), 4.11 (checkpoint format), 4.12 (RL core), 4.14 (two paper
ports), 4.15 (world models).

Task-level work, where a reference exists and the acceptance test is
mechanical: 4.3, 4.4, 4.6, 4.8, 4.9, 4.13, and every documentation and notebook
item in section 8. Task-level waves still get a review; the level is about who
writes, not about whether anyone checks.

## 10. Risks and unknowns

Each row states the risk plainly and names the experiment that resolves it. No
row is deferred without one.

| # | Risk or unknown | Experiment that resolves it |
| --- | --- | --- |
| 10.1 | The synchronous rollout may dominate step time, making GRPO throughput unacceptable, and the rollout batch is bounded from the other side by the KV cache: 64 rows of 2048 tokens at 0.6B is about 14 GiB in bf16 (`docs/design/post-training.md:340`) | 4.12's run reports `train/rollout_seconds` over `train/step_time_ms` at 0.6B, group 8, 256 new tokens, with the cache size that made it fit. Above about 3, a decoupled sampler process is justified and is a new seam |
| 10.2 | The grouped matmul may have no fast path on the hardware Dew has. tokamax's Triton and Mosaic paths do not lower on this Ada card (`docs/research/google-jax-stack.md:304-316`) | Before 4.7 writes the MoE block: benchmark `jax.lax.ragged_dot`, `tokamax.ragged_dot` and a dense masked reference at the 8-expert and 128-expert shapes, on the 4080 and on v5e-8. The winner is the default and the numbers go in `docs/performance.md` |
| 10.3 | Tensor parallelism may not pay at Dew's scale, and `docs/research/frontier-training.md:208` argues it never will | 4.5's four-configuration table on v5e-16 at fixed global batch. If the tensor axis never wins, it stays a rules entry with no recommended use and the documentation says so |
| 10.4 | Gemma 4 26B-A4B parity may be unreachable on available memory | Measure host RAM for a two-layer fp32 load of `google/gemma-4-26B-A4B-it` before 4.7. If it does not fit, parity is block-level plus a small-config end-to-end test, and the plan says that instead of promising logits |
| 10.5 | fp8 training may cost accuracy at Dew's scale. NVIDIA's own caution is that fp8 recipes validated below 8B and 1T tokens did not generalise (`docs/research/frontier-training.md:199`) | 4.10's paired 2000-step runs at 1B. Accept only on a stated loss delta; a failure means fp8 stays an inference and export path |
| 10.6 | The data pipeline may not survive the move to two hosts. `ShardByJaxProcess` and `jax.make_array_from_process_local_data` are multi-controller idioms (`docs/research/google-jax-stack.md:634`) | 4.6 asserts that the union of records seen by the two workers over 200 steps is the expected disjoint cover, from the iterator state, not from a log line |
| 10.7 | Dreamer v4's shortcut forcing objective is not specified in this plan | A reading pass over arXiv 2509.24527 producing a specification with equations, then a small reproduction: train the world model on one small video dataset and compare open-loop prediction error against a diffusion-forcing baseline on the same data. Only then does it become a wave |
| 10.8 | Muon may not transfer to Dew's parameter layouts. The labs' recipes assume a particular matrix-axis convention (`docs/research/frontier-training.md:183`) | 4.9's equal-token comparison, plus the dimension-numbers coverage test. If Muon loses at 0.4B, it stays an option and AdamW stays the default |
| 10.9 | The probe step may change the largest trainable configuration through captured activations | 4.3 measures peak HBM with and without capture at the largest configuration that fits on the 4080. If capture changes it, the probe step runs on a smaller batch, which is honest because it measures statistics, not throughput |
| 10.10 | Adding checkpointables changes a frozen layout | 4.11 ships the converter and a test that loads a checkpoint written by today's code, as `CONTRIBUTING.md:11` requires. Without that test the wave does not land |
| 10.11 | The one-way import rule for `dew.rl` may quietly break | The import-direction test in 5.1, in the suite from the first RL commit |
| 10.12 | Two documents may drift on the shared seams | `rollout`, the batch key names and the stage handoff are written identically here and in `docs/design/post-training.md`, and each names the other. A grep for `rollout(` across both documents is the check |

## Appendix A: source clones

| Source | Location | Commit | Date | Licence |
| --- | --- | --- | --- | --- |
| MaxText | `/tmp/plan/maxtext` | `c114e25` | 2026-09-02 | Apache 2.0 |
| tokamax | `/tmp/plan/tokamax` | `5e5e422` | 2026-09-02 | Apache 2.0 |
| Kauldron | `/tmp/plan/kauldron` | `19690ea` | 2026-09-02 | Apache 2.0 |
| gemma | `/tmp/plan/gemma` | `7b78599` | 2026-08-04 | Apache 2.0 |
| Grain | `/tmp/plan/grain` | `6bc3b4c` | 2026-08-31 | Apache 2.0 |
| Optax | `/tmp/plan/optax` | `d47bc8d` | 2026-08-30 | Apache 2.0 |
| Orbax | `/tmp/plan/orbax` | `1aac202` | 2026-09-02 | Apache 2.0 |
| Tunix | `/tmp/design/tunix` | read 2026-09-02 | 2026-09-01 | Apache 2.0 |
| TRL | `/tmp/design/trl` | read 2026-09-02 | n/a | Apache 2.0 |
| Brax | read through its README on GitHub | 2026-09-02 | n/a | Apache 2.0 |

Papers cited: Dreamer v3 arXiv 2301.04104; Dreamer v4 arXiv 2509.24527;
Flow-GRPO arXiv 2505.05470; DiffusionNFT arXiv 2509.16117; GRPO arXiv
2402.03300 (cited at `tunix rl/grpo/grpo_learner.py:63`); GSPO arXiv 2507.18071
(`:88`); aux-loss-free balancing arXiv 2408.15664 (cited at
`maxtext layers/moe.py:242`).
