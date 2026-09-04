# The JAX training stack Google publishes, and what Dew should take from it

Research note, 2026-09-02.

## How this was done, and how to read it

Every library below was cloned fresh and read as source. Claims are cited as `path:line` against the clone, or as a URL. Where a fact could not be found I say so rather than guess. Two things were checked by running them rather than by reading: tokamax on this workstation's RTX 4080 (appendix A), and the existence of the model releases named here, using `huggingface_hub.HfApi().list_models`.

Clones live in `/tmp/research/GoogleStack/<name>/`. The tokamax test script is `/tmp/research/GoogleStack/tokamax_gpu_test.py`.

Verdict tags, one or more per section:

| Tag | Meaning |
| --- | --- |
| [adopt: depend on it] | add the dependency, call its API |
| [borrow: reimplement the idea] | the idea is right, the dependency is wrong for Dew |
| [skip] | not for Dew, with a reason |
| [later] | right idea, wrong time, with the trigger that would make it right |

The Dew seams referred to throughout:

| Seam | Where |
| --- | --- |
| data source | `dew.data`, today `grain.python` via `dew/data/dataloaders.py` |
| kernel path | `dew.nn.attention.scaled_dot_product_attention(implementation=...)` |
| checkpointer | `dew.training.trainer`, an `ocp.CheckpointManager` |
| optimizer | `dew.training.optim.build_optimizer` and `OPTIMIZER_MAP` |
| sharding | `dew.training.distributed`, a `(data, fsdp)` mesh |
| objective | `dew.objectives` |
| post-training interop | `dew.interop`, safetensors |
| telemetry | `dew.telemetry` |

---

## 1. MaxText

**What it is.** Google's reference LLM training codebase in JAX, <https://github.com/AI-Hypercomputer/maxtext>. Apache 2.0. Very active: the clone's last commit is 2026-09-01. It is not a library you import; it is a configured program, and `src/maxtext/configs/base.yml` is 1432 lines of it. Read it as the answer key for "how does a lab actually configure a training run".

### Sharding: named axes, logical rules, two parallelism vectors

The mesh has twelve physical axes (`configs/base.yml:549`):

```
mesh_axes: ['diloco', 'data', 'stage', 'fsdp', 'fsdp_transpose', 'context',
            'context_usp_ulysses', 'context_autoregressive', 'tensor',
            'tensor_sequence', 'expert', 'autoregressive']
```

Every axis has an `ici_*` and a `dcn_*` size (`base.yml:679-704`). The product of the ICI sizes must equal devices per slice, the product of the DCN sizes must equal the number of slices, and one axis per vector may be `-1` to be solved for (`base.yml:675-678`). The defaults are `ici_fsdp_parallelism: -1` and `dcn_data_parallelism: -1`, which is exactly the recipe the scaling book argues for in section 10: shard inside the fast domain, replicate across the slow one. Four DCN axes are commented "never recommended": `dcn_sequence`, `dcn_tensor`, `dcn_tensor_sequence`, `dcn_autoregressive`.

Three axes beyond the usual list are worth naming:

| Axis | What it is |
| --- | --- |
| `diloco` | DiLoCo, infrequent outer-loop synchronisation between slices (`base.yml:679,692`) |
| `context_usp_ulysses` | the head-exchange dimension of unified sequence parallelism, used when `context_parallel_strategy='usp'` (`base.yml:698`) |
| `stage` | pipeline stages; `num_stages = ici_pipeline_parallelism * dcn_pipeline_parallelism` (`layers/pipeline.py:54`) |

The mesh is built by `create_device_mesh` (`utils/maxtext_utils.py:2169-2248`): fill the `-1`s with `fill_unspecified_mesh_axes`, then `mesh_utils.create_hybrid_device_mesh(ici_parallelism, dcn_parallelism, devices)` for multiple slices, or `mesh_utils.create_device_mesh(ici_parallelism, devices)` for one. Around that sit sub-slice selection, split physical axes, ring-reshaped custom meshes, a v6e-specific optimisation, and an elastic device list from `elastic_utils.live_devices`.

**The part worth copying is the indirection.** Arrays carry logical names, and the config maps each logical name to an ordered list of physical axes; the first axis that exists in the mesh wins. The table is 109 lines, `base.yml:550-658`. Five of them show the shape:

```
['activation_batch_attn', ['data', 'fsdp', 'fsdp_transpose', 'expert']],
['activation_heads',      ['tensor', 'tensor_sequence', 'autoregressive']],
['embed_vocab',           ['fsdp', 'fsdp_transpose', 'context', 'context_usp_ulysses', 'expert']],
['exp', 'expert'],
['mlp_moe', ['fsdp_transpose', 'tensor', 'tensor_sequence', 'autoregressive']],
```

The model says "this axis is `heads`". The config decides whether `heads` is sharded over `tensor` today. One model definition then serves data, FSDP, tensor, context, expert and pipeline parallelism with no edits. `override_logical_axis_rules` chooses merge or replace (`base.yml:25`), and evaluation gets its own `logical_axis_rules_for_eval` (`:670`).

One guardrail is worth stealing on its own: `sharding_tolerance: 0.02`, described as "the allowed percentage of non-sharded parameters" (`base.yml:672-673`). The run fails if too much of the model ended up replicated.

One more flag deserves a mention because Dew has made the same choice implicitly: `shard_mode: "auto"`, "can be either auto or explicit" (`base.yml:547`). Auto means GSPMD infers the collectives from the annotations, explicit means JAX's explicit axis types make them visible in the type. Dew builds its mesh with `axis_types=(AxisType.Auto, AxisType.Auto)` (`src/dew/training/distributed.py:38`) and says why in the docstring, that GSPMD should infer the collectives rather than Dew writing them by hand (`:27-28`). That is the right default, and it is worth knowing that the reference codebase treats it as a switch rather than a law, with `custom_mesh_and_rule` to swap the mesh and rules wholesale (`base.yml:548`).

### Rematerialisation

`remat_policy` picks one of `minimal_with_context`, `minimal`, `save_dot_with_context_except_mlp`, `save_dot_except_mlpwi`, `save_dot_except_mlp`, `save_qkv_proj`, `qkv_proj_offloaded`, `custom`, `minimal_offloaded`, `save_out_proj`, `full`, ordered fastest to slowest and highest to lowest HBM, default `full` (`base.yml:373-376`). With `custom`, about twenty named tensors each take `remat`, `device` or `offload`, including `mlpwi`, `mlpwo`, `qkv_proj`, `out_proj`, `moe_mlpwi_0`, `mla_q`, `mla_kv`, `context` and `indexer_cutoff_threshold` (`base.yml:377-400`). `decoder_layer_input` is pinned to `device` because it is the periodic checkpoint remat restarts from (`:379`). Host offload is a first-class third option, with separate `optimizer_memory_host_offload` and `parameter_memory_host_offload` switches (`:402-403`). The vision tower gets its own `remat_policy_for_vit` (`:1257`).

### Attention kernels, and attention variants

Two orthogonal flags (`base.yml:408-411`):

- `attention`: `autoselected`, `dot_product`, `flash`, `cudnn_flash_te`. Splash or flash on TPU, cuDNN flash through NVIDIA's Transformer Engine on GPU, autoselect by default.
- `attention_type`: `global`, `local_sliding`, `chunk`, `mla`, `full`, `compressed`, `block_diffusion`.

Around those sit the switches that define current architectures: `attention_sink` (`:414`), `sliding_window_size` (`:415`), `chunk_attn_window_size` (`:416`), `causal_block_size: 32` for block-causal diffusion attention citing arXiv:2503.09573, where tokens attend bidirectionally inside a block and causally across blocks (`:417-419`), `attn_logits_soft_cap` and `final_logits_soft_cap` (`:420-421`), `z_loss_multiplier` (`:422`), `use_post_attn_norm` and `use_post_ffw_norm`, the Gemma sandwich norms (`:423-424`), `qk_norm_with_scale` and `v_norm_with_scale` (`:425-426`), `fused_qkv` and `fused_mlp` (`:484-485`), MoBA with `moba_chunk_size: 1024` and `moba_topk: 8` (`:443-446`), and DeepSeek Sparse Attention (`:448`).

`use_qk_clip` with `qk_clip_threshold: 100.0` is labelled "QK-Clip (Muon Clip) Configuration" and "tau in the paper", supported in MLA with dot product or tokamax splash (`base.yml:479-481`). Note where it lives: QK-Clip is an attention-side rescale, not an optimizer feature, even though it arrived with Muon.

Gemma 4's small variants get named support: `hidden_size_per_layer_input` and `vocab_size_per_layer_input` add a per-layer-embedding sub-block (`base.yml:429-433`), `num_kv_shared_layers` makes trailing layers reuse K/V from the last non-shared layer of the same attention type, sliding to sliding and full to full (`:434-437`), and `use_double_wide_mlp` spends the saved parameters on a wider MLP (`:438-440`).

### MoE

`megablox: true` and `sparse_matmul: true` are the defaults, and `capacity_factor: -1.0` means "no dropping by default" (`base.yml:226-228`). MaxText is dropless out of the box and reaches for a grouped matmul rather than a padded capacity matmul. Three kernels can serve that matmul: megablox, JAX's `ragged_dot`, and tokamax's GMM via `use_tokamax_gmm`, `use_gmm_v2` and `use_gmm_v2_heuristic_tiling` (`base.yml:285-291`). The tiling comment is a useful hint about maturity: megablox and JAX ragged dot support only the six forward tile configs, while tokamax supports all eighteen (`:261-262`).

Token routing has its own kernels: SparseCore paths for ragged gather and ragged gather reduce, each with a JAX reference fallback and overridable FLOP and bytes cost estimates (`:252-259`), plus `moe_use_direct_token_gather` to avoid materialising top-k copies (`:250`) and `use_gather_mosaic_kernel` (`:251`). Sharding choices: `shard_exp_on_fsdp`, recommended only when `num_experts` is a multiple of the FSDP size, and `moe_fsdp_use_two_stage_all_gather` (`:296-299`). Numerics: `float32_weight_sum: true` sums expert weights in fp32 for stability, with `float32_gate_logits` available (`:208-209`). `norm_topk_prob` is called out as Qwen3-specific router weight normalisation (`:294`).

### Checkpointing

This is where Dew and MaxText overlap most, so the gaps are sharp.

| Feature | MaxText | Dew today |
| --- | --- | --- |
| Async save | `async_checkpointing: true` (`base.yml:56`) | yes, `enable_async_checkpointing=True` (`trainer.py:170`) |
| Orbax API version | v1 unconditionally; `enable_orbax_v1` is "DEPRECATED: Orbax v1 is now always used" (`base.yml:90-93`) | v0 style `ocp.CheckpointManager` with `ocp.args.PyTreeSave` (`trainer.py:171,347`) |
| Storage tuning | OCDBT and zarr3 on, 2GB target file size, 96GB concurrent I/O; the comment says chunking large arrays into sub-2GB files "can speed up distributed and over the network loading enormously" (`base.yml:78-88`) | defaults |
| Restore fan-out | `enable_single_replica_ckpt_restoring`, "one replica reads the ckpt then broadcasts to the rest" (`base.yml:60-61`) | every process reads |
| Multi-tier | `enable_multi_tier_checkpointing` with `local_checkpoint_directory`, `local_checkpoint_period` and a GCS backup interval; on restore, a local copy in any slice is broadcast to the others rather than fetched from GCS (`base.yml:493-500`) | none |
| Preemption | `enable_autocheckpoint` saves at the preemption step (`base.yml:102-103`) | none |
| Deletion | move to `checkpoint_todelete_subdir` before deleting (`base.yml:62-65`) | direct delete |
| Retention | `max_num_checkpoints_to_keep` (`base.yml:58`) | richer: `AnyPreservationPolicy([LatestN, BestN(get_metric_fn=_epoch_loss)])` (`trainer.py:163-169`) |
| Foreign formats | `source_checkpoint_layout: "orbax"` or `"safetensors"`, plus a `checkpoint_conversion_fn` hook (`base.yml:94-97`) | `dew.interop` safetensors conversion |

`src/maxtext/checkpoint_conversion/` holds `to_maxtext.py`, `to_huggingface.py`, `reshard_checkpoint.py`, `load_and_quantize_checkpoint.py`, `inspect_checkpoint.py` and `compare_hf_ckpt.py`. Conversion goes both ways, resharding is a first-class operation, and there is a tool whose only job is to diff against the Hugging Face checkpoint.

### Goodput

`enable_goodput_recording`, `monitor_goodput`, `goodput_upload_interval_seconds: 30`, `enable_pathways_goodput`, `enable_gcp_goodput_metrics: true` (`base.yml:1078-1084`). Goodput is the fraction of wall-clock time spent on useful training rather than on restarts, stalls and checkpoint waits. Dew records step time and loss, so it can say how fast a step was but not what fraction of the run was productive.

### Data

`dataset_type` is `synthetic`, `hf`, `grain` or `tfds` (`base.yml:795-797`), documented in `docs/guides/data_input_pipeline.md`. All four live in one codebase behind a flag, which is a useful reminder that the pipeline choice is configuration, not architecture. `tokenizer_type` is `sentencepiece`, `huggingface` or `tiktoken`; the grain and tfds pipelines take all three, the hf pipeline forces huggingface (`:734-736`).

### Flax NNX or Linen

Both, with NNX ahead. 89 files under `src/maxtext` import `flax.nnx` against 37 that import `flax.linen`. The NNX files spread across `models` (22), `layers` (20), `trainers` (14) and `utils` (13); the Linen files are in the same places. The migration is real and unfinished.

### Scan over layers

`scan_layers: true` with `param_scan_axis: 1` (`base.yml:404-406`). The comment gives the reason and a subtlety worth copying: set it false when using pipeline parallelism and scan the pipeline iterations instead, and "when resuming from a checkpoint, this flag is auto-determined from metadata". The checkpoint records whether the layer stack was stacked, so the flag cannot silently disagree with the weights on disk.

### Precision

Activations in bf16, master weights in fp32: `dtype: "bfloat16"` and `weight_dtype: "float32"` (`base.yml:125,185`). That is the same policy Dew already states in `src/dew/config/__init__.py:44-45`. Around it: `matmul_precision: "default"` (`:138`), `cast_logits_to_fp32: true` (`:204`), `float32_qk_product` and `float32_logits` for attention (`:205-206`), and `mu_dtype` to store Adam's first moment in something other than the weight dtype (`:1013`).

Quantization is one string with a per-layer config file beside it: `quantization` takes `int8` for dynamic range, `fp8` for NVIDIA 8-bit gemms, `nanoo_fp8` for AMD MI300 and MI325, and `fp8_full` for fp8 with static scaling (`base.yml:128-133`), plus `quant_cfg_path` (`:143`), `kv_quant_dtype: "int8"` (`:152`), `quantization_local_shard_count` (`:167`), and experimental fp8 quantization of Q and K inside splash attention with no scaling factors (`:1213-1214`).

### Models it ships

27 files in `src/maxtext/models/`: `gemma`, `gemma2`, `gemma3`, `gemma4`, `gemma4_small`, `gemma4_vision`, `deepseek`, `deepseek4`, `deepseek_batchsplit`, `deepseek_batchsplit_fp8`, `llama2`, `llama4`, `mistral`, `mixtral`, `qwen2`, `qwen3`, `qwen3_5`, `qwen3_5_vision`, `qwen3_custom`, `qwen3_vl_vision`, `gpt3`, `gpt_oss`, `olmo3`, `envy`, plus `models.py`, `simple_layer.py` and `__init__.py`.

Gemma 4, DeepSeek 4 and Qwen 3.5 are all present, so this codebase tracks releases closely. Gemma 4 is a real release, not a placeholder: `HfApi().list_models(search="gemma-4", author="google")` returns `google/gemma-4-31B-it`, `google/gemma-4-26B-A4B-it`, `google/gemma-4-E4B-it`, `google/gemma-4-E2B-it`, `google/gemma-4-12B-it` and a QAT w4a16 variant. There is a post-training script for it too, `trainers/post_train/rl/scripts/run_gemma4_e4b_rl.sh`.

### What Dew should do about it

Dew's mesh is `(data, fsdp)` with `AxisType.Auto`, and sharding is inferred from shape: shard the largest evenly divisible axis, replicate anything under 65536 elements (`src/dew/training/distributed.py:22-58`). That is a good default and a hard ceiling. Inferring rather than declaring costs two things. First, the heuristic cannot express any axis other than FSDP, so tensor, context and expert parallelism are unreachable, which by section 10's arithmetic pins Dew to a per-device batch of roughly 850 tokens on v5p-class hardware. Second, when no axis divides evenly the parameter is silently replicated and nothing says so.

The order of work: name logical axes in the model modules, add a rules table from logical names to physical axes, keep `(data, fsdp)` as the default rules so nothing changes today, then add axes one at a time.

[borrow: reimplement the idea] Logical axis rules, and `sharding_tolerance` as a startup assertion. Seam: `dew.training.distributed`, plus logical annotations in `dew.nn`.
[borrow: reimplement the idea] A named `remat_policy` instead of all-or-nothing `jax.checkpoint`. Seam: `dew.training.objective_trainer`.
[borrow: reimplement the idea] Vocabulary tiling for the LM head: `num_vocab_tiling` chunks the cross-entropy along batch-sequence and is "highly recommended for models with large vocabularies (e.g. Gemma)", with `vocab_tiling_ag_once` to gather the head once for the backward (`base.yml:720-729`). Seam: the LM objective.
[borrow: reimplement the idea] Goodput accounting. Seam: `dew.telemetry`.
[borrow: reimplement the idea] Recording `scan_layers` in checkpoint metadata so a resume cannot disagree with the weights. Seam: `dew.training.trainer`.
[skip] MaxText as a dependency. It is a program, not a library, and its configuration surface is larger than all of Dew.

---

## 2. Kauldron

**What it is.** DeepMind's research trainer, <https://github.com/google-research/kauldron>. Apache 2.0. On PyPI as `kauldron` 1.4.4, and it requires Python 3.12 or newer. The clone's last commit is 2026-09-02, and the message ("Skip test_overview_dashboard in Copybara export") shows this is an export of an internal repository rather than a repository developed on GitHub.

It is a real dependency, not only a design reference, and the proof is not the PyPI page: the `gemma` library depends on `kauldron>=1.4.4` (`gemma/pyproject.toml:45`) and uses it in its public model protocol (`gemma/gm/nn/_transformer_like.py:28,48`).

**The trainer is one flat dataclass.** `kauldron/train/trainer_lib.py:109` declares `class Trainer(config_util.BaseConfig)` and every part of a training run is a field on it (`:169-249`):

| Field group | Fields |
| --- | --- |
| identity | `seed`, `workdir` |
| data | `train_ds: data.Pipeline`, `eval_ds` |
| model | `model: nn.Module`, `rng_streams: RngStreams` |
| sharding | `sharding: sharding_utils.ShardingStrategy` |
| duration | `num_train_steps`, `stop_after_steps` |
| objective | `train_losses`, `train_metrics`, `train_summaries` |
| optimisation | `optimizer: optax.GradientTransformation`, `schedules` |
| output | `writer: metric_writer.WriterBase`, `profiler`, `log_metrics_every: 100`, `log_summaries_every: 1000` |
| persistence | `checkpointer: BaseCheckpointer`, `init_transform: InitTransform`, `exporter: ModelExporter` |
| step and eval | `trainstep: TrainStep`, `evals: Mapping[str, EvaluatorBase]` |
| infra | `setup: Setup`, `xm_job`, `raw_cfg` |

`from flax import linen as nn` (`trainer_lib.py:29`), so Kauldron models are Linen, like Dew's.

Two fields deserve attention because Dew has no equivalent. `init_transform: checkpoints.InitTransform` is a declared hook for "where do the initial weights come from", which is how a fine-tune differs from a pretrain without a different trainer. And `evals` is a mapping of named evaluators, so a run can carry several evaluation suites with different data and different cadence, rather than one validation loop.

**konfig** is the configuration system. You write a Python function that builds the object graph using the real classes, and every field is addressable from the command line. `kauldron/konfig/README.md:1-30` shows the pattern: `get_config(args: ConfigArgs)` returns a `kd.train.Trainer()` with fields assigned, and command-line overrides use paths like `--cfg.__args__.arg1=value`. The point is that the config is the constructor call, so there is no second schema to keep in sync with the code. Dew uses tyro over dataclasses, which achieves the same goal for flat configs and less for nested object graphs.

### What Dew should do about it

Not adopt. Kauldron requires Python 3.12 while Dew supports 3.11, it is a Copybara export with the API stability that implies, and its `Trainer` is a peer of Dew's rather than something to sit underneath it. Taking it would mean rewriting Dew's trainer as a Kauldron config, which is a different project.

Borrow two specific things. The first is the flat, fully substitutable trainer: Dew's `Trainer` already takes the objective as a parameter, and the fields Kauldron exposes that Dew hard-codes are the checkpointer, the writer, the profiler and the eval suite. The second is `init_transform`: a named seam for initial-weight provenance is exactly what Dew needs now that it can load Hugging Face checkpoints, and it belongs next to the checkpointer rather than inside a recipe.

[skip] as a dependency, for the Python 3.12 floor and the export model.
[borrow: reimplement the idea] the flat swappable-component trainer, and `init_transform` as a named weight-provenance seam. Seams: `dew.training.trainer`, `dew.training.objective_trainer`.

---

## 3. The gemma library

**What it is.** The official JAX library for running and fine-tuning Gemma, <https://github.com/google-deepmind/gemma>. Apache 2.0. PyPI `gemma` 4.0.1, and the clone declares 4.1.0 in `gemma/__init__.py`. Last commit 2026-08-04 in my clone. It implements Gemma 2, 3, 3n and 4, plus research variants T5Gemma and Diffusion Gemma.

**Linen, with no NNX at all.** Searching the whole repository for `nnx` returns zero files; 45 files import `flax.linen`. The `gm` namespace is not an NNX rewrite, it is a new-generation API surface over the same Linen models, organised as `gm.nn`, `gm.text`, `gm.ckpts`, `gm.data`, `gm.losses`, `gm.evals`, `gm.math`, `gm.tools` and `gm.sharding`.

**The sampler contract is a Protocol, and that is the interesting part.** `gemma/gm/nn/_transformer_like.py:79-158` defines:

```python
class TransformerLike(Protocol):
  """Protocol for a transformer model to be used with a Sampler.

  A model passed to a `Sampler` must implement `apply` and `init_cache`.
  """
  config: TransformerConfig
  INFO: ClassVar[ModelInfo]
  def __call__(self, tokens: Int['*B L'], *, images=None, positions=None,
               cache=None, attention_mask: Bool['*B L_with_mm cache_length'] | None = None,
               return_last_only=None, return_hidden_states=None) -> Output: ...
  def init_cache(self, *, batch_size, dtype, cache_length, sharding=None) -> Cache: ...
```

`Output` is a `flax.struct.dataclass` of `logits`, `cache` and `hidden_states` (`:53-66`). `ModelInfo` carries `tokenizer_version` and `default_ckpt` so the sampler can find the tokenizer and weights for a model class (`:69-76`). The type annotations come from Kauldron's `ktyping`, and the cache sharding type is `kd.sharding.ShardingTree` (`:28,48`).

The KV cache is a per-layer structure with left-aligned slices and an end index, and GQA is a reshaped einsum. There are no fused or Pallas kernels in the sampler path.

**Checkpoints are Orbax, not safetensors.** `gemma/gm/ckpts/_checkpoint.py` uses `ocp.StandardCheckpointer` with `save_concurrent_gb` and `restore_concurrent_gb` (`:207,245`), reading canonical paths from a `CheckpointPath` string enum pointing at `gs://gemma-data/checkpoints/...` (`gm/ckpts/_paths.py:20-49`). There is no Hugging Face safetensors loader. The loader auto-detects four on-disk layouts (nested, flat, stacked, Kauldron). Several comments wait on a feature Orbax does not have yet: "Once orbax supports partial restore, we would not need to ..." (`_checkpoint.py:410,597,608`).

**Fine-tuning is Kauldron.** The library's own dependency list includes `kauldron>=1.4.4` (`pyproject.toml:45`), and fine-tuning runs through `kd.train.Trainer` with LoRA, QAT, DPO and NPO losses provided by `gemma.peft` and `gm.losses`. `gemma/peft` does module surgery with Linen interceptors, which is a Linen-native way to add LoRA or int4 without editing the model.

Also in the dependency list, and worth a look for Dew: `hackable-diffusion @ git+https://github.com/google/hackable_diffusion.git` (`pyproject.toml:41`). That repository exists, is Apache 2.0, has 160 stars and was pushed 2026-08-18.

### What Dew should do about it

Do not depend on it: it is a model runner for one family, and it pulls Kauldron, `seqio`, `tensorflow-cpu` and `bagz` with it.

Do implement `TransformerLike` on Dew's `CausalTransformer`. The protocol needs `apply` and `init_cache` plus a keyword-compatible `__call__`, and Dew's decoder already has the substance; what it lacks is `init_cache` returning a gemma-shaped `Cache` and an `Output` struct. The payoff is that Dew models become usable with gemma's `Sampler`, `ChatSampler` and tool-calling sampler without Dew writing a sampler loop, and it gives Dew a published protocol to test its generation path against.

The second thing to take is the checkpoint layout normaliser. gemma detects four param-tree layouts and maps them to one. Dew's interop already does this for safetensors; the idea to copy is that the layout is detected and named rather than assumed.

[skip] as a dependency.
[borrow: reimplement the idea] `TransformerLike` as the shape of Dew's generation contract, and layout auto-detection on restore. Seams: `dew.sampling`, `dew.interop`, `dew.nn` decoder.
[later] gemma's Linen interceptor trick for LoRA, if and when Dew wants adapters. Seam: `dew.nn`.

---

## 4. Tunix

**What it is.** Google's post-training library for JAX, <https://github.com/google/tunix>. Apache 2.0. Last commit 2026-09-01, so it is active. PyPI has `tunix` at version 0.0.0, a placeholder, so installation is from GitHub. The README calls the project "in early development".

**What it covers.** SFT, DPO, PPO, GRPO and Dr.GRPO with citations in the README (`README.md:42-57`), knowledge distillation, and agentic RL with multi-turn agent and environment interaction, tool use and async rollout, released 2025-12 (`:67`). Rollout is served by vLLM or SGLang-JAX on TPU (`:74-76`). Gemma 4 support landed 2026-04 (`:65`), and the models use splash attention and a GMM MoE kernel (`:66`). The package tree is `sft`, `dpo`, `rl` (with `grpo`, `ppo`, `agentic`, `rollout`, `inference`, `rl_cluster.py`, `rl_learner.py`, `reward_manager.py`, `reshard.py`), `distillation`, `generate`, `models`, `perf`, `processors`, `diffusion` and `cli`.

**The model interface is NNX, and that is the blocker.** `PeftTrainer.__init__` takes `model: nnx.Module` (`tunix/sft/peft_trainer.py:347-349`), the gradient accumulator is itself an `nnx.Module` (`:201`), and the train step signature is `Concatenate[nnx.Module, P]` (`:463`). Across `tunix/`, 63 files use `flax.nnx` and 1 uses `flax.linen`.

So a Dew `CausalTransformer`, which is Linen, cannot be passed to a Tunix trainer as it stands. There is a bridge, and Tunix uses it for its own Gemma port: `module_from_linen_variables` (`tunix/models/gemma/model.py:755`, used at `:881`) builds an NNX module from a Linen variable dict. That is the shape of the work: convert the param tree, not the module.

Tunix also ships its own model zoo with `safetensors_loader.py`, `safetensors_saver.py`, `naming.py`, `registry.py` and `automodel.py`, including `create_model_from_safe_tensors` and `create_gemma_model_with_nnx_conversion` (`tunix/models/automodel.py:148,329`). Models present: gemma, gemma3, gemma4, llama3, qwen2, qwen3.

**Against verl.** verl (`volcengine/verl`) is the widely used RLHF and RL post-training framework, and it is PyTorch. For Dew the comparison is short: verl would mean leaving JAX for the post-training stage, moving weights across frameworks, and maintaining two model definitions. Tunix keeps everything in JAX and on TPU, at the cost of an NNX boundary and a library that has not shipped a real PyPI release. Neither is a dependency Dew should take today.

### What Dew should do about it

Dew's post-training story is safetensors. Dew can already export to and import from Hugging Face format, and that is the interchange both ecosystems accept. Post-training in Tunix means exporting weights, training there, importing back, which is a real workflow and needs no coupling.

If Dew wants deeper integration later, the cheapest path is a `dew.interop` function that converts a Dew Linen param tree into the NNX structure Tunix expects, modelled on `module_from_linen_variables`. That is an adapter in Dew, not a dependency on Tunix internals.

[skip] as a dependency today: PyPI 0.0.0, self-described early development, and an NNX interface against Dew's Linen.
[later] an NNX param-tree adapter, triggered by an actual need to run GRPO or DPO on a Dew-trained model. Seam: post-training interop in `dew.interop`.
[borrow: reimplement the idea] the separation Tunix draws between the learner and the rollout engine (`rl_learner.py` against `rl/rollout` and `rl/inference`), if Dew ever grows an RL objective. Seam: `dew.objectives`.

---

## 5. tokamax

**What it is.** A library of accelerator kernels written in Pallas, at <https://github.com/openxla/tokamax>. Note the org: `openxla/tokamax`, not `google/tokamax`, and the copyright header reads "DeepMind Technologies Limited" (`tokamax/__init__.py:1`). Apache 2.0 (`LICENSE`). Version 0.0.13 on PyPI. The clone's last commit is 2026-09-02. The version number is honest: a young library with a stable-looking front door.

**The front door.** Everything public is a plain JAX function with an `implementation` argument, exported from the top level (`tokamax/__init__.py:29-40`):

| Function | What it is | Backends in the source |
| --- | --- | --- |
| `dot_product_attention` | FlashAttention, a superset of `jax.nn.dot_product_attention` | `xla`, `xla_chunked`, `triton`, `mosaic` (GPU sm90/sm100, and TPU), `cudnn` |
| `ragged_dot`, `ragged_dot_general` | the grouped matmul MoE needs | `xla`, `triton`, `mosaic` GPU, `mosaic_tpu`, `mosaic_tpu_v2` |
| `layer_norm` | LayerNorm, and RMSNorm with `subtract_mean=False` | `xla`, `triton` |
| `gated_linear_unit` | SwiGLU and friends | `xla`, `triton` |
| `linear_softmax_cross_entropy_loss` | fused projection and loss, the large-vocabulary memory saver | `xla`, `triton` |
| `ragged_gather`, `ragged_gather_reduce`, `ragged_scatter` | MoE routing primitives | Pallas variants |
| `causal_conv1d_gated_delta_rule` | the gated delta rule mixer used by linear-attention hybrids | Pallas variants |
| `triangle_multiplication` | AlphaFold's triangle update | present |
| `flex_attention` | mask and score modification attention | present |

`docs/supported_ops.md:3-16` claims GPU support for attention, gated linear unit and layer norm, and GPU plus TPU for `ragged_dot`. The source is ahead of that document: `tokamax/_src/ops/attention/pallas_mosaic_tpu.py` exists, and `docs/splash_attention.md:1-4` calls splash attention "an experimental TPU op within Tokamax".

Three design choices are worth taking regardless of the dependency question:

1. `implementation=None` means "pick the best available, and you may pick differently for forward and backward"; a named implementation means "use it or raise" (`docs/basic_usage.md:22-31`). Dew's `scaled_dot_product_attention` already has this shape (`src/dew/nn/attention.py:97-182`), which is why the seam fits without redesign.
2. Autotuning results are values. `tokamax.autotune(f, *args)` returns an `AutotuningResult` that is also a context manager, with `dumps()` and `loads()` (`docs/basic_usage.md:47-69`). The docs state plainly that autotuning is non-deterministic and that different configs change numerics, so pinning a serialized result is how numerics stay stable across sessions (`:75-79`).
3. Benchmarking helpers measure accelerator time, not Python time, with a CUPTI path on GPU (`docs/basic_usage.md:106-125`). The docs call out timing around `block_until_ready` as the thing that does not work, which is what `tools/benchmark_step.py` in Dew currently does.

**Adoption evidence.** MaxText vendors a copy of the tokamax splash attention kernel at `src/maxtext/kernels/tokamax_splash_attention/splash_attention_kernel.py` and exposes `use_tokamax_splash` (`base.yml:1350`), `use_splash_scheduler` (`:1192`), `use_tokamax_gmm` (`:286`) and a GMM v2 switch (`:287`). The two projects are converging: tokamax is where MaxText's kernels are heading.

### Does it run on an RTX 4080 (Ada, sm_89)?

I installed it in a fresh venv and ran it. Script, output and reproduction commands are in appendix A. Result:

| Path | On RTX 4080, compute capability 8.9 |
| --- | --- |
| `implementation="xla"` | works. Attention 2.40 ms at B=2, T=2048, 8 heads, head_dim 128, bf16, causal, softcap 30; max absolute error 0.0112 against an fp32 reference. Backward pass works |
| `implementation="xla_chunked"` | works, 1.89 ms, error 0.0090 |
| `implementation="triton"` | fails, and not because of tokamax. JAX 0.11.1 keeps a device whitelist for its Pallas-Triton lowering (`jax/_src/pallas/triton/gpu_info.py:27-51`) listing A100, H100, H200, B200, GB200, GB300, GB10, L4, L40, T4, Thor, RTX 4090 and the RTX PRO cards. The RTX 4080 is absent, so `get_gpu_info()` raises `Unsupported GPU device kind: NVIDIA GeForce RTX 4080` |
| `implementation="mosaic"` | fails: `NotImplementedError: Only supported for sm90 and sm100 GPUs` (`tokamax/_src/ops/attention/pallas_mosaic_gpu.py:51-52`). The kernel files are `..._kernel_sm90.py` and `..._kernel_sm100.py`; there is no sm80 kernel |
| `implementation="cudnn"` | fails for this call: `NotImplementedError: logits_soft_cap not supported`. Worth knowing on its own, since the cuDNN path cannot do Gemma-style logit softcapping |
| `layer_norm`, and RMSNorm via `subtract_mean=False` | work on the `xla` path, error 0.0078 |
| `ragged_dot` | works on the `xla` path, relative error 2.3e-3 in bf16 |

Note the gate: `has_mosaic_gpu_support` returns True for anything at compute capability 8.0 or above (`tokamax/_src/gpu_utils.py:63-73`) and `is_sm80` buckets 8.0 to 9.0, so Ada reports as supported and the kernel then refuses. Advertised support and real support differ.

On this workstation, then, tokamax buys Dew nothing today: the only working paths are the XLA ones Dew already reaches through `jax.nn.dot_product_attention`. The value is on TPU and on H100 or newer.

### What Dew should do about it

Dew's `implementation` argument already routes `xla` and `cudnn` to `jax.nn.dot_product_attention` and `tpu` to `jax.experimental.pallas.ops.tpu.flash_attention` (`src/dew/nn/attention.py:165-182`). That TPU import is the older experimental kernel. tokamax is where the maintained TPU flash and splash kernels now live, and its signature is a superset of the one Dew already calls, including `logits_soft_cap` and `local_window_size`, both of which Dew needs for Gemma-style and sliding-window layers.

[adopt: depend on it] as an optional extra: add `'tokamax'` as an `implementation` value mapping straight through, and keep `auto` unchanged on GPU. Take `ragged_dot` as the MoE kernel when Dew grows an MoE objective, and `linear_softmax_cross_entropy_loss` for the LM head. Seams: `dew.nn.attention`, a future `dew.nn.moe`, the LM objective.
[borrow: reimplement the idea] serialized autotuning results, and kernel-time rather than wall-time benchmarking. Seam: `tools/benchmark_step.py`.
[skip] on consumer Ada GPUs, until JAX's Triton whitelist widens or a Mosaic sm80 kernel lands.

---

## 6. Orbax

**What it is.** The JAX checkpointing library, <https://github.com/google/orbax>. Apache 2.0. `orbax-checkpoint` is at 0.12.4 on PyPI, which is what Dew has. The clone's last commit is 2026-09-02.

Dew already uses more of Orbax than most projects do: a `CheckpointManager` with `enable_async_checkpointing=True`, an `AnyPreservationPolicy` combining `LatestN` with `BestN(get_metric_fn=_epoch_loss)`, sharded restore through `ocp.ArrayRestoreArgs(sharding=...)`, and a deliberate choice to hand sharded arrays straight to Orbax rather than gathering them on the host (`src/dew/training/trainer.py:162-171,285-347`). So this section is about the parts Dew has not reached. There are four.

### There is a v1 API, and MaxText already treats it as the only one

`orbax/checkpoint/experimental/v1/` is a full parallel API. Its training entry point is `Checkpointer`, a context manager with `should_save(step)`, `save_checkpointables`, `save_checkpointables_async`, `load_checkpointables`, `save_pytree` and `load_pytree` (`checkpoint/orbax/checkpoint/experimental/v1/_src/training/checkpointer.py:81,277,383,524,687,954-966`), exported alongside `save_decision_policies`, `preservation_policies`, `errors`, `CheckpointMetadata` and `RootMetadata` (`experimental/v1/training.py:19-30`).

A checkpoint is a set of named *checkpointables*, not one pytree. Model state, optimizer state and a data iterator are siblings, each with its own handler. Dew packs everything into one pytree and then works around the consequences, including a variable-length leaf for the data iterator position (`trainer.py:285-288`).

Two policies are separated where Dew's options conflate them: `SaveDecisionPolicy` decides whether to write at this step (`save_decision_policies.py:50`, with `SaveEveryNSteps` at `:81`), and `PreservationPolicy` decides what to keep afterwards (`preservation_policies.py:40`). Dew uses preservation policies already but expresses "when to save" as an if-statement in the loop.

That the v1 API is the intended future is not my inference. MaxText's config says it: `enable_orbax_v1` is marked "DEPRECATED: Orbax v1 is now always used for checkpointing; this flag is ignored and will be removed in a future release" (`maxtext/src/maxtext/configs/base.yml:90-93`).

### Replica-parallel writes

When an array is replicated across several replicas, the naive save has one replica write all the bytes while the others idle. Replica-parallel splits the write along an axis so each replica writes its fraction. `calculate_replica_parallel_axis_and_local_shape` picks the axis and the replica count (`_src/serialization/replica_slices.py:211-224`), and the user-facing knobs are in the v1 context options (`experimental/v1/_src/context/options.py:342-349,405-408`):

| Option | Meaning |
| --- | --- |
| `use_replica_parallel` | "Whether to parallelize saving across replicas" |
| `min_slice_bytes_for_replica_parallel` | only use it when bytes per replica reach this size |
| `max_replicas_for_replica_parallel` | cap on how many replicas share a write |
| `enable_replica_parallel_separate_folder` | write replica data to separate folders |

Nearby in the same options object: `use_compression: bool = True`, `ocdbt_target_data_file_size`, `enable_pinned_host_transfer` (currently GPU only), and `enable_post_merge_validation`, which validates parameters after the finalize step (`options.py:338-353,401-408`).

This matters to Dew exactly when its mesh has replicas, which is whenever `fsdp_size < device_count`. That is Dew's default configuration.

### Emergency and multi-tier checkpointing

`orbax/checkpoint/experimental/emergency/checkpoint_manager.py:1293-1305` describes itself:

> Provides both checkpoint management and emergency checkpointings. This class composes a local and a persistent checkpoint managers. The local manager saves checkpoints frequently to a fast local storage (like RAMFS). When a complete checkpoint exists at least one slice, restoration is possible, and the slice broadcasts the checkpoint to others. Additionally, the persistent manager checkpoints less frequently to a remote file system (e.g., GCS), providing a fail-safe if local checkpoints become unavailable due to issues like hardware failure or preemption.

Its options take `local`, `persistent` and `replica_axis_index` (`:267-287`). The package also holds `replicator_checkpoint_manager.py`, a `multi_tier_checkpointing/` subpackage, `mesh_consistency.py`, a `p2p/` transfer path and `process_metadata_checkpoint_handler.py`. MaxText wires it up as `enable_multi_tier_checkpointing` with `local_checkpoint_directory` and `local_checkpoint_period` (`base.yml:493-500`).

The trade is explicit: checkpoint every 20 steps to local RAM, back up to durable storage every 20 minutes, and accept that losing a slice's local copy falls back to the older durable one.

### What Dew should do about it

Two of these are worth doing now and two are not.

Worth doing now: turn on replica-parallel saving, because Dew's default mesh has replicas and the win scales with the replica count for no behaviour change; and register the Grain iterator as a separate checkpointable instead of a leaf in the model pytree, which removes the variable-length-leaf workaround at `trainer.py:285-288`. Both are reachable through the v1 API, which is the reason to look at it at all.

Not worth doing now: emergency and multi-tier checkpointing assume many slices, local NVMe or RAMFS, and a preemption rate Dew's users do not have. The trigger is clear though: the day Dew runs multi-slice, this is the first thing to add, because it is the difference between losing 20 steps and losing 10,000.

[adopt: depend on it] `use_replica_parallel` and its size and replica caps, plus separate checkpointables for model state and data iterator. Seam: checkpointer in `dew.training.trainer`.
[adopt: depend on it] a `SaveDecisionPolicy`, so "when to save" and "what to keep" are both declared rather than half declared. Seam: same.
[later] emergency and multi-tier checkpointing, triggered by the first multi-slice run. Seam: same.
[borrow: reimplement the idea] MaxText's storage tuning as Dew defaults: OCDBT on, a 2GB target data file size, and moving checkpoints to a trash directory before deleting. Seams: same, plus `dew.checkpoints.utils`.

---

## 7. Grain

**What it is.** Google's data pipeline library for JAX, <https://github.com/google/grain>. Apache 2.0. PyPI 0.2.18, which is what Dew has; the clone declares 0.2.19 (`pyproject.toml:7`). Last commit 2026-08-31. It does not require JAX to run (`README.md:31`).

**Two APIs, and neither is deprecated.** I looked for a deprecation of `grain.python` and there is none. `docs/api_choice.md:7-16` states the choice:

> If you need to do one of the following: mix multiple data sources, pack variable length elements, split dataset elements and globally shuffle the splits, then you should use `Dataset`, otherwise use simpler `DataLoader`.

`DataLoader` is the source, sampler and flat transformation list that Dew uses. `Dataset` is the chaining API: `MapDataset` for random access, `IterDataset` for iteration only, and `DatasetIterator`, the stateful iterator whose state saves and restores (`docs/api_choice.md:52-63`).

Two of the three reasons to move are things Dew's LM path needs.

| Capability | Where it lives | Dew today |
| --- | --- | --- |
| Sequence packing with segment ids | `grain.experimental.FirstFitPackIterDataset`, `BestFitPackIterDataset`, `ConcatThenSplitIterDataset` | none. `dew/data/sources/text.py:24-63` reads fixed `seq_len + 1` windows from a flat memmap at stride `seq_len` |
| Dataset mixing with weights | `MapDataset.mix` and `IterDataset.mix` (`dataset.py:250,960`), `MixedMapDataset` (`transformations/mix.py:87`) | none |
| Iterator state inside the Orbax checkpoint | `grain.checkpoint.CheckpointHandler`, with `CheckpointSave` and `CheckpointRestore` as `ocp.args.CheckpointArgs` (`checkpoint/handler.py:33,197,241`) | hand-rolled: `DevicePrefetchIterator.source_state` threaded into the trainer's pytree (`dew/training/distributed.py:96-104`, `dew/training/trainer.py:285-288`) |
| Elastic iterators across a changing host count | `ElasticIterDatasetIterator` with `get_shard_states()`, saved as `shard_state_<idx>.json` (`checkpoint/elastic_checkpoint.py`) | none |
| Sharding by process | `ShardByJaxProcess` sets `shard_index=process_index`, `shard_count=process_count` (`core/sharding.py:57-66`) | used already (`dew/data/dataloaders.py:190`) |

**Packing, concretely.** `FirstFitPackIterDataset` takes `length_struct={"x": 4}` and `num_packing_bins`, adds each element to the first bin with room, and emits all bins when an element does not fit (`transformations/packing.py:341-392`). The packer writes two extra features per packed feature, `f"{k}_segment_ids"` and `f"{k}_positions"`, both int32 (`transformations/packing_packed_batch.py:116-117`). Those are exactly the arrays an attention mask needs to stop attention crossing a document boundary, and a RoPE call needs to restart positions per document. The alternative strategy, `ConcatThenSplitIterDataset`, has an explicit `BOSHandling` enum (`transformations/packing_concat_then_split.py:72`). The docstring states the tradeoff: packing avoids splitting sequences by padding instead, and more bins means less padding but risks epoch leakage, where examples from two epochs land in one bin (`packing.py:344-347`).

**Performance knobs, with defaults.** `ReadOptions(num_threads=16, prefetch_buffer_size=500)` per process, and `MultiprocessingOptions(num_workers=0, per_worker_buffer_size=1, enable_profiling=False)` (`options.py:50-51,102-104`). Threads multiply: 8 threads and 10 workers is 80 readers (`options.py:33-35`). Grain warns when `prefetch_buffer_size < num_threads`, because that caps effective parallelism, and says the warning may become an error (`options.py:72-82`). Both read fields now accept an `AutotuneParameter` in place of an int (`options.py:50-51`). Dew passes one `Loading(workers=32, threads=64, read_buffer=128, worker_buffer=20)` per dataset spec (`dew/data/dataset.py`), so the tuning surface is wired; the defaults are just not documented on Dew's side.

### What Dew should do about it

Grain is already a dependency, so this is a version question, not an adoption question. The LM path should move from `DataLoader` to `Dataset` so it can pack. Today `TokenWindowSource` chops a concatenated token stream at a fixed stride, so a training window can straddle two documents with nothing marking the seam, attention runs across the boundary, and RoPE positions do not restart. Moving that path to `MapDataset.source(...).to_iter_dataset()` followed by `FirstFitPackIterDataset` yields packed windows plus `text_segment_ids` and `text_positions`, which the LM objective turns into a block-diagonal mask.

The second change is to stop hand-rolling iterator checkpointing and register Grain's `CheckpointHandler` as a second checkpointable, which pairs with the Orbax change in section 6.

[adopt: depend on it] the `Dataset` API for the LM path, `grain.experimental.FirstFitPackIterDataset` for packing, `grain.checkpoint.CheckpointHandler` for iterator state, and `mix` when a second corpus arrives. Seams: data source in `dew.data`, checkpointer in `dew.training.trainer`.
[later] `ElasticIterDatasetIterator`, triggered by training on a host count that can change mid-run.

---

## 8. Optax

**What it is.** The JAX optimizer library, <https://github.com/google-deepmind/optax>. Apache 2.0. PyPI 0.2.8, dated 2026-03-20; the clone declares `0.2.9.dev` (`optax/__init__.py:319`) and carries 2026 commits. Dew already depends on it.

Dew's `build_optimizer` offers `adam`, `adamw` and `lamb`, a warmup-cosine schedule, weight decay folded into the optimizer kwargs, `clip_by_global_norm`, and `MultiSteps` for gradient accumulation (`src/dew/training/optim.py:14-41`). Everything in `optax.contrib` is a plain `GradientTransformation`, so anything below drops into `OPTIMIZER_MAP` without touching the surrounding wiring.

### Muon

`optax/contrib/_muon.py`. Momentum with `beta=0.95` and Nesterov on by default, then a Newton-Schulz quintic iteration orthogonalises the 2D update: `ns_steps=5`, coefficients `(3.4445, -4.7750, 2.0315)`, a Frobenius rescale before the iteration, and a transpose for efficiency. The update is then scaled by `sqrt(max(1, fan_out/fan_in))`, or with `consistent_rms=0.2` by `sqrt(max(fan_in, fan_out))`, which makes the update RMS shape-independent so Muon and AdamW can share one parameter tree.

The practical detail is that the top-level `muon()` partitions parameters by `ndim == 2`. Matrices go through `scale_by_muon`, everything else goes through `optax.adamw`. Adopting Muon needs no restructuring of Dew's parameter tree, only a `weight_dimension_numbers` spec for arrays whose matrix axes are not `(0, 1)`.

**MuonClip is not in Optax.** A repository-wide search for `muon_clip` or `MuonClip` returns nothing. The mechanism exists elsewhere and belongs elsewhere: it is QK-Clip, a per-head rescale of the query and key projections when the observed maximum attention logit exceeds a threshold. MaxText implements it at the attention seam as `use_qk_clip` with `qk_clip_threshold: 100.0` (section 1).

### Schedule-free

`optax/contrib/_schedule_free.py`. `schedule_free_adamw()` is the drop-in candidate. It replaces Adam's first moment with `scale_by_rms(b1=0)`, adds decayed weights, and wraps the result in `schedule_free()`, which tracks three sequences: `y`, where gradients are evaluated, as `b1*x + (1-b1)*z`; `x`, the averaged evaluation sequence, as a running average of `z` weighted by `lr**weight_lr_power` with `weight_lr_power=2`; and `z`, the base iterates.

Two consequences for Dew. There is no learning-rate decay, so it pairs with warmup-then-constant, not warmup-cosine. And evaluation must use different parameters from training: `optax.contrib.schedule_free_eval_params(state, params)` returns `(y - (1-b1)*z)/b1`. The trainer therefore needs a params-for-eval path, both at evaluation and at checkpoint save.

### The rest of contrib

`optax/contrib/__init__.py` exports 26 modules. What is absent is as useful as what is present: no `distributed_shampoo`, no `mnm`, no LOMO variant, and `sm3` has been promoted out of contrib into the core (`optax/_src/alias.py:2164`). Notable entries:

| Entry | What it is | For Dew |
| --- | --- | --- |
| `muon` | orthogonalised updates, above | adopt |
| `schedule_free` | above | adopt |
| `galore` | rank-128 SVD low-rank gradient projection (`contrib/_galore.py:434`) | later |
| `ademamix` | three-alpha mixed EMA (`contrib/_ademamix.py:133`) | later |
| `split_real_and_imaginary` | complex parameters (`contrib/_complex_valued.py:87`) | later |
| `sophia`, `sam`, `dog`, `dowg`, `dadapt_adamw`, `prodigy`, `momo`, `madgrad`, `cocob`, `adopt`, `acprop`, `dpsgd`, `reduce_on_plateau` | research optimizers and wrappers | skip today |

**On sharded optimizer state.** Optax provides no wrap-to-shard helper. What it provides is a tested guarantee: `contrib/_sharding_test.py:54-75` and `_src/sharding_test.py:71-121` verify that every optimizer's state carries the input `NamedSharding` through `init` and `update` under an explicit mesh. For Dew that is the right news, because Dew derives optimizer-state sharding from parameter shapes already: `parameter_spec` is applied to every leaf of the train state, and the docstring says why, that moments and EMA copies have the same shapes as the parameters they track (`src/dew/training/distributed.py:44-52`).

### What Dew should do about it

Add two entries to `OPTIMIZER_MAP`: `optax.contrib.muon` and `optax.contrib.schedule_free_adamw`. Muon is the one that matters, because it is what several 2025 and 2026 frontier runs used, and because it costs one map entry. Two wiring details: pass the caller's schedule explicitly, since `muon` otherwise derives `adam_learning_rate` from `learning_rate`; and schedule-free needs the eval-params hook above, so it is the larger of the two changes.

[adopt: depend on it] `optax.contrib.muon`. Seam: `dew.training.optim.OPTIMIZER_MAP`.
[adopt: depend on it] `optax.contrib.schedule_free_adamw`, with a params-for-eval path. Seams: `dew.training.optim`, `dew.training.trainer`.
[borrow: reimplement the idea] QK-Clip, which is not in Optax and does not belong there. Seam: `dew.nn.attention`.
[later] `galore` and `ademamix`, triggered by a memory or stability problem that adamw and muon do not solve.

---

## 9. Qwix, and FP8 on GPU

**What it is.** Google's JAX quantization library, <https://github.com/google/qwix>. Apache 2.0. PyPI 0.1.8, despite the README still saying "Qwix doesn't provide a PyPI package yet" (`README.md:38-40`); tokamax depends on `qwix>=0.1.2` (`tokamax/pyproject.toml:27`), so the package is real and in use. Last commit 2026-08-31.

**What it covers** (`README.md:7-36`):

| Dimension | Options |
| --- | --- |
| Schemas | weight-only, dynamic-range, static-range |
| Modes | QAT, PTQ, ODML for LiteRT, LoRA and QLoRA |
| Numerics | native `int4`, `int8`, `fp8`; emulated `int1` to `int7`, `nf4` |
| Calibration | `absmax`, `minmax`, `rms`, `fixed` |
| Granularity | per-channel and sub-channel for `dot_general` and `einsum`, per-channel for `conv_general_dilated` |
| Integration | "any Flax Linen or NNX models via a single function call" |

**The API is the reason to care.** Quantization is expressed as regex rules over module paths and applied without editing the model (`README.md:74-101`):

```python
rules = [qwix.QuantizationRule(module_path='.*', weight_qtype='int8', act_qtype='int8')]
ptq_model = qwix.quantize_model(model, qwix.PtqProvider(rules))
```

The parameter tree then holds `QArray(qvalue=int8[...], scale=float32[...])` values wrapped in `WithAux`, visible under `jax.eval_shape(ptq_model.init, ...)` (`README.md:103-131`). Weight quantization is a separate call, `qwix.quantize_params`, because Linen modules are pure functions (`README.md:133-140`). The exported providers are `PtqProvider`, `QtProvider`, `LoraProvider`, `BoxedParamProvider`, `OdmlQatProvider` and `OdmlConversionProvider` (`qwix/__init__.py:28-37`). The README makes a point of shipping no preset recipes: schemas are combinations of rules.

**Quantized training is real, not only fake-quant.** `QtProvider` is "Quantization provider for Quantized Training (QT)" and overrides `dot_general` (`qwix/_src/providers/qt.py:66-84`). Its rule type adds the backward pass explicitly (`qt.py:32-54`):

| Field | Meaning |
| --- | --- |
| `bwd_qtype` | quantize the gradients to this type in the backward pass |
| `bwd_calibration_method` | default `absmax` |
| `bwd_weight_grad_tile_size` | sub-channel tiling for the weight gradient, applied to the incoming gradient and the residual activation |
| `disable_channelwise_axes` | forward and backward |
| `bwd_stochastic_rounding` | `uniform` or `low_bit_uniform` |

Stochastic rounding on gradients is the detail that separates a working low-precision recipe from a diverging one. There is also a `qwix/pallas.py`, which is how `QArray` reaches kernels; recall that tokamax's `dot_product_attention` and `ragged_dot` accept `QArray` inputs (`tokamax/_src/ops/attention/api.py:29,82-84`).

### FP8 on GPU: does Transformer Engine support Ada?

Yes, for per-tensor scaling. NVIDIA's Transformer Engine gates its JAX FP8 support on compute capability, and the check is explicit (`transformer_engine/jax/quantize/helper.py:81-87`, <https://github.com/NVIDIA/TransformerEngine>):

```python
if gpu_arch < 89:  # pre-ada
    return False, "Device compute capability 8.9 or higher required for FP8 execution."
if get_cublasLt_version() < 120103:
    return False, "CublasLt version 12.1.3.x or higher required for FP8 execution on Ada."
if get_cuda_version() < 12010:
    return False, "Cuda version 12.1 or higher required for FP8 execution on Ada."
```

So sm_89, which includes this workstation's RTX 4080, is the first architecture TE accepts for FP8, given cuBLASLt 12.1.3 and CUDA 12.1. Block-scaled MXFP8 is different: it requires compute capability 9.9 or above, cuBLASLt 12.8, CUDA 12.8 and JAX 0.5.3 (`helper.py:99-107`), so Ada and Hopper are excluded and it is Blackwell only. FP4 has its own check (`helper.py:110`). The public API is `is_fp8_available`, `fp8_autocast`, `autocast`, `get_quantization_recipe` and `get_supported_quantization_recipes` (`helper.py:49-63`). MaxText reaches TE for attention through `attention: cudnn_flash_te` and for gemms through `quantization: fp8` (section 1).

### What Dew should do about it

Qwix fits Dew's model code without touching it. Dew's parameters are Linen pytrees, and Qwix's entry point is one call on the module plus one on the params. Nothing in `dew.nn` needs a quantization branch, and nothing in the checkpoint format changes except that leaves become `QArray`.

The sensible order: int8 PTQ first, because it is cheap to validate (quantize, evaluate, compare) and it gives Dew a serving story. Then `QtProvider` for fp8 training, which is where the throughput is, on TPU v5e and above and on Ada or newer GPUs. Note that fp8 training and Dew's fp32 master weights are compatible: the weights stay fp32 while the gemms run in fp8.

Two things not to do. Do not write a quantization branch into Dew's modules; that is what Qwix's provider mechanism exists to avoid. And do not depend on Transformer Engine for the general case: it is the right dependency for cuDNN fused attention and fp8 gemms on NVIDIA and the wrong one for a framework that also runs on TPU, so it belongs behind the same `implementation` seam as everything else.

[adopt: depend on it] Qwix for int8 PTQ, as an optional extra. Seam: a quantization policy in `dew.config`, applied at trainer or export time, not inside `dew.nn`.
[later] `QtProvider` fp8 or int8 quantized training with `bwd_stochastic_rounding`, triggered by a run large enough for throughput to matter. Seam: same.
[later] Transformer Engine as an optional GPU kernel and fp8 gemm backend behind `dew.nn.attention`'s `implementation` argument. On this RTX 4080 it is the only fp8 path that exists, since tokamax's fast kernels do not lower here.
[skip] Qwix's ODML and LiteRT modes. Dew does not target on-device inference.

---

## 10. The scaling book, "How to Scale Your Model"

**What it is.** A book by Jacob Austin, Sholto Douglas, Roy Frostig, Anselm Levskaya, Charlie Chen, Sharad Vikram, Federico Lebron, Peter Choy, Vinay Ramasesh, Albert Webson and Reiner Pope, published by Google DeepMind on 2025-02-04 at <https://jax-ml.github.io/scaling-book/>. Twelve chapters. Chapter 12, on NVIDIA GPUs, is dated 2025-08-18. Not a library: it is the arithmetic that decides which parallelism to use.

**Why Dew needs it.** Dew's mesh has two axes and one rule: shard the largest evenly divisible axis over `fsdp`, replicate below 65536 elements (`src/dew/training/distributed.py:44-58`). Nothing in Dew's docs tells a user when that rule stops working. This book gives the thresholds.

### The chip numbers

TPU, from <https://jax-ml.github.io/scaling-book/tpus/>, section "TPU specs":

| Chip | Pod | HBM/chip | HBM BW/chip | bf16 FLOPs/s | int8 FLOPs/s | ICI BW/link bidi |
| --- | --- | --- | --- | --- | --- | --- |
| TPU v3 | 32x32 | 32GB | 9.0e11 | 1.4e14 | 1.4e14 | 2.0e11 |
| TPU v4p | 16x16x16 | 32GB | 1.2e12 | 2.75e14 | 2.75e14 | 9.0e10 |
| TPU v5p | 16x20x28 | 96GB | 2.8e12 | 4.59e14 | 9.18e14 | 1.8e11 |
| TPU v5e | 16x16 | 16GB | 8.2e11 | 1.97e14 | 3.94e14 | 9.0e10 |
| TPU v6e | 16x16 | 32GB | 1.6e12 | 9.20e14 | 1.84e15 | 1.8e11 |
| TPU7x | 4x4x576 | 192GB | 7.4e12 | 2.30e15 | 4.61e15 | 1.8e11 |

DCN egress is about 6.25e9 bytes/s per TPU, 12.5e9 on v6e and TPU7x, 3.125e9 on v5e. PCIe is about 1.6e10 bytes/s per TPU, 3.2e10 on v6e. Same page.

GPU, from <https://jax-ml.github.io/scaling-book/gpus/>, section "Summary of GPU specs":

| Chip | HBM/chip | HBM BW/chip | bf16 FLOPs/s | fp8/int8 FLOPs/s | fp4 |
| --- | --- | --- | --- | --- | --- |
| A100 | 80GB | 2.0e12 | 3.1e14 | 6.2e14 | none |
| H100 | 80GB | 3.4e12 | 9.9e14 | 2.0e15 | none |
| H200 | 141GB | 4.8e12 | 9.9e14 | 2.0e15 | none |
| B200 | 192GB | 8.0e12 | 2.3e15 | 4.5e15 | 9.0e15 |

An H100 egresses 450GB/s per GPU inside a node and 400GB/s per node beyond it on a DGX SuperPod. A B200 SuperPod doubles intra-node bandwidth to 900GB/s and leaves the scale-out network at 400GB/s, which makes the cross-node rule worse, not better: the critical per-GPU batch rises to `2250e12 / 400e9 = 5625`.

### The decision rules

Let `C` be per-chip FLOPs/s, `W` the relevant bidirectional bandwidth, `B` the total batch in tokens, `N` the chip count, `F` the feed-forward dimension, `X` the FSDP axis size, `Y` the tensor-parallel axis size, and `M_X`, `M_Y` the number of hardware mesh axes each strategy spans. `alpha = C / W_ici` is the ICI arithmetic intensity, 2550 for v5p in bf16.

| Question | Rule | Source |
| --- | --- | --- |
| Is a single chip busy at all? | per-chip batch above 240 tokens, or 267 counting D and F exactly | ch. 1 and ch. 2 quiz 4 |
| Can I use pure data parallelism? | only if parameters plus Adam state fit: `num_params < HBM_per_device / 10`, since bf16 weights plus fp32 moments cost 10 bytes per parameter. On v5p's 96GB that is about 9B, before activations | ch. 5 |
| Is DP or FSDP compute-bound? | per-device batch `B/X > C/W_ici`, so 2550 on v5p over one axis and `2550 / M_X` over several, which is 850 on three | ch. 5 |
| How much tensor parallelism can I afford? | `Y < M_Y * F / 2550`, which is 8 to 16 way for most models. Independent of precision, because int8 doubles the FLOPs and halves the bytes | ch. 5 |
| What is the best FSDP and TP split? | `X_opt = sqrt((B/F) * (M_X/M_Y) * N)` | ch. 5 |
| How low can the per-chip batch go? | mixed FSDP plus TP stays compute-bound while `B/N > alpha^2 / (M_X * M_Y * F)`, about `2550^2 / 2F`, near 100 for F=32768. Eight times better than FSDP alone | ch. 5 |
| When do I need more than one pod? | data parallelism across pods needs a per-pod batch above `C/W_dcn`, about 73,000 tokens on v5p | ch. 5 |
| The same questions on GPUs? | DP and ZeRO need about 2500 tokens per GPU on H100 or B200. TP is `Y < F * W_collective / C`, about `F/2200` inside a node and `F/2475` across nodes, so 8-way intra-node and rarely more | ch. 12 |
| What does MoE change? | the critical per-GPU batch for DP and ZeRO rises by `E/k`, total experts over activated experts. Expert parallelism spans 1 to 2 nodes when `F < 8*C/W_node`, and up to `E` nodes when `F > 8*C/W_node` | ch. 12 |
| Pipeline? | comms are tiny, `2BD / (W * N_MB)` per hop. The cost is the bubble and the code. On TPUs the book skips it because pods are densely connected; on GPUs it is how you cross nodes cheaply, and it usually rules out ZeRO-3, since you would AllGather per stage, so pair it with ZeRO-1 | ch. 5 and ch. 12 |

Two more facts belong in Dew's docs because they constrain model shape, not just sharding:

- Weight matrices are padded to at least 128 in both dimensions, 256 on TPU v6e (ch. 2, "Key Takeaways"). Head dimension and `d_ff` should be multiples of those.
- VMEM bandwidth is about 22 times HBM bandwidth, so an operation that fits in VMEM needs an arithmetic intensity of 10 to 20 rather than 240 (ch. 2). That is the whole argument for fused kernels.

### What Dew should do about it

Dew's docs describe `fsdp_size` as a number you pass. They should state the arithmetic that picks it: pure data parallelism until parameters plus optimizer state exceed `HBM/10`; then FSDP, valid while the per-device batch stays above `2550/M_X`; then a tensor axis capped at `F/2550`, split by `X_opt = sqrt((B/F)(M_X/M_Y)N)`. Dew cannot express a tensor axis yet, so the honest version of that document also says where the two-axis mesh runs out, which is a per-device batch near 850 tokens on v5p-class hardware.

[borrow: reimplement the idea] the decision rules, as a page in `docs/concepts/` and a docstring on `build_mesh`. Nothing to depend on; this is arithmetic.

---

## 11. Briefly: the rest of the ecosystem

### MaxDiffusion

<https://github.com/AI-Hypercomputer/maxdiffusion>, Apache 2.0, last commit 2026-09-01. The diffusion sibling of MaxText, and it shards DiTs with the same machinery: a `logical_axis_rules` table plus `ici_*` and `dcn_*` sizes for four axes, `data`, `fsdp`, `context` and `tensor` (`src/maxdiffusion/configs/base_flux_dev.yml:151-178`). The Flux rules are short enough to quote in full:

```
['batch', ['data','fsdp']], ['activation_batch', ['data','fsdp']],
['activation_heads', 'tensor'], ['activation_length', 'context'],
['activation_kv_length', 'context'], ['activation_kv', 'tensor'],
['mlp','tensor'], ['embed','fsdp'], ['heads', 'tensor'],
['conv_batch', ['data','fsdp']], ['out_channels', 'tensor'], ['conv_out', 'fsdp'],
```

Note two things. The sequence dimension of a DiT is sharded over a `context` axis, which is how long-sequence video models fit; and the UNet convolutions get their own logical names (`conv_batch`, `out_channels`, `conv_out`), so the same rules table serves transformers and convolutional stacks. `data_sharding: [['data', 'fsdp', 'context', 'tensor']]` (`:165`), and `dataset_type` is `tfrecord`, `hf`, `tf`, `grain` or `synthetic` (`:186`).

Model coverage, from `src/maxdiffusion/configs/`: Stable Diffusion 1.4, 1.5, 2.1 and XL, XL Lightning, Flux dev and schnell, Flux 2 Klein including a 9B config, Wan 1.3B, 14B, 27B, image-to-video and Animate, and Z-Image with a turbo variant. `src/maxdiffusion/models/` also holds `ltx_video`, `ltx2`, `qwen3_flax`, `controlnet_flax`, `lora.py`, `lora_nnx.py` and `quantizations.py`.

[borrow: reimplement the idea] the DiT and UNet logical axis rules above, as the concrete target for Dew's diffusion objective once logical axes exist. It is the closest published analogue to what Dew already trains. Seams: `dew.training.distributed`, `dew.nn.dit`.

### Levanter and Haliax

Both moved. Levanter's README carries a notice: "Levanter has been merged into Marin as of November 2025. All active development now happens in the Marin monorepo at `lib/levanter/`", with `pip install levanter` still working (`levanter/README.md:1-12`). Haliax says the same: "Development has moved into https://github.com/marin-community/marin monorepo" (`haliax/README.md:1-3`). Both are Apache 2.0. My Levanter clone's last commit is 2025-11-07, the merger notice itself; Haliax's is 2026-09-01, a README pointer.

Haliax is the interesting half: "a JAX library for building neural networks with named tensors", explicitly in the tradition of Tensor Considered Harmful, where "named tensors improve the legibility and compositionality of tensor programs by using named axes instead of positional indices" (`haliax/README.md:23-25`).

This is the same idea as MaxText's logical axis rules, taken further. MaxText keeps positional arrays and annotates them; Haliax makes the named axis the primitive, so an axis cannot be sharded by the wrong name because there is no position to confuse. For Dew the practical read is that the named-axis idea has two published forms, a light one (annotate and map, MaxText and MaxDiffusion, works with Linen today) and a heavy one (named tensors throughout, Haliax, a rewrite). The light one is the one to take.

[skip] both as dependencies: the development has moved into a monorepo aimed at a different project, and adopting Haliax means rewriting Dew's array code.
[borrow: reimplement the idea] the argument for names over positions, which section 1 already recommends in its lighter form.

### EasyDeL

<https://github.com/erfanzar/EasyDeL>, Apache 2.0, version 0.3.0, last commit 2026-04-22, so less active than the rest of this list. Community rather than Google. It is a broad JAX framework for training, fine-tuning and serving many Hugging Face architectures, with `trainers`, `inference`, `layers`, `modules`, `caching`, `operations` and an `axis.py`. It requires Python `>=3.11,<3.14` (`pyproject.toml:11`). It overlaps Dew's scope more than it complements it, and it is one maintainer's project rather than a lab's reference code.

[skip]. Nothing here that Dew needs and cannot get from a first-party library, and taking it would mean importing a second framework's opinions about trainers.

### Pathways on Cloud

Google's single-controller runtime, the one used internally to train Gemini, exposed on Cloud through GKE. Primary docs: <https://docs.cloud.google.com/ai-hypercomputer/docs/workloads/pathways-on-cloud/pathways-intro>. "Pathways simplifies large-scale machine learning computations by enabling a single JAX client to orchestrate workloads across multiple large TPU slices, potentially spanning thousands of TPU chips."

The difference that matters to Dew's code is the programming model, not the deployment. Under Pathways, `jax.process_index()` is always 0, and `jax.devices()` and `jax.local_devices()` return every device in the job (porting guide, "Process index"). Dew's data path assumes the opposite: `ShardByJaxProcess` shards by `process_index` and `process_count` (`grain/_src/core/sharding.py:57-66`, used at `dew/data/dataloaders.py:190`), and `shard_batch` calls `jax.make_array_from_process_local_data` (`dew/training/distributed.py:73-77`), which is a multi-controller idiom. Neither is wrong, but neither survives a move to Pathways unchanged.

The rest of the picture, from the same docs. `pathwaysutils` (<https://github.com/AI-Hypercomputer/pathways-utils>) provides a proxy JAX backend selected with `JAX_PLATFORMS=proxy`, profiling that covers the Pathways components as well as the user program, a custom Orbax `ArrayHandler` registered by `import pathwaysutils; pathwaysutils.initialize()` with `ENABLE_PATHWAYS_PERSISTENCE=1` so existing Orbax code keeps working, and elastic training primitives. Elastic training is configured with `max_elastic_down_event_count` and `max_reshard_retry_count`; a slice loss surfaces as a `jax.errors.JaxRuntimeError`, and the program is expected to rebuild its computation for the new device count (resilient-training guide). Data loading runs on a CPU VM rather than a TPU VM, which adds latency, and the recommended fix at scale is to run the input pipeline on the accelerator hosts with colocated Python; MaxText's `multihost_dataloading.py` `RemoteIterator` is named as the reference implementation. There is a persistent compilation cache in GCS, enabled by default through `--gcs_scratch_location`.

[later] Pathways, triggered by a multi-slice run. The preparatory work in Dew is small and worth knowing about now: keep process-count assumptions in one place, which today means `dew.data.dataloaders` and `dew.training.distributed`, so that switching to a single-controller runtime is a change in two files rather than everywhere.

### Flax NNX against Linen in 2026

Linen is not deprecated. The Flax team's own statement, in the NNX documentation index (`flax/docs_nnx/index.rst:20`):

> Flax Linen API is not going to be deprecated in the near future as most of Flax users still rely on this API. However, new users are encouraged to use Flax NNX.

The README frames NNX as the evolution of Linen and notes that Linen's documentation now lives on its own site (`flax/README.md:15-32`). Last commit in my clone 2026-08-21.

Where the ecosystem actually sits, from the counts in this research:

| Project | Linen | NNX |
| --- | --- | --- |
| gemma | 45 files | 0 files |
| Kauldron | `model: nn.Module` where `nn` is `flax.linen` (`trainer_lib.py:29,178`) | not the model type |
| MaxText | 37 files | 89 files |
| Tunix | 1 file | 63 files |
| Dew | all | none |

So the split is real and it runs along a line: the model libraries and the research trainer are Linen, the newer training and post-training code is NNX. Dew is in the larger, older camp, together with gemma and Kauldron, which are the two libraries Dew would most plausibly interoperate with.

[skip] migrating Dew to NNX now. There is no deprecation pressure, the two libraries Dew would borrow from most are Linen, and the cost is the whole model layer.
[later] revisit if Tunix becomes the post-training path Dew wants, since that is the one hard NNX boundary found in this research. The cheaper answer even then is a param-tree adapter, as in section 4.

### One thing not on the list: hackable_diffusion

While reading gemma's dependencies I found `hackable-diffusion @ git+https://github.com/google/hackable_diffusion.git` (`gemma/pyproject.toml:41`). The repository exists, is Apache 2.0, has 160 stars and was pushed 2026-08-18. gemma uses it for a diffusion SFT adapter alongside its Diffusion Gemma model. It was not in this assignment and I did not read its source, so I am flagging it rather than assessing it: a Google diffusion library that DeepMind's own Gemma library depends on is the most obviously relevant unexamined thing I ran into, given what Dew is.

---

## 12. Ranked: the ten things to take first

Ordered by expected value at scale, not by effort. Each line says why it matters with the evidence behind it.

**1. Logical axis rules instead of shape inference.** [borrow] Seam: `dew.training.distributed` plus annotations in `dew.nn`.
Evidence: MaxText expresses twelve physical axes through a 109-line rules table (`base.yml:549-658`), and MaxDiffusion applies the identical pattern to DiTs and UNets with four axes (`base_flux_dev.yml:151-178`). Dew's `parameter_spec` can only ever produce FSDP, so by section 10's arithmetic Dew is capped near a per-device batch of 850 tokens on v5p, where FSDP plus tensor parallelism reaches about 100. This one change is the precondition for tensor, context and expert parallelism, and therefore for training anything Dew cannot already train.

**2. Sequence packing with segment ids.** [adopt] Seam: data source in `dew.data`.
Evidence: `grain.experimental.FirstFitPackIterDataset` emits `_segment_ids` and `_positions` per feature (`packing_packed_batch.py:116-117`), which is what a block-diagonal mask and per-document RoPE need. Dew's `TokenWindowSource` chops a concatenated stream at fixed stride (`dew/data/sources/text.py:24-63`), so today every training window that straddles a document boundary trains attention across unrelated text with no marker. Grain's own guidance is that packing is one of exactly three reasons to use the `Dataset` API (`docs/api_choice.md:10-16`).

**3. `optax.contrib.muon`.** [adopt] Seam: `dew.training.optim.OPTIMIZER_MAP`.
Evidence: one dictionary entry, because `muon()` already partitions by `ndim == 2` and routes non-matrices to adamw internally (`optax/contrib/_muon.py`), so Dew's parameter tree needs no restructuring. It is the cheapest item on this list per unit of frontier parity.

**4. Vocabulary tiling for the LM head.** [borrow] Seam: the LM objective.
Evidence: MaxText's `num_vocab_tiling` chunks the cross-entropy along batch-sequence and is "highly recommended for models with large vocabularies (e.g. Gemma)", with `vocab_tiling_ag_once` to gather the head once for the backward (`base.yml:720-729`). tokamax ships the fused form as `linear_softmax_cross_entropy_loss`. For a 256k-vocabulary decoder the logits are the largest single activation in the step, so this is the highest memory saving available for the least architectural risk.

**5. A named remat policy with host offload.** [borrow] Seam: `dew.training.objective_trainer`.
Evidence: MaxText offers eleven named policies plus a `custom` mode where each of about twenty tensors takes `remat`, `device` or `offload` (`base.yml:373-403`), and pins `decoder_layer_input` to `device` because it is the remat restart point. All-or-nothing checkpointing forces a choice between speed and fitting; the named middle ground is what makes large models trainable on a given HBM budget.

**6. Replica-parallel checkpoint writes, and the iterator as a separate checkpointable.** [adopt] Seam: checkpointer in `dew.training.trainer`.
Evidence: `use_replica_parallel` with its size and replica caps (`orbax .../v1/_src/context/options.py:342-349,405-408`) divides write bytes by the replica count, and Dew's default mesh always has replicas when `fsdp_size < device_count`. Separately, making the data iterator its own checkpointable removes the variable-length-pytree-leaf workaround at `trainer.py:285-288` and lets Grain's own `CheckpointHandler` do the work (`grain checkpoint/handler.py:33`).

**7. tokamax as the TPU kernel path, and `ragged_dot` for MoE.** [adopt, optional extra] Seam: `dew.nn.attention`, future `dew.nn.moe`.
Evidence: Dew's `tpu` implementation imports `jax.experimental.pallas.ops.tpu.flash_attention` (`dew/nn/attention.py:177-182`), while MaxText has moved to a vendored tokamax splash kernel and exposes `use_tokamax_splash` and `use_tokamax_gmm` (`base.yml:286,1350`). tokamax's signature is a superset of the one Dew already calls, and it supports all eighteen MoE tile configs against six for megablox and JAX ragged dot (`base.yml:261-262`). My own test (appendix A) shows the fast paths do not reach consumer Ada, so this is a TPU and H100-and-newer item, which is exactly where it matters.

**8. `sharding_tolerance` as a startup assertion.** [borrow] Seam: `dew.training.distributed`.
Evidence: MaxText fails a run when more than 2 percent of parameters are unsharded (`base.yml:672-673`). Dew's `parameter_spec` silently returns `P()` when no axis divides evenly by `fsdp_size` (`distributed.py:56-58`), so a model whose dimensions do not match the device count trains at full replication and reports nothing. This is a dozen lines that turn a silent memory blow-up into an error, which is why it ranks above larger items.

**9. The scaling rules written into Dew's docs.** [borrow] Seam: `docs/concepts/`.
Evidence: the thresholds in section 10, notably per-device batch above `2550/M_X` for FSDP, `Y < F/2550` for tensor parallelism, `X_opt = sqrt((B/F)(M_X/M_Y)N)` for the split, and about 73,000 tokens per pod before DCN binds. A framework that cannot tell a user which mesh to pick makes the user guess, and the arithmetic is public.

**10. Goodput accounting.** [borrow] Seam: `dew.telemetry`.
Evidence: MaxText treats it as a first-class metric with five flags and a 30 second upload interval (`base.yml:1078-1084`). Dew measures step time, which answers "how fast is a step" and not "what fraction of the last day was training". Every item on this list about checkpointing and elasticity is justified by a goodput number, so the measurement should come before the mitigation.

Two honourable mentions, both cheap and both narrow. `optax.contrib.schedule_free_adamw` is one map entry plus an eval-params path, and it removes the decay schedule as a tuning axis. Qwix int8 PTQ is two calls and touches no model code (`qwix.quantize_model`, `qwix.quantize_params`), which gives Dew a serving story before it needs a training-precision story.

And one thing explicitly not on the list: migrating to Flax NNX. Linen is not deprecated (`flax/docs_nnx/index.rst:20`), and the two libraries Dew would interoperate with most, gemma and Kauldron, are both Linen.

---

## Appendix A: the tokamax GPU test

Run on this workstation, 2026-09-02. NVIDIA GeForce RTX 4080, compute capability 8.9, driver 595.84, CUDA 13.2. A sibling agent's process appeared on the GPU part-way through the session, so treat the millisecond figures as correctness evidence rather than clean benchmarks; the purpose was to find out which paths lower and run at all. At the start of the run `nvidia-smi --query-compute-apps=pid` showed only pid 3243, and the card reported 2535MHz SM clock against a 3105MHz maximum, 45.7W and 49C.

Reproduce:

```sh
mkdir -p /tmp/research/GoogleStack && cd /tmp/research/GoogleStack
uv venv --python 3.12 tokamax-venv
uv pip install --python ./tokamax-venv/bin/python "jax[cuda12]" tokamax triton
./tokamax-venv/bin/python tokamax_gpu_test.py
```

The script is at `/tmp/research/GoogleStack/tokamax_gpu_test.py`. Note that `triton` is not a dependency of the `tokamax` wheel; it is listed only in tokamax's `cuda` extra, along with `jax[cuda13]` (`tokamax/pyproject.toml:65-71`). Without it, the Triton path fails with a bare `ImportError` at import time rather than a useful message.

Output, with autotuning-cache warnings removed:

```
jax 0.11.1 | jaxlib 0.11.1
device NVIDIA GeForce RTX 4080 | compute_capability 8.9
tokamax 0.0.13
attention default impls: ('mosaic', 'triton', 'xla')
gpu_utils: is_sm80=True is_sm90=False is_sm100=False mosaic=True triton=True

== dot_product_attention  B=2 T=2048 heads=8/8 head_dim=128 bf16, causal, softcap=30 ==
  xla          ok    max_abs_err_vs_fp32=0.0112    2.40 ms/call
  xla_chunked  ok    max_abs_err_vs_fp32=0.0090    1.89 ms/call
  triton       FAIL  RuntimeError: No supported GPU devices found, please specify an abstract GPU device
  mosaic       FAIL  NotImplementedError: Only supported for sm90 and sm100 GPUs.
  cudnn        FAIL  NotImplementedError: `logits_soft_cap` not supported.

== backward pass (training path) ==
  xla          ok    dq finite=True shape=(2, 2048, 8, 128)
  triton       FAIL  RuntimeError: No supported GPU devices found ...

== layer_norm / rmsnorm  x=bf16[4096,2048] ==
  xla          ok    layer_norm err=0.0078  rmsnorm(subtract_mean=False) err=0.0078
  triton       FAIL  RuntimeError: No supported GPU devices found ...

== ragged_dot  E=8 groups of 128 tokens, D=2048, F=512 ==
  xla          ok    max_abs_err=0.4999 (rel 2.3e-03) out=(1024, 512)
  triton       FAIL  RuntimeError: No supported GPU devices found ...
  mosaic       FAIL  NotImplementedError: Unsupported GPU architecture.
```

Three findings worth keeping:

1. The Triton failure is JAX's, not tokamax's. `jax/_src/pallas/triton/gpu_info.py:27-51` enumerates supported device kinds, and the RTX 4080 is not among them while the RTX 4090 is; `get_gpu_info()` raises `Unsupported GPU device kind: NVIDIA GeForce RTX 4080`, which surfaces as the runtime error above.
2. tokamax reports Mosaic GPU as supported on Ada and then refuses. `has_mosaic_gpu_support` returns True at compute capability 8.0 and above (`tokamax/_src/gpu_utils.py:63-73`), but the attention kernel raises "Only supported for sm90 and sm100 GPUs" (`pallas_mosaic_gpu.py:51-52`) and ragged dot raises "Unsupported GPU architecture".
3. The cuDNN path cannot do logit softcapping. That is a real constraint for Gemma-style models on GPU, independent of tokamax.

## Appendix B: versions, licences and activity at a glance

All Apache 2.0 unless noted. "Last commit" is from my clone on 2026-09-02.

| Library | Version | Last commit | Hardware focus | Verdict |
| --- | --- | --- | --- | --- |
| MaxText | no PyPI package, run from source | 2026-09-01 | TPU first, GPU supported | borrow, do not depend |
| Kauldron | PyPI 1.4.4, needs Python 3.12+ | 2026-09-02 | TPU and GPU | borrow the design |
| gemma | PyPI 4.0.1, repo says 4.1.0 | 2026-08-04 | TPU and GPU | borrow `TransformerLike` |
| Tunix | PyPI 0.0.0, install from git | 2026-09-01 | TPU first | skip today, NNX boundary |
| tokamax | PyPI 0.0.13 | 2026-09-02 | TPU, GPU sm90+ for Mosaic, sm80+ nominally for Triton | adopt as optional extra |
| Orbax | `orbax-checkpoint` 0.12.4 | 2026-09-02 | any | adopt more of it |
| Grain | PyPI 0.2.18, clone 0.2.19 | 2026-08-31 | host side | adopt the `Dataset` API |
| Optax | PyPI 0.2.8, clone 0.2.9.dev | 2026-08-30 | any | adopt muon |
| Qwix | PyPI 0.1.8 | 2026-08-31 | TPU and GPU | adopt for int8 PTQ |
| MaxDiffusion | run from source | 2026-09-01 | TPU first | borrow the DiT rules |
| Levanter | PyPI `levanter`, development moved to Marin | 2025-11-07 | TPU and GPU | skip |
| Haliax | PyPI `haliax`, development moved to Marin | 2026-09-01 | TPU and GPU | skip, borrow the idea |
| EasyDeL | PyPI 0.3.0 | 2026-04-22 | TPU and GPU | skip |
| Flax | Linen and NNX both current | 2026-08-21 | any | stay on Linen |
| Transformer Engine | NVIDIA, Apache 2.0 | not cloned, read on GitHub | NVIDIA only, FP8 needs sm_89+ | later, GPU only |
