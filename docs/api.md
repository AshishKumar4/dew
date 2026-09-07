# Module index

For signatures, defaults, and behavior, start with the [core API reference](reference/core-api.md).
This generated index lists declared exports, or public classes and functions defined
directly in a module when it has no export list. An empty row does not mean the
module has no importable names.


## The package

| Module | Exports |
| --- | --- |
| `dew` | `__version__`, `Aux`, `CFG`, `Checkpoints`, `Condition`, `Dataset`, `EMASpec`, `Evaluation`, `Field`, `ImageGrid`, `InputSpec`, `Layout`, `MeshSpec`, `Objective`, `Process`, `Representations`, `Step`, `TextSamples`, `TokenScores`, `Tracker`, `TrainState`, `Trainer`, `VideoGrid`, `LocalTracker`, `Trackers`, `WandbTracker`, `datasets`, `encoders`, `evaluate`, `metrics`, `models`, `presets`, `sample`, `samplers` |
| `dew.registry` | `Registry`, `models`, `presets`, `samplers`, `datasets`, `encoders`, `metrics`, `objectives`, `mixers`, `towers`, `projectors`, `REGISTRIES`, `resolve_dtype`, `dtype_name`, `with_precision` |
| `dew.artifacts` | `ImageGrid`, `Representations`, `TextSamples`, `TokenScores`, `VideoGrid`, `agree_process_phase`, `broadcast_from_process_zero`, `collective_host`, `host` |
| `dew.config` | `ModelConfig`, `OptimConfig`, `RunConfig`, `TrainerConfig`, `Wandb` |

## Training

| Module | Exports |
| --- | --- |
| `dew.training` | `Aux`, `Checkpoints`, `DEFAULT_RULES`, `EMASpec`, `Evaluation`, `Layout`, `MeshSpec`, `Metric`, `Objective`, `Profile`, `Quantization`, `Rollout`, `Step`, `Tracker`, `TrainState`, `Trainer`, `LocalTracker`, `Trackers`, `WandbTracker`, `apply_quantization`, `build_mesh`, `build_optimizer`, `ema_update`, `evaluate`, `everything`, `prepare_process`, `run_timestamp`, `under`, `write_back` |
| `dew.training.distributed` | `DevicePrefetchIterator`, `Layout`, `MeshSpec`, `batch_shardings`, `build_mesh`, `local_rows`, `minimum_across_processes`, `parameter_spec`, `shard_batch` |
| `dew.training.optim` | `build_optimizer`, `muon_weight_dimension_numbers`, `scale_by_qk_clip` |
| `dew.training.quantization` | `Quantization`, `apply_quantization` |
| `dew.training.runtime` | `prepare_process`, `run_timestamp` |
| `dew.telemetry.instrumentation` | `compiled_flops`, `default_compilation_cache_dir`, `enable_compilation_cache`, `hlo_flops`, `model_flops_utilization`, `peak_flops`, `step_flops` |
| `dew.io` | `publish` |

## Objectives

| Module | Exports |
| --- | --- |
| `dew.objectives` |  |
| `dew.objectives.diffusion` | `DiffusionObjective`, `DiffusionRunConfig`, `MaskedDiffusionObjective`, `StableDiffusionAutoencoder`, `TextCondition`, `VALIDATION_SAMPLES` |
| `dew.objectives.jepa` |  |
| `dew.objectives.lm` |  |
| `dew.objectives.rl` | `ADVANTAGES_KEY`, `DPOObjective`, `GRPOObjective`, `IDS_KEY`, `OLD_LOG_PROBS_KEY`, `PREFERENCE_IDS_KEY`, `PREFERENCE_MASK_KEY`, `RESPONSE_MASK_KEY`, `REWARDS_KEY`, `Reward`, `SampledRollout`, `FlowGRPOObjective`, `FlowReward`, `FlowRollout` |

## Diffusion

| Module | Exports |
| --- | --- |
| `dew.diffusion` | `NoiseScheduler`, `GeneralizedNoiseScheduler`, `DiscreteNoiseScheduler`, `ContinuousNoiseScheduler`, `LinearNoiseScheduler`, `linear_beta_schedule`, `CosineNoiseScheduler`, `cosine_beta_schedule`, `ExpNoiseScheduler`, `exp_beta_schedule`, `CosineGeneralNoiseScheduler`, `CosineContinuousNoiseScheduler`, `SqrtContinuousNoiseScheduler`, `KarrasVENoiseScheduler`, `EDMNoiseScheduler`, `FlowMatchingScheduler`, `compute_resolution_shift`, `expand`, `PredictionTransform`, `EpsilonPredictionTransform`, `DirectPredictionTransform`, `VPredictionTransform`, `FlowMatchPredictionTransform`, `KarrasPredictionTransform`, `Weighting`, `ScheduleWeighting`, `MinSNR`, `broadcast_rates`, `Process`, `Denoiser`, `presets`, `discrete` |
| `dew.diffusion.presets` | `Cosine`, `EDM`, `Flow`, `Karras`, `Preset`, `Sqrt` |
| `dew.diffusion.schedules` | `NoiseScheduler`, `GeneralizedNoiseScheduler`, `DiscreteNoiseScheduler`, `ContinuousNoiseScheduler`, `LinearNoiseScheduler`, `linear_beta_schedule`, `CosineNoiseScheduler`, `cosine_beta_schedule`, `ExpNoiseScheduler`, `exp_beta_schedule`, `CosineGeneralNoiseScheduler`, `CosineContinuousNoiseScheduler`, `SqrtContinuousNoiseScheduler`, `KarrasVENoiseScheduler`, `EDMNoiseScheduler`, `FlowMatchingScheduler`, `compute_resolution_shift`, `expand` |
| `dew.diffusion.transforms` | `DirectPredictionTransform`, `EpsilonPredictionTransform`, `FlowMatchPredictionTransform`, `KarrasPredictionTransform`, `MinSNR`, `PredictionTransform`, `ScheduleWeighting`, `VPredictionTransform`, `Weighting`, `broadcast_rates` |
| `dew.diffusion.discrete` | `DiscreteDenoiser`, `DiscreteProcess`, `LogLinear`, `MDLM`, `MaskingSchedule`, `Unmask` |

## Sampling

| Module | Exports |
| --- | --- |
| `dew.sampling` | `Solver`, `DDPM`, `DDIM`, `Euler`, `EulerAncestral`, `Heun`, `RK4`, `MultiStepDPM`, `DPMSolverPP`, `CFG`, `sample`, `generate`, `Generation`, `Sampling`, `TextToImage`, `FlowSDE`, `FlowTrajectory`, `GaussianTransition`, `flow_transition`, `sample_trajectory` |
| `dew.sampling.solvers` | `Solver`, `DDPM`, `DDIM`, `Euler`, `EulerAncestral`, `Heun`, `RK4`, `MultiStepDPM`, `DPMSolverPP` |
| `dew.sampling.text` | `Generation`, `Sampling`, `generate` |

## Models

| Module | Exports |
| --- | --- |
| `dew.nn` |  |
| `dew.nn.backbones` | `Unet`, `UViT`, `SimpleUDiT`, `SimpleDiT`, `SimpleMMDiT`, `HierarchicalMMDiT`, `HybridSSMAttentionDiT`, `CausalTransformer`, `VideoDiT`, `UNet3D` |
| `dew.nn.autoencoders` | `AutoEncoder`, `StableDiffusionVAE`, `SimpleAutoEncoder` |
| `dew.nn.moe` | `ExpertLinear`, `ExpertMLP`, `Router`, `RouterMoments`, `SparseMLP`, `calculate_load_balance_updates`, `deepseek_v2_aux_loss`, `global_router_loss`, `load_balance_update`, `router_moments`, `sequence_router_losses` |
| `dew.nn.text_encoders` | `CLIP`, `CLIPAttention`, `CLIPEncoderLayer`, `CLIPModel`, `CLIPTextModel`, `CLIPTextTransformer`, `CLIPTowerOutput`, `CLIPVisionTransformer`, `MLP`, `T5Block`, `T5DenseGatedGeluDense`, `T5DenseReluDense`, `T5EncoderModel`, `T5EncoderTransformer`, `T5LayerNorm`, `T5SelfAttention`, `quick_gelu`, `translate_clip_config`, `translate_clip_weights`, `translate_config`, `translate_t5_config`, `translate_t5_weights`, `translate_vision_config`, `translate_weights` |
| `dew.nn.sharding` | `declared_axes`, `is_heuristic`, `logical_axes`, `microbatches`, `parameter_path`, `pipeline_microbatches`, `pipeline_stages`, `sequence_shards` |

## Inputs and data

| Module | Exports |
| --- | --- |
| `dew.inputs` | `Field`, `Condition`, `InputSpec`, `ConditionEncoder`, `CLIPText`, `T5Text`, `CharTable`, `rebuild`, `unit_range`, `pixel_field` |
| `dew.data` | `AestheticCoyo`, `AutoAudioProcessor`, `AutoTextTokenizer`, `Batch`, `ByteTokenizer`, `Checkpointable`, `Dataset`, `DatasetSpec`, `DiffusionDB`, `HFDatasetSource`, `HFImages`, `HFTokenizer`, `IDS_KEY`, `ImageDataset`, `Laion12mCoco`, `Laion2bAesthetic`, `LaionaCoco`, `LaionaCocoCoyo`, `Loading`, `LocalVideos`, `MASK_KEY`, `OnlineImages`, `OxfordFlowers`, `PackedTokens`, `PreferencePairs`, `Prompts`, `Role`, `TokenDocumentSource`, `TokenFileSource`, `TokenWindows`, `VideoDataset`, `VoxCeleb2`, `local_batch` |

## Evaluation and interop

| Module | Exports |
| --- | --- |
| `dew.eval` | `ImageMetric`, `frames`, `clip`, `clip_score`, `fid`, `frechet_distance`, `peak_signal_noise_ratio`, `psnr`, `structural_similarity`, `ssim` |
| `dew.interop` | `dequantize_checkpoint`, `dequantize_fp8_blocks`, `fp8_block`, `load_params`, `load_pretrained_decoder`, `pull_from_hub`, `push_to_hub`, `save_hf_layout`, `save_params`, `save_pretrained_decoder`, `translate_config`, `translate_weights` |
| `dew.rl` | `gae`, `group_advantage`, `masked_mean`, `masked_whiten`, `rloo_advantage`, `clipped_surrogate`, `k3_kl`, `preference_logsigmoid`, `sequence_log_ratio`, `token_log_ratio`, `token_mean` |
