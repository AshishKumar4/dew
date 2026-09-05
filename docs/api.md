# API

The public modules and the names each exports. This page is written by
`tools/api_page.py` from the code, and a test keeps them equal; edit the
`__all__` of a module, not this file.


## The package

| Module | Exports |
| --- | --- |
| `dew` | `__version__`, `Aux`, `CFG`, `Checkpoints`, `Condition`, `Dataset`, `EMASpec`, `Field`, `ImageGrid`, `InputSpec`, `Layout`, `MeshSpec`, `Objective`, `Process`, `Representations`, `Step`, `TextSamples`, `TokenScores`, `Tracker`, `TrainState`, `Trainer`, `VideoGrid`, `WandbTracker`, `datasets`, `encoders`, `metrics`, `models`, `presets`, `sample`, `samplers` |
| `dew.registry` | `Registry`, `models`, `presets`, `samplers`, `datasets`, `encoders`, `metrics`, `objectives`, `resolve_dtype`, `dtype_name`, `with_precision` |
| `dew.artifacts` | `ImageGrid`, `Representations`, `TextSamples`, `TokenScores`, `VideoGrid`, `host` |
| `dew.config` | `ModelConfig`, `OptimConfig`, `RunConfig`, `TrainerConfig`, `Wandb` |

## Training

| Module | Exports |
| --- | --- |
| `dew.training` | `Aux`, `Checkpoints`, `DEFAULT_RULES`, `EMASpec`, `Layout`, `MeshSpec`, `Metric`, `Objective`, `Profile`, `Step`, `Tracker`, `TrainState`, `Trainer`, `WandbTracker`, `build_mesh`, `build_optimizer`, `ema_update`, `everything`, `prepare_process`, `run_timestamp`, `under`, `write_back` |
| `dew.training.distributed` | `DevicePrefetchIterator`, `Layout`, `MeshSpec`, `batch_shardings`, `broadcast_from_process_zero`, `build_mesh`, `minimum_across_processes`, `parameter_spec`, `shard_batch` |
| `dew.training.optim` | `build_optimizer`, `muon_weight_dimension_numbers` |
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
| `dew.sampling` | `Solver`, `DDPM`, `DDIM`, `Euler`, `EulerAncestral`, `Heun`, `RK4`, `MultiStepDPM`, `CFG`, `sample`, `generate`, `TextToImage` |
| `dew.sampling.solvers` | `Solver`, `DDPM`, `DDIM`, `Euler`, `EulerAncestral`, `Heun`, `RK4`, `MultiStepDPM` |
| `dew.sampling.text` | `generate` |

## Models

| Module | Exports |
| --- | --- |
| `dew.nn` |  |
| `dew.nn.backbones` | `Unet`, `UViT`, `SimpleUDiT`, `SimpleDiT`, `SimpleMMDiT`, `HierarchicalMMDiT`, `HybridSSMAttentionDiT`, `CausalTransformer`, `VideoDiT`, `UNet3D` |
| `dew.nn.autoencoders` | `AutoEncoder`, `StableDiffusionVAE`, `SimpleAutoEncoder` |
| `dew.nn.moe` | `ExpertLinear`, `ExpertMLP`, `Router`, `SparseMLP`, `calculate_load_balance_updates` |
| `dew.nn.text_encoders` | `CLIP`, `CLIPAttention`, `CLIPEncoderLayer`, `CLIPMLP`, `CLIPModel`, `CLIPTextModel`, `CLIPTextTransformer`, `CLIPTowerOutput`, `CLIPVisionTransformer`, `T5Block`, `T5DenseGatedGeluDense`, `T5DenseReluDense`, `T5EncoderModel`, `T5EncoderTransformer`, `T5LayerNorm`, `T5SelfAttention`, `quick_gelu`, `translate_clip_config`, `translate_clip_weights`, `translate_config`, `translate_t5_config`, `translate_t5_weights`, `translate_vision_config`, `translate_weights` |
| `dew.nn.sharding` | `declared_axes`, `is_heuristic`, `logical_axes`, `parameter_path` |

## Inputs and data

| Module | Exports |
| --- | --- |
| `dew.inputs` | `Field`, `Condition`, `InputSpec`, `ConditionEncoder`, `CLIPText`, `T5Text`, `CharTable`, `rebuild`, `unit_range` |
| `dew.data` | `AestheticCoyo`, `AutoAudioProcessor`, `AutoTextTokenizer`, `Batch`, `ByteTokenizer`, `CC12M`, `CC3M`, `Combined30M`, `CombinedAesthetic`, `CombinedMsml612`, `CombinedOnline`, `Checkpointable`, `Dataset`, `DatasetSpec`, `DiffusionDB`, `HFDatasetSource`, `HFImages`, `HFTokenizer`, `ImageDataset`, `Laion12mCoco`, `Laion2bAesthetic`, `LaionaCoco`, `LaionaCocoCoyo`, `Loading`, `LocalVideos`, `OnlineImages`, `OxfordFlowers`, `PackedTokens`, `TokenDocumentSource`, `TokenFileSource`, `TokenWindows`, `VideoDataset`, `VoxCeleb2`, `local_batch` |

## Evaluation and interop

| Module | Exports |
| --- | --- |
| `dew.eval` | `ImageMetric`, `frames`, `clip`, `clip_score`, `fid`, `frechet_distance`, `peak_signal_noise_ratio`, `psnr`, `structural_similarity`, `ssim` |
| `dew.interop` | `load_params`, `load_pretrained_decoder`, `pull_from_hub`, `push_to_hub`, `save_hf_layout`, `save_params`, `save_pretrained_decoder`, `translate_config`, `translate_weights` |
| `dew.rl` | `gae`, `group_advantage`, `masked_mean`, `masked_whiten`, `rloo_advantage`, `clipped_surrogate`, `k3_kl`, `sequence_log_ratio`, `token_log_ratio`, `token_mean` |
