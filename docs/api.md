# API

The public modules, and what each one is for.

## Training

| Module | Contents |
| --- | --- |
| `dew.training` | `ObjectiveTrainer`, `SimpleTrainer`, `TrainState`, `SimpleTrainState`, `build_optimizer`, `prepare_process` |
| `dew.training.distributed` | `build_mesh`, `parameter_spec`, `state_sharding_tree`, `batch_sharding`, `shard_batch`, `DevicePrefetchIterator`, `DATA_AXIS`, `FSDP_AXIS`, `BATCH_SPEC`, `DEFAULT_MIN_SHARD_SIZE` |
| `dew.telemetry.instrumentation` | `step_flops`, `compiled_flops`, `model_flops_utilization`, `enable_compilation_cache`, `default_compilation_cache_dir`, `PEAK_FLOPS_PER_DEVICE` |
| `dew.checkpoints.utils` | `get_latest_checkpoint`, `serialize_model` |
| `dew.config` | `RunConfig`, `ModelConfig`, `DataConfig`, `OptimConfig`, `TrainerConfig` |

## Objectives

| Module | Contents |
| --- | --- |
| `dew.objectives` | `Objective`, `EMASpec` |
| `dew.objectives.diffusion` | `DiffusionObjective` |
| `dew.objectives.jepa` | `JepaObjective`, `MultiBlockMask`, `representation_health`, `get_linear_probe_metric`, `get_knn_probe_metric` |
| `dew.objectives.lm` | `LMObjective` |

## Diffusion

| Module | Contents |
| --- | --- |
| `dew.diffusion` | the schedules and prediction transforms below, re-exported |
| `dew.diffusion.schedules` | `LinearNoiseScheduler`, `CosineNoiseScheduler`, `ExpNoiseScheduler`, `CosineContinuousNoiseScheduler`, `CosineGeneralNoiseScheduler`, `SqrtContinuousNoiseScheduler`, `KarrasVENoiseScheduler`, `EDMNoiseScheduler`, `FlowMatchingScheduler` |
| `dew.diffusion.transforms` | `EpsilonPredictionTransform`, `DirectPredictionTransform`, `VPredictionTransform`, `FlowMatchPredictionTransform`, `KarrasPredictionTransform`, `get_diffusion_preset` |

## Models

| Module | Contents |
| --- | --- |
| `dew.registry` | `build_model`, `apply_precision_policy`, `canonicalize_architecture`, `map_config_strings`, `MODEL_REGISTRY` |
| `dew.nn.backbones` | `Unet`, `UNet3D`, `UViT`, `SimpleUDiT`, `SimpleDiT`, `SimpleMMDiT`, `HierarchicalMMDiT`, `HybridSSMAttentionDiT`, `VideoDiT` |
| `dew.nn.backbones.jepa` | `JepaEncoder`, `JepaVideoEncoder`, `JepaPredictor` |
| `dew.nn.backbones.causal_transformer` | `CausalTransformer` |
| `dew.nn.autoencoders` | `AutoEncoder`, `StableDiffusionVAE`, `SimpleAutoEncoder` |
| `dew.nn` | `attention`, `blocks`, `dit`, `vit`, `ssm`, `scan_orders` |

## Sampling and inference

| Module | Contents |
| --- | --- |
| `dew.sampling` | `DiffusionSampler`, `DDPMSampler`, `DDIMSampler`, `EulerSampler`, `EulerAncestralSampler`, `HeunSampler`, `RK4Sampler`, `MultiStepDPM` |
| `dew.sampling.pipelines` | `InferencePipeline`, `DiffusionInferencePipeline` |
| `dew.sampling.loading` | `load_from_checkpoint`, `load_from_wandb_run`, `load_from_wandb_registry`, `parse_config` |
| `dew.sampling.text` | `generate` |

## Inputs and data

| Module | Contents |
| --- | --- |
| `dew.inputs` | `DiffusionInputConfig`, `ConditionalInputConfig`, `ConditioningEncoder` |
| `dew.inputs.encoders` | `TextEncoder`, `CLIPTextEncoder`, `AudioEncoder`, `HFAudioEncoder` |
| `dew.inputs.processors` | `AutoTextTokenizer`, `AutoAudioProcessor`, `defaultTextEncodeModel` |
| `dew.data` | lazy re-exports of the loader factories, sources, augmenters and tokenizers, so `import dew.data` costs nothing; `load_data` and the name registries are not among them |
| `dew.data.dataloaders` | `load_data`, `get_dataset_grain`, `get_media_dataset_grain`, `get_token_dataset_grain`, `get_dataset_online`, `generate_collate_fn` |
| `dew.data.text` | `ByteTokenizer`, `HFTokenizer` |
| `dew.data.sources.text` | `TokenFileSource` |
| `dew.data.sources.hf` | `HFDatasetSource`, the `hf:<dataset>:<split>` route into the media loader |
| `dew.data.registry` | `datasetMap`, `onlineDatasetMap`, `mediaDatasetMap` |
| `dew.data.sources.base` | `DataSource`, `DataAugmenter`, `MediaDataset` |
| `dew.data.sources.images`, `.videos`, `.voxceleb2` | the TFDS, GCS, local video and VoxCeleb2 implementations |

## Evaluation and interop

| Module | Contents |
| --- | --- |
| `dew.eval` | `EvaluationMetric`, `get_fid_metric`, `get_clip_metric`, `get_clip_score_metric`, `get_psnr_metric`, `get_ssim_metric`, `get_perplexity_metric`, `psnr`, `ssim`, `frechet_distance` |
| `dew.interop` | `save_params`, `load_params`, `save_hf_layout`, `push_to_hub`, `pull_from_hub` |
| `dew.random_state` | `MarkovState`, `RandomMarkovState` |
| `dew.image_ops` | `clip_images`, `denormalize_images` |
