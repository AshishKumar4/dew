# Review of `main` at `102baa4`

Review, 2026-09-03. Read first hand: `training/`, `objectives/`, `diffusion/`, `sampling/`, `inputs/`, `registry.py`, `config/`, the causal transformer and attention, the chunked cross entropy, the DiT head, the data loaders and token sources, the eval metrics, the JEPA mask and probes, the examples, the LM recipe, and the trainer and sampler tests. Four read-only audits covered the data internals, the backbones, the periphery (interop, telemetry, eval, CLI, RL) and the test suite; every finding cited from them was re-read at the named lines. Every finding below was then executed on CPU in the project venv (jax 0.11.1, flax 0.12.9, Apple M3) before it was called confirmed. The companion design is `docs/design/api.md`.

## 1. Verdict

The mechanics are better than the API. The compiled step, the abstract-state sharded materialisation, cross-mesh restore, the checkpointable device prefetch, the attention kernel seam and KV cache, the chunked cross entropy, `generate` and the HF decoder translation are correct where checked. The suite (44 files) asserts values against references far more than most research code does: committed parity fixtures with their generators under `tools/`, mutation tests, and real `jax.distributed` process tests.

What is wrong is ownership. The trainer is a diffusion trainer with an escape hatch, the objective is handed the W&B run, data is an anonymous dict, randomness is a class, presets are tuples, registries are strings and configs are dicts that drop keys. The design document names each crossing with its line and the surface that replaces it.

## 2. Findings

Status: `fixed` names the commit, `held` means the fix changes numerics or belongs to the design waves, `withdrawn` means the reproduction disproved the claim. Line numbers are at `102baa4`.

| # | Where | Finding | Reproduction | Status |
| --- | --- | --- | --- | --- |
| 1 | `sampling/ddpm.py:23-29` | The noise coefficient equals the posterior std only under `alpha^2 + sigma^2 = 1`; on VE it is 0 | `gamma = 0` where the posterior std is 0.89; samples collapse to std 0.001 against data std 0.3 | fixed `911f5ec` |
| 2 | `sampling/loading.py:132` vs `recipes/diffusion/train.py:219` | Inference rebuilds the preset without `flow_shift` | config `flow_shift=3.0` gives a schedule with `shift=1.0` | fixed `a65d447` |
| 3 | `eval/psnr.py:59`, `eval/ssim.py:105` | Samples in `[-1, 1]` scored against uint8 batches | identical image: PSNR -37.2 dB, SSIM 0.000 | fixed `362b75f` |
| 4 | `data/dataloaders.py:904` | Missing `val.bin` validates on `train.bin` | `val_len == train_len`; first val row equals first train window | fixed `7c2ddd6` |
| 5 | `data/dataloaders.py:174,273,492,600,682,810` | `batch // process_count` truncates; `global_batch_size` still reports the request | 8 processes: batch 65 gives local 8 and reports 65; batch 7 gives local 0 | fixed `f3ea98b` |
| 6 | `data/dataloaders.py:499` | `dataset_length = 1000000` guessed for a source without `__len__` | code read | fixed `b52c6ec` |
| 7 | `schedules/discrete.py:68-70` | `get_posterior_variance` calls `int(steps)` and returns the std; no callers | array input raises TypeError; scalar returns `sqrt(var)` | fixed `a6c4e46` (deleted) |
| 8 | `sampling/multistep_dpm.py:26-60` | A VE-only integrator (`dx/dsigma = eps`) with no schedule guard | on a VP oracle the std is 0.266 against 0.3, an 11% bias inside the test tolerance | fixed `e25bba9` (guard) |
| 9 | `nn/blocks.py:69-83`, `schedules/flow.py:52-55` | The scale-16 Fourier time embedding decorrelates adjacent timesteps in the discrete and flow domains | cos(t=500, t=501) = -0.05; EDM's own adjacent grid points also give 0.089, which is how EDM's random Fourier features behave | withdrawn as a bug; the flow.py comment is backwards and SimpleDiT's embedder is EDM's, not DiT's (T9) |
| 10 | `schedules/discrete.py:16-17,47-52` | `p2_loss_weight_gamma=1` makes the cosine/v preset an x0 loss | `w(t) * (SNR + 1) = 1.0` at t = 10, 300, 600, 900 | held, documentation (T10) |
| 11 | `sampling/common.py:251-252`, `objectives/diffusion/objective.py:115-127` | Validation samples from a fixed seed every epoch | two ancestral runs bit-identical | fixed `5ab6400` |
| 12 | `objectives/diffusion/objective.py:100-110`, `trainer.py:611-622` | Validation hardcodes DDIM, guidance 3.0, 200 steps; a full sampler pass runs before step 0 | code read | held, design wave 2 (T12) |
| 13 | `objective_trainer.py:338-342,382-385`, `trainer.py:431-432` | Validation swallows every exception | a ZeroDivisionError in a metric prints and the run continues | fixed `6b747dc` |
| 14 | `sampling/loading.py:54` | `warnings.filterwarnings("ignore")` process-wide | `filters[0] == ('ignore', None)`; a later warning recorded 0 times | fixed `2cdd7ae` |
| 15 | `schedules/common.py:17-23` and every subclass | Constructors swallow unknown keywords | `sigma_mn`, `P_meen`, `beta_stat` accepted | fixed `7cc6bb3` |
| 16 | `registry.py:176-180` | `build_model` drops unknown keys with a print | `num_layerss`, `emb_feature` dropped | fixed `8b34869` |
| 17 | `nn/dit.py:174-185` | `learn_sigma` doubles the head and discards the variance half | 3 channels from a 96-wide head; sigma-half gradient exactly 0 | fixed `6ce8bef` (removed) |
| 18 | `nn/moe.py:51,142-149` | No MoE balancing runs: `calculate_load_balance_updates` has no caller, `expert_bias` is never written | grep | held, design wave 2 (T18) |
| 19 | `nn/text_encoders.py:341-379` via `inputs/__init__.py:131-154` | Frozen CLIP weights enter the train step as jit constants | the jaxpr carries 24,032 constant elements, the encoder's weight count | held, design wave 3 (T19) |
| 20 | `trainer.py:55-62` | `SimpleTrainer` is a `@dataclass` with a hand-written `__init__` | `__eq__` returns a meaningless False; dead `ema_decay` field | fixed `9e27b67` |
| 21 | `trainer.py:42-50` | `Metrics` is created empty, never updated, checkpointed | `state/metrics/{accuracy,loss}/{count,total}` written at step 1 | held, design wave 2 (T20) |
| 22 | `schedules/common.py:104-106` | `1 + sigma^2/sigma_max^2`, uncited, live for `CosineGeneralNoiseScheduler` | weights `[1.0, 1.0002, 1.0014]` against EDM lambda `[9.66, 1.99, 1.11]`; introduced in `2ccfbc3` beside the correct override | held, numerics (T21) |
| 23 | data package | Dead code: `CaptionDeletionTransform`, four unused parameters of `get_dataset_online`, `DataSource.create`, `sources/utils.py`, `import augmax`, `get_oxford_valset`, an unread `"filter"` key, duplicated exports | grep | fixed `e95189a` |
| 24 | `nn/dit.py:139` | Text conditioning mean-pools the padded CLIP rows | 70 padded rows move the vector by up to 2.33 | held, numerics (T23) |
| 25 | `objective_trainer.py:250-288` | A rejected dynamic-scale step advances `step` and applies the EMA | params unchanged, step 1 to 2, EMA 0.9 to 0.85 | fixed `d35ee68` |
| 26 | `nn/autoencoders/api.py:27,42` | Invented dunders `__encode__`, `__decode__` | code read | held (T25) |
| 27 | `trainer.py:482-540` | The first log tick includes the compile | code read; test in `5bc43e7` | fixed `5bc43e7` |
| 28 | `data/dataloaders.py:823-825` | Suspected: `shuffle(seed).repeat(n)` replays one permutation | grain reshuffles per epoch: `[0, 4, 6, ...]` then `[5, 7, 6, ...]` | withdrawn; the code and its docstring are right |
| 29 | `eval/inception.py:48` | Pickled weights reference `numpy.core.numeric`; the repo's warnings-as-errors rule fails the network FID test | reported by the metrics fix agent at base | open (T28) |
| 30 | `recipes/diffusion/train.py:35` | A second process-wide `warnings.filterwarnings("ignore")` | code read | open (T29) |
| 31 | `sampling/loading.py` `load_from_checkpoint` | Restores without sharding args; orbax warns in four inference tests at base | reported by the loading fix agent | open (T30) |

Verified correct and left alone: EDM `c_in`, `c_out`, `c_skip`, `c_noise` and `lambda(sigma)` (`transforms.py:112-140`, `karras.py:19-25`); the v-prediction algebra and every `target_error_scale`; the Euler step reducing to DDIM on VP and to Karras Euler on VE; DDIM `eta > 0` against eq. 16; Heun's `sigma = 0` fallback; flow matching rates, shift and velocity; rotate-half RoPE with packed positions honoured by the mask and by RoPE; `open_kv_cache` first-call semantics (`attention.py:72-92`); sliding-window agreement between the mask and the fused kernel; tied `head_weight`; the chunked cross entropy; JEPA mask disjointness; the MoE expert paths against the axis table; the packed token pipeline's chunk arithmetic.

## 3. The fix wave

Seven branches from `102baa4`, one owner each, disjoint files, the failing test committed before the fix, merged without conflicts as `3d30a35` through `b39c62a`. Every branch's covering test files were run green in its worktree before the merge, and the whole CPU suite ran once on `main` after it (section 4).

| Branch | Commits | Covering files |
| --- | --- | --- |
| `fix/psnr-ssim-range` | `362b75f` | `tests/test_metrics.py` |
| `fix/learn-sigma` | `6ce8bef` | `tests/test_models.py`, `tests/test_architectures.py` |
| `fix/schedules` | `7cc6bb3`, `a6c4e46` | `tests/test_schedulers.py` plus predictors, flow matching, samplers |
| `fix/samplers` | `911f5ec`, `e25bba9`, `5ab6400`, `b21ef60` | `tests/test_samplers.py`, `tests/test_objectives.py` |
| `fix/loading-registry` | `a65d447`, `2cdd7ae`, `8b34869` | `tests/test_config.py`, `tests/test_inference.py`, `tests/test_precision_policy.py`, `tests/test_architectures.py` |
| `fix/data` | `7c2ddd6`, `f3ea98b`, `b52c6ec`, `e95189a` | `tests/test_data.py`, `tests/test_text_data.py` |
| `fix/trainer` | `6b747dc`, `d35ee68`, `5bc43e7`, `9e27b67` | `tests/test_trainer.py`, `tests/test_objectives.py`, `tests/test_parallelism.py` |

Two things the wave did on purpose that are worth knowing. `b21ef60` removed the `SimpleDDPMSampler` and `generate_images` aliases; nothing outside the tests used them. `8b34869` makes an old logged config with a since-removed key fail to rebuild; per decision 7 in the design there is no migration for that.

## 4. Integration run

`PYTHONPATH=src JAX_PLATFORMS=cpu .venv/bin/python -m pytest -m "not network" -q -p no:cacheprovider -x` on `main` after the merges, Apple M3, CPU: 230 passed, 1 skipped (`test_data_real.py`, needs the tfds extra), then one failure, `tests/test_config_cli.py::test_grad_accum_steps_wraps_the_optimizer_and_reaches_the_trainer`, from `prepare_process` using the Linux-only `resource.RLIMIT_OFILE` name (`training/runtime.py:43`), a macOS portability bug older than this wave that CI on Linux never sees. Fixed in `f0e623a`; `tests/test_config_cli.py` then passes (16 passed). The full run after that fix was started and stopped before it finished; the whole suite still has to run once on a Linux box, which is the first step in section 6.

The reproduction scripts behind section 2 are under `tools/review/*_2026_09_03.py`. Run one with `PYTHONPATH=src JAX_PLATFORMS=cpu python tools/review/<name>.py` from the repository root; each prints a `CONFIRMED` or `NOT REPRODUCED` line per finding. `diffusion_2026_09_03.py` and `trainer_2026_09_03.py` import helpers from `tests/`. They are the evidence for the held decisions (T9, T10, T19, T21, T23) and are not tests.

## 5. Not done

No training run was made; the numerics-changing items (T21, T23) and the time embedding note (T9) wait for the lead's decision. `nn/ssm.py`, `scan_orders.py`, `uvit.py`, `unet*.py`, `video_dit.py`, the middle of `mmdit.py`, `eval/inception.py`, `online_loader.py` and `cli/tpu.py` were read by the audits, not first hand. The seam crossings in section 2 of the design are confirmed and are the design waves, not this wave.

## 6. How to continue

1. Recreate the environment: `uv venv .venv --python 3.12 && uv pip install --python .venv/bin/python -e '.[test,metrics,interop]'`. Then run the whole suite once: `JAX_PLATFORMS=cpu .venv/bin/python -m pytest -m "not network" -q -p no:cacheprovider`. Nothing in section 3 was merged without its covering files green, but the union was not run to completion.
2. Decide the held items: T9, T10, T21, T23 in the design's ticket table. Each changes either documentation or the numerics of an existing preset; the reproduction scripts print the numbers the decision rests on.
3. Start wave 1 of `docs/design/api.md` (the `Registry`); it is mechanical and touches no numerics.
4. Delete the seven `fix/*` branches after the suite is green on Linux; their commits are in `main` through the merge commits `3d30a35` to `b39c62a`.

