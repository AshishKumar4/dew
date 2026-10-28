# Changelog

Dew's releases, newest first. The version is `dew.__version__`, and a
release's tag is that version with a `v` in front (`v0.1.0`).

## Unreleased

- `NodePerturbation`, mixed into an objective, trains its model without backpropagation, by node
  perturbation of every Dense, DenseGeneral and Embed output as Dust does, the rest by weight-space
  evolution strategies, pooled over a split or accumulated update as its gradient is;
  `Objective.scored_units` states the units it credits.
- `Residuals` objectives train on the mean square of their residuals, by the gradient or by damped
  Gauss-Newton; `dew.nn.fourier.FourierPyramid` is a multi-resolution Fourier feature pyramid;
  `examples/pinn_burgers.py` solves the self-similar Burgers profile in float64.
- `Trainer.fit(resume=)` makes resuming explicit: `"auto"` continues the latest checkpoint where there
  is one, as before; `"require"` refuses a directory with none, for a job that must not start afresh;
  `"never"` refuses one that already holds a run (`Checkpoints.resumed`). A run configuration takes it
  as `trainer.resume` and refuses before writing its record over the directory's.
- `dew profile run.py --warmup 20 --steps 10` (`RunConfig.profile`) trains a run afresh in a directory of
  its own, traces a window of steps and prints where they spent their time (`ProfileReport`): step,
  input wait, compile, device time in collectives and the rest, analytic work and MFU.
  `Supervised(work=)` declares a batch's training FLOPs for a model Dew holds no count for.
- `Supervised(model, loss, inputs="x")` names the input field alone; the run reads its shape off its
  first batch as the model reads it, after the loader's device steps (`Objective.named_inputs`). An
  explicit `InputSpec` still states it.
- Under JAX's AutoPGLE (`JAX_ENABLE_PGLE=1`) the trainer guides its ahead-compiled step by a profile of
  its first runs, which JAX's jit never did for it, and records the profile in the step's plan.
- A token budget, a cadence of tokens and a schedule's stages read the committed-token count off the
  devices only where the steps since the last read could have reached what they read next
  (`Objective.row_tokens`), so the steps between dispatch without waiting on the one before.
- A schedule's stages each train at the row length their chain lays out, up to the model's
  `max_seq_len`, so a context-extension schedule (8k-token rows, then 64k) trains its later stage
  where the objective's one row length refused it.

- A tiled LM head's backward carries the state gradient over its vocabulary tiles as a compensated
  pair on every backend (`dew.nn.precision.paired_sum`), so its error is the tile products' own: on an
  H100 XLA took the plain sum's addition into each GEMM, which could then not split its sum, and the
  state gradient sat about twice as far from float64 as the whole head's.
- `divided`, IEEE division on every backend, moved from `dew.diffusion.schedules` to
  `dew.nn.precision`, beside the float32 algorithm it dispatches to; `dew.inputs.unit_range` and
  `dew.nn.mp.mp_silu` divide through it, where XLA took their division by a constant as a product
  with its rounded reciprocal.
- `dew.pipeline` is the one loader of a run published to the Hub: `TextToImage.from_pretrained` and the
  text tasks' `from_pretrained`, which pulled a run and loaded it with `from_run`, are gone; call
  `dew.pipeline(repo, revision=...)`, which pulls the run at the commit its metadata resolves and picks
  its task from the run's record. A saved run is pinned once: each task builds from the pinned run with
  `from_saved`, which `dew.pipeline`, `dew export` and a model soup (`dew.checkpoints.averaged_pinned`)
  call on the run they pinned, and `from_run` pins a directory and calls it. A plugin's saved task
  (`Objective.saved_task`) implements `from_saved`; deriving from `dew.inference.tasks.SavedRunTask`
  gives it `from_run` too.
- A server prefilling in pieces prefills a prefix block, or any group a layer reads whole, in one piece
  where the cut would split it, its members adjacent or not, where it refused a block longer than `chunk`.
  A served prompt carries its own image and block groups (`block_groups`, `image_groups`), slotted beside
  its tokens and kept in its row's cache, its own blocks standing over a prefix run's as `generate` keeps
  them; a prefix page is shared only between prompts that agree on their groups and their tokens up to
  the last grouped one. Disk banks read a
  family whose packed tensors split into views of themselves, fused experts among them.
- Every outer program Dew compiles goes through one seam (`dew.nn.kernels.options.program`): an image
  task's encode, noise and solver start, a bank's stacking, an autoencoder's encode and decode, LPIPS's
  and FID's networks and a host step's unstack, so two loads of one module share its programs. The seam
  holds placement trees by their structure and leaves. An objective's own programs and the serving
  kernel's donated step programs stay apart, as their docstrings say.
- The library holds no CPU-only algorithm (bible §5): a vocabulary row's fused logsumexp is one
  plain pass on every platform, which Dew's CPU test setting takes in float64 and rounds once
  (`dew.nn.precision.widened_reduction`, beside `accumulated`), in place of a CPU-only blocked
  reduction; a float64 schedule division is the plain division; and a plan over several devices is
  timed against the batch whole a rung up on every backend: it keeps the first plan unless the other's
  every sample beats it (`dew.training.planning.faster`), so samples that overlap never switch it (two
  runs timed apart may still keep different plans), and a resumed run continues its checkpoint's plan
  untimed. Dew's tests keep a frontier's first plan (`DEW_UNTIMED_FRONTIER`, a test setting like
  `DEW_WIDE_CPU_ARITHMETIC`).
- Gemma 4 Unified (`google/gemma-4-12B`, `model_type` `gemma4_unified`) loads as a native media wrapper:
  its encoder-free vision patches and raw audio frames project into the Gemma 4 decoder, either medium
  optional, and every attention kind reads each vision block whole. Its model FLOPs count its language
  model, as a vision projector's work is left out; a wrapper with an audio encoder refuses the count.
  A Gemma 4 audio clip's placeholders now read their valid frames by index, so a clip with masked frames
  between valid ones maps each placeholder to its own frame.
- `WeightLayout.source_array` assembles a source tensor in JAX at its compute dtype, traceable, as
  `export` lays it out; `export` keeps the host copy and the stored dtype.
- `record_case` takes no proxy: the recorded reference alone then sizes the count.
- `SafetensorsBanks` streams a media wrapper's language model and the native backbone of a decision
  release whose readout is stored apart (a slot release), besides a decoder: it reads the source through
  `dew.interop.pretrained.native_backbone`, `Pretrained.load`'s own translation and tree check over the
  safetensors alone. A wrapper's towers and projectors are read whole and stay resident. A config that
  states `expert_dtype` without a quantization config is no longer refused.
- `tools/slot_runtime_qualification.py` qualifies a released slot runtime against the authors' engine and
  its float64 truth, the truth's weights streamed from those banks: phases pool, native, environment,
  author, truth (`--partial` resumes a stopped phase) and verdict, one checkout's commit across them.
- A loaded decoder's pad row takes no gradient wherever its source's table pads it. Kimi Linear, Kimi K3,
  Ouro, Dream, LLaDA-MoE and LLaDA2 now pad `pad_token_id` as their released code does, at their config
  class's own default where a config states none; Dew trained that row before. Qwen 3.5's wrappers load
  a YaRN or Llama 3.1 rope record, their three-axis rotary turning by its ramp. Tier 2's probe runs past
  a rope ramp's original context and holds a convention's pad row to the source's table, a negative pad
  id counting back from the table's end. A verification of a repository's own code is recorded under
  the repository, Hub commit and folder it was fetched at, and a local directory's own code is verified
  at each load.
- A run trains its routers by one policy, `dew.objectives.routing.ExpertBalance`, which `LMObjective`, `MaskedDiffusionObjective`, `GRPOObjective` and `DPOObjective` take as `balance`, and `PPOObjective` takes as `balance` too and passes to its policy: the aux-loss-free balancing bias (`balance_rate`), DeepSeek V2's balance loss (`aux_loss_alpha`, `seq_aux`) and ST-MoE's router z-loss (`router_z_loss`), each optional. The policy adds its terms to the loss statistics and its bias counts to the step's effects, and the commit applies the counts once per update. `LMObjective`'s four balance arguments are replaced by `balance`; a run recorded with them is refused, as one recorded with any argument its objective no longer takes. A masked-diffusion run balances the routers of an LLaDA-MoE mixture as an LM does. A GRPO, DPO or PPO run counts the response tokens its loss trains, and a DPO pair's two halves both count. A record names `balance` as the `ExpertBalance` it is, so `--objective.balance` reads for each of these objectives. The decision objectives do not take the policy yet.
- Every token objective runs its vocabulary head by one policy (`dew.objectives.lm.head.HeadedObjective`):
  the masked- and block-diffusion objectives take `head_chunks` and `head_tile` as an LM's, keep the whole
  logits where they fit, and the fit ladder tiles them where they do not, where they scored a fixed
  1024 x 8192 tile before; GRPO's evaluation scores at its own tile. `head_cross_entropy` takes the tile it
  scores in, with no default.
- A prefix run trains and serves one mask: `LMObjective(prefix=N | "prompt")` makes each document's first
  tokens, or its prompt, a block the model's block-reading layers see whole, scores no target in its
  source's block, and gives a served prompt the same block (`TextSettings.prefix`, `TextGeneration`,
  `Server`). The data stage `dew.data.steps.blocks.Blocks` makes blocks for any chain from a prefix, a
  chat row's roles, or a user's per-token runs (`given=`), and a run with blocks refuses a model whose trunk
  reads none. A prediction depth reads blocks beyond its source, so a prefix run refuses `mtp_weight`, and
  blocks need a trunk layer that reads them with no attention chunk, which cuts a block at any origin. A
  prefix run's batch without blocks, a precompiled loader's, takes them at the scoring boundary, its
  documents' places counted over their real tokens across any hole in their validity, as the decoder
  counts them, in the data stage `Blocks` as in the loss; the stage a prefix run installs keeps a
  record's own blocks, as the loss keeps a batch's; `generate(..., prefix=)` refuses a model or a
  prefix the run would; lm-eval's likelihood (`dew.eval.harness`) scores a prefix run's continuation
  through the blocks a server makes of its context.
- MuonClip clips the attention of whatever objective a run trains, or refuses it: a run whose optimizer
  holds a QK-Clip (`dew.optim.clips_qk`, its state a `QKClipState`) asks its objective for the step's
  attention logit maxima (`Step.qk_stats`), so a masked- or block-diffusion run clips as an LM, RL or preference
  run does, and the run refuses a step whose objective returns none, where a decision run trained unclipped
  with no sign of it. `LMObjective`'s, `TokenScorer`'s and the RL objectives' `qk_stats`
  argument is removed: the run asks for them.
- Every run names its validation splits' metrics, `--val-metrics` (`RunConfig.val_metrics`), which only a
  diffusion run held: each metric scores the split it names (`MetricConfig.split`) or every split, so a
  decision run scores accuracy on `val` and the Brier score on `test`. Where it names none, each split
  scores the kind's own (`default_metrics`): a language model's perplexity, a decision run's accuracy,
  calibration error, risk-coverage area and log loss, a JEPA run's probes, an image run's CLIP.
  `DiffusionRunConfig.build_eval_metrics` is `validation_metrics(validates=...)`.
- A run weighs its splits' scores into named summaries, a board's weighted index (`--val-summaries`,
  `Trainer.fit(summaries=)`, `dew.training.evaluation.Summary`): each formed once a step, only where
  the step scored every split the summary weighs, logged as `evaluation/summary/<name>`, a namespace
  no split can take, and ranked by `Best` (`--trainer.best evaluation/summary/<name>`) and `Plateau`.
  A metric's name, like a split's, holds no '/'.
- A language-model run loops a `--pretrained` decoder as a decision run does: `--loop-steps`,
  `--loop-layers`, `--loop-backprop-steps` and `--loop-remat` are both runs' (`DecoderRunConfig`), and
  `looping` is `dew.config.looping`.
- Test-time training over the task loss: a decoder's continuum level (`CausalTransformer(continuum=ContinuumLevel(hidden))`) is stepped in context by the LM loss every `window` tokens of a document (`LMObjective(adaptation=ChunkAdaptation(...))`), Nested Learning's equation 71, each document from the trained start, training through every step.
- A label's cross-entropy derivative in its own logit is `p - 1` while the label holds at most half its row and minus the rest of the row past it (`dew.nn.categorical.label_entry`), for the whole-logits head and the tiled, fused and vocabulary-split heads alike; the rest alone sat two to three times as far from float64 on unconfident rows.
- `UNet2DCondition` stores its time embeddings' two maps flat, as `time_linear_1` and `time_linear_2`
  (and `additional_time_linear_1` and `_2`), where they were `time/in_proj` and `time/out_proj`; a
  published UNet loads as before. An AutoencoderKL's resnets are `dew.nn.blocks.ResidualBlock` called
  without a time embedding, and `dew.nn.autoencoders.vae.FlaxResnetBlock2D` is gone. `dew.nn.engram`
  draws its bucket primes with `sympy.nextprime`, so `sympy` is a dependency.
- A flow transition's density and KL hold at float32's edges: the density is JAX's isotropic
  `multivariate_normal.logpdf`, where `2 pi` times a variance near float32's largest overflowed, and the
  KL is one division of operands in range, the variance doubled where that fits and else the error
  halved, where 2 times such a variance or an error over a variance near float32's least overflowed and
  half an error near its least flushed to zero (a KL of 0 for 0.5); its derivative is each element's
  slope `(mean - reference) / variance` against the tangent, which neither overflows a tangent over a
  variance near float32's least nor flushes a cotangent over one near its largest.
  Schulman's k3 KL estimate is one helper through `expm1`
  (`dew.rl.surrogate.sampled_kl`), which a small log ratio's `exp(d) - 1` rounded away.
- A masked-diffusion row's reveals are its own under RBG keys too (`dew._distributions.counted`, the
  rest of ledger 1669), and TensorBoard histograms are its own `histogram_pb`'s, the plugin's tensor.
- The host CPU classifier is the CI's (tools/armada/host.py), not the library's: `dew._host` is gone.
- `OptimConfig` and `ScheduleSpec` move from `dew.config` to `dew.optim`, beside the optimizers and
  schedules they build; `Wandb` moves to `dew.training` (`dew.training.tracker`), beside the tracker that
  reads it; and
  `record_argument` moves from `dew.data.dataset` to `dew.registry`, beside the records it reads and
  writes. Import each from its new module.
- Every component record is the registry's class record, `{"class", "fields"}`, written by
  `dew.registry.to_record` and its class found by `dew.registry.component`. A component built around
  parameters states its record as `recorded_fields()` (`dew.registry.Recorded`): a condition encoder's
  `to_json` is now `recorded_fields`, and an autoencoder records its module and `record_arguments`, any
  subclass, no registration needed. `InputSpec.to_json`, `Condition.to_json` and `AutoEncoder.to_json`
  give way to `to_record(value, InputSpec)` and the like; `from_json` and `encoders.rebuild` to
  `InputSpec.bound`, `Condition.bound`, `AutoEncoder.bound` and `ConditionEncoder.bound`. A decision
  head records `{"class"}` for `{"name"}`; earlier runs' head and autoencoder records still load.
- The train state moves to `dew.state`, and a variables tree's operations (`Path`, `select`, `freeze`, `thaw`,
  `without`, `part`, `everything`, `under`, `PARAMETER_ROOTS`) from `dew.objectives.base` to `dew.batch`.
  `ModelConfig` moves from `dew.config` to `dew.registry`, the records' one owner; its wrappers are any
  `dew.registry.Wrapper`.
  `dew.coordination` becomes the private `dew._coordination`; a hook calls `dew.distributed` (`agreed`,
  `collective_host`, ...), whose `agreement_patience(seconds)` replaces `AGREEMENT_PATIENCE_SECONDS`.
  `dew.records` becomes the private `dew._records`. Import each name from its new module.
- Data plans are read by version 6 and cache generations by schema 2: a stream's
  shards are named by their files' places, which the same files mounted at
  another path keep, so a streamed cache fill serves there; and arrays are
  fingerprinted by their dtypes' names, so float8, int4 and other types of one
  byte layout no longer collide. A data position saved under plan version 5 is
  refused, and a generation or fill segment of schema 1 is refused as stale:
  fill the cache again.
- An earlier run's record is read through `dew.inference.tasks.earlier_shapes` at `SavedRun.pinned`, the one reader of earlier component shapes (a decision head or layout named by class name alone), carried to 1.0; `Head.from_record` reads an import path only, and `DecideSettings.of` is gone: a decision run's settings read as every task's do (`saved_settings`). An image run's input fields record through the registry.
- `dew.inference.tasks.SavedRun` is the one pinned context every run loader reads (a task's `from_run`, `Pretrained.from_run`, `Adapter.from_run`, `dew.pipeline`, `dew export`): `SavedRun.pinned(directory, step, trust)` replaces `run_record`, and `run.objective` and `run.variables(...)` replace each loader's own lookup and restore. `Pretrained.from_run` reads a run's task through its settings (`saved_settings`) instead of parsing their fields again. `Decide.from_run` and `DewLM.from_run` take `trust` as the other tasks do.
- A block- or masked-diffusion objective over a loaded source takes the source's
  end-of-sequence ids (`Pretrained.eos_token_ids`, or `eos_token_ids=`), and its
  task and saved run (`BlockSettings`, `MaskedSettings`) stop a response there,
  as the source's own task does; they generated past it.
- An LM, DPO, GRPO or PPO objective over a loaded source adds the source's
  end-of-sequence ids to every policy it builds whose sampling names none: its
  previews, task, saved run and the GRPO collector stop there.
- The batch contract moves to `dew.batch` and the mesh's axis names and rules to `dew.mesh`, so the
  data, models, objectives and trainer read them from one module below them all: `Batch`,
  `Variables`, `FROZEN`, `merge`, `VALID_ROWS`, `EXAMPLE_ROWS`, `Coordinates` and its token helpers
  (from `dew.objectives.base` and `dew.nn.inputs`) and `scored_targets` (from
  `dew.objectives.lm.objective`); the axis names, `DEFAULT_RULES`, `axis_rules`, `logical_spec`,
  `row_axes`, `batch_axes` and `LayoutRefused` (from `dew.nn.sharding`). Import them from their
  new modules; the old ones no longer hold them.
- A Hub split's shards are the files `datasets` resolves to it from the repository's listing, by its own
  default patterns (`dew.data.sources.hub.split_files`): `dev` and `valid` are `validation`'s, a sharded
  name's split is its own and a lone file is `train`'s, where a name match missed them. A listing of no
  file gives none. Every Hub call holds tqdm's bar lock first, so streamed sources discovered at once no
  longer delete it under each other's bars.
- An immutable upload's `retries` (`Cache.retry`) re-read only an object it created that reads back
  otherwise; a failed transfer raises after its transport's own retries, not Dew's on top of them.
- A decision run's calibration takes its probabilities through SciPy's `softmax` and fits each
  temperature with SciPy's bounded L-BFGS-B on its inverse, at least 1e-3, in place of a hand-written
  softmax and Newton loop; the grouping, ragged rows and soft targets are as before.
- An S3 store's conditional writes go through s3fs's own call (`call_s3`), as its reads and plain writes
  do, so a profile's server-side encryption (a KMS key, SSE-C) and its bucket's regional client reach
  them; a refused condition is still `PreconditionFailed`.
- On an Ampere GPU, the grouped GEMM's Mosaic launches run two programs a multiprocessor, eight warps as
  cuBLAS's kernels run, each in half its shared memory (`mosaic.Family.programs`, `Tiles.programs`): on an
  A100, 0.73 to 0.89 of one program's time (a product's geomean over 6 MoE and 4 dense rows), and the MoE
  forward runs the template by default, at 0.83 of Triton's time (CONSUMERS3).
- Dew's fused kernels, the delta rules' decay and memory products and the frozen FP8 product take
  their reverse mode through one `jax.experimental.hijax` linear map (`generation.linear_map`)
  instead of `linear_call`, which JAX deprecates in favour of hijax (jax-ml/jax#41782): their
  gradients still run the backward kernels, under a jit, a shard_map, a scan, a remat or a vmap.
  timed against the batch whole a rung up on every backend.
- 8-bit Adam state codes each moment value among the six codes around its estimate, its decade by its
  logarithm, instead of searching the whole map: on an A100, lm-dense's update takes 27.32 ms against
  60.39, its full-remat step 167.9 against 196.3 ms and compiles in 82.3 against 209.8 s, the codes
  the search's (`dew.optim.quantized`).
- `OptimConfig(state_dtype='int8')` stores adam's and adamw's moments blockwise in 8 bits
  (`dew.optim.int8_moments`, bitsandbytes' and torchao's 8-bit Adam): a quarter of float32's state, the
  update computed from the moments dequantized to fp32. On an A100, lm-dense's full-remat step peaks at
  5.88 GB against fp32 state's 8.09 and takes 196.3 ms against 151.7 (`tools/benchmark_kernels.py
  step --state-dtype int8`). bf16 and 8-bit Adam share one leafwise transform, and an 8-bit moment's
  codes and scales are placed by their parameters' axes.
- `Trainer.fit(examples=N)` and `--trainer.examples N` bound a run to the chain's
  first N records, counted before packing and each read once, its steps (and so
  its schedule) the ones the data plan counts for them. `Pause(fraction)`, as
  `stop=` or `--trainer.stop.fraction`, ends a run at that fraction of its
  budget, to resume as one that never paused. A training stream that runs out
  ends the run with its final checkpoint, on every process at the same step,
  instead of raising `StopIteration`.
- A JEPA run's target momentum ramps over its optimizer updates, the steps
  over its accumulation, where it ramped over its steps; its `--lora` adapts
  the encoder, which the run used to record and ignore. An LM run bounded by
  `--trainer.tokens` alone trains over data with no epoch length.
- `Plateau` counts an improvement as Flax's `EarlyStopping` does, which now runs its count: the best less the value must exceed `min_delta`. Its own test, the value below the best less `min_delta`, rounds that bound, so a best of 1 and a value of 1 - 2**-24 under a `min_delta` of 2**-24 - 2**-60 counted no improvement.
- A row drawn under its own key (text sampling's draw, masked diffusion's reveal) draws the same whatever rows it is drawn beside with an RBG key implementation (`rbg`, `unsafe_rbg`) too: `vmap` drew every row from the first key's stream, so each row key is made counter-based first, a key at a time. Threefry keys draw as before.
- A checkpoint ranking's best steps are Orbax's `BestN` over the steps it scored, and TensorBoard's text summaries are its own `text_pb`'s, each the same as before.
- A confident target's loss gradient is the rest of its row's probability summed directly, as arXiv:2610.31642's eq. 7 takes it, through every chunked head (the tiled, whole, fused and vocabulary-split heads) and the whole-logits loss (`logits_cross_entropy`). Each took the target's entry as its share less one, which rounds to 0 once the share rounds to 1, so the row's gradient stopped summing to zero by the rest mass, about 2e-9 at a margin of 20 and 4e-18 at 40. The forward pass reduces one more operand per row for it, and the loss values are unchanged.
- An autoencoder's latent normalization is one traced map, `dew.nn.autoencoders.LatentAffine`, which
  `AutoEncoder.encode` and `decode` apply through one hook, `latent_affine(params, dtype)`: a configured
  autoencoder's comes from its `LatentNormalization` (`affine`), a tuned REPA-E run's from its running
  statistics. `LatentNormalization.normalized`/`restored` and `TunedAutoEncoder.running_normalization` are
  gone: `latent_affine(...).normalized`/`restored` replace them.
- Two optimizers from Nested Learning (arXiv 2512.24695) as Optax transforms: `dew.optim.scale_by_m3`, M3's
  Muon over Adam's second moment with a slow momentum gathered once per chunk, and
  `dew.optim.delta_gradient_descent`, the step to eq. 56's minimizer given each kernel's input. M3
  orthogonalizes through optax's own Muon iteration (`optax.contrib.scale_by_muon` with no momentum).
- A filtered or packed read resumes on any number of readers once it is prepared: publish its rows (and
  a terminal pack's plan) as a generation with `dew data publish` and read it by index, and a new run
  lowers to the global order. A per-share read refused on another topology now says so.
- `dew.nn.attention.scaled_dot_product_attention(..., dropout_rate=..., dropout_rng=...)` drops attention
  weights on the selector's route (`generation.Route`): the reference path, or the attention template where
  it is named (`implementation='template'` or `generation.chosen(attention='template')`) and its plain-JAX
  mainloop runs the call: off a GPU, or on one before sm80. On an sm80 or later GPU the template's mainloops
  (the frozen Triton one and Mosaic's, whose weights hook comes next) take no dropout, so a dropping call
  takes the reference path there and a named template refuses it. The draw is the reference's mask, one hash
  of the call's seed and each weight's global batch row, query head, query and key
  (`dew.nn.kernels.templates.attention.Dropout`). A dropping call on a sharded sequence axis, or on an
  explicit fused kernel, is refused.
- A causal attention kind may read blocks whole: `AttentionMixer(bidirectional_blocks=True)` lets
  each token of a block (`block_groups=`, `AttentionMetadata.block_groups`, `[B, S]`, -1 outside
  any) see its whole block, whatever its window, as transformers' `block_sequence_ids` does, while
  `bidirectional_images` keeps Gemma's images, read whole within the window. A kind may read both,
  and every other kind stays causal. A cached call reads the groups of its own tokens, so no cache
  keeps them, and a layer that shares another's keys and values reads them too.
- `scaled_dot_product_attention(block_groups=, image_groups=)` reads blocks and images as the
  attention template's key spans, one interval a query, so an A100 runs a block or image layer on the
  template's fused kernels rather than xla's `[B, 1, S, S]` mask (268 MB a row a layer at 16k
  tokens), where it measured faster at every length; every other path takes the mask the spans mean. A window counts a call's places, padding
  inside a row skipped, in the spans too.
- `scaled_dot_product_attention(key_valid=, places=)` reads a call's real keys and the places its
  window counts as the template's spans, the invalid keys hidden as padding's document id hides
  them, so a row with padding inside it, a decoder's padded batch or a packed batch with holes, runs
  fused on a GPU rather than on xla's `[B, 1, S, S]` mask. Local attention's short windows and
  DeepSeek V4's window mask read the same spans.
- `dew.nn.attention.scaled_dot_product_attention(..., implementation='template')`
  runs the attention template's route
  (`dew.nn.kernels.templates.attention_route.attend`): causality, windows,
  documents, key lengths and starts, a softcap and sinks, differentiable in the
  query, key, value and sinks. It runs the Triton and Mosaic GPU mainloops on a
  GPU and the plain-JAX one elsewhere, per shard on a mesh. No target prefers it
  yet, so a call takes it only when it names it, or
  `generation.chosen(attention='template')` does.
- `dew.nn.moe.expert_projection` with `'pallas'` or `'auto'` runs its grouped
  matmuls through Dew's GEMM template and differentiates at every order:
  forward mode and higher-order derivatives no longer need `'xla'`. The old
  first-order-only path, `dew.nn.kernels.grouped_projection`, and
  `dew.nn.moe.grouped_matmul_kernel` are removed.
- The grouped GEMM's kernel gradient on Hopper streams its operand stages
  across a persistent tile loop, so the next tile's loads overlap this one's
  stores; it runs as fast as CUTLASS's grouped kernel gradient at the MoE
  shapes measured. A fused RMS norm's Mosaic GPU forward now lowers on Hopper.
- A causal attention kind may read blocks whole: `AttentionMixer(bidirectional_blocks=True)` lets
  each token of a block (`block_groups=`, `AttentionMetadata.block_groups`, `[B, S]`, -1 outside
  any) see its whole block, whatever its window, as transformers' `block_sequence_ids` does, while
  `bidirectional_images` keeps Gemma's images, read whole within the window. A kind may read both,
  and every other kind stays causal. A cached call reads the groups of its own tokens, so no cache
  keeps them, and a layer that shares another's keys and values reads them too.
- `dew.nn.categorical.cross_entropy` and `kl_divergence` differentiate by Barron,
  Gnanapragasam and Tschannen's fix (arXiv:2610.31642): a label whose margin
  rounds Optax's `p - 1` to 0 keeps its derivative, each row's derivative sums to
  zero, and a divergence from logits equal to its reference has derivative
  exactly zero. Their values are unchanged.
- A joint decision row's user turn is composed of segments, `JointLayout.segments`, a tuple of
  `Text`, `Media`, `State` and `Schema` in the order the row writes them: the default is the Laya
  joint release's turn, and another writes the schema before the state, or text of its own
  between the parts. A run records its layout and head by import path through `dew.registry`,
  so a layout or head of a package the run trusts reloads; a record of an earlier run, which
  names its class alone, reloads as before. `dew.decision.HEADS`, the closed table of heads a
  run could record, is removed.
- A joint row may read its schema as its context, `Schema(stream="context")`
  (`DecisionRunConfig.segments`): the backbone reads the schema in a pass of its own, with its own
  weights, its cross-attention layers read that pass beside the row, and the joint head pools the
  questions and options from it; the pass counts in `train/flops`.
- On a CPU, every float32 product a Dew model forms (the dot family, a bias's
  gradient, attention's reference path, which a CPU's attention now takes in
  every dtype) sums in float64 and rounds once (`dew.nn.precision.accumulated`),
  rather than XLA:CPU's float32 sum, whose order its shape and release choose.
  A CPU forward costs about 1.6 times, a training step 1.4 to 1.9 times;
  GPU and TPU programs are unchanged, and convolutions keep their float32 sum.
  `jax.default_matmul_precision("F32_F32_F32")` takes the float32 sum back.
- A run's W&B display name is its own, `Wandb.name` (`--trainer.wandb.name`), apart from the
  name its checkpoints are kept under; it shows the run's name where none is given.
- `Checkpoints.committed()` lists a run's committed steps at each call without opening or writing
  its store, so a process beside the one that trains (an evaluation worker) reads each step as it
  commits; the checkpoints guide shows the worker's loop.
- `dew.sampling.sample`'s `record` reads each step's clean prediction too,
  `record(x, state, denoised)`; a recorder written for two arguments takes a
  third.
- JEPA's `normalize_targets` is Flax's `LayerNorm` with no affine and exact
  variance. Its statistics are at least float32, as I-JEPA's `F.layer_norm`'s
  are, so a bfloat16 run's targets round once, and its epsilon is I-JEPA's,
  PyTorch's default 1e-5, where it was 1e-6.
- `Decide.systemone` takes a list of request bodies and answers each as it
  would alone, their rows scored together, a refused body's `ValueError` in its
  place. `dew.inference.ScoringServer` serves a task answered by one forward
  pass a batch (`Decide.scoring`), and `dew.inference.Engine` serves it and
  generation's `Server` from one thread; `examples/serve_decisions.py` answers
  concurrent requests in batches (`--batch-tokens`, `--batch-rows`) through it.
- `OptimConfig.learning_rate` is the learning rate, a number or a schedule's
  record, in place of the former `learning_rate` and `schedule` fields; on the
  command line it is one value, `--optim.learning-rate 3e-4` or a schedule's
  JSON record. A `ParamGroup`'s scale and shape of it is its `learning_rate`.
  The schedules name their rates as optax does: `peak_value`, `init_value` and
  `end_value` in place of `peak`, `init` and `end` (`Cosine`, `Power` and its
  `PowerTail`, `Linear`, `OneCycle`, `Exponential`), so a run record or a flag
  that names the old fields no longer reads.
- Module moves, each name at one path: the run configurations are
  `dew.config.lm.LMRunConfig`, `dew.config.jepa.JepaRunConfig` and
  `dew.config.diffusion.DiffusionRunConfig` (with `FlowRolloutConfig`); the
  condition records `TextCondition` and `AudioCondition` are `dew.inputs`';
  `PretrainedAutoencoder` is `dew.nn.autoencoders.pretrained`'s; the optimizers
  and schedules are `dew.optim`; the device facts and CUDA builds are
  `dew.devices` and `dew.builds`. A run's record names its run class by its
  alias (`"class": "diffusion"`, `dew.registry.runs`), so it reads back wherever
  the class moves; a run.json written before this names the old import path and
  no longer loads. A distillation's teacher run is the run config's
  `--teacher-run`, and the objectives take the loaded teacher
  (`teacher_variables=`, and guidance distillation's `teacher=`).
- A diffusion schedule's division of a state by a rate (`divided`: epsilon from a clean
  prediction, the clean sample from epsilon, a sigma solver's derivative) is IEEE's correctly
  rounded quotient whatever the backend's division flags, through the one correctly rounded
  divide the quantizers use (`dew.nn.precision.correctly_rounded_divide`, moved from
  `dew.nn.fake_quant`); a bfloat16 or float16 state's is the float32 quotient rounded to its
  dtype, as PyTorch's opmath gives it.
- `train/flops`, a step's analytical training work, is logged on every device,
  CPU and TPU included; `train/mfu` still needs a device with a known peak.
- `dew.figures` draws SVG figures for notebooks and docs. Their colors follow
  the page's light or dark theme (dewml.dev, Colab, JupyterLab, VS Code, and
  the system setting elsewhere), and their animations are CSS, so they play
  without script. `curve`, `bars`, `heatmap` and `tokens` draw charts from
  arrays; the tutorials use them for the figures their runs make.
- `train/mfu` now measures analytical model FLOPs over real row/segment lengths,
  rather than HLO model FLOPs. Old and new MFU values are not directly comparable;
  the old metric is `train/hlo_flops_util`, and hardware/remat work stays `train/hfu`.
  `train/tokens` counts real tokens per attempted microbatch and `train/tokens_per_s`
  reports their interval rate, independently of gradient accumulation.
- A `BlockDenoiser`'s call and `encode` take `positions`: block generation
  refines and commits each canvas at the logical positions that continue its
  prompt's, as text generation continues them.
- Data is one API, the chain: a store piped through the presets and steps
  (`dew.data.presets`), compiled by `dew.data.compile.compile`, and recorded in
  run.json as written (`dew.data.records`). `dew.config.data` (its recipes,
  `loader` and `Augment`) and `dew.data.load` are removed; a run config's
  `data=` is a chain, and `--set data.source.url=...` points it at another
  store. docs/design/data.md lists what each recipe chose and the chain that
  states it. A token store is a committed export (`dew data export
  --tokenizer`), one split each; a `train.bin`/`meta.json` directory is
  exported once. Every preset takes `batch=None`, and `prompts`,
  `preferences` and `canvas` take `cache=`.
- A committed export is named by what it holds, not where its writer put its
  shards, so hosts that each export the same rows read one store. A data
  position saved over a committed export before is refused as its source
  changed, and a cache or generation over one fills again.
- A holdout written before a published generation picks the inputs it picks
  without the cache: the generation stands in for the inputs each part keeps,
  by hash (by a field the cached step drops too) or by records, over missing
  rows and steps that drop or expand them. A records holdout over a
  generation a stream filled is served where the store states that its stream
  reads its records in index order (in-memory, token, TFDS, PyTorch, Grain
  and Arrow datasets, generations), and refused otherwise: publish those from
  a memory read, or an indexed one (`provision=True`). A step that reads its
  record's index there is refused, naming `Cache(url)`. A holdout part is named by
  what it selects of its source's identity, so selections stacked on one
  another stay apart; a data position saved over a holdout part before is
  refused as its source changed, and a generation published from one no
  longer stands in for it, where a `Cache(url)` over it fills again.
- A mixture reads its corpora by Grain's weighted selection and counts its
  epoch by it: until every corpus has come round once where the order reads
  on past the mix, and at its first corpus's end where a mix of corpora read
  once ends. Corpora that recur across phases continue where that selection
  left each, which reads other records after a phase switch for weights whose
  scaling Grain truncates (1, 10 and 15), so a data position saved by an
  earlier Dew is refused (data plan version 5).
- An epoch's steps count the step that reads its last records short of a
  batch where the order reads on into the next epoch (a training order that
  repeats for ever) or keeps its short last batch (`Loader.tail`): 39 windows
  in batches of 8 are 5 steps an epoch, not 4, so a run counted in epochs
  reads every record of each. An order that ends and drops its short last
  batch counts as before. A part of a mixture cut by a slice or a limit comes
  round once its cut records are read, and a packing's or token windows'
  facts keep none of the counts of the documents in them.
- `dew.data` and `dew.data.sources` export `from_torch`, `from_grain`,
  `from_hf` and `from_iter`, not the `Torch`, `Grain`, `HF` and `Iterated`
  classes they make, which a recorded chain still names by their modules. A
  Hugging Face stream saves its position through the iterator bridge every
  resumable iterator does, and a data position saved over one before is
  refused.
- A storage profile whose base is this host's disk (`file` or `local`) is
  that disk: an export commits to it, and ArrayRecord shards, offset indexes
  and Arrow files read there by their own readers.
- A text file is indexed by its lines' bytes, so a line that is not UTF-8 is
  refused when it is read, not when the file is provisioned.
- A packing plan a generation publishes is named by the digest a plan held
  in memory is (`dew.data.plan.fingerprint`). A generation published before
  keeps the name it stored, and a data position saved over it resumes; one
  published again from the same documents is named anew, and a position saved
  over the one before refuses it as its order changed.

## 0.1.0

The first release on PyPI, as `dewml`: `pip install dewml` for the CPU, with
the `cuda12`, `cuda13` and `tpu` extras for accelerators, on Python 3.12 or
newer.

- Training: one `Trainer` for every objective, on one device or a mesh
  (data, FSDP, tensor, sequence, expert and pipeline axes), with data and
  FSDP across hosts, gradient accumulation, EMA, checkpoints that resume the
  data position, time- or step-based checkpointing, validation over named
  splits each with its own metrics and cadence, LoRA, and quantized
  training.
- Diffusion: DiT, MMDiT, UNet and video backbones; flow matching, EDM and
  other processes; classifier-free guidance, CFG++, APG and autoguidance;
  few-step distillation (rCM, LADD, MeanFlow, shortcut models, guidance
  distillation); REPA and REPA-E.
- Language models: next-token training, masked and block diffusion language
  models, packed sequences, multi-token prediction, and Hugging Face decoders
  loaded into Dew's own modules.
- Post-training: GRPO, PPO, DPO, Flow-GRPO for diffusion models, and
  multi-turn tool episodes.
- Representation learning: image and video JEPA with linear and kNN probes.
- Decision models: classification heads on pretrained backbones with
  calibration and serving.
- Data: `dew.data.load` for Hugging Face text and image datasets and TFDS
  builders, with Grain and PyTorch datasets plugged in directly.
- Inference and serving: `dew.pipeline`, which loads a run, a source
  checkpoint or a Hub repository holding either as its task; samplers and
  solvers; a continuous-batching text server over a paged cache; and clients
  for Ollama and OpenAI-compatible servers.
- Interop: safetensors, GGUF and single-file diffusion checkpoints, export
  back to the source layout, and publishing to the Hugging Face Hub.
- The `dew` command trains a run from a Python file or its `run.json`,
  tokenizes a text corpus, exports a run to its family's published layout,
  starts a program on every accelerator of a cluster as one process pool
  (`dew launch`), and creates and reaches Cloud TPUs (`dew tpu`).
