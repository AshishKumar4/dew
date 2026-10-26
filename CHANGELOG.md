# Changelog

Dew's releases, newest first. The version is `dew.__version__`, and a
release's tag is that version with a `v` in front (`v0.1.0`).

## Unreleased

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
  orthogonalizes through `dew.optim.newton_schulz`, Muon's quintic iteration on each matrix alone.
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
