# Dew's architecture

Design record. Besides explaining how Dew works, this page owns the cross-cutting code boundaries, in its [shared contracts](#shared-contracts). [Training](training.md), [data](data.md) and [post-training](post-training.md) own their topic contracts, [algebraic optimization](algebra.md) and [kernel templates](kernel-templates.md) the two engines described here, and [testing](testing.md) behavioral qualification. The [roadmap](../roadmap.md) lists work still to complete or qualify.

This page follows one training run through Dew: the model and data you write, the step Dew builds from it, the point where that step is compiled, the rewrite Dew applies to its math, the kernel each operation runs, and how those kernels become machine code and run. Most of it is about the two parts Dew built itself: the algebra engine, which rewrites a program's math into a cheaper equal form, and the template engine, which generates GPU kernels from the device's own facts.

This page describes Dew as main has it. Two of its parts are built on base/kernels, Dew's kernel integration branch, and reach main when it promotes: the compile-time algebra pass, and the derived template forms with the cost model they share with it. Dependent launches come with them. Their sections here say what each does and why; the specs, [algebraic optimization](algebra.md) and [kernel templates](kernel-templates.md), hold their design, and base/kernels' own copy of this page describes their code. Where the code and the specs disagree, this page follows the code and says so.

![One training run, top to bottom. Teal marks Dew's templates, this page's subject; the trainer and the run config are Dew's too. The algebra pass, in amber, is built on base/kernels and is not on main yet.](https://raw.githubusercontent.com/AshishKumar4/dew/main/docs/assets/architecture-flow.svg)

*Figure 1. One training run, top to bottom. Teal marks Dew's templates, this page's subject; the trainer and the run config are Dew's too. The algebra pass, in amber, is built on base/kernels and is not on main yet.*

## The user's view

You write a plain Flax model, give it data, and train it from a Python script or a notebook. Here is a whole run: a one-layer model fitting a line.

```python
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn

from dew import Trainer
from dew.data import Memory
from dew.objectives.base import Aux, Objective

x = np.linspace(-1, 1, 32, dtype=np.float32).reshape(32, 1)
y = 2 * x + 1
data = Memory({"x": x, "y": y}).shuffle(seed=0).repeat().batch(32)
```

```python
class Regression(Objective):
    def __init__(self, model):
        self.model = model

    def init(self, key, variables=None):
        sample = jnp.zeros((1, 1), dtype=jnp.float32)
        return self.model.init(key, sample)

    def loss(self, variables, batch, step):
        prediction = self.model.apply(variables, batch["x"])
        errors = (prediction - batch["y"]) ** 2
        loss = self.row_mean(errors, batch)
        mse, _ = loss.mean()
        return loss, Aux(metrics={"mse": mse})


model = nn.Dense(features=1)
objective = Regression(model)
```

```python
trainer = Trainer(objective, optax.sgd(learning_rate=0.1),
                  key=jax.random.key(0))
state = trainer.fit(data, steps=100, log_every=50)
```

A model is any Flax module. An objective connects it to the trainer: `init` makes the variables and `loss` returns the quantity to minimize. Dew ships objectives for the common jobs, such as `LMObjective` (with its SFT path on chat data), `DiffusionObjective`, `DPOObjective` and `GRPOObjective`, so most runs write none. The data is any Dew `Dataset`: in memory as here, or files, Hugging Face or TFDS, read through Grain. `Trainer` takes the objective, an Optax optimizer and a key; `fit` places the variables on the device mesh, compiles one training step and runs it. Inside, one step is one `Transaction`: `jax.vjp` of the objective's loss, then the optimizer's update and the EMA. The step is compiled the first time `fit` sees each batch shape, and every step after that calls the executable directly, once per micro-batch.

The same run can come from a config. A `RunConfig` records the model by its import path and fields, the objective, the optimizer and the trainer's settings, in a `run.json` or a Python file whose `run()` returns it. `dew train run.json` runs it, and `--set path=value` changes any field. The command holds the XLA flags the run needs, re-executing the process if they change; `RunConfig.run` turns on the persistent compilation cache, builds the model and the objective around it, and constructs the same `Trainer`. Both paths then meet in `Trainer.fit`, and from there the step is compiled at the same point.

Which kernels run depends on the model, not on the path. Dew picks its templates inside its own layers: `dew.nn.Dense` and `DenseGeneral` send their product to the GEMM template (`nn/dense.py`), and Dew's attention, mixer and recurrence modules send theirs to the attention and recurrence templates. Nothing intercepts Flax's own ops, so a plain `flax.linen.Dense`, like the quickstart's, runs through XLA. A plain Flax model with any data gets the trainer and XLA; a model built from `dew.nn` layers gets the template kernels too.

Nothing in the user's code refers to kernels or the algebra engine. A model built from Dew's layers asks for attention, a projection or a norm through them, and which kernel runs, and in what algebraic form, is decided below that line. The few user-facing options that reach it are a run's precision (`trainer.precision` in its config: fast or accurate numerics), `attention_impl` (default `auto`) and the opt-in kernel tuner, `tune_kernels`, which is off by default.

*Where it lives: `dew.Trainer` (`fit`, `_training_loop`, `_compiled_for`, `compile`), `dew.objectives`, `dew.data`, `dew.training.transaction`; the config path in `dew.config.RunConfig` (`run`, `build_run`, `train`) and `dew.cli.main`. The [quickstart](../getting-started.md) walks through a first run step by step.*

## The compile point

Dew compiles its programs at one seam. Every program is an `OuterProgram`, whose `_jitted` jits its function. The resident training step is one: `Trainer._jitted` returns `owned_program(step, in_shardings=…, out_shardings=…, donate_argnums=0)`. So are a host-master step's phases, an evaluation, a load, a sampler, a serving step and an RL rollout. A lint rule (SLOP025) keeps it that way: the one other jit in the library, `dew.nn.nested`, only marks a sub-program inside an enclosing trace and compiles nothing at the top level. On base/kernels, the algebra pass sits on this seam ([below](#the-algebra-pass-on-basekernels)).

- `options.program(function, *models, step=None, **jit_arguments)`: The process-wide factory: one `OuterProgram` per function, compiler options and jit arguments, cached, so every call with the same inputs reuses it. Its compiler options come from the models it runs (`compiler_options`) and, for part of a training step, the step's own.
- `owned_program(...)`: The same program, kept by no cache of Dew's. A closure or bound method that a run makes, such as the training step, its placement's programs or its host phases, takes this form, because a process-wide cache would keep the trainer it closes over alive for good.
- `OuterProgram`: A jitted function whose compiles all go through the kernel report (`report.compiled`). When a launch's kernel spills registers, the report refuses that config and relowers the program once, traced afresh past JAX's trace cache, so the derivation takes its next config. Where the derivation has no next config, the first executable stays; a replacement that spills too is recorded, not retried. It keys its executables the way jit does: static arguments, the others' structure, shapes, dtypes and placements, and JAX's trace context.

The order of events for a training step is:

1. The program's `jax.jit` traces the step, and so produces a `ClosedJaxpr`. Autodiff runs during tracing, so this jaxpr already holds the backward pass.
2. The outer jit lowers to StableHLO. Each Pallas kernel appears there as a custom call.
3. The planner compiles candidate steps: one per split and rematerialization rung, each lowered and compiled under the step's XLA options (`step_compiler_options`). It keeps one that fits in memory. Compiling goes through `report.compiled`, which reads each Mosaic kernel's binary from the executable and relowers once on a spill.
4. The persistent compilation cache, on by default, saves XLA's work across runs. Tracing and lowering still run on every start. Ahead-of-time compilation (`compile_ahead`) exists, but a run that loads its result still traces and lowers, to check that the program it loads is the program it would compile.

### The algebra pass (on base/kernels)

Autodiff and plain code produce correct math, but not always cheap math. The spec's example is a unit norm's backward: a norm of norms makes its reductions wait on each other, so the backward reads each row several times, while three algebraic laws make every reduction independent, so they can all run in one pass. The algebra pass finds such forms in the traced step, between tracing and lowering, and puts the cheapest equal one in place.

It holds every equal form of a region in an e-graph, applies declared rules until no new form appears, and extracts the cheapest. Each rule carries a rounding class: exact rules change nothing a float computes, and reorderings change only rounding, admitted where the gate's contract for their class holds and where no new intermediate can overflow where the program did not. Accurate numerics admit only exact rules. Its limits come from the program's size, never the clock, so every process compiles the same program, which multi-controller JAX needs. A new form is priced by a model of the device and put in place only where it is cheaper; one switch, on by default, turns the pass off. Its design is the [algebraic optimization](algebra.md) spec's; base/kernels' copy of this page describes its code.

## Kernel routing

*One template per operation, with no selector, is planned.*

When one of Dew's layers calls an operation, a selector decides which kernel runs, while the program traces, from static facts alone. Each operation is an `Op`, a list of kernels in order. A kernel takes a call only if all of its requirements hold: the device's family, dtype, head width, the mask's form, sinks, softcap, dropout, determinism. The operation's preferred kernel runs where it takes the call, else the first in the operation's order that does, and a declared cell's old route runs instead where it takes the call; the route taken is recorded. Nothing is timed or ranked at runtime.

The device's family comes from its compute capability: sm80, sm86, sm89 and sm120 are `mma.sync`; sm90 is `wgmma`; sm100 and sm103 are `tcgen05`. An old route keeps a cell only where a measured receipt shows it faster, declared on the operation (`Op.declared`). The end state in the spec is no selector at all: one template per operation, and plain XLA by a fixed capability rule on a GPU outside every family a template claims.

| Operation | Default today | Template's place |
|---|---|---|
| Attention, bf16 or fp16, on H100 or A100 | FlashAttention, version 3 on sm90 and 2 on sm80, where it takes the call (declared cells, built into Dew's `dew-flash-attn` wheel) | Runs what FlashAttention refuses and the template supports: span masks, block groups, sinks, softcap, key validity. The template reads masks as spans, so an arbitrary dense mask goes to XLA or the reference implementation. The Mosaic mainloop is qualified on wgmma only, so on sm80 the template runs its Pallas-Triton mainloop. |
| Attention, fp32, on wgmma and mma.sync GPUs | XLA (declared) |  |
| Attention, sm100 and sm103 | Template, Pallas-Triton mainloop, or plain JAX | The tcgen05 mainloop is built but not routed. |
| Attention, decode-shaped | The Pallas-Triton decode kernel |  |
| Dense GEMM, sm80 and sm90 | XLA's dot, which calls cuBLAS (declared from measurements) | Dense calls reach the template through `template.dot_general`; every dense cell on these GPUs is declared to XLA by receipt. |
| Grouped GEMM (MoE) | Template: Mosaic GPU, then Mosaic TPU, then Triton | CUTLASS keeps the wgmma forward and input-gradient cells where installed; Triton keeps one mma.sync cell. |
| Linear recurrence: gated delta, SSD, selective scan | Template first | The ported kernels (fla, mamba_ssm's SSD) keep their declared wgmma and mma.sync cells. |
| RMSNorm and SwiGLU | XLA | Pallas-Triton on declared sm80 and sm90 cells: RMSNorm in bf16 and fp16, SwiGLU in bf16. |
| TPU | Splash attention; the Mosaic TPU grouped GEMM |  |
| CPU, and GPUs older than sm80 | XLA, or the template's plain-JAX mainloop |  |

So on today's H100 and A100, a plain causal bf16 attention runs FlashAttention and a dense projection runs cuBLAS. The templates run the calls those kernels cannot take, the grouped and recurrent operations, and every cell where they measured faster. The spec counts each declared old route as a defect in the template: the profile finds why it loses, the template is fixed, and the old route is deleted once the template matches it.

*Where it lives: `dew.nn.kernels.selection` (`Op`, `Kernel`, `select`), `dew.nn.kernels.generation` (`architecture`, `route`, `chosen`), `dew.nn.attention_kernels` (`ATTENTION`, `DECLARED`), `templates/attention_route.py`, `grouped_matmul.GROUPED`, `recurrence_chunk` (`GATED_DELTA`, `SCAN`, `SELECTIVE`), `norm_swiglu`.*

## The template engine

*Routing the tcgen05 bundle and a TPU bundle are planned.*

A template is one algorithm, written once, that generates the kernel for the device it runs on. Dew has three: attention, linear recurrence and GEMM. The spec's long-term goal is a whole forward and backward pass as one persistent kernel generated from the device's facts. That fixes two things about the templates: each derives its one kernel from facts, with no lookup tables and no sweeps, and each keeps its operands, outputs and resources explicit, so a later scheduler can place them.

![A template's layers. By design a lower layer holds no decision an upper one owns: a mainloop never masks, applies a hook, narrows a dtype or does softmax math, and the algebra never issues a product or knows its GPU. The recurrence mainloop does not meet that yet: it still splits products, casts and adds the skip term itself. TCGEN05 exists, but no route uses it.](https://raw.githubusercontent.com/AshishKumar4/dew/main/docs/assets/architecture-template.svg)

*Figure 3. A template's layers. By design a lower layer holds no decision an upper one owns: a mainloop never masks, applies a hook, narrows a dtype or does softmax math, and the algebra never issues a product or knows its GPU. The recurrence mainloop does not meet that yet: it still splits products, casts and adds the skip term itself. TCGEN05 exists, but no route uses it.*

### The algebra interface

The math of a template lives once, as named pure functions on tile values. For attention, the state carried along the keys is a `Partial`: the running maximum, the running denominator and the accumulator. `empty` gives the identity state, `plain_step` folds in one tile of scores and values, `merge` combines two partial states from a split walk, and `finish` returns the output, the maximum and the denominator. The spec calls these four `init`, `step`, `merge` and `finish`, and that is the shape the derived forms take ([derived forms](#derived-forms-on-basekernels)). Rounding points are algebra too: where values narrow to the operand dtype, and the dtype of every saved residual. The hand-written recurrence algebra has no such protocol yet. Its tile functions (`intra`, `carried`) take an `issue` callable that the mainloop supplies.

### Mainloops, toolkit and bundles

Everything that differs by GPU architecture lives once, in the shared Mosaic toolkit, as one `Family` per instruction family. A family holds its matrix instructions and their waits, where accumulators live, its register facts and its roles, and the streams and rings that load operands: `RingFamily` with a producer warpgroup feeding TMA rings, `ConsumerFed` with each consumer feeding its own. A mainloop holds only the order of its bundle's steps around the algebra. A template has one mainloop for every family whose overlap structure is the same, and a separate thin one only where it differs.

Four bundles exist. `WGMMA` serves sm90: a producer warpgroup and two consumers, with 227 KiB of shared memory. `FED` is wgmma with no producer, each consumer feeding its own ring, for tiles whose consumers need more registers than a producer leaves them (attention's backward takes it). `Mma` serves sm80 to sm89 with `mma.sync` and a single consumer, with 163 KiB of shared memory on sm80 and 99 KiB on sm86 and sm89. `TCGEN05` serves sm100's tensor-memory accumulators, but `bundles.of` does not return it yet, so no route reaches it. An sm120 GPU, `mma.sync` with TMA loads, has no bundle yet.

### Configs from device facts

Each template is a `Template`, Dew's launch protocol. Its `derived(call, target)` returns the one config for the call, or None, and None refuses the call with its reason. `legal` holds the hard constraints: the staged bytes fit the program's share of shared memory, the registers a role needs are held without spilling, and the compile report has not refused the config. `variants` lists the derived config and a few legal neighbors, which only the opt-in tuner times.

The facts come from `generation.GpuTarget`. The multiprocessor count, opt-in shared memory per block, clock and memory bandwidth are read from the device at runtime. Each instruction family's constants are fixed per architecture: tensor rates by precision, resident threads, whether it has wgmma and TMA, TMEM columns, measured load latency and L2 bytes per clock, and a TPU's VMEM limits and seconds per grid step. The register file is fixed: 64K registers per multiprocessor, at most 255 per thread, allocated in steps of 8. Dew's resource model budgets a wgmma consumer at 168 registers at three warpgroups, as if `setmaxnreg`'s raise were ignored, which it is: on main the Mosaic compile is relocatable, and ptxas then ignores the raise ([lowering](#lowering-to-machine-code)). Attention's forward derivation takes two consumers on wgmma when the step overlaps, the largest key tile that fits, and two stages.

The compile report checks each derived config against ptxas. It reads every Mosaic kernel's register count and spills from the compiled binary, refuses a spilling config and relowers the program once with the next one. Offline, a CPU compile gate in the tests holds the register model to ptxas at the cap each launch runs at.

> **Where the code and the spec part.** The spec's analytic launch model ranks nothing in the code: `templates.cost.seconds` has one caller, a test. The live derivations choose by fixed rules among legal configs: the GEMM takes the tile that reads its operands fewest times per FLOP, and attention's backward takes the legal fused tile with the largest blocks times consumers, then the most consumers. Nothing times configs on the default path. The spec puts sm80 and TPU attention on the Mosaic mainloop; in the code, sm80 attention runs on Pallas-Triton and TPU attention on Splash. The toolkit's names differ from the spec's (`products` and `issued` rather than `issue`; no `relayout` or `accumulators`), and the spec's public plain-JAX `xla_mainloop(config)` does not exist; the portable mainloop is `attention_portable.attend`.

*Where it lives: `templates/base.py` (`Template`), `templates/mosaic.py` (`Family`, `RingFamily`, `ConsumerFed`, `kernel`), `mosaic_sm80.py`, `mosaic_sm90.py`, `mosaic_sm100.py`, `bundles.py`, `resources.py`, `templates/cost.py`, `attention_algebra.py`, `attention_mosaic.py`, `templates/recurrence/`, `templates/gemm/`, `generation.py` (`GpuTarget`, `Architecture`).*

## Derived forms (on base/kernels)

A template's algebra is hand-written on main. The aim is to derive it from a declaration of what the operation computes, using the same e-graph engine as the pass: attention's softmax-weighted sum along the keys, a recurrence's update and readout along time. The derivation saturates the declaration's readings under rules that decouple a walked reduction into a carried state, the online softmax's rescale, log-sum-exp and moment merges among them, and extracts `init`, `step`, `merge` and `finish`, or refuses rather than emit half a derivation. It prices its candidates in the same units as the pass, with a model of one tile's walk on one multiprocessor in place of the whole device.

The derived forms are swapped in behind one switch, off by default, once they meet both bars: as fast on every scoreboard row the route serves, and on par by check 1 against float64. Their design is the [kernel templates](kernel-templates.md) and [algebraic optimization](algebra.md) specs'; base/kernels' copy of this page describes their code. A dev-time generator, which would check derived modules into the repository as reviewed code, is parked.

## Lowering to machine code

A Dew template kernel goes through these stages, top to bottom:

1. **Pallas.** The template builds its kernel with `mosaic.kernel`, which calls JAX's `plgpu.kernel` under warpgroup lowering semantics, with a kernel name the report can find.
2. **Mosaic GPU.** Pallas lowers the kernel to the Mosaic GPU MLIR dialect. A kernel that uses the grid-dependency primitives carries the PDL effect, and the launch is marked for it. The serialized module and its hash go into the program as a custom call, `mosaic_gpu_v2`.
3. **Dew's JAX.** Dew runs JAX from its fork (AshishKumar4/jax), published on pypi.dewml.dev, with Dew's build of its CUDA plugin; jaxlib and the PJRT plugin are the release's own. `pyproject.toml` and `constraints.txt` pin them, and the [installation page](../installation.md) lists the fork's fixes, each kept until upstream JAX carries it. Most are Mosaic GPU fixes the templates need: barriers at explicit parity, TF32 mma.sync, scoped atomics and fences and `griddepcontrol` through NVVM, the pre-Hopper pipeline's barrier placement, minimum blocks per multiprocessor, WGMMA descriptors formed at their instructions, TMA coordinates ptxas cannot fold, the MMA accumulator relaid out as the next product's operand on sm80.
4. **PTX, then ptxas, in the PJRT plugin.** The PJRT plugin (`xla_cuda_plugin.so`) links the Mosaic GPU custom call. When XLA builds the executable, the call's instantiate stage runs the Mosaic passes down to NVVM and PTX, then compiles the PTX with ptxas through XLA's compilation provider: in process where the build has the ptxas library, otherwise as the ptxas binary. The fork's patch that turns nvJitLink off sits in Dew's CUDA plugin, so it does not reach this compile: on main the PJRT plugin compiles relocatably, and ptxas then ignores `setmaxnreg` (C7504), which the resource model assumes. The result is cached per kernel hash.
5. **The cubin in XLA's executable.** The custom call keeps the cubin. It loads it with `cuModuleLoadData` before the first execution and launches it with `cuLaunchKernelEx`, with the cluster shape and programmatic stream serialization as launch attributes.

Dew then reads the cubin back. The kernel report walks the compiled executable's thunks and parses each Mosaic launch's ELF `.nv.info` section for its register count and stack frame. This is how a spill is caught and a config refused, on the compile path, before the step ever runs. The same lowering can target a GPU that is not attached: `generation.gpu_mesh` lowers Mosaic kernels for a named sm target through an abstract device, which the compile-only gate and ahead-of-time compiles use.

The other kernel paths are shorter. A Pallas-Triton kernel goes through Triton to a cubin inside jaxlib; JAX is removing that backend, and Dew keeps it only where it measures faster and jaxlib still runs it. FlashAttention 2 and 3 and CUTLASS's grouped GEMMs are compiled ahead of time into Dew-built wheels and called through `jax.ffi.ffi_call`. cuBLAS and cuDNN are reached through XLA and JAX.

*Where it lives: `templates/mosaic.py` (`kernel`, `compiler_params`), `templates/report.py` (`compiled`, `binaries`), `pyproject.toml` and `constraints.txt` (the pins), `docs/installation.md` (the fork's fixes); in the fork, `jaxlib/mosaic/gpu/custom_call.cc` and `target.cc`, and `jax_plugins/cuda/BUILD.bazel`.*

## Runtime

*Typed edges between launch pairs are planned.*

A resident training step, the default, is one XLA executable. Kernels are custom calls inside it, so one kernel hands off to the next on the device, with no trip to the host, and the trainer calls the executable directly, once per micro-batch part. A host-master step instead runs its compiled phases from a Python loop. Importing Dew before JAX initializes its backend supplies three XLA flags the run has not set itself: `--xla_allow_excess_precision=false`, so XLA keeps the roundings the program states, and on CUDA, allocator spatial partitioning off and IEEE-rounded fp32 division in PTX. An import after the backend is up changes nothing.

### Dependent launches (on base/kernels)

On sm90 and later, base/kernels makes every Mosaic launch a programmatic dependent launch: each kernel waits for the launch before it at its first memory access, exactly as without the feature, and lets the next launch start its setup once its own work is done, so a step's chain of kernels overlaps each launch's start with the last one's tail. The [kernel templates](kernel-templates.md) spec holds its design.

### Profiling

`dew profile run.json` runs a run's step for 20 warmup steps, captures 10 with XProf (through `dew.Profiler` over `jax.profiler`) and writes a `ProfileReport` beside XProf's own reports. A run can also capture a window during training (`TrainerConfig.profile`), which is off unless set. Region markers for the step, evaluation, compile and logging cost nothing without a capture. The kernel report records the route each operation took and each Mosaic kernel's registers and spills. The training profiler has no NVTX or Nsight Systems mode; only the PyTorch comparison tool (`tools/benchmark_torch.py --nsys`) brackets an Nsight capture.

*Where it lives: `dew.devices` (the import-time flags), `dew.telemetry.profile`, `dew.training.profiling`, `dew.cli.main.Profile`, `templates/report.py`.*

## Accuracy

*Check 2 covers far fewer sites than check 1.*

### The precision policy

A run's numerics are a `Numerics`, set with `trainer.precision`. It holds three things. The dtypes: parameters in fp32, compute in bf16 and outputs in fp32 by default. The precision of every product that states none: `default`, which is TF32 for fp32 on a GPU, or `high` or `highest`. And the forms: `fast` or `accurate`. The preset `accurate` runs fp32 throughout, every unstated product at `highest`, with the accurate forms. Code takes it for a block with `generation.run_precision(...)`. The default's contract is the fastest form that passes the gate on ordinary inputs and stays finite on adversarial ones; for attention, that is the base-2 fold and FlashAttention's backward arithmetic, with the anchored correction term for dQ. The Pallas-Triton attention and decode kernels still miss that contract on some extreme-logit families, a defect the testing spec lists.

Some choices hold whatever the policy says. RMS statistics are at least fp32 by default (RMSNorm's `fp32_statistics=False` keeps the input dtype), softmax is at least fp32 by default, gelu and silu compute in at least fp32, the vocabulary head returns fp32, the MoE router, recurrence state, SSD and chunked LM loss use exact fp32 products, and sums accumulate in fp32. The GEMM template sums in fp32 and rounds once.

### The gate: two checks against float64

Every numerical acceptance in Dew goes through one gate, with no tolerance of its own anywhere else.

- **Check 1, on par.** Over a site's declared seeds of drawn inputs, each output's RMS error from the float64 truth, pooled over the seeds, must be within 1.5 times the reference implementation's error on the same inputs (`PARITY`), with no floor. The worst row must be within 4 times the reference's (`ROW_FACTOR`), with wider factors for rows of one or two values (`NARROW_FACTORS`). Every element must be finite, and a leaf the reference gets exactly right must be exact. The decision is statistical. A leaf passes when the pooled ratio's bootstrap interval lies at or under 1.5, or when a well-resolved interval that straddles 1.5 has a point ratio at or under it; a typical-draw procedure can still resolve a leaf the pooled test leaves inconclusive. Any leaf that does not pass the first stage gets a second, larger draw of seeds, and a leaf still unresolved fails. The gate is `assert_as_exact`, which the numerical test files call.
- **Check 2, never wrong.** Every element of every equation must lie within a certified roundoff bound for that operation, on the operands the execution was given. A sum's bound is γ_n times the sum of absolute terms; comparisons, selections and moves must match bit for bit. Primitive certificates interpret one equation at a time. Native GEMM kernels are certified against intervals around their exact products, which account for tensor-core truncation and underflow, and the gated-delta recurrence has a native step witness (`tests/native_certificates.py`, `tests/recurrence_witness.py`). Attention, CUTLASS and the rest of the recurrence are still to come, and check 2 covers far fewer sites than check 1.

On base/kernels, the algebra engine plugs into the same gate: each compile-time rule is admitted by its rounding class and held to the gate's contract for that class ([algebraic optimization](algebra.md)).

*Where it lives: `dew.nn.precision` (`Numerics`, `Dtypes`, `ACCURATE`, `numerics_of`), `generation.run_precision`, `tests/reference_error.py` (`assert_as_exact`), `tests/check1_statistics.py`, `tests/certified.py` (`assert_certified`), `docs/design/testing.md`.*

## Shared contracts

A model is a plain Flax module. An objective defines variables, loss and validation. The trainer runs the objective with an Optax optimizer and owns execution, placement, checkpoints and logging. Sources and transforms compose through Grain. Post-training composes Dew's own objectives, rollouts and session sources.

### One owner per fact

| Fact or operation | Code owner | Contract |
|---|---|---|
| Device capabilities and peaks | `GpuTarget` and `TARGETS`, `src/dew/nn/kernels/generation.py`, interim | Device-specific choices read this table until the device facts read at runtime replace it ([kernel templates](kernel-templates.md#7-choice-the-one-kernel-each-template-derives)). A second hardware table is a second source of truth. |
| Kernel selection and routes | `GpuTarget.kernels`, `Route` and the selection interface in `generation.py`, interim | A layer requests an operation; the selector records the implementation that ran and the reason for any generic route. Kernel modules declare support as data. Each template's derivation from device and family facts, with each op's declared old-route cells, replaces the selector and its tables; nothing is picked or ranked at runtime ([kernel templates](kernel-templates.md#7-choice-the-one-kernel-each-template-derives)). |
| Launch configuration | `src/dew/nn/kernels/tuning.py`, on Tokamax's public API | Each launch takes the config its template derives from device facts (or its heuristic). An opt-in tuner times a few tested variants of it per device and shape and stores picks; a tuner win is a derivation gap. |
| Numerical acceptance | `tests/reference_error.py` | Every numerical consumer uses the [testing contract](testing.md#numerical-accuracy). No local tolerance policy. |
| Model work and MFU | `src/dew/nn/flops.py` and `WorkloadWork`, `src/dew/telemetry/workload.py` | One analytic FLOP formula per operation. MFU uses real, unpadded work over wall time and the measured device's peak. |
| Memory plan | `src/dew/training/memory.py` and `planning.py` | One search chooses a fitting micro-batch, rematerialization and head plan for the model and mesh. |
| Compilation cache | `src/dew/cache.py` | One cache dir and one autotune dir per (GPU model, jaxlib). A program's identity follows its computation and environment. |
| The batch contract | `src/dew/batch.py` | `Batch` and `Variables`, the frozen split and every variables-tree operation (`FROZEN`, `merge`, `freeze`, `thaw`, `select`, `part`), and the rows and tokens a batch carries (`VALID_ROWS`, `EXAMPLE_ROWS`, `padded_rows`, `Coordinates` and its token helpers) live here alone, below every package that reads a batch; it imports no other Dew package. |
| Mesh axes and rules | `src/dew/mesh.py` | The axis names (`MESH_AXES`, `BATCH_AXES`), the default rule table (`DEFAULT_RULES`), the spec rules give an array (`logical_spec`) and the axes a batch's rows split over (`row_axes`, `batch_axes`), below the models and the data alike; it imports no other Dew package. |
| Train state | `src/dew/state.py` | `TrainState` and the accumulation record a step takes and returns, read by objectives, checkpoints and the trainer from below them all. |
| Pool agreement | `src/dew/_coordination.py`, public through `dew.distributed` | One engine for host agreements, collective hosting and failure publication, below nn and data; hooks call `dew.distributed`. |
| Data execution | `src/dew/data/pipeline.py`, `analysis.py` and `lowering.py` | One `Dataset` chain, one analyzed plan, and the lowering that plan declares. |
| Weight layout | `src/dew/interop/` | One reversible source-to-model binding plan serves loading and export. Family adapters describe a source's schema. |
| Model capabilities | `src/dew/nn/protocols.py` | Objectives, tasks, serving and export ask for capabilities and call through Flax `apply`. |
| Training state and checkpoints | The training-state schema and `src/dew/checkpoints/` | Save and restore derive fields from the state. Casting and placement happen once. |
| Run configuration | `RunConfig.record()`, `src/dew/config/__init__.py` | A record pins every value its run built with. Resume and export read it rather than reconstructing a second configuration. |
| Outcomes and telemetry | `src/dew/telemetry/records.py` | Trackers, display and reports consume the same outcome records. |
| Environments and suite selection | Project extras and lock files; `.armada.json` + `tools/armada/ci.py` | Installers and CI read their package and test selections from these owners. No other package list anywhere. |
| Reinforcement-learning imports | `src/dew/rl` | `dew.rl` imports nothing from `dew.training`. Nothing under `src/dew` outside `src/dew/rl` and `src/dew/objectives/rl` may import `dew.rl`. A test walks the module tree and asserts it. |
| Staged training | `Recipe` and `Stage`, `recipes/chain.py` | Stages compose existing data, objectives and rollouts. `Objective.model_variables` defines what passes to the next stage. |

### Interfaces stay shared

Activations cross every layer-to-kernel boundary physically as `[batch, time, heads, head_dim]`. A kernel adapts its tiling to that interface. Its callers do not carry layout conversions for it. Physical layout checked in optimized HLO (no copies at the edge). Other tensors keep their own meaning.

Coordinates carry validity, positions, document identity and cache extent. Layers use those facts directly. A layer or kernel that receives a contract argument (segments, masks, validity, positions, cache, mesh) either uses it, refuses by name when it is set, or is independent of it by construction (a position-wise MLP needs no segments; say so in one comment). Refuse only where the alternative is a silent wrong result at a trust boundary (user input, files, external APIs). It is never silently dropped. Kernel-route qualification follows every registered model, target, mode and cache layout.

Layer identity survives fusion, scans and pipeline stages. The shared LM head owns logits, loss and scoring, including quantized and sharded forms. Guidance has one denoising interface; checkpoint restoration has one state schema. The guarantees exposed by these interfaces are described in [training's shared seams](training.md#9-the-shared-seams-as-a-user-sees-them).

A new family configures the shared attention, MLP, MoE, normalization, position and quantization modules. A missing capability belongs in the shared module. A source adapter maps source configuration and weight paths; it does not implement a second model. A new objective composes existing math and leaves model placement and execution to the trainer.

### Keep and compose

Start from the maintained implementation's public interface: JAX, Flax, Optax, Orbax, Grain, Tokamax or the system being integrated. Use it directly when it fits. A replacement needs a reason in the direction (not taste), and a parity list: every capability of the old code mapped to the new code or to a drop the user approved, checked before the old code is deleted.

A replacement is complete only when its callers move together. Before 1.0, the API changes directly, without aliases, migrations or parallel implementations.

Tests enumerate implementations from the registry that declares them. A new implementation joins the standing behavioral contracts in the same change. Its siblings are checked against the same contracts. Performance choices need same-session measurements at real shapes, including compilation cost, and [numerical qualification](testing.md#numerical-accuracy).

The [kernel roadmap](../roadmap.md#kernels-and-performance) records the approved template work. `TARGETS`, `GpuTarget.kernels` and the selector are interim: device facts read at runtime, each template's derivation and each op's declared old-route cells replace them ([kernel templates](kernel-templates.md#7-choice-the-one-kernel-each-template-derives)), as one owner each, never beside them; nothing is picked or ranked at runtime (owner, 2026-10-25).
