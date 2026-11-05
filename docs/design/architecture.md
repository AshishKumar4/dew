# Dew's architecture

Design record. Besides explaining how Dew works, this page owns the cross-cutting code boundaries, in its [shared contracts](#shared-contracts). [Training](training.md), [data](data.md) and [post-training](post-training.md) own their topic contracts, [algebraic optimization](algebra.md) and [kernel templates](kernel-templates.md) the two engines described here, and [testing](testing.md) behavioral qualification. The [roadmap](../roadmap.md) lists work still to complete or qualify.

This page follows one training run through Dew: the model and data you write, the step Dew builds from it, the point where that step is compiled, the rewrite Dew applies to its math, the kernel each operation runs, and how those kernels become machine code and run. Most of it is about the two parts Dew built itself: the algebra engine, which rewrites a program's math into a cheaper equal form, and the template engine, which generates GPU kernels from the device's own facts.

This page describes Dew's code as it stands. Parts the specs plan and the code does not have yet say so where they come up, and where the code and the specs disagree, the page follows the code and says so.

![One training run, top to bottom. Teal marks the algebra pass and the templates, this page's subject; the trainer and the run config are Dew's too.](https://raw.githubusercontent.com/AshishKumar4/dew/main/docs/assets/architecture-flow.svg)

*Figure 1. One training run, top to bottom. Teal marks the algebra pass and the templates, this page's subject; the trainer and the run config are Dew's too.*

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

The same run can come from a config. A `RunConfig` records the model by its import path and fields, the objective, the optimizer and the trainer's settings, in a `run.json` or a Python file whose `run()` returns it. `dew train run.json` runs it, and `--set path=value` changes any field. The command holds the XLA flags the run needs, re-executing the process if they change; `RunConfig.run` turns on the persistent compilation cache, builds the model and the objective around it, and constructs the same `Trainer`. Both paths then meet in `Trainer.fit`, and from there the step is compiled at the same point and takes the same algebra pass.

Which kernels run depends on the model, not on the path. Dew picks its templates inside its own layers: `dew.nn.Dense` and `DenseGeneral` send their product to the GEMM template (`nn/dense.py`), and Dew's attention, mixer and recurrence modules send theirs to the attention and recurrence templates. Nothing intercepts Flax's own ops, so a plain `flax.linen.Dense`, like the quickstart's, runs through XLA. A plain Flax model with any data gets the trainer, the pass and XLA; a model built from `dew.nn` layers gets the template kernels too.

Nothing in the user's code refers to kernels or the algebra engine. A model built from Dew's layers asks for attention, a projection or a norm through them, and which kernel runs, and in what algebraic form, is decided below that line. The few user-facing options that reach it are a run's precision (`trainer.precision` in its config: fast or accurate numerics), `attention_impl` (default `auto`) and the opt-in kernel tuner, `tune_kernels`, which is off by default.

*Where it lives: `dew.Trainer` (`fit`, `_training_loop`, `_compiled_for`, `compile`), `dew.objectives`, `dew.data`, `dew.training.transaction`; the config path in `dew.config.RunConfig` (`run`, `build_run`, `train`) and `dew.cli.main`. The [quickstart](../getting-started.md) walks through a first run step by step.*

## The compile point

Dew compiles its programs at one seam. Every program is an `OuterProgram`, whose `_jitted` jits `algebra.rewritten(function)`, so the algebra pass sits on that seam. The resident training step is one: `Trainer._jitted` returns `owned_program(step, in_shardings=…, out_shardings=…, donate_argnums=0)`. So are a host-master step's phases, an evaluation, a load, a sampler, a serving step and an RL rollout. A lint rule (SLOP025) keeps it that way: the one other jit in the library, `dew.nn.nested`, only marks a sub-program inside an enclosing trace and compiles nothing at the top level.

- `options.program(function, *models, step=None, **jit_arguments)`: The process-wide factory: one `OuterProgram` per function, compiler options and jit arguments, cached, so every call with the same inputs reuses it. Its compiler options come from the models it runs (`compiler_options`) and, for part of a training step, the step's own.
- `owned_program(...)`: The same program, kept by no cache of Dew's. A closure or bound method that a run makes, such as the training step, its placement's programs or its host phases, takes this form, because a process-wide cache would keep the trainer it closes over alive for good.
- `OuterProgram`: A jitted function whose compiles all go through the kernel report (`report.compiled`). When a launch's kernel spills registers, the report refuses that config and relowers the program once, traced afresh past JAX's trace cache, so the derivation takes its next config. Where the derivation has no next config, the first executable stays; a replacement that spills too is recorded, not retried. It keys its executables the way jit does: static arguments, the others' structure, shapes, dtypes and placements, and JAX's trace context.

The order of events for a training step is:

1. The program's `jax.jit` traces the step. The traced function is `algebra.rewritten(step)`, which binds static arguments as jit does, calls `jax.make_jaxpr` on the step, and so produces a `ClosedJaxpr`. Autodiff runs during tracing, so this jaxpr already holds the backward pass.
2. The algebra pass rewrites that jaxpr (next section), and the result is evaluated with `eval_jaxpr` inside the outer trace. The outer jit sees the rewritten program as its body, with the same inputs, outputs and output tree.
3. The outer jit lowers to StableHLO. Each Pallas kernel appears there as a custom call.
4. The planner compiles candidate steps: one per split and rematerialization rung, each lowered and compiled under the step's XLA options (`step_compiler_options`). It keeps one that fits in memory. Compiling goes through `report.compiled`, which reads each Mosaic kernel's binary from the executable and relowers once on a spill.
5. The persistent compilation cache, on by default, saves XLA's work across runs. Tracing, the algebra pass and lowering still run on every start. Ahead-of-time compilation (`compile_ahead`) exists, but a run that loads its result still traces and lowers, to check that the program it loads is the program it would compile.

### The algebra pass

*Rule sets beyond these, the dev-time generator and error-aware extraction are planned or parked.*

Autodiff and plain code produce correct math, but not always cheap math. The spec's example is a unit norm's backward: a norm of norms makes its reductions wait on each other, so the backward reads each row several times, while three algebraic laws make every reduction independent, so they can all run in one pass. Dew holds every equal form of a region in an e-graph, applies declared rules until no new form appears, and extracts the cheapest one. It uses [egglog](https://github.com/egraphs-good/egglog) through its Python bindings, pinned exactly in `pyproject.toml`.

![The compile-time pass. It runs while jax.jit traces, on the jaxpr, before anything is lowered. Its price comes from Dew's own model of the device.](https://raw.githubusercontent.com/AshishKumar4/dew/main/docs/assets/architecture-pass.svg)

*Figure 2. The compile-time pass. It runs while `jax.jit` traces, on the jaxpr, before anything is lowered. Its price comes from Dew's own model of the device.*

**Terms.** Each equation the language covers becomes a term: thirteen binary and twenty-three unary operations, six comparisons, three reductions, five layout operations, casts, three-way select and two-operand `dot_general`. Everything else becomes an opaque leaf the rules never touch: Pallas kernels, collectives, sharding constraints, random numbers, callbacks, effects, and anything inside a protected name scope, such as the error-free transforms of `dew.nn.precision`. Those transforms exist to keep a rounding residual, and reassociation would cancel exactly that residual.

**Windows.** The pass does not saturate a whole program. It finds heads where a rule can lower the price: a sum of two reductions, products or contractions of one kind, a reduction of a product, or a scaled contraction. Each head takes two levels of its producers, because no rule's left side is deeper than three. Windows that share equations merge, up to 64 equations, which keeps egglog's recursion bounded however long the program's chains are. Each cluster is saturated in an e-graph of its own; where two clusters share an equation, a value both yield takes the first cluster's form. The rest of the jaxpr stays as traced.

**Rules.** Three sets run at compile time. `identities` rounds exactly; today it holds only the commutativity of add and multiply. `linear` reorders rounding. Its ten rules factor `a·x + a·y` into `a·(x + y)`, join two contractions that share an operand into one contraction of the summed others, move a power-of-two scale onto either operand of a contraction, take a broadcast factor out of a sum, and take a row or column scale (a dequantization's) out of a matrix product. `joined` also reorders: it turns two sums along one axis into one sum of their terms. Every rule applies wherever the pass's cost model prices its rewrite cheaper, with no condition on XLA: where XLA would make the same rewrite the result is the same, and the pass's fire on floats where XLA's are held to integers, powers of two or fast math. `identities` only aids saturation (`Rules.aid`). A third set, `stable` (log1p, expm1, the log-sum-exp rearrangement), is registered for dev time only. A set is admitted by its rounding class: `exact` always, `reorders` under default numerics, never under accurate numerics.

**Range facts.** Each term carries a magnitude bound as an integer exponent e, meaning |value| ≤ 2^e (`terms.Magnitude`). A sum's exponent is one more than its larger term's, a product's is the sum of its factors', a contraction's adds the ceiling of log2 of its length, and `exp` is bounded only where its input's exponent is at most 6, at one power above its exact bound for XLA's rounding. Every bound is its output dtype's too: a boolean's at most 1, an integer's within its dtype's range and at it where an unsigned result wraps or a bitwise operation makes it; a conversion keeps a value at most 0 only where it keeps every such value's sign. A generated check (`tests/test_algebra_bounds.py`) proves each integer claim at every width, exhaustively at 8 bits and at the box's corners and critical points above, and checks each float claim differentially at its bound's edges. A reordering may fire only if every new intermediate's bound stays under a bound the program reaches as a number: one over its constants and declared ranges alone, with no unbounded operand in it, and at most the narrowest float's largest power of two. A bound over an unbounded operand bounds nothing, as the program's value may sit far below it while a new intermediate overflows, so a reordering whose new intermediate sums unbounded operands is refused. That keeps `0.5·x + 0.5·y` at `x = y = 3e38` as it is, where `0.5·(x + y)` would overflow; the tests hold each reordering rule finite wherever the program is.

**Saturation.** Each round runs each admitted set once, with the shape facts before each set and once more after, until nothing changes or a limit is hit. The limits come from the region's size, never from the clock: a budget of 64 nodes per equation, checked between rounds (so a round can pass it), and 4 rounds per level of the region's longest chain. A region that hits a limit keeps the cheapest form found so far, and the report says so. Because the limits come from the input, the same jaxpr gives the same program in any process. Multi-controller JAX needs that, since every process must compile the same program.

**Extraction and splicing.** egglog's greedy DAG extractor picks each root's form under Dew's cost model, which counts a shared value once. The pass prices each region before and after saturating, and splices the region's new forms only if the region as a whole is cheaper (`cost.cheaper`). No rewrite is left to XLA's cost analysis: the pass's cost model decides each. Each picked form is traced back into JAX primitives with their original parameters, checked for shape and dtype, and put in place of the equation that produced the old value. Each program's rewrites (the regions, prices before and after, the sets that fired and whether a limit stopped saturation) join the kernel report.

**The switch.** `dew.algebra.passes(on, target)` sets `dew.cache.ALGEBRA`, a JAX user context that is on by default. Because it lives in JAX's trace context, any program held under one setting retraces under another. The run's numerics reach the pass: under accurate numerics only the exact sets are admitted, and since those only aid saturation, the pass then leaves the program as it is. The spec has the accuracy gate run its reference arm with the pass off and the candidate with it on; no gate does that yet, and the pass-off comparison is a test of the pass's own.

> **Where the code and the spec part.** The spec names a `reuse` set, an `identities` set that also covers rsqrt and sqrt, exp and log, abs and sign and cast chains, and its own DAG extractor; the code has commutativity alone and uses egglog's greedy-DAG extractor. The spec says the planner's memory trace goes through the pass; in the code, `footprint` traces the loss without it, so the planner prices the unrewritten program.

*Where it lives: `dew.nn.kernels.options` (`program`, `owned_program`, `OuterProgram`), `dew.training.trainer.Trainer._jitted`, `dew.training.transaction.Transaction.step`, `dew.algebra.program` (`rewritten`, `rewrite`, `_windowed`, `_region`, `_splices`), `dew.algebra.terms`, `dew.algebra.rules`, `dew.algebra.cost`, `dew.cache.ALGEBRA`.*

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

The math of a template lives once, as named pure functions on tile values. For attention, the state carried along the keys is a `Partial`: the running maximum, the running denominator and the accumulator. `empty` gives the identity state, `plain_step` folds in one tile of scores and values, `merge` combines two partial states from a split walk, and `finish` returns the output, the maximum and the denominator. The spec calls these four `init`, `step`, `merge` and `finish`, and that is the shape the derived forms take ([derived forms](#derived-forms)). Rounding points are algebra too: where values narrow to the operand dtype, and the dtype of every saved residual. The hand-written recurrence algebra has no such protocol yet. Its tile functions (`intra`, `carried`) take an `issue` callable that the mainloop supplies.

### Mainloops, toolkit and bundles

Everything that differs by GPU architecture lives once, in the shared Mosaic toolkit, as one `Family` per instruction family. A family is an `Issuer` composed with a `Transport`. The issuer owns the matrix instructions, their waits, where accumulators live and the register facts. The transport owns loading and the pipeline: the roles and their register budgets, the walk that streams each operand's blocks through rings, persistent loops, loads, stores and the fix-up of split tiles. A mainloop holds only the order of its bundle's steps around the algebra. A template has one mainloop for every family whose overlap structure is the same, and a separate thin one only where it differs.

Five bundles exist. `WGMMA` serves sm90: a producer warpgroup and two consumers, with 227 KiB of shared memory. `FED` is wgmma with no producer, each consumer feeding its own ring, for tiles whose consumers need more registers than a producer leaves them (attention's backward takes it). `Mma` serves sm80 to sm89 with `mma.sync` and a single consumer, with 163 KiB of shared memory on sm80 and 99 KiB on sm86 and sm89. `MmaFed` is the same instructions with TMA loads, for sm120. `TCGEN05` serves sm100's tensor-memory accumulators, but `bundles.of` does not return it yet, so no route reaches it.

### Configs from device facts

Each template is a `Template`, Dew's launch protocol. Its `derived(call, target)` returns the one config for the call, or None, and None refuses the call with its reason. `legal` holds the hard constraints: the staged bytes fit the program's share of shared memory, the registers a role needs are held without spilling, and the compile report has not refused the config. `variants` lists the derived config and a few legal neighbors, which only the opt-in tuner times.

The facts come from `generation.GpuTarget`. The multiprocessor count, opt-in shared memory per block, clock and memory bandwidth are read from the device at runtime. Each architecture's facts are fixed per compute capability (`generation.Architecture`): tensor rates by precision, resident threads and blocks, its register file (64K registers per multiprocessor, at most 255 per thread, allocated in steps of 8), the shared memory each block reserves, its shared-memory and special-function rates, `setmaxnreg`'s range, whether it has wgmma and TMA, TMEM columns, a TPU's VMEM limits, and whether the GPU supports dependent launch; the resource and cost models read these facts, never the kind. Dew's resource model raises a wgmma consumer to its program's share of the register file, at most `resources.raise_ceiling()` (248), as the compile honours `setmaxnreg` ([lowering](#lowering-to-machine-code), `resources.RAISES_HONORED`). Attention's forward derivation takes two consumers on wgmma when the step overlaps, the largest walked tile that fits and two stages. The GEMM's Mosaic derivation is the most worked out: each rule names the fact it reads, from the tile that loads its operands the fewest times per FLOP within the consumers' registers, through the step, programs per multiprocessor, stages, cluster and raster band.

The compile report checks each derived config against ptxas. It reads every Mosaic kernel's register count and spills from the compiled binary, refuses a spilling config and relowers the program once with the next one. Offline, a CPU compile gate in the tests holds the register model to ptxas at the cap each launch runs at.

> **Where the code and the spec part.** The spec's analytic launch model is deleted in the code: `templates/cost.py` holds only the cost vocabulary ([derived forms](#derived-forms)). The live derivations choose by fixed rules among legal configs: the GEMM takes the tile that reads its operands fewest times per FLOP, and attention's backward takes the legal fused tile with the largest blocks times consumers, then the most consumers. Nothing times configs on the default path. The spec puts sm80 and TPU attention on the Mosaic mainloop; in the code, sm80 attention runs on Pallas-Triton and TPU attention on Splash. The toolkit's names differ from the spec's (`products` and `issued` rather than `issue`; no `relayout` or `accumulators`), and the spec's public plain-JAX `xla_mainloop(config)` does not exist; the portable mainloop is `attention_portable.attend`.

*Where it lives: `templates/base.py` (`Template`), `templates/mosaic.py` (`Issuer`, `Transport`, `Family`, `kernel`, `dependent`), `mosaic_sm80.py`, `mosaic_sm90.py`, `mosaic_sm100.py`, `bundles.py`, `resources.py`, `templates/cost.py`, `attention_algebra.py`, `attention_mosaic.py`, `templates/recurrence/`, `templates/gemm/`, `generation.py` (`GpuTarget`, `Architecture`).*

## Derived forms

*Behind a switch that is off by default. The dev-time generator and error-aware extraction are parked; the `linear_scan` and `error_free` rule sets are planned.*

The template's algebra is hand-written today. The aim is to derive it from a declaration of what the operation computes, using the same e-graph engine, and to swap the derived forms in behind one switch once they meet both bars: as fast on every scoreboard row the route serves, and on par by check 1 against float64.

![The derivation path of a Reduction, attention's. A Scan takes a shorter one: its own matcher and a fixed chunk algorithm, then the carry's extraction. Derivation runs at trace time today; the spec's dev-time generator, which would check the derived modules into the repository as reviewed code, is parked.](https://raw.githubusercontent.com/AshishKumar4/dew/main/docs/assets/architecture-derived.svg)

*Figure 4. The derivation path of a Reduction, attention's. A Scan takes a shorter one: its own matcher and a fixed chunk algorithm, then the carry's extraction. Derivation runs at trace time today; the spec's dev-time generator, which would check the derived modules into the repository as reviewed code, is parked.*

**Declarations.** A declaration states semantics only, never a form. Attention's is a `Reduction`: on whole operands, `m = max(s)`, `p = exp(s − m)`, `out = (p·v) / Σp`, with the keys as the walked axis and the exponent base (2 for the folded form, e for the natural one) stated by the template. A recurrence's is a `Scan`: the update `h ← exp(g)·h − β·k(kᵀ exp(g)·h) + β·k·vᵀ` (the middle term for the delta rule only), the readout `scale·qᵀh`, and time as the walked axis.

**Derivation.** `derive` traces the declaration at one tile into terms. Facts saturate first: which values lie along the walked axis, which reductions sum or take a maximum along it, and which values scale as `exp(−k)` of a walked maximum. Each walked reduction gets a slot, a carried state. The roots are the readings `stepped` and `merged` for each reduction and `finished` for each output, and the decoupled rules distribute them through the language: the running-maximum rescale of online softmax, log-sum-exp merges, moment merges. Extraction prices candidates with the tile walk. A reading no rule rewrote has no price, and the declaration is refused rather than emitted half-derived. The result is a `Derived` holding `init`, `step`, `merge` and `finish`.

**Binding.** With the switch on (`dew.algebra.forms.derived()`, which sets `dew.cache.DERIVED`), attention's `plain_step`, `merge` and `finish` take the derived forms. Those functions serve the Mosaic forward, Triton, portable and decode paths, for the forms the derivation covers, keys-first tiles among them. A dropping call's forward step stays hand-written: `plain_step` keeps its dropped weights as scaled parts, since the derivation narrows a dropped weight past fp16's range. The backward's derived forms do cover dropout, and mask each tile's score cotangents where its queries see no key; a query a score hook leaves seeing no key takes no output cotangent at all (`attention.launch_arguments`), as the reference's, and a common walk's holder wholly past the call's keys is masked (`Forward.walked_keys`). The recurrence's derived chunk restarts its decays' compensated prefix at a token whose decay vanishes in its dtype, and holds a walk's decay there, so no span's or walk's sum overflows. The forms it does not cover keep the template's hand-written implementation: headroom-guarded forms, which decode uses, and the accurate, natural and compensated forms. A derivation the engine starts and cannot complete refuses; it never falls back. The recurrence's portable branch takes its derived chunk form. The switch is off by default, and only tests turn it on.

> **Where the code and the spec part.** The spec's `derive(declaration, numerics, cost)` is `derive(declaration, *, accurate, saturation)` in code, and the spec's `linear_scan` and `error_free` sets do not exist. The spec notes its own drift: the scan's chunk form is still matched by its own symbolic matcher and emitted as one hard-coded chunk algorithm, not extracted under a kernel cost. The anchored training walk (`training_step`) never reads the switch.

### One cost vocabulary, two schedules

Both uses of the engine price their candidates in the same units, and the device facts come from one owner.

![One vocabulary, two schedules. A candidate's terms sum to one Work record over the chosen DAG, each shared value counted once; the schedule turns that record into seconds. Neither schedule compiles anything.](https://raw.githubusercontent.com/AshishKumar4/dew/main/docs/assets/architecture-cost.svg)

*Figure 5. One vocabulary, two schedules. A candidate's terms sum to one `Work` record over the chosen DAG, each shared value counted once; the schedule turns that record into seconds. Neither schedule compiles anything.*

The algebra counts its terms into `Work`: each input of the program its bytes once, a product's FLOPs and the shared-memory bytes it stages, a reduction's operations, a special function's exponentials, a cast's conversions, any other elementwise operation's operations, and nothing for a layout change. `cost.fused` models what XLA will do with an ordinary program: the program's inputs are read once from memory, elementwise work fuses into its consumer, and each unit runs at the whole device's rate, in series with that traffic. `cost.walk` models a template's inner loop: one tile on one multiprocessor, its tensor, shared-memory, SFU and CUDA-core time overlapped, its operands staged by the walk, so it charges no device bytes. Today the tile walk does not read register pressure or the walk's rings when it prices a candidate. Those belong to a whole chosen form, and the compile gate checks them afterwards. The prices are uncalibrated: neither schedule has receipts against measured profiles yet.

*Where it lives: `dew.algebra.derive` (`Reduction`, `derive`), `dew.algebra.scan` (`Scan`, `derive_scan`), `dew.algebra.carry`, `dew.algebra.forms` (`Derived`, `derived`), `dew.algebra.cost` (`work`, `fused`, `walk`), `templates/cost.py` (`Work`, `walked`), `attention_algebra` (`declaration`, `_derived_step`), `recurrence_chunk.derived_rule`.*

## Lowering to machine code

A Dew template kernel goes through these stages, top to bottom:

1. **Pallas.** The template builds its kernel with `mosaic.kernel`, which calls JAX's `plgpu.kernel` under warpgroup lowering semantics, with the minimum resident blocks per multiprocessor and a kernel name the report can find. The body is wrapped for dependent launch ([runtime](#runtime)).
2. **Mosaic GPU.** Pallas lowers the kernel to the Mosaic GPU MLIR dialect. A kernel that uses the grid-dependency primitives carries the PDL effect, and the launch is marked for it. The serialized module and its hash go into the program as a custom call, `mosaic_gpu_v2`.
3. **Dew's JAX.** Dew runs JAX from its fork (AshishKumar4/jax), published on pypi.dewml.dev, with Dew's builds of its CUDA and PJRT plugins; jaxlib is the release's own. `pyproject.toml` and `constraints.txt` pin them, and the [installation page](../installation.md) lists the fork's fixes, each kept until upstream JAX carries it. Most are Mosaic GPU fixes the templates need: barriers at explicit parity, TF32 mma.sync, scoped atomics and fences and `griddepcontrol` through NVVM, the pre-Hopper pipeline's barrier placement, minimum blocks per multiprocessor, WGMMA descriptors formed at their instructions, TMA coordinates ptxas cannot fold, the MMA accumulator relaid out as the next product's operand on sm80.
4. **PTX, then ptxas, in the PJRT plugin.** The PJRT plugin (`xla_cuda_plugin.so`) links the Mosaic GPU custom call. When XLA builds the executable, the call's instantiate stage runs the Mosaic passes down to NVVM and PTX, then compiles the PTX with ptxas through XLA's compilation provider: in process where the build has the ptxas library, otherwise as the ptxas binary. The fork's patch that turns nvJitLink off sits in Dew's PJRT plugin, so this compile is whole-program and ptxas honours `setmaxnreg`, where a relocatable compile ignored it. The result is cached per kernel hash.
5. **The cubin in XLA's executable.** The custom call keeps the cubin. It loads it with `cuModuleLoadData` before the first execution and launches it with `cuLaunchKernelEx`, with the cluster shape and programmatic stream serialization as launch attributes.

Dew then reads the cubin back. The kernel report walks the compiled executable's thunks and parses each Mosaic launch's ELF `.nv.info` section for its register count and stack frame. This is how a spill is caught and a config refused, on the compile path, before the step ever runs. The same lowering can target a GPU that is not attached: `generation.gpu_mesh` lowers Mosaic kernels for a named sm target through an abstract device, which the compile-only gate and ahead-of-time compiles use.

The other kernel paths are shorter. A Pallas-Triton kernel goes through XLA's Triton to a cubin inside the PJRT plugin; Dew routes to it where it measures faster. FlashAttention 2 and 3 and CUTLASS's grouped GEMMs are compiled ahead of time into Dew-built wheels and called through `jax.ffi.ffi_call`. cuBLAS and cuDNN are reached through XLA and JAX.

*Where it lives: `templates/mosaic.py` (`kernel`, `compiler_params`), `templates/report.py` (`compiled`, `binaries`), `pyproject.toml` and `constraints.txt` (the pins), `docs/installation.md` (the fork's fixes); in the fork, `jaxlib/mosaic/gpu/custom_call.cc` and `target.cc`, and `jax_plugins/cuda/BUILD.bazel`.*

## Runtime

*Typed edges between launch pairs are planned.*

A resident training step, the default, is one XLA executable. Kernels are custom calls inside it, so one kernel hands off to the next on the device, with no trip to the host, and the trainer calls the executable directly, once per micro-batch part. A host-master step instead runs its compiled phases from a Python loop. Importing Dew before JAX initializes its backend supplies three XLA flags the run has not set itself: `--xla_allow_excess_precision=false`, so XLA keeps the roundings the program states, and on CUDA, allocator spatial partitioning off and IEEE-rounded fp32 division in PTX. An import after the backend is up changes nothing.

### Dependent launches

On sm90 and later (sm90, sm100, sm103, sm120), every Mosaic launch is a programmatic dependent launch. `mosaic.dependent` wraps the kernel body so that it calls `griddepcontrol.wait` before touching any memory, so it is ordered against the launch before it exactly as without the feature. Once its work is done, it calls `griddepcontrol.launch_dependents`, which lets the next launch start its setup while this launch's last programs finish; whether they overlap is up to CUDA's scheduler and the free resources. The trigger sits at the end of the work, not at entry. On an H100's recurrence launches, a trigger at entry let the next launch's programs take multiprocessors this launch's later waves needed, and measured 0.6 to 1.0% slower than no dependent launch at all. CUTLASS triggers after its mainloop for the same reason. There is no switch. The wrapper passes the body through unchanged on sm80 to sm89, on TPUs and when interpreted. Triton and XLA launches do not overlap this way.

### Profiling

`dew profile run.json` runs a run's step for 20 warmup steps, captures 10 with XProf (through `dew.Profiler` over `jax.profiler`) and writes a `ProfileReport` beside XProf's own reports. A run can also capture a window during training (`TrainerConfig.profile`), which is off unless set. Region markers for the step, evaluation, compile and logging cost nothing without a capture. The kernel report records the route each operation took, every algebra rewrite and each Mosaic kernel's registers and spills. The training profiler has no NVTX or Nsight Systems mode; only the PyTorch comparison tool (`tools/benchmark_torch.py --nsys`) brackets an Nsight capture.

*Where it lives: `templates/mosaic.dependent`, `generation.Architecture.dependent_launch`, `dew.devices` (the import-time flags), `dew.telemetry.profile`, `dew.training.profiling`, `dew.cli.main.Profile`.*

## Accuracy

*The gate's use of the algebra switch is planned, and check 2 covers far fewer sites than check 1.*

### The precision policy

A run's numerics are a `Numerics`, set with `trainer.precision`. It holds three things. The dtypes: parameters in fp32, compute in bf16 and outputs in fp32 by default. The precision of every product that states none: `default`, which is TF32 for fp32 on a GPU, or `high` or `highest`. And the forms: `fast` or `accurate`. The preset `accurate` runs fp32 throughout, every unstated product at `highest`, with the accurate forms. Code takes it for a block with `generation.run_precision(...)`. The default's contract is the fastest form that passes the gate on ordinary inputs and stays finite on adversarial ones; for attention, that is the base-2 fold and FlashAttention's backward arithmetic, with the anchored correction term for dQ. The Pallas-Triton attention and decode kernels still miss that contract on some extreme-logit families, a defect the testing spec lists.

Some choices hold whatever the policy says. RMS statistics are at least fp32 by default (RMSNorm's `fp32_statistics=False` keeps the input dtype), softmax is at least fp32 by default, gelu and silu compute in at least fp32, the vocabulary head returns fp32, the MoE router, recurrence state, SSD and chunked LM loss use exact fp32 products, and sums accumulate in fp32. The GEMM template sums in fp32 and rounds once.

### The gate: two checks against float64

Every numerical acceptance in Dew goes through one gate, with no tolerance of its own anywhere else.

- **Check 1, on par.** Over a site's declared seeds of drawn inputs, each output's RMS error from the float64 truth, pooled over the seeds, must be within 1.5 times the reference implementation's error on the same inputs (`PARITY`), with no floor. The worst row must be within 4 times the reference's (`ROW_FACTOR`), with wider factors for rows of one or two values (`NARROW_FACTORS`). Every element must be finite, and a leaf the reference gets exactly right must be exact. The decision is statistical. A leaf passes when the pooled ratio's bootstrap interval lies at or under 1.5, or when a well-resolved interval that straddles 1.5 has a point ratio at or under it; a typical-draw procedure can still resolve a leaf the pooled test leaves inconclusive. Any leaf that does not pass the first stage gets a second, larger draw of seeds, and a leaf still unresolved fails. The gate is `assert_as_exact`, which the numerical test files call.
- **Check 2, never wrong.** Every element of every equation must lie within a certified roundoff bound for that operation, on the operands the execution was given. A sum's bound is γ_n times the sum of absolute terms; comparisons, selections and moves must match bit for bit. Primitive certificates interpret one equation at a time. Native GEMM kernels are certified against intervals around their exact products, which account for tensor-core truncation and underflow, and the gated-delta recurrence has a native step witness (`tests/native_certificates.py`, `tests/recurrence_witness.py`). Attention, CUTLASS and the rest of the recurrence are still to come, and check 2 covers far fewer sites than check 1.

The algebra engine plugs into the same gate. A compile-time rule is admitted by its rounding class and held to the gate's contract for that class. The gate runs every reference arm and float64 truth with `passes(on=False)` and the candidate as users run it, the pass on (`tests/arm_arithmetic.py`, `unrewritten`), so a program compared with its unrewritten self passes one function as both arms. The spec's error-aware extraction, the cheapest candidate whose measured check-1 error meets the policy, exists as `dew.algebra.error.choose`, with only tests calling it. It is meant for the dev-time generator, which is parked.

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
