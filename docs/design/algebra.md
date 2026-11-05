# Algebraic optimization: Dew derives the cheapest form of its own math

Design record, 2026-11-01. This is the binding design for how Dew rewrites the math of its programs and kernels
into cheaper or more accurate equivalent forms. Work items live in the [roadmap](../roadmap.md). It builds on
[the kernel templates](kernel-templates.md), [training](training.md) and [testing](testing.md).

## 0. The idea in one paragraph

Autodiff and plain code produce correct math, not cheap or stable math. A unit norm's backward, for example, reduces
over each row; a norm of norms makes those reductions wait on each other, so the backward reads every row several
times. Three algebraic laws (linearity of sums, pulling a per-row scalar out of a sum, distributivity) make every
reduction independent, so all of them run in one pass. Today such rewrites are derived by hand as custom VJPs or
template algebra. Dew instead holds every equivalent form of a computation in an e-graph, applies declared rules
until no new form appears (equality saturation), and extracts the cheapest form whose error meets the precision
policy. The engine runs in two modes. At dev time a generator derives Dew's own math once, and its output is
ordinary checked-in, reviewed code. At compile time a small, fast pass rewrites the one thing Dew cannot know in
advance: the autodiff output of the program a user composed. Nothing runs at runtime.

## 1. The engine: `dew.algebra`

### 1.1 Two modes

**Parked: the dev-time mode** (this section's dev-time row, §6.11). Derived forms run at trace time today
(`derive`, `scan`: the forms are built where a template calls them), and nothing has needed a generator's
checked-in output yet. The text stays as the design to resume.

| Mode | When | Input | Rule sets | Output |
|---|---|---|---|---|
| Dev time | `python -m dew.algebra.generate`, run by a developer when a declaration or a rule changes | declarations: template algebra (attention's reduction, a recurrence's update), stable numeric forms, surrogate gradients' bodies | every set, with sampled error estimates (§1.5) | Python modules and their tests, checked in and reviewed like hand-written code |
| Compile time | the program pass, once per program Dew compiles at its seam (§2) | the traced jaxpr of a training step, an evaluation, a rollout or a serving program, a step's backward included | `identities`, `contracted`, `linear`, `joined`: small and fast, admitted by rounding class (§1.4) | the rewritten jaxpr, compiled under the program's own options |
| Runtime | never | — | — | — |

- **Dev time** covers everything whose code Dew knows in advance: a template's forward and backward tiled forms,
  online and combine forms; numerically stable forms of Dew's own functions; and the internals of declared
  surrogate gradients (§1.3). The generator writes ordinary modules (`*_derived.py` beside the module that
  declares them) with a header naming the declaration, the rule sets and the egglog version. Each generated module
  ships with generated tests: the accuracy gate's sites for its forms, against the declaration at float64 (§3).
  CI regenerates them and fails when the checked-in output differs (`generate --check`), so a rule change shows up
  as a reviewed diff of generated code, never as a silent change in behavior.
- **Compile time** covers only what Dew cannot know in advance: the composition a user's model, objective and
  optimizer make, and the backward JAX's autodiff traces for it. It runs on every program Dew compiles at its
  seam: training steps, and the forwards of evaluation, rollouts, sampling and serving, which gain from it too,
  serving above all. Inside a program, the bodies Dew owns are marked (§6.10) and stay as dev time made them; the
  pass rewrites the composition and its autodiff output around them. Its rule sets are the ones whose saturation
  is fast and whose rounding class is static, so its cost is bounded by the program's size (§6.6) and it needs no
  sampling.
- **Runtime** runs nothing of the engine: a compiled program is ordinary XLA, and a generated module is ordinary
  Python.

### 1.2 Language, leaves and regions

- **E-graphs:** [egglog](https://github.com/egraphs-good/egglog), used through its Python bindings, not forked.
  Saturation stops at limits derived from the input's size (§6.6), never at a wall-clock limit, so the same input
  gives the same output in any process. Dew writes one piece itself: extraction that prices shared values once (a
  DAG cost), since a reduction read once and used twice costs one pass.
- **Language:** the arithmetic core of a jaxpr:
  - elementwise operations, casts, broadcasts, reshapes and transposes;
  - reductions (sum, max), `dot_general` and select;
  - rsqrt, sqrt, exp, log, log1p, expm1, abs and sign.
- **Opaque leaves,** never rewritten: Pallas kernels, collectives, `shard_map`, sharding constraints, random
  numbers, callbacks and effects.
- **Marked regions** (§6.10): Dew brackets a span of a computation with a marker primitive whose JVP and
  transpose rules bracket the span's tangent and cotangent too, so a span keeps its marker through autodiff. A
  marker carries its kind as an object, never a name string:
  - `Protected`: the error-free transforms of `dew.nn.precision` (`two_sum`, `two_product`, `compensated_add`,
    `compensated_difference`; `compensated_sum` and `widened_reduction` with their site's declaration, row 4026)
    and the numerics bodies of custom derivatives (§1.3). A protected span is a boundary no region crosses. At compile time it is one opaque leaf. At dev time
    the generator may propose another form for it, which extraction admits only when its error estimate meets the
    span's own (§1.5), and which lands only as reviewed generated code.
  - `Surrogate`: a declared surrogate gradient's forward or backward body (§1.3). Rules rewrite inside it, and no
    rule relates it to anything outside it.
  - `KernelGlue`: the XLA glue of a kernel's custom derivative around its Pallas launches, which rules rewrite as
    any other math; the launches stay opaque leaves.
- **Control flow:** the bodies of `scan`, `while`, `cond`, nested `jit` (`pjit`) and checkpoint (remat) are regions
  of their own, rewritten in place and recursively. Bodies are never unrolled, so a step's program depth stays
  fixed. Nothing moves across a body boundary or a loop carry.

Reassociation would cancel exactly the residual the error-free transforms exist to keep: `compensated_add`'s low
word, or the compensated sum that holds Mamba's D gradient inside check 1 (a plain sum puts it at 1.60 times the
reference's error, past the gate's 1.5).

### 1.3 Custom derivatives and surrogate gradients

Dew holds 58 custom-derivative sites (the inventory of 2026-11-01, by class: 24 kernel, 26 numerics, 8 surrogate),
and two more since: nGPT's join and normed hidden (`ngpt.stepped`, `ngpt.normed_hidden`, row 1308), which reverse in
two passes only to restructure the norm's algebra (§0's example). They stay hand-written: the one-pass form's
new intermediates are sums of the raw rows and of the joined rows `start + |alpha scale| (end - start)` against
the cotangent, and neither the cotangent nor the learned rate has a range the program proves or declares, so no
reordering to that form meets the dominance rule (§6.3). Autodiff's own backward is no safer: it forms the same
unscaled products (the pullback of `x r` to `r` is `sum(g x)`), and over residual, branch and cotangent scales
of 1 to 1e38 in fp32 the hand backward was never non-finite where autodiff's was finite, and finite in 137 cases
where autodiff's was not (row 4124). The engine sees through every site: it rewrites inside each custom VJP's and
JVP's forward and backward bodies, by the site's declared class:

- **Kernel** (the norm and SwiGLU kernels, the GEMM and attention templates, the recurrence kernels, the paged and
  Splash attention kernels): the launches are opaque leaves and their XLA glue is `KernelGlue`, rewritten freely.
  A kernel site's `linear_tangent` portable tangents are ordinary math and are rewritten as well.
- **Numerics** (`nn.categorical`'s label, soft-label and divergence tangents, `sampling.flow`'s range-safe KL,
  `nn.precision`'s compensated and correctly rounded functions, the recurrence algebra's decay prefix and readout,
  `exact_gelu`, the grouped projection's widened sums, among them): both bodies
  are `Protected`. They exist to restructure math for accuracy, often at a cost (the decay prefix costs 4–26% more
  for 3.6–4.5 times less error), so no compile-time rule undoes them. At dev time the generator searches each
  body's alternatives and proposes one only where its sampled error meets the body's own (§1.5).
- **Surrogate** (`Objective.with_gradients`' rules, node perturbation, the quantized convolution's
  straight-through backward, `rounded_operand`'s straight-through tangent, among them): a declared
  form, `surrogate(forward, derivative)` (row 3271), whose derivative is a rule, not the forward's. The engine
  optimizes inside each body (`Surrogate`) and never replaces the rule with autodiff's derivative of the forward,
  nor equates the two.

Each site declares its class where it defines its derivative (`dew.nn.derivatives`: `declared_derivative` or
`declared_tangent` with its kind, or `surrogate`; §6.10), so the class is read from the site, never inferred. A
bare custom derivative in `src/dew` is to fail a slop check, not take a default. Each declaring function wraps
`jax.custom_vjp` or `jax.custom_jvp` around its primal and its rules, each one marked call of its kind
(`dew.nn.derivatives.bound_span`: its arrays the operands, its other arguments and its `nondiff_argnums` bound
in), the custom rule outermost, so a differentiated call traces its rules alone, each where it runs, and the
check (SLOP026, row 4012) refuses `jax.custom_vjp` and
`jax.custom_jvp` anywhere in `src/dew` but `dew/nn/derivatives.py` (a reference to JAX's `custom_vjp_call_p` is
no application). Rows 4026 to 4030 declare each lane's sites; the check lands after them. Until a site is
declared, its custom call's bodies are rewritten as any glue is (§6.2), reorderings over its values held by
the dominance rule alone (§6.3).

### 1.4 Facts and rules

- **Facts about each value** (e-class analyses): its shape, and a bound on its magnitude where the program
  proves one (§6.3). Its dtype and parameters are the region's tables (`terms.Region`). Constant and sharded
  axes and a rounding class per value are not built: a hoisting rule reads a broadcast's laid axes, no price
  reads a link, and a rule set states its rounding class (§6.4). Error is measured per candidate, never held
  as a fact (§1.5).
- **Rules,** in declared sets in one place, each set tagged with the modes it runs in:
  - **linear algebra** (compile time and dev time): linearity of sums (`joined`), a laid factor hoisted out
    of a sum, distributivity, a product's bilinearity (a shared operand out of two products, a power of two's
    scale on either operand), a row or column scale moved through a contraction (a dequant's, or
    `(diag(1/t) E)ᵀ dO ≡ Eᵀ (diag(1/t) dO)`), and a positive scale out of a maximum or a minimum (`max(c x) = c max(x)` for `c > 0`, a literal's sign
    its fact): attention's derived step takes its raw products' maximum and scales it once, as the hand step
    does;
  - **identities** (compile time and dev time): commutativity, a saturation aid. A cast's widening round trip,
    a reciprocal square root's form, a negation negated and a magnitude's magnitude are XLA's algebraic
    simplifier's, which every compile runs. A value times its sign is its magnitude in exact arithmetic but not
    on a lane that flushes a subnormal product (the CPU's, `tests/certified.py`'s `Lane.flushes`: the product
    is zero where the magnitude keeps its bits), nor for a complex value; exp and log cancel only over a sign
    and a range the language holds no fact of. None of these is built;
  - **reuse** (compile time and dev time): a value the forward already computed. An equal term is one e-class
    and extraction prices a shared one once, so a value written twice is read once without a rule. A rule
    would relate a new value to another the program holds; over benchmark_step's and the parallelism matrix's
    step programs (2026-11-05, every program the pass saw, bodies included) none takes the square root and its
    reciprocal of one value or the exponentials of a value and its negation, a handful square a value by a
    product, and the only pattern of note is a divisor shared by several quotients (42 to 452 of 423 to 794
    divisions a suite). A shared reciprocal `a (1/y)` overflows where `a / y` is finite (a subnormal `y`) and
    makes a NaN of a zero `a`, so the dominance rule (§6.3) refuses it. Not built;
  - **decoupled reductions** (dev time): dependent nonlinear reductions broken with correction terms (the running-max
    rescale of online softmax, log-sum-exp merges and folds, moment merges);
  - **stable forms** (dev time; §1.5): `log1p` and `expm1`, and a sum's logarithm taken out of its exponent
    (the log-sum-exp fold), built; log-space divergences, compensated and widened sums, near-equal
    differences rearranged to cancel exactly, and products and quotients scaled to range are the generator's
    (parked, §6.11);
  - **linear scans** (dev time; §6.9), built as the derivation's own sets; **error-free transforms**
    (`error_free`, §6.9), not built;
  - **contracted** (compile time and dev time): a multiply its add or subtract reads first, as the fused
    multiply-add the compiler makes of the pair, a pricing aid that rounds as the form it replaces.
- **Rounding:** a rewrite that rounds identically (`exact`) always applies. One that reorders rounding (`reorders`:
  linearity, hoisting, distributivity) applies at compile time under default numerics and never under
  the accurate numerics (`precision="accurate"`). The compile-time pass estimates no error per program: a `reorders` rule is admitted
  under default numerics on the strength of its own gate, the default form's contract for its rounding class
  (§3), and a program's error is otherwise unknown, so nothing at compile time compares it to a parity. At dev
  time a reordering form is admitted under accurate numerics only where its sampled error meets the declared
  accuracy (§1.5), and it lands as reviewed code.

### 1.5 Error-aware extraction

Extraction picks the cheapest form whose error meets the precision policy's declared accuracy, as Herbie does for
floating-point expressions:

- **Error is a property of a candidate, not of an e-class.** The forms an e-class holds are equal in ℝ and differ
  in error, and a form's error depends on which form of each operand it reads, so the estimate belongs to a
  concrete candidate: an extracted DAG, every child chosen. Extraction proposes candidates in order of price, and
  each is measured before it is admitted (§6.5). Error is never an e-class fact.
- **Dev time measures each candidate** by sampling: the candidate evaluated in its dtype against the declaration
  at float64 on the gate's drawn and adversarial families (cancellation, wide ranges, near-equal operands, one-key
  rows), as check 1 measures error.
- **The declared accuracy** is the precision policy's (`dew.nn.precision`), read once:
  - default numerics: check 1 at the gate's parity (§3) against the form as written;
  - the accurate numerics (`precision="accurate"`): check 1 at the gate's parity against the most accurate candidate measured.
- **Among the candidates that meet it,** the generator takes the cheapest by time (§1.6). A candidate that
  misses it is never chosen, however cheap.
- **At compile time no candidate is measured;** admission is by rounding class alone (§1.4).
- **Stable rewrites** come mostly from dev time: `log(1 + x)` to `log1p(x)`, a KL in log space, a compensated or
  widened sum, `a − b` rearranged where `a` and `b` are near-equal sums. Two equal forms whose choice is decided by
  error alone, from the derived attention backward (feat/attention-derived 12f9398ba, check 1 against float64):
  - a scale moved through a contraction: normalizing before the contraction put causal queries-first dV at 1.64
    times the oracle's error, past 1.5, and the other side passes;
  - log-sum-exp folding: `exp(s − lse)` rounds `log(total)` into a value as large as the logits (1.6–1.7 times on
    mapped dK and dQ), while `exp(s − m) / total` is on par.

  They are the generator's first acceptance cases.

### 1.6 Cost

Each form is priced in time from the device facts Dew already owns, never from facts of its own:

- the program model reads `dew.nn.kernels.generation.gpu_target()` (memory bandwidth, dense peaks by precision,
  the part's measured latencies: a kernel's fixed seconds, a loop's and an in-kernel step's), so every process of a pool prices with the same numbers. No price reads
  links: every registered rule either keeps a reduction's axes and count or joins two reductions into one
  (`joined`; `linear`'s shared operand out of two products), which the device price already prefers, so a
  collective's bytes over `dew.training.distributed.link_figure` would only agree with it. A rule that moves a
  reduction onto or off a sharded axis brings that price with it;
- the kernel model is the templates' own (`dew.nn.kernels.templates.cost.walked`);
- a fact a price needs and no owner holds, such as a reduction's latency, is added to `GpuTarget` or its
  `Architecture`.

A region's seconds are computed once, from its composed terms over the chosen DAG (§6.5): bytes over bandwidth,
FLOPs over the dtype's rate, each kernel XLA's fusion makes of it (a reduction's
and a product's, each a fusion's root) at the part's measured fixed latency (its launch and first byte), so a
chain of dependent reductions pays a kernel's latency a link, and the span of a loop XLA lowers at its
iteration's latency (§6.5), combined as `templates.cost` combines a launch's. Row
4072's validation priced 1,580 profiled kernels: without a kernel's fixed latency an H100's XLA fusions came
to 0.02 of their measured time, with it 0.51 (p10 0.40). Prices are checked against measured profiles, not
fitted to them, and a rule set ships only with measured receipts.

## 2. Where it applies

- **Programs (compile time):** the pass runs where a function is jitted, so lowering, tracing, compiling and
  relowering all read one program:
  - An `OuterProgram` (`dew.nn.kernels.options`) rewrites the trace its jit makes (`_traced`,
    `dew.algebra.rewritten_program`), and its eager call runs `dew.algebra.rewritten(fn)`. Every outer program Dew
    builds gets the pass: `options.program` for evaluation and loads, `Objective.program` for the
    host-master step's forward and the evaluation, rollout and sample programs, and the serving kernel's step and
    decode programs, whose tuner lowers and compiles the same function.
  - The resident training step, `Trainer._jitted`'s `owned_program(step, …)` (`dew.training.trainer`), which every
    default `dew train` runs and the planner compiles through `compiled_if_it_fits`, is such an outer program, and so does every objective's step it builds, an RL update's included.
  - The host-master phases (`dew.training.transaction`) and the RL rollouts (`dew.objectives.rl.control`,
    `dreamer`, `dew.rl.simulator`, `dew.rl.flight`, `dew.rl.hyperlux`) jit theirs the same way: the rollout over
    an environment through `Objective.program`, the rest through `options.program` and `owned_program`, so none
    is built as an `OuterProgram` of its own (SLOP025).
  - A training step's traced jaxpr already holds the backward, since autodiff runs at trace time.
  - Rewrites stay within checkpoint (remat) regions and never change what a step saves.
  - The planner's gradient trace (`dew.training.footprint`) traces the same rewritten function before it prices
    the program, so the planner prices what it compiles.
  - `dew.algebra.rewritten(fn)` is a public transform, and the seam's switch (`dew.algebra.passes`, §6.7) lets the
    accuracy gate run Dew's own programs with the pass on or off per arm (§3).
- **Template hooks (compile time):** a hook is user code Dew cannot know in advance, a jaxpr, so it and its
  derivative pass through the same pass before the Pallas fuser places them in a kernel.
- **Template algebra (dev time):** a template's online, tiled and combine forms, forward and backward, are
  generated from a declaration (§6.9):
  - attention's score function, mask and reduction;
  - a recurrence's state update.

  The hand-written algebra modules are the reference a generated form is held to. It need not equal them. It
  replaces one when it meets both of the bible's bars (§4).
- **Stable numeric forms and surrogate bodies (dev time):** §1.3 and §1.5.
- **Fusion:** a search over algebraic forms, fusion boundaries and template compositions for a region of a step
  (a norm, a projection and an activation, for example), built from the toolkit's composed primitives and its work
  scheduler. Applied to a whole decoder layer, its output is a generated megakernel, checked in as a dev-time
  output.

## 3. Verification

All accuracy checks are the [accuracy gate](testing.md)'s, with no tolerance of their own:
- check 1 is `assert_as_exact` as it applies `tests/check1_statistics.py` today: each tensor's pooled RMS distance
  to the float64 truth against the reference's at `PARITY`, its worst row at `ROW_FACTOR`, and narrow rows at
  `NARROW_FACTORS`. This doc names those constants and never restates them, so it cannot drift from the gate.
- check 2 (bible §7, "never wrong"): where a rewritten operation has a certified bound (`tests/certified.py`),
  every element holds within it, at each rule's site and wherever the gate already runs check 2 for a program
  site.

- **Each rule,** by its rounding class, on the gate's terms for that class (testing.md's fast and accurate forms):
  - `exact` and `bounded` sets: the accurate form's contract, check 1 on every drawn and adversarial family;
  - `reorders` sets: the default form's contract, check 1 on drawn inputs, and on the adversarial families finite
    and keeping exact zeros.

  The rule's left side is the reference arm, the left side at float64 the truth, and the right side the
  candidate; `assert_certified` is added where a bound applies. Each rule's check 1 site names the rule.
- **Each program site the pass changes,** in one of two ways:
  - at a site the gate compiles itself: `assert_as_exact(rewritten(fn), fn, draw, truth_fn=fn at float64)`;
  - at a site running Dew's own programs (`dew_programs`, `as_given` trainer steps): the same arm under
    `dew.algebra.passes(on=True, target=…)` against `passes(on=False)`.

  The truth is the unrewritten program at float64 in both. The gate holds `passes(on=False)` by default for the
  reference arm and the float64 truth alone; the candidate runs as users run it, the pass on, priced at the gate's
  lowering target, as an arm is measured as its users run it (testing.md). A site that compares a program with its
  unrewritten self passes one function as both arms. The pass prices for the named target's facts (the generation
  the gate's lowering traces for), so a CPU lane checks the form a GPU of that target extracts.
- **Rewritten backwards:** `rewritten(jax.grad(fn))` at `tests/contract_support.py`'s edge draws, held to its
  finiteness rule against the unrewritten float64 gradient, and check 1 against `jax.grad(fn)`. A rule that only
  fires on backward patterns is caught there, as JAX's derivative of a rewritten forward never exercises it.
- **Marked regions:** a rule test injects each `Protected` transform and requires its residual to survive
  saturation and extraction; a `Surrogate` test requires its backward to stay the declared rule, not autodiff's.
- **Generated modules:** each ships its gate sites and its stability suite (bible §6); `generate --check` holds the
  checked-in output to the generator's.
- **Determinism:** one jaxpr in gives a bytewise-identical rewritten program out, compared as its lowered
  StableHLO, across fresh processes with different `PYTHONHASHSEED`s, checked on the multi-process CPU lane. Multi-controller JAX needs every
  process to compile the same program, and one process repeating itself cannot catch an ordering that follows
  object identity or string hashing. The e-graph is built in equation order and runs with one thread, and the
  prices read only pool-agreed facts (§1.6).
- **Compile time:** the pass's seconds per program are recorded beside lowering and XLA compile, in perf_gate and
  the eval, and measured on the 4B and 9B steps' compiles. It is a reported measurement, not a gate.
- **Report:** each program's rule classes that fired and the pass's seconds go in the one kernel report
  (`dew.nn.kernels.templates.report`, `report.compiled`).

## 4. What it replaces, and what it leaves

- **It replaces** custom derivatives that exist only to restructure algebra where its rules reach their form
  under the dominance rule (nGPT's join and normed hidden do not, §1.3), and hand-derived template algebra and numeric forms wherever a generated form meets both bars:
  - bible §4: at least as efficient on every scoreboard row and GPU type the route serves, at the derived default,
    in operator rows and the `benchmark_step` step;
  - bible §6: the template's stability suite on both forms, and check 1 and check 2 against the float64 truth.
- **It leaves:**
  - the templates' schedules, the toolkit and the families;
  - custom derivatives that implement kernels, and numerics and surrogate bodies except as §1.3 allows;
  - the planner's contract and the precision policy's meaning;
  - JAX, jaxlib and XLA, used through public interfaces only.
- **XLA keeps** fusion and code generation; it does not reassociate floating-point math by default, so chosen forms
  reach it intact.

## 5. Prior art

- egg and egglog (equality saturation);
- SPORES (sum-product rewriting for linear algebra);
- Tensat and TASO (verified tensor-graph rewrites);
- Herbie (rewriting for floating-point accuracy, error estimated by sampling);
- Neptune (fusing dependent reductions with correction terms, deriving FlashAttention-style forms);
- Mirage (joint search over algebra, schedules and kernels, with randomized equivalence checks);
- Constable (equality saturation inside XLA, costed against the real compiler's fusion);
- FlexAttention (variants as declared functions).

## 6. Interface

Draft, 2026-11-01. Every part of the work builds against this section, so a change to it is a change to the design.
The engine runs egglog 14.0.0 (MIT; wheels for CPython 3.12–3.14 on Linux, macOS and Windows), pinned exactly,
because its Python API is still marked as subject to change.

### 6.1 Modules

| Module | Holds |
|---|---|
| `dew.algebra.terms` | The term language and its conversion to and from jaxprs |
| `dew.algebra.rules` | The shape of each value a rule makes (`FACTS`, an e-class analysis; the magnitude bound is the region's, `terms.Region.magnitudes`, §6.3), `Rules` and `RuleFamily`, the registry and the sets registered: `identities`, `contracted`, `stable`, `linear`, `joined`, each declared as its rewrites with their instances (`reuse` and `error_free`, §1, are not built; `decoupled` and `carried` are the derivations' own sets, never registered) |
| `dew.algebra.linear_scan` | A recurrence's token update read as an affine map by rules, and its chunk form derived by identities and extracted under the work-span model (§6.9) |
| `dew.algebra.cost` | A node's work in the cost vocabulary (`work`, `templates.cost.Work`), the two schedules (`fused`, `walk`) that turn it into seconds, and the DAG cost models extraction reads (`model`, `stable`) |
| `dew.algebra.extract` | Extraction: egglog's greedy DAG (`cheapest`) and the exact one, its oracle (§6.5) |
| `dew.algebra.error` | Dev time: a function's candidate forms in order of price and the first its caller's admission takes (`choose`, §6.5) |
| `dew.nn.derivatives` | Below the engine, so every model module can import it: the marker primitive, its kinds, and the custom-derivative declarations (`declared_derivative`, `declared_tangent`, each of a kind, and `surrogate`, row 3271) |
| `dew.algebra.program` | The compile-time pass (`rewritten`), its switch (`passes`) and its report |
| `dew.algebra.derive` | A monoid walk's declaration (`Reduction`) and its derived forms; the derivation context every derivation keys on (`context`) |
| `dew.algebra.scan`, `dew.algebra.carry` | A recurrence's declaration (`Scan`) and its chunk forms, split into what every chunk computes at once and what its carry runs in sequence |
| `dew.algebra.forms` | What every derivation shares: the derived forms (`Derived`), their refusal, their evaluation, and the switch the templates take them behind (`derived`) |
| `dew.algebra.generate` | Parked (§6.11): the dev-time generator and its `--check` |
| `dew.algebra.bits` | Bit-level forms (§7): a declared GF(2)-linear transform of a stored word, its cheapest instruction form over the integer pipes and its proof |

### 6.2 Terms

`Term` is one egglog sort, `class Term(Expr)`. Each constructor is one jaxpr primitive of the language in §1.2, with
the primitive's parameters as primitive sorts:

- shapes, axes and permutations as `Vec[i64]`;
- dtypes as `String`;
- literals as `f64` together with their dtype.

```python
class Term(Expr):
    @classmethod
    def leaf(cls, index: i64Like) -> Term: ...                 # an input, or an equation outside the language
    @classmethod
    def literal(cls, index: i64Like) -> Term: ...              # by its index in the region's literals
    @classmethod
    def bundle(cls, roots: Vec[Term]) -> Term: ...             # a region's roots, extracted together
    def binary(self, name: StringLike, other: Term, params: i64Like) -> Term: ...  # add, sub, mul, div, max, ...
    def compare(self, name: StringLike, other: Term, params: i64Like) -> Term: ...  # lt, le, eq, ne, ge, gt
    def unary(self, name: StringLike, params: i64Like) -> Term: ...  # neg, exp, exp2, log, log1p, sqrt, ...
    def select(self, false: Term, true: Term) -> Term: ...
    def convert(self, params: i64Like) -> Term: ...
    def shaped(self, name: StringLike, params: i64Like) -> Term: ...  # broadcast_in_dim, reshape, transpose, ...
    def reduce(self, monoid: StringLike, axes: Vec[i64], params: i64Like) -> Term: ...  # sum, max or min (§6.3)
    def dot(self, other: Term, params: i64Like) -> Term: ...   # dot_general
    # the derivation's readings (§6.9), rewritten into the language and never extracted themselves
    @classmethod
    def state(cls, index: i64Like, side: i64Like) -> Term: ...
    def held(self, side: i64Like) -> Term: ...
    def stepped(self) -> Term: ...
    def merged(self) -> Term: ...
    def finished(self) -> Term: ...
```

- A term's `params` index the region's table of its equation's parameters (`terms.parameter_key`'s structural
  key, `Region.params`), so a rule reads a parameter as a fact of its own (a product's dimension numbers, a
  literal's value) and an equation's every parameter rebuilds it as traced.
- The error-free transformations (`two_sum`, `two_product`, the correctly rounded divide's significand) are
  `Protected` spans (§6.10): one leaf of the region each, no rule touching them, rather than a `semantic`
  constructor. An `iota`, a prefix sum (`cumsum`, `cummax`) and every other primitive outside the language are
  leaves; a recurrence's prefix is the scan derivation's (`linear_scan`, §6.9), which reads the token update,
  not a prefix term.
- `terms.region(closed) -> Region` converts a window's equations (`program._windows`):
  - every equation outside the language becomes a `leaf` for each of its outputs;
  - a `Protected` span (§6.10) is one marked call, a `leaf` for each of its outputs, its equations kept as
    they are;
  - every value the window reads but does not compute becomes a `leaf`.
- `terms.form_value(region, form, values)` evaluates an extracted form over its leaves' values, each equation
  rebuilt from its interned parameters, which a splice traces into the window's place (`program._splices`).
- A surviving custom-derivative call (`custom_jvp_call`, `custom_vjp_call`, in a program no transform has traced
  through) is a region of its own: its `call_jaxpr` is rewritten in place (a declared site's body is a marked
  call of its kind within it, read as §6.10 reads one), and the call, its rules and their derivative programs
  are rebuilt as they were, so the custom derivative stays the site's. Its
  rules are rewritten where autodiff materializes them, as marked bodies (§6.10).
- A region is a maximal set of equations of the language. It never crosses:
  - a checkpoint (remat) boundary;
  - an opaque equation;
  - a marker (§6.10): a `Protected` span is a leaf, and a `Surrogate` or `KernelGlue` span a region of its own;
  - a `scan`, `while`, `cond` or `pjit` body (those are regions of their own).

### 6.3 Facts

Each fact is an egglog `@function` from `Term` to a primitive sort, and its rules derive it from its
arguments' facts (`rules.FACTS`, the magnitudes' rules beside the reorderings they hold).

| Fact | Sort | Join | Read by |
|---|---|---|---|
| `shape(t)` | `Vec[i64]` | the first stated (every member's is one shape) | every rule that names an axis |
| `log2_bound(t)` | `i64`, a power of two over the value's magnitude | the first stated (each a true bound of every member) | the dominance rule: a reordering's new intermediates |
| `reach()` | `i64`, the narrowest float's largest power of two | — | the dominance rule |

A value's dtype and its equation's parameters are the region's tables (`terms.Region`), read by the rules
that need them. Not built, and read by nothing: constant axes (a hoisting rule reads a broadcast's laid axes),
sharded axes (no price reads a link, §1.6), a walked axis (the derivation's own `walked` fact, `derive.py`),
per-token, decay and structure facts (`linear_scan`'s affine forms hold them), and a rounding class per value
(each rule set states its own, §6.4).

The magnitude bound (`log2_bound`, `terms.Magnitude`): a number, `|x| <= 2**log2`, proved from the program's
constants, its declared ranges and its equations, or none. The region states it as it traces a program (`terms.Region.magnitudes`). A
constant's bound is its largest element's magnitude (a complex one's modulus), and a declared operand's is its
range (`scan.Scan.ranges`, `derive.Reduction.ranges`, by input, the program's constants apart: l2-normalised
keys 1, a sigmoid's gate 1, a recurrence's decays 1). An operand with no declared range, and an equation
outside the language, has none. Each equation in the language composes its operands' bounds: sums take twice
the larger, products add their powers, contractions and sums along an axis add their length's, and a bounded
function (`tanh`, `sin`, `logistic` of a real operand) is 2, one power past the reals' 1 because no native
implementation is certified within its range, a comparison or `sign` 1, a complex operand's sine
or tangent none (`sin(pi/2 + 89j)` is 2.2e38), and a rounding to an integer (`floor`, `ceil`, `round`) its
operand's or 1 (`ceil(2**-126)` is 1). Each bound is its output dtype's too: a boolean is at most 1 whatever
made it (a tiny float converted is 1); an integer at most its dtype's range, within which a sum or a product
wraps, and at that range for a bitwise operation and where an unsigned result may wrap from below zero
(`uint32(0) - 1`, a negation, int32's -1 converted: `2**32 - 1`). A conversion keeps a value at most 0 only
between signed dtypes. An exponential of an unbounded operand, or a quotient, has no
bound; an exponential of an operand at most 0 is at most 2, its native implementation's rounding given one
power past 1 (`ex2.approx`'s error is more than an ulp of 1). Every bound holds under its arithmetic's
contract: IEEE rounding to nearest, a contraction's mode one that rounds each operand once (its real bound)
or splits it into parts (bf16 x3 to x9, TF32 x3, `Precision.HIGH`: one more power), and none for a mode
whose rounding no source states (FP8 fast accumulation).

The dominance rule decides which reorderings fire. Its contract is bible §6: no infinity where the reference is
finite. A reordering is admitted only when every new intermediate it makes has a number bound under the
program's reach, the narrowest float's largest power of two, taken exactly (`math.frexp`: float64's `log2` of
its largest rounds to 1024) (`rules.reach`; `rules.dominated`): a number bound
under the float's largest is a finite value's wherever the program's values are. A value no number bounds
admits nothing: the reference's value may sit far below any bound over it while the new intermediate
overflows: `0.5 x + 0.5 y` at `x = y = 3e38` is finite, and `0.5 (x + y)` is not. Each rule computes its new
intermediates' bounds from its operands' (`rules.log2_bound`) and states them for the rules after it
(`rules.made`); a new reduction bounds its new sum of terms and itself, its length's power above them
(`rules.sum_length`): `sum(x) + sum(y)` of `x = [2**126, 2**126]` and `y = [2**126 - 2**102, 2**126 - 2**103]`
is float32's largest, while `sum(x + y)` overflows. A scan's solve inverse, a prepared operand, is bounded by
`(1 + a)^(C - 1)` over a chunk of `C` tokens, `a` an overlap's bound: the corrections' elementwise bounds'
product times the keys' count it contracts (two tokens of 16 unit keys overlap by 16), or a declared overlap
(unit-norm keys under a write gate at most `b`: `b`); a bound past float64's range is none.

So `a*x + a*y -> a*(x + y)` fires only where `x` and `y` carry declared ranges that keep their sum under what
the program reaches. The delta rule's carry does not distribute its solve over `U - W h`, which is fla's WY
form (two state products a chunk where the derivation keeps three): the values have no declared range, so
`L U` is no number bound. A dequant scale folds out of a matrix product where its codes are declared (FP8's
448), and not over unbounded operands. Error-aware extraction then checks precision against float64 (§1.5).

Notes on the table:
- Every fact is invariant under equality, as congruence requires. Error is not: it is measured per candidate
  (§1.5, §6.5).
- The inputs' facts come from the traced avals, their shardings, and a declaration's roles (§6.9).

A reduction names its monoid in the language (`Term.reduce`'s name), from the closed set the language holds
(`terms.REDUCTIONS`: `sum`, `max`, `min`). There is no open registry: the derivation reads each monoid's
algebra in its own rules (`derive.py`'s `summed` and `peaked` facts, the walked merge's `max` and `add`, the
softmax's rescaling by `exp(k_carried - k_new)`) and its identity in `derive._identity` (a sum's zero, a
maximum's least value), so a new reduction is a new name in `terms.REDUCTIONS` with the rules and the identity
that read it, each held by the laws (`tests/test_algebra_laws.py`). An open `Monoid` registry, whose
combine and identity were jaxprs a rule could read, was the earlier draft: no reduction the templates take
needs one, and a monoid's rules cannot be written from its combine alone (the softmax's rescaling is a rule
about `exp`, not about the monoid).

### 6.4 Rules

```python
@dataclass(frozen=True)
class RuleFamily:
    name: str                                # one rewrite of a set
    rules: tuple                             # its egglog rules, each direction and layout of it
    instance: Instance                       # the rewrite written as JAX, its accuracy test's (§3)

@dataclass(frozen=True)
class Rules:
    name: str                                # "identities", "contracted", "stable", "linear", "joined"
    rounding: Literal["exact", "reorders", "bounded"]   # the rounding class each of its rewrites gives
    ruleset: egglog.Ruleset                  # its families' rules (`Rules.declared`)
    receipt: str                             # the measurement it ships with: a ledger row and its receipts
    families: tuple[RuleFamily, ...] = ()
    dev: bool = False                        # dev time alone (stable): each candidate's error measured first
    aid: bool = False                        # only aids saturation (identities, contracted)

RULES: Mapping[str, Rules]                   # every registered set, by name
def register(rules: Rules) -> None: ...
def admitted(accurate: bool, *, dev: bool = False) -> tuple[Rules, ...]   # the sets numerics and a mode allow
```

- A set's rules are egglog `rewrite`/`birewrite`/`rule` objects in its `ruleset`.
- Each rule's check 1 site names the rule (§3).
- A rewrite's result carries its rounding class (§6.3).
- At compile time, under default numerics, the compile-time sets are admitted; under the accurate numerics
  (`precision="accurate"`), only the `exact` ones, since the pass estimates no error. The pass reads the run's
  numerics where it traces (`generation.numerics`, in JAX's trace context), so a program held across the two
  traces afresh.
- At dev time every set is admitted, and the declared accuracy (§1.5) decides which forms extraction may take.

### 6.5 Cost and extraction

```python
def work(graph: EGraph, node, region: Region, held: dict | None = None) -> templates.cost.Work: ...
    # a node's work alone: device and shared bytes at its dtypes' sizes, tensor-core and CUDA-core FLOPs by
    # mode, operations, exponentials, conversions, and the kernels XLA's fusion makes of it

class Schedule(Protocol):            # how a chosen DAG's composed work becomes seconds
    def pieces(self, done: Work) -> tuple[float, ...]: ...
    def spanned(self, done: Work) -> float: ...
def fused(facts: GpuTarget) -> Fused: ...  # the compile-time pass: XLA's fusion
def walk(facts: GpuTarget) -> Walk: ...    # a template's interface: the tile walk

def model(region: Region, schedule: Schedule) -> DagCostModel[Seconds]: ...  # egglog's DAG cost model
def cheapest(graph, root, model, *, exact=False, checked=False): ...  # extract.py: greedy-dag, each shared node once
def extracted(graph, root, model): ...          # the exact extractor (a MILP), the oracle under `checked`
```

- A cost model gives a node's work alone. Extraction composes it over the chosen DAG: bytes and FLOPs add up
  with each shared e-class counted once, and a span is the largest any chosen node states (`extract.Level`'s
  peak), extraction's composition being additive; a derivation that prices a path's length composes it itself
  (`linear_scan`'s candidates, their critical path). Seconds are computed once from the composed work, as
  `templates.cost` combines a launch's, so a span stays a latency and never becomes a count.
- A candidate is admitted at compile time where every rule set it drew on is (§6.4). At dev time
  `dew.algebra.error.choose` proposes candidates cheapest first (the form as written, the cheapest and the most
  stable extractions, and each root's variants) and takes the first its caller's `admits(candidate,
  reference)` passes, the gate's check of the two against the float64 truth (§1.5): under default numerics
  against the form as written, which is always a candidate, so one is always admitted; under the accurate
  numerics against every other candidate, which may admit none, and then nothing is chosen. No module calls
  `choose` yet (the generator is parked, §6.11).
- Extraction is egglog's `greedy-dag` (`dew.algebra.extract.cheapest`), each shared node priced once; its
  exact counterpart (`extract.extracted`) is the oracle. The exact extraction finds the roots' forms whose nodes
  together cost least as a mixed integer program, solved by HiGHS through SciPy: one binary choice per node; a
  used class takes one node, a used node's child classes are used, and an order over each cyclic class keeps the
  choice a DAG. The nodes and their prices are the model's own, read through one `greedy-dag` pass that records
  every node it prices, and the chosen DAG is extracted by egglog with every chosen node free. Over the
  algebra's and the templates' test derivations it priced no template's form below greedy's and took twice its
  time, so greedy stays the default. The test suite holds each derivation's greedy choice to the exact optimum
  (`dew._cache.Passes.checked`); a declaration greedy prices dearer is refused there until it is extracted
  exactly (`derive.Reduction.exact`).
- A price states itself as levels compared in order (`extract.Linear`). Each level is the larger of its
  pieces, each a sum over the chosen nodes, and its nodes' largest peak (a span), plus what the nodes add
  beside it. A schedule's seconds are exactly such a level (`Schedule.pieces`, `spanned`), and its tiebreak
  the next. Each level is solved in turn with the ones before it held at their optimum; a level whose
  additions are infinite on every DAG (a reading the derivation left) ties, as its prices compare.
- `fired` comes from the nodes each set's run added, recorded between runs of the sets.
- The facts are the owners' (§1.6): `gpu_target()` for bandwidth, peaks by precision and latencies, so every
  process prices with the same numbers. A fact no owner holds is
  added to `GpuTarget` or its `Architecture`, never to the engine. Prices are checked against measured profiles,
  never fitted to them; a form whose price and profile disagree is a finding against the cost model.
- One cost vocabulary, two schedules. Every price is told in one vocabulary, the toolkit's `templates.cost.Work`:
  an operation's work by unit (tensor-core FLOPs by mode, the 16-bit mode's, TF32's and FP8's, `fp8_flops`;
  CUDA-core fp32 operations, a product at `Precision.HIGHEST` among them; SFU exponentials; conversions) and its bytes by
  level (device memory, shared memory). The algebra's terms price through it (`cost.work`: a product's FLOPs in
  its mode, from its operands' dtypes and its precision as the product declaration reads them, a dot
  algorithm's passes counted, and its operand bytes; a reduction's operations; a special function's
  exponentials, a conversion's conversions, any other elementwise node's operations, a layout change nothing;
  a leaf's bytes, read once). A template's instructions price through the same record. No per-operation price
  exists anywhere else.
- A schedule turns summed work into seconds over `GpuTarget`'s facts, the one source of device facts, each
  unit's rate from them (`templates.cost.rates`); a unit a target states no rate of (a TPU's vector and
  transcendental units, which no primary source states) is priced by none, and a form is cheaper only where it
  takes fewer seconds and no more of that work (`cost.cheaper`), so it is cheaper whatever the rate. A process
  of no accelerator prices as the interpreted templates' H100, a GPU no row names by its family's and its
  runtime's facts (`cost.target`):
  - XLA's fusion (`cost.fused`), for the compile-time pass over a program, error-aware extraction and the
    derived scan's carry split (`carry.split`, whose carry runs as a loop XLA lowers, a chunk an iteration, its
    span priced at that loop's step). A program's operands are read from device memory once, however many
    reductions read them (XLA's fusion reads a sibling reduction's inputs once), and elementwise work fuses
    into its consumer. Each unit's work runs at the whole device's rate, in series with the memory traffic.
  - The template's tile walk (`cost.walk`, `templates.cost.walked`), for a form a template's algebra interface
    takes (`derive`). A tile's work runs on its program with its units overlapped, the longest
    unit's time. Its operands are staged by the walk, so its device bytes are the launch's.
- Either schedule's time is the work-span model's (Brent's): the larger of a form's work at its units'
  rates and its span, its critical path's dependent steps, at one step's latency. On XLA's path a step is
  an iteration of a loop XLA lowers, its launches and its predicate (`GpuTarget.step_seconds`, each GPU's
  measured as the slope of a loop's time over its iterations: an A100's 5.96 µs, an H100's 4.93 µs, an RTX
  PRO 6000's 3.68 µs); a token loop's tokens and a substitution's rows are a step each. On the template's
  path a step is a dependent reduction inside a program, measured as one (`GpuTarget.kernel_step_seconds`:
  a Pallas program's 128-lane reduction read by the next, an A100's 225.3 ns, an H100's 142.7 ns, an RTX PRO
  6000's 137.7 ns). The algebra's terms state no span of their own, extraction composing a span as its largest
  node's (`extract.Level`), so a derived tile form's span is the derivation's to state. A whole launch's step,
  its tile's load and products where its ring does not hide them, is the launch model's (row 4123).
- Every rule is the pass's own, applied wherever its cost model prices the rewrite cheaper, with no
  condition on XLA: where XLA would make the same rewrite the result is the same, and where XLA's factoring
  and reassociation are held to integers, powers of two or fast math, the pass's fire on floats wherever the
  range facts admit them. A study of the pinned pipeline (an H100's and the CPU's optimized programs with the
  pass off and on) informs the cost model and conditions nothing: XLA's programs keep every
  rewrite the reordering sets made where they fired (a tensor factor's `a x + a y`, a shared product
  operand's `a@x + a@y` and `x@a + y@a`, a laid factor out of a sum, two sums joined), and its
  `AlgebraicSimplifier::HandleAdd` factored `a c + b c` itself by a power-of-two constant. Commutativity
  only aids saturation (`Rules.aid`), changing no program on its own, so a pass whose admitted sets are all
  aids (the accurate numerics') does not run.
- The pass saturates and extracts each window cluster in an e-graph of its own (§6.7); two clusters whose
  union passes `CLUSTER` stay apart though they share an equation, and a value both yield takes the first's
  form. One e-graph of a program's every window grew its extraction with the program: a gemma3n training
  step's took past 500 seconds, 14 million cost additions in one extraction. A term keys a dictionary by its
  declaration (`terms.key`), not its text, which egglog formats through black at about a millisecond a term.
- Extraction for a template's interface (`derive`) prices candidates through the tile walk; the compile-time
  pass and the derived scan's carry split price them through XLA's fusion. Each shared term is priced once either way.
- The tile walk reads no registers or rings when pricing a candidate: both are properties of a whole chosen DAG
  at a tile size the derivation does not yet know. The template's register model (`MosaicForward.registers`,
  `_hook_live`) sizes its tiles to the form chosen, and its compile gate refuses a spill.

### 6.6 Saturation

```python
@dataclass(frozen=True)
class Saturation:
    growth: int             # nodes allowed a region, per equation of it
    depth: int              # iterations allowed, per level of the region's longest chain

    def limits(self, region) -> tuple[int, int]: ...   # (nodes, iterations) for this region
```

- The limits are derived from the region's size where they are set (`limits`): nodes at `growth` times its
  equations, iterations at `depth` times its longest chain. egglog has no node limit of its own, so the count is
  checked between iterations.
- `growth` and `depth` are measured per mode. At compile time they sit above the largest the compile-time sets
  reach at their fixpoint on Dew's step programs: none of 756 windows (a UNet's fsdp2 x tensor2 matrix cell,
  benchmark_step's cpu-smoke steps) reached either of 64 and 4, the largest using under a quarter of each,
  and none priced lower at four times both (`SATURATION`'s receipt). At dev time they are measured on the
  declarations when the generator resumes. They are re-measured when a set changes.
- A region that reaches a limit keeps the cheapest form found so far, and the report says so.
- Sets run in registry order, one iteration each, until no set changes the graph or a limit is reached.
- The e-graph runs with one thread and is built in equation order, so the same input gives the same output in any
  process (§3).

### 6.7 The program pass

```python
def rewritten_program(closed: jax.extend.core.Jaxpr, *, saturation: Saturation) -> jax.extend.core.Jaxpr: ...
    # a jit's own trace, rewritten where the switch is on; the same inputs, outputs, avals and shardings, and
    # `closed` itself where no rule changes it (the numerics read from the trace context, the prices from
    # `cost.target`)

def rewritten(fn: Callable, *, saturation: Saturation, static_argnums=(), static_argnames=()) -> Callable: ...
    # fn traced over its dynamic arguments, its static ones bound first as jit binds them, rewritten as
    # the seam would, and evaluated, its outputs' pytree as fn's; `dew.algebra.rewritten` and
    # `dew.algebra.rewritten_program` bind `SATURATION`

@dataclass(frozen=True)
class Passes:                         # dew._cache
    on: bool = True                   # the pass runs
    target: str | None = None         # the GpuTarget whose facts it prices for; None: the process's own
    checked: bool = False             # each derivation's greedy extraction held to the exact optimum

ALGEBRA = jax.make_user_context(default_value=Passes())   # dew._cache, in JAX's trace context

@contextlib.contextmanager
def passes(on: bool = True, target: str | None = None) -> Iterator[None]: ...   # sets ALGEBRA

@dataclass(frozen=True)
class Rewrite:                        # one region's
    region: terms.Region
    picked: Mapping[Hashable, terms.Term]   # each root's cheaper form by its key; empty where none is
    report: AlgebraReport
```

- `rewritten(fn)` binds `fn`'s static arguments as the jit around it does (the same `static_argnums` and
  `static_argnames`), traces it over the dynamic ones with `jax.make_jaxpr`, rewrites the result with the
  compile-time sets, and returns a function that evaluates the rewritten jaxpr (`jax.extend.core` and
  `eval_jaxpr`) and rebuilds the output pytree. Where it is jitted, the jit options and shardings stay the
  program's own; a static argument keeps the program's Python control flow concrete (a rollout's length, a
  policy's `random`).
- The pass sits where a function is jitted (§2): `OuterProgram._jitted` jits `rewritten(fn)`, so `lower`,
  `trace`, `compiled` and every relowering read one program; `Trainer._jitted` jits the rewritten step, the
  host-master phases and the rollouts theirs. `footprint`'s `make_jaxpr` traces the same rewritten function.
- A region is a window cluster (`program._windows`): each head a rule may lower the price at with the
  equations of the language that make its operands, `WINDOW` deep, windows that share an equation joined up to
  `CLUSTER` equations, each saturated and extracted in an e-graph of its own (§6.5), its cheaper roots' forms
  spliced where their equations stood. A region whose only bounded terms are its literals builds no e-graph
  (`rules.reorderable`): every reordering makes its new intermediate of an operand the program bounds (§6.3),
  and commutativity alone leaves every price as it was. Most of a training step's regions are its own math
  over unbounded values: a gemma3n step's 2879 each fired commutativity alone.
- A region's rewrite is derived once per structure in a process (`program._derived`, bounded by
  `WINDOWS_HELD`): keyed by what its derivation reads (`program.structure`: its values' avals, their
  placement included, each equation's primitive, parameters as the region holds them, protection,
  operands by position or literal value, and the values it yields), the numerics' forms, the saturation
  and the target it is priced for.
  Every layer of a stack and every compile of one step (the planner's candidates, a resumed run's
  relowering) take the first derivation, spliced into each program apart.
- `passes` is the seam's switch. `ALGEBRA` lives in JAX's trace context, so every jit and every executable map
  that keys on `trace_context` (jit's own, `OuterProgram._keyed`) retraces when it changes: a program already held
  never runs a form traced under another switch or priced for another target. The gate opens it per arm: off for
  the reference and the float64 truth, on for the candidate, priced at a named target (§3). Outside any `passes`,
  the pass is on and prices for the process's own GPU. Tests hold a retained step to a new executable when the
  switch or the target changes.
- The admitted set names key the program beside its compiler options, never inside them (XLA refuses a key it
  does not know): the switch's state is in the trace context, and the sets are fixed per Dew version, so a
  rewritten program and an unrewritten one never share an executable.

### 6.8 The report

`AlgebraReport` joins the one kernel report (`dew.nn.kernels.templates.report.rewrites`), beside the kernels'
misses, as each program traces; a relowering's rewrites join it only where its replacement compiles. A compiled
step records its search's rewrites with its kernels (`StepCompiled.plan`'s `kernels`). It holds:
- the regions rewritten;
- the price before and after;
- the rule sets that fired, with counts;
- the saturation's iterations and node count, its limits, and whether a limit stopped it;
- the pass's seconds, beside lowering's and XLA's compile's.

### 6.9 Derived template forms

A walk's derivation (`derive.py`) runs through the e-graph. The declaration, traced at one tile, enters as
terms. Facts saturate first: each value's walked axis, which values read no walked reduction, which reductions
sum or take the maximum along the walk, and which values scale as `exp(-k)` with a function `k` of walked
maxima. The derivation's readings (`Term.stepped`, `merged`, `finished`, `held`) distribute over the language
under its rules (`decoupled`). At a walked reduction a monoid's identity holds, a sum that scales with a maximum
rescaled by `exp(k_carried - k_new)` first. A carried state is `Term.state(index, side)`. The engine's sets the
numerics admit saturate in the same e-graph, so a form is extracted from the language and the engine's rewrites
together, under the program's cost. The readings cost without bound: a value no rule rewrites leaves its form
unpriced, and the declaration is refused. Each tile shape's rewrite joins the kernel report (§6.8).

A scan's chunk forms are split by extraction (`carry.py`). The chunk function, traced at one chunk, enters
the e-graph with the values that read the state marked. The products' linear algebra saturates over them: a
transposed operand's product as a plain product of its transpose, a product distributed over a sum that reads
the state, two products associated so the state is last, two products of one operand factored. The carried
outputs are extracted under a two-part price, the work that reads the state first and the rest second, so the
carry reads the state through the fewest products; the emitted outputs take the program's cost. The forms'
largest terms that read no state are the walk's prepared values (`Derived.prepare`), computed for every chunk
at once; the carry alone runs in sequence. The delta rule's carry is one product of the state by a matrix the
chunk prepared, its solve against the state off the carry.

The chunk function's algorithm is derived by rules (`linear_scan.py`); nothing of its form is written. The
token's update and readout, traced at one token, enter an e-graph as their algebra over the state
(`Token`): scalars, vectors along the keys or the values laid as columns or rows, products, sums,
exponentials, and a vector's contraction with a matrix over the keys. A value free of the state whose
equation the language has no constructor for is a leaf. The algebra's identities saturate: products and
sums commute and associate, products distribute and factor, and a contraction is linear in each operand.
Where the update is affine, its class then holds the form: a decay `exp(z)` (or `exp2`) on the state, a
scalar's or each key's, a rank-one input `B U^T`, and corrections along `B` that read the decayed state
(`Affine.decayed`, `corrected`). The readout's form is a contraction with a vector plus a free part
(`Affine.readout`). An update or readout that reaches no such form is refused. The affine update then
enters a second e-graph as its chunk's token loop, and the identities of affine scans hold every equal
chunk algorithm side by side:
- `wy`: the corrections' loop is a plain loop over the inputs that a unit lower-triangular system solves;
- `dual`: a plain loop is its outputs' masked quadratic form plus its state's decayed sum;
- a solve is its inverse by doubling or a forward substitution;
- a solve is linear in its given values, so the part of the system that reads the state is off it.

The cheapest under the work-span model is the chunk's form, which `scan.py` evaluates and `carry.py`
splits. At a kernel's chunk on an A100 or an H100 (64 tokens, 128 keys and values) the extraction is the
compact WY form with its inverse by doubling, the hand-written chunk's form. Both rule sets are proven
sound over the reals from their declarations (tests/test_linear_scan.py).

A derived algebra's laws are proven before its forms ship, in the suite (tests/algebra_laws.py): identity, associativity, consistency
(a tile folded into a state is a walk of it merged), and the declared function (a walk of two tiles finishes
as the function of them joined, an output along the walk each tile's own joined). Together they make a
walk's result independent of its chunking and its merge tree. The proof is over the reals: the forms traced
and evaluated over SymPy symbols, a float's largest read as a finite value below every other (the empty
state's anchor), each comparison decided both ways and each maximum by each order of its arguments, and
the two sides' difference zero as a rational function of exponentials. A declaration that carries no state
(a tile's) is certified tile by tile. Extraction is priced, so a declaration's forms can take another
structure at another tile. The suite enumerates every tile the mainloops' derivations emit for an A100, an
H100 and an RTX PRO 6000 and certifies each distinct structure at a small tile of it, each axis's extent
past one and the kernel tile's equal extents equal (tests/test_algebra_laws.py). No proof runs at trace
time.

Behind the one switch (`dew.algebra.forms.derived`, in JAX's trace context), the templates take their derived
forms where the declarations state the walk:

- The attention forward. Every mainloop's step, merge and finish take the declaration's forms, with the sums
  in the mainloop's own accumulator and every product through its own issuer. The declaration covers
  natural and base-2 exponents, keys-first tiles, initial states and score hooks; its products sum in fp32.
  A dropping call's step keeps the template's: the declaration narrows a dropped weight unnormalized, past
  fp16's largest at a rate near 1, where the template's scaled parts keep it. A walk's merge and finish name
  the forms they read by the walk's tiles (`Derived.merge`, `finish`: a state's values alone name no walk, a
  bf16 walk's fp32 sums being an fp32 walk's too); the attention state's merge and finish read the same
  forms at any keys a tile or values' dtype, so the template names its base's walk at the state's rows and
  width (`attention_algebra._stated`), whatever walk came before, over every row of states of any leading
  axes (a split decode's, `_rows`).
- The attention backward. The plain-JAX mainloop walks dQ over the keys and dK and dV over the queries by
  the backward declaration's forms, each launch's tile reader and hook gradients shared with its
  hand-written algebra; its dP sums in fp32, its score cotangents are masked where the tile's queries see
  no key, before and after a hook's derivative (`softmax_cotangents`, `pulled`: an empty query's product or
  a padded query's hook derivative need not be finite), and they take the scale before their products, the
  reference's order. Every Mosaic backward launch takes its tiles' weights and score cotangents from the
  tile declaration (`softmax_cotangents`) over its own products. Both cover hooks' slopes, dropout, the
  first pass's D and initial states.
- The recurrence. The plain-JAX path runs `derive_scan`'s chunk forms with their carry split
  (`derived_rule`), differentiated at every order (the decays' and the solve's rules through
  `generation.linear_tangent`: reverse mode their protected rules, forward mode their own tangent), its
  carries typed as every operand varies, its decays' prefix restarting at a token whose decay vanishes in
  its dtype (`dew.nn.precision.vanishing`, a segmented compensated scan: a span through it is zero, and no later span's sum
  holds its exponent: a prefix of -2**127 twice is -inf, one of -2**70 loses a later -1), a pair no decay
  joins selected away before its decay multiplies it. The Mosaic mainloop takes its system's pairs and its readout's scores from the derived forms of tile
  declarations (`recurrence_algebra.pairs_of`, `scores_of`) over its own products and its compensated
  prefix's decays, the one path: the derived forms are the hand-written ones' arithmetic, which they replace.

The template's own forms run in three places:
- the accurate forms (the compensated exponent, the largest weight held to 1, raw logits, the anchor test,
  the keyed sums);
- compensated running sums and the headroom guard's shifted sums;
- representations the template keeps: the compensated prefix, the inverse by doubling, the launch structure.
Those are rounding and representation choices the fast class does not make. The replacement rule holds:
a derived form replaces a hand-written one only where it states the walk, passes check 1 and is no slower.

**Drift to fix.**
- The accurate forms (row 4228, theme AL03) are the declaration's numerics, not rewrites: the accurate
  numerics admit only the `exact` sets, so no rule could reach them. A declaration states its exponential's
  implementation (`Reduction.exponential`) and the rescale a carried state takes from one anchor to the next
  (`Reduction.rescale`, the language's binary `rescale`, `terms.RESCALE`, which no rule rewrites), and may
  hold its largest weight to 1 by a select (`exp(where(x < b, x - b, 0))`, `b` the broadcast of `x`'s own
  maximum). A compensated walk's running sums are the mainloop's own additions over the derived forms
  (`Derived.step`'s `stepped`, the mainloop's one step over each sum's parts, `Accumulated` and `Added`). The template's hand forms run until it
  takes these (row 4228); the guard's shifted sums follow, its control flow the template's. Error-aware
  extraction (§1.5) waits for a cell whose cheapest form fails check 1 where an equal one passes: row 3310's
  was the test's schedule, not a form.
- The chunk's decays (`scan._decays`: the compensated prefix and its exponentials) and a gram's evaluation
  by decay kind (`scan._paired`) are written, not extracted. The compensated prefix is the declaration's
  numerics, not a candidate: a plain prefix loses about 1e-6 of a deep decay, which no price weighs.
- A Mosaic launch built from the derived chunk forms' terms is the generator's work (§6.11). Until then the
  Mosaic recurrence takes derived tile algebra inside its own launches.
- The derived forms run at trace time.

Derivation runs at trace time today (`derive.derive`, `scan.derive_scan`, where a template calls them); the
design moves it to dev time, in the generator (parked, §6.11), its output a checked-in module. A declaration states
semantics only: plain JAX on whole operands, or token functions along a walked axis. It never states a
hand-written form.

```python
@dataclass(frozen=True)
class Reduction:                     # attention, and any walk whose state is a monoid's
    function: Callable               # the math on whole operands: softmax_keys(where(seen, score(q, k), -inf)) @ v
    operands: Mapping[str, Axes]     # each operand's named axes: q (rows, d), k (keys, d), v (keys, e)
    walked: str                      # the axis the kernel walks in tiles: "keys"
    exponent: Literal[2, "e"]        # stated by each template per call (a scale folded with log2(e)), never found
    ranges: Mapping[str, float] = {} # operands' magnitude bounds by name (§6.3)
    exact: bool = False              # extracted exactly where greedy prices dearer (§6.5)
    exponential: Callable | None = None   # (x, scale=1.0) -> exp(x scale), its exponential's implementation
    rescale: Callable | None = None  # (carried, now) -> exp((carried - now) c), c its weights' literal or 1

@dataclass(frozen=True)
class Scan:                          # a recurrence
    update: Callable                 # (h, x_t) -> h, read as h <- A_t h + B_t U_t^T by rules
    readout: Callable                # (h, x_t) -> o_t
    operands: Mapping[str, Axes]
    walked: str                      # "time"
    state: tuple[tuple[int, ...], jnp.dtype]   # the state's shape (keys by values) and dtype
    reset: str | None = None         # an operand whose tokens start a document
    ranges: Mapping[str, float | tuple[float, float]] = {}   # bounds, "state" and "overlap" among them

class Derived(NamedTuple):           # dew.algebra.forms
    init: Callable                   # shapes -> the empty state
    step: Callable                   # (state, tiles) -> state; a scan's (state, prepared)
    merge: Callable                  # two states of disjoint walks joined: split walks, the work scheduler's fix-up
    finish: Callable                 # (state, operands) -> the outputs the walk reduces
    mapped: Callable                 # a tile's outputs along the walked axis; a scan's (prepared, state)
    report: Mapping                  # what the derivation decoupled and its rounding class
    backward: Callable | None = None # a scan's chunk adjoint; a reduction's backward is a declaration of its own
    prepare: Callable = ...          # a tile's values that read no state, for every tile at once
    slots: Callable | None = None    # a reduction's carried states' roles (max, sum, contraction) and shapes

def derive(declaration: Reduction, *, accurate: bool, saturation: Saturation) -> Derived: ...
def derive_scan(declaration: Scan, *, saturation: Saturation, splits_held: int = FORMS_HELD) -> Derived: ...
```

- `step` takes the walk's tile as the template forms it: logits scaled, coordinates, and the mask and score
  hook applied. Places and tile layouts are the schedule's, not the algebra's.
- The hand-written forms (`attention_algebra`, `recurrence_algebra`) are adapted to the same coarse interface.
  The derived and hand-written forms then swap behind one switch, which check 1 drives both ways.
- A generated form replaces the hand-written one when it meets both bars of §4: bible §4's efficiency on every
  scoreboard row and GPU type the route serves, and bible §6's stability suite and accuracy gate against the
  float64 truth. Its internals need not decompose into the hand-written functions.
- The rule sets derivation needs beside §6.4's:
  - `decoupled`: dependent reductions split with correction terms; a running maximum chosen by `range`;
    log-sum-exp and moment merges;
  - `linear_scan`: a token's algebra over its state and its affine forms; the chunk's identities, compact
    WY, the state-space dual, a solve's inverse by doubling or its substitution, and a solve's linearity.
    A scan's adjoint is its forms' derivative;
  - `error_free`: the compensated sum, the natural exponent through `two_product` and `2**k`, and an argmax
    anchor's split identity. Under accurate numerics, extraction keeps these wherever the hand-written form's
    error bound requires them.

### 6.10 Marked regions and custom derivatives

These live in `dew.nn.derivatives`, below `dew.algebra`, `surrogate` (row 3271's declaration) among them. What exists: the four kinds, `marked_call_p` with its transform rules and their prototype test,
`surrogate`, the precision transforms marked `Protected`, and the pass reading a marked call's kind
(`program._within`, `program._opened`: a protected call an opaque leaf, a surrogate's and a kernel glue's body
rewritten within its call, an Algebra span opened into the region around it, its call kept where no form
lands), the declaring functions (`declared_derivative`, `declared_tangent`, `surrogate`), and the slop check
(SLOP026), which lands once each lane's sites are declared through them (rows 4026 to 4030).

```python
class Protected: ...                  # error-free transforms and numerics bodies (§1.2, §1.3)
class Surrogate: ...                  # a declared surrogate gradient's bodies
class KernelGlue: ...                 # a kernel derivative's XLA around its launches
class Algebra: ...                    # a derivative written only to restructure algebra: rewritten freely, and
                                      # retired once the engine derives its form

Kind = Protected | Surrogate | KernelGlue | Algebra

marked_call_p: jax.extend.core.Primitive   # a call: params call_jaxpr (a ClosedJaxpr) and kind (a Kind instance)

def marked(kind: Kind, function: Callable) -> Callable: ...   # function traced as one marked_call of its body

def declared_derivative(kind, primal, forward, backward, *, nondiff_argnums=()) -> Callable: ...  # a custom_vjp
def declared_tangent(kind: TangentKind, primal, tangent, *, nondiff_argnums=()) -> Callable: ...  # a custom_jvp
def surrogate(forward, derivative, *, nondiff_argnums=()) -> Callable: ...
    # row 3271: the value forward's exactly; derivative(primals, tangents, value) the declared output tangent,
    # linear in the tangents, so reverse mode transposes it and higher orders differentiate it
```

- A marked body is one call equation carrying its body's jaxpr and its kind, as `closed_call` carries a body, so
  it stays atomic through every transform; two sentinel equations would not, since partial evaluation splits a
  body's primal, residual and tangent parts and dead-code elimination drops an unused end. Its rules are
  registered through `jax.interpreters`' registries as `closed_call`'s are: its JVP is a marked call of the
  body's JVP, its partial evaluation splits the body into a known and an unknown marked call of the same kind
  (the residuals among the known one's outputs), its transpose is a marked call of the transposed body, and its
  batching rule a marked call of the batched body. Its lowering inlines the body, so a marked call costs nothing
  with the pass off.
- A prototype test holds the marker before anything builds on it: a body's forward, tangent and cotangent each
  arrive as marked calls of their kind through `jax.grad`, `jvp`, `linearize`, `vmap`, `jax.checkpoint` and
  `shard_map`.
- The pass reads `kind` as an object (`isinstance`), never a name or a scope string, and the pass inlines every
  marked call (`dew.nn.programs.inlined`, through `program._opened`), so the compiled program holds none.
- `dew.nn.precision`'s error-free transforms are `marked(Protected(), …)`, replacing the name scope the first cut
  reads.
- Every custom-derivative site declares itself through one of the three declarations, in place of a bare
  `jax.custom_vjp` or `jax.custom_jvp`. `linear_tangent` is to take its caller's declared kind (row 4026, after
  the fork's staged custom_jvp batching, row 4067): `KernelGlue` at a kernel site, `Protected` at a numerics one
  (`sampling.flow`'s KL, `nn.precision`'s products), and `Algebra` at nGPT's join and normed hidden, which the
  compile-time sets rewrite and which stay (§1.3). Once every site is declared, a bare custom derivative in
  `src/dew` fails a slop check (row 4012), so the inventory of §1.3 stays complete.
- `surrogate(forward, derivative)`: the forward runs as written and its tangent is `derivative`'s, both bodies
  `Surrogate`. No rule set holds a rule relating a `Surrogate` span to autodiff's derivative of its forward, so the
  engine optimizes inside each body and never replaces the rule.

### 6.11 The generator

**Parked** (§1.1): derived forms run at trace time today, and nothing has needed the generator yet.

```python
@dataclass(frozen=True)
class Generated:
    declaration: str                  # the declaring module and name
    module: str                       # the generated module, beside the declaring one: *_derived.py
    tests: str                        # its gate sites and stability suite
    rules: tuple[str, ...]            # the rule sets and their receipts
    egglog: str                       # the pinned version

@dataclass(frozen=True)
class Cases:                          # where a declaration is generated and checked
    operands: tuple[tuple[jax.ShapeDtypeStruct, ...], ...]   # each case's operands, by dtype and shape
    draw: Callable[[int, tuple[jax.ShapeDtypeStruct, ...]], tuple]   # the gate's drawn and adversarial families
    seeds: int                        # the declared seeds of its check 1 site
    policies: tuple[str, ...]         # the precision policies it is generated for: "default", "accurate"
    targets: tuple[str, ...]          # the GpuTargets it is priced for

@dataclass(frozen=True)
class StableForm:                     # a numeric function of Dew's own
    function: Callable                # its math as written: the form as written and, at float64, the truth
    cases: Cases

@dataclass(frozen=True)
class SurrogateBodies:                # a surrogate's two bodies (row 3271), each optimized, neither replaced
    forward: Callable
    derivative: Callable
    cases: Cases

DECLARATIONS: Mapping[str, Declaration | StableForm | SurrogateBodies]   # registered by the declaring modules

def generate(names: Sequence[str] | None = None, *, check: bool = False) -> tuple[Generated, ...]: ...
```

- `python -m dew.algebra.generate [names] [--check]` traces each registered declaration at each of its cases,
  saturates it with every set (`Mode.DEV`), measures each candidate's error on its cases' draws against the
  declaration at float64 (`dew.algebra.error`), extracts per §1.5 under each policy the cases name, priced for
  each target, and writes the module and its tests. `Reduction` and `Scan` carry `cases` as well.
- The output is ordinary Python: plain JAX functions with no import of `dew.algebra`, formatted, linted and typed
  as any module, and a header naming the declaration, the sets, their receipts and the egglog version. Nothing of
  the engine is imported at runtime.
- `--check` regenerates in memory and fails where the checked-in output differs, in CI's lint lane; a changed
  rule or declaration therefore lands as a reviewed diff of generated code.
- Sampling is seeded and its families fixed, so generation is deterministic across processes (§3).

## 7. Bit-level forms

Design record, 2026-11-05 (theme T82). The e-graph engine derives real-valued forms; this part derives the
cheapest instruction form of a declared bit-level transform, a low-precision format's decode first, over a
GPU's integer pipes, and proves it exact on every input. Its output is a form a template takes, so nothing
about it is an estimate.

### 7.1 What it derives

- **A transform** (`BitTransform`) maps a stored word of at most 32 bits to registers, each with a mask: the
  bits its consumer reads, every other bit zero. It is GF(2)-linear, each bit read the parity of some input
  bits (`rows`), so a format's decode is one: its codes' fields placed where the consumer's multiply reads
  them (`BitTransform.placed`), the exponent's bias folded into the block scale's multiply (an E2M1 code at a
  bfloat16's mantissa's top bit and its exponent's two low bits is the code's value times 2^-126). A transform
  that is not linear is not declarable: its proof over a 32-bit word would need a solver, and no consumer has
  one yet (a rounding pack's own search waits for its consumer).
- **Its layout** is fixed (the stored word is the input) or free: the stored word may be any invertible
  re-encoding of the input, chosen with the form. The encoder runs once where the weights are stored (Dew's
  quantization codec), the decoder each time they are read. Its `units` are the pipes its form may issue on:
  the ALU always (every LOP3's), the FMA pipe unless the consumer's own math holds it.
- **A form** (`BitForm`) is a list of instructions over the stored word, the register each output is, and the
  encoder (its bit `j` the parity of the input's bits `encoder[j]` names).

### 7.2 Vocabulary and price

- The instructions are those a decode needs, each GF(2)-linear (`BitOp`): a logical right shift and a left shift
  (SHF on the ALU, or IMAD.HI and IMAD.SHL, a multiply by a power of two, on the FMA pipe), a rotation (SHF's
  funnel shift, the ALU's alone), and LOP3 masking the XOR of one or two values (the mask an immediate). Every
  architecture from sm75 to sm120 issues them alike.
- **Price** (`BitForm.price`): the busiest pipe's issues, then the instruction count. Nsight Compute's Kernel
  Profiling Guide (its pipelines) puts logic, shifts and every integer instruction but IMAD and IMUL on the
  ALU, and IMAD and IMUL on the FMA pipe; the CUDA C++ Best Practices Guide 13 (Table 5) gives a shift, a
  bitwise operation and an integer multiply-add 64 results a clock per multiprocessor on every compute
  capability from 7.5 to 12.1. A decoder's words are many and independent, so throughput prices it, not its
  chain's latency, and a shift goes on whichever pipe balances the two. A form is checked against measured
  kernels where a consumer takes it, never against instruction counts alone (§7.4).

### 7.3 Search and proof

- **Search** (`synthesized`): one LOP3 per output register, `mask & (u ^ v)` or `mask & u`, over the stored
  word, at most one right shift or rotation of it by any amount, and up to `lefts` left shifts of it or of
  that, each by at most `reach` places; a register may read an earlier one, and a fixed layout's mask leaves
  out the bits its transform holds zero. Shapes (which shifts) are walked in increasing price,
  so the first form found is the cheapest the vocabulary holds. A free layout's registers need only be
  independent over GF(2) (an invertible read), checked row by row as each register is chosen, after a
  necessary count (each register's bit a parity of the sources' bits at its place) rejects most shift sets
  before any; a fixed layout's must equal the transform's rows. Every shift must be read, else a cheaper
  shape reads the same. The encoder is the form's map inverted, after the transform.
- **Proof** (`proven`): every instruction is GF(2)-linear, so a form and its encoder are linear maps, and two
  linear maps that agree on each basis word agree on all 2^width; a word of 16 bits or fewer is checked on
  every word besides. A fixed layout's encoder must be the identity: its form reads the input as stored. A
  form not proven is never returned, and a checked-in form is proven again by its consumer's tests. That
  proof replaces the sampled checks a real-valued form gets.
- The fp4 challenge, the evidence for the technique, searched this space in C: its NVFP4 decode into
  bfloat16 pairs takes 4 LOP3 + 1 SHF + 3 IMAD.SHL against the 7 + 2 before, bit-identical, +0.4% end to end.
  The search here re-derives it from the decode's declaration (2026-11-05, one armada core): at left shifts of
  up to 31 places it walks every shape of four issues on the busiest pipe and finds none, then returns a form
  of the challenge's price, a rotation by 6, shifts by 2 and 6 and the rotation's by 2 and four masked XORs,
  in 27 seconds (left shifts to 6 places: 2.3 seconds, the test's). No form without a right-moving shift
  exists within three left shifts.
- **Not in the vocabulary yet** (row 4050): PRMT's byte permutes, LOP3's other linear tables (a three-way
  XOR, a bit select), and transforms that are not linear (a rounding pack, an integer decode through fp16
  arithmetic), whose search over LOP3's truth tables, IADD3 and HFMA2 is a bit-vector synthesis (Z3) and whose
  proof over a 32-bit word is Z3's equivalence. Each joins when a consumer's transform needs it: sm80's FP8
  and FP4 epilogue packing (row 3733) is the first. No target varies the vocabulary: the pipes and rates above
  hold from sm75 to sm120, so the search takes no target.

### 7.4 Interface with the GEMM template (its first consumer, row 3733)

Agreed with the GEMM template's owner (2026-11-03):

- A declaration names the fragment layout the family's MMA reads, which the family bundle supplies: on sm90
  the wgmma A-operand fragment, the weight taken as A and swapped onto M at decode shapes so the decoder writes
  straight into it; on sm100 the tcgen05 and tensor-memory layouts. A thread's input is the packed words it
  loads: a 32-bit word holds 8 E2M1 codes.
- Block scales (NVFP4's 1×16, MX's 1×32) are a separate transform of a fixed layout, which feeds the
  template's promotion and epilogue: a form may re-encode the codes, never the scales.
- Each decode a template takes is declared and checked in once, in `dew.algebra.bits`, the one source the
  engine's and the consumer's tests prove: `NVFP4_BFLOAT16`, eight E2M1 codes of a word in their natural
  order (code `i` at bits `4i`, a byte's low nibble the earlier element) into four bfloat16 pairs, each the
  code's value times 2^-126, and its form `NVFP4_BFLOAT16_FORM`, the search's at shifts of up to 6 places.
  The codec packs codes naturally; the consumer's prepare step stores them in the form's layout once
  (`BitForm.prepared`, in `jax.numpy`), and its portable path reads a word tile `[r, c]` back as `[r, 8c]`
  elements in depth order (`BitForm.values`). An E2M1 subnormal (±0.5) decodes to a bfloat16 subnormal, so
  the portable path scales it in integer exponent arithmetic, never through a lane that flushes subnormals.
- The consumer records the form it takes: its encoder and the encoder's version (in the
  checkpoint), the decoder's live registers per thread (which `resources.py` prices before a tile is admitted)
  and its issues by pipe (`BitForm.issues`). `Work` holds no integer pipe yet: the consumer's head adds the
  ALU's and the FMA pipe's integer issues to it, each 64 results a clock per multiprocessor (§7.2), so
  `cost.py` prices a form's mix beside the tile's other work. Its tests prove the form again.
- The decoder is emitted inline at a matrix-operand hook the GEMM template adds (row 3621): `BitForm.ptx(i)`
  writes output register `i` as one PTX block over the stored word (`$1`) into the register (`$0`), each step
  as its pipe's instruction (`shr`, `shl` and `shf` on the ALU, `mul.lo` and `mul.hi` by a power of two on the
  FMA pipe, `lop3` for a mask), for Mosaic GPU's or Triton's inline assembly (constraints `=r,r`, a register
  a block). `BitForm.traced` computes the same registers in `jax.numpy` for the hook's portable path and the
  interpreter. Both evaluate one definition of each step (`bits._run`), the one `proven` checks. PTX fixes the
  arithmetic, not the SASS: ptxas chooses the instructions, which the consumer's compile report shows.
- Scope: NVFP4 and MXFP4 on sm90 and sm80; FP8 on sm80, which has no FP8 MMA; FP8 and FP4 packing in the
  epilogue. FP8 on sm90 needs no decode: wgmma multiplies E4M3 and E5M2 from shared memory.
- A form is taken only where it is bit-identical over its whole domain and timed no slower on the real GEMM.

### 7.5 Prior art

- Massalin's superoptimizer (ASPLOS 1987): exhaustive search over short instruction sequences, which this
  generalises.
- STOKE (Schkufza, Sharma and Aiken, "Stochastic Superoptimization", ASPLOS 2013): long sequences explored by
  MCMC and verified by a solver; forms this short are found exhaustively, their optimum included.
- Souper (Sasnauskas et al., arXiv 1711.04422): LLVM IR rewrites synthesized with SMT, on IR rather than target
  instructions, with no layout freedom.
- Rosette (Torlak and Bodik, Onward! 2013 and PLDI 2014): bit-vector synthesis by symbolic evaluation into
  SMT, the kind of query a non-linear transform's search would make.
- New here: the search is joint with a free invertible re-encoding of the stored word, which none of the above
  search, and it is priced per pipe for a GPU's ALU and FMA.
