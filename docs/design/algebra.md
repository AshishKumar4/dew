# Algebraic optimization: Dew derives the cheapest form of its own math

Design record, 2026-11-01. This is the binding design for how Dew rewrites the math of its programs and kernels
into cheaper equivalent forms. Work items live in the [roadmap](../roadmap.md). It builds on
[the kernel templates](kernel-templates.md), [training](training.md) and [testing](testing.md).

## 0. The idea in one paragraph

Autodiff and plain code produce correct math, not cheap math. A unit norm's backward, for example, reduces over
each row; a norm of norms makes those reductions wait on each other, so the backward reads every row several times.
Three algebraic laws (linearity of sums, pulling a per-row scalar out of a sum, distributivity) make every
reduction independent, so all of them run in one pass. Today such rewrites are derived by hand as custom VJPs or
template algebra. Dew instead holds every equivalent form of a program in an e-graph, applies declared algebraic
rules until no new form appears (equality saturation), and extracts the cheapest form under a cost model. One
engine serves three uses: whole programs, the templates' algebra, and the search over fusions.

## 1. The engine: `dew.algebra`

- **E-graphs:** [egglog](https://github.com/egraphs-good/egglog), used through its Python bindings, not forked.
  Saturation stops at node and iteration limits, never at a wall-clock limit, so the same input gives the same
  output. Dew writes one piece itself: extraction that prices shared values once (a DAG cost), since a reduction
  read once and used twice costs one pass.
- **Language:** the arithmetic core of a jaxpr:
  - elementwise operations, casts, broadcasts, reshapes and transposes;
  - reductions (sum, max), `dot_general` and select;
  - rsqrt, sqrt, exp, log, abs and sign.
- **Opaque leaves,** never rewritten: Pallas kernels, custom derivatives, collectives, `shard_map`, sharding
  constraints, random numbers, callbacks and effects.
- **Facts about each value** (e-class analyses):
  - its shape and dtype;
  - the axes it is constant along;
  - whether each axis is local or sharded (a sum over a sharded axis is a collective);
  - the rounding class of the forms that produce it.
- **Rules,** in declared sets in one place:
  - **linear algebra:** linearity of sums, scalars hoisted out of reductions along constant axes,
    distributivity, `dot_general` ⇄ multiply and sum, inner products moved across linear maps (⟨x, L y⟩ = ⟨Lᵀ x, y⟩);
  - **identities:** rsqrt and sqrt, exp and log, abs and sign, cast chains;
  - **reuse:** a value the forward already computed;
  - **decoupled reductions:** dependent nonlinear reductions broken with correction terms (the running-max
    rescale of online softmax, log-sum-exp merges, moment merges).
- **Cost:**
  - passes over large tensors, in bytes, counting elementwise work as fused into its consumers, as XLA fuses it;
  - then the depth of dependent reductions, then collectives, then FLOPs;
  - calibrated against measured profiles, and a rule set ships only with measured receipts.
- **Rounding:** a rewrite that rounds identically always applies. One that reorders rounding (linearity,
  hoisting, distributivity) applies under default numerics and never under `--precision accurate`, through the
  typed precision policy.

## 2. Where it applies

- **Programs:** every outer program Dew compiles passes through `options.program` (training steps, serving,
  evaluation, loads).
  - The traced jaxpr already holds the backward, since autodiff runs at trace time.
  - The engine rewrites it, and the result compiles under the same `jit` options and shardings, through JAX's
    public jaxpr API.
  - Rewrites stay within checkpoint (remat) regions and never change what a step saves.
  - The planner prices the rewritten program, so there is one program description.
- **Template hooks:** they are jaxprs, so the same pass simplifies them before the Pallas fuser places them in a
  kernel.
- **Template algebra:** a template's online, tiled and combine forms, forward and backward, are derived from a
  declaration:
  - attention's score function, mask and reduction;
  - a recurrence's state update.

  The hand-written algebra modules are the oracle: a derived form replaces a hand-written one only after matching
  it in float64 and in kernel speed. A new variant then needs only its declaration and its schedule.
- **Fusion:** a search over algebraic forms, fusion boundaries and template compositions for a region of a step
  (a norm, a projection and an activation, for example), built from the toolkit's composed primitives and its work
  scheduler. Applied to a whole decoder layer, its output is a generated megakernel.

## 3. Verification

- Each rule has a float64 identity test over random and adversarial inputs.
- Each class of rewritten program passes the [accuracy gate](testing.md): its distance to the float64 truth is no
  worse than the original's, beyond the gate's derived factor.
- Rewritten backwards pass the derivative contracts.
- The same input jaxpr gives a bytewise-identical output.
- Each program's compile report lists the rule classes that fired.

## 4. What it replaces, and what it leaves

- **It replaces** custom derivatives that exist only to restructure algebra, and, once matched, hand-derived
  template algebra.
- **It leaves:**
  - the templates' schedules, the toolkit and the families;
  - custom derivatives that implement kernels;
  - the planner's contract and the precision policy's meaning;
  - JAX, jaxlib and XLA, used through public interfaces only.
- **XLA keeps** fusion and code generation; it does not reassociate floating-point math by default, so chosen forms
  reach it intact.

## 5. Prior art

- egg and egglog (equality saturation);
- SPORES (sum-product rewriting for linear algebra);
- Tensat and TASO (verified tensor-graph rewrites);
- Herbie (rewriting for floating-point accuracy);
- Neptune (fusing dependent reductions with correction terms, deriving FlashAttention-style forms);
- Mirage (joint search over algebra, schedules and kernels, with randomized equivalence checks);
- Constable (equality saturation inside XLA, costed against the real compiler's fusion);
- FlexAttention (variants as declared functions).
