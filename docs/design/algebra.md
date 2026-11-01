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
  Saturation stops at limits derived from the program's size (§6.6), never at a wall-clock limit, so the same input
  gives the same output in any process. Dew writes one piece itself: extraction that prices shared values once (a
  DAG cost), since a reduction read once and used twice costs one pass.
- **Language:** the arithmetic core of a jaxpr:
  - elementwise operations, casts, broadcasts, reshapes and transposes;
  - reductions (sum, max), `dot_general` and select;
  - rsqrt, sqrt, exp, log, abs and sign.
- **Opaque leaves,** never rewritten: Pallas kernels, collectives, `shard_map`, sharding constraints, random
  numbers, callbacks and effects. A custom derivative is opaque where its primitive survives, in a program no
  transform has traced through. Under `jax.grad`, JAX traces its forward and backward rules inline, so nothing of
  it is left to hold opaque.
- **Protected regions,** never rewritten in any numerics mode: `dew.nn.precision`'s error-free transforms
  (`two_sum`, `two_product`, `compensated_*`, `widened_reduction`) and any region a precision module marks. The
  module emits a marker the pass reads, which survives autodiff because the marked ops trace inline under it. The
  rounding-class fact of their values is seeded from the marker. Reassociation would cancel exactly the residual
  these transforms exist to keep: `compensated_add`'s low word, or the compensated sum that holds Mamba's D gradient
  inside check 1 (a plain sum puts it at 1.60 times the reference's error, past the gate's 1.5).
- **Control flow:** the bodies of `scan`, `while`, `cond`, nested `jit` (`pjit`) and checkpoint (remat) are regions
  of their own, rewritten in place and recursively. Bodies are never unrolled, so a step's program depth stays
  fixed. Nothing moves across a body boundary or a loop carry.
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
- **Cost:** each form is priced in time, from the device's facts:
  - passes over large tensors, in bytes over memory bandwidth, counting elementwise work as fused into its
    consumers, as XLA fuses it;
  - collectives, in bytes over the link's bandwidth;
  - FLOPs over the FLOP rate;
  - the depth of dependent reductions as latency.

  The facts are the device's own, measured or published, and the prices are checked against measured profiles,
  not fitted to them. A rule set ships only with measured receipts.
- **Rounding:** a rewrite that rounds identically always applies. One that reorders rounding (linearity,
  hoisting, distributivity) applies under default numerics and never under `--precision accurate`, through the
  typed precision policy.

## 2. Where it applies

- **Programs:** the seam is `OuterProgram` (`dew.nn.kernels.options`). Every outer program Dew compiles is one:
  `options.program` builds it for evaluation and loads, `Objective.program` builds it for training steps, and the
  serving kernel builds it for its step and decode programs.
  - The traced jaxpr already holds the backward, since autodiff runs at trace time.
  - The pass runs between trace and lower in `OuterProgram.compiled`, as one function, `dew.algebra.rewritten`.
    The result compiles under the same `jit` options and shardings, through JAX's public jaxpr API.
  - Rewrites stay within checkpoint (remat) regions and never change what a step saves.
  - The planner's gradient trace (`dew.training.footprint`) calls the same function before it prices the
    program, so there is one program description.
  - `dew.algebra.rewritten(fn)` is also a public transform, so the accuracy gate can compile an unrewritten
    reference arm beside a rewritten candidate (§3).
- **Template hooks:** they are jaxprs, so the same pass simplifies them before the Pallas fuser places them in a
  kernel.
- **Template algebra:** a template's online, tiled and combine forms, forward and backward, are derived from a
  declaration:
  - attention's score function, mask and reduction;
  - a recurrence's state update.

  The hand-written algebra modules are the reference a derived form is held to. It need not equal them. It replaces
  one when it passes the [accuracy gate](testing.md) against it and is at least as fast in the kernel: check 1, its
  distance to the float64 truth within the fixed 1.5 parity of the hand-written form's, the worst row included,
  and check 2 where it applies. A different form that is as good or better is the point. A new variant then needs
  only its declaration and its schedule.
- **Fusion:** a search over algebraic forms, fusion boundaries and template compositions for a region of a step
  (a norm, a projection and an activation, for example), built from the toolkit's composed primitives and its work
  scheduler. Applied to a whole decoder layer, its output is a generated megakernel.

## 3. Verification

All accuracy checks are the [accuracy gate](testing.md)'s, with no tolerance of their own.

- **Each rule:** `assert_as_exact` over drawn and adversarial families. The rule's left side is the reference arm,
  the left side at float64 the truth, and the right side the candidate. `assert_certified` is added where a bound
  applies.
- **Each program site the pass changes:** `assert_as_exact(rewritten(fn), fn, draw, truth_fn=fn at float64)`. That
  is check 1 (the fixed 1.5 parity, the worst row included) and check 2 where it applies, at each gate site whose
  program the pass rewrites.
- **Protected regions:** a rule test injects a compensated sum and requires its residual to survive saturation and
  extraction.
- **Rewritten backwards:** they pass the derivative contracts, nested `jvp` of `grad` and forward over reverse
  (`tests/contract_support.py`'s `assert_contract`).
- **Determinism:** two fresh processes with different `PYTHONHASHSEED` give the same rewritten program, compared as
  its lowered StableHLO, for the same input, checked on the multi-process CPU lane. Multi-controller JAX needs every
  process to compile the same program, and one process repeating itself cannot catch an ordering that follows
  object identity or string hashing. The e-graph is built in equation order and runs with one thread.
- **Compile time:** the pass's seconds per program are recorded beside lowering and XLA compile, in perf_gate and
  the eval.
- **Report:** each program's rule classes that fired and the pass's seconds go in the one kernel report
  (`dew.nn.kernels.templates.report`, `report.compiled`).

## 4. What it replaces, and what it leaves

- **It replaces** custom derivatives that exist only to restructure algebra, and hand-derived template algebra
  wherever a derived form passes the accuracy gate against it and is at least as fast.
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

## 6. Interface

Draft, 2026-11-01. Every part of the work builds against this section, so a change to it is a change to the design.
The engine runs egglog 14.0.0 (MIT; wheels for CPython 3.12–3.14 on Linux, macOS and Windows), pinned exactly,
because its Python API is still marked as subject to change.

### 6.1 Modules

| Module | Holds |
|---|---|
| `dew.algebra.terms` | The term language and its conversion to and from jaxprs |
| `dew.algebra.facts` | Facts about each value (e-class analyses) |
| `dew.algebra.rules` | `Rules`, `Monoid`, their registries and the sets: `linear`, `identities`, `reuse`, `decoupled`, `linear_scan`, `error_free` |
| `dew.algebra.cost` | `Price`, `CostModel` and DAG extraction |
| `dew.algebra.program` | The pass at `OuterProgram` (`rewritten`), its public transform and its report |
| `dew.algebra.derive` | Template declarations and their derived forms |

### 6.2 Terms

`Term` is one egglog sort, `class Term(Expr)`. Each constructor is one jaxpr primitive of the language in §1, with
the primitive's parameters as primitive sorts:

- shapes, axes and permutations as `Vec[i64]`;
- dtypes as `String`;
- literals as `f64` together with their dtype.

```python
class Term(Expr):
    @classmethod
    def input(cls, index: i64Like) -> Term: ...                 # a region's argument
    @classmethod
    def literal(cls, value: f64Like, dtype: StringLike) -> Term: ...
    @classmethod
    def opaque(cls, equation: i64Like, output: i64Like) -> Term: ...  # never rewritten (§1)
    @classmethod
    def semantic(cls, name: StringLike, operands: Vec[Term], output: i64Like) -> Term: ...
    def __add__(self, other: Term) -> Term: ...                 # add, sub, mul, div, max, min, ...
    def unary(self, name: StringLike) -> Term: ...             # neg, exp, exp2, log, sqrt, rsqrt, abs, sign
    def compare(self, kind: StringLike, other: Term) -> Term: ...  # lt, le, eq, ne, ge, gt
    @classmethod
    def iota(cls, shape: Vec[i64], dimension: i64Like) -> Term: ...
    def prefix(self, monoid: StringLike, axis: i64Like, reverse: BoolLike) -> Term: ...  # cumsum, cummax, ...
    def convert(self, dtype: StringLike) -> Term: ...
    def broadcast(self, shape: Vec[i64], dimensions: Vec[i64]) -> Term: ...
    def reshape(self, shape: Vec[i64]) -> Term: ...
    def transpose(self, permutation: Vec[i64]) -> Term: ...
    def reduce(self, monoid: StringLike, axes: Vec[i64]) -> Term: ...   # a registered monoid (§6.3)
    @classmethod
    def dot(cls, lhs: Term, rhs: Term, contracting: Vec[i64], batch: Vec[i64], result: StringLike) -> Term: ...
    @classmethod
    def select(cls, predicate: Term, true: Term, false: Term) -> Term: ...
```

- `semantic` is a primitive whose meaning is declared, not computed by the e-graph. Rules can equate it with other
  terms, but never rewrite inside it. These primitives are:
  - the error-free transformations, `two_sum`, `two_product` and scaling by `2**k`, exact barring overflow;
  - `bounded_split(x, dtype)`, hi = dtype(x) and lo = dtype(x − hi), whose declared error bound is the dtype's
    rounding of lo;
  - `triangular_inverse`.
- `terms.region(jaxpr, equations) -> (Term roots, Leaves)` converts a region of equations:
  - every equation outside the language becomes `opaque`;
  - every value the region reads but does not compute becomes an `input`.
- `terms.rebuilt(roots, leaves) -> jax.core.Jaxpr` converts an extracted term back to equations.
- A region is a maximal set of equations of the language. It never crosses:
  - a checkpoint (remat) boundary;
  - an opaque equation;
  - a scan or while body (those are regions of their own).

### 6.3 Facts

Each fact is an egglog `@function` from `Term` to a primitive sort. Its `merge` is the lattice join, and its rules
derive it from its arguments' facts.

| Fact | Sort | Join | Read by |
|---|---|---|---|
| `shape(t)` | `Vec[i64]` | equal or fail | every rule that names an axis |
| `dtype(t)` | `String` | equal or fail | casts, rounding |
| `constant_axes(t)` | `Set[i64]` | union | hoisting a scalar out of a reduction |
| `sharded_axes(t)` | `Map[i64, String]` (axis to mesh axes) | equal or fail | a sum over a sharded axis is a collective |
| `walked_axis(t)` | `i64` (−1 for none) | equal or fail | the toolkit's derived forms |
| `per_token(t)` | `Bool` (varies along the walked axis) | or | hoisting across tiles; factors that commute through WY |
| `decay(t)` | `String` (`scalar`, `per_head`, `per_key`) | equal or fail | a recurrence's decay hoisting |
| `structure(t)` | `String` (a state update's `A_t`: `scalar`, `diagonal`, `rank_one_correction`) | equal or fail | the linear-scan rules |
| `range(t)` | `Pair[f64, f64]` (bounds, where cheap) | intersect | a stable correction (exp arguments at most 0) |
| `rounding(t)` | `String` (`exact`, `reorders`, `bounded`) and `error(t)`, an `f64` bound | the worse | accurate extraction |

Two notes on the table:
- "equal or fail" means members of one e-class that disagree raise, naming the rule that united them.
- The inputs' facts come from the traced avals, their shardings, and a declaration's roles (§6.9).

Monoids are an open registry, so a reduction, a prefix and a derived form's merge all name one:

```python
@dataclass(frozen=True)
class Monoid:
    name: str                          # "sum", "max", "softmax" (m, total, acc), "affine" (A, b),
                                       # "compensated" (sum, error), "argmax", "moments", ...
    combine: jax.core.ClosedJaxpr      # (state, state) -> state
    identity: jax.core.ClosedJaxpr     # () -> state
    associative: bool                  # False: an ordered fold, never reassociated
    rounding: str                      # its combine's rounding class
    tie_break: str | None = None       # a total order where combine chooses (argmax: lowest position)

MONOIDS: Mapping[str, Monoid]
def register_monoid(monoid: Monoid) -> None: ...
```

### 6.4 Rules

```python
@dataclass(frozen=True)
class Rules:
    name: str                                # "linear", "identities", "reuse", "decoupled", "linear_scan", ...
    rounding: Literal["exact", "reorders", "bounded"]   # the rounding class each of its rewrites gives
    ruleset: egglog.Ruleset
    receipt: str                             # the measurement it ships with: a ledger row and its receipts

RULES: Mapping[str, Rules]                   # every registered set, by name
def register(rules: Rules) -> None: ...
def admitted(numerics) -> tuple[Rules, ...]  # the sets a program's precision policy allows
```

- A set's rules are egglog `rewrite`/`birewrite`/`rule` objects in its `ruleset`.
- Each rule's float64 identity test names the rule (§3).
- A rewrite's result joins its e-class with its rounding class and error bound (§6.3). Under default numerics
  every set is admitted. Under `--precision accurate`:
  - `exact` sets are admitted;
  - `bounded` and `reorders` forms are admitted only where extraction finds their error bound within the
    hand-written form's (§6.5).

### 6.5 Cost and extraction

```python
@dataclass(frozen=True, order=True)
class Price:                # compared in this order
    error: float            # under --precision accurate only: the form's error bound; 0 under default numerics
    seconds: float          # the form's time from the device's facts (§1)
    bytes: int = field(compare=False)        # its terms, recorded beside the price
    depth: int = field(compare=False)        # the longest chain of dependent reductions
    collectives: int = field(compare=False)  # bytes over links
    flops: int = field(compare=False)

@dataclass(frozen=True)
class Device:               # the facts a price reads, the device's own
    memory_bandwidth: float # bytes a second
    link_bandwidth: float   # bytes a second, per collective's group
    flop_rate: float        # by dtype
    reduction_latency: float

class CostModel(Protocol):
    def price(self, node: Node, facts: Facts) -> Price: ...    # the node alone; extraction prices sharing

def extracted(graph: egglog.EGraph, roots: Sequence[Term], cost: CostModel) -> Extraction: ...

@dataclass(frozen=True)
class Extraction:
    roots: tuple[Term, ...]
    price: Price                     # the region's cost, each shared value priced once
    fired: Mapping[str, int]         # rule set name: the extracted nodes it introduced
```

- A cost model prices a node alone. Extraction composes the prices over the chosen DAG: `bytes`,
  `collectives` and `flops` add up with each shared e-class counted once, and `depth` is the longest path of
  reductions.
- Extraction reads the e-graph through egglog's serialization. A DAG extractor prices each shared e-class once.
  egglog's `greedy-dag` is the baseline it has to beat.
- Ties break by node order in the serialization, which is deterministic with one thread.
- `fired` comes from the nodes each set's run added, recorded between runs of the sets.
- A device's facts are measured or published for that device. Prices are checked against its measured profiles,
  never fitted to them; a form whose price and profile disagree is a finding against the cost model.
- Two cost models implement the protocol:
  - the program model, in passes over HBM;
  - the kernel model, for derived template forms: values live per thread (registers), instructions per tile and
    the dependent-reduction depth inside a step, read from the toolkit's register model.

### 6.6 Saturation

```python
@dataclass(frozen=True)
class Saturation:
    growth: float           # nodes allowed a region, per equation of it
    depth: int              # iterations allowed, per level of the region's longest chain

def limits(region, saturation: Saturation) -> tuple[int, int]: ...   # (nodes, iterations) for this region
```

- The limits are derived from the region's size where they are set (`limits`): nodes at `growth` times its
  equations, iterations at `depth` times its longest chain. egglog has no node limit of its own, so the count is
  checked between iterations.
- `growth` and `depth` are measured. They are the largest the registered rule sets reach at their fixpoint on the
  eval's programs, recorded with the rule sets' receipts and re-measured when a set changes.
- A region that reaches a limit keeps the cheapest form found so far, and the report says so.
- Sets run in registry order, one iteration each, until no set changes the graph or a limit is reached.
- The e-graph runs with one thread and is built in equation order, so the same input gives the same output in any
  process (§3).

### 6.7 The program pass

```python
def rewritten(closed: jax.core.ClosedJaxpr, *, numerics, cost: CostModel,
              saturation: Saturation) -> Rewrite: ...

def rewritten_function(fn: Callable, **options) -> Callable: ...   # public as dew.algebra.rewritten(fn)

@dataclass(frozen=True)
class Rewrite:
    closed: jax.core.ClosedJaxpr     # same inputs, outputs, avals and shardings
    report: AlgebraReport
```

- `OuterProgram.compiled` traces its function with `jax.make_jaxpr`, rewrites the result, and jits a function
  that evaluates the rewritten jaxpr (`jax.core.eval_jaxpr`). The jit options and shardings stay the program's
  own. Every builder of an `OuterProgram` gets the pass: `options.program`, `Objective.program` and the serving
  kernel's programs.
- The program's compiler options gain `algebra`: the admitted set names, or none. It joins the program's cache
  key, so a rewritten program and an unrewritten one never share an executable.
- The planner prices the rewritten program: `footprint`'s traced gradient passes through `rewritten` before it is
  priced.
- `dew.algebra.rewritten(fn)` rewrites a function as the seam would. The gate uses it to build a rewritten
  candidate beside an unrewritten reference, which it compiles with plain `jit`.

### 6.8 The report

`AlgebraReport` joins the one kernel report (`dew.nn.kernels.templates.report`, through `report.compiled`), so it
is recorded with each compiled step (`StepCompiled`) and each `OuterProgram` compile. It holds:
- the regions rewritten;
- the price before and after;
- the rule sets that fired, with counts;
- the saturation's iterations and node count, its limits, and whether a limit stopped it;
- the pass's seconds, beside lowering's and XLA's compile's.

### 6.9 Derived template forms

A declaration states semantics only: plain JAX on whole operands, or token functions along a walked axis. It never
states a hand-written form.

```python
@dataclass(frozen=True)
class Reduction:                     # attention, and any walk whose state is a monoid's
    function: jax.core.ClosedJaxpr   # the math on whole operands: softmax_keys(where(seen, score(q, k), -inf)) @ v
    operands: Mapping[str, Axes]     # each operand's named axes: q (rows, d), k (keys, d), v (keys, e)
    walked: str                      # the axis the kernel walks in tiles: "keys"
    exponent: Literal[2, "e"]        # stated by each template per call (a scale folded with log2(e)), never found
    hooks: tuple[Hook, ...] = ()     # score, mask and output hooks with their parameter kinds

@dataclass(frozen=True)
class Scan:                          # a recurrence
    update: jax.core.ClosedJaxpr     # (h, x_t) -> h, seen as h <- A_t h + b_t
    readout: jax.core.ClosedJaxpr    # (h, x_t) -> o_t
    operands: Mapping[str, Axes]
    walked: str                      # "time"
    structure: str                   # A_t: "scalar", "diagonal" or "rank_one_correction"

Declaration = Reduction | Scan

class Derived(NamedTuple):
    state: Pytree                    # the per-row values carried along the walked axis
    init: Callable                   # rows -> state, the empty partial
    step: Callable                   # (state, tile as the template forms it, coords) -> state
    merge: Callable                  # (state, state) -> state: split walks, the work scheduler's fix-up
    finish: Callable                 # state -> outputs and the residuals a backward reads
    backward: Derived | None         # the same for the backward walk
    report: AlgebraReport            # rule sets fired, price, each form's rounding class and error bound

def derive(declaration: Declaration, numerics, cost: CostModel) -> Derived: ...
```

- `step` takes the walk's tile as the template forms it: logits scaled, coordinates, and the mask and score
  hook applied. Places and tile layouts are the schedule's, not the algebra's.
- The hand-written forms (`attention_algebra`, `recurrence_algebra`) are adapted to the same coarse interface.
  The derived and hand-written forms then swap behind one switch, which check 1 drives both ways.
- A derived form replaces the hand-written one when it passes the accuracy gate against the float64 truth and
  is at least as fast in the kernel (§2). Its internals need not decompose into the hand-written functions.
- The rule sets derivation needs beside §6.4's:
  - `decoupled`: dependent reductions split with correction terms; a running maximum chosen by `range`;
    log-sum-exp and moment merges;
  - `linear_scan`: the affine monoid, compact WY with `triangular_inverse`, decay hoisting as exp of prefix
    sums (the compensated prefix under accurate numerics), and a scan's adjoint;
  - `error_free`: the compensated sum, the natural exponent through `two_product` and `2**k`, and an argmax
    anchor's split identity. Under accurate numerics, extraction keeps these wherever the hand-written form's
    error bound requires them.
