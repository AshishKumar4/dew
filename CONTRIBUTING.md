# Contributing

Dew is small on purpose. Every line has to earn its place. These rules apply to people and to agents alike, and a review checks each one.

## Design

- Compose before you write. Look for the primitive first: `jax.nn.dot_product_attention` (grouped-query heads, causal and windowed masks, the fused kernels), `flax.linen` (norms, embeddings, attention with its decode cache, `scan`, `remat`), `optax` (losses, schedules, transforms), `orbax` (checkpoints and retention), `grain` (sources, sharding, batching, packing), and Google's own JAX code (MaxText, tokamax, the gemma library) for anything they already do well. A reimplementation needs a reason a reader can check: a measured inefficiency, a missing feature, or a parameter layout we must match. Write that reason where the code is.
- One path. A capability has one implementation, one config field, one registry entry. No fallbacks, flags or compatibility layers without a demonstrated need.
- The seams are the contract. Models are plain Flax modules that know nothing about training. Objectives own parameters, loss and validation. The trainer owns the mesh, the compiled step, EMA, checkpoints and logging. Data sources produce records; transforms are Grain transforms. A new architecture is a module and a registry entry; a new modality is an objective. If a change needs to cross these lines, the design is wrong, not the lines.
- Prefer deep modules: a small interface over real complexity. Delete an abstraction if inlining it makes the code clearer.
- Frozen: parameter tree names, the checkpoint layout, wandb metric keys, the Objective methods, and the Hugging Face parameter layout of `CausalTransformer`. A change to any of these is a migration, with a converter and a test that loads the old form.

## Reference parity

Anything that implements a published architecture, schedule, sampler or loss is a port, and a port is correct only when it reproduces the reference.

- Identical design, not an equivalent one. The same parameter layout, the same operation order where numerics depend on it (norm placement, RoPE convention, softcapping, attention scaling, the dtype each step runs in), the same defaults. A rearrangement is allowed only with a test that shows it agrees with the reference to the stated tolerance.
- A parity test ships with the port. It loads the same weights into Dew and into the reference (the transformers implementation for model families, the authors' code for a paper, the equation for a schedule), runs the same inputs at fp32, and asserts the outputs agree: identical argmax and a stated maximum absolute difference for logits, a stated tolerance for everything else. The tolerance and the largest observed difference are written in the test.
- Fixtures are reproducible. The script that generates reference outputs is committed under `tools/`, the fixtures it produced are small and committed under `tests/fixtures/`, and a network-marked test regenerates them against the real checkpoint when it is available. A fixture nobody can regenerate is not evidence.
- Configuration round-trips. A reference `config.json` translates into Dew's fields and back without loss; the translation is tested on the real configs of the smallest checkpoint of each family.

## Code

- Smallest correct version. No dead parameters, no helper used once, no branch for a case that cannot happen.
- Fix causes, not symptoms. No suppressed warnings, no special-cased inputs, no zero-filled fallbacks.
- Types are narrow and honest. No `Any`, no casts to make a checker quiet.
- Comments say why, never what or what changed. Docstrings describe the code as it is.
- Performance is measured, not assumed. A change that claims to be faster ships with the number, the command that produced it and the hardware it ran on. Defaults are the fast ones.
- Performance never costs anything else. An optimization is accepted only if the loss, gradients and outputs match the code it replaces to fp32 tolerance, nothing observable is removed, and no reduced-precision path, clipping or approximation is introduced. A test that would fail if a term were dropped ships with it.

## Tests

- Every behaviour has a test that can fail, at the seam where the behaviour lives: a scheduler against its paper's invariants, a sampler against an analytic denoiser, a model through `fit` on the simulated 8-device mesh, a parity check against the reference for anything that ports a published design.
- A bug fix comes with the test that fails before and passes after.
- Tests are deterministic and run on CPU. The GPU lane exists for kernels and dtypes, not for logic.
- `tests/test_architectures.py` fails when a registry entry has no training case. Keep it that way.

## Writing

Plain sentences. Short. No em dashes. No words that sell (robust, seamless, leverage, cutting-edge, comprehensive). Tables for comparisons. The README and docs describe what the code does today; a claim without code behind it is a bug.

## Before a merge

1. The suite passes on CPU (`JAX_PLATFORMS=cpu pytest -m "not network" -q`) and the touched files pass on a GPU.
2. Every new number in docs has its reproduction command, and every port has its parity test.
3. An independent review has read the code, not the description of it.
