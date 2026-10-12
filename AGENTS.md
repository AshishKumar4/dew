# Working on Dew

Read `CONTRIBUTING.md` for design, reference parity, code, tests, and writing standards. This checklist adds agent workflow rules; it does not duplicate that contract. Follow the user's current scope and approval boundaries.

## Scope and implementation

- Read the affected module, its callers, tests, and relevant design section before editing. Check JAX, Flax, Optax, Orbax, Grain, and the reference implementation for an existing primitive.
- Preserve checkpoint and reference-layout requirements described in `CONTRIBUTING.md`. Before 1.0, migrate callers together without compatibility shims.
- Keep code that owns meaningful complexity. Apply the deletion test to helpers; caller count alone does not decide whether a helper belongs.
- Preserve outputs and precision when optimizing an existing path. Treat requested quantization or approximate methods as explicit features with separate accuracy and performance evidence.
- Fix a failure by design, not by a guard. Derive a fact from its one source instead of passing it in and checking it, order steps so races cannot occur, and give each fact one owner. Add a conditional, timeout or assert only when no design change removes the need; timeouts belong at external boundaries such as networks and cloud allocation.
- Fix a failure where its cause lives, in the shared piece, not in the script or test that hit it, and remove the special case or copy that allowed it.
- Compose on the existing primitive. Before adding code, find the piece that already does the job; one concept has one implementation and one source of truth (for example `GpuTarget` for device facts and `KERNELS` for kernel choices).
- Duplication is banned everywhere, not only in `src/`: a dependency list, version pin, table, flag set, test list, policy or algorithm exists in one place. CI workflows, installers, armada recipes, private tooling and docs read or generate from that one owner. A second hand-kept copy is a bug, and a head that adds one does not land.
- Put optimizations in the shared core so every model family and backend benefits. Keep model- or hardware-specific code small and say why it cannot be general.
- Activations cross layer-to-kernel interfaces as `[batch, time, heads, head_dim]`. A kernel adapts to that layout; layer code does not permute activations for a kernel. Other tensors keep their own meaning.

## Parallel work and integration

- Give each parallel writer an isolated worktree and clear file ownership. Coordinate shared-file edits and GPU use before starting them.
- Inspect GPU processes and current lane ownership before launching GPU work. A process name alone does not establish whether the device is available.
- Commit completed work as `Ashish Kumar Singh <ashishkmr472@gmail.com>` with a concise conventional commit message. A lane pushes its own branch; only the coordinating agent merges to `main`. Never force-push.
- Preserve tracked and untracked work before integration. Remove a worktree only after its work is committed, verified, merged, and no process still needs it.
- Deliver one head per area of work, landed when that piece is complete and gated; fold small fixes into it rather than landing them separately. Merge `main` into an active head at least daily and before landing.
- Land complete work early. A change whose correctness the CPU suite proves lands without waiting on unrelated GPU measurements; a default changes only with the measurement that justifies it.
- Give a finished large head two or three independent reviews in parallel, each with a different focus, instead of reviewing small pieces one by one.
- A review checks the head against the whole project, not only its diff: for every list, table, helper or policy the head adds, it searches the repository and the private tooling for an existing owner. A duplicate it finds blocks the head.
- Keep private development tooling (personal cloud runners and session drivers, provisioning scripts, one-off session scripts) out of this repository. Tooling that Dew's own CI, tests, docs or users need, such as the CI entry points and the profiling and attribution tools, stays here.

## Verification

- Run the affected test files with pytest's cache enabled. Let the coordinating agent run integration suites; capture complete output and exit status once instead of rerunning a suite to recover failure names.
- In a worktree, pytest uses its configured source path; run scripts with `PYTHONPATH=src` so they import that worktree. Use the project's environment and commands documented in `CONTRIBUTING.md`.
- Use CPU tests for fast logic checks. For kernel code, CPU checks are limited to compiling for each target and interpret-mode math; correctness and performance are proven on the real GPUs and TPUs the kernel targets, and kernel code lands only after its touched GPU tests pass on each of them. Keep test cases small enough to isolate the failure.
- For numerical changes, record reference version, inputs, dtype, backend, command, observed error, and the reason for the tolerance. Measure the distance from a float64 reference per output and gradient, at real shapes on the target device; the bound is no looser than the reference implementation's own measured distance. Investigate a failing bound before changing it, and never loosen one to pass.
- Run heavy work (the full suite, type checks, builds, benchmarks, training) on remote CPU and GPU runners, not on a developer's workstation.
- A shell pipeline's last command succeeding does not prove tests passed. Stop integration on a failed check; fix and rerun the affected behavior.

## Performance investigations

- Start with [JAX/XProf](https://docs.jax.dev/en/latest/201/profiling.html) and [Grain's native diagnostics](https://google-grain.readthedocs.io/en/latest/tutorials/dataset_debugging_tutorial.html). Add instrumentation only for a demonstrated visibility gap.
- Separate compilation, data warm-up, steady execution, and checkpoint/inference costs. Choose capture windows from settled timings and compile events, not a fixed warm-up assumption.
- Preserve asynchronous execution. Synchronize timing boundaries, not every step; distinguish dispatch time from completed device work.
- Keep profiling-disabled execution free of timers, trace allocations, retained arrays, extra transfers, and synchronization. Measure active-capture overhead separately.
- Attribute stalls across input production, placement, dispatch, collectives, and kernels before changing code. Busy-device percentages alone do not establish useful throughput or compute efficiency.
- Use compile-time JAX naming inside compiled code. Inspect compilation, shapes, static arguments, and executable reuse before blaming missing JIT or replacing kernels.
- Cache deterministic work before random transformations unless frozen augmentation is intentional. Preserve record order, randomness, restart position, sharding, and buffer memory bounds.
- Profile the actual execution configuration. Label diagnostic variants; preserve native reports and missing-data warnings rather than presenting unavailable counters as zero.
- Report MFU as model FLOPs on real tokens over wall time, and name the precision and peak each utilization figure uses. A run below 50% MFU is profiled and its three largest losses fixed; the goal is past 75%.
- The bar for a kernel is the best existing implementation for that operation on that device, measured in the same session at real training shapes. Write a custom kernel when the supplied one is the bottleneck. Use open projects as design references and keep their licence notices for anything ported.
- Claim a speed comparison only for the workload and hardware that were measured.

## Reporting

- Distinguish implemented behavior, tested behavior, research findings, and open work. Keep each requested item tracked until its acceptance checks pass or the user defers it.
- Report benchmark conditions and commands, exact checks run, remaining risks, and blocked prerequisites. Do not infer full framework parity or production readiness from small fixtures or simulated devices.
- Preserve the author's voice. Use the writing standards in `CONTRIBUTING.md` for docs, comments, commits, and replies.
