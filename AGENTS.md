# Working on Dew

Read `CONTRIBUTING.md` first: it owns the design, reference parity, code, test-authoring and writing rules, and this file does not restate them; [testing](docs/design/testing.md) owns the scenario inventory, the numerical contracts and GPU landing. This file adds what an agent working in the repository does. The maintainers' own process (lanes and worktrees, commit identity, the land queue and its trains, review and approval by kind of work, CI scope) is set by their operating rules (the bible, §4 for review and approval), and only security paths (`.github`'s secrets, the installers, credentials) wait on a maintainer's approval. Follow the user's current scope and approval boundaries.

## Scope and implementation

- Read the affected module, its callers, tests, and relevant design section before editing. Check JAX, Flax, Optax, Orbax, Grain, and the reference implementation for an existing primitive.
- Activations cross layer-to-kernel interfaces as `[batch, time, heads, head_dim]`. A kernel adapts to that layout; layer code does not permute activations for a kernel. Other tensors keep their own meaning.

## Parallel work and integration

- Inspect GPU processes and current lane ownership before launching GPU work. A process name alone does not establish whether the device is available.
- Keep private development tooling (personal cloud runners and session drivers, provisioning scripts, one-off session scripts) out of this repository. Tooling that Dew's own CI, tests, docs or users need, such as the CI entry points and the profiling and attribution tools, stays here.

## Verification

- Run the affected test files with pytest's cache enabled. Let the coordinating agent run integration suites; capture complete output and exit status once instead of rerunning a suite to recover failure names.
- In a worktree, pytest uses its configured source path; run scripts with `PYTHONPATH=src` so they import that worktree. Use the project's environment and commands documented in `CONTRIBUTING.md`.
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
- Report MFU as model FLOPs on real tokens over wall time, and name the precision and peak each utilization figure uses. A run below 50% MFU is profiled and its three largest losses fixed; the target is 50–60% or more on every workload that matters.
- The bar for a kernel is the best existing implementation for that operation on that device, measured in the same session at real training shapes. Write a custom kernel when the supplied one is the bottleneck. Use open projects as design references and keep their licence notices for anything ported.
- Claim a speed comparison only for the workload and hardware that were measured.

## Reporting

- Distinguish implemented behavior, tested behavior, research findings, and open work. Keep each requested item tracked until its acceptance checks pass or the user defers it.
- Report benchmark conditions and commands, exact checks run, remaining risks, and blocked prerequisites. Do not infer full framework parity or production readiness from small fixtures or simulated devices.
- Preserve the author's voice. Use the writing standards in `CONTRIBUTING.md` for docs, comments, commits, and replies.
