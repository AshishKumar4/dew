# Working on Dew

Read `CONTRIBUTING.md` for design, reference parity, code, tests, and writing standards. This checklist adds agent workflow rules; it does not duplicate that contract. Follow the user's current scope and approval boundaries.

## Scope and implementation

- Read the affected module, its callers, tests, and relevant design section before editing. Check JAX, Flax, Optax, Orbax, Grain, and the reference implementation for an existing primitive.
- Preserve checkpoint and reference-layout requirements described in `CONTRIBUTING.md`. Before 1.0, migrate callers together without compatibility shims.
- Keep code that owns meaningful complexity. Apply the deletion test to helpers; caller count alone does not decide whether a helper belongs.
- Preserve outputs and precision when optimizing an existing path. Treat requested quantization or approximate methods as explicit features with separate accuracy and performance evidence.

## Parallel work and integration

- Give each parallel writer an isolated worktree and clear file ownership. Coordinate shared-file edits and GPU use before starting them. Keep at most four expert/slow sessions active; use no Sonnet models.
- Inspect GPU processes and current lane ownership before launching GPU work. A process name alone does not establish whether the device is available.
- Commit completed work as `Ashish Kumar Singh <ashishkmr472@gmail.com>` with a concise conventional commit message. Subagents report commits without pushing. The coordinating agent may push when the user authorizes it; never force-push.
- Preserve tracked and untracked work before integration. Remove a worktree only after its work is committed, verified, merged, and no process still needs it.

## Verification

- Run the affected test files with pytest's cache enabled. Let the coordinating agent run integration suites; capture complete output and exit status once instead of rerunning a suite to recover failure names.
- In a worktree, pytest uses its configured source path; run scripts with `PYTHONPATH=src` so they import that worktree. Use the project's environment and commands documented in `CONTRIBUTING.md`.
- Use small deterministic cases for logic and representative GPU/TPU cases for device behavior. Confirm the test size exercises the intended failure without constructing a production model by accident.
- For numerical changes, record reference version, inputs, dtype, backend, command, observed error, and the reason for the tolerance. Investigate a failing bound before changing it.
- A shell pipeline's last command succeeding does not prove tests passed. Stop integration on a failed check; fix and rerun the affected behavior.

## Reporting

- Distinguish implemented behavior, tested behavior, research findings, and open work. Keep each requested item tracked until its acceptance checks pass or the user defers it.
- Report benchmark conditions and commands, exact checks run, remaining risks, and blocked prerequisites. Do not infer full framework parity or production readiness from small fixtures or simulated devices.
- Preserve the author's voice. Use the writing standards in `CONTRIBUTING.md` for docs, comments, commits, and replies.
