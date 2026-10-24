# fla family coverage (ledger rows 1446-1460): work in progress

Stopped before any family landed. No code changed. No ledger row was set in-progress.

## What was read

- Spec rows: docs/design/kernel-templates.md Appendix B Table 1 (a753b5933), rows for log-linear, MesaNet,
  TTT/Titans, DeltaFormer, MoM, CyFA, HGRN1, Momentum DeltaNet, RWKV4, Raven, Rodimus, PGDN, PKDA, PaTH.
- Ledger rows 1446-1460 (all `open P3`).
- The template: recurrence_algebra.py (PRODUCTS, decays, factor, intra, apply, carry, readout, compose,
  transition), recurrence_chunk.py (chunked, chunk_delta_rule and its custom VJP, chunk), recurrence.py
  (Hooks, update, step_reference), tests/test_recurrence_chunk.py (draw, token_by_token, check 1 sites),
  tests/reference_error.py (assert_as_exact, with_gradients, torch_with_gradients),
  tests/delta_rule_support.py (fla_side: fla in its own interpreter, DEW_FLA_PYTHON).

## Finding: the queue's fla reference is 0.5.2, the spec's rows cite 0.6.0

- The node provisions DEW_FLA_PYTHON with flash-linear-attention==0.5.2 and fla-core==0.5.2 on
  torch==2.14.1+cu126 (ops-stable dewops/node/session.py, `FLA`).
- fla 0.5.2 has no cyfa, momentum_delta_rule, precond_gated_delta_rule, precond_kda, atk or lightnet ops
  (its fla/ops list, /mnt/local/dew/tmp/DeafMoth/fla-0.5.2/fla/ops). Those families' executed references
  need 0.6.0.
- fla-core 0.6.0 is on PyPI (pure Python, requires einops and torch>=2.7, triton>=3.3). Its fla/ops is
  413ad2c's byte for byte, apart from the READMEs and one PDF (`diff -rq` of the wheel against fla-src).
- Proposed route for the A100 check 1 against fla, without changing the queue's 0.5.2 pin: the job
  installs 0.6.0 as an overlay into the reference's interpreter, and the test's fla side prepends it to
  sys.path:
  `uv pip install --python "$DEW_FLA_PYTHON" --target "$OUT/fla060" --no-deps fla-core==0.6.0`. The
  helper reads the overlay's path from an environment variable and fails by name where it is unset, as
  fla_side does for DEW_FLA_PYTHON. The GPU tier deselects only `network` tests, so the reference test
  should be marked to need the overlay (or the queue pins 0.6.0 beside 0.5.2: the ops owner's decision).

## Planned order and the composition each family takes (none started)

1. MoM: GDN per memory (chunk with the scalar-decay delta hooks) over routed tokens, top-2 of 8. Group
   metadata as ragged per-memory sequences (packed `starts`, not padded max_len), then a routing-weighted
   combine in a fixed order. Truth: per-memory token_by_token in float64 with the same routing.
2. Log-linear: scalar decay with Fenwick-level weights. A bank of per-level states on the static bit
   schedule of the chunk index, a level-indexed intra mask, and readout Σ_ℓ λ_ℓ qᵀS_ℓ.
3. HGRN1: per-value decay at key width 1, factor skipped, carry only. Existing shapes; the vector decay
   with Dk = 1 and the values as the write.
4. Momentum DeltaNet: (S, M) under the block T [[α, −βμ], [0, μ]] with the untied erase (p, ηk).
5. MesaNet: H_kk and H_kv share the scalar decay, with a readout by 30 static CG steps.
6. TTT-Linear, Titans (Titans has no fla kernel, so its speed bar is unknown), DeltaFormer, CyFA, RWKV4,
   Raven, Rodimus, PGDN, PKDA, PaTH: per the spec rows.

## Worktree

/mnt/local/dew/worktrees/recurrence-families, branch feat/recurrence-families, based on
origin/perf/recurrence-sm90-fwd e3f6786d74.
