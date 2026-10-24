# GEMM rows 12, 10 and 4 (WIP notes; no code yet)

Base: origin/base/kernels d8073b98d8.

## Row 12: per-row scale

- The forward already exists. A 1-D `(s, ("rows",))` epilogue side whose hook broadcasts it itself
  (`acc * s[:, None]`) gets past `_split`/`_launchable` without a squeezed position, so
  `_Mosaic._hooks_lower`'s broadcast refusal (grouped_matmul.py:655-663) does not apply. Only the
  `[M, 1]` `("rows", None)` form is refused on Mosaic GPU.
- tests/test_gemm_template.py:111 "row scaled" (`scaled_rows`) is in CASES (forward on sm80, sm90 and v5e,
  interpreted) and in the cooperative-tile test (:208). It came in with eda2753570.
- Gaps:
  1. "row scaled" is not in GRADIENT_CASES (:123), so the gradient is untested.
  2. No H100 lowering test covers a rows-side epilogue: test_the_hooked_mosaic_launches_lower_for_an_h100
     (:821) uses only a columns bias. Whether Mosaic's layout inference takes a `[rows_half]` vector's
     broadcast onto the WGMMA accumulator is unknown until a native lowering or H100 run shows it.
  3. The backward is poor. `_reads_accumulator` is True, so template.py:101-104 recomputes the whole
     product into fp32. The VJP's d_acc = s*dy is fp32, so `narrow` is False and work=fp32: dx and dW
     run as fp32 products (Mosaic is not eligible, Triton runs tf32 or XLA). That makes three GEMMs, two
     of them fp32, where the unfused sequence has two bf16 GEMMs.
- Cheaper d_s, which avoids the recompute: d_s = rowsum(dy * (xW)) = rowsum((dy W^T) * x). The dx GEMM
  computes g = dy W^T on bf16 dy. Its epilogue applies dx = s*g, and rowsum(g*x) is taken either as
  per-column-tile partials or as an XLA reduce. dW = (s*x)^T dy goes through tgmm's prologue on 'rows'.
  This keeps every product bf16. Design C.12 says "d_s via per-tile partials, reduce over column
  tiles", and this identity puts those partials in the dx launch rather than in a recompute.
- MoE consumer (moe.py:946-951 `_combine`): today slots come out bf16, and XLA fuses the weights
  multiply into the gather and k-sum (an fp32 einsum). The forward saving from fusion is about zero. The
  backward needs d_w = <dy, slots>; today that reads the saved slots. The fused form needs the recompute
  (a loss) or the identity above (no saved slots: a memory win of [T*k, D] per layer). Not measured yet:
  it needs an H100 A/B of down plus combine, fwd and fwd+bwd, time and peak.

## Row 10: per-expert bias [G, N]

- Not started. Design C.10 names a new side axis 'groups', loaded at the tile's group from the tile
  table (Mosaic: `GroupInfo.group_id`; Triton ragged_dot: the program's group). dbias is a per-group
  column sum of dy, which is XLA's segment_sum over sorted rows or partials per (group, column tile).
  Today's consumer is moe.py:426 `gather_expert_bias` plus an add pass.

## Row 4: batched

- Not started. Plan: equal static group sizes (the batch as the group axis, design C.4). Route through
  the template only where an H100 measures it faster than XLA's batched dot_general.
