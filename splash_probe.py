"""Where Splash's dq is nonzero at the float64 oracle's exact zeros, and how many keys those query rows see."""
import sys
sys.path.insert(0, "tests")
import lane_environment  # noqa: F401
import numpy as np
import jax
from test_attention_splash import (ORACLE_CASES, qkv, value_and_grads, exact_value_and_grads, reference,
                                   document_mask, packed_ids)
import jax.numpy as jnp

for case in ("mask", "packed-window", "window", "causal", "grouped", "full"):
    q_shape, kv_shape, structure = ORACLE_CASES[case]
    query, _, _ = qkv(q_shape, jnp.float32)
    _, key, value = qkv(kv_shape or q_shape, jnp.float32, seed=1)
    structure = dict(structure)
    structure.pop("sinks", None)
    if structure.pop("mask", False):
        structure["mask"] = np.asarray(document_mask(packed_ids((100, 84, 72))[:1]))[:, None]
    if structure.pop("segment_ids", False):
        structure["segment_ids"] = packed_ids((100, 84, 50), (30, 190, 10))
        structure["live"] = structure["segment_ids"] != 0
    splash = value_and_grads('tpu', query, key, value, **structure)
    truth = exact_value_and_grads(query, key, value, **structure)
    ref = reference(query, key, value, **structure)
    dq, dq_true, dq_ref = splash[1], truth[1], ref[1]
    # visible keys per (b, q, h): rebuild the keep mask as exact_value_and_grads does
    b, s, h, d = query.shape
    kl = key.shape[1]
    rows, cols = np.arange(s)[:, None], np.arange(kl)[None, :]
    keep = np.ones((1, 1, s, kl), bool)
    if structure.get("causal"): keep = keep & (cols <= rows)
    if structure.get("sliding_window") is not None: keep = keep & (np.abs(cols - rows) < structure["sliding_window"])
    if structure.get("mask") is not None: keep = keep & np.asarray(structure["mask"], bool)
    if structure.get("segment_ids") is not None:
        ids = np.asarray(structure["segment_ids"]); keep = keep & (ids[:, None, :, None] == ids[:, None, None, :])
    visible = np.broadcast_to(keep, (b, 1, s, kl)).sum(-1)[:, 0]  # [b, s]
    live = np.asarray(structure.get("live", np.ones((b, s), bool)))
    zero = dq_true == 0
    bad = zero & (dq != 0)
    rms = np.sqrt(np.mean(dq_true ** 2))
    rows_bad = np.unique(np.argwhere(bad.any(-1))[:, :2], axis=0) if bad.any() else np.zeros((0, 2), int)
    print(f"== {case}: true zeros {zero.sum()}, splash nonzero there {bad.sum()} (max {np.abs(dq[bad]).max() if bad.any() else 0:.3e}, "
          f"{(np.abs(dq[bad]).max() / rms) if bad.any() else 0:.2e} of RMS); reference nonzero there {(zero & (dq_ref != 0)).sum()}")
    if len(rows_bad):
        vis = [int(visible[bq, qq]) for bq, qq in rows_bad]
        lv = [bool(live[bq, qq]) for bq, qq in rows_bad]
        print("   rows:", len(rows_bad), "visible-key counts:", sorted(set(vis)), "live:", sorted(set(lv)))
        print("   first rows:", [(int(a), int(c), visible[a, c], bool(live[a, c])) for a, c in rows_bad[:12]])
    zrows = np.argwhere(zero.all(-1).all(-1)) if zero.ndim == 4 else []
    fully = (visible == 0)
    print("   fully masked rows:", int(fully.sum()), " single-key rows:", int((visible == 1).sum()), " dead rows:", int((~live).sum()))
