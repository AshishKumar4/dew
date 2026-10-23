"""Check 1's worst-row rule on equally exact pairs as sites run it: both arms on one truth (shared row
sizes), two stages (8 declared seeds, then 40), R rows a seed of w values, truths and errors standard
normal. Counts false reds at a factor c, for a chunk of runs from its own stream."""
import json, sys
import numpy as np

R = 256

def reds(w, c, runs, chunk, block=200):
    rng = np.random.default_rng([0x5A, w, chunk])
    count = 0
    for start in range(0, runs, block):
        b = min(block, runs - start)
        t = rng.standard_normal((b, 40, R, w))
        s = np.sqrt(np.mean(t * t, axis=3))
        r_d = np.sqrt(np.mean(np.square(rng.standard_normal((b, 40, R, w))), axis=3)) / s
        r_r = np.sqrt(np.mean(np.square(rng.standard_normal((b, 40, R, w))), axis=3)) / s
        stage1 = r_d[:, :8].max(axis=(1, 2)) > c * r_r[:, :8].max(axis=(1, 2))
        stage2 = r_d.max(axis=(1, 2)) > c * r_r.max(axis=(1, 2))
        count += int((stage1 & stage2).sum())
    return count

w, c, runs, chunk = int(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
print(json.dumps({"w": w, "c": c, "runs": runs, "chunk": chunk, "reds": reds(w, c, runs, chunk)}))
