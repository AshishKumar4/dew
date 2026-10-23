import sys
sys.path[:0] = ["tests", "src", "."]
import tools.gate_null_runs as g
from reference_error import REDRAW, _drawn, _staged
reference, draw, truth, orbit = g.CLASSES["mean-loss-rows"]()
span = (1 + REDRAW) * 8
shown = 0
for run in range(g.OFFSET, g.OFFSET + 200):
    base = run * 2 * span
    def drawn(start, stop, base=base):
        return _drawn(reference, reference, draw, range(base + start, base + stop), truth,
                      reference_seeds=range(base + span + start, base + span + stop))
    over, unresolved, stage = _staged(drawn, 8, "mean-loss-rows")
    if over:
        print("OVER", run, stage, " | ".join(over)[:400], flush=True)
        shown += 1
        if shown >= 6:
            break
