"""Scratch sizing plugin (`-p tests.seeds_sizing`): every check-1 site the run reaches is sized paired
(`seeds_for` with Dew's arm as `dew_fn`) and then judged at the larger of that count and its declared
one; the counts, and each leaf's reference error from the truth over the truth's size on two draws (a
truth of other draws reads near one), are printed at the end of the run."""
import numpy as np

sized = []


def pytest_configure(config):
    # tests/ is on the path once its conftest has loaded, which is before this hook.
    import jax
    import reference_error

    judged = reference_error.assert_as_exact

    def relative(reference_fn, draw, truth_fn):
        """Each leaf's reference error over its truth's size on seed 0: the median, and the five worst
        with their sizes."""
        inputs = draw(0)
        with jax.enable_x64(new_val=True):
            truth = (truth_fn or reference_fn)(*reference_error.widened(inputs))
        rows = []
        for (path, mine), true in zip(jax.tree_util.tree_leaves_with_path(reference_fn(*inputs)),
                                      jax.tree.leaves(truth), strict=True):
            true = np.asarray(true, np.float64)
            size = float(np.sqrt(np.mean(np.square(true))))
            error = float(np.sqrt(np.mean(np.square(np.asarray(mine, np.float64) - true))))
            rows.append((error / size if size > 0 else float("inf"), size, jax.tree_util.keystr(path)))
        rows.sort()
        worst = "; ".join(f"{name} {ratio:.2e} (size {size:.2e})" for ratio, size, name in rows[-5:])
        return f"median {rows[len(rows) // 2][0]:.2e} of {len(rows)} leaves; worst {worst}"

    def sizing(dew_fn, reference_fn, draw, label, *, seeds, truth_fn=None, orders=None):
        worst = relative(reference_fn, draw, truth_fn)
        try:
            count = reference_error.seeds_for(reference_fn, draw, label, truth_fn=truth_fn, dew_fn=dew_fn,
                                              most=1024)
        except AssertionError as refused:
            sized.append(f"{label!r}: declared {seeds}, paired seeds_for refused: {str(refused)[:300]}; "
                         f"reference/truth {worst}")
            raise
        sized.append(f"{label!r}: declared {seeds}, paired seeds_for {count}; reference/truth {worst}")
        judged(dew_fn, reference_fn, draw, label, seeds=max(count, seeds), truth_fn=truth_fn, orders=orders)

    reference_error.assert_as_exact = sizing


def pytest_terminal_summary(terminalreporter):
    for line in sized:
        terminalreporter.write_line(f"SEEDS {line}")
