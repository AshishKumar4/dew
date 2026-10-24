"""Scratch plugin (`-p tests.seeds_spread`): each check-1 site's per-seed errors over 64 paired draws,
for its eight widest leaves: the quantiles of Dew's and the reference's RMS errors and of their per-seed
ratio, and the count of seeds the two arms agree on to the bit. Passes every site."""
import numpy as np

lines = []


def pytest_configure(config):
    import reference_error

    def spread(dew_fn, reference_fn, draw, label, *, seeds, truth_fn=None, orders=None):
        start = reference_error.SIZING_FROM
        drawn = reference_error._drawn(dew_fn, reference_fn, draw, range(start, start + 64), truth_fn)
        widths = []
        for index, name in enumerate(drawn.names):
            mine, theirs = drawn.mine[:, index], drawn.theirs[:, index]
            if np.sum(theirs) == 0:
                continue
            low, high = reference_error._interval(mine, theirs)
            widths.append((high / max(low, 1e-300), index, name))
        widths.sort(reverse=True)
        for width, index, name in widths[:8]:
            mine, theirs = np.sqrt(drawn.mine[:, index]), np.sqrt(drawn.theirs[:, index])
            ratio = mine / np.maximum(theirs, 1e-300)
            same = int(np.sum(mine == theirs))
            quantiles = (0.0, 0.1, 0.5, 0.9, 1.0)
            lines.append(f"{label!r}{name}: width {width:.2f}; dew {np.quantile(mine, quantiles)}; "
                         f"reference {np.quantile(theirs, quantiles)}; ratio {np.quantile(ratio, quantiles)}; "
                         f"{same} of 64 seeds alike")

    reference_error.assert_as_exact = spread


def pytest_terminal_summary(terminalreporter):
    for line in lines:
        terminalreporter.write_line(f"SPREAD {line}")
