"""Scratch sizing plugin (`-p tests.seeds_unpaired`): every check-1 site the run reaches is sized
unpaired (`seeds_for`, the reference against itself, up to 4096 seeds) and then judged at the larger
of that count and its declared one; the counts are printed at the end of the run."""

sized = []


def pytest_configure(config):
    # tests/ is on the path once its conftest has loaded, which is before this hook.
    import reference_error

    judged = reference_error.assert_as_exact

    def sizing(dew_fn, reference_fn, draw, label, *, seeds, truth_fn=None, orders=None):
        try:
            count = reference_error.seeds_for(reference_fn, draw, label, truth_fn=truth_fn, most=4096)
        except AssertionError as refused:
            sized.append(f"{label!r}: declared {seeds}, unpaired seeds_for refused: {str(refused)[:400]}")
            raise
        sized.append(f"{label!r}: declared {seeds}, unpaired seeds_for {count}")
        judged(dew_fn, reference_fn, draw, label, seeds=max(count, seeds), truth_fn=truth_fn, orders=orders)
        sized.append(f"{label!r}: judged at {max(count, seeds)}: pass")

    reference_error.assert_as_exact = sizing


def pytest_terminal_summary(terminalreporter):
    for line in sized:
        terminalreporter.write_line(f"SEEDS {line}")
