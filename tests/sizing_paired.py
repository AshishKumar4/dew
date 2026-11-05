"""A pytest plugin: each check-1 site sized by `seeds_for` paired with its Dew arm (up to MOST_SEEDS), timed,
in place of its verdict. One SIZED or UNRESOLVED line a site.

    pytest -p sizing_paired -s <tests>
"""

import functools
import time


def _sized(dew_fn, reference_fn, draw, label, *, seeds, truth_fn=None, orders=None, **_):
    import reference_error

    began = time.perf_counter()
    try:
        count = reference_error.seeds_for(reference_fn, draw, label, truth_fn=truth_fn, dew_fn=dew_fn,
                                          orders=orders)
        print(f"\nSIZED {label!r} paired {count} declared {seeds} {time.perf_counter() - began:.1f} s", flush=True)
    except AssertionError as error:
        print(f"\nUNRESOLVED {label!r} {time.perf_counter() - began:.1f} s: {str(error)[:400]}", flush=True)


def pytest_collection_modifyitems(session, config, items):
    for module in {item.module for item in items if getattr(item, "module", None) is not None}:
        if hasattr(module, "assert_as_exact"):
            module.assert_as_exact = _sized
