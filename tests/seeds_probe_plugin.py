"""Replace check 1 with seeds_for, printing each site's declared seed count."""
import reference_error


def _sized(dew_fn, reference_fn, draw, label, *, seeds, truth_fn=None):
    count = reference_error.seeds_for(reference_fn, draw, label, truth_fn=truth_fn)
    print(f"SEEDS {label!r}: declared {seeds}, seeds_for {count}", flush=True)


reference_error.assert_as_exact = _sized
