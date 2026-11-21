"""A recording of a reference over seeds read whole (`reference_error.indexed_recording`'s path for a
recording that keeps the reference's outputs, not its evidence alone): its draw, reference and truth, each
seed's recorded row, a pilot's (from SIZING_FROM on) where the recording holds pilots; and a recorded
reference's pilots read as a view of it (`recorded_pilots`)."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
from arm_arithmetic import drawn_on_the_host
from arm_forms import as_recorded
from reference_error import SIZING_FROM, input_digest

from dew.parity_stages import REDRAW, staged_units


def whole_recording(stored, site: str, seeds: int, inputs_of):
    """`indexed_recording`'s draw, reference and truth of a whole recording `stored` of `site`, which
    covers check 1 over `seeds` (its account's, `reference_error.indexed_recording`), its pilots
    (`<site>.pilot.input.<name>`, `.reference.`, `.truth.`, `.replayed.`, and `.input_digest` where
    `inputs_of` draws its inputs again) read at seeds from SIZING_FROM on and carried by its reference
    (`as_recorded`'s `pilots`), so a sizing of it takes them alone (`seeds_for`)."""
    def tree(kind):
        prefix = f"{site}.{kind}."
        return {name.removeprefix(prefix): value for name, value in stored.items() if name.startswith(prefix)}

    inputs, reference, truth, replayed = tree("input"), tree("reference"), tree("truth"), tree("replayed")
    narrow = [name for name, value in truth.items()
              if jnp.issubdtype(value.dtype, jnp.floating) and value.dtype != np.float64]
    if narrow:
        raise AssertionError(f"{site}: the recorded truth {narrow} is not float64; record it in float64, as "
                             f"no later widening recovers what a narrower recording dropped")
    if inputs_of is None and not inputs:
        raise AssertionError(f"{site}: no inputs recorded and no `inputs_of` to draw them again")
    if int(stored.get(f"{site}.redraw", -1)) != REDRAW:
        raise AssertionError(f"{site}: the recording holds no `{site}.redraw` of {REDRAW}, which "
                             f"record_case writes and the fetched store's rule reads it by")
    digests = stored.get(f"{site}.input_digest")
    if inputs_of is not None and digests is None:
        raise AssertionError(f"{site}: `inputs_of` draws the inputs again and the recording keeps no "
                             f"digest to check them by; record the outputs alone (`store_inputs=False`)")
    recorded = [*reference.values(), *truth.values(),
                *(inputs.values() if inputs_of is None else [digests])]
    count = min(len(value) for value in recorded)
    if count < staged_units(seeds):
        raise AssertionError(f"{site}: {count} recorded seeds, and check 1 over {seeds} may draw "
                             f"{staged_units(seeds)}; record (1 + REDRAW) times the declared seeds")

    pilot = {kind: tree(f"pilot.{kind}") for kind in ("input", "reference", "truth", "replayed")}
    pilot_digests = stored.get(f"{site}.pilot.input_digest")
    pilots = _pilots(site, pilot, pilot_digests, {"input": inputs, "reference": reference, "truth": truth,
                                                  "replayed": replayed}, drawn_again=inputs_of is not None)

    def at(seed: int, verdict: dict, piloted: dict):
        """`seed`'s row: a verdict seed's in `verdict`, a pilot's (from SIZING_FROM on) in `piloted`."""
        if seed < SIZING_FROM:
            return verdict, seed
        if seed >= SIZING_FROM + pilots:
            raise AssertionError(f"{site}: seed {seed} is past the recording's {pilots} pilots from "
                                 f"SIZING_FROM")
        return piloted, seed - SIZING_FROM

    def draw(seed):
        if inputs_of is None:
            held, row = at(seed, inputs, pilot["input"])
            drawn = {name: value[row] for name, value in held.items()}
        else:
            drawn = drawn_on_the_host(inputs_of, seed)
            kept, row = at(seed, {"digest": digests}, {"digest": pilot_digests})
            if input_digest(drawn) != kept["digest"][row]:
                raise AssertionError(f"{site}: seed {seed}'s inputs drawn again are not the recorded ones "
                                     f"(their digest differs); the generator's draw changed since the "
                                     f"recording")
        held, row = at(seed, replayed, pilot["replayed"])
        return np.int32(seed), {**drawn, **{name: value[row] for name, value in held.items()}}

    def lookup(values, piloted):
        # A draw's recorded leaves are read from the host at the draw's index, in the dtypes the caller's
        # context holds them in. A recording is never a program's constants, which exhaust a host's memory
        # where a weighed recording keeps thousands of draws whole. An arm that wraps this one and is traced
        # reads at run time (`jax.pure_callback`); JAX keeps a callback as long as its program, so the
        # gate's own read, its index in hand, goes through none.
        def read(index):
            held, row = at(int(index), values, piloted)
            return {name: value[row] for name, value in held.items()}

        def recorded(index, inputs):
            if not isinstance(index, jax.core.Tracer):
                return {name: jnp.asarray(value) for name, value in read(index).items()}
            kept = {name: jax.ShapeDtypeStruct(value.shape[1:], jax.dtypes.canonicalize_dtype(value.dtype))
                    for name, value in values.items()}
            return jax.pure_callback(
                lambda index: {name: np.asarray(value, kept[name].dtype)
                               for name, value in read(index).items()},
                kept, index, vmap_method="sequential")
        # Its pilots, where it holds them, a sizing's draws (`seeds_for`, `_paired_most`).
        return as_recorded(recorded, range(SIZING_FROM, SIZING_FROM + pilots) if pilots else None)

    return draw, lookup(reference, pilot["reference"]), lookup(truth, pilot["truth"])


def _pilots(site: str, pilot: dict, digests, verdict: dict, *, drawn_again: bool) -> int:
    """How many pilots a whole recording holds: none where it holds no pilot entry; else each kind's leaves
    (`pilot`: input, reference, truth, replayed) named as the verdict's are, each row of the verdict's shape
    and dtype, its inputs' digests (`digests`) beside them where its inputs are drawn again (`drawn_again`),
    and every array one row a pilot: a sizing on pilots that hold some of the verdict's leaves alone, or
    another workload's, would certify a count none of the verdict's draws sized."""
    if not any(pilot.values()) and digests is None:
        return 0
    differing = sorted(kind for kind, leaves in verdict.items() if set(pilot[kind]) != set(leaves))
    if differing or (drawn_again and digests is None):
        raise AssertionError(f"{site}: its pilots hold other leaves than its verdict seeds' ({differing}) or "
                             f"no digest of inputs drawn again; record every leaf of every pilot")
    unlike = sorted(f"{kind}.{name}" for kind, leaves in verdict.items() for name, value in leaves.items()
                    if (np.shape(pilot[kind][name])[1:], np.asarray(pilot[kind][name]).dtype)
                    != (np.shape(value)[1:], np.asarray(value).dtype))
    if unlike:
        raise AssertionError(f"{site}: its pilots' {unlike} differ from its verdict seeds' in a row's shape "
                             f"or dtype; a pilot is a draw of the verdict's workload")
    counts = {len(value) for leaves in pilot.values() for value in leaves.values()}
    counts |= set() if digests is None else {len(digests)}
    if len(counts) != 1:
        raise AssertionError(f"{site}: its pilot arrays hold {sorted(counts)} rows; one row a pilot in each")
    return counts.pop()


def recorded_pilots(reference_fn, draw, count: int):
    """A view of `reference_fn`, a recorded reference (one `.recorded`), that holds `count` pilot draws
    from SIZING_FROM on, disjoint from every verdict's seeds, the reference and its truth recorded on the
    inputs Dew's arm takes live: the draws a paired sizing of it takes (`seeds_for`). The view is the site's
    own; `reference_fn` is left as it is. Every pilot is read now (`draw`'s inputs at each), so a
    recording that does not hold them all is refused here, not at the doubling that reaches one."""
    if not getattr(reference_fn, "recorded", False):
        raise ValueError("pilots belong to a recorded reference; a live one draws its own from SIZING_FROM")
    if count < 1:
        raise ValueError(f"a recording of {count} pilots")
    for seed in range(SIZING_FROM, SIZING_FROM + count):
        reference_fn(*draw(seed))

    @functools.wraps(reference_fn)
    def view(*inputs):
        return reference_fn(*inputs)
    return as_recorded(view, range(SIZING_FROM, SIZING_FROM + count))
