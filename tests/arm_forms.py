"""How the gate runs an arm of one draw beyond one call: over a site's windows of stacked draws (`windows`,
`over_windows`) and declared to vary from run to run (`run_to_run`), each an adapter `reference_error`'s
check 1 reads through (`_layers`)."""

import jax
import jax.numpy as jnp
import numpy as np
from arm_arithmetic import Unrewritten


def _stack_draws(trees):
    """A draw's leaves of each window stacked on a new leading axis. A JAX array or Python scalar stacks
    as JAX stacks it, its weak type kept, and a host array as NumPy stacks it, its dtype kept."""
    def stack(*leaves):
        if all(isinstance(leaf, jax.Array | bool | int | float | complex) for leaf in leaves):
            return jnp.stack([jnp.asarray(leaf) for leaf in leaves])
        return np.stack([np.asarray(leaf) for leaf in leaves])
    return jax.tree.map(stack, *trees)


def stacked_on_host(trees):
    """Outputs (an arm's of each window, a site's of each example) stacked on a new leading axis as host
    arrays of their own dtypes."""
    return jax.tree.map(lambda *leaves: np.stack([np.asarray(leaf) for leaf in leaves]), *trees)


def _window_count(arguments) -> int:
    """The windows an arm's stacked arguments hold, the leading axis of each."""
    leaves = jax.tree.leaves(arguments)
    count = len(leaves[0]) if leaves else 0
    if count < 1:
        raise ValueError(f"an arm over {count} windows")
    return count


def windows(draw, count: int):
    """A check-1 draw of `count` windows a seed. Seed `s` gives the site's draws of seeds `s * count` to
    `s * count + count - 1`, each leaf stacked on a new leading axis, and every argument is a window. A
    seed's evidence then pools `count` draws, as independent as the site's draws of different seeds
    are. A leaf that is one row in a draw, such as a parameter's gradient, becomes `count` rows. Its
    arms take the windows through `over_windows`."""
    if count < 1:
        raise ValueError(f"windows of {count} draws")

    def drawn(seed):
        return _stack_draws([draw(seed * count + index) for index in range(count)])
    return drawn


class _Windowed:
    """An arm of one draw (`fn`) over `windows`' stacked arguments, window by window (`over_windows`):
    the gate compiles `fn` as it compiles any arm (`_arms`) and runs the windows through that program."""

    def __init__(self, fn):
        self.fn = fn

    def __call__(self, *arguments):
        return stacked_on_host([self.fn(*jax.tree.map(lambda leaf, index=index: leaf[index], arguments))
                            for index in range(_window_count(arguments))])


class _RunToRun:
    """An arm whose outputs vary in their last bits from run to run on the same inputs (`run_to_run`):
    the gate compiles `fn` as it compiles any arm (`_arms`) and judges each draw by the one run it makes."""

    def __init__(self, fn, why: str):
        self.fn, self.why = fn, why

    def __call__(self, *inputs, **options):
        return self.fn(*inputs, **options)

    def __getattr__(self, name):
        return getattr(self.fn, name)


def _layers(fn):
    """`fn` and every arm it adapts, outermost first: through a declaration (`_RunToRun`), a window loop
    (`_Windowed`) and a reference's unrewritten scope (`Unrewritten`)."""
    while True:
        yield fn
        if isinstance(fn, _RunToRun | _Windowed):
            fn = fn.fn
        elif isinstance(fn, Unrewritten):
            fn = fn.wrapped
        else:
            return


def _varies(fn) -> bool:
    """Whether `fn`, or an arm it adapts, is declared to vary from run to run (`run_to_run`)."""
    return any(isinstance(layer, _RunToRun) for layer in _layers(fn))


def run_to_run(fn, why: str):
    """`fn`, an arm declared to vary from run to run on a GPU on the same inputs, for `why`: an
    accumulation in arrival order (atomic adds), which a deterministic form would make 2-2.7 times as slow
    (row 4448). Check 1 pairs it with the other arm within each draw's one run, as it pairs every
    arm, and never holds it to a run of itself again (`_of_its_inputs`) but on a CPU, where its interpreted
    program is a function of its inputs: a draw's evidence is one run's, and the redraw and every further
    seed are runs of their own, so its spread across runs is the spread across draws check 1 weighs. A
    window loop's arm (`over_windows`) is declared inside its loop, which the gate reads outermost."""
    if isinstance(fn, _Windowed):
        return _Windowed(_RunToRun(fn.fn, why))
    return _RunToRun(fn, why)


def over_windows(fn):
    """`fn`, an arm of one draw, run over `windows`' stacked arguments window by window, each window's
    arguments sliced from the stack, with its outputs stacked on their leading axis as host arrays.

    Each window runs the program one draw runs, never a batched one: a batch axis changes the program
    XLA compiles (its fusions, its reductions' order, its dots' batching, a kernel's route), which XLA
    holds to no common bits with one draw's, so check 1 would judge a program users never run. The gate
    compiles `fn` as it compiles an arm of one draw (`_arms`), once a signature, and every window calls
    that program; a torch or NumPy reference passes `as_given`. The loop keeps `fn`, which a recorded
    truth's rounding count traces on one window (`indexed_recording`)."""
    return _Windowed(fn)


def _one_draws_program(fn, arguments):
    """`fn` and its `arguments` as one draw's program reads them: a window loop's (`over_windows`) own
    function on its first window's arguments, any other function on them all."""
    if not isinstance(fn, _Windowed):
        return fn, arguments
    return fn.fn, jax.tree.map(lambda leaf: leaf[0], arguments)
