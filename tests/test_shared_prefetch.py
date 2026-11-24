"""Two mma.sync consumers' collective cp.async walk prefetching each step's first fragments (row 12695)."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
from generation_support import INTERPRETS
from jax.experimental.pallas import mosaic_gpu as plgpu
from jax.sharding import Mesh

from dew.nn.kernels import generation
from dew.nn.kernels.templates import bundles

INTERPRETED = plgpu.InterpretGPUParams()
ROWS, WIDTH, STEPS, CONSUMERS = 64, 64, 4, 2


def _prefetched_walk(x, segments: tuple[tuple[int, int], ...], interpret=INTERPRETED):
    """Two consumers sharing one cp.async pipeline walk `segments` of `x`'s `ROWS`-row steps, stated common,
    their bounds traced: each step adds its tile to an fp32 sum, its first row read ahead by the pipeline's
    prefetch; compiled (not `interpret`), its tile's square through the family's fragments
    (`Family.products_across`, `first_fragments`: mma.sync, which the interpreter runs none of). Returns
    each consumer's sum and the prefetches each ran (`[CONSUMERS]`, counted in global memory as they run)."""
    family = bundles.named("mma.sync", jnp.bfloat16, None)

    def body(x_gmem, sums_gmem, counts_gmem):
        allocated = family.streams({'x': ((ROWS, WIDTH), jnp.bfloat16)}, 2, CONSUMERS)
        wg = jax.lax.axis_index("wg")
        counts_gmem[wg] = jnp.int32(0)

        def walked(*, streams):
            first = jnp.int32(0)
            views = {'x': (x_gmem, lambda j: (j, 0))}
            feed = family.opened(streams, views)

            def prefetch(slots):
                counts_gmem[wg] = counts_gmem[wg] + 1
                if interpret is False:
                    return family.first_fragments(slots['x'], slots['x'])
                return slots['x'][0, :].astype(jnp.float32)

            def step(at, slots, carry, accumulators, *, prefetched, boundary):
                del at
                (acc,) = accumulators
                if interpret is False:
                    acc, following = family.products_across(acc, slots['x'], slots['x'], prefetched,
                                                            boundary)
                    return (carry, (acc,)), following
                tile = slots['x'][...].astype(jnp.float32)
                return (carry, (acc + tile,)), boundary()

            walk_segments = tuple((step, first + lo, first + hi) for lo, hi in segments)

            def walk(streams, lo, hi, start):
                del lo, hi, start
                return family.walk(streams, walk_segments, (), ((ROWS, WIDTH),), interpret,
                                   first=first, prefetch=prefetch)[1][0]

            streamed = (first, first + STEPS)
            return family.walk_own(feed, views, streamed, streamed, wg, walk, common=True)

        sums_gmem[wg] = family.scoped(walked, streams=allocated)

    sums = jax.ShapeDtypeStruct((CONSUMERS, ROWS, WIDTH), jnp.float32)
    counts = jax.ShapeDtypeStruct((CONSUMERS,), jnp.int32)
    return family.kernel(body, out_type=(sums, counts), grid=(1,), grid_names=("programs",),
                         consumers=CONSUMERS, interpret=interpret, name="prefetched")(x)


@INTERPRETS
def test_a_traced_empty_common_segment_prefetches_nothing():
    """Interpreted: two consumers' common segments, their bounds traced, of no step, one step and three steps
    each prefetch once as they start and once at each step's boundary, so the one-step and three-step
    segments prefetch 2 and 4 times; the empty one, its pipeline not run, none (where it prefetched its
    first slot, which no copy filled). Each consumer's sum is every walked step's tile."""
    x = jnp.asarray(np.arange(STEPS * ROWS * WIDTH).reshape(STEPS * ROWS, WIDTH) % 7, jnp.bfloat16)
    sums, counts = _prefetched_walk(x, ((0, 0), (0, 1), (1, STEPS)))
    np.testing.assert_array_equal(np.asarray(counts), [2 + 4] * CONSUMERS)
    expected = np.asarray(x, np.float32).reshape(STEPS, ROWS, WIDTH).sum(axis=0)
    for summed in np.asarray(sums):
        np.testing.assert_array_equal(summed, expected)


def test_a_prefetching_common_walk_lowers_for_an_a100():
    """The prefetching walk's mma.sync products and fragments, its traced segments' pipelines each under a
    branch on its trips, lower for an A100 on any host."""
    x = jax.ShapeDtypeStruct((STEPS * ROWS, WIDTH), jnp.bfloat16)
    mesh = generation.gpu_mesh(Mesh(np.array(jax.devices()[:1]), ("data",)), "sm80")
    walked = functools.partial(_prefetched_walk, segments=((0, 0), (0, 1), (1, STEPS)), interpret=False)
    with generation.assumed("sm80"), jax.sharding.use_abstract_mesh(mesh):
        jax.jit(walked).trace(x).lower(lowering_platforms=("cuda",))
