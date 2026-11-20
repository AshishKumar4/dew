"""One builder of a Blackwell's gathered persistent walk for the tests that run or read one
(`gathered_kernel`): the protocol checker's walks (tests/test_ring_protocol.py) and the strided-table cell
(tests/test_gemm_gather.py), each its tiles, its streams' views and its stores (row 8679: two copies of the
walk's body)."""

from collections.abc import Callable

import jax
import jax.numpy as jnp

from dew.nn.kernels import generation
from dew.nn.kernels.templates import mosaic_sm100


def tile_value(tile, values: tuple[int, ...]):
    """A tile's value of `values`, scalars alone (a kernel closes over no array)."""
    return sum(jnp.where(tile == index, jnp.int32(value), jnp.int32(0)) for index, value in enumerate(values))


def gathered_kernel(blocks: dict, gathered: tuple[str, ...], tiles: tuple[int, ...], views: Callable,
                    accumulator: tuple[int, int], *, stages: int, out: jax.ShapeDtypeStruct,
                    stored: bool, name: str) -> Callable:
    """The tcgen05 family's persistent walk (`mosaic_sm100.RINGS.persist`) over `blocks`' streams (`{name:
    (block, dtype)}`, those `gathered` names `gathered.Committed`'s: each step's rows where a table says, its
    tiles spread over the issuing warpgroup's load warps; `w` the plain one), `tiles[t]` steps a tile,
    `views(operands, tile)` each stream's view of the kernel's operands at a tile; each step a tcgen05
    product of `w`'s slot times each gathered one into `accumulator`, the products' commit releasing them;
    where `stored`, each tile's accumulator stored to the kernel's last operand (`out`), else nothing:
    a kernel over its operands, built as an assumed B200 builds it."""
    family = mosaic_sm100.RINGS

    def body(*operands):
        def scope(*, streams):
            def schedule(index):
                tile = index[0]
                steps = tile_value(tile, tiles)
                return steps > 0, steps, views(operands, tile), tile

            def issue(wg, tile, walk, *, active):
                del wg, tile, active

                def step(at, slots, carry, accumulators):
                    del at
                    (acc,) = accumulators
                    for each in gathered:
                        acc = family.products(acc, slots["w"], slots[each])
                    return carry, (acc,)

                return walk(step, None, (accumulator,))[1] if stored else walk(step, None, (accumulator,))

            def finish(wg, tile, accs, store, *, active):
                del wg, tile, active
                if not stored:
                    return
                for first, (value,) in family.chunks(accs):
                    store(0, value, operands[-1], (0, 0, accumulator[0], first, accumulator[0]))

            family.persist(streams, (len(tiles),), schedule, issue, finish, consumers=1, axis="programs",
                           outputs=((accumulator, jnp.float32),) if stored else (), interpret=False,
                           accumulators=(accumulator,))

        family.scoped(scope, streams=family.streams(blocks, stages, 1, gathered=gathered))

    with generation.assumed("sm100"):
        return family.kernel(body, out_type=out, grid=(1,), grid_names=("programs",), consumers=1,
                             interpret=False, name=name)
