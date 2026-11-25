"""The dense GEMM's routes and resources as its receipts and its reference's state them: each GPU
generation's declared cells derived from the receipts that timed its products (`grouped_cells`,
`grouped_matmul.DECLARED`), and a split launch's fp32 partials within the workspace the reference's
library holds (`GpuTarget.reference_workspace`, row 12850)."""

import dataclasses
import functools

import jax
import jax.numpy as jnp
import pytest

from dew.nn.kernels import gemm, generation, grouped_cells, grouped_matmul, selection


def _dense_case(generation_name: str, made: grouped_cells.Made, matrix: tuple[int, int], rows: int,
                dtype=jnp.bfloat16, stored=None) -> grouped_matmul.GroupedCase:
    """The `GroupedCase` of a dense product of `dtype` operands on `generation_name` making `made` of `matrix`
    `[in, out]` over `rows` tokens into `stored` (the receipts' own output dtype, `grouped_cells.STORED`,
    where None)."""
    from dew.nn.kernels.templates.product import declared

    width_in, width_out = matrix
    stored = jnp.dtype(grouped_cells.STORED[made] if stored is None else stored)
    shaped = functools.partial(jax.ShapeDtypeStruct, dtype=dtype)
    sizes = jax.ShapeDtypeStruct((1,), jnp.int32)
    product = declared(dtype, None, outputs=(stored,))
    with generation.assumed(generation_name):
        if made == grouped_cells.Made.KERNEL_GRADIENT:
            call = grouped_matmul.tgmm_call(shaped((rows, width_in)), shaped((rows, width_out)), sizes,
                                            out_dtype=stored, interpret=False, dense=True)
        else:
            transposed = made == grouped_cells.Made.INPUT_GRADIENT
            tokens = shaped((rows, width_out if transposed else width_in))
            call = grouped_matmul.gmm_call(tokens, shaped((1, width_in, width_out)), sizes, out_dtype=stored,
                                           trans_rhs=transposed, interpret=False, dense=True)
        return grouped_matmul.grouped_case(*call, product=product, whole=True)


def _route(case: grouped_matmul.GroupedCase) -> str:
    """The kernel the grouped selector runs `case` on, as its target's process would."""
    with generation.assumed(case.call.target.generation):
        return selection.select(grouped_matmul.GROUPED, case).kernel


def test_a_timed_product_is_recognized_over_its_receipts_rows_and_arithmetic_alone():
    """A receipt is evidence for the products it timed alone (`grouped_cells.Timed`, row 6794): each product a
    receipt timed (`grouped_cells.timings`) is recognized as its matrix over that receipt's rows
    (`Kind.span`) in the arithmetic and output dtype it timed, and not past those rows (a fine-tune's 2,048
    took the 4B key's cell and its launches made the step's products 1.75 ms slower) nor in another
    arithmetic (a float32 call's TF32 kernel) or output dtype (another staging and memory)."""
    for timing in grouped_cells.timings():
        if timing.formed:
            continue
        rows = timing.timed.fewest
        assert _dense_case(timing.generation, timing.made, timing.matrix, rows).kind.span == timing.timed.span
        others = grouped_cells.MEASURED[timing.generation, timing.made, False]
        stored = jnp.dtype(grouped_cells.STORED[timing.made]).name
        if not any(timed.holds(timing.matrix, rows - 1, timing.timed.dtype, stored) for timed in others):
            assert _dense_case(timing.generation, timing.made, timing.matrix, rows - 1).kind.matrix is None
        wide = _dense_case(timing.generation, timing.made, timing.matrix, rows, dtype=jnp.float32,
                           stored=jnp.float32)
        assert wide.kind.matrix is None, timing
        if timing.timed.stored is not None:
            other = "bfloat16" if timing.timed.stored == "float32" else "float32"
            assert _dense_case(timing.generation, timing.made, timing.matrix, rows,
                               stored=other).kind.matrix is None, timing


@pytest.mark.parametrize("board", grouped_cells.SCOREBOARDS, ids=lambda board: board.evidence.split()[0])
def test_a_scoreboards_products_run_the_template_where_at_parity_and_its_kind_elsewhere(board):
    """A dense kind runs the template whole only where every product any receipt timed of it is at parity in
    time and temporaries (kernel-templates.md §1; bible §6, 2026-11-24); a kind with a product lagging keeps
    XLA's dot, its products at parity the template's over their timed rows, its lagging ones and every
    product it did not time XLA's (`grouped_cells.Scoreboard`)."""
    untimed = (3 * 1024 + 64, 8 * 1024 + 64)
    for made, projections in board.times.items():
        lags = any(timing.winner == "xla" for timing in grouped_cells.timings()
                   if (timing.generation, timing.made) == (board.generation, made))
        kind = "xla" if lags else board.winner(made)
        assert _route(_dense_case(board.generation, made, untimed, board.rows)) == kind, made
        for name in projections:
            route = _route(_dense_case(board.generation, made, grouped_cells.PROJECTIONS[name], board.rows))
            expected = "mosaic" if kind == "mosaic" or board.at_parity(made, name) else "xla"
            assert route == expected, (made, name)


def test_a_generation_no_receipt_timed_runs_the_template():
    """A generation no receipt timed (a B300) runs the template at every dense kind, whatever its
    instruction family's other generations measured (the B200's lagging 4B cells are the B200's)."""
    for made in grouped_cells.Made:
        if made != grouped_cells.Made.EXACT_KERNEL_GRADIENT:
            assert _route(_dense_case("sm103", made, (2560, 9216), 24576)) == "mosaic", made


def test_a_pinned_split_runs_only_where_its_partials_fit_the_workspace():
    """A pinned two-way split runs where its fp32 partials, as its launch declares them, fit the 4 MiB an
    A100's cuBLAS GEMM's workspace is (`GpuTarget.reference_workspace`, row 12850), and is refused
    (`tuning.Unpinnable`) where one thing grows them past it: a 1985-wide product, launched 2016 wide, of
    248 rows (3.8 MiB) or 264 (4.1 MiB)."""
    from dew.nn.kernels import tuning

    launch = grouped_matmul.MosaicGmm()
    bf16 = functools.partial(jax.ShapeDtypeStruct, dtype=jnp.bfloat16)

    def shapes(rows):
        return bf16((rows, 1024)), bf16((1024, 1985))

    def run(x, w):
        return gemm.gemm(x, w, None, out_dtype=jnp.bfloat16)

    with generation.assumed("sm80"):
        target = generation.gpu_target()
        x, w = shapes(248)
        ba = launch.bind(x, bf16((1, *w.shape)), jax.ShapeDtypeStruct((1,), jnp.int32), out_dtype="bfloat16",
                         trans_rhs=False, interpret=False, dense=True)
        tile = dataclasses.replace(launch.derived(ba, target), splits=2)
        with generation.chosen(grouped_matmul="mosaic"):
            with tuning.pinned({grouped_matmul.MosaicGmm: tile}), tuning.seen() as launches:
                jax.jit(run).trace(*shapes(248))
            assert tile in {config for (op, _), (config, _) in zip(launches, launches.taken, strict=True)
                            if isinstance(op, grouped_matmul.MosaicGmm)}
            with tuning.pinned({grouped_matmul.MosaicGmm: tile}), pytest.raises(tuning.Unpinnable):
                jax.jit(run).trace(*shapes(264))
