"""A compile's live profile (`dew.training.memory.LiveProfile`): the bytes its schedule holds at its peak, of
the buffers a rung would keep, as the heap simulation counts the memory it occupies, and what a place of
the fit ladder frees of it (`dew.training.planning.priced`)."""

from dew.nn.protocols import COPIES_SCOPE, HEAD_SCOPE, RecomputeStatement
from dew.objectives.lm.chunked import tile_logits
from dew.training.live_buffers import LiveInterval
from dew.training.memory import Held, LiveProfile, LiveSet
from dew.training.planning import Rung, priced

FORWARD = "jit(step)/jvp(Model)/dew_rung/dot_general"
BACKWARD = "jit(step)/transpose(jvp(Model))/dot_general"
HEAD = f"jit(step)/jvp(Head)/{HEAD_SCOPE}/dot_general"


def held(size, start, end, buffer, storage=None, *, scaled=False, residual=False, trace=0, op_name=FORWARD,
         names=(), shape=()):
    stored = buffer if storage is None else storage
    return Held(size, op_name, names, scaled, residual, trace, start, end, buffer, stored, shape)


def interval(size, start, end, buffer, *, trace=0, timeline=0, op_name=FORWARD):
    return LiveInterval(bytes=size, shape=(size,), dtype="u8", kind="temporary", instruction=f"i{buffer}",
                        op_name=op_name, source="", allocation=trace, trace=trace, timeline=timeline,
                        whole=True, complete=True, start=start, end=end, buffer=buffer, storage=buffer,
                        charged=True)


def test_the_peak_is_the_most_the_schedule_holds_at_once():
    """Buffers live over overlapping positions add; apart, the larger stands; an interval is half-open, so
    one freed where another starts is not held beside it."""
    profile = LiveProfile((held(10, 0, 4, 0), held(7, 2, 6, 1), held(30, 6, 8, 2)))
    assert profile.peak() == 30
    assert profile.peak(lambda buffer: buffer.buffer != 2) == 17


def test_a_storage_holds_the_most_its_live_buffers_hold_once():
    """Two buffers in one storage (an in-place fusion's output over its operand) hold its bytes once;
    dropping one frees the storage only where no buffer kept in it is live; a smaller buffer reusing a
    storage freed before it holds its own bytes there, not the first one's."""
    profile = LiveProfile((held(8, 0, 4, 0), held(8, 2, 6, 1, storage=0), held(5, 1, 3, 2)))
    assert profile.peak() == 13
    assert profile.peak(lambda buffer: buffer.buffer != 0) == 13
    assert profile.peak(lambda buffer: buffer.buffer == 2) == 5
    reused = LiveProfile((held(8, 0, 2, 0), held(3, 2, 4, 1, storage=0), held(4, 3, 5, 2)))
    assert reused.peak() == 8
    assert reused.peak(lambda buffer: buffer.buffer != 0) == 7


def test_a_split_divides_the_bytes_a_microbatchs_rows_scale():
    """At a split of four, a buffer the batch's rows scale holds a quarter of its bytes (rounded up), every
    other buffer its whole."""
    profile = LiveProfile((held(40, 0, 2, 0, scaled=True), held(6, 0, 2, 1)))
    assert profile.peak(split=4) == 16


def test_a_rung_frees_what_its_peak_falls_by_wherever_the_peak_then_stands():
    """A step whose peak is its layers' backward, its forward's activations live through it, and whose head
    peaks apart: a rung dropping the activations frees what the peak falls by to the head's, not the
    activations' bytes at the old peak, which `LiveSet.frees` reads where the compile states its profile
    (the 4B decoder's peak moved from its layers' backward to its head); a rung drops residuals alone, never
    the head's own values or a backward's temporaries."""
    profile = LiveProfile((held(50, 4, 10, 0, names=("mlp",), residual=True),
                           held(40, 6, 8, 1, op_name=BACKWARD), held(45, 1, 3, 2, op_name=HEAD)))
    assert profile.peak() == 90
    assert profile.frees(lambda op_name, names: "mlp" in names) == 90 - 45
    assert profile.frees(lambda op_name, names: True) == 90 - 45
    at_peak = LiveSet(0, 0, ((50, FORWARD, ("mlp",)), (40, BACKWARD, ())))
    assert at_peak.frees(lambda op_name, names: "mlp" in names) == 50
    profiled = LiveSet(0, 0, at_peak.made, profile)
    assert profiled.frees(lambda op_name, names: "mlp" in names) == 45


def test_a_residual_is_a_forward_value_live_into_the_backward():
    """Read off a compile's intervals, a forward's value freed before the backward makes its first buffer
    is its own scratch, which a rung recomputing the forward makes again, and frees none of the peak;
    one live past it is a residual the rung drops."""
    profile = LiveProfile.of((interval(30, 0, 3, 0), interval(20, 1, 6, 1),
                              interval(25, 4, 6, 2, op_name=BACKWARD)), frozenset())
    assert profile is not None
    assert [buffer.residual for buffer in profile.held] == [False, True, False]
    assert profile.most == 50
    assert profile.frees(lambda op_name, names: True) == 50 - 30


def test_a_heaps_pages_hold_their_buffers_together_and_heaps_apart_read_no_profile():
    """A heap's pages copy one trace (`LiveInterval.timeline`): their buffers live at one position add. Two
    heaps XLA simulates apart order their positions on no common clock, so no profile is read of them."""
    pages = LiveProfile.of((interval(100, 0, 4, 0), interval(90, 0, 4, 1, trace=1)), frozenset())
    assert pages is not None and pages.most == 190
    apart = (interval(100, 0, 4, 0), interval(90, 0, 4, 1, trace=1, timeline=1))
    assert LiveProfile.of(apart, frozenset()) is None


class Remat:
    """A module stating that its rung recomputes every value its blocks' forward makes."""

    def recompute_statement(self) -> RecomputeStatement:
        return RecomputeStatement("dew_rung", frozenset())


def rung(modules=(), tile=None, *, fused=False, narrow=True) -> Rung:
    return Rung(modules, tile, fused=fused, xla_defaults=False, narrow=narrow)


START = rung()


def test_a_whole_heads_tile_frees_its_logits_less_the_tile_it_holds_fused_or_not():
    """A place tiling a head the start held whole drops the whole logits (`HEAD_SCOPE`) and holds a tile of
    them in their place, a fused head's chunk of tokens over every column or an unfused head's tile; a tile
    or fusion change of a head already tiled is priced by no name."""
    tokens, vocab = 4096, 65536
    logits = 4 * tokens * vocab
    profile = LiveProfile((held(logits, 0, 4, 0, op_name=HEAD, shape=(tokens, vocab)),
                           held(1 << 20, 0, 4, 1)))
    for fused in (False, True):
        tiled = rung(tile=(1024, 8192), fused=fused)
        assert priced(tiled, START, profile, 0) == logits - tile_logits(tokens, vocab, (1024, 8192), fused)
    assert priced(rung(tile=(1024, 8192)), rung(tile=(1024, 8192), fused=True), profile, 0) is None


def test_a_place_frees_each_buffer_once_and_the_copies_bank_where_it_drops_them():
    """A place recomputing the forward and reading no narrow copies frees each residual once, the copies'
    casts (`COPIES_SCOPE`) and their bank among the step's arguments; a place reading copies the start read
    none of is priced by none."""
    casts = f"jit(step)/{COPIES_SCOPE}/convert"
    profile = LiveProfile((held(64, 0, 6, 0, residual=True), held(16, 1, 6, 1, op_name=casts, residual=True),
                           held(8, 4, 6, 2, op_name=BACKWARD)))
    assert priced(rung((Remat(),), narrow=False), START, profile, bank=32) == 88 - 8 + 32
    assert priced(START, rung(narrow=False), profile, 0) is None
