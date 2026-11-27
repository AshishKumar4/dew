"""A compile's live profile (`dew.training.memory.LiveProfile`): the bytes its schedule holds at its peak, of
the buffers a rung would keep, as the heap simulation counts the memory it occupies, and what a place of
the fit ladder frees of it (`dew.training.planning.priced`)."""

from dew.nn.protocols import COPIES_SCOPE, HEAD_SCOPE, WHOLE_LOGITS, RecomputeStatement
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
    is its own scratch, which a rung recomputing the forward makes again, and frees none of the peak; one
    made before it and live past it is a residual the rung drops; a forward replayed after the backward
    began (a resident transaction's) makes scratch a rung makes again too."""
    profile = LiveProfile.of((interval(30, 0, 3, 0), interval(20, 1, 6, 1),
                              interval(25, 4, 6, 2, op_name=BACKWARD), interval(5, 5, 7, 3)), frozenset())
    assert profile is not None
    assert [buffer.residual for buffer in profile.held] == [False, True, False, False]
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


def tiled(held: int):
    """A head policy whose tile holds `held` bytes whatever the logits and tile."""
    return lambda whole, tile, fused: held


def test_a_whole_heads_tile_frees_its_logits_and_holds_its_tile_where_they_were():
    """A place tiling a head the start held whole drops the whole head's buffers (`HEAD_SCOPE`) and holds
    its tile over the positions they held: where the head's phase is the peak (100) and the layers' is
    apart (80), a 10-byte tile frees 20, the peak then the layers'; a tile or fusion change of a head
    already tiled, or a head policy that prices no tile, is priced by none."""
    profile = LiveProfile((held(100, 0, 3, 0, op_name=HEAD, names=(WHOLE_LOGITS,), shape=(64, 512)),
                           held(80, 3, 6, 1)))
    assert priced(rung(tile=(16, 128), fused=True), START, profile, 0, tiled(10)) == 20
    assert priced(rung(tile=(16, 128)), START, profile, 0, tiled(90)) == 10
    assert priced(rung(tile=(16, 128)), START, profile, 0, lambda whole, tile, fused: None) is None
    assert priced(rung(tile=(16, 128)), rung(tile=(16, 128), fused=True), profile, 0, tiled(10)) is None


def test_a_place_frees_each_buffer_once_and_the_copies_bank_where_it_drops_them():
    """A place recomputing the forward and reading no narrow copies frees each residual once, the copies'
    casts (`COPIES_SCOPE`) and their bank among the step's arguments; a place reading copies the start read
    none of is priced by none."""
    casts = f"jit(step)/{COPIES_SCOPE}/convert"
    profile = LiveProfile((held(64, 0, 6, 0, residual=True), held(16, 1, 6, 1, op_name=casts, residual=True),
                           held(8, 4, 6, 2, op_name=BACKWARD)))
    assert priced(rung((Remat(),), narrow=False), START, profile, 32, None) == 88 - 8 + 32
    assert priced(START, rung(narrow=False), profile, 0, None) is None


def test_a_finer_split_frees_its_rows_share_beside_what_the_place_drops():
    """Priced at twice the microbatches its profile read, a place frees half of each buffer a microbatch's
    rows scale too."""
    profile = LiveProfile((held(64, 0, 6, 0, residual=True), held(40, 0, 6, 1, scaled=True)))
    assert priced(rung((Remat(),)), START, profile, 0, None, ratio=2) == 104 - 20


def test_a_heads_tile_holds_what_its_route_holds():
    """A head tiled unfused holds a tile of its tokens by its columns' width over its chunks; fused, a chunk
    of its tokens over the whole vocabulary, gathered whole where the whole head's columns were a split of
    it: 4 bytes a logit."""
    from dew.nn.backbones.causal_transformer import CausalTransformer
    from dew.objectives.lm import LMObjective

    model = CausalTransformer(vocab_size=65536, emb_features=8, num_layers=1, num_heads=1, mlp_features=16,
                              max_seq_len=8)
    objective = LMObjective(model, seq_len=8, head_chunks=16)
    assert objective.tiled_logits((4096, 65536), (1024, 8192), fused=False) == 4 * 1024 * 4096
    assert objective.tiled_logits((4096, 32768), (1024, 8192), fused=False) == 4 * 1024 * 2048
    assert objective.tiled_logits((4096, 32768), (1024, 8192), fused=True) == 4 * 1024 * 65536
