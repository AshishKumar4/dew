"""A compile's live profile (`dew.training.memory.LiveProfile`): the bytes its schedule holds at its peak, of
the buffers a rung would keep, as the heap simulation counts the memory it occupies, and what a place of
the fit ladder frees of it (`dew.training.planning.priced`)."""

import jax
import jax.numpy as jnp

from dew.nn.protocols import COPIES_SCOPE, HEAD_SCOPE, UPDATE_SCOPE, WHOLE_LOGITS, RecomputeStatement
from dew.training.live_buffers import LiveInterval
from dew.training.memory import Held, LiveProfile, LiveSet
from dew.training.planning import FitSearch, Rung, priced

FORWARD = "jit(step)/jvp(Model)/dew_rung/dot_general"
BACKWARD = "jit(step)/transpose(jvp(Model))/dot_general"
HEAD = f"jit(step)/jvp(Head)/{HEAD_SCOPE}/dot_general"


def held(size, start, end, buffer, storage=None, *, scaled=False, residual=False, trace=0, op_name=FORWARD,
         names=(), shape=(), until=None, stack=None):
    stored = buffer if storage is None else storage
    return Held(size, op_name, names, scaled, residual, trace, start, end, buffer, stored, shape, until,
                stack)


def interval(size, start, end, buffer, *, trace=0, timeline=0, op_name=FORWARD, uses=(), storage=None,
             names=(), stack=None, filling=False):
    return LiveInterval(bytes=size, shape=(size,), dtype="u8", kind="temporary", instruction=f"i{buffer}",
                        op_name=op_name, source="", names=names, stack=stack, filling=filling,
                        allocation=trace, trace=trace,
                        timeline=timeline, uses=uses, whole=True, complete=True, start=start, end=end,
                        buffer=buffer, storage=buffer if storage is None else storage, charged=True)


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
                           held(40, 6, 8, 1, op_name=BACKWARD), held(55, 1, 3, 2, op_name=HEAD)))
    assert profile.peak() == 90
    assert profile.frees(lambda op_name, names: "mlp" in names) == 90 - 55
    assert profile.frees(lambda op_name, names: True) == 90 - 55
    at_peak = LiveSet(0, 0, ((50, FORWARD, ("mlp",)), (40, BACKWARD, ())))
    assert at_peak.frees(lambda op_name, names: "mlp" in names) == 50
    profiled = LiveSet(0, 0, at_peak.made, profile)
    assert profiled.frees(lambda op_name, names: "mlp" in names) == 35


def test_a_residual_is_a_forward_value_live_into_the_backward():
    """Read off a compile's intervals, a forward's value freed before the backward makes its first buffer
    is its own scratch, which a rung recomputing the forward makes again, and frees none of the peak; one
    made before it and live past it is a residual the rung drops; a forward replayed after the backward
    began is scratch unless its own pullback reads it while it holds it (a resident transaction's), and a
    read past its interval is another call's, none of its own."""
    profile = LiveProfile.of((interval(30, 0, 3, 0), interval(20, 3, 6, 1),
                              interval(25, 4, 6, 2, op_name=BACKWARD),
                              interval(5, 5, 7, 3, uses=((9, BACKWARD),)),
                              interval(8, 5, 9, 4, uses=((8, BACKWARD),))), frozenset())
    assert profile is not None
    assert [buffer.residual for buffer in profile.held] == [False, True, False, False, True]
    assert profile.most == 58
    assert profile.frees(lambda op_name, names: True) == 58 - 38


def test_a_residual_is_held_through_its_forwards_reads_before_the_backward():
    """A residual a rung recomputes is made and read by its forward all the same: held through its forward's
    last read before the backward began (`LiveInterval.uses`, `Held.until`), where it shares the forward's
    peak with its scratch dropping it frees none of that peak; read last before the scratch, it frees the
    scratch's share of it; a read named the forward's at or past the backward's start is the backward's."""
    def profile(read: int) -> LiveProfile:
        found = LiveProfile.of((interval(100, 0, 6, 0, uses=((read, FORWARD), (5, BACKWARD))),
                                interval(70, 1, 3, 1), interval(10, 4, 6, 2, op_name=BACKWARD)), frozenset())
        assert found is not None
        return found

    assert profile(2).most == profile(0).most == profile(4).most == 170
    assert profile(2).frees(lambda op_name, names: True) == 0
    assert profile(0).frees(lambda op_name, names: True) == 170 - 100
    assert profile(4).frees(lambda op_name, names: True) == 170 - 100


def test_a_residual_read_by_an_op_of_no_name_is_held_to_its_end():
    """A residual a read of no op name reads (a fusion whose computation names none: no phase told of it)
    is held to its interval's end, as one read placed nowhere: freeing it at its forward's last named read
    would credit a release the compile may not make (train 1406); read by its forward alone, it frees the
    scratch's share."""
    def profile(uses) -> LiveProfile:
        found = LiveProfile.of((interval(100, 0, 6, 0, uses=uses), interval(70, 1, 3, 1),
                                interval(10, 4, 6, 2, op_name=BACKWARD)), frozenset())
        assert found is not None
        return found

    assert profile(((0, FORWARD), (5, BACKWARD))).frees(lambda op_name, names: True) == 170 - 100
    assert profile(((0, FORWARD), (2, ""), (5, BACKWARD))).frees(lambda op_name, names: True) == 0


def test_a_replayed_residual_is_held_through_its_forwards_last_read():
    """A forward replayed after the backward began (a resident transaction's) makes the value its own
    pullback reads all the same, and a rung recomputing it holds it until that forward last reads it
    (`LiveInterval.uses`, `Held.until`): where it shares the replay's peak with its scratch, dropping it
    frees none of that peak; read last before the scratch, it frees the scratch's share of it."""
    def profile(read: int) -> LiveProfile:
        found = LiveProfile.of((interval(10, 0, 1, 0, op_name=BACKWARD),
                                interval(100, 1, 7, 1, uses=((read, FORWARD), (6, BACKWARD))),
                                interval(70, 2, 4, 2), interval(10, 5, 7, 3, op_name=BACKWARD)), frozenset())
        assert found is not None
        return found

    assert profile(3).most == profile(1).most == 170
    assert profile(3).frees(lambda op_name, names: True) == 0
    assert profile(1).frees(lambda op_name, names: True) == 170 - 100


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
    return lambda whole, tile, fused, itemsize: held


def test_a_stack_a_place_recomputes_is_built_nowhere_where_the_forward_reads_it_nowhere():
    """A scan's stack is one stack in each buffer holding it (`Held.stack`): the zeros its loop fills (40
    bytes over 0-4), each step's write into them in place (3-8) and the loop's value the backward reads
    (7-14), of which only the last is a residual. A place recomputing it builds none of them where the
    forward reads the stack nowhere past its loop, so the forward loop's peak beside its scratch (70) goes
    with it, the jump's price (`priced`) and the remat order's (`LiveSet.frees`) alike (row 12463: a cold 4B
    start's remat places priced at 37.8 GiB of its 180.2 GiB miss, its forward loop holding 167 GiB of its
    stacks' in-loop values); where any op but the backward's reads it past its loop, the loop builds it all
    the same (review of row 12463)."""

    def profile(*read: tuple[int, str]) -> LiveProfile:
        loop = (7, 1)
        found = LiveProfile.of((
            interval(40, 0, 4, 0, stack=loop, filling=True),
            interval(40, 3, 8, 1, storage=0, stack=loop, filling=True),
            interval(40, 7, 14, 2, storage=0, names=("mlp",), stack=loop, uses=(*read, (11, BACKWARD))),
            interval(30, 2, 9, 3), interval(50, 10, 13, 4, op_name=BACKWARD)), frozenset())
        assert found is not None
        return found

    unread = profile()
    assert [buffer.residual for buffer in unread.held] == [False, False, True, False, False]
    assert unread.most == 90
    assert unread.frees(lambda op_name, names: "mlp" in names) == 90 - 50
    assert priced(rung((Remat(),)), START, unread, 0, None) == 90 - 50
    assert LiveSet(0, 0, (), unread).frees(lambda op_name, names: "mlp" in names) == 90 - 50
    # A place keeping the stack keeps every buffer of it.
    assert unread.frees(lambda op_name, names: False) == 0
    # Read past its loop by the forward, before the backward's start or after it, or by the update, it is
    # built all the same, the place freeing the residual's span past its forward's reads alone (the peak
    # then the forward loop's, 70); read by an op no phase is told of, it is held whole.
    update = f"jit(step)/{UPDATE_SCOPE}/mul"
    assert [profile(read).frees(lambda op_name, names: "mlp" in names)
            for read in ((9, FORWARD), (12, FORWARD), (12, update), (12, ""))] == [90 - 70] * 3 + [0]
    # Its zeros made before an early backward buffer and live past it, a residual its loop alone reads,
    # neither ask for the stack nor drop it: a place dropping the zeros' name and keeping the output's
    # keeps it whole (review of row 12463).
    def early(zeros: tuple[str, ...], scratch: tuple[int, int, int], later: int) -> LiveProfile:
        found = LiveProfile.of((
            interval(40, 0, 4, 0, names=zeros, stack=(7, 1), filling=True),
            interval(40, 3, 8, 1, storage=0, stack=(7, 1), filling=True),
            interval(40, 7, 14, 2, storage=0, names=("mlp",), stack=(7, 1), uses=((11, BACKWARD),)),
            interval(scratch[0], scratch[1], scratch[2], 3), interval(1, 1, 13, 4, op_name=BACKWARD),
            interval(later, 10, 13, 5, op_name=BACKWARD)), frozenset())
        assert found is not None and found.held[0].residual
        return found

    unread = early((), (30, 2, 9), 50)
    assert unread.frees(lambda op_name, names: "mlp" in names) == unread.most - 51
    # Its zeros held whole beside the scratch they meet before the stack's first write (71), not cut where
    # their name is dropped (review of row 12463: 61).
    saved = early(("attn",), (30, 2, 3), 20)
    assert saved.most == 71 and saved.frees(lambda op_name, names: "attn" in names) == 0


def test_a_whole_heads_tile_frees_its_logits_and_holds_its_tile_where_they_were():
    """A place tiling a head the start held whole drops the whole head's buffers (`HEAD_SCOPE`) and holds
    its tile over the positions they held: where the head's phase is the peak (100) and the layers' is
    apart (80), a 10-byte tile frees 20, the peak then the layers'; a tile or fusion change of a head
    already tiled, or a head policy that prices no tile, is priced by none."""
    profile = LiveProfile((held(100, 0, 3, 0, op_name=HEAD, names=(WHOLE_LOGITS,), shape=(64, 512)),
                           held(80, 3, 6, 1)))
    assert priced(rung(tile=(16, 128), fused=True), START, profile, 0, tiled(10)) == 20
    assert priced(rung(tile=(16, 128)), START, profile, 0, tiled(90)) == 10
    assert priced(rung(tile=(16, 128)), START, profile, 0, lambda whole, tile, fused, itemsize: None) is None
    assert priced(rung(tile=(16, 128)), rung(tile=(16, 128), fused=True), profile, 0, tiled(10)) is None


def test_a_place_frees_each_buffer_once_and_the_copies_bank_where_it_drops_them():
    """A place recomputing the forward and reading no narrow copies frees each buffer once: a residual its
    span past its forward's last read of it, the copies' casts (`COPIES_SCOPE`) whole, made by no step that
    reads none, and their bank among the step's arguments; a place reading copies the start read none of is
    priced by none."""
    casts = f"jit(step)/{COPIES_SCOPE}/convert"
    profile = LiveProfile((held(64, 2, 6, 0, residual=True, until=3),
                           held(16, 1, 6, 1, op_name=casts, residual=True),
                           held(8, 4, 6, 2, op_name=BACKWARD)))
    assert priced(rung((Remat(),), narrow=False), START, profile, 32, None) == 88 - 64 + 32
    assert priced(START, rung(narrow=False), profile, 0, None) is None


def test_a_finer_split_frees_its_rows_share_beside_what_the_place_drops():
    """Priced at twice the microbatches its profile read, a place frees half of each buffer a microbatch's
    rows scale."""
    profile = LiveProfile((held(64, 0, 6, 0), held(40, 0, 6, 1, scaled=True)))
    assert priced(START, START, profile, 0, None, ratio=2) == 104 - (64 + 20)


def test_a_heads_tile_holds_what_its_route_holds():
    """A head tiled unfused holds a tile of its tokens by its columns' width over its chunks; fused, a chunk
    of its tokens over every column it scores, its whole logits' bytes a logit (4 at float32, 8 at
    float64, never fewer than float32's: a tile forms them so); a fused head whose whole head's columns
    were a split of what it scores, its tokens then its group's gathered, is priced by none. A head scoring
    fewer columns than its table's rows (Inkling 2's 200058 of 201024, `output_vocab`) is priced at those
    columns (train 1406)."""
    from dew.nn.backbones.causal_transformer import CausalTransformer
    from dew.objectives.lm import LMObjective

    def objective(**options):
        model = CausalTransformer(vocab_size=65536, emb_features=8, num_layers=1, num_heads=1,
                                  mlp_features=16, max_seq_len=8, **options)
        return LMObjective(model, seq_len=8, head_chunks=16)

    whole = objective()
    assert whole.tiled_logits((4096, 65536), (1024, 8192), fused=False, itemsize=4) == 4 * 1024 * 4096
    assert whole.tiled_logits((4096, 32768), (1024, 8192), fused=False, itemsize=4) == 4 * 1024 * 2048
    assert whole.tiled_logits((4096, 65536), (1024, 8192), fused=True, itemsize=4) == 4 * 1024 * 65536
    assert whole.tiled_logits((4096, 65536), (1024, 8192), fused=True, itemsize=8) == 8 * 1024 * 65536
    # A tile's logits at least float32's, whatever the whole head held (a single GPU's bf16 logits).
    assert whole.tiled_logits((4096, 65536), (1024, 8192), fused=False, itemsize=2) == 4 * 1024 * 4096
    assert whole.tiled_logits((4096, 32768), (1024, 8192), fused=True, itemsize=4) is None
    scoring = objective(output_vocab=65000)
    assert scoring.tiled_logits((4096, 65000), (1024, 8192), fused=True, itemsize=4) == 4 * 1024 * 65000
    assert scoring.tiled_logits((4096, 65536), (1024, 8192), fused=True, itemsize=4) is None


def test_a_compiles_residuals_are_read_by_its_backward_where_it_holds_them():
    """Off a compile-only H100's buffer assignment, a tanh chain's gradient: some forward value is a
    residual, its backward reading it; each of its reads is placed in its trace within its interval, its
    forward's last before its backward's, so a rung recomputing it holds it through that read alone."""
    from test_compile_ahead import gpus

    from dew.training.live_buffers import live_intervals

    def step(x, w):
        for _ in range(3):
            x = jnp.tanh(x @ w)
        return x.sum()

    device = gpus("sm90")[0]
    placed = jax.sharding.SingleDeviceSharding(device)
    tile = jax.ShapeDtypeStruct((256, 256), jnp.float32, sharding=placed)
    compiled = jax.jit(jax.grad(step, argnums=(0, 1))).lower(tile, tile).compile()
    profile = LiveProfile.of(live_intervals(compiled), frozenset())
    assert profile is not None
    residuals = [buffer for buffer in profile.held if buffer.residual]
    assert residuals
    for buffer in residuals:
        assert buffer.until is not None and buffer.start < buffer.until <= buffer.end, buffer
    # Its forward's reads decoded off the assignment and placed: some residual is held past its own placing
    # and released before its end.
    assert any(buffer.until is not None and buffer.start + 1 < buffer.until < buffer.end
               for buffer in residuals)


def test_a_searchs_places_hold_each_rungs_copy_free_form_beside_it():
    """The places a search reads past a start reading narrow copies, the jump's and the frontier's alike
    (`FitSearch.placed`): its copy-only release, then each rung of the ladder and its form without the
    copies beside it, so a plan the jump admits copy-free (a tiled head without its copies) is one the
    frontier lists lighter rungs below (train 1409: the frontier's order held the start's release alone, a
    copy-free tile in none of it, its frontier then empty). A start reading none places its rungs alone, and
    a step that reads no copies at all each rung once, as it stands (row 14129: each priced twice)."""
    tile = rung(tile=(16, 128))
    assert FitSearch.placed(START, [tile]) == [rung(narrow=False), tile, rung(tile=(16, 128), narrow=False)]
    copy_free = rung(tile=(16, 128), narrow=False)
    assert FitSearch.placed(rung(narrow=False), [copy_free]) == [copy_free]
    assert FitSearch.placed(START, [tile], copies=False) == [tile]


def test_a_custom_step_reads_no_narrow_copies():
    """A narrow rung's step reads narrow copies only where the trainer composes it: a custom step traces no
    loss to read its casts from, so its narrow rung and its copy-free form are one program, which a
    frontier reads once (`copies_narrow`; train 1437), as it does a scaled loss's and DiLoCo's."""
    from types import SimpleNamespace

    from dew.nn.kernels.generation import chosen
    from dew.training._candidates import copies_narrow

    state = SimpleNamespace(variables={"params": {}})
    built = SimpleNamespace(step=None, dynamic_scale=False, diloco=None)
    with chosen(narrow_copies="on"):
        assert copies_narrow(built, state)
        custom = SimpleNamespace(**{**vars(built), "step": lambda objective, optimizer: None})
        assert not copies_narrow(custom, state)
        assert not copies_narrow(SimpleNamespace(**{**vars(built), "dynamic_scale": True}), state)


def test_a_rungs_stacks_are_parted_at_once_only_where_its_peak_holds_the_update():
    """A rung that missed by 47 within what its joined stacks hold by count (60) is asked parted at once only
    where its peak holds as much of the update's own values (`LiveSet.updating`: neither its forward's nor
    its backward's), the stacks' phase; a peak of its layers' residuals and their gradients holds none, so
    its parting waits for the climb's end, as it does where the peak's largest value states no op (row
    12463: a 4B cold start parted against a residual peak; its review)."""
    from dew.training.planning import Parting, Rung

    rung = Rung((), None, fused=False, xla_defaults=False, narrow=True)
    residual = LiveSet(0, 0, ((40, FORWARD, ("mlp",)), (30, BACKWARD, ())))
    updating = LiveSet(0, 0, ((20, BACKWARD, ()), (50, f"jit(step)/{UPDATE_SCOPE}/dot_general", ())))
    # A value no op states is of no phase known, the update's none.
    unknown = LiveSet(0, 0, ((80, "", ()), (10, f"jit(step)/{UPDATE_SCOPE}/dot_general", ())))
    assert residual.updating() == 0 and updating.updating() == 50 and unknown.updating() == 10
    assert Parting(60, lambda verdict: [verdict]).after(rung, 47, unknown.updating()) is None
    assert Parting(60, lambda verdict: [verdict]).after(rung, 47, residual.updating()) is None
    parted = Parting(60, lambda verdict: [verdict]).after(rung, 47, updating.updating())
    assert parted is not None and parted.ns_group == 1


def test_a_rungs_miss_and_its_update_bytes_are_read_in_a_devices_bytes():
    """A search on an allocator whose temporaries count twice (factor 2) reads a rung's miss of 80 as a
    device's 40 and its peak's update values as the live set states them, 60, unhalved: a joined rung its
    stacks' 60 cover is parted at once (review: the update's bytes halved to 30 deferred it)."""
    from types import SimpleNamespace

    from test_miss_driven_rungs import scripted_search

    from dew.training.memory import MemoryMiss
    from dew.training.planning import Parting

    search = scripted_search(lambda mine: [mine])
    search.factor = 2
    at = search.rung()
    search.probes.append(SimpleNamespace(split=1, rung=search.current()))
    search.misses[(at, 1)] = MemoryMiss(1, scaled=0, shortfall=80, factor=2)
    search.live[(at, 1)] = LiveSet(0, 0, ((60, f"jit(step)/{UPDATE_SCOPE}/dot_general", ()),))
    assert search.missed_with_update(at) == (40, 60)
    parted = Parting(60, lambda verdict: [verdict]).after(
        Rung((), None, fused=False, xla_defaults=False, narrow=True), *search.missed_with_update(at))
    assert parted is not None and parted.ns_group == 1

