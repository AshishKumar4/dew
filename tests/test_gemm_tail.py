"""A dense product's balanced tail (`templates.tail`): its ranges over the programs, split alike over fewer
ranges than programs where that prices least (`tail._partitions`), each program's whole waves its own."""

from dew.nn.kernels.templates import tail as tail_of


def test_a_tail_split_over_fewer_ranges_than_programs_leaves_each_its_whole_waves():
    """A balanced tail whose tiles' depth is split alike over fewer ranges than the launch's programs
    (`tail._partitions`): every program's whole-wave tile is its own, fixed up by it alone (its gathered
    programs one: the native epilogue stores it), and the programs past the ranges walk no tail step."""
    programs, steps, tiles = 8, 4, 11
    split = tail_of.Tail(waves=1, iterations=3 * steps, partitions=6, steps=steps)
    for program in range(programs):
        tile, start, count, (first, gathered, _) = split.segment((0, program), programs, tiles)
        assert (int(tile), int(start), int(count)) == (program, 0, steps)
        assert (int(first), int(gathered)) == (program, program + 1)
        if program >= split.partitions:
            assert int(split.segment((1, program), programs, tiles)[2]) == 0
