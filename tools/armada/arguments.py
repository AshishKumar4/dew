"""The command line of tools/armada/ci.py: each operation's arguments (`parser`), apart from what the
operations do. armada's commands fill the plan's and a task's (.armada.json), a GPU node's session a lane's,
the land queue's tools the rest."""

import argparse
from collections.abc import Sequence
from pathlib import Path


def plan_arguments(planning: argparse.ArgumentParser) -> None:
    """The `plan` operation's arguments, which .armada.json's plan command fills, and to which armada's `run`
    appends what follows its `--`."""
    planning.add_argument("--target", type=float, required=True)
    planning.add_argument("--timings", type=Path, required=True)
    planning.add_argument("--census", action="store_true",
                          help="the census (check 1's CENSUS): the whole suite with the test cache off, "
                               "every test "
                               "process compiling check 1's arms with YNNPACK's fusions on; armada passes it "
                               "after `--`, so the run's verdict is stored as no commit's")


def task_arguments(running: argparse.ArgumentParser, pythons: Sequence[str], deadline: float) -> None:
    """The `task` operation's arguments, which armada's task command fills from a plan entry; `deadline`, a
    task's own where none is named (session.py's DEADLINE)."""
    running.add_argument("--python", choices=pythons, required=True)
    running.add_argument("--tests", required=True)
    running.add_argument("--split", default="")
    running.add_argument("--out", type=Path, required=True)
    running.add_argument(
        "--cache", required=True,
        help="the plan's DEW_TEST_CACHE (`policy`): on, or off")
    running.add_argument("--whole", choices=("0", "1"), default="0",
                         help="the plan's DEW_WHOLE_SUITE: 1 on the whole suite's run, which runs its cells")
    running.add_argument("--census", choices=("0", "1"), default="0",
                         help="the plan's census: 1 runs the tests under check 1's CENSUS, recording no row")
    running.add_argument("--deadline", type=float, default=deadline,
                         help="seconds before armada's own task timeout (.armada.json) to interrupt at")
    running.add_argument("--coverage", type=Path, default=None,
                         help="run the files under coverage.py, its data file in this directory")


def release_arguments(planning: argparse.ArgumentParser, checking: argparse.ArgumentParser,
                      pythons: Sequence[str]) -> None:
    """The `release-plan` and `release` operations' arguments: a whole suite's second job (`release_plan`)."""
    planning.add_argument("verdict", type=Path, help="armada verdict <sha> --json")
    checking.add_argument("--python", choices=pythons, required=True)
    checking.add_argument("--tests", required=True)
    checking.add_argument("--requirement", required=True)
    checking.add_argument("--out", type=Path, required=True)


def tier_arguments(choosing: argparse.ArgumentParser) -> None:
    """`gpu-tier`'s arguments: the tier it names is the checkout's gpu-marked files (`gpu_tier`), each shard's
    on a line, then each file's pinned GPU generations on a line of JSON (`marked`)."""
    choosing.add_argument("--evidence", action="store_true",
                          help="each file of the tier with its marked tests and its records' backend reads")
    choosing.add_argument("--shards", type=int, default=1, help="how many GPUs share the tier")


def parser(pythons: Sequence[str], deadline: float) -> argparse.ArgumentParser:
    """ci.py's command line, every operation's arguments: the tested Pythons are `pythons`, a task's own
    deadline `deadline`."""
    parser = argparse.ArgumentParser(prog="tools/armada/ci.py")
    operations = parser.add_subparsers(dest="operation", required=True)
    deciding = operations.add_parser("policy", help="the test cache's policy for this checkout's change")
    deciding.add_argument("base", help="the commit the change is read from (its merge-base with HEAD)")
    plan_arguments(operations.add_parser("plan"))
    task_arguments(operations.add_parser("task"), pythons, deadline)
    selecting = operations.add_parser(
        "pytest", help="pytest on the arguments as a CI task runs it, CI's selection first; its exit status")
    selecting.add_argument("--python", choices=pythons, required=True)
    selecting.add_argument("--report", type=Path, required=True, help="where pytest writes its JUnit report")
    selecting.add_argument("--tests", default="", help="the files --split's group is of, space-separated")
    selecting.add_argument("--split", default="",
                           help="--tests' group as CI names it: 1/3, a check-1 site's test or its draws")
    selecting.add_argument("--whole", action="store_true",
                           help="select as main's whole run does, its whole_suite cells too (a bisect)")
    selecting.add_argument("arguments", nargs=argparse.REMAINDER,
                           help="after --, pytest's own: files, node ids and options")
    measuring = operations.add_parser("coverage")
    stages = measuring.add_subparsers(dest="stage", required=True)
    itemizing = stages.add_parser("items")
    itemizing.add_argument("--target", type=float, required=True)
    itemizing.add_argument("--timings", type=Path, required=True)
    reporting = stages.add_parser("report")
    reporting.add_argument("directory", type=Path)
    tier_arguments(operations.add_parser("gpu-tier"))
    lane = operations.add_parser("gpu-lane")
    lane.add_argument("--tests", help="the test files; with --network, every file holding a "
                                       "network-marked test by default (`network_files`)")
    lane.add_argument("--network", action="store_true",
                      help="the network-marked tests alone, with the Hub open, on the network lane's class")
    lane.add_argument("--prepare", action="store_true",
                      help="the files' prepare on the node's CPU, before their GPU run (`gpu_prepared`)")
    lane.add_argument("--out", type=Path, required=True)
    lane.add_argument("--cache", help="the test cache's policy (`policy`): every lane's but a prepare's")
    lane.add_argument("--processes", type=int, required=True, help="the most test files running at once")
    lane.add_argument("--deadline", type=float, default=3600.0, help="seconds a file may run")
    operations.add_parser("gangs")
    versions = operations.add_parser("pythons")
    versions.add_argument("--check", action="store_true",
                          help="refuse a .python-version other than the oldest, or a smoke of other versions")
    flagging = operations.add_parser(
        "slow", help="each test of a verdict past the group target, a JSON line each; --new: exit 1 on any")
    flagging.add_argument("verdict", type=Path, help="armada verdict <sha> --json")
    flagging.add_argument("--new", type=Path, default=None,
                          help="the durations record the change is measured against: only the tests it lacks")
    release_arguments(
        operations.add_parser("release-plan", help="a whole suite verdict's release checks, a task each"),
        operations.add_parser("release", help="one requirement's release checks; exit 1 on a red row"),
        pythons)
    whole = operations.add_parser(
        "partition", help="whether a whole verdict's rows ran each collected test once; exit 1 where not")
    whole.add_argument("verdict", type=Path, help="armada verdict <sha> --json")
    whole.add_argument("collected", type=Path, nargs="+",
                       help="pytest --collect-only -q of the commit's test files under CI's selection")
    recording = operations.add_parser("durations")
    recording.add_argument("verdict", type=Path, help="main's CI verdict: armada verdict <sha> --json")
    return parser
