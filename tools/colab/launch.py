"""Launch a pinned packed session with a durable systemd supervisor."""

import argparse
import shlex
import subprocess
import sys

from tools.colab.common import REPO, STATE, name


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("attempt", type=name)
    parser.add_argument("session", type=name)
    parser.add_argument("--rate", required=True, type=float)
    parser.add_argument("--budget-cu", type=float)
    separator = sys.argv.index("--")
    args = parser.parse_args(sys.argv[1:separator])
    if args.budget_cu is not None and not 0 < args.budget_cu < float("inf"):
        parser.error("budget-cu must be finite and positive")
    directory = STATE / "colab-ops" / args.attempt
    driver = [sys.executable, "-m", "tools.colab.ops_driver", args.session, *sys.argv[separator + 1 :]]
    subprocess.run([*driver, "--dry-run"], cwd=REPO, check=True)
    supervisor = [sys.executable, "-m", "tools.colab.ops_supervisor"]
    reconciler = "dew-colab-ops-reconcile.timer"
    loaded = subprocess.run(
        ["systemctl", "--user", "show", "--property=LoadState", "--value", reconciler],
        capture_output=True,
        text=True,
        check=True,
    )
    if loaded.stdout.strip() == "loaded":
        subprocess.run(["systemctl", "--user", "start", reconciler], check=True)
    else:
        subprocess.run(
            [
                "systemd-run",
                "--user",
                "--unit=dew-colab-ops-reconcile",
                "--on-active=60",
                "--on-unit-active=60",
                "--property=WorkingDirectory=" + str(REPO),
                *supervisor,
                "--reconcile",
            ],
            cwd=REPO,
            check=True,
        )
    finalizer = [*supervisor, "--run-dir", str(directory), "--finalize-only"]
    budget = ["--budget-cu", str(args.budget_cu)] if args.budget_cu is not None else []
    subprocess.run(
        [
            "systemd-run",
            "--user",
            "--unit=dew-colab-" + args.attempt,
            "--property=Type=exec",
            "--property=Nice=10",
            "--property=CPUQuota=50%",
            "--property=MemoryMax=1G",
            "--property=TimeoutStopSec=240",
            "--property=WorkingDirectory=" + str(REPO),
            "--property=ExecStopPost=" + shlex.join(finalizer),
            *supervisor,
            "--run-dir",
            str(directory),
            "--session",
            args.session,
            "--rate",
            str(args.rate),
            *budget,
            "--",
            *driver,
        ],
        cwd=REPO,
        check=True,
    )


if __name__ == "__main__":
    main()
